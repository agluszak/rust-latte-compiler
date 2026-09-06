use crate::ast;
use crate::ast::Literal;
use crate::const_cond::const_bool;
use crate::ir::{BinaryOpCode, BlockId, Ir, Terminator, UnaryOpCode, ValueId};
use crate::lower::BasicBlockContinuation::{ContinueBlock, Stop};
use crate::ssa::{BuildingValue, IrBuilder};
use crate::types::Type;
use crate::typed_ast::{TypedBlock, TypedExpr, TypedExprKind, TypedFnDecl, TypedStmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BasicBlockContinuation {
    ContinueBlock(BlockId),
    Stop,
}

impl Ir {
    pub fn translate_function(&mut self, decl: TypedFnDecl) {
        let mut ir = IrBuilder::new();
        let ty = decl.ty();
        let entry_block = ir.new_block();
        ir.seal_block(entry_block);
        for (arg, i) in decl.args.into_iter().zip(0..) {
            let argument = ir.emit(entry_block, BuildingValue::Argument(i), arg.value.ty);
            ir.write_variable(arg.value.var_id, entry_block, argument);
        }

        let continuation = translate_block(&mut ir, decl.body.value, entry_block);
        if let ContinueBlock(block_id) = continuation {
            let Type::Function(_, ret) = &ty else {
                panic!("function type must be a function");
            };
            if **ret == Type::Void {
                ir.finish_block(block_id, Terminator::ReturnNoValue);
            } else {
                panic!("non-void function falls through after successful return checking");
            }
        }
        let function_name = decl.name.value.0;
        let function_ir = ir.finish(ty, entry_block);

        self.functions.insert(function_name, function_ir);
    }
}

fn translate_expr(
    context: &mut IrBuilder,
    expr: TypedExpr,
    block_id: BlockId,
) -> (ValueId, BlockId) {
    let (value, block_id) = match expr.expr {
        TypedExprKind::Variable(_, id) => (context.read_variable(id, block_id), block_id),
        TypedExprKind::Literal(lit) => {
            let val = match lit {
                Literal::Int(i) => context.emit(
                    block_id,
                    BuildingValue::Int(i32::try_from(i).expect("int range checked")),
                    Type::Int,
                ),
                Literal::String(s) => {
                    context.emit(block_id, BuildingValue::String(s), Type::LatteString)
                }
                Literal::Bool(b) => context.emit(block_id, BuildingValue::Bool(b), Type::Bool),
            };
            (val, block_id)
        }
        TypedExprKind::Binary { lhs, op, rhs } => {
            // Logical operators are represented exclusively by short-circuit CFG.
            if matches!(op.value, ast::BinaryOp::And | ast::BinaryOp::Or) {
                let (lhs, lhs_block) = translate_expr(context, lhs.value, block_id);

                let rhs_block = context.new_block();
                let join_block = context.new_block();

                // `a && b` evaluates b only when a is true, so the direct
                // edge to the join carries `false` (a itself); `a || b`
                // evaluates b only when a is false, so its direct edge
                // carries `true` (again a itself). Either way the phi
                // merges the actual operand values, no synthetic blocks
                // or booleans needed.
                let (then_block, else_block) = match op.value {
                    ast::BinaryOp::And => (rhs_block, join_block),
                    ast::BinaryOp::Or => (join_block, rhs_block),
                    _ => unreachable!(),
                };

                context
                    .finish_block(lhs_block, Terminator::Branch(lhs, then_block, else_block));
                context.seal_block(rhs_block);
                let (rhs, rhs_end) = translate_expr(context, rhs.value, rhs_block);
                context.finish_block(rhs_end, Terminator::Jump(join_block));
                context.seal_block(join_block);

                let phi = context.new_phi(join_block, Type::Bool);
                context.add_phi_incoming(phi, lhs_block, lhs);
                context.add_phi_incoming(phi, rhs_end, rhs);
                let phi = context.try_remove_trivial_phi(phi);
                return (phi, join_block);
            }

            let op = match op.value {
                ast::BinaryOp::Add => BinaryOpCode::Add,
                ast::BinaryOp::Sub => BinaryOpCode::Sub,
                ast::BinaryOp::Mul => BinaryOpCode::Mul,
                ast::BinaryOp::Div => BinaryOpCode::Div,
                ast::BinaryOp::Mod => BinaryOpCode::Mod,
                ast::BinaryOp::Gt => BinaryOpCode::Gt,
                ast::BinaryOp::Lt => BinaryOpCode::Lt,
                ast::BinaryOp::Gte => BinaryOpCode::Gte,
                ast::BinaryOp::Lte => BinaryOpCode::Lte,
                ast::BinaryOp::Eq => BinaryOpCode::Eq,
                ast::BinaryOp::Neq => BinaryOpCode::Neq,
                ast::BinaryOp::And | ast::BinaryOp::Or => unreachable!(),
            };

            let (lhs, block_id) = translate_expr(context, lhs.value, block_id);
            let (rhs, block_id) = translate_expr(context, rhs.value, block_id);

            let val = context.emit(block_id, BuildingValue::BinaryOp(op, lhs, rhs), expr.ty);
            (val, block_id)
        }
        TypedExprKind::Unary { op, expr: target } => {
            let (val, block_id) = translate_expr(context, target.value, block_id);
            let op = match op.value {
                ast::UnaryOp::Neg => UnaryOpCode::Neg,
                ast::UnaryOp::Not => UnaryOpCode::Not,
            };
            let val = context.emit(block_id, BuildingValue::UnaryOp(op, val), expr.ty);
            (val, block_id)
        }
        TypedExprKind::Application { target, args } => {
            let TypedExprKind::Variable(_, id) = target.value.expr else {
                panic!("This should have been caught by the typechecker")
            };

            let mut arg_values = Vec::new();
            let mut current_block_id = block_id;
            for arg in args {
                let (arg, block_id) =
                    translate_expr(context, arg.value, current_block_id);
                current_block_id = block_id;
                arg_values.push(arg);
            }

            let val = context.emit(
                current_block_id,
                BuildingValue::Call(id, arg_values),
                expr.ty,
            );
            (val, current_block_id)
        }
    };
    (value, block_id)
}

fn translate_block(
    context: &mut IrBuilder,
    block: TypedBlock,
    block_id: BlockId,
) -> BasicBlockContinuation {
    let mut block_id = block_id;
    for stmt in block.0 {
        match translate_stmt(context, stmt.value, block_id) {
            ContinueBlock(new_block_id) => block_id = new_block_id,
            Stop => return Stop,
        }
    }
    ContinueBlock(block_id)
}

fn default_value(ty: &Type) -> BuildingValue {
    match ty {
        Type::Int => BuildingValue::Int(0),
        Type::Bool => BuildingValue::Bool(false),
        Type::LatteString => BuildingValue::String(String::new()),
        Type::Function(_, _) => panic!("Function cannot have a default value"),
        Type::Void => panic!("Void cannot have a default value"),
    }
}

#[allow(clippy::too_many_lines)]
fn translate_stmt(
    context: &mut IrBuilder,
    stmt: TypedStmt,
    block_id: BlockId,
) -> BasicBlockContinuation {
    match stmt {
        TypedStmt::Empty => ContinueBlock(block_id),
        TypedStmt::Block(block) => translate_block(context, block.value, block_id),
        TypedStmt::Decl(decl) => {
            let mut block_id = block_id;
            for item in decl.value.items {
                if let Some(expr) = item.value.init {
                    let (expr, continuation_block) =
                        translate_expr(context, expr.value, block_id);
                    block_id = continuation_block;
                    context.write_variable(item.value.var_id, block_id, expr);
                } else {
                    let default = default_value(&item.value.ty);
                    let default = context.emit(block_id, default, item.value.ty);
                    context.write_variable(item.value.var_id, block_id, default);
                }
            }
            ContinueBlock(block_id)
        }
        TypedStmt::Assignment {
            target: _,
            target_id,
            expr,
        } => {
            let (expr, block_id) = translate_expr(context, expr.value, block_id);
            context.write_variable(target_id, block_id, expr);
            ContinueBlock(block_id)
        }
        TypedStmt::Return(expr) => {
            let expr = expr.map(|expr| translate_expr(context, expr.value, block_id));
            if let Some((expr, block_id)) = expr {
                context.finish_block(block_id, Terminator::Return(expr));
            } else {
                context.finish_block(block_id, Terminator::ReturnNoValue);
            }
            Stop
        }
        TypedStmt::If {
            cond,
            then,
            otherwise,
        } => {
            if let Some(constant) = const_bool(&cond.value) {
                return if constant {
                    translate_stmt(context, then.value, block_id)
                } else if let Some(otherwise) = otherwise {
                    translate_stmt(context, otherwise.value, block_id)
                } else {
                    ContinueBlock(block_id)
                };
            }
            let (cond, block_id) = translate_expr(context, cond.value, block_id);
            let cond = context.resolve_alias(cond);

            let then_block = context.new_block();
            let then_continuation = translate_stmt(context, then.value, then_block);
            if let Some(otherwise) = otherwise {
                let else_block = context.new_block();
                let else_continuation =
                    translate_stmt(context, otherwise.value, else_block);
                context
                    .finish_block(block_id, Terminator::Branch(cond, then_block, else_block));
                context.seal_block(else_block);
                context.seal_block(then_block);

                if let (Stop, Stop) = (then_continuation, else_continuation) {
                    return Stop;
                }

                let after_block = context.new_block();

                if let ContinueBlock(after_then_block) = then_continuation {
                    context.finish_block(after_then_block, Terminator::Jump(after_block));
                }

                if let ContinueBlock(after_else_block) = else_continuation {
                    context.finish_block(after_else_block, Terminator::Jump(after_block));
                }
                context.seal_block(after_block);

                ContinueBlock(after_block)
            } else {
                let after_block = context.new_block();

                context
                    .finish_block(block_id, Terminator::Branch(cond, then_block, after_block));
                context.seal_block(then_block);

                if let ContinueBlock(after_then_block) = then_continuation {
                    context.finish_block(after_then_block, Terminator::Jump(after_block));
                }

                context.seal_block(after_block);

                ContinueBlock(after_block)
            }
        }
        TypedStmt::While { cond, body } => {
            if let Some(constant) = const_bool(&cond.value) {
                return if !constant {
                    ContinueBlock(block_id)
                } else {
                    let loop_header = context.new_block();
                    context.finish_block(block_id, Terminator::Jump(loop_header));
                    let body_block = context.new_block();
                    context.finish_block(loop_header, Terminator::Jump(body_block));
                    context.seal_block(body_block);
                    let body_continuation =
                        translate_stmt(context, body.value, body_block);
                    if let ContinueBlock(after_body_block) = body_continuation {
                        context.finish_block(after_body_block, Terminator::Jump(loop_header));
                    }
                    context.seal_block(loop_header);
                    Stop
                };
            }
            let loop_header = context.new_block();
            context.finish_block(block_id, Terminator::Jump(loop_header));

            let (cond, condition_exit) =
                translate_expr(context, cond.value, loop_header);

            let cond = context.resolve_alias(cond);

            let after_block = context.new_block();
            let body_block = context.new_block();

            context.finish_block(
                condition_exit,
                Terminator::Branch(cond, body_block, after_block),
            );
            context.seal_block(body_block);
            context.seal_block(after_block);

            let body_continuation = translate_stmt(context, body.value, body_block);
            if let ContinueBlock(after_body_block) = body_continuation {
                context.finish_block(after_body_block, Terminator::Jump(loop_header));
            }

            context.seal_block(loop_header);
            ContinueBlock(after_block)
        }
        TypedStmt::Expr(expr) => {
            let (_, block_id) = translate_expr(context, expr.value, block_id);
            ContinueBlock(block_id)
        }
        TypedStmt::Incr(expr) => {
            let TypedExprKind::Variable(_, var_id) = expr.value.expr else {
                panic!("This should have been caught by the typechecker")
            };
            let (expr, block_id) = translate_expr(context, expr.value, block_id);
            let one = context.emit(block_id, BuildingValue::Int(1), Type::Int);
            let op = context.emit(
                block_id,
                BuildingValue::BinaryOp(BinaryOpCode::Add, expr, one),
                Type::Int,
            );
            context.write_variable(var_id, block_id, op);
            ContinueBlock(block_id)
        }
        TypedStmt::Decr(expr) => {
            let TypedExprKind::Variable(_, var_id) = expr.value.expr else {
                panic!("This should have been caught by the typechecker")
            };
            let (expr, block_id) = translate_expr(context, expr.value, block_id);
            let one = context.emit(block_id, BuildingValue::Int(1), Type::Int);
            let op = context.emit(
                block_id,
                BuildingValue::BinaryOp(BinaryOpCode::Sub, expr, one),
                Type::Int,
            );
            context.write_variable(var_id, block_id, op);
            ContinueBlock(block_id)
        }
    }
}
