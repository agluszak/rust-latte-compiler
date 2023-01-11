use crate::ast::Ident;
use crate::ir::{
    BlockId, Context, Function, IrFunctionType, IrType, SsaName, Terminator, Value, ValueBuilder,
    VariableId,
};
use crate::typechecker::Type;
use crate::typed_ast::{TypedDecl, TypedExpr, TypedExprKind, TypedStmt};
use crate::{ast, ir};
use std::collections::HashMap;
use std::thread::current;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Environment {
    variables: HashMap<ast::Ident, VariableId>,
    parent: Option<Box<Environment>>,
}

impl Environment {
    fn global() -> Self {
        Environment {
            variables: HashMap::new(),
            parent: None,
        }
    }

    fn local(&self) -> Self {
        Environment {
            variables: HashMap::new(),
            parent: Some(Box::new(self.clone())),
        }
    }

    fn get_variable(&self, ident: &ast::Ident) -> Option<VariableId> {
        self.variables.get(ident).cloned().or_else(|| {
            self.parent
                .as_ref()
                .and_then(|parent| parent.get_variable(ident))
        })
    }

    fn add_variable(&mut self, ident: ast::Ident, id: VariableId) {
        self.variables.insert(ident, id);
    }
}

fn lower_type(ty: &Type) -> IrType {
    match ty {
        Type::Int => IrType::Int,
        Type::Bool => IrType::Bool,
        Type::LatteString => IrType::String,
        Type::Void => IrType::Void,
        Type::Function(_, _) => unimplemented!("Function types are not supported"),
    }
}

fn lower_expr(
    function: &mut ir::Function,
    env: &mut Environment,
    expr: TypedExpr,
    current_block: BlockId,
    context: &Context,
) -> SsaName {
    let ty = lower_type(&expr.ty);
    let ssa_name = match expr.expr {
        TypedExprKind::Variable(name) => {
            let variable = env.get_variable(&name).unwrap();
            function.read_variable(variable, current_block)
        }
        TypedExprKind::Literal(lit) => {
            let value = match lit {
                ast::Literal::Int(i) => ValueBuilder::constant_int(i),
                ast::Literal::String(s) => ValueBuilder::constant_string(s),
                ast::Literal::Bool(b) => ValueBuilder::constant_bool(b),
            };
            let value_id = function.new_value(value, ty);
            function.new_ssa_name(value_id)
        }
        TypedExprKind::Binary { lhs, op, rhs } => {
            let lhs = lower_expr(function, env, lhs.value, current_block, context);
            let rhs = lower_expr(function, env, rhs.value, current_block, context);
            let value = match op.value {
                ast::BinaryOp::Add => ValueBuilder::add(lhs, rhs),
                ast::BinaryOp::Sub => ValueBuilder::sub(lhs, rhs),
                ast::BinaryOp::Mul => ValueBuilder::mul(lhs, rhs),
                ast::BinaryOp::Div => ValueBuilder::div(lhs, rhs),
                ast::BinaryOp::Mod => ValueBuilder::mod_(lhs, rhs),
                ast::BinaryOp::Eq => ValueBuilder::eq(lhs, rhs),
                ast::BinaryOp::Neq => ValueBuilder::neq(lhs, rhs),
                ast::BinaryOp::Lt => ValueBuilder::lt(lhs, rhs),
                ast::BinaryOp::Lte => ValueBuilder::lte(lhs, rhs),
                ast::BinaryOp::Gt => ValueBuilder::gt(lhs, rhs),
                ast::BinaryOp::Gte => ValueBuilder::gte(lhs, rhs),
                ast::BinaryOp::And => ValueBuilder::and(lhs, rhs),
                ast::BinaryOp::Or => ValueBuilder::or(lhs, rhs),
            };
            let value_id = function.new_value(value, ty);
            function.new_ssa_name(value_id)
        }
        TypedExprKind::Unary { op, expr } => {
            let expr = lower_expr(function, env, expr.value, current_block, context);
            let value = match op.value {
                ast::UnaryOp::Neg => ValueBuilder::neg(expr),
                ast::UnaryOp::Not => ValueBuilder::not(expr),
            };
            let value_id = function.new_value(value, ty);
            function.new_ssa_name(value_id)
        }
        TypedExprKind::Application { target, args } => {
            let function_name = expect_variable(target.value);
            let function_id = context.get_function(&function_name);
            let args = args
                .into_iter()
                .map(|arg| lower_expr(function, env, arg.value, current_block, context))
                .collect();
            let call = function.new_call(function_id, args);
            function.new_ssa_name(call)
        }
    };
    function.define_ssa_name(current_block, ssa_name);
    ssa_name
}

fn expect_variable(expr: TypedExpr) -> ast::Ident {
    match expr.expr {
        TypedExprKind::Variable(name) => name,
        _ => panic!("Expected variable"),
    }
}

fn lower_stmt(
    function: &mut ir::Function,
    env: &mut Environment,
    stmt: TypedStmt,
    current_block: BlockId,
    context: &Context,
) -> BlockId {
    match stmt {
        TypedStmt::Block(block) => {
            let mut local_env = env.local();
            let mut current_block = current_block;
            for stmt in block.value.0 {
                current_block =
                    lower_stmt(function, &mut local_env, stmt.value, current_block, context);
            }
            current_block
        }
        TypedStmt::Empty => current_block,
        TypedStmt::Expr(expr) => {
            lower_expr(function, env, expr.value, current_block, context);
            current_block
        }
        TypedStmt::If {
            cond,
            then,
            otherwise,
        } => {
            let condition = lower_expr(function, env, cond.value, current_block, context);
            function.define_ssa_name(current_block, condition);
            let then_block = function.new_block();
            function.declare_predecessor(then_block, current_block);
            function.seal_block(then_block);
            let then_out_block = lower_stmt(function, env, then.value, then_block, context);

            let merge_block = function.new_block();
            function.declare_predecessor(merge_block, then_out_block);

            if let Some(otherwise) = otherwise {
                let otherwise_block = function.new_block();
                function.declare_predecessor(otherwise_block, current_block);
                function.seal_block(otherwise_block);
                let otherwise_out_block =
                    lower_stmt(function, env, otherwise.value, otherwise_block, context);
                function.declare_predecessor(merge_block, otherwise_out_block);
                let terminator = Terminator::ConditionalJump {
                    condition,
                    then_block,
                    otherwise_block,
                };
                function.fill_block(current_block, terminator);
            } else {
                function.declare_predecessor(merge_block, current_block);
                let terminator = Terminator::ConditionalJump {
                    condition,
                    then_block,
                    otherwise_block: merge_block,
                };
                function.fill_block(current_block, terminator);
            }

            function.seal_block(merge_block);

            merge_block
        }
        TypedStmt::Decl(decl) => lower_var_decl(function, env, decl.value, current_block, context),
        TypedStmt::Assignment { target, expr } => {
            let ssa_name = lower_expr(function, env, expr.value, current_block, context);
            function.define_ssa_name(current_block, ssa_name);
            let variable = env.get_variable(&target.value).unwrap();
            function.set_variable(variable, current_block, ssa_name);
            current_block
        }
        TypedStmt::Return(Some(ret)) => {
            let ssa_name = lower_expr(function, env, ret.value, current_block, context);
            function.define_ssa_name(current_block, ssa_name);
            let terminator = Terminator::Return(ssa_name);
            function.fill_block(current_block, terminator);
            current_block
        }
        TypedStmt::Return(None) => {
            let value = ValueBuilder::void();
            let value = function.new_value(value, IrType::Void);
            let ssa_name = function.new_ssa_name(value);
            function.define_ssa_name(current_block, ssa_name); // TODO: Is this necessary?
            let terminator = Terminator::Return(ssa_name);
            function.fill_block(current_block, terminator);
            current_block
        }
        TypedStmt::While { cond, body } => {
            let condition_block = function.new_block();
            function.declare_predecessor(condition_block, current_block);

            let condition = lower_expr(function, env, cond.value, condition_block, context);
            function.define_ssa_name(condition_block, condition);

            let exit_block = function.new_block();
            function.declare_predecessor(exit_block, condition_block);
            function.seal_block(exit_block);

            let body_block = function.new_block();
            function.declare_predecessor(body_block, condition_block);
            function.seal_block(body_block);
            let body_out_block = lower_stmt(function, env, body.value, body_block, context);
            function.declare_predecessor(condition_block, body_out_block);
            let terminator = Terminator::jump(condition_block);
            function.fill_block(body_out_block, terminator);

            let terminator = Terminator::conditional_jump(condition, body_block, exit_block);
            function.fill_block(condition_block, terminator);

            exit_block
        }
        op @ TypedStmt::Incr(_) | op @ TypedStmt::Decr(_) => {
            let is_incr = matches!(op, TypedStmt::Incr(_));

            let target = match op {
                TypedStmt::Incr(target) => target,
                TypedStmt::Decr(target) => target,
                _ => unreachable!(),
            };

            let variable = expect_variable(target.value);
            let variable = env.get_variable(&variable).unwrap();
            let one = ValueBuilder::constant_int(1);
            let one = function.new_value(one, IrType::Int);
            let one = function.new_ssa_name(one);
            function.define_ssa_name(current_block, one);
            let current_value = function.read_variable(variable, current_block);
            let add = if is_incr {
                ValueBuilder::add(current_value, one)
            } else {
                ValueBuilder::sub(current_value, one)
            };
            let add = function.new_value(add, IrType::Int);
            let add = function.new_ssa_name(add);
            function.define_ssa_name(current_block, add);
            function.set_variable(variable, current_block, add);

            current_block
        }
    }
}

fn lower_var_decl(
    function: &mut Function,
    env: &mut Environment,
    decl: TypedDecl,
    current_block: BlockId,
    context: &Context,
) -> BlockId {
    let decl = match decl {
        TypedDecl::Var { ty, items } => {
            let ty = lower_type(&ty);
            for item in items {
                let name = item.value.ident.value;
                let init = item.value.init;
                if let Some(init) = init {
                    let init = lower_expr(function, env, init.value, current_block, context);
                    function.define_ssa_name(current_block, init);
                    let variable = function.new_variable(ty.clone());
                    function.set_variable(variable, current_block, init);
                    env.add_variable(name, variable);
                } else {
                    // Zero-initialize
                    let init = ValueBuilder::zero_value(&ty);
                    let init = function.new_value(init, ty.clone());
                    let init = function.new_ssa_name(init);
                    function.define_ssa_name(current_block, init);
                    let variable = function.new_variable(ty.clone());
                    function.set_variable(variable, current_block, init);
                    env.add_variable(name, variable);
                }
            }
        }
        _ => unreachable!(),
    };
    current_block
}

pub fn lower_fn_decl(decl: TypedDecl) {
    match decl {
        TypedDecl::Fn {
            return_type,
            name,
            args,
            body,
        } => {
            let return_ty = lower_type(&return_type);
            let arg_names: Vec<Ident> = args
                .iter()
                .map(|arg| arg.value.name.value.clone())
                .collect();
            // FIXME
            let args: Vec<IrType> = args
                .into_iter()
                .map(|arg| lower_type(&arg.value.ty))
                .collect();
            let function_type = IrFunctionType {
                return_ty,
                args: args.clone(),
            };
            // TODO: insert self into context? (not here)

            let mut context = ir::Context::default();
            let mut function = ir::Function::new(&context);
            let mut env = Environment::global();

            let start_block = function.new_block();
            function.seal_block(start_block);

            for (i, (name, ty)) in arg_names.into_iter().zip(args).enumerate() {
                let id = function.new_variable(ty.clone());
                env.add_variable(name, id);
                let arg = function.new_value(ValueBuilder::arg(i as u32), ty.clone());
                let arg = function.new_ssa_name(arg);
                function.set_variable(id, start_block, arg);
            }

            for stmt in body.value.0 {
                lower_stmt(&mut function, &mut env, stmt.value, start_block, &context);
            }

            function.dump()
        }
        _ => panic!("Expected function declaration"),
    }
}
