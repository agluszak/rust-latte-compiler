use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;
use crate::ast;
use crate::ast::Literal;
use crate::ir::BasicBlockContinuation::{ChangeBlock, Continue, Stop};
use crate::ir::Value::{BinaryOp, Undef};
use crate::typechecker::Type;
use crate::typed_ast::{TypedBlock, TypedDecl, TypedExpr, TypedExprKind, TypedStmt, VariableId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct BlockId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ValueId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum BinaryOpCode {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Gt,
    Lt,
    Gte,
    Lte,
    Eq,
    Neq,
    And,
    Or
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum UnaryOpCode {
    Neg,
    Not
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Phi {
    incoming: BTreeMap<BlockId, ValueId>,
    block: BlockId,
    users: BTreeSet<ValueId>,
}

impl Phi {
    fn new(block: BlockId) -> Phi {
        Phi {
            incoming: BTreeMap::new(),
            block,
            users: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Value {
    Int(i64),
    String(String),
    Bool(bool),
    Call(VariableId, Vec<ValueId>),
    Argument(u32),
    BinaryOp(BinaryOpCode, ValueId, ValueId),
    UnaryOp(UnaryOpCode, ValueId),
    Phi(Phi),
    Rerouted(ValueId),
    Undef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Block {
    instructions: Vec<ValueId>,
    terminator: Option<Terminator>,
    preds: BTreeSet<BlockId>,
    sealed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Terminator {
    Return(ValueId),
    ReturnNoValue,
    Branch(ValueId, BlockId, BlockId),
    Jump(BlockId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IrContext {
    values_in_blocks: BTreeMap<VariableId, BTreeMap<BlockId, ValueId>>,
    variable_types: BTreeMap<VariableId, Type>,
    values: BTreeMap<ValueId, Value>,
    value_types: BTreeMap<ValueId, Type>,
    blocks: BTreeMap<BlockId, Block>,
    incomplete_phis: BTreeMap<BlockId, BTreeMap<VariableId, Phi>>,
}

impl IrContext {
    pub fn new() -> IrContext {
        IrContext {
            values_in_blocks: BTreeMap::new(),
            variable_types: BTreeMap::new(),
            values: BTreeMap::new(),
            value_types: BTreeMap::new(),
            blocks: BTreeMap::new(),
            incomplete_phis: BTreeMap::new(),
        }
    }

    fn add_instruction(&mut self, block: BlockId, value: ValueId) {
        let block = self.blocks.get_mut(&block).unwrap_or_else(|| panic!("Block {:?} not found", block));
        assert!(block.terminator.is_none());
        block.instructions.push(value);
    }

    fn terminated(&self, block_id: BlockId) -> bool {
        self.blocks.get(&block_id).unwrap_or_else(|| panic!("Block {:?} not found", block_id)).terminator.is_some()
    }

    fn add_terminator(&mut self, block_id: BlockId, terminator: Terminator) {
        let mut block = self.blocks.remove(&block_id).unwrap_or_else(|| panic!("Block {:?} not found", block_id));
        assert!(block.terminator.is_none());
        match &terminator {
            Terminator::Return(_) | Terminator::ReturnNoValue => {}
            Terminator::Branch(_, then, else_) => {
                self.add_predecessor(*then, block_id);
                self.add_predecessor(*else_, block_id);
            }
            Terminator::Jump(target) => {
                self.add_predecessor(*target, block_id);
            }
        }
        block.terminator = Some(terminator);
        self.blocks.insert(block_id, block);
    }

    fn add_predecessor(&mut self, block: BlockId, pred: BlockId) {
        let block = self.blocks.get_mut(&block).unwrap_or_else(|| panic!("Block {:?} not found", block));
        assert!(!block.sealed);
        block.preds.insert(pred);
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.insert(id, Block {
            instructions: Vec::new(),
            terminator: None,
            preds: BTreeSet::new(),
            sealed: false,
        });
        id
    }

    pub fn write_variable(&mut self, variable: VariableId, block: BlockId, value: ValueId) {
        let value_ty = self.value_types.get(&value).cloned().unwrap_or_else(|| panic!("Value {:?} not found", value));
        self.variable_types.insert(variable, value_ty);
        self.values_in_blocks.entry(variable).or_default().insert(block, value);
    }

    pub fn new_value(&mut self, value: Value, ty: Type) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.insert(id, value);
        self.value_types.insert(id, ty);
        id
    }

    pub fn read_variable(&mut self, variable: VariableId, block_id: BlockId) -> ValueId {
        if let Some(value) = self.values_in_blocks.get(&variable).and_then(|map| map.get(&block_id)) {
            *value
        } else {
            let block = self.blocks.get(&block_id).expect(&format!("Block {:?} not found", block_id));
            let ty = self.variable_types.get(&variable).cloned().unwrap_or_else(|| panic!("Variable {:?} not found", variable));
            let val = if !block.sealed {
                // Incomplete CFG
                let phi = Phi::new(block_id);
                self.incomplete_phis.entry(block_id).or_default().insert(variable, phi.clone());

                self.new_value(Value::Phi(phi), ty)
            } else if block.preds.len() == 1 {
                // Optimize the common case of a single predecessor: no phi needed
                let pred = block.preds.iter().next().unwrap();
                self.read_variable(variable, *pred)
            } else {
                // Break potential cycles with operandless phi
                let phi = Phi::new(block_id);
                let val = self.new_value(Value::Phi(phi), ty);
                self.write_variable(variable, block_id, val);
                let val = self.add_phi_operands(variable, val);
                val
            };
            self.write_variable(variable, block_id, val);
            val
        }
    }

    fn add_phi_operands(&mut self, variable: VariableId, phi_id: ValueId) -> ValueId {
        let mut phi = self.get_phi(phi_id);
        let block = self.blocks.get(&phi.block).expect(&format!("Block {:?} not found", phi.block));
        for pred in block.preds.clone() {
            let pred_val = self.read_variable(variable, pred);
            phi.incoming.insert(pred, pred_val);
        }
        self.values.insert(phi_id, Value::Phi(phi));
        self.try_remove_trivial_phi(phi_id)
    }

    fn get_phi(&self, phi_id: ValueId) -> Phi {
        if let Value::Phi(phi) = self.values[&phi_id].clone() {
            phi
        } else {
            panic!("Value {:?} is not a phi", phi_id)
        }
    }

    pub fn seal_block(&mut self, block_id: BlockId) {
        let mut block = self.blocks.get(&block_id).cloned().expect(&format!("Block {:?} not found", block_id));
        block.sealed = true;
        self.blocks.insert(block_id, block);
        if let Some(incomplete_phis) = self.incomplete_phis.remove(&block_id) {
            for (variable, phi) in incomplete_phis {
                let ty = self.variable_types.get(&variable).cloned().unwrap_or_else(|| panic!("Variable {:?} not found", variable));
                let val = self.new_value(Value::Phi(phi), ty);
                self.add_phi_operands(variable, val);
            }
        }
    }

    fn try_remove_trivial_phi(&mut self, phi_id: ValueId) -> ValueId {
        let mut phi = self.get_phi(phi_id);
        let mut same = None;
        for &op in phi.incoming.values() {
            if let Some(same) = same {
                if same != op && same != phi_id {
                    // This phi merges at least two different values, so it's not trivial
                    return phi_id;
                } else {
                    continue;
                }
            } else {
                same = Some(op);
            }
        }
        if same.is_none() {
            // This phi is unreachable or in the entry block
            let ty = self.value_types.get(&phi_id).cloned().unwrap_or_else(|| panic!("Value {:?} not found", phi_id));
            let undef = self.new_value(Undef, ty);
            same = Some(undef);
        }
        // Remember all users except the phi itself
        phi.users.remove(&phi_id);
        // Reroute all uses of phi to same and remove phi
        self.values.insert(phi_id, Value::Rerouted(same.unwrap()));

        // Try to recursively remove all phi users, which might have become trivial
        for &user in &phi.users {
            if let Value::Phi(_) = self.values[&user] {
                self.try_remove_trivial_phi(user);
            }
        }
        same.unwrap()
    }
}

#[derive(Debug, Clone, Copy,  PartialEq, Eq)]
enum BasicBlockContinuation {
    Continue,
    ChangeBlock(BlockId),
    Stop
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ir {
    ir: IrContext,
    ty: Type,
}

impl Ir {
    pub fn dump(&self) {
        // print values
        for value in self.ir.values.keys() {
            println!("Value {:?}: {:?}", value, self.ir.values.get(value).unwrap());
        }

        for (var, values) in &self.ir.values_in_blocks {
            println!("Var {:?}", var);
            for (block, value) in values {
                println!("  {:?} - {:?}", block, value);
            }
        }

        for block in self.ir.blocks.keys() {
            println!("Block {:?}", block);
            let block = self.ir.blocks.get(block).unwrap();
            let sealed = if block.sealed { "sealed" } else { "unsealed" };
            println!("  {}", sealed);
            let preds = block.preds.iter().map(|id| format!("{:?}", id)).collect::<Vec<_>>().join(", ");
            println!("  preds: {:?}", preds);
            for instr in &block.instructions {
                println!("    {:?}: {:?}", instr.0, self.ir.values.get(instr).unwrap());
            }
            if let Some(terminator) = &block.terminator {
                println!("  {:?}", terminator);
            }

        }
    }

    pub fn translate_function(decl: TypedDecl) -> Ir {
        let mut ir = IrContext::new();
        let ty = decl.ty();
        match decl {
            TypedDecl::Var { .. } => panic!("Global variables are not supported yet"),
            TypedDecl::Fn { name, body, args, .. } => {
                let entry_block = ir.new_block();
                ir.seal_block(entry_block);
                for (arg, i) in args.into_iter().zip(0..) {
                    let undef = ir.new_value(Value::Argument(i), arg.value.ty);
                    ir.write_variable(arg.value.var_id, entry_block, undef);
                }

                let continuation = Self::translate_block(&mut ir, body.value, entry_block);
                if let ChangeBlock(block_id) = continuation {
                    ir.add_terminator(block_id, Terminator::ReturnNoValue);
                }
            }
        }
        Ir {
            ir,
            ty
        }
    }

    fn translate_expr(context: &mut IrContext, expr: TypedExpr, block_id: BlockId) -> ValueId {
        let id = match expr.expr {
            TypedExprKind::Variable(_, id) => context.read_variable(id, block_id),
            TypedExprKind::Literal(lit) => match lit {
                Literal::Int(i) => context.new_value(Value::Int(i), Type::Int),
                Literal::String(s) => context.new_value(Value::String(s), Type::LatteString),
                Literal::Bool(b) => context.new_value(Value::Bool(b), Type::Bool)
            }
            TypedExprKind::Binary { lhs, op, rhs } => {
                let lhs = Self::translate_expr(context, lhs.value, block_id);
                let rhs = Self::translate_expr(context, rhs.value, block_id);
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
                    ast::BinaryOp::And => BinaryOpCode::And,
                    ast::BinaryOp::Or => BinaryOpCode::Or,
                };
                context.new_value(Value::BinaryOp(op, lhs, rhs), expr.ty)

            }
            TypedExprKind::Unary { op, expr: target } => {
                let val = Self::translate_expr(context, target.value, block_id);
                let op = match op.value {
                    ast::UnaryOp::Neg => UnaryOpCode::Neg,
                    ast::UnaryOp::Not => UnaryOpCode::Not,
                };
                context.new_value(Value::UnaryOp(op, val), expr.ty)
            }
            TypedExprKind::Application { target, args } => {
                let TypedExprKind::Variable(_, id) = target.value.expr else {
                    panic!("This should have been caught by the typechecker")
                };
                let args = args.into_iter().map(|arg| Self::translate_expr(context, arg.value, block_id)).collect();
                context.new_value(Value::Call(id, args), expr.ty)
            }
        };
        context.add_instruction(block_id, id);
        id
    }

    fn translate_block(context: &mut IrContext, block: TypedBlock, block_id: BlockId) -> BasicBlockContinuation {
        let mut block_id = block_id;
        let mut final_continuation = Continue;
        for stmt in block.0 {
            let continuation = Self::translate_stmt(context, stmt.value, block_id);
            match continuation {
                Continue => final_continuation = Continue,
                ChangeBlock(new_block_id)=> {
                    block_id = new_block_id;
                    final_continuation = ChangeBlock(new_block_id);
                }
                Stop => {
                    return Stop
                }
            }
        }
        final_continuation
    }

    fn translate_stmt(context: &mut IrContext, stmt: TypedStmt, block_id: BlockId) -> BasicBlockContinuation {
        match stmt {
            TypedStmt::Empty => Continue,
            TypedStmt::Block(block) => Self::translate_block(context, block.value, block_id),
            TypedStmt::Decl(decl) => match decl.value {
                TypedDecl::Var { items, .. } => {
                    for item in items {
                        if let Some(expr) = item.value.init {
                            let expr = Self::translate_expr(context, expr.value, block_id);
                            context.write_variable(item.value.var_id, block_id, expr);
                        } else {
                            let undef = context.new_value(Undef, item.value.ty);
                            context.add_instruction(block_id, undef); // TODO or some default?
                            context.write_variable(item.value.var_id, block_id, undef);
                        }
                    }
                    Continue
                }
                TypedDecl::Fn { .. } => panic!("Nested functions are not supported yet")
            }
            TypedStmt::Assignment { target, target_id, expr } => {
                let expr = Self::translate_expr(context, expr.value, block_id);
                context.write_variable(target_id, block_id, expr);
                Continue
            }
            TypedStmt::Return(expr) => {
                let expr = expr.map(|expr| Self::translate_expr(context, expr.value, block_id));
                if let Some(expr) = expr {
                    context.add_terminator(block_id, Terminator::Return(expr));
                } else {
                    context.add_terminator(block_id, Terminator::ReturnNoValue);
                }
                Stop
            }
            TypedStmt::If { cond, then, otherwise } => {
                let cond = Self::translate_expr(context, cond.value, block_id);
                let after_block = context.new_block();
                let then_block = context.new_block();
                let then_continuation = Self::translate_stmt(context, then.value, then_block);
                if let Some(otherwise) = otherwise {
                    let else_block = context.new_block();
                    let else_continuation = Self::translate_stmt(context, otherwise.value, else_block);
                    context.add_terminator(block_id, Terminator::Branch(cond, then_block, else_block));
                    context.seal_block(else_block);
                    context.seal_block(then_block);

                    if let (Stop, Stop) = (then_continuation, else_continuation) {
                        return Stop
                    }

                    if let ChangeBlock(after_then_block) = then_continuation {
                        context.add_terminator(after_then_block, Terminator::Jump(after_block));
                    } else if let Continue = then_continuation {
                        context.add_terminator(then_block, Terminator::Jump(after_block));
                    }

                    if let ChangeBlock(after_else_block) = else_continuation {
                        context.add_terminator(after_else_block, Terminator::Jump(after_block));
                    } else if let Continue = else_continuation {
                        context.add_terminator(else_block, Terminator::Jump(after_block));
                    }
                    context.seal_block(after_block);

                    ChangeBlock(after_block)
                } else {
                    context.add_terminator(block_id, Terminator::Branch(cond, then_block, after_block));
                    context.seal_block(then_block);
                    context.seal_block(after_block);

                    if let ChangeBlock(after_then_block) = then_continuation {
                        context.add_terminator(after_then_block, Terminator::Jump(after_block));
                    } else if let Continue = then_continuation {
                        context.add_terminator(then_block, Terminator::Jump(after_block));
                    }

                    ChangeBlock(after_block)
                }
            }
            TypedStmt::While { cond, body } => {
                let cond_block = context.new_block();
                let body_block = context.new_block();
                let after_block = context.new_block();

                context.add_terminator(block_id, Terminator::Jump(cond_block));

                let cond = Self::translate_expr(context, cond.value, block_id);
                context.add_terminator(cond_block, Terminator::Branch(cond, body_block, after_block));
                context.seal_block(body_block);

                let body_continuation = Self::translate_stmt(context, body.value, body_block);
                if let ChangeBlock(after_body_block) = body_continuation {
                    context.add_terminator(after_body_block, Terminator::Jump(cond_block));
                } else if let Continue = body_continuation {
                    context.add_terminator(body_block, Terminator::Jump(cond_block));
                }

                context.seal_block(cond_block);
                ChangeBlock(after_block)
            }
            TypedStmt::Expr(expr) => {
                let expr = Self::translate_expr(context, expr.value, block_id);
                Continue
            }
            TypedStmt::Incr(expr) => {
                let expr = Self::translate_expr(context, expr.value, block_id);
                let one = context.new_value(Value::Int(1), Type::Int);
                context.add_instruction(block_id, one);
                let op = context.new_value(Value::BinaryOp(BinaryOpCode::Add, expr, one), Type::Int);
                context.add_instruction(block_id, op);
                Continue
            }
            TypedStmt::Decr(expr) => {
                let expr = Self::translate_expr(context, expr.value, block_id);
                let one = context.new_value(Value::Int(1), Type::Int);
                context.add_instruction(block_id, one);
                let op = context.new_value(Value::BinaryOp(BinaryOpCode::Sub, expr, one), Type::Int);
                context.add_instruction(block_id, op);
                Continue
            }
        }
    }
}
