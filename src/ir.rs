use std::collections::{BTreeMap, BTreeSet};
use crate::ast;
use crate::ast::Literal;
use crate::ir::Value::Undef;
use crate::typed_ast::{TypedBlock, TypedDecl, TypedExpr, TypedExprKind, TypedStmt, VariableId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct BlockId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ValueId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum OpCode {
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
    Literal(i64),
    Op(OpCode, ValueId, ValueId),
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
    values: BTreeMap<ValueId, Value>,
    blocks: BTreeMap<BlockId, Block>,
    incomplete_phis: BTreeMap<BlockId, BTreeMap<VariableId, Phi>>,
}

impl IrContext {
    pub fn new() -> IrContext {
        IrContext {
            values_in_blocks: BTreeMap::new(),
            values: BTreeMap::new(),
            blocks: BTreeMap::new(),
            incomplete_phis: BTreeMap::new(),
        }
    }

    fn add_instruction(&mut self, block: BlockId, value: ValueId) {
        let block = self.blocks.get_mut(&block).unwrap_or_else(|| panic!("Block {:?} not found", block));
        assert!(block.terminator.is_none());
        block.instructions.push(value);
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
        self.values_in_blocks.entry(variable).or_default().insert(block, value);
    }

    pub fn new_value(&mut self, value: Value) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.insert(id, value);
        id
    }

    pub fn read_variable(&mut self, variable: VariableId, block_id: BlockId) -> ValueId {
        if let Some(value) = self.values_in_blocks.get(&variable).and_then(|map| map.get(&block_id)) {
            *value
        } else {
            let block = self.blocks.get(&block_id).expect(&format!("Block {:?} not found", block_id));
            let val = if !block.sealed {
                // Incomplete CFG
                let phi = Phi::new(block_id);
                self.incomplete_phis.entry(block_id).or_default().insert(variable, phi.clone());
                self.new_value(Value::Phi(phi))
            } else if block.preds.len() == 1 {
                // Optimize the common case of a single predecessor: no phi needed
                let pred = block.preds.iter().next().unwrap();
                self.read_variable(variable, *pred)
            } else {
                // Break potential cycles with operandless phi
                let phi = Phi::new(block_id);
                let val = self.new_value(Value::Phi(phi));
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
                let val = self.new_value(Value::Phi(phi));
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
            let undef = self.new_value(Undef);
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionContext {
    ir: IrContext,
}

impl FunctionContext {
    pub fn new() -> FunctionContext {
        FunctionContext {
            ir: IrContext::new(),
        }
    }

    pub fn dump(&self) {
        for block in self.ir.blocks.keys() {
            println!("Block {:?}", block);
            let block = self.ir.blocks.get(block).unwrap();
            let sealed = if block.sealed { "sealed" } else { "unsealed" };
            println!("  {:?}", sealed);
            let preds = block.preds.iter().map(|id| format!("{:?}", id)).collect::<Vec<_>>().join(", ");
            println!("  preds: {:?}", preds);
            for instr in &block.instructions {
                println!("  {:?}: {:?}", instr.0, self.ir.values.get(instr).unwrap());
            }
            if let Some(terminator) = &block.terminator {
                println!("  {:?}", terminator);
            }

        }
    }

    pub fn translate_function(&mut self, decl: TypedDecl) {
        match decl {
            TypedDecl::Var { .. } => {}
            TypedDecl::Fn { name, body, args, .. } => {
                let entry_block = self.ir.new_block();
                for arg in args {
                    let undef = self.ir.new_value(Undef);
                    self.ir.write_variable(arg.value.var_id, entry_block, undef);
                }
                self.ir.seal_block(entry_block);
                let exit_block = self.ir.new_block();
                self.translate_block(body.value, entry_block, exit_block);
                self.ir.seal_block(exit_block);
            }
        }
    }

    fn translate_expr(&mut self, expr: TypedExpr, block_id: BlockId) -> ValueId {
        match expr.expr {
            TypedExprKind::Variable(_, id) => self.ir.read_variable(id, block_id),
            TypedExprKind::Literal(lit) => if let Literal::Int(i) = lit {
                self.ir.new_value(Value::Literal(i))
            } else {
                todo!("Expected int literal")
            }
            TypedExprKind::Binary { lhs, op, rhs } => {
                let lhs = self.translate_expr(lhs.value, block_id);
                let rhs = self.translate_expr(rhs.value, block_id);
                let op = match op.value {
                    ast::BinaryOp::Add => OpCode::Add,
                    ast::BinaryOp::Sub => OpCode::Sub,
                    ast::BinaryOp::Mul => OpCode::Mul,
                    ast::BinaryOp::Div => OpCode::Div,
                    ast::BinaryOp::Mod => OpCode::Mod,
                    ast::BinaryOp::Gt => OpCode::Gt,
                    ast::BinaryOp::Lt => OpCode::Lt,
                    ast::BinaryOp::Gte => OpCode::Gte,
                    ast::BinaryOp::Lte => OpCode::Lte,
                    ast::BinaryOp::Eq => OpCode::Eq,
                    ast::BinaryOp::Neq => OpCode::Neq,
                    ast::BinaryOp::And => todo!(),
                    ast::BinaryOp::Or => todo!(),
                };
                self.ir.new_value(Value::Op(op, lhs, rhs))
            }
            TypedExprKind::Unary { op, expr } => {
                let expr = self.translate_expr(expr.value, block_id);
                match op.value {
                    ast::UnaryOp::Neg => todo!(),
                    ast::UnaryOp::Not => todo!(),
                }
            }
            TypedExprKind::Application { .. } => todo!()
        }
    }

    fn translate_block(&mut self, block: TypedBlock, block_id: BlockId, succeeding_block: BlockId) {
        let after_block = self.ir.new_block();
        for stmt in block.0 {
            self.translate_stmt(stmt.value, block_id, after_block);
        }
        self.ir.seal_block(after_block);
        self.ir.add_terminator(after_block, Terminator::Jump(succeeding_block));
    }

    fn translate_stmt(&mut self, stmt: TypedStmt, block_id: BlockId, succeeding_block: BlockId) {
        match stmt {
            TypedStmt::Empty => {}
            TypedStmt::Block(block) => self.translate_block(block.value, block_id, succeeding_block),
            TypedStmt::Decl(decl) => match decl.value {
                TypedDecl::Var { items, .. } =>
                    for item in items {
                        if let Some(expr) = item.value.init {
                            let expr = self.translate_expr(expr.value, block_id);
                            self.ir.write_variable(item.value.var_id, block_id, expr);
                            self.ir.add_instruction(block_id, expr);
                        } else {
                            let undef = self.ir.new_value(Undef);
                            self.ir.write_variable(item.value.var_id, block_id, undef);
                        }
                    }
                TypedDecl::Fn { .. } => todo!()
            }
            TypedStmt::Assignment { target, target_id, expr } => {
                let expr = self.translate_expr(expr.value, block_id);
                self.ir.write_variable(target_id, block_id, expr);
                self.ir.add_instruction(block_id, expr);
            }
            TypedStmt::Return(expr) => {
                let expr = expr.map(|expr| self.translate_expr(expr.value, block_id));
                if let Some(expr) = expr {
                    self.ir.add_terminator(block_id, Terminator::Return(expr));
                } else {
                    self.ir.add_terminator(block_id, Terminator::ReturnNoValue);
                }
            }
            TypedStmt::If { cond, then, otherwise } => {
                let cond = self.translate_expr(cond.value, block_id);
                let then_block = self.ir.new_block();
                self.translate_stmt(then.value, then_block, succeeding_block);
                if let Some(otherwise) = otherwise {
                    let else_block = self.ir.new_block();
                    self.translate_stmt(otherwise.value, else_block, succeeding_block);
                    self.ir.add_terminator(block_id, Terminator::Branch(cond, then_block, else_block));
                    self.ir.seal_block(else_block);
                } else {
                    self.ir.add_terminator(block_id, Terminator::Branch(cond, then_block, succeeding_block));
                }
                self.ir.seal_block(then_block);
            }
            TypedStmt::While { cond, body } => {
                let cond_block = self.ir.new_block();
                let body_block = self.ir.new_block();

                self.ir.add_terminator(block_id, Terminator::Jump(cond_block));

                let cond = self.translate_expr(cond.value, block_id);
                self.ir.add_terminator(cond_block, Terminator::Branch(cond, body_block, succeeding_block));
                self.ir.seal_block(body_block);

                self.translate_stmt(body.value, body_block, cond_block);
                self.ir.seal_block(cond_block);
            }
            TypedStmt::Expr(expr) => {
                let expr = self.translate_expr(expr.value, block_id);
                self.ir.add_instruction(block_id, expr);
            }
            TypedStmt::Incr(expr) => {
                let expr = self.translate_expr(expr.value, block_id);
                let one = self.ir.new_value(Value::Literal(1));
                let op = self.ir.new_value(Value::Op(OpCode::Add, expr, one));
                self.ir.add_instruction(block_id, op);
            }
            TypedStmt::Decr(expr) => {
                let expr = self.translate_expr(expr.value, block_id);
                let one = self.ir.new_value(Value::Literal(1));
                let op = self.ir.new_value(Value::Op(OpCode::Sub, expr, one));
                self.ir.add_instruction(block_id, op);
            }
        }
    }
}
