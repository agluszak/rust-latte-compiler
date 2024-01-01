use crate::ast;
use crate::ast::Literal;
use crate::ir::BasicBlockContinuation::{ChangeBlock, Continue, Stop};
use crate::ir::Value::Undef;
use crate::typechecker::Type;
use crate::typed_ast::{TypedBlock, TypedDecl, TypedExpr, TypedExprKind, TypedStmt, VariableId};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(u32);

impl Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(u32);

impl Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BinaryOpCode {
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
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UnaryOpCode {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi {
    incoming: BTreeMap<BlockId, ValueId>,
    block: BlockId,
    users: BTreeSet<ValueId>,
}

impl Phi {
    pub fn incoming(&self) -> impl Iterator<Item = (BlockId, ValueId)> + '_ {
        self.incoming.iter().map(|(block, value)| (*block, *value))
    }
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
pub enum Value {
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

impl Block {
    fn ready(self) -> ReadyBlock {
        assert!(self.sealed);
        assert!(self.terminator.is_some());
        ReadyBlock {
            instructions: self.instructions,
            terminator: self.terminator.unwrap(),
            preds: self.preds,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Return(ValueId),
    ReturnNoValue,
    Branch(ValueId, BlockId, BlockId),
    Jump(BlockId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrContext {
    variable_names: BTreeMap<VariableId, String>,
    values_in_blocks: BTreeMap<VariableId, BTreeMap<BlockId, ValueId>>,
    variable_types: BTreeMap<VariableId, Type>,
    values: BTreeMap<ValueId, Value>,
    value_types: BTreeMap<ValueId, Type>,
    blocks: BTreeMap<BlockId, Block>,
    incomplete_phis: BTreeMap<BlockId, BTreeMap<VariableId, Phi>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyBlock {
    pub instructions: Vec<ValueId>,
    pub terminator: Terminator,
    pub preds: BTreeSet<BlockId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyIr {
    pub values: BTreeMap<ValueId, Value>,
    pub types: BTreeMap<ValueId, Type>,
    pub blocks: BTreeMap<BlockId, ReadyBlock>,
}

impl IrContext {
    pub fn new() -> IrContext {
        IrContext {
            variable_names: BTreeMap::new(),
            values_in_blocks: BTreeMap::new(),
            variable_types: BTreeMap::new(),
            values: BTreeMap::new(),
            value_types: BTreeMap::new(),
            blocks: BTreeMap::new(),
            incomplete_phis: BTreeMap::new(),
        }
    }

    pub fn ready(self) -> ReadyIr {
        assert!(self.incomplete_phis.is_empty());
        let blocks = self
            .blocks
            .into_iter()
            .map(|(id, block)| (id, block.ready()))
            .collect();
        ReadyIr {
            values: self.values,
            types: self.value_types,
            blocks,
        }
    }

    fn add_instruction(&mut self, block: BlockId, value: ValueId) {
        let block = self
            .blocks
            .get_mut(&block)
            .unwrap_or_else(|| panic!("Block {:?} not found", block));
        assert!(block.terminator.is_none());
        block.instructions.push(value);
    }

    fn pop_instruction(&mut self, block: BlockId) {
        let block = self
            .blocks
            .get_mut(&block)
            .unwrap_or_else(|| panic!("Block {:?} not found", block));
        assert!(block.terminator.is_none());
        block.instructions.pop().unwrap_or_else(|| panic!("Block {:?} is empty", block));
    }

    fn add_terminator(&mut self, block_id: BlockId, terminator: Terminator) {
        let mut block = self
            .blocks
            .remove(&block_id)
            .unwrap_or_else(|| panic!("Block {:?} not found", block_id));
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

    fn terminated(&self, block_id: BlockId) -> bool {
        self.blocks.get(&block_id).unwrap().terminator.is_some()
    }

    fn add_predecessor(&mut self, block: BlockId, pred: BlockId) {
        let block = self
            .blocks
            .get_mut(&block)
            .unwrap_or_else(|| panic!("Block {:?} not found", block));
        assert!(!block.sealed);
        block.preds.insert(pred);
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.insert(
            id,
            Block {
                instructions: Vec::new(),
                terminator: None,
                preds: BTreeSet::new(),
                sealed: false,
            },
        );
        id
    }

    pub fn write_variable(&mut self, variable: VariableId, block: BlockId, value: ValueId) {
        let value_ty = self
            .value_types
            .get(&value)
            .cloned()
            .unwrap_or_else(|| panic!("Value {:?} not found", value));
        self.variable_types.insert(variable, value_ty);
        self.values_in_blocks
            .entry(variable)
            .or_default()
            .insert(block, value);
    }

    pub fn new_value(&mut self, value: Value, ty: Type) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.insert(id, value);
        self.value_types.insert(id, ty);
        id
    }

    pub fn remove_value(&mut self, value: ValueId) {
        self.values.remove(&value);
        self.value_types.remove(&value);
    }

    pub fn read_variable(&mut self, variable: VariableId, block_id: BlockId) -> ValueId {
        if let Some(value) = self
            .values_in_blocks
            .get(&variable)
            .and_then(|map| map.get(&block_id))
        {
            *value
        } else {
            let block = self
                .blocks
                .get(&block_id)
                .unwrap_or_else(|| panic!("Block {:?} not found", block_id));
            let ty = self
                .variable_types
                .get(&variable)
                .cloned()
                .unwrap_or_else(|| panic!("Variable {:?} not found", variable));
            let val = if !block.sealed {
                // Incomplete CFG
                let phi = Phi::new(block_id);
                self.incomplete_phis
                    .entry(block_id)
                    .or_default()
                    .insert(variable, phi.clone());

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

                self.add_phi_operands(variable, val)
            };
            self.write_variable(variable, block_id, val);
            val
        }
    }

    fn add_phi_operands(&mut self, variable: VariableId, phi_id: ValueId) -> ValueId {
        let mut phi = self.get_phi(phi_id);
        let block = self
            .blocks
            .get(&phi.block)
            .unwrap_or_else(|| panic!("Block {:?} not found", phi.block));
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
        let mut block = self
            .blocks
            .get(&block_id)
            .cloned()
            .unwrap_or_else(|| panic!("Block {:?} not found", block_id));
        block.sealed = true;
        self.blocks.insert(block_id, block);
        if let Some(incomplete_phis) = self.incomplete_phis.remove(&block_id) {
            for (variable, phi) in incomplete_phis {
                let ty = self
                    .variable_types
                    .get(&variable)
                    .cloned()
                    .unwrap_or_else(|| panic!("Variable {:?} not found", variable));
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
            let ty = self
                .value_types
                .get(&phi_id)
                .cloned()
                .unwrap_or_else(|| panic!("Value {:?} not found", phi_id));
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasicBlockContinuation {
    Continue,
    ChangeBlock(BlockId),
    Stop,
}

pub struct Ir {
    pub names: BTreeMap<VariableId, String>,
    pub functions: BTreeMap<String, FunctionIr>,
}

impl Default for Ir {
    fn default() -> Self {
        Self::new()
    }
}

impl Ir {
    pub fn new() -> Self {
        Ir {
            names: BTreeMap::new(),
            functions: BTreeMap::new(),
        }
    }

    pub fn translate_function(&mut self, decl: TypedDecl) {
        let mut ir = IrContext::new();
        let ty = decl.ty();
        let function_name = match decl {
            TypedDecl::Var { .. } => panic!("Global variables are not supported yet"),
            TypedDecl::Fn {
                name, body, args, ..
            } => {
                let entry_block = ir.new_block();
                ir.seal_block(entry_block);
                for (arg, i) in args.into_iter().zip(0..) {
                    let argument = ir.new_value(Value::Argument(i), arg.value.ty);
                    ir.write_variable(arg.value.var_id, entry_block, argument);
                    ir.add_instruction(entry_block, argument);
                }

                let continuation = FunctionIr::translate_block(&mut ir, body.value, entry_block);
                if let ChangeBlock(block_id) = continuation {
                    ir.add_terminator(block_id, Terminator::ReturnNoValue);
                } else if let Continue = continuation {
                    ir.add_terminator(entry_block, Terminator::ReturnNoValue);
                }
                name.value.0
            }
        };
        let ir = ir.ready();
        let function_ir = FunctionIr { ir, ty };

        self.functions.insert(function_name, function_ir);
    }

    pub fn dump(&self) -> String {
        let mut result = String::new();
        for (name, function) in &self.functions {
            result.push_str(&format!("Function {}\n", name));
            for (id, block) in &function.ir.blocks {
                result.push_str(&format!("{}:\n", id));
                for instr in &block.instructions {
                    result.push_str(&format!("  {:?}: {:?}\n", instr, function.ir.values.get(instr).unwrap()));
                }
                result.push_str(&format!("  {:?}\n", block.terminator));
            }
        }
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionIr {
    pub ir: ReadyIr,
    pub ty: Type,
}

impl FunctionIr {
    fn dfa_binary(context: &mut IrContext, op: BinaryOpCode, lhs_id: ValueId, rhs_id: ValueId) -> Option<ValueId> {
        let lhs = context.values[&lhs_id].clone();
        let rhs = context.values[&rhs_id].clone();

        let result = match (lhs, rhs, op) {
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Add) => context.new_value(Value::Int(lhs + rhs), Type::Int),
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Sub) => context.new_value(Value::Int(lhs - rhs), Type::Int),
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Mul) => context.new_value(Value::Int(lhs * rhs), Type::Int),
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Div) => context.new_value(Value::Int(lhs / rhs), Type::Int),
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Mod) => context.new_value(Value::Int(lhs % rhs), Type::Int),
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Gt) => context.new_value(Value::Bool(lhs > rhs), Type::Bool),
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Lt) => context.new_value(Value::Bool(lhs < rhs), Type::Bool),
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Gte) => context.new_value(Value::Bool(lhs >= rhs), Type::Bool),
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Lte) => context.new_value(Value::Bool(lhs <= rhs), Type::Bool),
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Eq) => context.new_value(Value::Bool(lhs == rhs), Type::Bool),
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Neq) => context.new_value(Value::Bool(lhs != rhs), Type::Bool),
            (Value::String(lhs), Value::String(rhs), BinaryOpCode::Add) => context.new_value(Value::String(lhs + &rhs), Type::LatteString),
            (Value::String(lhs), Value::String(rhs), BinaryOpCode::Eq) => context.new_value(Value::Bool(lhs == rhs), Type::Bool),
            (Value::String(lhs), Value::String(rhs), BinaryOpCode::Neq) => context.new_value(Value::Bool(lhs != rhs), Type::Bool),
            (Value::Bool(lhs), Value::Bool(rhs), BinaryOpCode::And) => context.new_value(Value::Bool(lhs && rhs), Type::Bool),
            (Value::Bool(lhs), Value::Bool(rhs), BinaryOpCode::Or) => context.new_value(Value::Bool(lhs || rhs), Type::Bool),
            (Value::Bool(lhs), Value::Bool(rhs), BinaryOpCode::Eq) => context.new_value(Value::Bool(lhs == rhs), Type::Bool),
            (Value::Bool(lhs), Value::Bool(rhs), BinaryOpCode::Neq) => context.new_value(Value::Bool(lhs != rhs), Type::Bool),
            _ => return None,
        };
        Some(result)
    }

    fn dfa_unary(context: &mut IrContext, op: UnaryOpCode, expr_id: ValueId) -> Option<ValueId> {
        let expr = context.values[&expr_id].clone();

        let result = match (expr, op) {
            (Value::Int(expr), UnaryOpCode::Neg) => context.new_value(Value::Int(-expr), Type::Int),
            (Value::Bool(expr), UnaryOpCode::Not) => context.new_value(Value::Bool(!expr), Type::Bool),
            _ => return None,
        };
        Some(result)
    }

    fn translate_expr(context: &mut IrContext, expr: TypedExpr, block_id: BlockId) -> ValueId {
        if let TypedExprKind::Variable(_, id) = expr.expr {
            context.read_variable(id, block_id)
        } else {
            let id = match expr.expr {
                TypedExprKind::Variable(_, id) => panic!("Caught above"),
                TypedExprKind::Literal(lit) => match lit {
                    Literal::Int(i) => context.new_value(Value::Int(i), Type::Int),
                    Literal::String(s) => context.new_value(Value::String(s), Type::LatteString),
                    Literal::Bool(b) => context.new_value(Value::Bool(b), Type::Bool),
                },
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
                    if let Some(dfa) = Self::dfa_binary(context, op, lhs, rhs) {
                        // context.remove_value(lhs);
                        // context.remove_value(rhs);
                        // context.pop_instruction(block_id);
                        // context.pop_instruction(block_id);
                        dfa
                    } else {
                        context.new_value(Value::BinaryOp(op, lhs, rhs), expr.ty)
                    }
                }
                TypedExprKind::Unary { op, expr: target } => {
                    let val = Self::translate_expr(context, target.value, block_id);
                    let op = match op.value {
                        ast::UnaryOp::Neg => UnaryOpCode::Neg,
                        ast::UnaryOp::Not => UnaryOpCode::Not,
                    };
                    if let Some(dfa) = Self::dfa_unary(context, op, val) {
                        // context.remove_value(val);
                        // context.pop_instruction(block_id);
                        dfa
                    } else {
                        context.new_value(Value::UnaryOp(op, val), expr.ty)
                    }
                }
                TypedExprKind::Application { target, args } => {
                    let TypedExprKind::Variable(_, id) = target.value.expr else {
                        panic!("This should have been caught by the typechecker")
                    };
                    let args = args
                        .into_iter()
                        .map(|arg| Self::translate_expr(context, arg.value, block_id))
                        .collect();
                    context.new_value(Value::Call(id, args), expr.ty)
                }
            };
            context.add_instruction(block_id, id);
            id
        }
    }

    fn translate_block(
        context: &mut IrContext,
        block: TypedBlock,
        block_id: BlockId,
    ) -> BasicBlockContinuation {
        let mut block_id = block_id;
        let mut final_continuation = Continue;
        for stmt in block.0 {
            let continuation = Self::translate_stmt(context, stmt.value, block_id);
            match continuation {
                Continue => final_continuation = Continue,
                ChangeBlock(new_block_id) => {
                    block_id = new_block_id;
                    final_continuation = ChangeBlock(new_block_id);
                }
                Stop => return Stop,
            }
        }
        final_continuation
    }

    fn default_value(ty: &Type) -> Value {
        match ty {
            Type::Int => Value::Int(0),
            Type::Bool => Value::Bool(false),
            Type::LatteString => Value::String(String::new()),
            Type::Function(_, _) => panic!("Function cannot have a default value"),
            Type::Void => panic!("Void cannot have a default value"),
        }
    }

    fn translate_stmt(
        context: &mut IrContext,
        stmt: TypedStmt,
        block_id: BlockId,
    ) -> BasicBlockContinuation {
        match stmt {
            TypedStmt::Empty => Continue,
            TypedStmt::Block(block) => Self::translate_block(context, block.value, block_id),
            TypedStmt::Decl(decl) => match decl.value {
                TypedDecl::Var { items, .. } => {
                    for item in items {
                        context
                            .variable_names
                            .insert(item.value.var_id, item.value.ident.value.0);
                        if let Some(expr) = item.value.init {
                            let expr = Self::translate_expr(context, expr.value, block_id);
                            context.write_variable(item.value.var_id, block_id, expr);
                        } else {
                            let default = Self::default_value(&item.value.ty);
                            let default = context.new_value(default, item.value.ty);
                            context.add_instruction(block_id, default);
                            context.write_variable(item.value.var_id, block_id, default);
                        }
                    }
                    Continue
                }
                TypedDecl::Fn { .. } => panic!("Nested functions are not supported yet"),
            },
            TypedStmt::Assignment {
                target: _,
                target_id,
                expr,
            } => {
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
            TypedStmt::If {
                cond,
                then,
                otherwise,
            } => {
                let cond = Self::translate_expr(context, cond.value, block_id);
                if let Value::Bool(cond) = context.values[&cond] {
                    return if cond {
                        Self::translate_stmt(context, then.value, block_id)
                    } else if let Some(otherwise) = otherwise {
                        Self::translate_stmt(context, otherwise.value, block_id)
                    } else {
                        Continue
                    }
                }

                let after_block = context.new_block();
                let then_block = context.new_block();
                let then_continuation = Self::translate_stmt(context, then.value, then_block);
                if let Some(otherwise) = otherwise {
                    let else_block = context.new_block();
                    let else_continuation =
                        Self::translate_stmt(context, otherwise.value, else_block);
                    context
                        .add_terminator(block_id, Terminator::Branch(cond, then_block, else_block));
                    context.seal_block(else_block);
                    context.seal_block(then_block);

                    if let (Stop, Stop) = (then_continuation, else_continuation) {
                        return Stop;
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
                    context.add_terminator(
                        block_id,
                        Terminator::Branch(cond, then_block, after_block),
                    );
                    context.seal_block(then_block);

                    if let ChangeBlock(after_then_block) = then_continuation {
                        context.add_terminator(after_then_block, Terminator::Jump(after_block));
                    } else if let Continue = then_continuation {
                        context.add_terminator(then_block, Terminator::Jump(after_block));
                    }

                    context.seal_block(after_block);

                    ChangeBlock(after_block)
                }
            }
            TypedStmt::While { cond, body } => {
                let cond_block = context.new_block();
                let body_block = context.new_block();
                let after_block = context.new_block();

                context.add_terminator(block_id, Terminator::Jump(cond_block));

                let cond = Self::translate_expr(context, cond.value, block_id);
                context.add_terminator(
                    cond_block,
                    Terminator::Branch(cond, body_block, after_block),
                );
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
                let _expr = Self::translate_expr(context, expr.value, block_id);
                Continue
            }
            TypedStmt::Incr(expr) => {
                let expr = Self::translate_expr(context, expr.value, block_id);
                let one = context.new_value(Value::Int(1), Type::Int);
                context.add_instruction(block_id, one);
                let op =
                    context.new_value(Value::BinaryOp(BinaryOpCode::Add, expr, one), Type::Int);
                context.add_instruction(block_id, op);
                Continue
            }
            TypedStmt::Decr(expr) => {
                let expr = Self::translate_expr(context, expr.value, block_id);
                let one = context.new_value(Value::Int(1), Type::Int);
                context.add_instruction(block_id, one);
                let op =
                    context.new_value(Value::BinaryOp(BinaryOpCode::Sub, expr, one), Type::Int);
                context.add_instruction(block_id, op);
                Continue
            }
        }
    }
}
