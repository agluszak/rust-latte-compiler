use crate::ast::Literal;
use crate::ir::BasicBlockContinuation::{ContinueBlock, Stop};
use crate::ir::Value::Undef;
use crate::typechecker::Type;
use crate::typed_ast::{TypedBlock, TypedDecl, TypedExpr, TypedExprKind, TypedStmt, VariableId};
use crate::{ast, DBG};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BlockId(u32);

impl Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
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

impl Value {
    pub fn const_evaluable(&self) -> bool {
        !matches!(self, Value::Call(_, _))
    }
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
    incomplete_phis: BTreeMap<BlockId, BTreeMap<VariableId, ValueId>>,
    defined_in_instructions: BTreeSet<ValueId>,
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

impl Default for IrContext {
    fn default() -> Self {
        Self::new()
    }
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
            defined_in_instructions: BTreeSet::new(),
        }
    }

    pub fn ready(self) -> ReadyIr {
        assert!(self.incomplete_phis.is_empty());
        if DBG.load(std::sync::atomic::Ordering::Relaxed) {
            dbg!((&self.variable_names, &self.values, &self.blocks));
        }
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
        if let Value::Phi(_) = self.values[&value].clone() {
            // TODO: Phi nodes can be added post-sealing, but they should be added to the beginning
        } else {
            assert!(block.terminator.is_none());
        }
        if !self.defined_in_instructions.contains(&value) {
            self.defined_in_instructions.insert(value);
            block.instructions.push(value);
        }
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
                let id = self.new_value(Value::Phi(phi), ty);
                self.incomplete_phis
                    .entry(block_id)
                    .or_default()
                    .insert(variable, id);
                id
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
        let phi_block = self.get_phi(phi_id).unwrap().block;
        let block = self
            .blocks
            .get(&phi_block)
            .unwrap_or_else(|| panic!("Block {:?} not found", phi_block));
        for pred in block.preds.clone() {
            let pred_val = self.read_variable(variable, pred);
            if let Some(pred_phi) = self.get_phi(pred_val) {
                pred_phi.users.insert(phi_id);
            }
            self.get_phi(phi_id)
                .unwrap()
                .incoming
                .insert(pred, pred_val);
        }
        self.try_remove_trivial_phi(phi_id)
    }

    fn get_phi(&mut self, phi_id: ValueId) -> Option<&mut Phi> {
        if let Some(Value::Rerouted(id)) = self.values.get(&phi_id) {
            self.get_phi(*id)
        } else if let Some(Value::Phi(phi)) = self.values.get_mut(&phi_id) {
            Some(phi)
        } else {
            None
        }
    }

    pub fn seal_block(&mut self, block_id: BlockId) {
        let mut block = self
            .blocks
            .get(&block_id)
            .cloned()
            .unwrap_or_else(|| panic!("Block {:?} not found", block_id));
        assert!(!block.sealed);
        block.sealed = true;
        self.blocks.insert(block_id, block);
        if let Some(incomplete_phis) = self.incomplete_phis.remove(&block_id) {
            for (variable, phi_id) in incomplete_phis {
                self.add_phi_operands(variable, phi_id);
            }
        }
    }

    fn try_remove_trivial_phi(&mut self, phi_id: ValueId) -> ValueId {
        let mut phi = self.get_phi(phi_id).cloned().unwrap();
        let mut same = None;
        for &op in phi.incoming.values() {
            if let Some(same) = same {
                if op == same || op == phi_id {
                    // Unique value or self−reference
                    continue;
                } else {
                    // This phi merges at least two different values, so it's not trivial
                    return phi_id;
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
            if let Some(_phi) = self.get_phi(user) {
                let target = self.try_remove_trivial_phi(user);
                // TODO: is this necessary?
                self.values.insert(user, Value::Rerouted(target));
            }
        }
        same.unwrap()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BasicBlockContinuation {
    ContinueBlock(BlockId),
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
                if let ContinueBlock(block_id) = continuation {
                    ir.add_terminator(block_id, Terminator::ReturnNoValue);
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
                    result.push_str(&format!(
                        "  {:?}: {:?}\n",
                        instr,
                        function.ir.values.get(instr).unwrap()
                    ));
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
    fn dfa_binary(
        context: &mut IrContext,
        op: BinaryOpCode,
        lhs_id: ValueId,
        rhs_id: ValueId,
    ) -> Option<ValueId> {
        let lhs = context.values[&lhs_id].clone();
        let rhs = context.values[&rhs_id].clone();

        let result = match (lhs, rhs, op) {
            // Constant folding
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Add) => {
                context.new_value(Value::Int(lhs + rhs), Type::Int)
            }

            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Sub) => {
                context.new_value(Value::Int(lhs - rhs), Type::Int)
            }

            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Mul) => {
                context.new_value(Value::Int(lhs * rhs), Type::Int)
            }

            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Div) => {
                context.new_value(Value::Int(lhs / rhs), Type::Int)
            }
            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Mod) => {
                context.new_value(Value::Int(lhs % rhs), Type::Int)
            }

            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Gt) => {
                context.new_value(Value::Bool(lhs > rhs), Type::Bool)
            }

            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Lt) => {
                context.new_value(Value::Bool(lhs < rhs), Type::Bool)
            }

            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Gte) => {
                context.new_value(Value::Bool(lhs >= rhs), Type::Bool)
            }

            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Lte) => {
                context.new_value(Value::Bool(lhs <= rhs), Type::Bool)
            }

            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Eq) => {
                context.new_value(Value::Bool(lhs == rhs), Type::Bool)
            }

            (Value::Int(lhs), Value::Int(rhs), BinaryOpCode::Neq) => {
                context.new_value(Value::Bool(lhs != rhs), Type::Bool)
            }

            (Value::String(lhs), Value::String(rhs), BinaryOpCode::Add) => {
                context.new_value(Value::String(lhs + &rhs), Type::LatteString)
            }

            (Value::String(lhs), Value::String(rhs), BinaryOpCode::Eq) => {
                context.new_value(Value::Bool(lhs == rhs), Type::Bool)
            }

            (Value::String(lhs), Value::String(rhs), BinaryOpCode::Neq) => {
                context.new_value(Value::Bool(lhs != rhs), Type::Bool)
            }

            (Value::Bool(lhs), Value::Bool(rhs), BinaryOpCode::And) => {
                context.new_value(Value::Bool(lhs && rhs), Type::Bool)
            }

            (Value::Bool(lhs), Value::Bool(rhs), BinaryOpCode::Or) => {
                context.new_value(Value::Bool(lhs || rhs), Type::Bool)
            }

            (Value::Bool(lhs), Value::Bool(rhs), BinaryOpCode::Eq) => {
                context.new_value(Value::Bool(lhs == rhs), Type::Bool)
            }

            (Value::Bool(lhs), Value::Bool(rhs), BinaryOpCode::Neq) => {
                context.new_value(Value::Bool(lhs != rhs), Type::Bool)
            }

            // Expression simplification
            (lhs, rhs, BinaryOpCode::Eq)
                if lhs == rhs && lhs.const_evaluable() && rhs.const_evaluable() =>
            {
                context.new_value(Value::Bool(true), Type::Bool)
            }
            (lhs, rhs, BinaryOpCode::Neq)
                if lhs == rhs && lhs.const_evaluable() && rhs.const_evaluable() =>
            {
                context.new_value(Value::Bool(false), Type::Bool)
            }
            (Value::Int(0), _rhs, BinaryOpCode::Add) => rhs_id,
            (Value::Int(0), _rhs, BinaryOpCode::Sub) => {
                context.new_value(Value::UnaryOp(UnaryOpCode::Neg, rhs_id), Type::Int)
            }
            (Value::Int(1), _rhs, BinaryOpCode::Mul) => rhs_id,
            (Value::Int(0), rhs, BinaryOpCode::Mul) if rhs.const_evaluable() => {
                context.new_value(Value::Int(0), Type::Int)
            }
            (_lhs, Value::Int(0), BinaryOpCode::Add) => lhs_id,
            (_lhs, Value::Int(0), BinaryOpCode::Sub) => lhs_id,
            (_lhs, Value::Int(1), BinaryOpCode::Mul) => lhs_id,
            (lhs, Value::Int(0), BinaryOpCode::Mul) if lhs.const_evaluable() => {
                context.new_value(Value::Int(0), Type::Int)
            }
            (_lhs, Value::Int(1), BinaryOpCode::Div) => lhs_id,
            // Short-circuiting SHOULD NOT overoptimize
            (_lhs, Value::Bool(true), BinaryOpCode::And) => lhs_id,
            (lhs, Value::Bool(false), BinaryOpCode::And) if lhs.const_evaluable() => {
                context.new_value(Value::Bool(false), Type::Bool)
            }
            (lhs, Value::Bool(true), BinaryOpCode::Or) if lhs.const_evaluable() => {
                context.new_value(Value::Bool(true), Type::Bool)
            }
            (_lhs, Value::Bool(false), BinaryOpCode::Or) => lhs_id,
            (Value::Bool(true), _rhs, BinaryOpCode::And) => rhs_id,
            (Value::Bool(false), _rhs, BinaryOpCode::Or) => rhs_id,
            // Short circuiting SHOULD optimize
            (Value::Bool(false), _rhs, BinaryOpCode::And) => {
                context.new_value(Value::Bool(false), Type::Bool)
            }
            (Value::Bool(true), _rhs, BinaryOpCode::Or) => {
                context.new_value(Value::Bool(true), Type::Bool)
            }
            _ => return None,
        };
        Some(result)
    }

    fn dfa_unary(context: &mut IrContext, op: UnaryOpCode, expr_id: ValueId) -> Option<ValueId> {
        let expr = context.values[&expr_id].clone();

        let result = match (expr, op) {
            (Value::Int(expr), UnaryOpCode::Neg) => context.new_value(Value::Int(-expr), Type::Int),
            (Value::Bool(expr), UnaryOpCode::Not) => {
                context.new_value(Value::Bool(!expr), Type::Bool)
            }
            _ => return None,
        };
        Some(result)
    }

    fn join_blocks(context: &mut IrContext, block_ids: &[BlockId]) -> BlockId {
        let mut block_ids = block_ids.to_vec();
        block_ids.sort();
        block_ids.dedup();
        if block_ids.len() == 1 {
            return block_ids[0];
        }

        let join_block = context.new_block();
        for block_id in block_ids {
            context.add_terminator(block_id, Terminator::Jump(join_block));
        }
        context.seal_block(join_block);
        join_block
    }

    fn translate_expr(
        context: &mut IrContext,
        expr: TypedExpr,
        block_id: BlockId,
    ) -> (ValueId, BlockId) {
        let (value, block_id) = match expr.expr {
            TypedExprKind::Variable(_, id) => (context.read_variable(id, block_id), block_id),
            TypedExprKind::Literal(lit) => {
                let val = match lit {
                    Literal::Int(i) => context.new_value(Value::Int(i), Type::Int),
                    Literal::String(s) => context.new_value(
                        Value::String(
                            s.strip_prefix('"')
                                .unwrap()
                                .strip_suffix('"')
                                .unwrap()
                                .to_string(),
                        ),
                        Type::LatteString,
                    ),
                    Literal::Bool(b) => context.new_value(Value::Bool(b), Type::Bool),
                };
                (val, block_id)
            }
            TypedExprKind::Binary { lhs, op, rhs } => {
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

                // Handle short-circuiting
                // TODO: const eval
                if let BinaryOpCode::And = op {
                    let (lhs, lhs_block) = Self::translate_expr(context, lhs.value, block_id);

                    let true_block = context.new_block();
                    let rhs_block = context.new_block();
                    let false_block = context.new_block();
                    let join_block = context.new_block();

                    context
                        .add_terminator(lhs_block, Terminator::Branch(lhs, rhs_block, false_block));
                    context.seal_block(rhs_block);
                    let (rhs, rhs_block) = Self::translate_expr(context, rhs.value, rhs_block);
                    context.add_terminator(
                        rhs_block,
                        Terminator::Branch(rhs, true_block, false_block),
                    );
                    context.seal_block(true_block);
                    context.seal_block(false_block);
                    context.add_terminator(true_block, Terminator::Jump(join_block));
                    context.add_terminator(false_block, Terminator::Jump(join_block));
                    context.seal_block(join_block);
                    let true_ = context.new_value(Value::Bool(true), Type::Bool);
                    let false_ = context.new_value(Value::Bool(false), Type::Bool);
                    context.add_instruction(join_block, true_);
                    context.add_instruction(join_block, false_);
                    let mut phi = Phi::new(join_block);
                    phi.incoming.insert(true_block, true_);
                    phi.incoming.insert(false_block, false_);
                    let phi = context.new_value(Value::Phi(phi), Type::Bool);
                    context.add_instruction(join_block, phi);
                    return (phi, join_block);
                } else if let BinaryOpCode::Or = op {
                    let (lhs, lhs_block) = Self::translate_expr(context, lhs.value, block_id);

                    let true_block = context.new_block();
                    let rhs_block = context.new_block();
                    let false_block = context.new_block();
                    let join_block = context.new_block();

                    context
                        .add_terminator(lhs_block, Terminator::Branch(lhs, true_block, rhs_block));
                    context.seal_block(rhs_block);
                    let (rhs, rhs_block) = Self::translate_expr(context, rhs.value, rhs_block);
                    context.add_terminator(
                        rhs_block,
                        Terminator::Branch(rhs, true_block, false_block),
                    );
                    context.seal_block(true_block);
                    context.seal_block(false_block);
                    context.add_terminator(true_block, Terminator::Jump(join_block));
                    context.add_terminator(false_block, Terminator::Jump(join_block));
                    context.seal_block(join_block);
                    let true_ = context.new_value(Value::Bool(true), Type::Bool);
                    let false_ = context.new_value(Value::Bool(false), Type::Bool);
                    context.add_instruction(join_block, true_);
                    context.add_instruction(join_block, false_);
                    let mut phi = Phi::new(join_block);
                    phi.incoming.insert(true_block, true_);
                    phi.incoming.insert(false_block, false_);
                    let phi = context.new_value(Value::Phi(phi), Type::Bool);
                    context.add_instruction(join_block, phi);
                    return (phi, join_block);
                }

                let (lhs, lhs_block) = Self::translate_expr(context, lhs.value, block_id);
                let (rhs, rhs_block) = Self::translate_expr(context, rhs.value, block_id);
                let block_id = Self::join_blocks(context, &[lhs_block, rhs_block]);

                if let Some(dfa) = Self::dfa_binary(context, op, lhs, rhs) {
                    (dfa, block_id)
                } else {
                    let val = context.new_value(Value::BinaryOp(op, lhs, rhs), expr.ty);
                    (val, block_id)
                }
            }
            TypedExprKind::Unary { op, expr: target } => {
                let (val, block_id) = Self::translate_expr(context, target.value, block_id);
                let op = match op.value {
                    ast::UnaryOp::Neg => UnaryOpCode::Neg,
                    ast::UnaryOp::Not => UnaryOpCode::Not,
                };
                if let Some(dfa) = Self::dfa_unary(context, op, val) {
                    (dfa, block_id)
                } else {
                    let val = context.new_value(Value::UnaryOp(op, val), expr.ty);
                    (val, block_id)
                }
            }
            TypedExprKind::Application { target, args } => {
                let TypedExprKind::Variable(_, id) = target.value.expr else {
                    panic!("This should have been caught by the typechecker")
                };

                let mut arg_values = Vec::new();
                let mut current_block_id = block_id;
                for arg in args {
                    let (arg, block_id) =
                        Self::translate_expr(context, arg.value, current_block_id);
                    current_block_id = block_id;
                    arg_values.push(arg);
                }

                let val = context.new_value(Value::Call(id, arg_values), expr.ty);
                (val, current_block_id)
            }
        };
        if let Value::Phi(phi) = &context.values[&value] {
            context.add_instruction(phi.block, value);
        } else {
            context.add_instruction(block_id, value);
        }
        (value, block_id)
    }

    fn translate_block(
        context: &mut IrContext,
        block: TypedBlock,
        block_id: BlockId,
    ) -> BasicBlockContinuation {
        let mut block_id = block_id;
        let mut final_continuation = ContinueBlock(block_id);
        for stmt in block.0 {
            let continuation = Self::translate_stmt(context, stmt.value, block_id);
            match continuation {
                ContinueBlock(new_block_id) => {
                    block_id = new_block_id;
                    final_continuation = ContinueBlock(new_block_id);
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
            TypedStmt::Empty => ContinueBlock(block_id),
            TypedStmt::Block(block) => Self::translate_block(context, block.value, block_id),
            TypedStmt::Decl(decl) => match decl.value {
                TypedDecl::Var { items, .. } => {
                    let mut expr_blocks = Vec::new();
                    for item in items {
                        context
                            .variable_names
                            .insert(item.value.var_id, item.value.ident.value.0);
                        if let Some(expr) = item.value.init {
                            let (expr, block_id) =
                                Self::translate_expr(context, expr.value, block_id);
                            context.write_variable(item.value.var_id, block_id, expr);
                            expr_blocks.push(block_id);
                        } else {
                            let default = Self::default_value(&item.value.ty);
                            let default = context.new_value(default, item.value.ty);
                            context.add_instruction(block_id, default);
                            context.write_variable(item.value.var_id, block_id, default);
                            expr_blocks.push(block_id);
                        }
                    }
                    let block_id = Self::join_blocks(context, &expr_blocks);
                    ContinueBlock(block_id)
                }
                TypedDecl::Fn { .. } => panic!("Nested functions are not supported yet"),
            },
            TypedStmt::Assignment {
                target: _,
                target_id,
                expr,
            } => {
                let (expr, block_id) = Self::translate_expr(context, expr.value, block_id);
                context.write_variable(target_id, block_id, expr);
                ContinueBlock(block_id)
            }
            TypedStmt::Return(expr) => {
                let expr = expr.map(|expr| Self::translate_expr(context, expr.value, block_id));
                if let Some((expr, block_id)) = expr {
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
                let (cond, block_id) = Self::translate_expr(context, cond.value, block_id);
                if let Value::Bool(cond) = context.values[&cond] {
                    return if cond {
                        Self::translate_stmt(context, then.value, block_id)
                    } else if let Some(otherwise) = otherwise {
                        Self::translate_stmt(context, otherwise.value, block_id)
                    } else {
                        ContinueBlock(block_id)
                    };
                }

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

                    let after_block = context.new_block();

                    if let ContinueBlock(after_then_block) = then_continuation {
                        context.add_terminator(after_then_block, Terminator::Jump(after_block));
                    }

                    if let ContinueBlock(after_else_block) = else_continuation {
                        context.add_terminator(after_else_block, Terminator::Jump(after_block));
                    }
                    context.seal_block(after_block);

                    ContinueBlock(after_block)
                } else {
                    let after_block = context.new_block();

                    context.add_terminator(
                        block_id,
                        Terminator::Branch(cond, then_block, after_block),
                    );
                    context.seal_block(then_block);

                    if let ContinueBlock(after_then_block) = then_continuation {
                        context.add_terminator(after_then_block, Terminator::Jump(after_block));
                    }

                    context.seal_block(after_block);

                    ContinueBlock(after_block)
                }
            }
            TypedStmt::While { cond, body } => {
                let cond_block = context.new_block();
                context.add_terminator(block_id, Terminator::Jump(cond_block));

                let (cond, cond_block) = Self::translate_expr(context, cond.value, cond_block);

                if let Value::Bool(cond) = context.values[&cond] {
                    return if !cond {
                        let after_block = context.new_block();
                        context.add_terminator(cond_block, Terminator::Jump(after_block));
                        context.seal_block(cond_block);
                        context.seal_block(after_block);
                        ContinueBlock(after_block)
                    } else {
                        let body_block = context.new_block();
                        context.add_terminator(cond_block, Terminator::Jump(body_block));
                        context.seal_block(body_block);
                        let body_continuation =
                            Self::translate_stmt(context, body.value, body_block);
                        if let ContinueBlock(after_body_block) = body_continuation {
                            context.add_terminator(after_body_block, Terminator::Jump(cond_block));
                        }
                        context.seal_block(cond_block);
                        Stop
                    };
                }

                let after_block = context.new_block();
                let body_block = context.new_block();

                context.add_terminator(
                    cond_block,
                    Terminator::Branch(cond, body_block, after_block),
                );
                context.seal_block(body_block);
                context.seal_block(after_block);

                let body_continuation = Self::translate_stmt(context, body.value, body_block);
                if let ContinueBlock(after_body_block) = body_continuation {
                    context.add_terminator(after_body_block, Terminator::Jump(cond_block));
                }

                context.seal_block(cond_block);
                ContinueBlock(after_block)
            }
            TypedStmt::Expr(expr) => {
                let (_, block_id) = Self::translate_expr(context, expr.value, block_id);
                ContinueBlock(block_id)
            }
            TypedStmt::Incr(expr) => {
                let TypedExprKind::Variable(_, var_id) = expr.value.expr else {
                    panic!("This should have been caught by the typechecker")
                };
                let (expr, block_id) = Self::translate_expr(context, expr.value, block_id);
                let one = context.new_value(Value::Int(1), Type::Int);
                context.add_instruction(block_id, one);
                let op =
                    context.new_value(Value::BinaryOp(BinaryOpCode::Add, expr, one), Type::Int);
                context.add_instruction(block_id, op);
                context.write_variable(var_id, block_id, op);
                ContinueBlock(block_id)
            }
            TypedStmt::Decr(expr) => {
                let TypedExprKind::Variable(_, var_id) = expr.value.expr else {
                    panic!("This should have been caught by the typechecker")
                };
                let (expr, block_id) = Self::translate_expr(context, expr.value, block_id);
                let one = context.new_value(Value::Int(1), Type::Int);
                context.add_instruction(block_id, one);
                let op =
                    context.new_value(Value::BinaryOp(BinaryOpCode::Sub, expr, one), Type::Int);
                context.add_instruction(block_id, op);
                context.write_variable(var_id, block_id, op);
                ContinueBlock(block_id)
            }
        }
    }
}
