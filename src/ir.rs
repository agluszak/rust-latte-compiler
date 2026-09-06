use crate::types::Type;
use crate::typed_ast::VariableId;
use std::collections::BTreeMap;
use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct BlockId(pub(crate) u32);

impl Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ValueId(pub(crate) u32);

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UnaryOpCode {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi {
    pub incoming: Vec<(BlockId, ValueId)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Int(i32),
    String(String),
    Bool(bool),
    Call(VariableId, Vec<ValueId>),
    Argument(u32),
    BinaryOp(BinaryOpCode, ValueId, ValueId),
    UnaryOp(UnaryOpCode, ValueId),
    Phi(Phi),
    Undef,
}

impl Value {
    pub(crate) fn rewrite_operands(&mut self, mut f: impl FnMut(ValueId) -> ValueId) {
        match self {
            Value::Call(_, args) => {
                for arg in args {
                    *arg = f(*arg);
                }
            }
            Value::BinaryOp(_, lhs, rhs) => {
                *lhs = f(*lhs);
                *rhs = f(*rhs);
            }
            Value::UnaryOp(_, operand) => *operand = f(*operand),
            Value::Phi(phi) => {
                for (_, value) in &mut phi.incoming {
                    *value = f(*value);
                }
            }
            Value::Int(_)
            | Value::String(_)
            | Value::Bool(_)
            | Value::Argument(_)
            | Value::Undef => {}
        }
    }

    pub(crate) fn operands(&self) -> impl Iterator<Item = ValueId> + '_ {
        match self {
            Value::Call(_, args) => args.iter().copied().collect::<Vec<_>>().into_iter(),
            Value::BinaryOp(_, lhs, rhs) => vec![*lhs, *rhs].into_iter(),
            Value::UnaryOp(_, operand) => vec![*operand].into_iter(),
            Value::Phi(phi) => phi.incoming.iter().map(|(_, v)| *v).collect::<Vec<_>>().into_iter(),
            Value::Int(_)
            | Value::String(_)
            | Value::Bool(_)
            | Value::Argument(_)
            | Value::Undef => Vec::new().into_iter(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueData {
    pub ty: Type,
    pub kind: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    Return(ValueId),
    ReturnNoValue,
    Branch(ValueId, BlockId, BlockId),
    Jump(BlockId),
}

impl Terminator {
    pub(crate) fn successors(&self) -> impl Iterator<Item = BlockId> {
        let successors = match *self {
            Terminator::Return(_) | Terminator::ReturnNoValue => [None, None],
            Terminator::Branch(_, then_block, else_block) => [Some(then_block), Some(else_block)],
            Terminator::Jump(target) => [Some(target), None],
        };
        successors.into_iter().flatten()
    }

    pub(crate) fn rewrite_operands(&mut self, mut f: impl FnMut(ValueId) -> ValueId) {
        match self {
            Terminator::Return(value) => *value = f(*value),
            Terminator::Branch(condition, _, _) => *condition = f(*condition),
            Terminator::ReturnNoValue | Terminator::Jump(_) => {}
        }
    }

    pub(crate) fn operands(&self) -> impl Iterator<Item = ValueId> + '_ {
        match *self {
            Terminator::Return(value) | Terminator::Branch(value, _, _) => {
                vec![value].into_iter()
            }
            Terminator::ReturnNoValue | Terminator::Jump(_) => Vec::new().into_iter(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicBlock {
    pub phis: Vec<ValueId>,
    pub instructions: Vec<ValueId>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionIr {
    pub ty: Type,
    pub entry: BlockId,
    pub values: BTreeMap<ValueId, ValueData>,
    pub blocks: BTreeMap<BlockId, BasicBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ir {
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
            functions: BTreeMap::new(),
        }
    }

    pub fn dump(&self) -> String {
        let mut result = String::new();
        for (name, function) in &self.functions {
            result.push_str(&format!("Function {}\n", name));
            for (id, block) in &function.blocks {
                result.push_str(&format!("{}:\n", id));
                for phi in &block.phis {
                    result.push_str(&format!("  {:?}: {:?}\n", phi, function.values[phi]));
                }
                for instr in &block.instructions {
                    result.push_str(&format!("  {:?}: {:?}\n", instr, function.values[instr]));
                }
                result.push_str(&format!("  {:?}\n", block.terminator));
            }
        }
        result
    }
}
