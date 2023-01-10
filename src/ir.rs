use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::mem;

#[derive(Debug, Default)]
pub struct Context {
    pub functions: HashMap<FunctionId, IrFunctionType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Instruction {
    Define(SsaName, ValueId),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Terminator {
    UnconditionalJump(BlockId),
    ConditionalJump {
        condition: ValueId,
        then_block: BlockId,
        else_block: BlockId,
    },
    Return(ValueId),
}

impl Terminator {
    pub fn conditional_jump(condition: ValueId, then_block: BlockId, else_block: BlockId) -> Self {
        Terminator::ConditionalJump {
            condition,
            then_block,
            else_block,
        }
    }

    pub fn jump(block: BlockId) -> Self {
        Terminator::UnconditionalJump(block)
    }

    pub fn return_(value: ValueId) -> Self {
        Terminator::Return(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SsaName(u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub preds: BTreeSet<BlockId>,
    pub instructions: Vec<Instruction>,
    pub sealed: bool,
    incomplete_phis: HashMap<VariableId, ValueId>,
}

impl Block {
    pub fn new() -> Self {
        Self {
            preds: BTreeSet::new(),
            instructions: Vec::new(),
            sealed: false,
            incomplete_phis: HashMap::new(),
        }
    }

    fn add_incopmlete_phi(&mut self, var: VariableId, value: ValueId) {
        self.incomplete_phis.insert(var, value);
    }

    fn take_incomplete_phis(&mut self) -> HashMap<VariableId, ValueId> {
        mem::take(&mut self.incomplete_phis)
    }

    fn seal(&mut self) {
        self.sealed = true;
    }

    fn declare_pred(&mut self, pred: BlockId) {
        if self.sealed {
            panic!("Cannot add a predecessor to a sealed block");
        }
        self.preds.insert(pred);
    }

    fn add_instruction(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct BlockId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct VariableId(u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum IrType {
    Bool,
    Int,
    String,
    Void,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct IrFunctionType {
    pub args: Vec<IrType>,
    pub return_ty: IrType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValueContainer {
    value: Value,
    ty: IrType,
    users: HashSet<ValueId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VariableContainer {
    ty: IrType,
    block_values: HashMap<BlockId, ValueId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BlockContainer {
    block: Block,
}

#[derive(Debug, Default)]
struct Blocks(HashMap<BlockId, BlockContainer>);

impl Blocks {
    fn all(&self) -> impl Iterator<Item = (BlockId, &Block)> {
        self.0.iter().map(|(id, container)| (*id, &container.block))
    }

    fn get(&self, id: BlockId) -> &Block {
        &self.0[&id].block
    }

    fn get_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.0.get_mut(&id).unwrap().block
    }

    fn new(&mut self) -> BlockId {
        let id = BlockId(self.0.len() as u32);
        self.0.insert(
            id,
            BlockContainer {
                block: Block::new(),
            },
        );
        id
    }

    fn declare_predecessor(&mut self, block: BlockId, pred: BlockId) {
        self.get_mut(block).declare_pred(pred);
    }

    fn predecessors(&self, block: BlockId) -> BTreeSet<BlockId> {
        self.get(block).preds.clone()
    }

    fn single_predecessor(&self, block: BlockId) -> Option<BlockId> {
        let preds = self.predecessors(block);
        if preds.len() == 1 {
            Some(preds.into_iter().next().unwrap())
        } else {
            None
        }
    }
}

#[derive(Debug, Default)]
struct Variables(HashMap<VariableId, VariableContainer>);

#[derive(Debug, Default)]
struct Values {
    map: HashMap<ValueId, ValueContainer>,
    cache: HashMap<(IrType, Value), ValueId>,
}

impl Values {
    fn new(&mut self, value: Value, ty: IrType) -> ValueId {
        let value = self.fold_constant(value);
        let pair = (ty, value);
        if let Some(id) = self.cache.get(&pair) {
            return *id;
        }
        let (ty, value) = pair;

        let id = ValueId(self.map.len() as u32);
        self.cache.insert((ty.clone(), value.clone()), id);
        self.map.insert(
            id,
            ValueContainer {
                value,
                ty,
                users: HashSet::new(),
            },
        );

        id
    }

    fn fold_constant(&mut self, value: Value) -> Value {
        match value {
            Value::BinaryOperation(operation) => match operation {
                BinaryOperation::Add(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Int(lhs + rhs)),
                        _ => ValueBuilder::add(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Sub(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Int(lhs - rhs)),
                        _ => ValueBuilder::sub(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Mul(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Int(lhs * rhs)),
                        _ => ValueBuilder::mul(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Div(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Int(lhs / rhs)),
                        _ => ValueBuilder::div(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Mod(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Int(lhs % rhs)),
                        _ => ValueBuilder::mod_(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::And(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Bool(lhs)),
                            Value::Constant(Constant::Bool(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs && rhs)),
                        _ => ValueBuilder::and(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Or(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Bool(lhs)),
                            Value::Constant(Constant::Bool(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs || rhs)),
                        _ => ValueBuilder::or(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Eq(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs == rhs)),
                        (
                            Value::Constant(Constant::Bool(lhs)),
                            Value::Constant(Constant::Bool(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs == rhs)),
                        (
                            Value::Constant(Constant::String(lhs)),
                            Value::Constant(Constant::String(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs == rhs)),
                        _ => ValueBuilder::eq(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Neq(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs != rhs)),
                        (
                            Value::Constant(Constant::Bool(lhs)),
                            Value::Constant(Constant::Bool(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs != rhs)),
                        (
                            Value::Constant(Constant::String(lhs)),
                            Value::Constant(Constant::String(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs != rhs)),
                        _ => ValueBuilder::neq(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Lt(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs < rhs)),
                        _ => ValueBuilder::lt(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Lte(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs <= rhs)),
                        _ => ValueBuilder::lte(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Gt(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs > rhs)),
                        _ => ValueBuilder::gt(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Gte(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::Int(lhs)),
                            Value::Constant(Constant::Int(rhs)),
                        ) => Value::Constant(Constant::Bool(lhs >= rhs)),
                        _ => ValueBuilder::gte(operands.lhs, operands.rhs),
                    }
                }
                BinaryOperation::Concat(operands) => {
                    let lhs = self.get(operands.lhs).clone();
                    let lhs = self.fold_constant(lhs);
                    let rhs = self.get(operands.rhs).clone();
                    let rhs = self.fold_constant(rhs);
                    match (lhs, rhs) {
                        (
                            Value::Constant(Constant::String(lhs)),
                            Value::Constant(Constant::String(rhs)),
                        ) => Value::Constant(Constant::String(lhs + &rhs)),
                        _ => ValueBuilder::concat(operands.lhs, operands.rhs),
                    }
                }
            },
            Value::UnaryOperation(operation) => match operation {
                UnaryOperation::Not(operand_id) => {
                    let operand = self.get(operand_id).clone();
                    let operand = self.fold_constant(operand);
                    match operand {
                        Value::Constant(Constant::Bool(operand)) => {
                            Value::Constant(Constant::Bool(!operand))
                        }
                        _ => ValueBuilder::not(operand_id),
                    }
                }
                UnaryOperation::Neg(operand_id) => {
                    let operand = self.get(operand_id).clone();
                    let operand = self.fold_constant(operand);
                    match operand {
                        Value::Constant(Constant::Int(operand)) => {
                            Value::Constant(Constant::Int(-operand))
                        }
                        _ => ValueBuilder::neg(operand_id),
                    }
                }
            },
            other => other,
        }
    }

    fn get(&self, id: ValueId) -> &Value {
        &self.map[&id].value
    }

    fn get_mut(&mut self, id: ValueId) -> &mut Value {
        &mut self.map.get_mut(&id).unwrap().value
    }

    fn ty(&self, id: ValueId) -> IrType {
        self.map[&id].ty.clone()
    }

    fn add_user(&mut self, id: ValueId, user: ValueId) {
        self.map.get_mut(&id).unwrap().users.insert(user);
    }

    fn users(&self, id: ValueId) -> HashSet<ValueId> {
        self.map[&id].users.clone()
    }
}

impl Variables {
    fn new(&mut self, ty: IrType) -> VariableId {
        let id = VariableId(self.0.len() as u32);
        self.0.insert(
            id,
            VariableContainer {
                ty,
                block_values: HashMap::new(),
            },
        );
        id
    }

    fn ty(&self, id: VariableId) -> IrType {
        self.0[&id].ty.clone()
    }

    fn get_in_block(&self, id: VariableId, block: BlockId) -> Option<ValueId> {
        self.0[&id].block_values.get(&block).cloned()
    }

    fn set_in_block(&mut self, id: VariableId, block: BlockId, value: ValueId) {
        self.0
            .get_mut(&id)
            .unwrap()
            .block_values
            .insert(block, value);
    }
}

#[derive(Debug)]
pub struct Function<'a> {
    // pub name: String, // TODO: interning
    blocks: Blocks,
    variables: Variables,
    values: Values,
    call_counter: u32,
    ssa_counter: u32,
    context: &'a Context,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ValueId(u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Phi {
    pub block: BlockId,
    pub operands: BTreeMap<BlockId, ValueId>,
}

impl Phi {
    pub fn new(block: BlockId) -> Self {
        Self {
            block,
            operands: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Commutative {
    lhs: ValueId,
    rhs: ValueId,
}

impl Commutative {
    fn new(lhs: ValueId, rhs: ValueId) -> Self {
        if lhs < rhs {
            Self { lhs, rhs }
        } else {
            Self { lhs: rhs, rhs: lhs }
        }
    }

    fn replace_use(&mut self, old: ValueId, new: ValueId) {
        if self.lhs == old {
            self.lhs = new;
        }
        if self.rhs == old {
            self.rhs = new;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct NonCommutative {
    lhs: ValueId,
    rhs: ValueId,
}

impl NonCommutative {
    fn new(lhs: ValueId, rhs: ValueId) -> Self {
        Self { lhs, rhs }
    }

    fn replace_use(&mut self, old: ValueId, new: ValueId) {
        if self.lhs == old {
            self.lhs = new;
        }
        if self.rhs == old {
            self.rhs = new;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum BinaryOperation {
    Add(Commutative),
    Sub(NonCommutative),
    Mul(Commutative),
    Div(NonCommutative),
    Mod(NonCommutative),
    Eq(Commutative),
    Neq(Commutative),
    Lt(NonCommutative),
    Gt(NonCommutative),
    Lte(NonCommutative),
    Gte(NonCommutative),
    Or(Commutative),
    And(Commutative),
    Concat(NonCommutative),
}

impl BinaryOperation {
    fn replace_use(&mut self, old: ValueId, new: ValueId) {
        match self {
            Self::Add(operands)
            | Self::Mul(operands)
            | Self::Eq(operands)
            | Self::Neq(operands)
            | Self::Or(operands)
            | Self::And(operands) => operands.replace_use(old, new),
            Self::Sub(operands)
            | Self::Div(operands)
            | Self::Mod(operands)
            | Self::Lt(operands)
            | Self::Gt(operands)
            | Self::Lte(operands)
            | Self::Gte(operands)
            | Self::Concat(operands) => operands.replace_use(old, new),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum UnaryOperation {
    Not(ValueId),
    Neg(ValueId),
}

impl UnaryOperation {
    fn replace_use(&mut self, old: ValueId, new: ValueId) {
        match self {
            Self::Not(value) | Self::Neg(value) => {
                if *value == old {
                    *value = new;
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum Constant {
    Int(i32),
    Bool(bool),
    String(String), // TODO: interning
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash, Copy)]
pub struct CallId(u32);

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct Call {
    id: CallId,
    function: FunctionId,
    arguments: Vec<ValueId>,
}

impl Call {
    fn replace_use(&mut self, old: ValueId, new: ValueId) {
        for argument in &mut self.arguments {
            if *argument == old {
                *argument = new;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct FunctionId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ArgumentId(u32);

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum Value {
    Phi(Phi),
    Constant(Constant),
    Argument(ArgumentId),
    BinaryOperation(BinaryOperation),
    UnaryOperation(UnaryOperation),
    Call(Call),
    Undefined,
}

impl Value {
    pub fn as_phi(&self) -> &Phi {
        match self {
            Value::Phi(phi) => phi,
            _ => panic!("not a phi"),
        }
    }

    pub fn as_phi_mut(&mut self) -> &mut Phi {
        match self {
            Value::Phi(phi) => phi,
            _ => panic!("not a phi"),
        }
    }

    pub fn replace_use(&mut self, old: ValueId, new: ValueId) {
        match self {
            Value::Phi(phi) => {
                for (_, operand) in phi.operands.iter_mut() {
                    if *operand == old {
                        *operand = new;
                    }
                }
            }
            Value::BinaryOperation(operation) => operation.replace_use(old, new),
            Value::UnaryOperation(operation) => operation.replace_use(old, new),
            Value::Call(call) => call.replace_use(old, new),
            Value::Argument(_) | Value::Constant(_) | Value::Undefined => {}
        }
    }
}

pub struct ValueBuilder;

impl ValueBuilder {
    pub fn constant_int(value: i32) -> Value {
        Value::Constant(Constant::Int(value))
    }

    pub fn constant_bool(value: bool) -> Value {
        Value::Constant(Constant::Bool(value))
    }

    pub fn constant_string(value: String) -> Value {
        Value::Constant(Constant::String(value))
    }

    pub fn add(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Add(Commutative::new(lhs, rhs)))
    }

    pub fn sub(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Sub(NonCommutative::new(lhs, rhs)))
    }

    pub fn mul(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Mul(Commutative::new(lhs, rhs)))
    }

    pub fn div(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Div(NonCommutative::new(lhs, rhs)))
    }

    pub fn mod_(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Mod(NonCommutative::new(lhs, rhs)))
    }

    pub fn eq(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Eq(Commutative::new(lhs, rhs)))
    }

    pub fn neq(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Neq(Commutative::new(lhs, rhs)))
    }

    pub fn lt(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Lt(NonCommutative::new(lhs, rhs)))
    }

    pub fn gt(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Gt(NonCommutative::new(lhs, rhs)))
    }

    pub fn lte(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Lte(NonCommutative::new(lhs, rhs)))
    }

    pub fn gte(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Gte(NonCommutative::new(lhs, rhs)))
    }

    pub fn concat(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Concat(NonCommutative::new(lhs, rhs)))
    }

    pub fn and(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::And(Commutative::new(lhs, rhs)))
    }

    pub fn or(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Or(Commutative::new(lhs, rhs)))
    }

    pub fn not(value: ValueId) -> Value {
        Value::UnaryOperation(UnaryOperation::Not(value))
    }

    pub fn neg(value: ValueId) -> Value {
        Value::UnaryOperation(UnaryOperation::Neg(value))
    }

    pub fn arg(id: u32) -> Value {
        Value::Argument(ArgumentId(id))
    }

    pub fn void() -> Value {
        Value::Undefined
    }
}

impl<'a> Function<'a> {
    pub fn new(context: &'a Context) -> Function<'a> {
        Self {
            blocks: Default::default(),
            variables: Default::default(),
            values: Default::default(),
            call_counter: 0,
            ssa_counter: 0,
            context,
        }
    }

    pub fn call(
        &mut self,
        block: BlockId,
        function: FunctionId,
        arguments: Vec<ValueId>,
    ) -> ValueId {
        let id = self.call_counter;
        self.call_counter += 1;
        let value = Value::Call(Call {
            id: CallId(id),
            function,
            arguments,
        });
        let ty = self
            .context
            .functions
            .get(&function)
            .cloned()
            .unwrap()
            .return_ty;
        self.values.new(value, ty)
    }

    pub fn new_variable(&mut self, ty: IrType) -> VariableId {
        self.variables.new(ty)
    }

    pub fn new_block(&mut self) -> BlockId {
        self.blocks.new()
    }

    pub fn new_ssa_name(&mut self) -> SsaName {
        let id = self.ssa_counter;
        self.ssa_counter += 1;
        SsaName(id)
    }

    pub fn write_variable(&mut self, var: VariableId, block: BlockId, value: Value) -> ValueId {
        let ty = self.variables.ty(var);
        let value = self.values.new(value, ty);
        self.variables.set_in_block(var, block, value);
        value
    }

    pub fn read_variable(&mut self, var: VariableId, block_id: BlockId) -> ValueId {
        let ty = self.variables.ty(var);
        if let Some(value) = self.variables.get_in_block(var, block_id) {
            value
        } else {
            let block = &self.blocks.get(block_id);
            let sealed = block.sealed;
            let val = if !sealed {
                // Incomplete CFG
                let val = Value::Phi(Phi::new(block_id));
                let val = self.values.new(val, ty);
                self.blocks.get_mut(block_id).add_incopmlete_phi(var, val);
                val
            } else if let Some(pred) = self.blocks.single_predecessor(block_id) {
                // Optimize the common case of one predecessor: No phi needed
                self.read_variable(var, pred)
            } else {
                // Break potential cycles with operandless phi
                let val = Value::Phi(Phi::new(block_id));
                let val = self.write_variable(var, block_id, val);
                self.add_phi_operands(var, val)
            };
            self.variables.set_in_block(var, block_id, val);
            val
        }
    }

    fn add_phi_operand(&mut self, phi_id: ValueId, block: BlockId, value: ValueId) {
        let phi = self.values.get_mut(phi_id).as_phi_mut();
        phi.operands.insert(block, value);
        self.values.add_user(value, phi_id);
    }

    fn remove_trivial_phi(&mut self, phi: ValueId) -> ValueId {
        let mut same = None;
        for (_, value) in &self.values.get(phi).as_phi().operands {
            debug_assert_eq!(self.values.ty(*value), self.values.ty(phi));
            if *value == phi {
                continue; // ignore self-references
            }
            if let Some(same) = same {
                if same == *value {
                    continue; // ignore duplicates
                } else {
                    return phi; // not trivial, merges multiple values
                }
            } else {
                same = Some(*value);
            }
        }

        let same = if let Some(same) = same {
            same
        } else {
            let ty = self.values.ty(phi);
            let undefined = self.values.new(Value::Undefined, ty);
            undefined // this phi is unreachable or in start block
        };

        // Remember all users except the phi itself
        let users = {
            let mut users = self.values.users(phi);
            users.remove(&phi);
            users
        };

        for user_id in users {
            // Reroute all uses of phi to same
            let user = self.values.get_mut(user_id);
            user.replace_use(phi, same);
            if let Value::Phi(_) = user {
                // Try to recursively remove all phi users, which might have become trivial
                self.remove_trivial_phi(user_id);
            }
        }

        // TODO: remove pi?

        same
    }

    fn add_phi_operands(&mut self, var: VariableId, phi_id: ValueId) -> ValueId {
        // Determine operands from predecessors
        let phi = self.values.get_mut(phi_id).as_phi_mut();
        let block = self.blocks.get(phi.block);
        for pred in &block.preds.clone() {
            let pred_value = self.read_variable(var, *pred);
            self.add_phi_operand(phi_id, *pred, pred_value);
        }

        self.remove_trivial_phi(phi_id)
    }

    pub fn seal_block(&mut self, block_id: BlockId) {
        let block = self.blocks.get_mut(block_id);
        let phis = block.take_incomplete_phis();
        for (var, phi) in phis {
            self.add_phi_operands(var, phi);
        }
        let block = self.blocks.get_mut(block_id);
        block.seal();
    }

    // fn dump(&self) {
    //     for (id, block) in self.blocks.all() {
    //         println!("Block {:?}:", id);
    //         println!("  preds: {:?}", block.preds);
    //
    //         for instr in &block.instructions {
    //             match instr {
    //                 Instruction::UnconditionalJump(b) => println!("  jump {:?}", b),
    //                 Instruction::ConditionalJump {
    //                     test_value,
    //                     true_block,
    //                     false_block,
    //                 } => {
    //                     let value = self.values.get(*test_value);
    //                     println!(
    //                         "  if {:?} ({:?}) jump {:?} else {:?}",
    //                         value, test_value, true_block, false_block
    //                     )
    //                 }
    //                 Instruction::DefineValue(id) => {
    //                     let value = &self.values.get(*id);
    //                     println!("  {:?} = {:?}", id, value);
    //                 }
    //             }
    //         }
    //     }
    // }
}

mod tests {
    use super::*;

    #[test]
    fn int_constant_folding() {
        let context = Context::default();

        let mut function = Function::new(&context);
        let block = function.new_block();
        let lhs = function
            .values
            .new(ValueBuilder::constant_int(7), IrType::Int);
        let rhs = function
            .values
            .new(ValueBuilder::constant_int(3), IrType::Int);

        let sum = function
            .values
            .new(ValueBuilder::add(lhs, rhs), IrType::Int);
        let sum = function.values.get(sum);
        assert_eq!(sum, &ValueBuilder::constant_int(10));

        let diff = function
            .values
            .new(ValueBuilder::sub(lhs, rhs), IrType::Int);
        let diff = function.values.get(diff);
        assert_eq!(diff, &ValueBuilder::constant_int(4));

        let prod = function
            .values
            .new(ValueBuilder::mul(lhs, rhs), IrType::Int);
        let prod = function.values.get(prod);
        assert_eq!(prod, &ValueBuilder::constant_int(21));

        let quot = function
            .values
            .new(ValueBuilder::div(lhs, rhs), IrType::Int);
        let quot = function.values.get(quot);
        assert_eq!(quot, &ValueBuilder::constant_int(2));

        let rem = function
            .values
            .new(ValueBuilder::mod_(lhs, rhs), IrType::Int);
        let rem = function.values.get(rem);
        assert_eq!(rem, &ValueBuilder::constant_int(1));

        let neg = function.values.new(ValueBuilder::neg(lhs), IrType::Int);
        let neg = function.values.get(neg);
        assert_eq!(neg, &ValueBuilder::constant_int(-7));
    }

    #[test]
    fn const_folding_bool() {
        let context = Context::default();

        let mut function = Function::new(&context);
        let block = function.new_block();
        let lhs = function
            .values
            .new(ValueBuilder::constant_bool(true), IrType::Bool);
        let rhs = function
            .values
            .new(ValueBuilder::constant_bool(false), IrType::Bool);

        let and = function
            .values
            .new(ValueBuilder::and(lhs, rhs), IrType::Bool);
        let and = function.values.get(and);
        assert_eq!(and, &ValueBuilder::constant_bool(false));

        let or = function
            .values
            .new(ValueBuilder::or(lhs, rhs), IrType::Bool);
        let or = function.values.get(or);
        assert_eq!(or, &ValueBuilder::constant_bool(true));

        let not = function.values.new(ValueBuilder::not(lhs), IrType::Bool);
        let not = function.values.get(not);
        assert_eq!(not, &ValueBuilder::constant_bool(false));
    }

    #[test]
    fn const_folding_comparison() {
        let context = Context::default();

        let mut function = Function::new(&context);
        let block = function.new_block();
        let lhs = function
            .values
            .new(ValueBuilder::constant_int(7), IrType::Int);
        let rhs = function
            .values
            .new(ValueBuilder::constant_int(3), IrType::Int);

        let eq = function
            .values
            .new(ValueBuilder::eq(lhs, rhs), IrType::Bool);
        let eq = function.values.get(eq);
        assert_eq!(eq, &ValueBuilder::constant_bool(false));

        let neq = function
            .values
            .new(ValueBuilder::neq(lhs, rhs), IrType::Bool);
        let neq = function.values.get(neq);
        assert_eq!(neq, &ValueBuilder::constant_bool(true));

        let lt = function
            .values
            .new(ValueBuilder::lt(lhs, rhs), IrType::Bool);
        let lt = function.values.get(lt);
        assert_eq!(lt, &ValueBuilder::constant_bool(false));

        let gt = function
            .values
            .new(ValueBuilder::gt(lhs, rhs), IrType::Bool);
        let gt = function.values.get(gt);
        assert_eq!(gt, &ValueBuilder::constant_bool(true));

        let leq = function
            .values
            .new(ValueBuilder::lte(lhs, rhs), IrType::Bool);
        let leq = function.values.get(leq);
        assert_eq!(leq, &ValueBuilder::constant_bool(false));

        let geq = function
            .values
            .new(ValueBuilder::gte(lhs, rhs), IrType::Bool);
        let geq = function.values.get(geq);
        assert_eq!(geq, &ValueBuilder::constant_bool(true));
    }

    #[test]
    fn const_folding_nested() {
        let context = Context::default();

        let mut function = Function::new(&context);
        let block = function.new_block();
        let arg1 = function
            .values
            .new(ValueBuilder::constant_int(7), IrType::Int);
        let arg2 = function
            .values
            .new(ValueBuilder::constant_int(3), IrType::Int);
        let arg3 = function
            .values
            .new(ValueBuilder::constant_int(2), IrType::Int);

        let sum1_2 = function
            .values
            .new(ValueBuilder::add(arg1, arg2), IrType::Int);

        let sum1_2_3 = function
            .values
            .new(ValueBuilder::add(sum1_2, arg3), IrType::Int);

        let comp = function
            .values
            .new(ValueBuilder::eq(sum1_2_3, arg1), IrType::Bool);
        assert_eq!(
            function.values.get(comp),
            &ValueBuilder::constant_bool(false)
        );

        let false_ = function
            .values
            .new(ValueBuilder::constant_bool(false), IrType::Bool);

        let or = function
            .values
            .new(ValueBuilder::or(comp, false_), IrType::Bool);

        let or = function.values.get(or);
        assert_eq!(or, &ValueBuilder::constant_bool(false));
    }

    #[test]
    fn local_value_numbering_constant() {
        let context = Context::default();

        let mut function = Function::new(&context);
        let block = function.new_block();
        let constant = function
            .values
            .new(ValueBuilder::constant_int(7), IrType::Int);
        let constant2 = function
            .values
            .new(ValueBuilder::constant_int(7), IrType::Int);
        assert_eq!(constant, constant2);
    }

    #[test]
    fn local_value_numbering_variable() {
        let context = Context::default();

        let mut function = Function::new(&context);
        let block = function.blocks.new();
        let var = function.variables.new(IrType::Int);
        let value_written = function.write_variable(var, block, ValueBuilder::constant_int(42));
        let value_read = function.read_variable(var, block);
        assert_eq!(value_written, value_read);
        let value_written2 = function.write_variable(var, block, ValueBuilder::constant_int(42));
        assert_eq!(value_written, value_written2);

        let var2 = function.variables.new(IrType::Int);
        let value_written3 = function.write_variable(var2, block, ValueBuilder::constant_int(42));
        assert_eq!(value_written, value_written3);
    }

    #[test]
    fn global_value_numbering() {
        let context = Context::default();

        let mut function = Function::new(&context);
        let changing = function.variables.new(IrType::Int);
        let not_changing = function.variables.new(IrType::Int);
        let not_changing_2 = function.variables.new(IrType::Int);
        let start = function.blocks.new();
        let value_changing_start =
            function.write_variable(changing, start, ValueBuilder::constant_int(42));
        let value_not_changing_start =
            function.write_variable(not_changing, start, ValueBuilder::constant_int(314));
        let value_not_changing_start_2 =
            function.write_variable(not_changing_2, start, ValueBuilder::constant_int(42));

        let true_ = function
            .values
            .new(ValueBuilder::constant_bool(true), IrType::Bool);
        function.seal_block(start);
        let pred1 = function.blocks.new();
        let value_changing_pred1 =
            function.write_variable(changing, pred1, ValueBuilder::constant_int(420));
        function.blocks.declare_predecessor(pred1, start);
        function.seal_block(pred1);
        let pred2 = function.blocks.new();
        function.blocks.declare_predecessor(pred2, start);
        function.seal_block(pred2);
        let join = function.blocks.new();
        function.blocks.declare_predecessor(join, pred1);
        function.blocks.declare_predecessor(join, pred2);
        function.seal_block(join);
        let value_changing_join = function.read_variable(changing, join);
        let value_changing_join = function.values.get(value_changing_join);
        assert_eq!(
            value_changing_join,
            &Value::Phi(Phi {
                block: join,
                operands: BTreeMap::from([
                    (pred1, value_changing_pred1),
                    (pred2, value_changing_start)
                ])
            })
        );

        let value_not_changing_join = function.read_variable(not_changing, join);
        assert_eq!(value_not_changing_start, value_not_changing_join);

        let value_not_changing_join_2 = function.read_variable(not_changing_2, join);
        assert_eq!(value_not_changing_start_2, value_not_changing_join_2);
    }

    // #[test]
    // fn test() {
    //     let context = Context::default();
    //
    //     let mut function = Function::new(&context);
    //     let start = function.blocks.new();
    //     let true_block = function.blocks.new();
    //     function.blocks.declare_predecessor(true_block, start);
    //     let false_block = function.blocks.new();
    //     function.blocks.declare_predecessor(false_block, start);
    //     let join_block = function.blocks.new();
    //     function.blocks.declare_predecessor(join_block, true_block);
    //     function.blocks.declare_predecessor(join_block, false_block);
    //
    //     // %1 = 1
    //     // if %1 goto true_block else false_block
    //     let test_value = function
    //         .values
    //         .new(ValueBuilder::constant_int(1), IrType::Int);
    //     function.add_instruction(start, Instruction::DefineValue(test_value));
    //     function.add_instruction(
    //         start,
    //         Instruction::ConditionalJump {
    //             test_value,
    //             true_block,
    //             false_block,
    //         },
    //     );
    //     function.seal_block(start);
    //
    //     // var = 2
    //     // temp = var
    //     // var = 3
    //     // var = var + temp
    //     let variable = function.new_variable(IrType::Int);
    //     function.write_variable(variable, true_block, ValueBuilder::constant_int(2));
    //     let last = function.read_variable(variable, true_block);
    //     function.add_instruction(true_block, Instruction::DefineValue(last));
    //     function.write_variable(variable, true_block, ValueBuilder::constant_int(3));
    //     let last_2 = function.read_variable(variable, true_block);
    //     function.add_instruction(true_block, Instruction::DefineValue(last_2));
    //
    //     let sum = function.write_variable(variable, true_block, ValueBuilder::add(last, last_2));
    //     function.add_instruction(true_block, Instruction::DefineValue(sum));
    //     function.add_instruction(true_block, Instruction::UnconditionalJump(join_block));
    //     function.seal_block(true_block);
    //
    //     // var = 4
    //     let set_in_false =
    //         function.write_variable(variable, false_block, ValueBuilder::constant_int(3));
    //     function.add_instruction(false_block, Instruction::DefineValue(set_in_false));
    //     function.seal_block(false_block);
    //
    //     // var = var + 5
    //     let last = function.read_variable(variable, join_block);
    //     function.add_instruction(join_block, Instruction::DefineValue(last));
    //     let const_5 = function
    //         .values
    //         .new(ValueBuilder::constant_int(5), IrType::Int);
    //     function.add_instruction(join_block, Instruction::DefineValue(const_5));
    //     let sum = function
    //         .values
    //         .new(ValueBuilder::add(last, const_5), IrType::Int);
    //     function.add_instruction(join_block, Instruction::DefineValue(sum));
    //     function.seal_block(join_block);
    //
    //     function.dump();
    // }
}
