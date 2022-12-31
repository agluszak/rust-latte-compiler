use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::Hash;
use std::mem;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Instruction {
    UnconditionalJump(BlockId),
    ConditionalJump {
        test_value: ValueId,
        true_block: BlockId,
        false_block: BlockId,
    },
    DefineValue(ValueId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub preds: HashSet<BlockId>,
    pub instructions: Vec<Instruction>,
    pub sealed: bool,
    incomplete_phis: HashMap<VariableId, ValueId>,
}

impl Block {
    pub fn new() -> Self {
        Self {
            preds: HashSet::new(),
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
    Pointer(Box<IrType>),
    Function {
        args: Vec<IrType>,
        return_type: Option<Box<IrType>>,
    },
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

    fn predecessors(&self, block: BlockId) -> HashSet<BlockId> {
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

#[derive(Debug, Default)]
pub struct Function {
    // pub name: String, // TODO: interning
    blocks: Blocks,
    variables: Variables,
    values: Values,
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
    Call(FunctionId, Vec<ValueId>),
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
            Value::Call(_, operands) => {
                for operand in operands.iter_mut() {
                    if *operand == old {
                        *operand = new;
                    }
                }
            }
            Value::Argument(_) | Value::Constant(_) | Value::Undefined => {}
        }
    }
}

struct ValueBuilder;

impl ValueBuilder {
    fn constant_int(value: i32) -> Value {
        Value::Constant(Constant::Int(value))
    }

    fn constant_bool(value: bool) -> Value {
        Value::Constant(Constant::Bool(value))
    }

    fn constant_string(value: String) -> Value {
        Value::Constant(Constant::String(value))
    }

    fn add(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Add(Commutative::new(lhs, rhs)))
    }

    fn sub(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Sub(NonCommutative::new(lhs, rhs)))
    }

    fn mul(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Mul(Commutative::new(lhs, rhs)))
    }

    fn div(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Div(NonCommutative::new(lhs, rhs)))
    }

    fn mod_(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Mod(NonCommutative::new(lhs, rhs)))
    }

    fn eq(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Eq(Commutative::new(lhs, rhs)))
    }

    fn neq(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Neq(Commutative::new(lhs, rhs)))
    }

    fn lt(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Lt(NonCommutative::new(lhs, rhs)))
    }

    fn gt(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Gt(NonCommutative::new(lhs, rhs)))
    }

    fn lte(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Lte(NonCommutative::new(lhs, rhs)))
    }

    fn gte(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Gte(NonCommutative::new(lhs, rhs)))
    }

    fn concat(lhs: ValueId, rhs: ValueId) -> Value {
        Value::BinaryOperation(BinaryOperation::Concat(NonCommutative::new(lhs, rhs)))
    }
}

impl Function {
    pub fn add_instruction(&mut self, block: BlockId, instruction: Instruction) {
        self.blocks.get_mut(block).add_instruction(instruction);
    }

    pub fn new_variable(&mut self, ty: IrType) -> VariableId {
        self.variables.new(ty)
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

    fn dump(&self) {
        for (id, block) in self.blocks.all() {
            println!("Block {:?}:", id);
            println!("  preds: {:?}", block.preds);

            for instr in &block.instructions {
                match instr {
                    Instruction::UnconditionalJump(b) => println!("  jump {:?}", b),
                    Instruction::ConditionalJump {
                        test_value,
                        true_block,
                        false_block,
                    } => {
                        let value = self.values.get(*test_value);
                        println!(
                            "  if {:?} ({:?}) jump {:?} else {:?}",
                            value, test_value, true_block, false_block
                        )
                    }
                    Instruction::DefineValue(id) => {
                        let value = &self.values.get(*id);
                        println!("  {:?} = {:?}", id, value);
                    }
                }
            }
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn local_value_numbering_constant() {
        let mut function = Function::default();
        let constant = function
            .values
            .new(ValueBuilder::constant_int(42), IrType::Int);
        let constant2 = function
            .values
            .new(ValueBuilder::constant_int(42), IrType::Int);
        assert_eq!(constant, constant2);
    }

    #[test]
    fn local_value_numbering_variable() {
        let mut function = Function::default();
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
        let mut function = Function::default();
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

    #[test]
    fn test() {
        let mut function = Function::default();
        let start = function.blocks.new();
        let true_block = function.blocks.new();
        function.blocks.declare_predecessor(true_block, start);
        let false_block = function.blocks.new();
        function.blocks.declare_predecessor(false_block, start);
        let join_block = function.blocks.new();
        function.blocks.declare_predecessor(join_block, true_block);
        function.blocks.declare_predecessor(join_block, false_block);

        // %1 = 1
        // if %1 goto true_block else false_block
        let test_value = function
            .values
            .new(ValueBuilder::constant_int(1), IrType::Int);
        function.add_instruction(start, Instruction::DefineValue(test_value));
        function.add_instruction(
            start,
            Instruction::ConditionalJump {
                test_value,
                true_block,
                false_block,
            },
        );
        function.seal_block(start);

        // var = 2
        // temp = var
        // var = 3
        // var = var + temp
        let variable = function.new_variable(IrType::Int);
        function.write_variable(variable, true_block, ValueBuilder::constant_int(2));
        let last = function.read_variable(variable, true_block);
        function.add_instruction(true_block, Instruction::DefineValue(last));
        function.write_variable(variable, true_block, ValueBuilder::constant_int(3));
        let last_2 = function.read_variable(variable, true_block);
        function.add_instruction(true_block, Instruction::DefineValue(last_2));

        let sum = function.write_variable(variable, true_block, ValueBuilder::add(last, last_2));
        function.add_instruction(true_block, Instruction::DefineValue(sum));
        function.add_instruction(true_block, Instruction::UnconditionalJump(join_block));
        function.seal_block(true_block);

        // var = 4
        let set_in_false =
            function.write_variable(variable, false_block, ValueBuilder::constant_int(3));
        function.add_instruction(false_block, Instruction::DefineValue(set_in_false));
        function.seal_block(false_block);

        // var = var + 5
        let last = function.read_variable(variable, join_block);
        function.add_instruction(join_block, Instruction::DefineValue(last));
        let const_5 = function
            .values
            .new(ValueBuilder::constant_int(5), IrType::Int);
        function.add_instruction(join_block, Instruction::DefineValue(const_5));
        let sum = function
            .values
            .new(ValueBuilder::add(last, const_5), IrType::Int);
        function.add_instruction(join_block, Instruction::DefineValue(sum));
        function.seal_block(join_block);

        function.dump();
    }
}
