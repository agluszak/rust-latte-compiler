use crate::ir::Value::Constant;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub enum Instruction {
    UnconditionalJump(BlockId),
    ConditionalJump {
        test_value: ValueId,
        true_block: BlockId,
        false_block: BlockId,
    },
    DefineValue(ValueId),
}

pub struct Block {
    pub preds: HashSet<BlockId>,
    pub instructions: Vec<Instruction>,
}

impl Block {
    pub fn new() -> Self {
        Self {
            preds: HashSet::new(),
            instructions: Vec::new(),
        }
    }

    fn declare_pred(&mut self, pred: BlockId) {
        self.preds.insert(pred);
    }

    fn add_instruction(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariableId(u32);

pub struct Function {
    // pub name: String, // TODO: interning
    pub blocks: HashMap<BlockId, Block>,
    pub variables: HashMap<VariableId, HashMap<BlockId, ValueId>>,
    pub incomplete_phis: HashMap<BlockId, HashMap<VariableId, ValueId>>,
    pub sealed_blocks: HashSet<BlockId>,
    pub values: HashMap<ValueId, Value>,
    pub value_users: HashMap<ValueId, HashSet<ValueId>>,
}

impl Function {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            variables: HashMap::new(),
            incomplete_phis: HashMap::new(),
            sealed_blocks: HashSet::new(),
            values: HashMap::new(),
            value_users: HashMap::new(),
        }
    }

    pub fn add_instruction(&mut self, block: BlockId, instruction: Instruction) {
        self.blocks
            .get_mut(&block)
            .unwrap()
            .add_instruction(instruction);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phi {
    pub block: BlockId,
    pub operands: Vec<(BlockId, ValueId)>,
    pub users: HashSet<ValueId>,
}

impl Phi {
    pub fn new(block: BlockId) -> Self {
        Self {
            block,
            operands: Vec::new(),
            users: HashSet::new(),
        }
    }

    fn add_user(&mut self, user: ValueId) {
        self.users.insert(user);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Phi(Phi),
    Constant(i32),
    Add(ValueId, ValueId),
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
            _ => {}
        }
    }
}

impl Function {
    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.insert(id, Block::new());
        id
    }

    fn declare_predecessor(&mut self, block: BlockId, pred: BlockId) {
        self.blocks.get_mut(&block).unwrap().declare_pred(pred);
    }

    fn new_variable(&mut self) -> VariableId {
        let id = VariableId(self.variables.len() as u32);
        self.variables.insert(id, HashMap::new());
        id
    }

    fn new_value(&mut self, value: Value) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.insert(id, value);
        id
    }

    pub fn write_variable(&mut self, var: VariableId, block: BlockId, value: Value) -> ValueId {
        let value = self.new_value(value);
        self.write_variable_with_id(var, block, value);
        value
    }

    fn write_variable_with_id(&mut self, var: VariableId, block: BlockId, value: ValueId) {
        self.variables.entry(var).or_default().insert(block, value);
    }

    pub fn read_variable(&mut self, var: VariableId, block: BlockId) -> ValueId {
        if let Some(value) = self.variables[&var].get(&block) {
            *value
        } else {
            self.read_variable_recursive(var, block)
        }
    }

    fn add_phi_operand(&mut self, phi_id: ValueId, block: BlockId, value: ValueId) {
        let phi = self.values.get_mut(&phi_id).unwrap().as_phi_mut();
        phi.operands.push((block, value));
        let value = self.values.get_mut(&value).unwrap();
        if let Value::Phi(value_phi) = value {
            value_phi.add_user(phi_id);
        }
    }

    fn remove_trivial_phi(&mut self, phi: ValueId) -> ValueId {
        let mut same = None;
        for (_, value) in &self.values[&phi].as_phi().operands {
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
            let undefined = self.new_value(Value::Undefined);
            undefined // this phi is unreachable or in start block
        };

        // Remember all users except the phi itself
        let users = {
            let mut users = self.value_users[&phi].clone();
            users.remove(&phi);
            users
        };

        for user_id in users {
            // Reroute all uses of phi to same
            let user = self.values.get_mut(&user_id).unwrap();
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
        let phi = self.values.get_mut(&phi_id).unwrap().as_phi_mut();
        let block = &self.blocks[&phi.block];
        for pred in &block.preds.clone() {
            let pred_value = self.read_variable(var, *pred);
            self.add_phi_operand(phi_id, *pred, pred_value);
        }

        self.remove_trivial_phi(phi_id)
    }

    fn single_pred(&self, block: BlockId) -> Option<BlockId> {
        let preds = &self.blocks[&block].preds;
        if preds.len() == 1 {
            Some(*preds.iter().next().unwrap())
        } else {
            None
        }
    }

    fn read_variable_recursive(&mut self, var: VariableId, block: BlockId) -> ValueId {
        let val = if !self.sealed_blocks.contains(&block) {
            // Incomplete CFG
            let val = Value::Phi(Phi::new(block));
            let val = self.new_value(val);
            self.incomplete_phis
                .entry(block)
                .or_default()
                .insert(var, val);
            val
        } else if let Some(pred) = self.single_pred(block) {
            // Optimize the common case of one predecessor: No phi needed
            self.read_variable(var, pred)
        } else {
            // Break potential cycles with operandless phi
            let val = Value::Phi(Phi::new(block));
            let val = self.write_variable(var, block, val);
            self.add_phi_operands(var, val)
        };
        self.write_variable_with_id(var, block, val);
        val
    }

    fn seal_block(&mut self, block: BlockId) {
        if let Some(phis) = self.incomplete_phis.remove(&block) {
            for (var, phi) in phis {
                self.add_phi_operands(var, phi);
            }
        }
        self.sealed_blocks.insert(block);
    }

    fn dump(&self) {
        for (id, block) in &self.blocks {
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
                        let value = &self.values[test_value];
                        println!(
                            "  if {:?} ({:?}) jump {:?} else {:?}",
                            value, test_value, true_block, false_block
                        )
                    }
                    Instruction::DefineValue(id) => {
                        let value = &self.values[id];
                        println!("  {:?} = {:?}", id, value);
                    }
                }
            }
        }
    }
}

#[test]
fn test() {
    let mut function = Function::new();
    let start = function.new_block();
    let true_block = function.new_block();
    function.declare_predecessor(true_block, start);
    let false_block = function.new_block();
    function.declare_predecessor(false_block, start);
    let join_block = function.new_block();
    function.declare_predecessor(join_block, true_block);
    function.declare_predecessor(join_block, false_block);

    // %1 = 1
    // if %1 goto true_block else false_block
    let test_value = function.new_value(Constant(1));
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
    let variable = function.new_variable();
    function.write_variable(variable, true_block, Constant(2));
    let last = function.read_variable(variable, true_block);
    function.add_instruction(true_block, Instruction::DefineValue(last));
    function.write_variable(variable, true_block, Constant(3));
    let last_2 = function.read_variable(variable, true_block);
    function.add_instruction(true_block, Instruction::DefineValue(last_2));

    let sum = function.write_variable(variable, true_block, Value::Add(last, last_2));
    function.add_instruction(true_block, Instruction::DefineValue(sum));
    function.add_instruction(true_block, Instruction::UnconditionalJump(join_block));
    function.seal_block(true_block);

    // var = 4
    let set_in_false = function.write_variable(variable, false_block, Constant(4));
    function.add_instruction(false_block, Instruction::DefineValue(set_in_false));
    function.seal_block(false_block);

    // var = var + 5
    let last = function.read_variable(variable, join_block);
    function.add_instruction(join_block, Instruction::DefineValue(last));
    let const_5 = function.new_value(Constant(5));
    function.add_instruction(join_block, Instruction::DefineValue(const_5));
    let sum = function.new_value(Value::Add(last, const_5));
    function.add_instruction(join_block, Instruction::DefineValue(sum));
    function.seal_block(join_block);

    function.dump();
}
