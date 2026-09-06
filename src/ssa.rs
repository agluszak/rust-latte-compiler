use crate::ir::{BasicBlock, BlockId, FunctionIr, Phi, Terminator, Value, ValueData, ValueId};
use crate::types::Type;
use crate::typed_ast::VariableId;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildingPhi {
    pub(crate) block: BlockId,
    pub(crate) incoming: Vec<(BlockId, ValueId)>,
    pub(crate) users: Vec<ValueId>,
}

impl BuildingPhi {
    pub(crate) fn new(block: BlockId) -> Self {
        Self {
            incoming: Vec::new(),
            block,
            users: Vec::new(),
        }
    }

    pub(crate) fn add_incoming(&mut self, block: BlockId, value: ValueId) {
        self.incoming.push((block, value));
    }

    pub(crate) fn add_user(&mut self, user: ValueId) {
        if !self.users.contains(&user) {
            self.users.push(user);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BuildingValue {
    Int(i32),
    String(String),
    Bool(bool),
    Call(VariableId, Vec<ValueId>),
    Argument(u32),
    BinaryOp(crate::ir::BinaryOpCode, ValueId, ValueId),
    UnaryOp(crate::ir::UnaryOpCode, ValueId),
    Phi(BuildingPhi),
    Undef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildingValueData {
    pub(crate) ty: Type,
    pub(crate) kind: BuildingValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildingBlock {
    pub(crate) phis: Vec<ValueId>,
    pub(crate) instructions: Vec<ValueId>,
    pub(crate) terminator: Option<Terminator>,
    pub(crate) predecessors: Vec<BlockId>,
    pub(crate) sealed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IrBuilder {
    next_value_id: u32,
    next_block_id: u32,
    current_definitions: BTreeMap<VariableId, BTreeMap<BlockId, ValueId>>,
    variable_types: BTreeMap<VariableId, Type>,
    values: BTreeMap<ValueId, BuildingValueData>,
    aliases: BTreeMap<ValueId, ValueId>,
    blocks: BTreeMap<BlockId, BuildingBlock>,
    incomplete_phis: BTreeMap<BlockId, BTreeMap<VariableId, ValueId>>,
}

impl IrBuilder {
    pub(crate) fn new() -> Self {
        Self {
            next_value_id: 0,
            next_block_id: 0,
            current_definitions: BTreeMap::new(),
            variable_types: BTreeMap::new(),
            values: BTreeMap::new(),
            aliases: BTreeMap::new(),
            blocks: BTreeMap::new(),
            incomplete_phis: BTreeMap::new(),
        }
    }

    fn new_value_id(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    fn new_block_id(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    fn allocate(&mut self, kind: BuildingValue, ty: Type) -> ValueId {
        let id = self.new_value_id();
        self.values.insert(id, BuildingValueData { ty, kind });
        id
    }

    pub(crate) fn emit(&mut self, block: BlockId, kind: BuildingValue, ty: Type) -> ValueId {
        assert!(self.blocks[&block].terminator.is_none());
        let id = self.allocate(kind, ty);
        self.blocks.get_mut(&block).unwrap().instructions.push(id);
        id
    }

    pub(crate) fn new_phi(&mut self, block: BlockId, ty: Type) -> ValueId {
        let id = self.allocate(BuildingValue::Phi(BuildingPhi::new(block)), ty);
        self.blocks.get_mut(&block).unwrap().phis.push(id);
        id
    }

    pub(crate) fn resolve_alias(&mut self, id: ValueId) -> ValueId {
        let Some(&next) = self.aliases.get(&id) else {
            return id;
        };
        let resolved = self.resolve_alias(next);
        self.aliases.insert(id, resolved);
        resolved
    }

    pub(crate) fn finish_block(&mut self, block_id: BlockId, terminator: Terminator) {
        assert!(self.blocks[&block_id].terminator.is_none());
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
        self.blocks.get_mut(&block_id).unwrap().terminator = Some(terminator);
    }

    fn add_predecessor(&mut self, block: BlockId, pred: BlockId) {
        let block = self.blocks.get_mut(&block).unwrap();
        assert!(!block.sealed);
        if !block.predecessors.contains(&pred) {
            block.predecessors.push(pred);
        }
    }

    pub(crate) fn new_block(&mut self) -> BlockId {
        let id = self.new_block_id();
        self.blocks.insert(
            id,
            BuildingBlock {
                phis: Vec::new(),
                instructions: Vec::new(),
                terminator: None,
                predecessors: Vec::new(),
                sealed: false,
            },
        );
        id
    }

    pub(crate) fn write_variable(&mut self, variable: VariableId, block: BlockId, value: ValueId) {
        let value = self.resolve_alias(value);
        let value_ty = self.values[&value].ty.clone();
        self.variable_types.insert(variable, value_ty);
        self.current_definitions
            .entry(variable)
            .or_default()
            .insert(block, value);
    }

    pub(crate) fn read_variable(&mut self, variable: VariableId, block_id: BlockId) -> ValueId {
        if let Some(value) = self
            .current_definitions
            .get(&variable)
            .and_then(|map| map.get(&block_id))
            .copied()
        {
            self.resolve_alias(value)
        } else {
            let sealed = self.blocks[&block_id].sealed;
            let predecessors = self.blocks[&block_id].predecessors.clone();
            let ty = self
                .variable_types
                .get(&variable)
                .cloned()
                .unwrap_or_else(|| panic!("Variable {:?} not found", variable));
            let val = if !sealed {
                // Incomplete CFG
                let id = self.new_phi(block_id, ty);
                self.incomplete_phis
                    .entry(block_id)
                    .or_default()
                    .insert(variable, id);
                id
            } else if predecessors.len() == 1 {
                // Optimize the common case of a single predecessor: no phi needed
                self.read_variable(variable, predecessors[0])
            } else {
                // Break potential cycles with operandless phi
                let val = self.new_phi(block_id, ty);
                self.write_variable(variable, block_id, val);

                self.add_phi_operands(variable, val)
            };
            self.write_variable(variable, block_id, val);
            val
        }
    }

    fn add_phi_operands(&mut self, variable: VariableId, phi_id: ValueId) -> ValueId {
        let phi_id = self.resolve_alias(phi_id);
        let phi_block = self.get_phi(phi_id).unwrap().block;
        for pred in self.blocks[&phi_block].predecessors.clone() {
            let pred_val = self.read_variable(variable, pred);
            self.add_phi_incoming(phi_id, pred, pred_val);
        }
        self.try_remove_trivial_phi(phi_id)
    }

    /// Adds `(block, value)` as an incoming edge of the building phi `phi`.
    ///
    /// Centralizes the SSA invariant that registering a phi operand must also
    /// update the use relation when the operand is itself a building phi.
    pub(crate) fn add_phi_incoming(&mut self, phi: ValueId, block: BlockId, value: ValueId) {
        let value = self.resolve_alias(value);

        if let Some(operand_phi) = self.get_phi(value) {
            operand_phi.add_user(phi);
        }

        self.get_phi(phi).unwrap().add_incoming(block, value);
    }

    fn get_phi(&mut self, phi_id: ValueId) -> Option<&mut BuildingPhi> {
        match &mut self.values.get_mut(&phi_id)?.kind {
            BuildingValue::Phi(phi) => Some(phi),
            _ => None,
        }
    }

    pub(crate) fn seal_block(&mut self, block_id: BlockId) {
        assert!(!self.blocks[&block_id].sealed);
        self.blocks.get_mut(&block_id).unwrap().sealed = true;
        if let Some(incomplete_phis) = self.incomplete_phis.remove(&block_id) {
            for (variable, phi_id) in incomplete_phis {
                self.add_phi_operands(variable, phi_id);
            }
        }
    }

    pub(crate) fn try_remove_trivial_phi(&mut self, phi_id: ValueId) -> ValueId {
        let phi_id = self.resolve_alias(phi_id);
        let mut phi = self.get_phi(phi_id).cloned().unwrap();
        let mut same = None;
        for &(_, operand) in &phi.incoming {
            let op = self.resolve_alias(operand);
            if op == phi_id {
                continue;
            }
            if let Some(same) = same {
                if op == same {
                    // Another edge carrying the same unique value
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
            let ty = self.values[&phi_id].ty.clone();
            let undef = self.allocate(BuildingValue::Undef, ty);
            self.blocks
                .get_mut(&phi.block)
                .unwrap()
                .instructions
                .push(undef);
            same = Some(undef);
        }
        // Remember all users except the phi itself
        phi.users.retain(|&user| user != phi_id);
        let replacement = self.resolve_alias(same.unwrap());
        self.aliases.insert(phi_id, replacement);

        // Transfer the users to the replacement phi, so that if the
        // replacement itself becomes trivial later, these users are
        // reconsidered as well.
        if let Some(BuildingValue::Phi(replacement_phi)) =
            self.values.get_mut(&replacement).map(|data| &mut data.kind)
        {
            for user in &phi.users {
                replacement_phi.add_user(*user);
            }
        }

        // Try to recursively remove all phi users, which might have become trivial
        for &user in &phi.users {
            let user = self.resolve_alias(user);
            if matches!(self.values[&user].kind, BuildingValue::Phi(_)) {
                self.try_remove_trivial_phi(user);
            }
        }
        replacement
    }

    pub(crate) fn finish(mut self, ty: Type, entry: BlockId) -> FunctionIr {
        assert!(self.incomplete_phis.is_empty());
        assert!(self.blocks.values().all(|block| block.sealed));
        assert!(self.blocks.values().all(|block| block.terminator.is_some()));

        // Path-compress every alias so surviving operands resolve in one step.
        let aliased: Vec<ValueId> = self.aliases.keys().copied().collect();
        for id in aliased {
            self.resolve_alias(id);
        }

        let aliases = std::mem::take(&mut self.aliases);
        let remap = |id: ValueId| {
            let mut id = id;
            while let Some(&next) = aliases.get(&id) {
                id = next;
            }
            id
        };

        let building_values = std::mem::take(&mut self.values);
        let values = building_values
            .into_iter()
            .filter_map(|(id, data)| {
                if aliases.contains_key(&id) {
                    // The value was replaced by another one; the finalized IR
                    // simply does not contain it anymore.
                    return None;
                }
                Some((
                    id,
                    ValueData {
                        ty: data.ty,
                        kind: finalize_value(data.kind, &remap),
                    },
                ))
            })
            .collect();

        let building_blocks = std::mem::take(&mut self.blocks);
        let blocks = building_blocks
            .into_iter()
            .map(|(id, block)| {
                let operands = |ids: Vec<ValueId>| -> Vec<ValueId> {
                    ids.into_iter()
                        .filter(|id| !aliases.contains_key(id))
                        .map(remap)
                        .collect()
                };
                (
                    id,
                    BasicBlock {
                        phis: operands(block.phis),
                        instructions: operands(block.instructions),
                        terminator: match block.terminator.unwrap() {
                            Terminator::Return(value) => Terminator::Return(remap(value)),
                            Terminator::ReturnNoValue => Terminator::ReturnNoValue,
                            Terminator::Branch(condition, then_block, else_block) => {
                                Terminator::Branch(remap(condition), then_block, else_block)
                            }
                            Terminator::Jump(target) => Terminator::Jump(target),
                        },
                    },
                )
            })
            .collect();

        FunctionIr {
            ty,
            entry,
            values,
            blocks,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_set_phi_incoming(
        &mut self,
        phi: ValueId,
        incoming: Vec<(BlockId, ValueId)>,
    ) {
        self.get_phi(phi).unwrap().incoming = incoming;
    }

    #[cfg(test)]
    pub(crate) fn test_add_phi_user(&mut self, phi: ValueId, user: ValueId) {
        self.get_phi(phi).unwrap().add_user(user);
    }

    #[cfg(test)]
    pub(crate) fn test_incomplete_phis(
        &self,
    ) -> &std::collections::BTreeMap<BlockId, std::collections::BTreeMap<VariableId, ValueId>>
    {
        &self.incomplete_phis
    }
}

fn finalize_value(kind: BuildingValue, remap: &impl Fn(ValueId) -> ValueId) -> Value {
    match kind {
        BuildingValue::Int(value) => Value::Int(value),
        BuildingValue::String(value) => Value::String(value),
        BuildingValue::Bool(value) => Value::Bool(value),
        BuildingValue::Argument(index) => Value::Argument(index),
        BuildingValue::Call(function, args) => {
            Value::Call(function, args.into_iter().map(remap).collect())
        }
        BuildingValue::BinaryOp(op, lhs, rhs) => Value::BinaryOp(op, remap(lhs), remap(rhs)),
        BuildingValue::UnaryOp(op, operand) => Value::UnaryOp(op, remap(operand)),
        BuildingValue::Phi(phi) => Value::Phi(Phi {
            incoming: phi
                .incoming
                .into_iter()
                .map(|(block, value)| (block, remap(value)))
                .collect(),
        }),
        BuildingValue::Undef => Value::Undef,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOpCode, Terminator};
    use std::collections::BTreeSet;

    fn assert_operands_are_valid(ir: &FunctionIr) {
        crate::verify::verify(ir).unwrap();
    }

    fn assert_values_are_single_definitions(ir: &FunctionIr) {
        let mut seen = BTreeSet::new();
        for block in ir.blocks.values() {
            for &id in block.phis.iter().chain(block.instructions.iter()) {
                assert!(seen.insert(id), "value {:?} defined twice", id);
            }
        }
    }

    #[test]
    fn recursive_trivial_phis_are_removed_and_values_are_canonicalized() {
        let mut builder = IrBuilder::new();
        let entry = builder.new_block();
        let left = builder.new_block();
        let right = builder.new_block();
        let join = builder.new_block();
        builder.seal_block(entry);

        let condition = builder.emit(entry, BuildingValue::Bool(true), Type::Bool);
        let value = builder.emit(entry, BuildingValue::Int(7), Type::Int);
        builder.finish_block(entry, Terminator::Branch(condition, left, right));
        builder.seal_block(left);
        builder.seal_block(right);
        builder.finish_block(left, Terminator::Jump(join));
        builder.finish_block(right, Terminator::Jump(join));
        builder.seal_block(join);

        let first = builder.new_phi(join, Type::Int);
        builder.test_set_phi_incoming(first, [(left, value), (right, first)].into());
        let second = builder.new_phi(join, Type::Int);
        builder.test_set_phi_incoming(second, [(left, first), (right, second)].into());
        builder.test_add_phi_user(first, second);

        assert_eq!(builder.try_remove_trivial_phi(first), value);
        assert_eq!(builder.resolve_alias(second), value);
        builder.finish_block(join, Terminator::Return(second));

        let ir = builder.finish(Type::Function(Vec::new(), Box::new(Type::Int)), entry);
        assert!(
            ir.values
                .iter()
                .all(|(_, value)| !matches!(value.kind, Value::Phi(_)))
        );
        assert_eq!(ir.values.len(), 2);
        assert_operands_are_valid(&ir);
        assert_values_are_single_definitions(&ir);
    }

    #[test]
    fn users_are_transferred_when_a_trivial_phi_is_replaced_by_another_phi() {
        let mut builder = IrBuilder::new();
        let entry = builder.new_block();
        let left = builder.new_block();
        let right = builder.new_block();
        let join = builder.new_block();
        builder.seal_block(entry);

        let condition = builder.emit(entry, BuildingValue::Bool(true), Type::Bool);
        let x = builder.emit(entry, BuildingValue::Int(7), Type::Int);
        builder.finish_block(entry, Terminator::Branch(condition, left, right));
        builder.seal_block(left);
        builder.seal_block(right);
        builder.finish_block(left, Terminator::Jump(join));
        builder.finish_block(right, Terminator::Jump(join));
        builder.seal_block(join);

        // B merges the same value on both edges, but is not removed yet.
        let b = builder.new_phi(join, Type::Int);
        builder.test_set_phi_incoming(b, [(left, x), (right, x)].into());
        // A is trivially replaced by B.
        let a = builder.new_phi(join, Type::Int);
        builder.test_set_phi_incoming(a, [(left, b), (right, b)].into());
        // C uses A and X, so it is not trivial while A is still alive.
        let c = builder.new_phi(join, Type::Int);
        builder.test_set_phi_incoming(c, [(left, a), (right, x)].into());
        builder.test_add_phi_user(a, c);

        // A -> B. C must be transferred to B's users.
        assert_eq!(builder.try_remove_trivial_phi(a), b);

        // B -> X later. C has effectively become phi(X, X) and must go too.
        assert_eq!(builder.try_remove_trivial_phi(b), x);
        assert_eq!(builder.resolve_alias(c), x);

        builder.finish_block(join, Terminator::Return(c));

        let ir = builder.finish(Type::Function(Vec::new(), Box::new(Type::Int)), entry);
        assert!(
            ir.values
                .iter()
                .all(|(_, value)| !matches!(value.kind, Value::Phi(_))),
            "a trivial phi survived finalization"
        );
        assert_operands_are_valid(&ir);
        assert_values_are_single_definitions(&ir);
    }

    #[test]
    fn loop_header_incomplete_phi_becomes_canonical_ssa() {
        let mut builder = IrBuilder::new();
        let entry = builder.new_block();
        let header = builder.new_block();
        let body = builder.new_block();
        let exit = builder.new_block();
        builder.seal_block(entry);
        let variable = VariableId::new(0);

        let initial = builder.emit(entry, BuildingValue::Int(0), Type::Int);
        builder.write_variable(variable, entry, initial);
        builder.finish_block(entry, Terminator::Jump(header));

        let header_value = builder.read_variable(variable, header);
        assert!(builder.test_incomplete_phis()[&header].contains_key(&variable));
        let condition = builder.emit(header, BuildingValue::Bool(true), Type::Bool);
        builder.finish_block(header, Terminator::Branch(condition, body, exit));
        builder.seal_block(body);
        builder.seal_block(exit);

        let body_value = builder.read_variable(variable, body);
        let one = builder.emit(body, BuildingValue::Int(1), Type::Int);
        let next = builder.emit(
            body,
            BuildingValue::BinaryOp(BinaryOpCode::Add, body_value, one),
            Type::Int,
        );
        builder.write_variable(variable, body, next);
        builder.finish_block(body, Terminator::Jump(header));
        builder.seal_block(header);
        builder.finish_block(exit, Terminator::Return(header_value));

        let ir = builder.finish(Type::Function(Vec::new(), Box::new(Type::Int)), entry);
        let phis: Vec<_> = ir
            .values
            .values()
            .filter_map(|value| match &value.kind {
                Value::Phi(phi) => Some(phi),
                _ => None,
            })
            .collect();
        assert_eq!(phis.len(), 1);
        assert_eq!(phis[0].incoming.len(), 2);
        assert_operands_are_valid(&ir);
        assert_values_are_single_definitions(&ir);

        // Phis are structural: they live in block.phis, never in instructions.
        let header_block = &ir.blocks[&header];
        assert_eq!(header_block.phis.len(), 1);
        assert!(
            header_block
                .instructions
                .iter()
                .all(|&id| !matches!(ir.values[&id].kind, Value::Phi(_)))
        );
    }
}
