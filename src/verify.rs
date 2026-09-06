use crate::cfg::{Cfg, Dominators};
use crate::ir::{BinaryOpCode, BlockId, FunctionIr, UnaryOpCode, Value, ValueId};
use crate::types::Type;
use std::collections::{BTreeMap, BTreeSet};

/// Consolidated well-formedness checks for finalized SSA functions.
pub(crate) fn verify(ir: &FunctionIr) -> Result<(), String> {
    if !ir.blocks.contains_key(&ir.entry) {
        return Err(format!("entry {} is not in the function", ir.entry));
    }
    // Test fragments use a bare return type; accept it as the return type.
    let (arg_types, ret) = match &ir.ty {
        Type::Function(args, ret) => (args.clone(), (**ret).clone()),
        other => (Vec::new(), other.clone()),
    };

    let cfg = Cfg::compute(ir);

    // Definition ownership and placement.
    let mut def_block: BTreeMap<ValueId, BlockId> = BTreeMap::new();
    for (&block, data) in &ir.blocks {
        for &phi in &data.phis {
            let Some(value) = ir.values.get(&phi) else {
                return Err(format!("phi {phi} in {block} has no value"));
            };
            if !matches!(value.kind, Value::Phi(_)) {
                return Err(format!("block.phis of {block} contains non-phi {phi}"));
            }
            if def_block.insert(phi, block).is_some() {
                return Err(format!("value {phi} defined twice"));
            }
        }
        for &instr in &data.instructions {
            let Some(value) = ir.values.get(&instr) else {
                return Err(format!("instruction {instr} in {block} has no value"));
            };
            if matches!(value.kind, Value::Phi(_)) {
                return Err(format!("phi {instr} must live in block.phis, not instructions"));
            }
            if def_block.insert(instr, block).is_some() {
                return Err(format!("value {instr} defined twice"));
            }
        }
    }
    if def_block.len() != ir.values.len() {
        let orphan: Vec<_> = ir
            .values
            .keys()
            .filter(|id| !def_block.contains_key(id))
            .collect();
        return Err(format!("values without a defining block: {orphan:?}"));
    }

    // Phi ownership: every phi value appears in exactly one block's phi list.
    {
        let mut phi_blocks: BTreeMap<ValueId, BlockId> = BTreeMap::new();
        for (&block, data) in &ir.blocks {
            for &phi in &data.phis {
                if phi_blocks.insert(phi, block).is_some() {
                    return Err(format!("phi {phi} appears in multiple blocks"));
                }
            }
        }
        for (&id, data) in &ir.values {
            if matches!(data.kind, Value::Phi(_)) && !phi_blocks.contains_key(&id) {
                return Err(format!("phi {id} is not owned by any block"));
            }
        }
    }

    // Operand existence.
    for (&id, data) in &ir.values {
        for operand in data.kind.operands() {
            if !ir.values.contains_key(&operand) {
                return Err(format!("value {id} uses missing value {operand}"));
            }
        }
    }
    for (&block, data) in &ir.blocks {
        for operand in data.terminator.operands() {
            if !ir.values.contains_key(&operand) {
                return Err(format!("terminator of {block} uses missing value {operand}"));
            }
        }
    }

    // Successor validity is already enforced by Cfg::compute (panics on
    // missing successors). Check phi/predecessor correspondence exactly.
    for (&block, data) in &ir.blocks {
        let preds: BTreeSet<BlockId> = cfg.predecessors[&block].iter().copied().collect();
        for &phi in &data.phis {
            let Value::Phi(phi_data) = &ir.values[&phi].kind else {
                unreachable!()
            };
            let incoming_blocks: Vec<BlockId> =
                phi_data.incoming.iter().map(|(b, _)| *b).collect();
            let incoming_set: BTreeSet<BlockId> = incoming_blocks.iter().copied().collect();
            if incoming_set != preds {
                return Err(format!(
                    "phi {phi} incoming {incoming_set:?} does not match CFG predecessors {preds:?} of {block}"
                ));
            }
            if incoming_blocks.len() != incoming_set.len() {
                return Err(format!("phi {phi} has duplicate incoming edges"));
            }
        }
    }

    // Dominance: ordinary uses dominate their block; phi operands dominate
    // the predecessor they arrive on. Intra-block order is checked for
    // non-phi instructions and terminators.
    let dominators = Dominators::compute_from_cfg(ir, &cfg);
    let reachable: BTreeSet<BlockId> = cfg.reverse_postorder.iter().copied().collect();
    let block_order_index: BTreeMap<BlockId, BTreeMap<ValueId, usize>> = ir
        .blocks
        .iter()
        .map(|(&block, data)| {
            let mut index = BTreeMap::new();
            for (i, &id) in data.phis.iter().chain(&data.instructions).enumerate() {
                index.insert(id, i);
            }
            (block, index)
        })
        .collect();
    let is_phi = |id: ValueId| matches!(ir.values[&id].kind, Value::Phi(_));
    for (&block, data) in &ir.blocks {
        if !reachable.contains(&block) {
            // Unreachable block: still require operands to exist (checked
            // above) but skip dominance, which is only defined for
            // reachable blocks.
            continue;
        }
        for &phi in &data.phis {
            let Value::Phi(phi_data) = &ir.values[&phi].kind else {
                unreachable!()
            };
            for (pred, operand) in &phi_data.incoming {
                let def = def_block[operand];
                if !dominators.dominates(def, *pred) {
                    return Err(format!(
                        "phi {phi} operand {operand} (defined in {def}) does not dominate predecessor {pred}"
                    ));
                }
            }
        }
        for &instr in &data.instructions {
            for operand in ir.values[&instr].kind.operands().collect::<Vec<_>>() {
                let def = def_block[&operand];
                if def == block {
                    let use_idx = block_order_index[&block][&instr];
                    let def_idx = block_order_index[&block][&operand];
                    if is_phi(operand) || def_idx < use_idx {
                        continue;
                    }
                    return Err(format!(
                        "value {instr} in {block} uses {operand} defined later in the same block"
                    ));
                } else if !dominators.dominates(def, block) {
                    return Err(format!(
                        "value {instr} in {block} uses {operand} defined in non-dominating {def}"
                    ));
                }
            }
        }
        for operand in data.terminator.operands().collect::<Vec<_>>() {
            let def = def_block[&operand];
            if def != block && !dominators.dominates(def, block) {
                return Err(format!(
                    "terminator of {block} uses {operand} defined in non-dominating {def}"
                ));
            }
        }
    }

    // Basic type and return consistency.
    for (&id, data) in &ir.values {
        match &data.kind {
            Value::Int(_) if data.ty == Type::Int => {}
            Value::Bool(_) if data.ty == Type::Bool => {}
            Value::String(_) if data.ty == Type::LatteString => {}
            Value::Int(_) | Value::Bool(_) | Value::String(_) => {
                return Err(format!("literal {id} has wrong type {}", data.ty));
            }
            Value::Argument(i) => {
                if !arg_types.is_empty() {
                    let idx = *i as usize;
                    if idx >= arg_types.len() {
                        return Err(format!("argument {id} index {i} out of range"));
                    }
                    if data.ty != arg_types[idx] {
                        return Err(format!("argument {id} type mismatch"));
                    }
                }
            }
            Value::BinaryOp(op, lhs, rhs) => {
                let lhs_ty = &ir.values[lhs].ty;
                let rhs_ty = &ir.values[rhs].ty;
                match op {
                    BinaryOpCode::Add => {
                        if lhs_ty == &Type::Int && rhs_ty == &Type::Int {
                            if data.ty != Type::Int {
                                return Err(format!("int add {id} must produce int"));
                            }
                        } else if lhs_ty == &Type::LatteString
                            && rhs_ty == &Type::LatteString
                        {
                            if data.ty != Type::LatteString {
                                return Err(format!("string add {id} must produce string"));
                            }
                        } else {
                            return Err(format!("add {id} has invalid operand types"));
                        }
                    }
                    BinaryOpCode::Sub
                    | BinaryOpCode::Mul
                    | BinaryOpCode::Div
                    | BinaryOpCode::Mod => {
                        if lhs_ty != &Type::Int || rhs_ty != &Type::Int || data.ty != Type::Int {
                            return Err(format!("arithmetic {id} must be int"));
                        }
                    }
                    BinaryOpCode::Gt
                    | BinaryOpCode::Lt
                    | BinaryOpCode::Gte
                    | BinaryOpCode::Lte => {
                        if lhs_ty != &Type::Int || rhs_ty != &Type::Int || data.ty != Type::Bool
                        {
                            return Err(format!("comparison {id} must compare ints to bool"));
                        }
                    }
                    BinaryOpCode::Eq | BinaryOpCode::Neq => {
                        if lhs_ty != rhs_ty || data.ty != Type::Bool {
                            return Err(format!("equality {id} needs matching operands to bool"));
                        }
                        if !matches!(
                            lhs_ty,
                            Type::Int | Type::Bool | Type::LatteString
                        ) {
                            return Err(format!("equality {id} has unsupported type"));
                        }
                    }
                }
            }
            Value::UnaryOp(op, operand) => {
                let operand_ty = &ir.values[operand].ty;
                match op {
                    UnaryOpCode::Neg => {
                        if operand_ty != &Type::Int || data.ty != Type::Int {
                            return Err(format!("neg {id} must be int"));
                        }
                    }
                    UnaryOpCode::Not => {
                        if operand_ty != &Type::Bool || data.ty != Type::Bool {
                            return Err(format!("not {id} must be bool"));
                        }
                    }
                }
            }
            Value::Phi(phi) => {
                for (_, operand) in &phi.incoming {
                    if ir.values[operand].ty != data.ty {
                        return Err(format!("phi {id} incoming type mismatch"));
                    }
                }
            }
            Value::Call(_, _) | Value::Undef => {}
        }
    }
    for (&block, data) in &ir.blocks {
        match &data.terminator {
            crate::ir::Terminator::Return(value) => {
                if ret == Type::Void {
                    return Err(format!("void function returns a value in {block}"));
                }
                if ir.values[value].ty != ret {
                    return Err(format!("return type mismatch in {block}"));
                }
            }
            crate::ir::Terminator::ReturnNoValue => {
                if ret != Type::Void {
                    return Err(format!("non-void function has value-less return in {block}"));
                }
            }
            crate::ir::Terminator::Branch(cond, _, _) => {
                if ir.values[cond].ty != Type::Bool {
                    return Err(format!("branch condition in {block} must be bool"));
                }
            }
            crate::ir::Terminator::Jump(_) => {}
        }
    }

    Ok(())
}
