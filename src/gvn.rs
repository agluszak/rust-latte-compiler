use crate::ir::{BinaryOpCode, BlockId, FunctionIr, UnaryOpCode, Value, ValueId};
use crate::typechecker::Type;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq, Eq)]
struct Dominators {
    children: BTreeMap<BlockId, Vec<BlockId>>,
}

fn predecessors(ir: &FunctionIr) -> BTreeMap<BlockId, Vec<BlockId>> {
    let mut predecessors: BTreeMap<_, Vec<_>> =
        ir.blocks.keys().map(|&block| (block, Vec::new())).collect();
    for (&block, data) in &ir.blocks {
        for successor in data.terminator.successors() {
            predecessors
                .get_mut(&successor)
                .unwrap_or_else(|| panic!("successor {successor} is not in the function"))
                .push(block);
        }
    }
    predecessors
}

fn reachable_reverse_postorder(ir: &FunctionIr) -> Vec<BlockId> {
    fn visit(
        ir: &FunctionIr,
        block: BlockId,
        visited: &mut BTreeSet<BlockId>,
        postorder: &mut Vec<BlockId>,
    ) {
        if !visited.insert(block) {
            return;
        }
        let data = ir
            .blocks
            .get(&block)
            .unwrap_or_else(|| panic!("block {block} is not in the function"));
        for successor in data.terminator.successors() {
            visit(ir, successor, visited, postorder);
        }
        postorder.push(block);
    }

    let mut visited = BTreeSet::new();
    let mut postorder = Vec::new();
    visit(ir, ir.entry, &mut visited, &mut postorder);
    postorder.reverse();
    postorder
}

impl Dominators {
    fn compute(ir: &FunctionIr) -> Self {
        let predecessors = predecessors(ir);
        let reverse_postorder = reachable_reverse_postorder(ir);
        let reachable: BTreeSet<_> = reverse_postorder.iter().copied().collect();

        let mut sets: BTreeMap<BlockId, BTreeSet<BlockId>> = reverse_postorder
            .iter()
            .map(|&block| {
                let initial = if block == ir.entry {
                    BTreeSet::from([block])
                } else {
                    reachable.clone()
                };
                (block, initial)
            })
            .collect();

        loop {
            let mut changed = false;
            for &block in reverse_postorder.iter().skip(1) {
                let mut reachable_predecessors = predecessors[&block]
                    .iter()
                    .copied()
                    .filter(|predecessor| reachable.contains(predecessor));
                let first = reachable_predecessors.next().unwrap_or_else(|| {
                    panic!("reachable non-entry block {block} has no predecessor")
                });
                let mut next = sets[&first].clone();
                for predecessor in reachable_predecessors {
                    next = next.intersection(&sets[&predecessor]).copied().collect();
                }
                next.insert(block);
                if next != sets[&block] {
                    sets.insert(block, next);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut children: BTreeMap<_, Vec<_>> = reverse_postorder
            .iter()
            .map(|&block| (block, Vec::new()))
            .collect();
        for &block in reverse_postorder.iter().skip(1) {
            let immediate = sets[&block]
                .iter()
                .copied()
                .filter(|&dominator| dominator != block)
                .max_by_key(|dominator| sets[dominator].len())
                .unwrap();
            children.get_mut(&immediate).unwrap().push(block);
        }
        for dominated in children.values_mut() {
            dominated.sort();
        }

        Self { children }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ValueNumber(ValueId);

#[derive(Debug, PartialEq, Eq)]
struct ValueNumbers {
    values: BTreeMap<ValueId, ValueNumber>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum InitialClass {
    Int(i32),
    Bool(bool),
    String(String),
    Unary(UnaryOpCode, Type),
    Binary(BinaryOpCode, Type),
    Phi(BlockId, Type),
    Opaque(ValueId),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Signature {
    Leaf,
    Unary {
        operand: ValueNumber,
    },
    Binary {
        lhs: ValueNumber,
        rhs: ValueNumber,
    },
    Phi {
        incoming: Vec<(BlockId, ValueNumber)>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RefinementKey {
    previous: ValueNumber,
    signature: Signature,
}

impl ValueNumbers {
    fn compute(ir: &FunctionIr) -> Self {
        Self::compute_with_iterations(ir).0
    }

    fn compute_with_iterations(ir: &FunctionIr) -> (Self, usize) {
        let phi_blocks = Self::phi_blocks(ir);
        let mut initial_groups: BTreeMap<InitialClass, Vec<ValueId>> = BTreeMap::new();
        for (&id, data) in &ir.values {
            let class = match &data.kind {
                Value::Int(value) => InitialClass::Int(*value),
                Value::Bool(value) => InitialClass::Bool(*value),
                Value::String(value) => InitialClass::String(value.clone()),
                Value::UnaryOp(op, _) => InitialClass::Unary(*op, data.ty.clone()),
                Value::BinaryOp(op, _, _) => InitialClass::Binary(*op, data.ty.clone()),
                Value::Phi(_) => InitialClass::Phi(phi_blocks[&id], data.ty.clone()),
                Value::Argument(_) | Value::Call(_, _) | Value::Undef => InitialClass::Opaque(id),
            };
            initial_groups.entry(class).or_default().push(id);
        }

        let mut current = Self::number_groups(initial_groups.into_values());
        let mut iterations = 0;
        loop {
            iterations += 1;
            let mut groups: BTreeMap<RefinementKey, Vec<ValueId>> = BTreeMap::new();
            for (&id, data) in &ir.values {
                let signature = match &data.kind {
                    Value::Int(_)
                    | Value::Bool(_)
                    | Value::String(_)
                    | Value::Argument(_)
                    | Value::Call(_, _)
                    | Value::Undef => Signature::Leaf,
                    Value::UnaryOp(_, operand) => Signature::Unary {
                        operand: current[operand],
                    },
                    Value::BinaryOp(op, lhs, rhs) => {
                        let mut lhs = current[lhs];
                        let mut rhs = current[rhs];
                        let commutative = matches!(op, BinaryOpCode::Eq | BinaryOpCode::Neq)
                            || (data.ty == Type::Int
                                && matches!(op, BinaryOpCode::Add | BinaryOpCode::Mul));
                        if commutative && rhs < lhs {
                            std::mem::swap(&mut lhs, &mut rhs);
                        }
                        Signature::Binary { lhs, rhs }
                    }
                    Value::Phi(phi) => {
                        let mut incoming: Vec<_> = phi
                            .incoming
                            .iter()
                            .map(|(block, value)| (*block, current[value]))
                            .collect();
                        incoming.sort();
                        Signature::Phi { incoming }
                    }
                };
                groups
                    .entry(RefinementKey {
                        previous: current[&id],
                        signature,
                    })
                    .or_default()
                    .push(id);
            }

            let next = Self::number_groups(groups.into_values());
            if next == current {
                return (Self { values: next }, iterations);
            }
            current = next;
        }
    }

    fn number_groups(
        groups: impl IntoIterator<Item = Vec<ValueId>>,
    ) -> BTreeMap<ValueId, ValueNumber> {
        let mut numbers = BTreeMap::new();
        for group in groups {
            let number = ValueNumber(*group.iter().min().unwrap());
            for id in group {
                numbers.insert(id, number);
            }
        }
        numbers
    }

    fn phi_blocks(ir: &FunctionIr) -> BTreeMap<ValueId, BlockId> {
        let mut blocks = BTreeMap::new();
        for (&block, data) in &ir.blocks {
            for &phi in &data.phis {
                assert!(
                    matches!(
                        ir.values.get(&phi).map(|data| &data.kind),
                        Some(Value::Phi(_))
                    ),
                    "block {block} contains non-phi {phi} in its phi list"
                );
                assert!(
                    blocks.insert(phi, block).is_none(),
                    "phi {phi} appears in multiple blocks"
                );
            }
        }
        let value_phis: BTreeSet<_> = ir
            .values
            .iter()
            .filter_map(|(&id, data)| matches!(data.kind, Value::Phi(_)).then_some(id))
            .collect();
        assert_eq!(blocks.keys().copied().collect::<BTreeSet<_>>(), value_phis);
        blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BasicBlock, Phi, Terminator, ValueData};
    use crate::typechecker::Type;
    use crate::typed_ast::VariableId;

    fn block(terminator: Terminator) -> BasicBlock {
        BasicBlock {
            phis: Vec::new(),
            instructions: Vec::new(),
            terminator,
        }
    }

    fn data(ty: Type, kind: Value) -> ValueData {
        ValueData { ty, kind }
    }

    fn single_block(
        values: BTreeMap<ValueId, ValueData>,
        instructions: Vec<ValueId>,
    ) -> FunctionIr {
        let entry = BlockId(10);
        FunctionIr {
            ty: Type::Void,
            entry,
            values,
            blocks: BTreeMap::from([(
                entry,
                BasicBlock {
                    phis: Vec::new(),
                    instructions,
                    terminator: Terminator::ReturnNoValue,
                },
            )]),
        }
    }

    #[test]
    fn computes_dominator_tree_for_diamond() {
        let entry = BlockId(7);
        let left = BlockId(9);
        let right = BlockId(11);
        let join = BlockId(13);
        let ir = FunctionIr {
            ty: Type::Void,
            entry,
            values: BTreeMap::new(),
            blocks: BTreeMap::from([
                (
                    entry,
                    block(Terminator::Branch(crate::ir::ValueId(1), left, right)),
                ),
                (left, block(Terminator::Jump(join))),
                (right, block(Terminator::Jump(join))),
                (join, block(Terminator::ReturnNoValue)),
            ]),
        };

        let dominators = Dominators::compute(&ir);
        assert_eq!(dominators.children[&entry], vec![left, right, join]);
        assert!(dominators.children[&left].is_empty());
        assert!(dominators.children[&right].is_empty());
        assert!(dominators.children[&join].is_empty());
    }

    #[test]
    fn generated_blocks_are_reachable_from_explicit_entry() {
        let source = r#"int main() { int x = 0; while (x < 3) { x++; } return x; }"#;
        let parsed = crate::parser::latte::ProgramParser::new()
            .parse(crate::lexer::Lexer::new(source))
            .unwrap();
        let (program, _) = crate::typechecker::typecheck_program(parsed).unwrap();
        let mut translated = crate::ir::Ir::new();
        translated.translate_function(program.0.into_iter().next().unwrap().value);
        let ir = translated.functions.get("main").unwrap();

        let reachable: BTreeSet<_> = reachable_reverse_postorder(ir).into_iter().collect();
        assert_eq!(reachable, ir.blocks.keys().copied().collect());
    }

    #[test]
    fn numbers_equal_literals_and_expressions_without_mutating_ir() {
        let a = ValueId(3);
        let b = ValueId(7);
        let one_a = ValueId(12);
        let one_b = ValueId(18);
        let sum_a = ValueId(25);
        let sum_b = ValueId(31);
        let ir = single_block(
            BTreeMap::from([
                (a, data(Type::Int, Value::Argument(0))),
                (b, data(Type::Int, Value::Argument(1))),
                (one_a, data(Type::Int, Value::Int(1))),
                (one_b, data(Type::Int, Value::Int(1))),
                (
                    sum_a,
                    data(Type::Int, Value::BinaryOp(BinaryOpCode::Add, a, one_a)),
                ),
                (
                    sum_b,
                    data(Type::Int, Value::BinaryOp(BinaryOpCode::Add, a, one_b)),
                ),
            ]),
            vec![a, b, one_a, one_b, sum_a, sum_b],
        );
        let before = ir.clone();

        let numbers = ValueNumbers::compute(&ir);

        assert_eq!(numbers.values[&one_a], numbers.values[&one_b]);
        assert_eq!(numbers.values[&sum_a], numbers.values[&sum_b]);
        assert_eq!(ir, before);
    }

    #[test]
    fn canonicalizes_only_genuinely_commutative_binary_operations() {
        let a = ValueId(1);
        let b = ValueId(2);
        let int_ab = ValueId(3);
        let int_ba = ValueId(4);
        let string_a = ValueId(5);
        let string_b = ValueId(6);
        let string_ab = ValueId(7);
        let string_ba = ValueId(8);
        let ir = single_block(
            BTreeMap::from([
                (a, data(Type::Int, Value::Argument(0))),
                (b, data(Type::Int, Value::Argument(1))),
                (
                    int_ab,
                    data(Type::Int, Value::BinaryOp(BinaryOpCode::Add, a, b)),
                ),
                (
                    int_ba,
                    data(Type::Int, Value::BinaryOp(BinaryOpCode::Add, b, a)),
                ),
                (string_a, data(Type::LatteString, Value::Argument(2))),
                (string_b, data(Type::LatteString, Value::Argument(3))),
                (
                    string_ab,
                    data(
                        Type::LatteString,
                        Value::BinaryOp(BinaryOpCode::Add, string_a, string_b),
                    ),
                ),
                (
                    string_ba,
                    data(
                        Type::LatteString,
                        Value::BinaryOp(BinaryOpCode::Add, string_b, string_a),
                    ),
                ),
            ]),
            vec![
                a, b, int_ab, int_ba, string_a, string_b, string_ab, string_ba,
            ],
        );

        let numbers = ValueNumbers::compute(&ir);
        assert_eq!(numbers.values[&int_ab], numbers.values[&int_ba]);
        assert_ne!(numbers.values[&string_ab], numbers.values[&string_ba]);
    }

    #[test]
    fn calls_and_undefs_remain_opaque() {
        let call_a = ValueId(4);
        let call_b = ValueId(9);
        let undef_a = ValueId(14);
        let undef_b = ValueId(20);
        let ir = single_block(
            BTreeMap::from([
                (
                    call_a,
                    data(Type::Int, Value::Call(VariableId::new(0), Vec::new())),
                ),
                (
                    call_b,
                    data(Type::Int, Value::Call(VariableId::new(0), Vec::new())),
                ),
                (undef_a, data(Type::Int, Value::Undef)),
                (undef_b, data(Type::Int, Value::Undef)),
            ]),
            vec![call_a, call_b, undef_a, undef_b],
        );

        let numbers = ValueNumbers::compute(&ir);
        assert_ne!(numbers.values[&call_a], numbers.values[&call_b]);
        assert_ne!(numbers.values[&undef_a], numbers.values[&undef_b]);
    }

    fn induction_variables(second_initial: i32) -> (FunctionIr, [ValueId; 4]) {
        let entry = BlockId(1);
        let header = BlockId(2);
        let back = BlockId(3);
        let exit = BlockId(4);
        let zero_x = ValueId(10);
        let zero_y = ValueId(11);
        let one_x = ValueId(12);
        let one_y = ValueId(13);
        let condition = ValueId(14);
        let phi_x = ValueId(20);
        let phi_y = ValueId(21);
        let add_x = ValueId(30);
        let add_y = ValueId(31);
        let values = BTreeMap::from([
            (zero_x, data(Type::Int, Value::Int(0))),
            (zero_y, data(Type::Int, Value::Int(second_initial))),
            (one_x, data(Type::Int, Value::Int(1))),
            (one_y, data(Type::Int, Value::Int(1))),
            (condition, data(Type::Bool, Value::Argument(0))),
            (
                phi_x,
                data(
                    Type::Int,
                    Value::Phi(Phi {
                        incoming: vec![(entry, zero_x), (back, add_x)],
                    }),
                ),
            ),
            (
                phi_y,
                data(
                    Type::Int,
                    Value::Phi(Phi {
                        incoming: vec![(entry, zero_y), (back, add_y)],
                    }),
                ),
            ),
            (
                add_x,
                data(Type::Int, Value::BinaryOp(BinaryOpCode::Add, phi_x, one_x)),
            ),
            (
                add_y,
                data(Type::Int, Value::BinaryOp(BinaryOpCode::Add, phi_y, one_y)),
            ),
        ]);
        let blocks = BTreeMap::from([
            (
                entry,
                BasicBlock {
                    phis: vec![],
                    instructions: vec![zero_x, zero_y, one_x, one_y, condition],
                    terminator: Terminator::Jump(header),
                },
            ),
            (
                header,
                BasicBlock {
                    phis: vec![phi_x, phi_y],
                    instructions: vec![add_x, add_y],
                    terminator: Terminator::Branch(condition, back, exit),
                },
            ),
            (back, block(Terminator::Jump(header))),
            (exit, block(Terminator::Return(phi_x))),
        ]);
        (
            FunctionIr {
                ty: Type::Int,
                entry,
                values,
                blocks,
            },
            [phi_x, phi_y, add_x, add_y],
        )
    }

    #[test]
    fn optimistic_refinement_finds_parallel_cyclic_induction_variables() {
        let (ir, [phi_x, phi_y, add_x, add_y]) = induction_variables(0);
        let numbers = ValueNumbers::compute(&ir);

        assert_eq!(numbers.values[&phi_x], numbers.values[&phi_y]);
        assert_eq!(numbers.values[&add_x], numbers.values[&add_y]);
    }

    #[test]
    fn cyclic_classes_split_when_an_initial_value_differs() {
        let (ir, [phi_x, phi_y, add_x, add_y]) = induction_variables(7);
        let (numbers, iterations) = ValueNumbers::compute_with_iterations(&ir);

        assert_ne!(numbers.values[&phi_x], numbers.values[&phi_y]);
        assert_ne!(numbers.values[&add_x], numbers.values[&add_y]);
        assert!(
            iterations > 1,
            "a split must propagate around the cyclic fixed point"
        );
    }

    #[test]
    fn phis_in_different_blocks_start_in_different_classes() {
        let left = BlockId(1);
        let right = BlockId(2);
        let source = ValueId(1);
        let phi_left = ValueId(2);
        let phi_right = ValueId(3);
        let ir = FunctionIr {
            ty: Type::Int,
            entry: left,
            values: BTreeMap::from([
                (source, data(Type::Int, Value::Int(1))),
                (
                    phi_left,
                    data(
                        Type::Int,
                        Value::Phi(Phi {
                            incoming: vec![(left, source)],
                        }),
                    ),
                ),
                (
                    phi_right,
                    data(
                        Type::Int,
                        Value::Phi(Phi {
                            incoming: vec![(left, source)],
                        }),
                    ),
                ),
            ]),
            blocks: BTreeMap::from([
                (
                    left,
                    BasicBlock {
                        phis: vec![phi_left],
                        instructions: vec![source],
                        terminator: Terminator::Jump(right),
                    },
                ),
                (
                    right,
                    BasicBlock {
                        phis: vec![phi_right],
                        instructions: vec![],
                        terminator: Terminator::Return(phi_right),
                    },
                ),
            ]),
        };

        let numbers = ValueNumbers::compute(&ir);
        assert_ne!(numbers.values[&phi_left], numbers.values[&phi_right]);
    }
}
