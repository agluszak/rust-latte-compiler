use crate::ir::{BlockId, FunctionIr};
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BasicBlock, Terminator};
    use crate::typechecker::Type;

    fn block(terminator: Terminator) -> BasicBlock {
        BasicBlock {
            phis: Vec::new(),
            instructions: Vec::new(),
            terminator,
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
}
