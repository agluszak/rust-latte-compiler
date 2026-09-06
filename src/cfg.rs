use crate::ir::{BlockId, FunctionIr};
use std::collections::{BTreeMap, BTreeSet};

/// Control-flow analysis shared by optimization, verification, and codegen.
///
/// `predecessors` contains one entry per predecessor block: a branch with
/// identical targets contributes a single predecessor relationship, matching
/// the builder's deduplication. `reverse_postorder` covers reachable blocks
/// only; [`Cfg::layout_order`] appends unreachable blocks in ID order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Cfg {
    pub(crate) predecessors: BTreeMap<BlockId, Vec<BlockId>>,
    pub(crate) reverse_postorder: Vec<BlockId>,
}

impl Cfg {
    pub(crate) fn compute(ir: &FunctionIr) -> Self {
        let predecessors = predecessors(ir);
        let reverse_postorder = reachable_reverse_postorder(ir);
        Self {
            predecessors,
            reverse_postorder,
        }
    }

    /// Generation order for backends: reachable blocks in reverse postorder,
    /// followed by unreachable blocks in ID order.
    pub(crate) fn layout_order(&self, ir: &FunctionIr) -> Vec<BlockId> {
        let reachable: BTreeSet<BlockId> =
            self.reverse_postorder.iter().copied().collect();
        let mut order = self.reverse_postorder.clone();
        for &block in ir.blocks.keys() {
            if !reachable.contains(&block) {
                order.push(block);
            }
        }
        order
    }
}

pub(crate) fn predecessors(ir: &FunctionIr) -> BTreeMap<BlockId, Vec<BlockId>> {
    let mut predecessors: BTreeMap<_, Vec<_>> =
        ir.blocks.keys().map(|&block| (block, Vec::new())).collect();
    for (&block, data) in &ir.blocks {
        for successor in data.terminator.successors() {
            let preds = predecessors
                .get_mut(&successor)
                .unwrap_or_else(|| panic!("successor {successor} is not in the function"));
            if !preds.contains(&block) {
                preds.push(block);
            }
        }
    }
    for preds in predecessors.values_mut() {
        preds.sort();
    }
    predecessors
}

pub(crate) fn reachable_reverse_postorder(ir: &FunctionIr) -> Vec<BlockId> {
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

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Dominators {
    pub(crate) immediate: BTreeMap<BlockId, BlockId>,
    pub(crate) children: BTreeMap<BlockId, Vec<BlockId>>,
}

impl Dominators {
    pub(crate) fn compute(ir: &FunctionIr) -> Self {
        let cfg = Cfg::compute(ir);
        Self::compute_from_cfg(ir, &cfg)
    }

    fn compute_from_cfg(ir: &FunctionIr, cfg: &Cfg) -> Self {
        let predecessors = &cfg.predecessors;
        let reverse_postorder = &cfg.reverse_postorder;
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

        let mut immediate: BTreeMap<BlockId, BlockId> = BTreeMap::new();
        let mut children: BTreeMap<_, Vec<_>> = reverse_postorder
            .iter()
            .map(|&block| (block, Vec::new()))
            .collect();
        for &block in reverse_postorder.iter().skip(1) {
            let idom = sets[&block]
                .iter()
                .copied()
                .filter(|&dominator| dominator != block)
                .max_by_key(|dominator| sets[dominator].len())
                .unwrap();
            immediate.insert(block, idom);
            children.get_mut(&idom).unwrap().push(block);
        }
        for dominated in children.values_mut() {
            dominated.sort();
        }

        Self {
            immediate,
            children,
        }
    }
}
