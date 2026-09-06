//! Operation semantics and backend contract.
//!
//! These distinctions are not interchangeable:
//! - `can_compute_equivalent`: the operation is deterministic for its inputs.
//! - `can_reuse_dominating_result`: a dominating congruent definition may
//!   replace this one (GVN). Calls are excluded: they may observe or affect
//!   state beyond their inputs.
//! - `can_remove_when_unused`: GVN never performs DCE; this query exists so
//!   future passes do not conflate reuse with deletion (e.g. division traps).
//! - `can_hoist`: reusing a dominating result does not authorize moving the
//!   operation. No current pass hoists.
//!
//! Backend contract (existing language):
//! - 32-bit two's-complement ints; add/sub/mul wrap. Signed division and
//!   remainder trap on zero divisors (and `INT_MIN / -1`); no explicit
//!   Latte-level check is emitted.
//! - Booleans are normalized LLVM `i1` (0/1). Comparisons produce `i1`;
//!   `!` is bitwise not on `i1`.
//! - String equality is byte equality via the runtime (`len` + `memcmp`).
//!   String concatenation allocates via the runtime; reusing a dominating
//!   string result is allowed, merging allocations is not assumed.
//! - `Undef` appears only for entry/unreachable trivial phis. It may be any
//!   value of its type and must never be observed on a reachable path.

use crate::ir::Value;

pub(crate) fn can_reuse_dominating_result(value: &Value) -> bool {
    match value {
        Value::Int(_)
        | Value::Bool(_)
        | Value::String(_)
        | Value::UnaryOp(_, _)
        | Value::BinaryOp(_, _, _)
        | Value::Phi(_) => true,
        Value::Argument(_) | Value::Call(_, _) | Value::Undef => false,
    }
}

pub(crate) fn can_remove_when_unused(_value: &Value) -> bool {
    // No current pass deletes unused definitions based on this query.
    // Division, remainder, and calls must not be treated as freely deletable
    // without settling exceptional arithmetic and observable effects first.
    false
}

pub(crate) fn can_hoist(_value: &Value) -> bool {
    false
}
