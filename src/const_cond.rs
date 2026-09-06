use crate::ast::{BinaryOp, Literal, UnaryOp};
use crate::typed_ast::TypedExpr;
use crate::typed_ast::TypedExprKind;

/// Syntactically constant boolean condition shared by return analysis and lowering.
///
/// Limited scope: boolean literals combined with `!`, `==`, `!=`, `&&`, `||`.
/// This is intentionally not general constant folding.
pub(crate) fn const_bool(expr: &TypedExpr) -> Option<bool> {
    match &expr.expr {
        TypedExprKind::Literal(Literal::Bool(value)) => Some(*value),
        TypedExprKind::Unary { op, expr } if op.value == UnaryOp::Not => {
            const_bool(&expr.value).map(|value| !value)
        }
        TypedExprKind::Binary { lhs, op, rhs } => {
            let lhs = const_bool(&lhs.value)?;
            let rhs = const_bool(&rhs.value)?;
            match op.value {
                BinaryOp::Eq => Some(lhs == rhs),
                BinaryOp::Neq => Some(lhs != rhs),
                BinaryOp::And => Some(lhs && rhs),
                BinaryOp::Or => Some(lhs || rhs),
                _ => None,
            }
        }
        _ => None,
    }
}
