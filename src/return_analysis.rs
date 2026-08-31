use crate::ast::{BinaryOp, Literal, UnaryOp};
use crate::typechecker::{Type, TypecheckingError};
use crate::typed_ast::{TypedBlock, TypedDecl, TypedExpr, TypedExprKind, TypedStmt};

fn const_bool(expr: &TypedExpr) -> Option<bool> {
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

fn block_always_returns(block: &TypedBlock) -> bool {
    block
        .0
        .iter()
        .any(|stmt| statement_always_returns(&stmt.value))
}

fn statement_always_returns(stmt: &TypedStmt) -> bool {
    match stmt {
        TypedStmt::Return(_) => true,
        TypedStmt::Block(block) => block_always_returns(&block.value),
        TypedStmt::If {
            cond,
            then,
            otherwise,
        } => match const_bool(&cond.value) {
            Some(true) => statement_always_returns(&then.value),
            Some(false) => otherwise
                .as_ref()
                .is_some_and(|otherwise| statement_always_returns(&otherwise.value)),
            None => {
                statement_always_returns(&then.value)
                    && otherwise
                        .as_ref()
                        .is_some_and(|otherwise| statement_always_returns(&otherwise.value))
            }
        },
        TypedStmt::While { cond, .. } => const_bool(&cond.value) == Some(true),
        TypedStmt::Empty
        | TypedStmt::Decl(_)
        | TypedStmt::Assignment { .. }
        | TypedStmt::Expr(_)
        | TypedStmt::Incr(_)
        | TypedStmt::Decr(_) => false,
    }
}

pub fn check_function_returns(decl: &TypedDecl) -> Result<(), TypecheckingError> {
    if let TypedDecl::Fn {
        return_type, body, ..
    } = decl
        && return_type != &Type::Void
        && !block_always_returns(&body.value)
    {
        return Err(TypecheckingError::missing_return(body.span.clone()));
    }
    Ok(())
}
