use crate::const_cond::const_bool;
use crate::typechecker::TypecheckingError;
use crate::types::Type;
use crate::typed_ast::{TypedBlock, TypedFnDecl, TypedStmt};

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

pub fn check_function_returns(decl: &TypedFnDecl) -> Result<(), TypecheckingError> {
    if decl.return_type != Type::Void && !block_always_returns(&decl.body.value) {
        return Err(TypecheckingError::missing_return(decl.body.span.clone()));
    }
    Ok(())
}
