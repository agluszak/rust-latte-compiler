use crate::ast;

use crate::typechecker::{Type, TypecheckingError};
use crate::typed_ast::{TypedBlock, TypedDecl, TypedExpr, TypedExprKind, TypedStmt};
use std::collections::HashMap;
use std::ops::Not;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Environment {
    parent: Option<Box<Environment>>,
    bools: HashMap<ast::Ident, TriLogic>,
}

impl Environment {
    fn global() -> Environment {
        Environment {
            parent: None,
            bools: HashMap::new(),
        }
    }

    fn local(&self) -> Environment {
        Environment {
            parent: Some(Box::new(self.clone())),
            bools: HashMap::new(),
        }
    }

    fn mark_bool(&mut self, name: &ast::Ident, value: TriLogic) {
        self.bools.insert(name.clone(), value);
    }

    fn get_bool(&self, name: &ast::Ident) -> Option<TriLogic> {
        self.bools.get(name).cloned().or_else(|| {
            self.parent
                .as_ref()
                .and_then(|parent| parent.get_bool(name))
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TriLogic {
    True,
    False,
    Unknown,
}

impl TriLogic {
    fn trilogic_equal(self, other: TriLogic) -> TriLogic {
        match (self, other) {
            (TriLogic::True, TriLogic::True) => TriLogic::True,
            (TriLogic::False, TriLogic::False) => TriLogic::True,
            _ => TriLogic::Unknown,
        }
    }

    fn trilogic_different(self, other: TriLogic) -> TriLogic {
        match (self, other) {
            (TriLogic::True, TriLogic::False) => TriLogic::True,
            (TriLogic::False, TriLogic::True) => TriLogic::True,
            _ => TriLogic::Unknown,
        }
    }

    fn or(self, other: Self) -> Self {
        match (self, other) {
            (TriLogic::True, _) => TriLogic::True,
            (_, TriLogic::True) => TriLogic::True,
            (TriLogic::False, TriLogic::False) => TriLogic::False,
            _ => TriLogic::Unknown,
        }
    }

    fn and(self, other: Self) -> Self {
        match (self, other) {
            (TriLogic::True, TriLogic::True) => TriLogic::True,
            (TriLogic::False, _) => TriLogic::False,
            (_, TriLogic::False) => TriLogic::False,
            _ => TriLogic::Unknown,
        }
    }
}

impl From<bool> for TriLogic {
    fn from(b: bool) -> Self {
        if b {
            TriLogic::True
        } else {
            TriLogic::False
        }
    }
}

impl Not for TriLogic {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            TriLogic::True => TriLogic::False,
            TriLogic::False => TriLogic::True,
            TriLogic::Unknown => TriLogic::Unknown,
        }
    }
}

/// All variables *must* be declared before DFA
fn data_flow_analysis(expr: &TypedExpr, env: &Environment) -> TriLogic {
    match &expr.expr {
        TypedExprKind::Literal(ast::Literal::Bool(true)) => TriLogic::True,
        TypedExprKind::Literal(ast::Literal::Bool(false)) => TriLogic::False,
        TypedExprKind::Literal(_) => TriLogic::Unknown,
        TypedExprKind::Variable(ident, _) => {
            if let Some(value) = env.get_bool(ident) {
                value
            } else {
                TriLogic::Unknown
            }
        }
        TypedExprKind::Unary { op, expr } => {
            let expr = data_flow_analysis(&expr.value, env);
            match op.value {
                ast::UnaryOp::Not => !expr,
                ast::UnaryOp::Neg => TriLogic::Unknown,
            }
        }
        TypedExprKind::Binary { op, lhs, rhs } => {
            let lhs = data_flow_analysis(&lhs.value, env);
            let rhs = data_flow_analysis(&rhs.value, env);
            match &op.value {
                ast::BinaryOp::Add
                | ast::BinaryOp::Sub
                | ast::BinaryOp::Mul
                | ast::BinaryOp::Div
                | ast::BinaryOp::Mod => TriLogic::Unknown,
                // We could check for integers etc. here
                ast::BinaryOp::Eq => lhs.trilogic_equal(rhs),
                ast::BinaryOp::Neq => lhs.trilogic_different(rhs),
                ast::BinaryOp::Lt | ast::BinaryOp::Lte | ast::BinaryOp::Gt | ast::BinaryOp::Gte => {
                    TriLogic::Unknown
                }
                ast::BinaryOp::And => lhs.and(rhs),
                ast::BinaryOp::Or => lhs.or(rhs),
            }
        }
        TypedExprKind::Application { .. } => TriLogic::Unknown,
    }
}

fn return_analysis_block(
    block: &TypedBlock,
    env: &mut Environment,
) -> Result<bool, TypecheckingError> {
    let mut always_returns = false;
    for stmt in &block.0 {
        always_returns = always_returns || return_analysis_stmt(&stmt.value, env)?;
    }
    Ok(always_returns)
}

fn return_analysis_stmt(
    stmt: &TypedStmt,
    env: &mut Environment,
) -> Result<bool, TypecheckingError> {
    let always_returns = match stmt {
        TypedStmt::Expr(_) => false,
        TypedStmt::Return(_) => true,
        TypedStmt::If {
            cond,
            then,
            otherwise,
        } => {
            // An if statement returns on all control paths if both
            // the "if" and "else" branches return on all control paths
            // or the "if" statement is always true.
            let dfa_value = data_flow_analysis(&cond.value, env);
            let then_always_returns = return_analysis_stmt(&then.value, env)?;
            match otherwise {
                Some(otherwise) => {
                    let otherwise_always_returns = return_analysis_stmt(&otherwise.value, env)?;
                    match dfa_value {
                        TriLogic::True => then_always_returns,
                        TriLogic::False => otherwise_always_returns,
                        TriLogic::Unknown => then_always_returns && otherwise_always_returns,
                    }
                }
                None => {
                    if dfa_value == TriLogic::True {
                        then_always_returns
                    } else {
                        false
                    }
                }
            }
        }
        TypedStmt::While { cond, body: _ } => {
            // A while loop returns on all control paths only
            // if the "while" condition is always true, because it loops
            // forever and it doesn't matter if the body returns or not.
            let dfa_value = data_flow_analysis(&cond.value, env);
            match dfa_value {
                TriLogic::True => true,
                TriLogic::False => false,
                TriLogic::Unknown => false,
            }
        }
        TypedStmt::Incr(_) => false,
        TypedStmt::Decr(_) => false,
        TypedStmt::Assignment { target, expr, .. } => {
            let dfa_value = data_flow_analysis(&expr.value, env);
            env.mark_bool(&target.value, dfa_value);
            false
        }
        TypedStmt::Empty => false,
        TypedStmt::Block(block) => {
            let mut env = env.local();
            return_analysis_block(&block.value, &mut env)?
        }
        TypedStmt::Decl(decl) => return_analysis_decl(&decl.value, env)?,
    };
    Ok(always_returns)
}

fn return_analysis_decl(
    decl: &TypedDecl,
    env: &mut Environment,
) -> Result<bool, TypecheckingError> {
    match decl {
        TypedDecl::Fn {
            return_type,
            name: _,
            args,
            body,
        } => {
            let mut env = env.local();
            for arg in args {
                if arg.value.ty == Type::Bool {
                    env.mark_bool(&arg.value.name.value, TriLogic::Unknown);
                }
            }
            let always_returns = return_analysis_block(&body.value, &mut env)?;
            if !always_returns && return_type != &Type::Void {
                return Err(TypecheckingError::missing_return(body.span.clone()));
            }
            Ok(false)
        }
        TypedDecl::Var { ty, items } => {
            if ty == &Type::Bool {
                for item in items {
                    if let Some(expr) = &item.value.init {
                        let dfa_value = data_flow_analysis(&expr.value, env);
                        env.mark_bool(&item.value.ident.value, dfa_value);
                    } else {
                        env.mark_bool(&item.value.ident.value, TriLogic::False);
                    }
                }
            }
            Ok(false)
        }
    }
}

pub fn top_level_return_analysis(decl: &TypedDecl) -> Result<(), TypecheckingError> {
    let mut env = Environment::global();
    return_analysis_decl(decl, &mut env)?;
    Ok(())
}
