use crate::ast::{Decl, Expr, Stmt};
use crate::lexer::{Span, Spanned};
use crate::{ast, lexer};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Environment {
    names: HashMap<ast::Ident, (Type, Span)>,
}

impl Environment {
    fn global() -> Self {
        // void printInt(int)
        // void printString(string)
        // void error()
        // int readInt()
        // string readString()
        let fictional_span = Span::from(0..0);

        let predefined = [
            (
                ast::Ident::new("printInt"),
                (
                    Type::Function(vec![Type::Int], Box::new(Type::Void)),
                    fictional_span.clone(),
                ),
            ),
            (
                ast::Ident::new("printString"),
                (
                    Type::Function(vec![Type::LatteString], Box::new(Type::Void)),
                    fictional_span.clone(),
                ),
            ),
            (
                ast::Ident::new("error"),
                (
                    Type::Function(vec![], Box::new(Type::Void)),
                    fictional_span.clone(),
                ),
            ),
            (
                ast::Ident::new("readInt"),
                (
                    Type::Function(vec![], Box::new(Type::Int)),
                    fictional_span.clone(),
                ),
            ),
            (
                ast::Ident::new("readString"),
                (
                    Type::Function(vec![], Box::new(Type::LatteString)),
                    fictional_span.clone(),
                ),
            ),
        ];

        Environment {
            names: HashMap::from(predefined), // TODO: predefined funcs?
        }
    }

    fn get_type(&self, name: &ast::Ident) -> Option<&Type> {
        self.names.get(name).map(|(ty, _)| ty)
    }

    fn get_span(&self, name: &ast::Ident) -> Option<&Span> {
        self.names.get(name).map(|(_, span)| span)
    }

    fn insert(&mut self, name: Spanned<ast::Ident>, ty: Type) -> Result<(), TypecheckingError> {
        let span = name.span;
        let name = name.value;

        let already_present = self.names.insert(name.clone(), (ty, span.clone()));
        if let Some((_, old_declaration)) = already_present {
            Err(TypecheckingError::redeclaration(
                name,
                old_declaration,
                span,
            ))
        } else {
            Ok(())
        }
    }

    fn overwrite(&mut self, name: Spanned<ast::Ident>, ty: Type) {
        self.names.insert(name.value, (ty, name.span));
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Int,
    Bool,
    LatteString,
    Void,
    Function(Vec<Type>, Box<Type>),
}

#[derive(Debug, Clone)]
pub enum TypecheckingErrorKind {
    UnknownName(ast::Ident),
    TypeMismatch {
        expected: Vec<Type>,
        found: Type,
    },
    NotCallable(Type),
    WrongArgumentCount {
        expected: usize,
        found: usize,
    },
    Redeclaration {
        name: ast::Ident,
        old_declaration: Span,
    },
    UnkownType(ast::Type),
    InconsistentReturnTypes {
        expected: Type,
        found: Type,
    },
    IncrDecrOnNonInt,
}

#[derive(Debug, Clone)]
pub struct TypecheckingError {
    kind: TypecheckingErrorKind,
    location: lexer::Span,
}

impl TypecheckingError {
    fn unknown_name(name: ast::Ident, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::UnknownName(name),
            location,
        }
    }

    fn type_mismatch(expected: impl TypeMatch, found: Type, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::TypeMismatch {
                expected: expected.into_vec(),
                found,
            },
            location,
        }
    }

    fn not_callable(ty: Type, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::NotCallable(ty),
            location,
        }
    }

    fn wrong_argument_count(expected: usize, found: usize, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::WrongArgumentCount { expected, found },
            location,
        }
    }

    fn redeclaration(name: ast::Ident, old_declaration: Span, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::Redeclaration {
                name,
                old_declaration,
            },
            location,
        }
    }

    fn unknown_type(ty: ast::Type, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::UnkownType(ty),
            location,
        }
    }

    fn inconsistent_return_types(expected: Type, found: Type, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::InconsistentReturnTypes { expected, found },
            location,
        }
    }

    fn incr_decr_on_non_int(location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::IncrDecrOnNonInt,
            location,
        }
    }
}

trait TypeMatch {
    fn matches(&self, other: &Type) -> bool;
    fn into_vec(self) -> Vec<Type>;
}

impl TypeMatch for &Type {
    fn matches(&self, other: &Type) -> bool {
        *self == other
    }

    fn into_vec(self) -> Vec<Type> {
        vec![self.clone()]
    }
}

impl TypeMatch for Type {
    fn matches(&self, other: &Type) -> bool {
        self == other
    }
    fn into_vec(self) -> Vec<Type> {
        vec![self]
    }
}

impl<const N: usize> TypeMatch for [Type; N] {
    fn matches(&self, other: &Type) -> bool {
        self.iter().any(|t: &Type| t.matches(other))
    }
    fn into_vec(self) -> Vec<Type> {
        self.to_vec()
    }
}

impl TypeMatch for Vec<Type> {
    fn matches(&self, other: &Type) -> bool {
        self.iter().any(|t: &Type| t.matches(other))
    }
    fn into_vec(self) -> Vec<Type> {
        self
    }
}

fn ensure_type(
    expected: impl TypeMatch,
    found: &Type,
    location: lexer::Span,
) -> Result<Type, TypecheckingError> {
    if expected.matches(&found) {
        Ok(found.clone())
    } else {
        Err(TypecheckingError::type_mismatch(
            expected,
            found.clone(),
            location,
        ))
    }
}

trait SpannedAst<T> {
    fn span(&self) -> lexer::Span;
    fn value(&self) -> &T;
}

impl<T> SpannedAst<T> for Spanned<T> {
    fn span(&self) -> lexer::Span {
        self.span.clone()
    }
    fn value(&self) -> &T {
        &self.value
    }
}

impl<T> SpannedAst<T> for Box<Spanned<T>> {
    fn span(&self) -> lexer::Span {
        self.span.clone()
    }
    fn value(&self) -> &T {
        &self.value
    }
}

fn typecheck_expr(
    expr: &impl SpannedAst<Expr>,
    env: &Environment,
) -> Result<Type, TypecheckingError> {
    match expr.value() {
        ast::Expr::Error => unreachable!(),
        ast::Expr::Variable(ident) => env
            .get_type(ident)
            .cloned()
            .ok_or(TypecheckingError::unknown_name(ident.clone(), expr.span())),
        ast::Expr::Literal(literal) => match literal {
            ast::Literal::Int(_) => Ok(Type::Int),
            ast::Literal::Bool(_) => Ok(Type::Bool),
            ast::Literal::String(_) => Ok(Type::LatteString),
        },
        ast::Expr::Unary { op, expr } => {
            let expr_type = typecheck_expr(expr, env)?;
            match &op.value {
                ast::UnaryOp::Neg => ensure_type(Type::Int, &expr_type, op.span()),
                ast::UnaryOp::Not => ensure_type(Type::Bool, &expr_type, op.span()),
            }
        }
        ast::Expr::Binary { lhs, op, rhs } => {
            let lhs_type = typecheck_expr(lhs, env)?;
            let rhs_type = typecheck_expr(rhs, env)?;
            match &op.value {
                // Both strings and ints can be added
                ast::BinaryOp::Add => {
                    ensure_type([Type::Int, Type::LatteString], &lhs_type, op.span.clone())
                }
                ast::BinaryOp::Sub
                | ast::BinaryOp::Mul
                | ast::BinaryOp::Div
                | ast::BinaryOp::Mod => {
                    ensure_type(Type::Int, &lhs_type, lhs.span.clone())?;
                    ensure_type(Type::Int, &rhs_type, rhs.span.clone())?;
                    Ok(Type::Int)
                }
                ast::BinaryOp::Lt | ast::BinaryOp::Lte | ast::BinaryOp::Gt | ast::BinaryOp::Gte => {
                    ensure_type(Type::Int, &lhs_type, lhs.span.clone())?;
                    ensure_type(Type::Int, &rhs_type, rhs.span.clone())?;
                    Ok(Type::Bool)
                }
                ast::BinaryOp::Eq | ast::BinaryOp::Neq => {
                    ensure_type(lhs_type, &rhs_type, rhs.span.clone())?;
                    Ok(Type::Bool)
                }
                ast::BinaryOp::And | ast::BinaryOp::Or => {
                    ensure_type(Type::Bool, &lhs_type, lhs.span.clone())?;
                    ensure_type(Type::Bool, &rhs_type, rhs.span.clone())?;
                    Ok(Type::Bool)
                }
            }
        }
        ast::Expr::Application { target, args } => {
            let target_type = typecheck_expr(target, env)?;
            let arg_types = args
                .iter()
                .map(|arg| typecheck_expr(arg, env))
                .collect::<Result<Vec<_>, _>>()?;
            match target_type {
                Type::Function(expected_arg_types, return_type) => {
                    if expected_arg_types.len() != arg_types.len() {
                        return Err(TypecheckingError::wrong_argument_count(
                            expected_arg_types.len(),
                            arg_types.len(),
                            target.span(),
                        ));
                    }
                    for ((expected, arg_expr), found) in
                        expected_arg_types.iter().zip(args).zip(arg_types)
                    {
                        ensure_type(expected.clone(), &found, arg_expr.span())?;
                    }
                    Ok(*return_type)
                }
                _ => Err(TypecheckingError::not_callable(target_type, target.span())),
            }
        }
    }
}

fn typecheck_block(
    stmt: &impl SpannedAst<ast::Block>,
    env: &mut Environment,
    expected_return_type: &Type,
) -> Result<Option<Type>, TypecheckingError> {
    let mut return_type: Option<Type> = None;
    for stmt in &stmt.value().0 {
        let stmt_type = typecheck_stmt(stmt, env, expected_return_type)?;
        if let Some(stmt_type) = stmt_type {
            if let Some(return_type) = return_type.as_ref() {
                if return_type != &stmt_type {
                    return Err(TypecheckingError::inconsistent_return_types(
                        return_type.clone(),
                        stmt_type,
                        stmt.span(),
                    ));
                }
            } else {
                return_type = Some(stmt_type);
            }
        }
    }
    Ok(return_type)
}

fn resolve_type(
    name: &impl SpannedAst<ast::Type>,
    env: &Environment,
) -> Result<Type, TypecheckingError> {
    // TODO: This will change once we have classes
    // TODO: And also if we implement string interning
    match name.value().0.as_str() {
        "int" => Ok(Type::Int),
        "bool" => Ok(Type::Bool),
        "string" => Ok(Type::LatteString),
        _ => Err(TypecheckingError::unknown_type(
            name.value().clone(),
            name.span(),
        )),
    }
}

struct FunctionHeader {
    name: Spanned<ast::Ident>,
    function_type: Type,
    args: Vec<(Spanned<ast::Ident>, Type)>,
    return_type: Type,
}

fn resolve_function_header(
    name: &Spanned<ast::Ident>,
    return_type: &Spanned<ast::Type>,
    args: &[Spanned<ast::Arg>],
    env: &Environment,
) -> Result<FunctionHeader, TypecheckingError> {
    let return_type = resolve_type(return_type, env)?;
    let args = args
        .iter()
        .map(|arg: &Spanned<ast::Arg>| {
            let ty = resolve_type(&arg.value().ty, env);
            let name = arg.value().name.clone();
            ty.map(|ty| (name, ty))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let function_type = Type::Function(
        args.clone().into_iter().map(|(_, ty)| ty).collect(),
        Box::new(return_type.clone()),
    );

    let header = FunctionHeader {
        name: name.clone(),
        function_type,
        args,
        return_type,
    };

    Ok(header)
}

fn typecheck_decl(
    decl: &impl SpannedAst<ast::Decl>,
    env: &mut Environment,
) -> Result<(), TypecheckingError> {
    match decl.value() {
        ast::Decl::Fn {
            return_type,
            name,
            args,
            body,
        } => {
            let header = resolve_function_header(name, return_type, args, env)?;

            // Define all the arguments in the environment
            for arg in header.args {
                env.overwrite(arg.0, arg.1.clone());
            }

            // Define the function itself for recursive calls
            env.overwrite(name.clone(), header.function_type);

            typecheck_block(body, env, &header.return_type)?;
            Ok(())
        }

        ast::Decl::Var { ty, items } => {
            let ty = resolve_type(ty, env)?;
            for item in items {
                let name = &item.value().ident;
                let init = &item.value().init;
                if let Some(init) = init {
                    let init_type = typecheck_expr(init, env)?;
                    ensure_type(ty.clone(), &init_type, init.span())?;
                }
                env.overwrite(name.clone(), ty.clone());
            }
            Ok(())
        }
    }
}

// TODO: Change return type
fn typecheck_stmt(
    stmt: &impl SpannedAst<ast::Stmt>,
    env: &mut Environment,
    expected_return_type: &Type,
) -> Result<Option<Type>, TypecheckingError> {
    match stmt.value() {
        Stmt::Error => unreachable!(),
        Stmt::Block(block) => {
            let mut env = env.clone();
            typecheck_block(block, &mut env, expected_return_type)
        }
        Stmt::Decl(decl) => {
            let mut env = env.clone();
            typecheck_decl(decl, &mut env)?;
            Ok(None)
        }
        Stmt::Assignment { target, expr } => {
            let target_type =
                env.get_type(target.value())
                    .ok_or(TypecheckingError::unknown_name(
                        target.value().clone(),
                        target.span(),
                    ))?;
            let expr_type = typecheck_expr(expr, env)?;
            ensure_type(target_type.clone(), &expr_type, expr.span())?;
            Ok(None)
        }
        Stmt::Return(init) => {
            if let Some(init) = init {
                let init_type = typecheck_expr(init, env)?;
                ensure_type(expected_return_type.clone(), &init_type, init.span())?;
                Ok(Some(expected_return_type.clone()))
            } else {
                ensure_type(Type::Void, expected_return_type, stmt.span())?;
                Ok(Some(expected_return_type.clone()))
            }
        }
        Stmt::If {
            cond,
            then,
            otherwise,
        } => {
            let cond_type = typecheck_expr(cond, env)?;
            ensure_type(Type::Bool, &cond_type, cond.span())?;
            let then_type = typecheck_stmt(then, env, expected_return_type)?;
            if let Some(otherwise) = otherwise {
                let otherwise_type = typecheck_stmt(otherwise, env, expected_return_type)?;
                if let (Some(then_type), Some(otherwise_type)) =
                    (then_type.clone(), otherwise_type.clone())
                {
                    if then_type != otherwise_type {
                        return Err(TypecheckingError::inconsistent_return_types(
                            then_type.clone(),
                            otherwise_type,
                            stmt.span(),
                        ));
                    }
                }
                Ok(then_type.or(otherwise_type))
            } else {
                Ok(then_type)
            }
        }
        Stmt::While { cond, body } => {
            let cond_type = typecheck_expr(cond, env)?;
            ensure_type(Type::Bool, &cond_type, cond.span())?;
            typecheck_stmt(body, env, expected_return_type)
        }
        Stmt::Expr(expr) => {
            typecheck_expr(expr, env)?;
            Ok(None)
        }
        Stmt::Incr(target) | Stmt::Decr(target) => {
            if let ast::Expr::Variable(ident) = target.value() {
                let target_type = env.get_type(ident).ok_or(TypecheckingError::unknown_name(
                    ident.clone(),
                    target.span(),
                ))?;
                ensure_type(Type::Int, &target_type, target.span())?;
                Ok(None)
            } else {
                Err(TypecheckingError::incr_decr_on_non_int(target.span()))
            }
        }
    }
}

pub fn typecheck_program(program: &ast::Program) -> Result<(), TypecheckingError> {
    let mut env = Environment::global();
    // If we supported creating new types, we would have to add them to the environment here

    // Before typechecking bodies, first add all the function declarations to the environment
    for decl in &program.0 {
        match decl.value() {
            ast::Decl::Fn {
                return_type,
                name,
                args,
                body: _,
            } => {
                let header = resolve_function_header(name, return_type, args, &mut env)?;
                env.insert(name.clone(), header.function_type)?;
            }
            ast::Decl::Var { ty: _, items: _ } => {
                // Skip globals for now
            }
        }
    }

    // Typecheck bodies
    for decl in &program.0 {
        match decl.value() {
            ast::Decl::Fn {
                return_type,
                name: _,
                args: _,
                body,
            } => {
                let return_type = resolve_type(return_type, &mut env)?;
                typecheck_block(body, &mut env, &return_type)?;
            }
            _ => {}
        }
    }

    Ok(())
}
