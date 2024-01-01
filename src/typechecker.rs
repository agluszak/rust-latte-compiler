use crate::dfa::top_level_return_analysis;
use crate::lexer::{Span, Spanned};
use crate::typed_ast::{
    TypedArg, TypedBlock, TypedDecl, TypedExpr, TypedExprKind, TypedItem, TypedProgram, TypedStmt,
    VariableId,
};
use crate::{ast, lexer};
use std::collections::{BTreeMap, BTreeSet};

use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Environment {
    parent: Option<Box<Environment>>,
    names: BTreeMap<ast::Ident, VariableData>,
    next_variable_id: u32,
}

impl Environment {
    pub fn ready(self) -> ReadyEnvironment {
        let mut globals = BTreeMap::new();
        let mut names = BTreeMap::new();

        for (name, data) in self.names {
            let id = data.id;
            globals.insert(name.0.clone(), data.ty);
            names.insert(id, name.0);
        }

        ReadyEnvironment { globals, names }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadyEnvironment {
    pub globals: BTreeMap<String, Type>,
    pub names: BTreeMap<VariableId, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VariableData {
    ty: Type,
    span: Span, // built-ins don't have span
    id: VariableId,
}

impl VariableData {
    fn new(ty: Type, span: Span, id: VariableId) -> Self {
        Self { ty, span, id }
    }
}

impl Environment {
    fn fresh_variable_id(&mut self) -> VariableId {
        let id = self.next_variable_id;
        self.next_variable_id += 1;
        VariableId::new(id)
    }

    fn add_predefined_fn(&mut self, name: &str, args: Vec<Type>, ret: Type) {
        let id = self.fresh_variable_id();
        self.names.insert(
            ast::Ident::new(name),
            VariableData {
                ty: Type::Function(args, Box::new(ret)),
                span: 0..0,
                id,
            },
        );
    }

    fn global() -> Self {
        let mut env = Environment {
            parent: None,
            names: BTreeMap::new(),
            next_variable_id: 0,
        };

        env.add_predefined_fn("printInt", vec![Type::Int], Type::Void);
        env.add_predefined_fn("printString", vec![Type::LatteString], Type::Void);
        env.add_predefined_fn("error", vec![], Type::Void);
        env.add_predefined_fn("readInt", vec![], Type::Int);
        env.add_predefined_fn("readString", vec![], Type::LatteString);

        env
    }

    fn local(&self) -> Self {
        Environment {
            parent: Some(Box::new(self.clone())),
            names: BTreeMap::new(),
            next_variable_id: self.next_variable_id,
        }
    }

    fn with_local<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, TypecheckingError>,
    ) -> Result<T, TypecheckingError> {
        let mut local = self.local();

        let result = f(&mut local);

        self.next_variable_id = local.next_variable_id;

        result
    }

    fn get_type(&self, name: &ast::Ident) -> Option<Type> {
        self.get_data(name).map(|data| data.ty.clone())
    }

    fn get_data(&self, name: &ast::Ident) -> Option<&VariableData> {
        self.names.get(name).or_else(|| {
            self.parent
                .as_ref()
                .and_then(|parent| parent.get_data(name))
        })
    }

    fn get_span(&self, name: &ast::Ident) -> Option<&Span> {
        self.get_data(name).map(|data| &data.span)
    }

    fn insert_data(
        &mut self,
        name: ast::Ident,
        data: VariableData,
    ) -> Result<(), TypecheckingError> {
        let new_span = data.span.clone();
        let already_present = self.overwrite_data(name.clone(), data);
        if let Some(old_data) = already_present {
            Err(TypecheckingError::redeclaration(
                name,
                old_data.span,
                new_span,
            ))
        } else {
            Ok(())
        }
    }

    fn overwrite_data(&mut self, name: ast::Ident, data: VariableData) -> Option<VariableData> {
        self.names.insert(name, data)
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

impl Display for Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Bool => write!(f, "bool"),
            Type::LatteString => write!(f, "string"),
            Type::Void => write!(f, "void"),
            Type::Function(args, ret) => {
                write!(f, "function(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ") -> {ret}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypecheckingErrorKind {
    UndefinedVariable(ast::Ident),
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
    UnknownType(ast::TypeName),
    DuplicateArgument(ast::Ident),
    MissingReturn,
    VoidVariable,
    VoidReturn,
    NoMain,
    InvalidLvalue,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypecheckingError {
    pub kind: TypecheckingErrorKind,
    pub location: lexer::Span,
}

impl TypecheckingError {
    pub fn undefined_variable(name: ast::Ident, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::UndefinedVariable(name),
            location,
        }
    }

    pub fn type_mismatch(expected: impl TypeMatch, found: Type, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::TypeMismatch {
                expected: expected.into_vec(),
                found,
            },
            location,
        }
    }

    pub fn not_callable(ty: Type, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::NotCallable(ty),
            location,
        }
    }

    pub fn wrong_argument_count(expected: usize, found: usize, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::WrongArgumentCount { expected, found },
            location,
        }
    }

    pub fn redeclaration(name: ast::Ident, old_declaration: Span, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::Redeclaration {
                name,
                old_declaration,
            },
            location,
        }
    }

    pub fn unknown_type(ty: ast::TypeName, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::UnknownType(ty),
            location,
        }
    }

    pub fn duplicate_argument(name: ast::Ident, location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::DuplicateArgument(name),
            location,
        }
    }

    pub fn missing_return(location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::MissingReturn,
            location,
        }
    }

    pub fn void_variable(location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::VoidVariable,
            location,
        }
    }

    pub fn void_return(location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::VoidReturn,
            location,
        }
    }

    pub fn no_main(location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::NoMain,
            location,
        }
    }

    pub fn invalid_lvalue(location: lexer::Span) -> Self {
        Self {
            kind: TypecheckingErrorKind::InvalidLvalue,
            location,
        }
    }
}

pub trait TypeMatch {
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
    if expected.matches(found) {
        Ok(found.clone())
    } else {
        Err(TypecheckingError::type_mismatch(
            expected,
            found.clone(),
            location,
        ))
    }
}

trait SpannedAst {
    type Inner;
    type Output<Inner>;
    fn span(&self) -> lexer::Span;
    fn value(&self) -> &Self::Inner;
    fn into_value(self) -> Self::Inner;
    fn map<F, MappedInner>(self, f: F) -> Self::Output<MappedInner>
    where
        F: FnOnce(Self::Inner) -> MappedInner;
}

impl<T> SpannedAst for Spanned<T> {
    type Inner = T;
    type Output<Inner> = Spanned<Inner>;
    fn span(&self) -> lexer::Span {
        self.span.clone()
    }
    fn value(&self) -> &T {
        &self.value
    }
    fn into_value(self) -> T {
        self.value
    }
    fn map<F, MappedInner>(self, f: F) -> Spanned<MappedInner>
    where
        F: FnOnce(T) -> MappedInner,
    {
        Spanned {
            span: self.span,
            value: f(self.value),
        }
    }
}

impl<T> SpannedAst for Box<Spanned<T>> {
    type Inner = T;
    type Output<Inner> = Box<Spanned<Inner>>;
    fn span(&self) -> lexer::Span {
        self.span.clone()
    }
    fn value(&self) -> &T {
        &self.value
    }
    fn into_value(self) -> T {
        self.value
    }
    fn map<F, MappedInner>(self, f: F) -> Box<Spanned<MappedInner>>
    where
        F: FnOnce(T) -> MappedInner,
    {
        Box::new(Spanned {
            span: self.span,
            value: f(self.value),
        })
    }
}

fn typecheck_expr(
    expr: impl SpannedAst<Inner = ast::Expr>,
    env: &mut Environment,
) -> Result<Spanned<TypedExpr>, TypecheckingError> {
    let span = expr.span();
    let typed_expr = match expr.into_value() {
        ast::Expr::Variable(ident) => {
            let data = env.get_data(&ident).cloned().ok_or_else(|| {
                TypecheckingError::undefined_variable(ident.clone(), span.clone())
            })?;
            let ty = data.ty;
            let expr = TypedExprKind::Variable(ident, data.id);
            Ok(TypedExpr { expr, ty })
        }
        ast::Expr::Literal(literal) => {
            let ty = match literal {
                ast::Literal::Int(_) => Type::Int,
                ast::Literal::Bool(_) => Type::Bool,
                ast::Literal::String(_) => Type::LatteString,
            };
            Ok(TypedExpr {
                expr: TypedExprKind::Literal(literal),
                ty,
            })
        }
        ast::Expr::Unary { op, expr } => {
            let typed_expr = typecheck_expr(expr, env)?;
            let expr_ty = &typed_expr.value().ty;
            let ty = match &op.value {
                ast::UnaryOp::Neg => ensure_type(Type::Int, expr_ty, op.span())?,
                ast::UnaryOp::Not => ensure_type(Type::Bool, expr_ty, op.span())?,
            };
            Ok(TypedExpr {
                expr: TypedExprKind::Unary {
                    op,
                    expr: Box::new(typed_expr),
                },
                ty,
            })
        }
        ast::Expr::Binary { lhs, op, rhs } => {
            let lhs_span = lhs.span();
            let rhs_span = rhs.span();

            let lhs_typed_expr = typecheck_expr(lhs, env)?;
            let rhs_typed_expr = typecheck_expr(rhs, env)?;
            let lhs_ty = &lhs_typed_expr.value().ty;
            let rhs_ty = &rhs_typed_expr.value().ty;
            let ty = match &op.value {
                // Both strings and ints can be added
                ast::BinaryOp::Add => {
                    ensure_type([Type::Int, Type::LatteString], lhs_ty, lhs_span)?;
                    ensure_type(lhs_ty, rhs_ty, rhs_span)?;
                    lhs_ty.clone()
                }
                ast::BinaryOp::Sub
                | ast::BinaryOp::Mul
                | ast::BinaryOp::Div
                | ast::BinaryOp::Mod => {
                    ensure_type(Type::Int, lhs_ty, lhs_span)?;
                    ensure_type(Type::Int, rhs_ty, rhs_span)?;
                    Type::Int
                }
                ast::BinaryOp::Lt | ast::BinaryOp::Lte | ast::BinaryOp::Gt | ast::BinaryOp::Gte => {
                    ensure_type(Type::Int, lhs_ty, lhs_span)?;
                    ensure_type(Type::Int, rhs_ty, rhs_span)?;
                    Type::Bool
                }
                ast::BinaryOp::Eq | ast::BinaryOp::Neq => {
                    ensure_type(lhs_ty, rhs_ty, rhs_span)?;
                    Type::Bool
                }
                ast::BinaryOp::And | ast::BinaryOp::Or => {
                    ensure_type(Type::Bool, lhs_ty, lhs_span)?;
                    ensure_type(Type::Bool, rhs_ty, rhs_span)?;
                    Type::Bool
                }
            };
            Ok(TypedExpr {
                expr: TypedExprKind::Binary {
                    lhs: Box::new(lhs_typed_expr),
                    op,
                    rhs: Box::new(rhs_typed_expr),
                },
                ty,
            })
        }
        ast::Expr::Application { target, args } => {
            let target_span = target.span();
            let target_typed_expr = typecheck_expr(target, env)?;
            let arg_typed_exprs = args
                .clone()
                .into_iter()
                .map(|arg| typecheck_expr(arg, env))
                .collect::<Result<Vec<_>, _>>()?;
            let arg_types: Vec<Type> = arg_typed_exprs
                .iter()
                .map(|arg| arg.value().ty.clone())
                .collect();
            let target_ty = &target_typed_expr.value().ty;
            let ty = *match target_ty {
                Type::Function(expected_arg_types, return_type) => {
                    if expected_arg_types.len() != arg_types.len() {
                        Err(TypecheckingError::wrong_argument_count(
                            expected_arg_types.len(),
                            arg_typed_exprs.len(),
                            target_span,
                        ))?;
                    }
                    for ((expected, arg_expr), found) in
                        expected_arg_types.iter().zip(args.iter()).zip(arg_types)
                    {
                        ensure_type(expected.clone(), &found, arg_expr.span())?;
                    }
                    return_type.clone()
                }
                _ => Err(TypecheckingError::not_callable(
                    target_ty.clone(),
                    target_span,
                ))?,
            };
            Ok(TypedExpr {
                expr: TypedExprKind::Application {
                    target: Box::new(target_typed_expr),
                    args: arg_typed_exprs,
                },
                ty,
            })
        }
    }?;
    Ok(Spanned::new(span, typed_expr))
}

fn typecheck_block(
    stmt: impl SpannedAst<Inner = ast::Block>,
    env: &mut Environment,
    expected_return_type: &Type,
) -> Result<Spanned<TypedBlock>, TypecheckingError> {
    let span = stmt.span();
    let typed_stmts = stmt
        .into_value()
        .0
        .into_iter()
        .map(|stmt| typecheck_stmt(stmt, env, expected_return_type))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Spanned::new(span, TypedBlock(typed_stmts)))
}

fn resolve_type(
    name: &impl SpannedAst<Inner = ast::TypeName>,
    _env: &Environment,
) -> Result<Type, TypecheckingError> {
    // TODO: This will change once we have classes
    // TODO: And also if we implement string interning
    match name.value().0.as_str() {
        "int" => Ok(Type::Int),
        "boolean" => Ok(Type::Bool),
        "string" => Ok(Type::LatteString),
        "void" => Ok(Type::Void),
        _ => Err(TypecheckingError::unknown_type(
            name.value().clone(),
            name.span(),
        )),
    }
}

struct FunctionHeader {
    function_type: Type,
    args: Vec<(Spanned<ast::Ident>, Type)>,
    return_type: Type,
}

fn resolve_function_header(
    return_type: &Spanned<ast::TypeName>,
    args: &[Spanned<ast::Arg>],
    env: &Environment,
) -> Result<FunctionHeader, TypecheckingError> {
    let return_type = resolve_type(return_type, env)?;
    let args = args
        .iter()
        .map(|arg: &Spanned<ast::Arg>| {
            let ty = resolve_type(&arg.value().ty, env).and_then(|ty| {
                if ty == Type::Void {
                    Err(TypecheckingError::void_variable(arg.span()))
                } else {
                    Ok(ty)
                }
            });
            let name = arg.value().name.clone();
            ty.map(|ty| (name, ty))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Ensure that argument names are unique
    let mut arg_names = BTreeSet::new();
    for (arg_name, _) in &args {
        if !arg_names.insert(arg_name.value().0.clone()) {
            return Err(TypecheckingError::duplicate_argument(
                arg_name.value().clone(),
                arg_name.span(),
            ));
        }
    }

    let function_type = Type::Function(
        args.clone().into_iter().map(|(_, ty)| ty).collect(),
        Box::new(return_type.clone()),
    );

    let header = FunctionHeader {
        function_type,
        args,
        return_type,
    };

    Ok(header)
}

fn typecheck_decl(
    decl: impl SpannedAst<Inner = ast::Decl>,
    env: &mut Environment,
) -> Result<Spanned<TypedDecl>, TypecheckingError> {
    let span = decl.span();
    let typed_decl = match decl.into_value() {
        ast::Decl::Fn {
            return_type,
            name,
            args,
            body,
        } => {
            let header = resolve_function_header(&return_type, &args, env)?;

            let (args, body) = env.with_local(|env| {
                let mut typed_args = Vec::new();

                // Define all the arguments in the environment
                for (name, ty) in header.args {
                    let var_id = env.fresh_variable_id();
                    let span = name.span();
                    let typed_arg = Spanned::new(span, TypedArg { name: name.clone(), ty: ty.clone(), var_id });
                    let data = VariableData::new(ty, name.span, var_id);
                    env.overwrite_data(name.value, data);
                    typed_args.push(typed_arg);
                }

                let typed_block = typecheck_block(body, env, &header.return_type)?;

                Ok((typed_args, typed_block))
            })?;

            TypedDecl::Fn {
                return_type: header.return_type,
                name,
                args,
                body,
            }
        }

        ast::Decl::Var { ty, items } => {
            let ty = resolve_type(&ty, env)?;
            if ty == Type::Void {
                return Err(TypecheckingError::void_variable(span));
            }

            let typed_items = items
                .into_iter()
                .map(|item| {
                    let span = item.span();
                    let item = item.into_value();
                    let ident = item.ident;
                    let ty = ty.clone();
                    let typed_init = if let Some(init) = item.init {
                        let typed_init = typecheck_expr(init, env)?;
                        ensure_type(ty.clone(), &typed_init.value().ty, typed_init.span())?;
                        Some(typed_init)
                    } else {
                        None
                    };
                    let var_id = env.fresh_variable_id();
                    let data = VariableData::new(ty.clone(), span.clone(), var_id);
                    env.insert_data(ident.value.clone(), data)?;
                    Ok(Spanned::new(
                        span,
                        TypedItem {
                            ident,
                            ty,
                            var_id,
                            init: typed_init,
                        },
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;

            TypedDecl::Var {
                ty,
                items: typed_items,
            }
        }
    };
    Ok(Spanned::new(span, typed_decl))
}

fn typecheck_stmt(
    stmt: impl SpannedAst<Inner = ast::Stmt>,
    env: &mut Environment,
    expected_return_type: &Type,
) -> Result<Spanned<TypedStmt>, TypecheckingError> {
    let span = stmt.span();
    let typed_stmt = match stmt.into_value() {
        ast::Stmt::Empty => TypedStmt::Empty,
        ast::Stmt::Block(block) => {
            let typed_block =
                env.with_local(|env| typecheck_block(block, env, expected_return_type))?;
            TypedStmt::Block(typed_block)
        }
        ast::Stmt::Decl(decl) => {
            let typed_decl = typecheck_decl(decl, env)?;
            TypedStmt::Decl(typed_decl)
        }
        ast::Stmt::Assignment { target, expr } => {
            let target_type = env
                .get_data(target.value())
                .ok_or_else(|| {
                    TypecheckingError::undefined_variable(target.value().clone(), target.span())
                })?
                .clone();
            let typed_expr = typecheck_expr(expr, env)?;
            let expr_ty = typed_expr.value().ty.clone();
            ensure_type(target_type.ty.clone(), &expr_ty, typed_expr.span())?;
            let target_id = target_type.id;
            TypedStmt::Assignment {
                target,
                target_id,
                expr: typed_expr,
            }
        }
        ast::Stmt::Return(init) => {
            if let Some(init) = init {
                let typed_init = typecheck_expr(init, env)?;
                let init_ty = &typed_init.value().ty;
                ensure_type(expected_return_type.clone(), init_ty, typed_init.span())?;
                if expected_return_type == &Type::Void {
                    return Err(TypecheckingError::void_return(typed_init.span()));
                }
                TypedStmt::Return(Some(typed_init))
            } else {
                ensure_type(expected_return_type, &Type::Void, span.clone())?;
                TypedStmt::Return(None)
            }
        }
        ast::Stmt::If {
            cond,
            then,
            otherwise,
        } => {
            let typed_cond = typecheck_expr(cond, env)?;
            let cond_ty = typed_cond.value().ty.clone();
            ensure_type(Type::Bool, &cond_ty, typed_cond.span())?;
            let typed_then = typecheck_stmt(then, env, expected_return_type)?;
            let typed_otherwise = if let Some(otherwise) = otherwise {
                Some(Box::new(typecheck_stmt(
                    otherwise,
                    env,
                    expected_return_type,
                )?))
            } else {
                None
            };
            TypedStmt::If {
                cond: typed_cond,
                then: Box::new(typed_then),
                otherwise: typed_otherwise,
            }
        }
        ast::Stmt::While { cond, body } => {
            let typed_cond = typecheck_expr(cond, env)?;
            let cond_ty = typed_cond.value().ty.clone();
            ensure_type(Type::Bool, &cond_ty, typed_cond.span())?;
            let typed_body = typecheck_stmt(body, env, expected_return_type)?;
            TypedStmt::While {
                cond: typed_cond,
                body: Box::new(typed_body),
            }
        }
        ast::Stmt::Expr(expr) => {
            let typed_expr = typecheck_expr(expr, env)?;
            TypedStmt::Expr(typed_expr)
        }
        ast::Stmt::Incr(target) => {
            let typed_target = typecheck_incr_decr_target(target, env)?;
            TypedStmt::Incr(typed_target)
        }
        ast::Stmt::Decr(target) => {
            let typed_target = typecheck_incr_decr_target(target, env)?;
            TypedStmt::Decr(typed_target)
        }
    };
    Ok(Spanned::new(span, typed_stmt))
}

fn typecheck_incr_decr_target(
    target: impl SpannedAst<Inner = ast::Expr>,
    env: &mut Environment,
) -> Result<Spanned<TypedExpr>, TypecheckingError> {
    // TODO: Lvalues...
    let typed_target = typecheck_expr(target, env)?;
    if let TypedExprKind::Variable(ident, _) = &typed_target.value().expr {
        let target_type = env.get_type(ident).ok_or_else(|| {
            TypecheckingError::undefined_variable(ident.clone(), typed_target.span())
        })?;
        ensure_type(Type::Int, &target_type, typed_target.span())?;
        Ok(typed_target)
    } else {
        Err(TypecheckingError::invalid_lvalue(typed_target.span()))?
    }
}

pub fn typecheck_program(
    program: ast::Program,
) -> Result<(TypedProgram, ReadyEnvironment), Vec<TypecheckingError>> {
    let mut errors = Vec::new();
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
                let header = resolve_function_header(return_type, args, &env);
                match header {
                    Ok(header) => {
                        let var_id = env.fresh_variable_id();
                        let data =
                            VariableData::new(header.function_type, name.span.clone(), var_id);
                        let result = env.insert_data(name.value.clone(), data);
                        if let Err(err) = result {
                            errors.push(err);
                        }
                    }
                    Err(err) => {
                        errors.push(err);
                    }
                }
            }
            ast::Decl::Var { .. } => {
                panic!("Global variables are not supported");
            }
        }
    }

    let mut typed_decls = Vec::new();

    // Typecheck bodies
    for decl in program.0 {
        match typecheck_decl(decl, &mut env) {
            Ok(decl) => typed_decls.push(decl),
            Err(err) => errors.push(err),
        }
    }

    // Check if the main function exists and has the correct type
    let main_ident = ast::Ident::new("main".to_string());
    if let Some(main_type) = env.get_type(&main_ident) {
        if let Err(err) = ensure_type(
            Type::Function(Vec::new(), Box::new(Type::Int)),
            &main_type,
            env.get_span(&main_ident).unwrap().clone(),
        ) {
            errors.push(err);
        }
    } else {
        errors.push(TypecheckingError::no_main(Span::default()));
    }

    // Return analysis
    for decl in &typed_decls {
        if let Err(err) = top_level_return_analysis(&decl.value) {
            errors.push(err);
        }
    }

    if errors.is_empty() {
        Ok((TypedProgram(typed_decls), env.ready()))
    } else {
        Err(errors)
    }
}
