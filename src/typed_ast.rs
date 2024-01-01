use crate::ast::{BinaryOp, Ident, Literal, UnaryOp};
use crate::lexer::Spanned;
use crate::typechecker::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VariableId(u32);

impl VariableId {
    pub fn new(id: u32) -> Self {
        VariableId(id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedProgram(pub Vec<Spanned<TypedDecl>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedBlock(pub Vec<Spanned<TypedStmt>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedStmt {
    Empty,
    Block(Spanned<TypedBlock>),
    Decl(Spanned<TypedDecl>),
    Assignment {
        target: Spanned<Ident>,
        target_id: VariableId,
        expr: Spanned<TypedExpr>,
    },
    Return(Option<Spanned<TypedExpr>>),
    If {
        cond: Spanned<TypedExpr>,
        then: Box<Spanned<TypedStmt>>,
        otherwise: Option<Box<Spanned<TypedStmt>>>,
    },
    While {
        cond: Spanned<TypedExpr>,
        body: Box<Spanned<TypedStmt>>,
    },
    Expr(Spanned<TypedExpr>),
    Incr(Spanned<TypedExpr>),
    Decr(Spanned<TypedExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedExpr {
    pub expr: TypedExprKind,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedExprKind {
    Variable(Ident, VariableId),
    Literal(Literal),
    Binary {
        lhs: Box<Spanned<TypedExpr>>,
        op: Spanned<BinaryOp>,
        rhs: Box<Spanned<TypedExpr>>,
    },
    Unary {
        op: Spanned<UnaryOp>,
        expr: Box<Spanned<TypedExpr>>,
    },
    Application {
        target: Box<Spanned<TypedExpr>>,
        args: Vec<Spanned<TypedExpr>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedDecl {
    Var {
        ty: Type,
        items: Vec<Spanned<TypedItem>>,
    },
    Fn {
        return_type: Type,
        name: Spanned<Ident>,
        args: Vec<Spanned<TypedArg>>, // TODO: or params?
        body: Spanned<TypedBlock>,
    },
}

impl TypedDecl {
    pub fn ty(&self) -> Type {
        match self {
            TypedDecl::Var { ty, .. } => ty.clone(),
            TypedDecl::Fn {
                return_type, args, ..
            } => {
                let mut arg_types = Vec::new();
                for arg in args {
                    arg_types.push(arg.value.ty.clone());
                }
                Type::Function(arg_types, Box::new(return_type.clone()))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedItem {
    pub ty: Type,
    pub ident: Spanned<Ident>,
    pub var_id: VariableId,
    pub init: Option<Spanned<TypedExpr>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedArg {
    pub ty: Type,
    pub name: Spanned<Ident>,
    pub var_id: VariableId,
}
