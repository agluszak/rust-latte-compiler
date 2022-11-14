use crate::lexer::Spanned;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program(Vec<Spanned<TopDef>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopDef {
    Fn {
        ty: Spanned<Type>,
        ident: Spanned<Ident>,
        args: Vec<Spanned<Arg>>,
        block: Spanned<Block>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block(Vec<Spanned<Stmt>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Block(Spanned<Block>),
    Empty,
    Assignment {
        ident: Spanned<Ident>,
        expr: Spanned<Expr>,
    },
    Return(Option<Spanned<Expr>>),
    If {
        cond: Spanned<Expr>,
        then: Box<Spanned<Stmt>>,
        otherwise: Option<Box<Spanned<Stmt>>>,
    },
    While {
        cond: Spanned<Expr>,
        body: Box<Spanned<Stmt>>,
    },
    Expr(Spanned<Expr>),
    Incr(Spanned<Expr>),
    Decr(Spanned<Expr>),

}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Error,
    Variable(Ident),
    Literal(Literal),
    Binary {
        lhs: Box<Spanned<Expr>>,
        op: Spanned<BinaryOp>,
        rhs: Box<Spanned<Expr>>,
    },
    Unary {
        op: Spanned<UnaryOp>,
        expr: Box<Spanned<Expr>>,
    },
    Application {
        target: Spanned<Box<Expr>>,
        args: Vec<Spanned<Expr>>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Int(i64),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decl {
    Decl {
        ty: Spanned<Type>,
        items: Vec<Spanned<Item>>,
    },
    Fn {
        ty: Spanned<Type>,
        ident: Spanned<Ident>,
        args: Vec<Spanned<Arg>>,
        block: Spanned<Block>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    ident: Spanned<Ident>,
    init: Option<Spanned<Expr>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arg {
    name: Spanned<Ident>,
    ty: Spanned<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ident(pub String); // TODO: interned

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type(pub String); // TODO: interned