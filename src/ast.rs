use crate::lexer::Spanned;
use std::fmt;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program(pub Vec<Spanned<Decl>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block(pub Vec<Spanned<Stmt>>);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Error,
    Empty,
    Block(Spanned<Block>),
    Decl(Spanned<Decl>),
    Assignment {
        target: Spanned<Ident>,
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
        target: Box<Spanned<Expr>>,
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
    Var {
        ty: Spanned<Type>,
        items: Vec<Spanned<Item>>,
    },
    Fn {
        return_type: Spanned<Type>,
        name: Spanned<Ident>,
        args: Vec<Spanned<Arg>>,
        body: Spanned<Block>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub ident: Spanned<Ident>,
    pub init: Option<Spanned<Expr>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arg {
    pub ty: Spanned<Type>,
    pub name: Spanned<Ident>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ident(pub String); // TODO: interned

impl Ident {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Type(pub String); // TODO: interned

impl<T> Display for Spanned<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for def in &self.0 {
            write!(f, "{}", def)?;
        }
        Ok(())
    }
}

impl Display for Stmt {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Stmt::Error => write!(f, "error"),
            Stmt::Empty => write!(f, ";"),
            Stmt::Block(stmts) => {
                write!(f, "{{")?;
                for stmt in &stmts.value.0 {
                    write!(f, "{}", stmt)?;
                }
                write!(f, "}}")
            }
            Stmt::Assignment {
                target: ident,
                expr,
            } => write!(f, "{} = {};", ident, expr),
            Stmt::Return(expr) => match expr {
                Some(expr) => write!(f, "return {};", expr),
                None => write!(f, "return;"),
            },
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                write!(f, "if ({}) {}", cond, then)?;
                if let Some(otherwise) = otherwise {
                    write!(f, " else {}", otherwise)?;
                }
                Ok(())
            }
            Stmt::While { cond, body } => write!(f, "while ({}) {}", cond, body),
            Stmt::Expr(expr) => write!(f, "{};", expr),
            Stmt::Incr(expr) => write!(f, "{}++;", expr),
            Stmt::Decr(expr) => write!(f, "{}--;", expr),
            Stmt::Decl(decl) => write!(f, "{}", decl),
        }
    }
}

impl Display for Decl {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Decl::Var { ty, items } => {
                write!(f, "{} ", ty)?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, ";")
            }
            Decl::Fn {
                return_type: ty,
                name: ident,
                args,
                body,
            } => {
                write!(f, "{} {}(", ty, ident)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ") {{")?;
                for stmt in &body.value.0 {
                    write!(f, "{}", stmt)?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Expr::Error => write!(f, "error"),
            Expr::Variable(ident) => write!(f, "{}", ident),
            Expr::Literal(lit) => write!(f, "{}", lit),
            Expr::Binary { lhs, op, rhs } => write!(f, "({} {} {})", lhs, op, rhs),
            Expr::Unary { op, expr } => write!(f, "({}{})", op, expr),
            Expr::Application { target, args } => {
                write!(f, "{}(", target)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Literal::Int(i) => write!(f, "{}", i),
            Literal::String(s) => write!(f, "\"{}\"", s),
            Literal::Bool(b) => write!(f, "{}", b),
        }
    }
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
        }
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Mod => write!(f, "%"),
            BinaryOp::Eq => write!(f, "=="),
            BinaryOp::Neq => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::Lte => write!(f, "<="),
            BinaryOp::Gte => write!(f, ">="),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
        }
    }
}

impl Display for Arg {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} {}", self.name, self.ty)
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.ident)?;
        if let Some(init) = &self.init {
            write!(f, " = {}", init)?;
        }
        Ok(())
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for Ident {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
