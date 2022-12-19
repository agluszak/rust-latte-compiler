use logos::Logos;
use std::fmt::Display;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Logos)]
pub enum Token<'input> {
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[regex(r"[0-9]+", |lex| lex.slice().parse())]
    Num(i64),
    #[regex(r#""([^"\\]|\\.)*""#)]
    Str(&'input str),
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Ident(&'input str),
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("while")]
    While,
    #[token("return")]
    Return,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("!")]
    Bang,
    #[token("=")]
    Equal,
    #[token("==")]
    EqualEqual,
    #[token("!=")]
    BangEqual,
    #[token("<")]
    Less,
    #[token("<=")]
    LessEqual,
    #[token(">")]
    Greater,
    #[token(">=")]
    GreaterEqual,
    #[token("&&")]
    AmpersandAmpersand,
    #[token("||")]
    PipePipe,
    #[token("++")]
    PlusPlus,
    #[token("--")]
    MinusMinus,
    #[error]
    #[regex(r"[ \t\n\f]+", logos::skip)] // whitespace
    #[regex(r"//[^\n\r]*", logos::skip)] // single line comment
    #[regex(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", logos::skip)] // multi line comment
    Error,
}

pub struct Lexer<'a> {
    lexer: logos::Lexer<'a, Token<'a>>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            lexer: Token::lexer(input),
        }
    }
}

impl<'input> Iterator for Lexer<'input> {
    type Item = Result<(usize, Token<'input>, usize), ()>;

    fn next(&mut self) -> Option<Self::Item> {
        let token = self.lexer.next()?;
        let start = self.lexer.span().start;
        let end = self.lexer.span().end;
        Some(Ok((start, token, end)))
    }
}

impl<'input> Display for Token<'input> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
