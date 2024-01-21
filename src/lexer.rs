use logos::Logos;
use std::fmt::Display;
use std::num::ParseIntError;
use std::ops::Range;

pub type Span = Range<usize>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    pub span: Span,
    pub value: T,
}

impl<T> Spanned<T> {
    pub fn new(span: Span, value: T) -> Self {
        Self { span, value }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LexingError {
    NumberTooLong,
    UnexpectedEscape(char),
    InvalidUnicodeEscape(String),
    InvalidUnicodeChar(u32),
    UnclosedEscape,
    #[default]
    Other,
}

impl From<ParseIntError> for LexingError {
    fn from(_: ParseIntError) -> Self {
        LexingError::NumberTooLong
    }
}

fn escape_str(s: &str) -> Result<String, LexingError> {
    let mut result = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(c) = chars.next() {
                    match c {
                        'n' => result.push('\n'),
                        'r' => result.push('\r'),
                        't' => result.push('\t'),
                        '0' => result.push('\0'),
                        '\\' => result.push('\\'),
                        '"' => result.push('"'),
                        '\'' => result.push('\''),
                        'u' => {
                            let mut code = String::new();
                            for _ in 0..4 {
                                if let Some(c) = chars.next() {
                                    code.push(c);
                                } else {
                                    return Err(LexingError::InvalidUnicodeEscape(code));
                                }
                            }
                            let code = u32::from_str_radix(&code, 16)
                                .map_err(|_| LexingError::InvalidUnicodeEscape(code))?;
                            result.push(
                                std::char::from_u32(code)
                                    .ok_or(LexingError::InvalidUnicodeChar(code))?,
                            );
                        }
                        _ => return Err(LexingError::UnexpectedEscape(c)),
                    }
                } else {
                    return Err(LexingError::UnclosedEscape);
                }
            }
            _ => result.push(c),
        }
    }
    Ok(result)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Logos)]
#[logos(error = LexingError)]
pub enum Token {
    #[token("new")]
    New,
    #[token("int")]
    Int,
    #[token("string")]
    String,
    #[token("bool")]
    Bool,
    #[token("void")]
    Void,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[regex(r"[0-9]+", |lex| lex.slice().parse())]
    Num(i64),
    #[regex(r#""([^"\\]|\\.)*""#, |lex| escape_str(lex.slice()))]
    Str(String),
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_owned())]
    Ident(String),
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
    #[regex(r"[ \t\n\f]+", logos::skip)] // whitespace
    #[regex(r"//[^\n\r]*", logos::skip)] // single line comment
    #[regex(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", logos::skip)] // multi line comment
    Ignored,
}

pub struct Lexer<'a> {
    lexer: logos::Lexer<'a, Token>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            lexer: Token::lexer(input),
        }
    }
}

impl<'input> Iterator for Lexer<'input> {
    type Item = Result<(usize, Token, usize), Spanned<LexingError>>;

    fn next(&mut self) -> Option<Self::Item> {
        let token = self.lexer.next()?;
        let start = self.lexer.span().start;
        let end = self.lexer.span().end;

        Some(
            token
                .map(|token| (start, token, end))
                .map_err(|err| Spanned::new(start..end, err)),
        )
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! lexer_tests {
            ($($name:ident),*) => {
            $(
                #[test]
                fn $name() {
                    let input = include_str!(concat!("../inputs/", stringify!($name), ".lat"));
                    let tokens = Lexer::new(input).collect::<Vec<_>>();

                    insta::with_settings!({
                        description => input,
                        omit_expression => true
                    }, {
                        insta::assert_debug_snapshot!(tokens);
                    });
                }
            )*
            };
    }

    lexer_tests!(_lexer_comments, hello_world, _lexer_simple);
}
