use std::fmt::{Display, Formatter};
use std::ops::Range;

use chumsky::prelude::*;

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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Token {
    Bool(bool),
    Num(String),
    Str(String),
    Op(Op),
    Ctrl(Ctrl),
    Ident(String),
    If,
    Else,
    While,
    Return,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ctrl {
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Bang,
    Equal,
    EqualEqual,
    BangEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    AmpersandAmpersand,
    PipePipe,
    PlusPlus,
    MinusMinus,
}

pub fn lexer() -> impl Parser<char, Vec<(Token, Span)>, Error = Simple<char>> {
    // A parser for decimal numbers (only digits)
    let number = text::digits(10).map(Token::Num).labelled("number");

    let escape = just('\\').ignore_then(
        just('\\')
            .or(just('/'))
            .or(just('"'))
            .or(just('b').to('\x08'))
            .or(just('f').to('\x0C'))
            .or(just('n').to('\n'))
            .or(just('r').to('\r'))
            .or(just('t').to('\t'))
            .or(just('u').ignore_then(
                filter(|c: &char| c.is_ascii_hexdigit())
                    .repeated()
                    .exactly(4)
                    .collect::<String>()
                    .validate(|digits, span, emit| {
                        char::from_u32(u32::from_str_radix(&digits, 16).unwrap()).unwrap_or_else(
                            || {
                                emit(Simple::custom(span, "invalid unicode character"));
                                '\u{FFFD}' // unicode replacement character
                            },
                        )
                    }),
            )),
    );

    // A parser for strings (anything between double quotes)
    let string = just('"')
        .ignore_then(filter(|c| *c != '\\' && *c != '"').or(escape).repeated())
        .then_ignore(just('"'))
        .collect::<String>()
        .map(Token::Str)
        .labelled("string");

    // A parser for operators
    let operator = just('+')
        .then_ignore(just('+'))
        .to(Op::PlusPlus)
        .or(just('+').to(Op::Plus))
        .or(just('-').then_ignore(just('-')).to(Op::MinusMinus))
        .or(just('-').to(Op::Minus))
        .or(just('*').to(Op::Star))
        .or(just('/').to(Op::Slash))
        .or(just('%').to(Op::Percent))
        .or(just('!').then_ignore(just('=')).to(Op::BangEqual))
        .or(just('!').to(Op::Bang))
        .or(just('=').then_ignore(just('=')).to(Op::EqualEqual))
        .or(just('=').to(Op::Equal))
        .or(just('<').then_ignore(just('=')).to(Op::LessEqual))
        .or(just('<').to(Op::Less))
        .or(just('>').then_ignore(just('=')).to(Op::GreaterEqual))
        .or(just('>').to(Op::Greater))
        .or(just('&').then_ignore(just('&')).to(Op::AmpersandAmpersand))
        .or(just('|').then_ignore(just('|')).to(Op::PipePipe))
        .map(Token::Op)
        .labelled("operator");

    // A parser for control characters
    let control = just('{')
        .to(Ctrl::LBrace)
        .or(just('}').to(Ctrl::RBrace))
        .or(just('(').to(Ctrl::LParen))
        .or(just(')').to(Ctrl::RParen))
        .or(just('[').to(Ctrl::LBracket))
        .or(just(']').to(Ctrl::RBracket))
        .or(just(',').to(Ctrl::Comma))
        .or(just(';').to(Ctrl::Semicolon))
        .map(Token::Ctrl)
        .labelled("control character");

    // A parser for identifiers and keywords
    let identifier_or_keyword = text::ident()
        .map(|s: String| match s.as_str() {
            "true" => Token::Bool(true),
            "false" => Token::Bool(false),
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "return" => Token::Return,
            _ => Token::Ident(s),
        })
        .labelled("identifier or keyword");

    // A single token can be any of the above
    let token = number
        .or(string)
        .or(operator)
        .or(control)
        .or(identifier_or_keyword)
        .recover_with(skip_then_retry_until([]));

    let single_line_comment = just("#")
        .or(just("//"))
        .then(take_until(just("\n")))
        .ignored();

    let multi_line_comment = just("/*").then(take_until(just("*/"))).ignored();

    let comment = single_line_comment
        .or(multi_line_comment)
        .padded()
        .labelled("comment");

    token
        .map_with_span(|tok, span| (tok, span))
        .padded_by(comment.repeated())
        .padded()
        .repeated()
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Bool(b) => write!(f, "{b}"),
            Token::Num(n) => write!(f, "{n}"),
            Token::Str(s) => write!(f, "\"{s}\""),
            Token::Op(op) => write!(f, "{op}"),
            Token::Ctrl(ctrl) => write!(f, "{ctrl}"),
            Token::Ident(ident) => write!(f, "{ident}"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::While => write!(f, "while"),
            Token::Return => write!(f, "return"),
        }
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Plus => write!(f, "+"),
            Op::Minus => write!(f, "-"),
            Op::Star => write!(f, "*"),
            Op::Slash => write!(f, "/"),
            Op::Percent => write!(f, "%"),
            Op::Bang => write!(f, "!"),
            Op::Equal => write!(f, "="),
            Op::EqualEqual => write!(f, "=="),
            Op::BangEqual => write!(f, "!="),
            Op::Less => write!(f, "<"),
            Op::LessEqual => write!(f, "<="),
            Op::Greater => write!(f, ">"),
            Op::GreaterEqual => write!(f, ">="),
            Op::AmpersandAmpersand => write!(f, "&&"),
            Op::PipePipe => write!(f, "||"),
            Op::PlusPlus => write!(f, "++"),
            Op::MinusMinus => write!(f, "--"),
        }
    }
}

impl Display for Ctrl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Ctrl::LBrace => write!(f, "{{"),
            Ctrl::RBrace => write!(f, "}}"),
            Ctrl::LParen => write!(f, "("),
            Ctrl::RParen => write!(f, ")"),
            Ctrl::LBracket => write!(f, "["),
            Ctrl::RBracket => write!(f, "]"),
            Ctrl::Comma => write!(f, ","),
            Ctrl::Semicolon => write!(f, ";"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chumsky::stream::Stream;

    macro_rules! lexer_tests {
            ($($name:ident),*) => {
            $(
                #[test]
                fn $name() {
                    let input = include_str!(concat!("../inputs/", stringify!($name), ".lat"));
                    let stream = Stream::from(input);
                    let tokens = lexer().parse(stream);

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
