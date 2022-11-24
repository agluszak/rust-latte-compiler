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

    // A parser for strings (anything between double quotes)
    let string = just('"')
        .ignore_then(filter(|c| *c != '"').repeated())
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

    let comment = single_line_comment.or(multi_line_comment).padded();

    token
        .map_with_span(|tok, span| (tok, span))
        .padded_by(comment.repeated())
        .padded()
        .repeated()
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
                    let mut stream = Stream::from(input);
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

    lexer_tests!(comments, hello_world, simple);
}
