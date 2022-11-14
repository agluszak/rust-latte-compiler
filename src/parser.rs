use chumsky::error::Simple;
use chumsky::{Error, Parser, select};
use chumsky::prelude::{filter_map, just, nested_delimiters, recursive};
use crate::ast::{BinaryOp, Expr, Literal, Program, Ident, Type};
use crate::lexer::{Ctrl, Op, Spanned, Token};
use crate::lexer::Ctrl::{LBrace, LBracket, LParen, RBrace, RBracket, RParen};

fn binary_expr_parser(lower_precedence: impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone,
                      op: impl Parser<Token, Spanned<BinaryOp>, Error = Simple<Token>> + Clone) -> impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone {
    lower_precedence.clone().then(op.then(lower_precedence).repeated())
        .foldl(|lhs, (op, rhs)| {
            let span = lhs.span.start..rhs.span.end;
            Spanned::new(span, Expr::Binary { lhs: Box::new(lhs), op, rhs: Box::new(rhs) })
        })
}

fn expr_parser() -> impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> {
    recursive(|expr| {
        let literal = chumsky::primitive::filter_map(move |span, x| match x {
            Token::Bool(b) => Ok(Expr::Literal(Literal::Bool(b))),
            Token::Str(s) => Ok(Expr::Literal(Literal::String(s))),
            Token::Num(n) =>
                n.parse()
                    .map(|n| Expr::Literal(Literal::Int(n)))
                    .map_err(|e| Simple::custom(span, format!("invalid number: {}", e))),

            _ => Err(Error::expected_input_found(span, None, Some(x))),
        })
            .labelled("literal");

        let ident = select! {
            Token::Ident(s) => Ident(s),
        }.labelled("identifier");

        let variable= ident
            .map_with_span(|ident, span| Expr::Variable(ident))
            .labelled("variable");

        // A list of expressions
        let items = expr
            .clone()
            .separated_by(just(Token::Ctrl(Ctrl::Comma)))
            .allow_trailing();

        let atom = literal
            .or(variable)
            .map_with_span(|expr, span| Spanned::new(span, expr))
            .or(expr.clone().delimited_by(just(Token::Ctrl(LParen)), just(Token::Ctrl(RParen))))
            .recover_with(nested_delimiters(Token::Ctrl(LParen), Token::Ctrl(RParen),
            [(Token::Ctrl(LBrace), Token::Ctrl(RBrace)),
                (Token::Ctrl(LBracket), Token::Ctrl(RBracket))], |span| Spanned::new(span, Expr::Error)))
            .labelled("atom");

        let call = atom.then(
            items.delimited_by(just(Token::Ctrl(LParen)), just(Token::Ctrl(RParen)))
                .map_with_span(|arg, span| Spanned::new(span, arg))
                .repeated()).foldl(|target, args| {
            let span = target.span.start..args.span.end;
            Spanned::new(span, Expr::Application {
                target: Spanned::new(target.span, Box::new(target.value)),
                args: args.value
            })});

        let product_op = select! {
            Token::Op(Op::Star) => BinaryOp::Mul,
            Token::Op(Op::Slash) => BinaryOp::Div,
            Token::Op(Op::Percent) => BinaryOp::Mod,
        }
            .labelled("product operator");

        let product = binary_expr_parser(call, product_op.map_with_span(|e, s| Spanned::new(s, e)));


        let sum_op = select! {
            Token::Op(Op::Plus) => BinaryOp::Add,
            Token::Op(Op::Minus) => BinaryOp::Sub,
        }
            .labelled("sum operator");

        let sum = binary_expr_parser(product, sum_op.map_with_span(|e, s| Spanned::new(s, e)));

        let comparison_op = select! {
            Token::Op(Op::Equal) => BinaryOp::Eq,
            Token::Op(Op::BangEqual) => BinaryOp::Neq,
            Token::Op(Op::Less) => BinaryOp::Lt,
            Token::Op(Op::LessEqual) => BinaryOp::Lte,
            Token::Op(Op::Greater) => BinaryOp::Gt,
            Token::Op(Op::GreaterEqual) => BinaryOp::Gte,
        }
            .labelled("comparison operator");

        let comparison = binary_expr_parser(sum, comparison_op.map_with_span(|e, s| Spanned::new(s, e)));

        comparison
    })
}

// fn parser() -> impl Parser<Spanned<Token>, Program, Error = Simple<Token>> {
//
// }

#[cfg(test)]
mod tests {
    use crate::lexer::{Ctrl, Op, Token};
    use crate::parser::expr_parser;
    use chumsky::{Parser, Stream};

    #[test]
    fn mul_todo() {
        let input = "a + b(3, 1 % 2) * c";
        let input_len = input.len();
        let mut lexer = crate::lexer::lexer();
        let tokens = lexer.parse(input).unwrap();
        let stream = Stream::from_iter(input_len..input_len+1, tokens.into_iter());
        let result = expr_parser().parse(stream);
        println!("{:?}", result);
    }
}