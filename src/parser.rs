use crate::ast::{
    Arg, BinaryOp, Block, Decl, Expr, Ident, Item, Literal, Program, Stmt, Type, UnaryOp,
};
use crate::lexer::Ctrl::{LBrace, LBracket, LParen, RBrace, RBracket, RParen, Semicolon};

use crate::lexer::{Ctrl, Op, Spanned, Token};
use chumsky::error::Simple;
use chumsky::prelude::{end, just, nested_delimiters, recursive};
use chumsky::{select, Error, Parser};

fn binary_expr_parser(
    lower_precedence: impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone,
    op: impl Parser<Token, Spanned<BinaryOp>, Error = Simple<Token>> + Clone,
) -> impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone {
    lower_precedence
        .clone()
        .then(op.then(lower_precedence).repeated())
        .foldl(|lhs, (op, rhs)| {
            let span = lhs.span.start..rhs.span.end;
            Spanned::new(
                span,
                Expr::Binary {
                    lhs: Box::new(lhs),
                    op,
                    rhs: Box::new(rhs),
                },
            )
        })
}

fn unary_expr_parser(
    lower_precedence: impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone,
    op: impl Parser<Token, Spanned<UnaryOp>, Error = Simple<Token>> + Clone,
) -> impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> + Clone {
    op.repeated().then(lower_precedence).foldr(|op, rhs| {
        let span = op.span.start..rhs.span.end;
        Spanned::new(
            span,
            Expr::Unary {
                op,
                expr: Box::new(rhs),
            },
        )
    })
}

// TODO: int literal with - sign
fn expr_parser() -> impl Parser<Token, Spanned<Expr>, Error = Simple<Token>> {
    recursive(|expr| {
        let literal = chumsky::primitive::filter_map(move |span, x| match x {
            Token::Bool(b) => Ok(Expr::Literal(Literal::Bool(b))),
            Token::Str(s) => Ok(Expr::Literal(Literal::String(s))),
            Token::Num(n) => n
                .parse()
                .map(|n| Expr::Literal(Literal::Int(n)))
                .map_err(|e| Simple::custom(span, format!("invalid number: {}", e))),

            _ => Err(Error::expected_input_found(span, None, Some(x))),
        })
        .labelled("literal");

        let ident = select! {
            Token::Ident(s) => Ident(s),
        }
        .labelled("identifier");

        let variable = ident
            .map_with_span(|ident, _span| Expr::Variable(ident))
            .labelled("variable");

        // A list of expressions
        let items = expr
            .clone()
            .separated_by(just(Token::Ctrl(Ctrl::Comma)))
            .allow_trailing()
            .labelled("items");

        let atom = literal
            .or(variable)
            .map_with_span(|expr, span| Spanned::new(span, expr))
            .or(expr
                .clone()
                .delimited_by(just(Token::Ctrl(LParen)), just(Token::Ctrl(RParen))))
            .recover_with(nested_delimiters(
                Token::Ctrl(LParen),
                Token::Ctrl(RParen),
                [
                    (Token::Ctrl(LBrace), Token::Ctrl(RBrace)),
                    (Token::Ctrl(LBracket), Token::Ctrl(RBracket)),
                ],
                |span| Spanned::new(span, Expr::Error),
            ))
            .labelled("atom");

        let call = atom
            .then(
                items
                    .delimited_by(just(Token::Ctrl(LParen)), just(Token::Ctrl(RParen)))
                    .map_with_span(|arg, span| Spanned::new(span, arg))
                    .repeated(),
            )
            .foldl(|target, args| {
                let span = target.span.start..args.span.end;
                Spanned::new(
                    span,
                    Expr::Application {
                        target: Box::new(Spanned::new(target.span, target.value)),
                        args: args.value,
                    },
                )
            })
            .labelled("call");

        let unary_op = select! {
            Token::Op(Op::Minus) => UnaryOp::Neg,
            Token::Op(Op::Bang) => UnaryOp::Not,
        }
        .labelled("unary operator");

        let unary_expr = unary_expr_parser(call, unary_op.map_with_span(|e, s| Spanned::new(s, e)))
            .labelled("unary expression");

        let product_op = select! {
            Token::Op(Op::Star) => BinaryOp::Mul,
            Token::Op(Op::Slash) => BinaryOp::Div,
            Token::Op(Op::Percent) => BinaryOp::Mod,
        }
        .labelled("product operator");

        let product = binary_expr_parser(
            unary_expr,
            product_op.map_with_span(|e, s| Spanned::new(s, e)),
        );

        let sum_op = select! {
            Token::Op(Op::Plus) => BinaryOp::Add,
            Token::Op(Op::Minus) => BinaryOp::Sub,
        }
        .labelled("sum operator");

        let sum = binary_expr_parser(product, sum_op.map_with_span(|e, s| Spanned::new(s, e)));

        let comparison_op = select! {
            Token::Op(Op::EqualEqual) => BinaryOp::Eq,
            Token::Op(Op::BangEqual) => BinaryOp::Neq,
            Token::Op(Op::Less) => BinaryOp::Lt,
            Token::Op(Op::LessEqual) => BinaryOp::Lte,
            Token::Op(Op::Greater) => BinaryOp::Gt,
            Token::Op(Op::GreaterEqual) => BinaryOp::Gte,
        }
        .labelled("comparison operator");

        let comparison =
            binary_expr_parser(sum, comparison_op.map_with_span(|e, s| Spanned::new(s, e)))
                .labelled("comparison");

        let logical_and_op = select! {
            Token::Op(Op::AmpersandAmpersand) => BinaryOp::And
        }
        .labelled("logical and operator");

        let logical_and = binary_expr_parser(
            comparison,
            logical_and_op.map_with_span(|e, s| Spanned::new(s, e)),
        )
        .labelled("logical and");

        let logical_or_op = select! {
            Token::Op(Op::PipePipe) => BinaryOp::Or
        }
        .labelled("logical or operator");

        let logical_or = binary_expr_parser(
            logical_and,
            logical_or_op.map_with_span(|e, s| Spanned::new(s, e)),
        )
        .labelled("logical or");

        logical_or
    })
}

fn block_parser(
    stmt_parser: impl Parser<Token, Spanned<Stmt>, Error = Simple<Token>>,
) -> impl Parser<Token, Spanned<Block>, Error = Simple<Token>> {
    stmt_parser
        .repeated()
        .collect::<Vec<_>>()
        .delimited_by(just(Token::Ctrl(LBrace)), just(Token::Ctrl(RBrace)))
        .map_with_span(|stmts, span| Spanned::new(span, Block(stmts)))
        // Attempt to recover anything that looks like a block but contains errors
        .recover_with(nested_delimiters(
            Token::Ctrl(LBrace),
            Token::Ctrl(RBrace),
            [
                (Token::Ctrl(LParen), Token::Ctrl(RParen)),
                (Token::Ctrl(LBracket), Token::Ctrl(RBracket)),
            ],
            |span| Spanned::new(span, Block(Vec::new())),
        ))
}

fn type_parser() -> impl Parser<Token, Spanned<Type>, Error = Simple<Token>> {
    select! { Token::Ident(ident) => Type(ident) }
        .map_with_span(|tpe, span| Spanned::new(span, tpe))
        .labelled("type")
}

fn ident_parser() -> impl Parser<Token, Spanned<Ident>, Error = Simple<Token>> {
    select! { Token::Ident(ident) => Ident(ident) }
        .map_with_span(|ident, span| Spanned::new(span, ident))
        .labelled("identifier")
}

fn item_parser() -> impl Parser<Token, Spanned<Item>, Error = Simple<Token>> {
    ident_parser()
        .then(
            just(Token::Op(Op::Equal))
                .ignore_then(expr_parser())
                .or_not(),
        )
        .map_with_span(|(ident, init), span| Spanned::new(span, Item { ident, init }))
}

fn arg_parser() -> impl Parser<Token, Spanned<Arg>, Error = Simple<Token>> {
    type_parser()
        .then(ident_parser())
        .map_with_span(|(ty, name), span| Spanned::new(span, Arg { ty, name }))
}

fn decl_parser(
    stmt_parser: impl Parser<Token, Spanned<Stmt>, Error = Simple<Token>>,
) -> impl Parser<Token, Spanned<Decl>, Error = Simple<Token>> {
    let var = type_parser()
        .then(
            item_parser()
                .separated_by(just(Token::Ctrl(Ctrl::Comma)))
                .at_least(1),
        )
        .then_ignore(just(Token::Ctrl(Ctrl::Semicolon)))
        .map_with_span(|(ty, items), span| Spanned::new(span, Decl::Var { ty, items }))
        .labelled("variable declaration");

    let fun = type_parser()
        .then(ident_parser())
        .then(
            arg_parser()
                .separated_by(just(Token::Ctrl(Ctrl::Comma)))
                .delimited_by(just(Token::Ctrl(LParen)), just(Token::Ctrl(RParen))),
        )
        .then(block_parser(stmt_parser))
        .map_with_span(|(((ty, ident), args), body), span| {
            Spanned::new(
                span,
                Decl::Fn {
                    return_type: ty,
                    name: ident,
                    args,
                    body,
                },
            )
        });

    var.or(fun)
}

fn stmt_parser() -> impl Parser<Token, Spanned<Stmt>, Error = Simple<Token>> {
    let stmt = recursive(|stmt| {
        let expr_stmt = expr_parser()
            .then_ignore(just(Token::Ctrl(Semicolon)))
            .map_with_span(|expr, span| Spanned::new(span, Stmt::Expr(expr)))
            .labelled("expression statement");

        let block = block_parser(stmt.clone())
            .map_with_span(|block, span| Spanned::new(span, Stmt::Block(block)))
            .labelled("block");

        let if_ = recursive(|if_| {
            just(Token::If)
                .ignore_then(
                    expr_parser()
                        .delimited_by(just(Token::Ctrl(LParen)), just(Token::Ctrl(RParen))),
                )
                .then(stmt.clone())
                .then(just(Token::Else).ignore_then(stmt.clone()).or_not())
                .map_with_span(|((cond, then), otherwise), span| {
                    let if_stmt = Stmt::If {
                        cond,
                        then: Box::new(then),
                        otherwise: otherwise.map(Box::new),
                    };
                    Spanned::new(span, if_stmt)
                })
        })
        .labelled("if statement");

        let while_ = just(Token::While)
            .ignore_then(
                expr_parser().delimited_by(just(Token::Ctrl(LParen)), just(Token::Ctrl(RParen))),
            )
            .then(stmt.clone())
            .map_with_span(|(cond, body), span| {
                let while_stmt = Stmt::While {
                    cond,
                    body: Box::new(body),
                };
                Spanned::new(span, while_stmt)
            })
            .labelled("while statement");

        let assignment = ident_parser()
            .then_ignore(just(Token::Op(Op::Equal)))
            .then(expr_parser())
            .then_ignore(just(Token::Ctrl(Semicolon)))
            .map_with_span(|(ident, expr), span| {
                Spanned::new(
                    span,
                    Stmt::Assignment {
                        target: ident,
                        expr,
                    },
                )
            })
            .labelled("assignment statement");

        let decl =
            decl_parser(stmt).map_with_span(|decl, span| Spanned::new(span, Stmt::Decl(decl)));

        let return_ = just(Token::Return)
            .ignore_then(expr_parser().or_not())
            .then_ignore(just(Token::Ctrl(Semicolon)))
            .map_with_span(|expr, span| Spanned::new(span, Stmt::Return(expr)))
            .labelled("return statement");

        let inc = expr_parser()
            .then_ignore(just(Token::Op(Op::PlusPlus)))
            .then_ignore(just(Token::Ctrl(Semicolon)))
            .map_with_span(|ident, span| Spanned::new(span, Stmt::Incr(ident)))
            .labelled("increment statement");

        let dec = expr_parser()
            .then_ignore(just(Token::Op(Op::MinusMinus)))
            .then_ignore(just(Token::Ctrl(Semicolon)))
            .map_with_span(|ident, span| Spanned::new(span, Stmt::Decr(ident)))
            .labelled("decrement statement");

        let empty = just(Token::Ctrl(Semicolon))
            .map_with_span(|_, span| Spanned::new(span, Stmt::Empty))
            .labelled("empty statement");

        decl.or(assignment)
            .or(expr_stmt)
            .or(if_)
            .or(while_)
            .or(return_)
            .or(block)
            .or(inc)
            .or(dec)
            .or(empty)
    });
    stmt
}

pub fn program_parser() -> impl Parser<Token, Spanned<Program>, Error = Simple<Token>> {
    let decl = decl_parser(stmt_parser());
    let program = decl
        .repeated()
        .then_ignore(end())
        .map_with_span(|decls, span| Spanned::new(span, Program(decls)))
        .labelled("program");
    program
}

#[cfg(test)]
mod tests {

    use crate::parser::program_parser;

    use chumsky::{Parser, Stream};

    macro_rules! parser_tests {
            ($($name:ident),*) => {
            $(
                #[test]
                fn $name() {
                    let input = include_str!(concat!("../inputs/", stringify!($name), ".lat"));
                    let input_len = input.len();
                    let lexer = crate::lexer::lexer();
                    let tokens = lexer.parse(input).unwrap();
                    let stream = Stream::from_iter(input_len..input_len + 1, tokens.into_iter());
                    let result = program_parser().parse(stream);

                    insta::with_settings!({
                        description => input,
                        omit_expression => true
                    }, {
                        insta::assert_debug_snapshot!(result);
                    });
                }
            )*
            };
    }

    parser_tests!(_parser_ugly, hello_world, factorial);
}
