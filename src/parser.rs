use crate::ast::{Arg, BinaryOp, Block, Decl, Expr, Ident, Item, Literal, Program, Stmt, Type};
use crate::lexer::Ctrl::{LBrace, LBracket, LParen, RBrace, RBracket, RParen, Semicolon};
use crate::lexer::Op::Equal;
use crate::lexer::{Ctrl, Op, Spanned, Token};
use chumsky::error::Simple;
use chumsky::prelude::{filter_map, just, nested_delimiters, recursive};
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
                        target: Spanned::new(target.span, Box::new(target.value)),
                        args: args.value,
                    },
                )
            });

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

        let comparison =
            binary_expr_parser(sum, comparison_op.map_with_span(|e, s| Spanned::new(s, e)));

        comparison
    })
}

fn block_parser() -> impl Parser<Token, Spanned<Block>, Error = Simple<Token>> {
    stmt_parser()
        .delimited_by(just(Token::Ctrl(LBrace)), just(Token::Ctrl(RBrace)))
        // Attempt to recover anything that looks like a block but contains errors
        .recover_with(nested_delimiters(
            Token::Ctrl(LBrace),
            Token::Ctrl(RBrace),
            [
                (Token::Ctrl(LParen), Token::Ctrl(RParen)),
                (Token::Ctrl(LBracket), Token::Ctrl(RBracket)),
            ],
            |span| Spanned::new(span, Stmt::Error),
        ))
        .repeated()
        .collect::<Vec<_>>()
        .map_with_span(|stmts, span| Spanned::new(span, Block(stmts)))
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
        .then(just(Token::Op(Op::Equal)).ignore_then(expr_parser().or_not()))
        .map_with_span(|(ident, init), span| Spanned::new(span, Item { ident, init }))
}

fn arg_parser() -> impl Parser<Token, Spanned<Arg>, Error = Simple<Token>> {
    type_parser()
        .then(ident_parser())
        .map_with_span(|(ty, name), span| Spanned::new(span, Arg { ty, name }))
}

fn decl_parser() -> impl Parser<Token, Spanned<Decl>, Error = Simple<Token>> {
    let var = type_parser()
        .then(
            item_parser()
                .separated_by(just(Token::Ctrl(Ctrl::Comma)))
                .at_least(1),
        )
        .map_with_span(|(ty, items), span| Spanned::new(span, Decl::Var { ty, items }))
        .labelled("variable declaration");

    let fun = type_parser()
        .then(ident_parser())
        .then(
            arg_parser()
                .separated_by(just(Token::Ctrl(Ctrl::Comma)))
                .delimited_by(just(Token::Ctrl(LParen)), just(Token::Ctrl(RParen))),
        )
        .then(block_parser())
        .map_with_span(|(((ty, ident), args), body), span| {
            Spanned::new(
                span,
                Decl::Fn {
                    ty,
                    ident,
                    args,
                    body,
                },
            )
        });

    var.or(fun)
}

// TODO: make sure it consumes all input
fn stmt_parser() -> impl Parser<Token, Spanned<Stmt>, Error = Simple<Token>> {
    let stmt = recursive(|stmt| {
        let expr_stmt = expr_parser()
            .then_ignore(just(Token::Ctrl(Semicolon)))
            .map_with_span(|expr, span| Spanned::new(span, Stmt::Expr(expr)))
            .labelled("expression statement");

        let block =
            // stmt
            // .clone()
            // .repeated()
            // .delimited_by(just(Token::Ctrl(LBrace)), just(Token::Ctrl(RBrace)))
            // .collect::<Vec<_>>()
            block_parser()
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

        // let decl = type_parser()
        //     .then(
        //         ident_parser()
        //             .separated_by(just(Token::Ctrl(Ctrl::Comma)))
        //             .at_least(1),
        //     )
        //     .then_ignore(just(Token::Ctrl(Semicolon)))
        //     .map_with_span(|(ty, idents), span| {
        //         let decl = Stmt::Decl { ty, idents };
        //         Spanned::new(span, decl)
        //     })
        //     .labelled("declaration");

        expr_stmt.or(if_).or(while_).or(block)
    });
    stmt
}

#[cfg(test)]
mod tests {
    use crate::lexer::{Ctrl, Op, Token};
    use crate::parser::{expr_parser, stmt_parser};
    use chumsky::{Parser, Stream};

    #[test]
    fn mul_todo() {
        let input = "a + b(3, 1 % 2) * c";
        let input_len = input.len();
        let mut lexer = crate::lexer::lexer();
        let tokens = lexer.parse(input).unwrap();
        let stream = Stream::from_iter(input_len..input_len + 1, tokens.into_iter());
        let result = expr_parser().parse(stream);
        println!("{:?}", result);
    }

    #[test]
    fn stmts() {
        let input = r#"
        if (a) { 
            b; 
        } else { 
            while (true) { 
                int test;
                c; 
            } 
        }"#;
        // let input = "";
        // let input = "if (a) {} else { a; }";
        let input_len = input.len();
        let mut lexer = crate::lexer::lexer();
        let tokens = lexer.parse(input).unwrap();
        println!("{:?}", tokens);
        let stream = Stream::from_iter(input_len..input_len + 1, tokens.into_iter());
        let result = stmt_parser().parse(stream);
        println!("{}", result.unwrap());
    }
}
