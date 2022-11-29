extern crate core;

use crate::errors::{parsing_reports, typechecking_reports};
use crate::lexer::lexer;
use crate::parser::program_parser;
use crate::typechecker::typecheck_program;
use ariadne::Report;
use chumsky::Parser;
use chumsky::Stream;

mod ast;
mod errors;
mod lexer;
mod parser;
mod typechecker;

pub fn compile(input: &str) -> Vec<Report> {
    let mut error_reports = Vec::new();

    // Lex
    let (tokens, lexer_errs) = lexer().parse_recovery(input);
    error_reports.extend(parsing_reports(lexer_errs));

    if let Some(tokens) = tokens {
        let input_len = input.len();
        let stream = Stream::from_iter(input_len..input_len + 1, tokens.into_iter());
        // Parse
        let (ast, parser_errs) = program_parser().parse_recovery(stream);
        error_reports.extend(parsing_reports(parser_errs));
        if let Some(ast) = ast {
            if error_reports.is_empty() {
                // Typecheck
                match typecheck_program(&ast.value) {
                    Ok(_) => {}
                    Err(errs) => {
                        error_reports.extend(typechecking_reports(errs));
                    }
                }
            }
        }
    }

    error_reports
}
