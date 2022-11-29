extern crate core;

use crate::errors::{parsing_reports, typechecking_reports};
use crate::input::read_input;
use crate::lexer::lexer;
use crate::parser::program_parser;
use crate::typechecker::typecheck_program;
use ariadne::Source;
use chumsky::Parser;
use chumsky::Stream;

use std::process::ExitCode;

mod ast;
mod dfa;
mod errors;
mod input;
mod lexer;
mod parser;
mod typechecker;

fn main() -> ExitCode {
    let mut error_reports = Vec::new();

    let input = {
        match read_input() {
            Ok(input) => input,
            Err(err) => {
                eprintln!("Error: {}", err);
                return ExitCode::FAILURE;
            }
        }
    };

    // Lex
    let (tokens, lexer_errs) = lexer().parse_recovery(input.source.as_str());
    error_reports.extend(parsing_reports(lexer_errs));

    if let Some(tokens) = tokens {
        let input_len = input.source.len();
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

    if !error_reports.is_empty() {
        println!("ERROR");
        for report in error_reports {
            report.eprint(Source::from(&input.source)).unwrap_or(());
        }
        ExitCode::FAILURE
    } else {
        println!("OK");
        ExitCode::SUCCESS
    }
}
