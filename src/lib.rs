extern crate core;

use crate::errors::{parsing_reports, typechecking_reports};
use crate::lexer::Lexer;
use crate::parser::latte::ProgramParser;
use crate::typechecker::{typecheck_program, TypecheckingError};
use ariadne::Report;

use crate::typed_ast::TypedProgram;
use crate::typed_ast_lowering::lower_fn_decl;
use std::ops::Range;

mod ast;
mod dfa;
mod errors;
pub mod input;
mod ir;
pub mod lexer;
pub mod parser;
mod typechecker;
mod typed_ast;
mod typed_ast_lowering;

pub fn compile<'a>(input: &'a str, filename: &'a str) -> Vec<Report<'a, (String, Range<usize>)>> {
    // let mut error_reports = Vec::new();
    //
    // // Lex
    // let (tokens, lexer_errs) = lexer().parse_recovery(input);
    // error_reports.extend(parsing_reports(lexer_errs, filename));
    //
    // if let Some(tokens) = tokens {
    //     let input_len = input.len();
    //     let stream = Stream::from_iter(input_len..input_len + 1, tokens.into_iter());
    //     // Parse
    //     let (ast, parser_errs) = program_parser().parse_recovery(stream);
    //     error_reports.extend(parsing_reports(parser_errs, filename));
    //     if let Some(ast) = ast {
    //         if error_reports.is_empty() {
    //             // Typecheck
    //             match typecheck_program(&ast.value) {
    //                 Ok(_) => {}
    //                 Err(errs) => {
    //                     error_reports.extend(typechecking_reports(errs, filename));
    //                 }
    //             }
    //         }
    //     }
    // }

    let lexer = Lexer::new(input);
    let actual = ProgramParser::new().parse(lexer);

    match actual {
        Ok(actual) => match typecheck_program(actual) {
            Ok(program) => {
                for f in program.0 {
                    lower_fn_decl(f.value);
                }
                todo!("compile")
            }

            Err(errors) => typechecking_reports(errors, filename),
        },
        Err(err) => parsing_reports(err, filename),
    }
}
