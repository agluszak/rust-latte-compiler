extern crate core;

use crate::errors::typechecking_reports;
use crate::lexer::lexer;
use crate::lexer2::Lexer;

use crate::parser2::latte::ProgramParser;
use crate::typechecker::typecheck_program;
use ariadne::{Color, Label, Report, ReportKind};
use chumsky::Parser;

use lalrpop_util::ParseError;
use std::ops::Range;

mod ast;
mod errors;
pub mod input;
mod lexer;
pub mod lexer2;
mod parser;
pub mod parser2;
mod typechecker;

pub fn compile(input: &str, filename: &str) -> Vec<Report<(String, Range<usize>)>> {
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
        Ok(actual) => {
            let errors = typecheck_program(&actual);

            if let Err(errs) = errors {
                typechecking_reports(errs, filename)
            } else {
                Vec::new()
            }
        }
        Err(err) => match err {
            ParseError::InvalidToken { location } => {
                let mut report = Report::build(ReportKind::Error, filename.to_string(), location)
                    .with_message("Invalid token".to_string());

                report = report.with_label(
                    Label::new((filename.to_string(), location..location))
                        .with_message("Invalid token".to_string())
                        .with_color(Color::Red),
                );

                let report = report.finish();

                vec![report]
            }
            ParseError::UnrecognizedToken {
                token: (start, token, end),
                expected: _,
            } => {
                let mut report = Report::build(ReportKind::Error, filename.to_string(), start)
                    .with_message("Unrecognized token".to_string());

                report = report.with_label(
                    Label::new((filename.to_string(), start..end))
                        .with_message(format!("Unrecognized token `{token}`"))
                        .with_color(Color::Red),
                );

                let report = report.finish();

                vec![report]
            }
            ParseError::ExtraToken {
                token: (start, token, end),
            } => {
                let mut report = Report::build(ReportKind::Error, filename.to_string(), start)
                    .with_message("Extra token".to_string());

                report = report.with_label(
                    Label::new((filename.to_string(), start..end))
                        .with_message(format!("Extra token `{token}`"))
                        .with_color(Color::Red),
                );

                let report = report.finish();

                vec![report]
            }
            ParseError::User { error } => {
                let mut report = Report::build(ReportKind::Error, filename.to_string(), 0)
                    .with_message("User error".to_string());

                report = report.with_label(
                    Label::new((filename.to_string(), 0..0))
                        .with_message(format!("User error `{error:?}`"))
                        .with_color(Color::Red),
                );

                let report = report.finish();

                vec![report]
            }
            ParseError::UnrecognizedEOF {
                location,
                expected: _,
            } => {
                let mut report = Report::build(ReportKind::Error, filename.to_string(), location)
                    .with_message("Unrecognized EOF".to_string());

                report = report.with_label(
                    Label::new((filename.to_string(), location..location))
                        .with_message("Unrecognized EOF".to_string())
                        .with_color(Color::Red),
                );

                let report = report.finish();

                vec![report]
            }
        },
    }
}
