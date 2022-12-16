use crate::lexer::LexingError;
use crate::lexer::Span;
use crate::parser::ParsingError;
use crate::typechecker::{TypecheckingError, TypecheckingErrorKind};
use ariadne::{Color, Label, Report, ReportBuilder, ReportKind};

use lalrpop_util::ParseError;

use std::ops::Range;

pub fn typechecking_reports(
    errs: Vec<TypecheckingError>,
    filename: &str,
) -> Vec<Report<(String, Range<usize>)>> {
    let color = Color::Red;

    errs.into_iter()
        .map(|err| {
            let report = Report::build(ReportKind::Error, filename.to_string(), err.location.start)
                .with_message("Type error".to_string());

            let report = match err.kind {
                TypecheckingErrorKind::DuplicateArgument(name) => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!("Duplicate argument `{name}`"))
                        .with_color(color),
                ),
                TypecheckingErrorKind::IncrDecrOnNonInt => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message("Increment/decrement can only be applied to integers")
                        .with_color(color),
                ),
                TypecheckingErrorKind::NotCallable(found) => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!("`{found}` is not callable"))
                        .with_color(color),
                ),
                TypecheckingErrorKind::MissingReturn => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message("Missing return statement")
                        .with_color(color),
                ),
                TypecheckingErrorKind::Redeclaration {
                    name,
                    old_declaration,
                } => {
                    if old_declaration == (0..0) {
                        report.with_label(
                            Label::new((filename.to_string(), err.location))
                                .with_message(format!("Redeclaration of a built-in `{name}`"))
                                .with_color(color),
                        )
                    } else {
                        report
                            .with_label(
                                Label::new((filename.to_string(), err.location.clone()))
                                    .with_message(format!("Redeclaration of `{name}`"))
                                    .with_color(color),
                            )
                            .with_label(
                                Label::new((filename.to_string(), err.location))
                                    .with_message(format!("Previous declaration of `{name}`"))
                                    .with_color(color),
                            )
                    }
                }
                TypecheckingErrorKind::UndefinedVariable(name) => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!("Unknown variable `{name}`"))
                        .with_color(color),
                ),
                TypecheckingErrorKind::TypeMismatch { expected, found } => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!(
                            "Type mismatch: expected `{}`, found `{}`",
                            expected
                                .iter()
                                .map(|t| t.to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                            found
                        ))
                        .with_color(color),
                ),
                TypecheckingErrorKind::UnknownType(name) => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!("Unknown type `{name}`"))
                        .with_color(color),
                ),
                TypecheckingErrorKind::VoidReturn => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message("Void functions cannot return a value")
                        .with_color(color),
                ),
                TypecheckingErrorKind::VoidVariable => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message("Cannot assign to void variable")
                        .with_color(color),
                ),
                TypecheckingErrorKind::WrongArgumentCount { expected, found } => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!(
                            "Wrong number of arguments: expected {expected}, found {found}"
                        ))
                        .with_color(color),
                ),
                TypecheckingErrorKind::NoMain => report.with_message("No main function found"),
            };

            report.finish()
        })
        .collect()
}

fn syntax_error(filename: &str, offset: usize) -> ReportBuilder<(String, Span)> {
    Report::build(ReportKind::Error, filename, offset).with_message("Syntax error".to_string())
}

pub fn parsing_reports(err: ParsingError, filename: &str) -> Vec<Report<(String, Range<usize>)>> {
    let color = Color::Red;

    let mut reports = Vec::new();

    let report = match err {
        ParseError::InvalidToken { location } => syntax_error(filename, location).with_label(
            Label::new((filename.to_string(), location..location))
                .with_message("Invalid token")
                .with_color(color),
        ),
        ParseError::UnrecognizedToken {
            token: (start, token, end),
            expected,
        } => syntax_error(filename, start)
            .with_label(
                Label::new((filename.to_string(), start..end))
                    .with_message(format!("Unrecognized token `{}`", token))
                    .with_color(color),
            )
            .with_label(
                Label::new((filename.to_string(), start..end))
                    .with_message(format!("Expected one of: {}", expected.join(", ")))
                    .with_color(color),
            ),
        ParseError::UnrecognizedEOF { location, expected } => syntax_error(filename, location)
            .with_label(
                Label::new((filename.to_string(), location..location))
                    .with_message("Unexpected end of file")
                    .with_color(color),
            )
            .with_label(
                Label::new((filename.to_string(), location..location))
                    .with_message(format!("Expected one of: {}", expected.join(", ")))
                    .with_color(color),
            ),
        ParseError::ExtraToken {
            token: (start, token, end),
        } => syntax_error(filename, start).with_label(
            Label::new((filename.to_string(), start..end))
                .with_message(format!("Unexpected token `{}`", token))
                .with_color(color),
        ),
        ParseError::User { error } => {
            let span = error.span;
            let error = error.value;
            match error {
                LexingError::NumberTooLong => syntax_error(filename, span.start).with_label(
                    Label::new((filename.to_string(), span))
                        .with_message("Number too long")
                        .with_color(color),
                ),
                LexingError::UnexpectedEscape(c) => syntax_error(filename, span.start).with_label(
                    Label::new((filename.to_string(), span))
                        .with_message(format!("Unexpected escape character `{}`", c))
                        .with_color(color),
                ),

                LexingError::InvalidUnicodeEscape(s) => syntax_error(filename, span.start)
                    .with_label(
                        Label::new((filename.to_string(), span))
                            .with_message(format!("Invalid unicode escape sequence `{}`", s))
                            .with_color(color),
                    ),

                LexingError::InvalidUnicodeChar(c) => syntax_error(filename, span.start)
                    .with_label(
                        Label::new((filename.to_string(), span))
                            .with_message(format!("Invalid unicode character `{}`", c))
                            .with_color(color),
                    ),
                LexingError::UnclosedEscape => syntax_error(filename, span.start).with_label(
                    Label::new((filename.to_string(), span))
                        .with_message("Unclosed escape sequence")
                        .with_color(color),
                ),
                LexingError::Other => syntax_error(filename, span.start).with_label(
                    Label::new((filename.to_string(), span))
                        .with_message("Lexing error")
                        .with_color(color),
                ),
            }
        }
    };

    reports.push(report.finish());
    reports
}
