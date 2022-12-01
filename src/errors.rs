use crate::typechecker::{TypecheckingError, TypecheckingErrorKind};
use ariadne::{Color, ColorGenerator, Fmt, Label, Report, ReportBuilder, ReportKind};
use chumsky::error::SimpleReason;
use std::hash::Hash;
use std::ops::Range;

pub fn typechecking_reports(
    errs: Vec<TypecheckingError>,
    filename: &str,
) -> Vec<Report<(String, Range<usize>)>> {
    let mut colors = ColorGenerator::new();

    errs.into_iter()
        .map(|err| {
            let report = Report::build(ReportKind::Error, filename.to_string(), err.location.start)
                .with_message("Type error".to_string());

            let report = match err.kind {
                TypecheckingErrorKind::DuplicateArgument(name) => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!("Duplicate argument `{}`", name))
                        .with_color(colors.next()),
                ),
                TypecheckingErrorKind::IncrDecrOnNonInt => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message("Increment/decrement can only be applied to integers")
                        .with_color(colors.next()),
                ),
                TypecheckingErrorKind::NotCallable(found) => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!("`{}` is not callable", found))
                        .with_color(colors.next()),
                ),
                TypecheckingErrorKind::MissingReturn => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message("Missing return statement")
                        .with_color(colors.next()),
                ),
                TypecheckingErrorKind::Redeclaration {
                    name,
                    old_declaration,
                } => {
                    if old_declaration == (0..0) {
                        report.with_label(
                            Label::new((filename.to_string(), err.location))
                                .with_message(format!("Redeclaration of a built-in `{}`", name))
                                .with_color(colors.next()),
                        )
                    } else {
                        report
                            .with_label(
                                Label::new((filename.to_string(), err.location.clone()))
                                    .with_message(format!("Redeclaration of `{}`", name))
                                    .with_color(colors.next()),
                            )
                            .with_label(
                                Label::new((filename.to_string(), err.location))
                                    .with_message(format!("Previous declaration of `{}`", name))
                                    .with_color(colors.next()),
                            )
                    }
                }
                TypecheckingErrorKind::UndefinedVariable(name) => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!("Unknown variable `{}`", name))
                        .with_color(colors.next()),
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
                        .with_color(colors.next()),
                ),
                TypecheckingErrorKind::UnkownType(name) => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!("Unknown type `{}`", name))
                        .with_color(colors.next()),
                ),
                TypecheckingErrorKind::VoidReturn => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message("Void functions cannot return a value")
                        .with_color(colors.next()),
                ),
                TypecheckingErrorKind::VoidVariable => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message("Cannot assign to void variable")
                        .with_color(colors.next()),
                ),
                TypecheckingErrorKind::WrongArgumentCount { expected, found } => report.with_label(
                    Label::new((filename.to_string(), err.location))
                        .with_message(format!(
                            "Wrong number of arguments: expected {}, found {}",
                            expected, found
                        ))
                        .with_color(colors.next()),
                ),
                TypecheckingErrorKind::NoMain => report.with_message("No main function found"),
            };

            report.finish()
        })
        .collect()
}

pub fn parsing_reports<T: ToString + Hash + Eq>(
    errs: Vec<chumsky::error::Simple<T>>,
    filename: &str,
) -> Vec<Report<(String, Range<usize>)>> {
    let errs: Vec<chumsky::error::Simple<String>> =
        errs.into_iter().map(|e| e.map(|c| c.to_string())).collect();

    errs.into_iter()
        .map(|err| {
            let report: ReportBuilder<(String, Range<usize>)> =
                Report::build(ReportKind::Error, filename.to_string(), err.span().start);
            let report = match err.reason() {
                SimpleReason::Unclosed { span, delimiter } => report
                    .with_message(format!(
                        "Unclosed delimiter {}",
                        delimiter.fg(Color::Yellow)
                    ))
                    .with_label(
                        Label::new((filename.to_string(), span.clone()))
                            .with_message(format!(
                                "Unclosed delimiter {}",
                                delimiter.fg(Color::Yellow)
                            ))
                            .with_color(Color::Yellow),
                    ),
                SimpleReason::Unexpected => report
                    .with_message(format!(
                        "{}, expected one of: {}",
                        if err.found().is_some() {
                            "Unexpected token in input"
                        } else {
                            "Unexpected end of input"
                        },
                        if err.expected().len() == 0 {
                            "something else".to_string()
                        } else {
                            err.expected()
                                .map(|expected| match expected {
                                    Some(expected) => expected.to_string(),
                                    None => "end of input".to_string(),
                                })
                                .collect::<Vec<_>>()
                                .join(", ")
                        }
                    ))
                    .with_label(
                        Label::new((filename.to_string(), err.span()))
                            .with_message(format!(
                                "Unexpected token {}",
                                err.found()
                                    .unwrap_or(&"end of file".to_string())
                                    .fg(Color::Red)
                            ))
                            .with_color(Color::Red),
                    ),
                SimpleReason::Custom(msg) => report.with_message(msg).with_label(
                    Label::new((filename.to_string(), err.span()))
                        .with_message(format!("{}", msg.fg(Color::Red)))
                        .with_color(Color::Red),
                ),
            };

            report.finish()
        })
        .collect()
}
