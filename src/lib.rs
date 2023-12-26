extern crate core;

use crate::errors::{parsing_reports, typechecking_reports};
use crate::lexer::Lexer;
use crate::parser::latte::ProgramParser;
use crate::typechecker::{typecheck_program};
use ariadne::Report;


use std::ops::Range;
use crate::ir::FunctionContext;

mod ast;
mod dfa;
mod errors;
pub mod input;
pub mod lexer;
pub mod parser;
mod typechecker;
mod typed_ast;
mod ir;
mod junk;

type AriadneReport<'a> = Report<'a, (String, Range<usize>)>;

pub fn compile<'a>(input: &'a str, filename: &'a str) -> Result<(), Vec<AriadneReport<'a>>> {

    let lexer = Lexer::new(input);
    let parsed = ProgramParser::new().parse(lexer).map_err(|err| parsing_reports(err, filename))?;
    let typechecked = typecheck_program(parsed).map_err(|errs| typechecking_reports(errs, filename))?;

    for decl in typechecked.0 {
        let mut ir = FunctionContext::new();
        ir.translate_function(decl.value);
        ir.dump();
    }

    Ok(())
}
