extern crate core;

use crate::errors::{parsing_reports, typechecking_reports};
use crate::lexer::Lexer;
use crate::parser::latte::ProgramParser;
use crate::typechecker::typecheck_program;
use ariadne::Report;

use crate::ir::Ir;
use crate::llvm_generator::CodeGen;
use inkwell::context::Context;
use std::ops::Range;

mod ast;
mod dfa;
mod errors;
pub mod input;
pub mod ir;
mod junk;
pub mod lexer;
pub mod llvm_generator;
pub mod parser;
mod typechecker;
mod typed_ast;

type AriadneReport<'a> = Report<'a, (String, Range<usize>)>;

pub fn compile<'a>(input: &'a str, filename: &'a str) -> Result<String, Vec<AriadneReport<'a>>> {
    let lexer = Lexer::new(input);
    let parsed = ProgramParser::new()
        .parse(lexer)
        .map_err(|err| parsing_reports(err, filename))?;
    let (typechecked, env) =
        typecheck_program(parsed).map_err(|errs| typechecking_reports(errs, filename))?;

    #[cfg(feature = "dbg")]
    {
        dbg!(&typechecked);
        dbg!(&env);
    }

    let mut ir = Ir::new();

    for decl in typechecked.0 {
        ir.translate_function(decl.value);
    }

    #[cfg(feature = "dbg")]
    {
        println!("{}", ir.dump());
    }

    let context = Context::create();
    let codegen = CodeGen::new(&context, filename, env);

    for (name, func) in &ir.functions {
        codegen.declare(name, func);
    }

    for (name, func) in ir.functions {
        codegen.generate(&name, &func);
    }

    Ok(codegen.compile_to_string())
}
