extern crate core;

use crate::errors::{parsing_reports, typechecking_reports};
use crate::lexer::Lexer;
use crate::parser::latte::ProgramParser;
use crate::typechecker::{typecheck_program};
use ariadne::Report;


use std::ops::Range;
use inkwell::context::Context;
use crate::ir::{FunctionIr, Ir};
use crate::llvm_generator::CodeGen;

mod ast;
mod dfa;
mod errors;
pub mod input;
pub mod lexer;
pub mod parser;
mod typechecker;
mod typed_ast;
pub mod ir;
pub mod llvm_generator;
mod junk;

type AriadneReport<'a> = Report<'a, (String, Range<usize>)>;

pub fn compile<'a>(input: &'a str, filename: &'a str) -> Result<(), Vec<AriadneReport<'a>>> {

    let lexer = Lexer::new(input);
    let parsed = ProgramParser::new().parse(lexer).map_err(|err| parsing_reports(err, filename))?;
    let (typechecked, env) = typecheck_program(parsed).map_err(|errs| typechecking_reports(errs, filename))?;

    let mut ir = Ir::new();

    for decl in typechecked.0 {
        ir.translate_function(decl.value);
    }

    let context = Context::create();
    let codegen = CodeGen::new(&context, filename, env);

    for (name, func) in ir.functions {
        codegen.generate(&name, &func);
    }

    let new_filename = tempfile::Builder::new().suffix(".ll").tempfile().unwrap();

    let compiled_filename = new_filename.path().to_str().unwrap().to_string().replace(".ll", ".bc");

    codegen.compile(&new_filename);

    // spawn llvm-as
    let output = std::process::Command::new("llvm-as")
        .arg(new_filename.path())
        .output()
        .expect("failed to execute process");


    if !output.status.success() {
        println!("llvm-as failed");
        println!("{}", String::from_utf8_lossy(&output.stderr));
        return Err(vec![]);
    }

    println!("llvm-as succeeded");
    println!("{}", compiled_filename);
    Ok(())
}
