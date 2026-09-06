extern crate core;

use crate::errors::{parsing_reports, typechecking_reports};
use crate::lexer::Lexer;
use crate::parser::latte::ProgramParser;
use crate::typechecker::typecheck_program;
use ariadne::Report;

use crate::ir::Ir;
use crate::llvm_generator::CodeGen;
use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;
use inkwell::support::LLVMString;
use std::ops::Range;

mod ast;
mod cfg;
mod const_cond;
mod errors;
mod gvn;
pub mod input;
pub mod ir;
pub mod lexer;
pub mod llvm_generator;
mod lower;
pub mod parser;
mod passes;
mod return_analysis;
pub mod ssa;
pub mod typechecker;
pub mod typed_ast;
pub mod types;
mod symbols;
mod verify;

type AriadneReport<'a> = Report<'a, (String, Range<usize>)>;

static RUNTIME_BITCODE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/runtime.bc"));

pub struct ProgramIr {
    pub ir: Ir,
    pub env: crate::typechecker::ReadyEnvironment,
}

pub fn lower_program(
    typechecked: crate::typed_ast::TypedProgram,
    env: crate::typechecker::ReadyEnvironment,
) -> ProgramIr {
    let mut ir = Ir::new();
    for decl in typechecked.0 {
        ir.translate_function(decl.value);
    }
    ProgramIr { ir, env }
}

pub fn optimize_program(program: &mut ProgramIr) {
    passes::optimize_program(&mut program.ir);
}

pub fn emit_llvm<'ctx>(context: &'ctx Context, filename: &str, program: &ProgramIr) -> Module<'ctx> {
    let codegen = CodeGen::new(context, filename, program.env.clone());

    for (name, func) in &program.ir.functions {
        codegen.declare(name, func);
    }

    for (name, func) in &program.ir.functions {
        codegen.generate(name, func);
    }

    codegen.into_module()
}

pub fn compile<'ctx, 'src>(
    context: &'ctx Context,
    input: &'src str,
    filename: &'src str,
) -> Result<Module<'ctx>, Vec<AriadneReport<'src>>> {
    let lexer = Lexer::new(input);
    let parsed = ProgramParser::new()
        .parse(lexer)
        .map_err(|err| parsing_reports(err, filename))?;
    let (typechecked, env) =
        typecheck_program(parsed).map_err(|errs| typechecking_reports(errs, filename))?;

    let mut program = lower_program(typechecked, env);
    optimize_program(&mut program);

    Ok(emit_llvm(context, filename, &program))
}

pub fn link_runtime(module: &Module<'_>) -> Result<(), LLVMString> {
    let runtime_buffer = MemoryBuffer::create_from_memory_range(RUNTIME_BITCODE, "runtime.bc");
    let runtime = Module::parse_bitcode_from_buffer(&runtime_buffer, module.get_context())?;
    module.link_in_module(runtime)
}
