use rust_latte_compiler::{compile_ir, emit_llvm, link_runtime, optimize_program};

#[path = "common/mod.rs"]
mod common;

fn run_source(source: &str, stdin_data: &str) -> (String, i32) {
    // Lower once, then compare the unoptimized program against an optimized copy.
    let unoptimized = compile_ir(source, "test").expect("test source must compile");
    let mut optimized = unoptimized.clone();
    optimize_program(&mut optimized);

    let context = inkwell::context::Context::create();
    let unopt_module = emit_llvm(&context, "unopt", &unoptimized);
    link_runtime(&unopt_module).unwrap();
    unopt_module.verify().unwrap();
    let unopt_out = execute(&unopt_module, stdin_data);

    let context = inkwell::context::Context::create();
    let opt_module = emit_llvm(&context, "opt", &optimized);
    link_runtime(&opt_module).unwrap();
    opt_module.verify().unwrap();
    let opt_out = execute(&opt_module, stdin_data);

    assert_eq!(
        unopt_out, opt_out,
        "unoptimized and optimized executions must agree"
    );
    unopt_out
}

fn execute(module: &inkwell::module::Module<'_>, stdin_data: &str) -> (String, i32) {
    let bitcode = tempfile::NamedTempFile::new().unwrap();
    assert!(module.write_bitcode_to_path(bitcode.path()));
    let output = common::run_lli(bitcode.path(), stdin_data);
    assert!(!output.timed_out, "lli timed out");
    (
        String::from_utf8(output.stdout).unwrap(),
        output.code.unwrap_or(-1),
    )
}

#[test]
fn gvn_preserves_global_redundancy_result() {
    let source = "int main() { int x = 3; int y = x + 1; int z = x + 1; printInt(y + z); return 0; }";
    assert_eq!(run_source(source, ""), ("8\n".to_string(), 0));
}

#[test]
fn loop_with_short_circuit_condition_terminates() {
    let source = "int main() { int i = 0; while (i < 3 && i < 5) { i++; } printInt(i); return 0; }";
    assert_eq!(run_source(source, ""), ("3\n".to_string(), 0));
}

#[test]
fn constant_condition_return_needs_no_synthetic_return() {
    let source = "int main() { if (!false) return 0; }";
    assert_eq!(run_source(source, ""), ("".to_string(), 0));
}

#[test]
fn nested_loop_forwarded_phi_survives_outer_seal() {
    let source = "int main() { int x = 7; while (readInt() > 0) { while (x > 0) { if (readInt() > 0) printInt(x); return 0; } } return 0; }";
    assert_eq!(run_source(source, "1\n1\n"), ("7\n".to_string(), 0));
}
