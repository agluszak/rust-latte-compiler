use rust_latte_compiler::{emit_llvm, link_runtime, lower_program, optimize_program};
use std::io::Write;
use std::process::{Command, Stdio};

fn run_source(source: &str, stdin_data: &str) -> (String, i32) {
    let lexer = rust_latte_compiler::lexer::Lexer::new(source);
    let parsed = rust_latte_compiler::parser::latte::ProgramParser::new()
        .parse(lexer)
        .expect("test source must parse");
    let (checked, env) = rust_latte_compiler::typechecker::typecheck_program(parsed)
        .expect("test source must typecheck");

    // Lower once, then compare the unoptimized program against an optimized copy.
    let unoptimized = lower_program(checked, env);
    let mut optimized = rust_latte_compiler::ProgramIr {
        ir: unoptimized.ir.clone(),
        env: unoptimized.env.clone(),
    };
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
    let mut child = Command::new("lli")
        .arg(bitcode.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin_data.as_bytes())
        .unwrap();
    let output = wait_with_timeout(child, std::time::Duration::from_secs(10));
    (
        String::from_utf8(output.stdout).unwrap(),
        output.status.code().unwrap_or(-1),
    )
}

fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: std::time::Duration,
) -> std::process::Output {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait().unwrap() {
            Some(_) => return child.wait_with_output().unwrap(),
            None => {
                if std::time::Instant::now() >= deadline {
                    child.kill().unwrap();
                    panic!("lli timed out");
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }
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
fn constant_condition_return_agrees() {
    let source = "int main() { if (!false) { return 0; } return 1; }";
    assert_eq!(run_source(source, ""), ("".to_string(), 0));
}
