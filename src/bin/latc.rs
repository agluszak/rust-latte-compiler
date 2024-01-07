use rust_latte_compiler::compile;
use rust_latte_compiler::input::Input;

use std::fs::File;
use std::io::{Read, Write};
use std::process::{Command, ExitCode};

fn read_from_path(path: &str) -> Result<Input, String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut text = String::new();
    file.read_to_string(&mut text).map_err(|e| e.to_string())?;
    Ok(Input::new(text, path.to_string()))
}

fn read_from_stdin() -> Result<Input, String> {
    let mut source = String::new();
    std::io::stdin()
        .read_to_string(&mut source)
        .map_err(|e| e.to_string())?;
    Ok(Input::new(source, "<stdin>".to_string()))
}

pub fn read_input() -> Result<Input, String> {
    match std::env::args().collect::<Vec<_>>().as_slice() {
        [_, path, dbg] if dbg == "--dbg" => {
            rust_latte_compiler::DBG.store(true, std::sync::atomic::Ordering::Relaxed);
            read_from_path(path)
        }
        [_, path] => read_from_path(path),
        [_] => read_from_stdin(),
        [this, ..] => Err(format!("Usage: {this} <file> [--dbg]")),
        &[] => unreachable!(),
    }
}

fn compile_llvm_ir(ir: &str, filename: &str) -> anyhow::Result<()> {
    let output_basename = filename.replace(".lat", "");

    let ll_path = format!("{}.ll", output_basename);
    let bc_path = format!("{}.bc", output_basename);

    let mut ll_file = File::create(&ll_path)?;
    ll_file.write_all(ir.as_bytes())?;

    let status = Command::new("llvm-as")
        .args(["-o", &bc_path, &ll_path])
        .status()?;

    if !status.success() {
        anyhow::bail!("llvm-as failed");
    }

    Ok(())
}

fn main() -> ExitCode {
    let input = {
        match read_input() {
            Ok(input) => input,
            Err(err) => {
                eprintln!("Error: {err}");
                return ExitCode::FAILURE;
            }
        }
    };

    let result = compile(&input.text, &input.filename);

    match result {
        Ok(ir) => {
            if rust_latte_compiler::DBG.load(std::sync::atomic::Ordering::Relaxed) {
                println!("{}", ir);
            }

            if let Err(err) = compile_llvm_ir(&ir, &input.filename) {
                eprintln!("ERROR\n {err}");
                return ExitCode::FAILURE;
            }

            println!("OK");
            ExitCode::SUCCESS
        }
        Err(error_reports) => {
            println!("ERROR");
            for report in error_reports {
                report.eprint(&input).unwrap_or(());
            }
            ExitCode::FAILURE
        }
    }
}
