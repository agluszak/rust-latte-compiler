use rust_latte_compiler::compile;
use rust_latte_compiler::input::Input;

use inkwell::context::Context;
use std::io::Read;
use std::path::Path;
use std::process::ExitCode;

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

    let context = Context::create();
    let result = compile(&context, &input.text, &input.filename);

    match result {
        Ok(module) => {
            if rust_latte_compiler::DBG.load(std::sync::atomic::Ordering::Relaxed) {
                println!("{}", module.print_to_string().to_string());
            }

            if let Err(err) = module.verify() {
                eprintln!("ERROR\n generated invalid LLVM IR: {err}");
                return ExitCode::FAILURE;
            }

            let output = Path::new(&input.filename).with_extension("bc");
            if !module.write_bitcode_to_path(&output) {
                eprintln!(
                    "ERROR\n failed to write LLVM bitcode to {}",
                    output.display()
                );
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
