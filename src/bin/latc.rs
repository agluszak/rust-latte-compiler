use rust_latte_compiler::compile;
use rust_latte_compiler::input::Input;
use std::io::Read;
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

    let result = compile(&input.text, &input.filename);

    match result {
        Ok(ir) => {
            println!("{}", ir);
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

// let new_filename = tempfile::Builder::new().suffix(".ll").tempfile().unwrap();
//
//     let compiled_filename = new_filename.path().to_str().unwrap().to_string().replace(".ll", ".bc");
//
//     println!("{}", codegen.compile_to_string());
//     // codegen.compile(&new_filename);
//     //
//     // // spawn llvm-as
//     // let output = std::process::Command::new("llvm-as")
//     //     .arg(new_filename.path())
//     //     .output()
//     //     .expect("failed to execute process");
//     //
//     //
//     // if !output.status.success() {
//     //     println!("llvm-as failed");
//     //     println!("{}", String::from_utf8_lossy(&output.stderr));
//     //     return Err(vec![]);
//     // }
//     //
//     // println!("llvm-as succeeded");
//     // println!("{}", compiled_filename);
//     Ok(())
