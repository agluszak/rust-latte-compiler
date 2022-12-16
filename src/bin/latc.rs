use rust_latte_compiler::compile;
use rust_latte_compiler::input::Input;
use std::io::Read;
use std::process::ExitCode;

pub fn read_input() -> Result<Input, String> {
    match std::env::args().collect::<Vec<_>>().as_slice() {
        [_, path] => {
            let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
            let mut text = String::new();
            file.read_to_string(&mut text).map_err(|e| e.to_string())?;
            Ok(Input::new(text, path.to_string()))
        }
        [_] => {
            let mut source = String::new();
            std::io::stdin()
                .read_to_string(&mut source)
                .map_err(|e| e.to_string())?;
            Ok(Input::new(source, "<stdin>".to_string()))
        }
        [this, ..] => Err(format!("Usage: {this} [file]")),
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

    let error_reports = compile(&input.text, &input.filename);

    if !error_reports.is_empty() {
        println!("ERROR");
        for report in error_reports {
            report.eprint(&input).unwrap_or(());
        }
        ExitCode::FAILURE
    } else {
        println!("OK");
        ExitCode::SUCCESS
    }
}
