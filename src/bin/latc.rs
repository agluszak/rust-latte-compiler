use ariadne::Source;
use rust_latte_compiler::compile;
use std::io::Read;
use std::process::ExitCode;

pub struct Input {
    pub source: String,
    pub filename: String,
}

pub fn read_input() -> Result<Input, String> {
    match std::env::args().collect::<Vec<_>>().as_slice() {
        [_, path] => {
            let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
            let mut source = String::new();
            file.read_to_string(&mut source)
                .map_err(|e| e.to_string())?;
            Ok(Input {
                source,
                filename: path.to_string(),
            })
        }
        [_] => {
            let mut source = String::new();
            std::io::stdin()
                .read_to_string(&mut source)
                .map_err(|e| e.to_string())?;

            Ok(Input {
                source,
                filename: "<stdin>".to_string(),
            })
        }
        [this, ..] => Err(format!("Usage: {} [file]", this)),
        &[] => unreachable!(),
    }
}

fn main() -> ExitCode {
    let input = {
        match read_input() {
            Ok(input) => input,
            Err(err) => {
                eprintln!("Error: {}", err);
                return ExitCode::FAILURE;
            }
        }
    };

    let error_reports = compile(&input.source);

    if !error_reports.is_empty() {
        println!("ERROR");
        for report in error_reports {
            report.eprint(Source::from(&input.source)).unwrap_or(());
        }
        ExitCode::FAILURE
    } else {
        println!("OK");
        ExitCode::SUCCESS
    }
}
