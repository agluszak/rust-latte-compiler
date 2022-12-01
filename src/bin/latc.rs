use ariadne::{Cache, Source};
use rust_latte_compiler::compile;
use std::io::Read;
use std::process::ExitCode;

pub struct Input {
    pub source: Source,
    pub text: String, // TODO: Don't store this
    pub filename: String,
}

impl Cache<String> for &Input {
    fn fetch(&mut self, id: &String) -> Result<&Source, Box<dyn std::fmt::Debug + '_>> {
        if id == &self.filename {
            Ok(&self.source)
        } else {
            Err(Box::new(format!("File not found: {}", id)))
        }
    }

    fn display<'a>(&self, id: &'a String) -> Option<Box<dyn std::fmt::Display + 'a>> {
        if id == &self.filename {
            Some(Box::new(id))
        } else {
            None
        }
    }
}

pub fn read_input() -> Result<Input, String> {
    match std::env::args().collect::<Vec<_>>().as_slice() {
        [_, path] => {
            let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
            let mut text = String::new();
            file.read_to_string(&mut text).map_err(|e| e.to_string())?;
            let source = Source::from(text.clone());
            Ok(Input {
                source,
                text,
                filename: path.to_string(),
            })
        }
        [_] => {
            let mut source = String::new();
            std::io::stdin()
                .read_to_string(&mut source)
                .map_err(|e| e.to_string())?;
            let text = source.clone();
            let source = Source::from(source);

            Ok(Input {
                source,
                text,
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
