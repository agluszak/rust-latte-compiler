extern crate core;

use crate::lexer::{lexer, Span, Token};
use crate::parser::program_parser;
use crate::typechecker::typecheck_program;
use chumsky::Parser;
use chumsky::Stream;
use std::io::Read;
use std::process::exit;

mod ast;
mod example;
mod grammar;
mod lexer;
mod parser;
mod typechecker;

fn main() -> anyhow::Result<()> {
    // Read stdin
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    // Lex
    let input_len = input.len();
    match lexer().parse(input) {
        Ok(tokens) => {
            // Parse
            let stream = Stream::from_iter(input_len..input_len + 1, tokens.into_iter());
            match program_parser().parse(stream) {
                Ok(program) => {
                    // Typecheck
                    match typecheck_program(&program.value) {
                        Ok(program) => {
                            // Print
                            println!("OK");
                        }
                        Err(err) => {
                            println!("ERROR");
                            eprintln!("Type error: {:?}", err);
                            exit(1);
                        }
                    }
                }
                Err(err) => {
                    println!("ERROR");
                    eprintln!("Parse error: {:?}", err);
                    exit(1);
                }
            }
        }
        Err(err) => {
            println!("ERROR");
            println!("Lexing error: {:?}", err);
            exit(1);
        }
    }

    Ok(())
}
