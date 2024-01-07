use crate::lexer::{LexingError, Spanned, Token};
use lalrpop_util::lalrpop_mod;

lalrpop_mod!(#[allow(clippy::all, dead_code)] pub latte); // synthesized by LALRPOP

pub type ParsingError = lalrpop_util::ParseError<usize, Token, Spanned<LexingError>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    macro_rules! parser_tests {
            ($($name:ident),*) => {
            $(
                #[test]
                fn $name() {
                    let input = include_str!(concat!("../inputs/", stringify!($name), ".lat"));

                    let lexer = Lexer::new(input);
                    let result = latte::ProgramParser::new().parse(lexer);

                    insta::with_settings!({
                        description => input,
                        omit_expression => true
                    }, {
                        insta::assert_debug_snapshot!(result);
                    });
                }
            )*
            };
    }

    parser_tests!(_parser_ugly, hello_world, factorial);
}
