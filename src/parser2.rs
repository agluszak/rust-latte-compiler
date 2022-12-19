use lalrpop_util::lalrpop_mod;
lalrpop_mod!(#[allow(clippy::all, dead_code)] pub latte); // synthesized by LALRPOP

pub enum ParsingErrorKind {
    NumberParsingError,
}

#[cfg(test)]
mod tests {

    #[test]
    fn addition() {
        // read stdin
    }
}
