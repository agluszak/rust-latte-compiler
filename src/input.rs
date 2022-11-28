use std::io::Read;

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
