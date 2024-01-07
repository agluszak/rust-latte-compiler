use ariadne::{Cache, Source};

pub struct Input {
    pub source: Source,
    pub text: String, // TODO: Don't store this
    pub filename: String,
}

// FIXME: huh, ariadne is weird
impl Cache<String> for &Input {
    type Storage = String;

    fn fetch(&mut self, _id: &String) -> Result<&Source, Box<dyn std::fmt::Debug + '_>> {
        Ok(&self.source)
    }

    fn display<'a>(&self, id: &'a String) -> Option<Box<dyn std::fmt::Display + 'a>> {
        Some(Box::new(id))
    }
}

impl Input {
    pub fn new(text: String, filename: String) -> Self {
        let source = Source::from(text.clone());
        Self {
            source,
            text,
            filename,
        }
    }
}
