extern crate lalrpop;

use std::fs::read_dir;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

fn read_latte_inputs(directory: &str) -> Vec<PathBuf> {
    let directory = read_dir(directory).unwrap();
    let mut files = directory
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_file())
        .map(|path| path.canonicalize().unwrap())
        .filter(|path| path.extension().unwrap() == "lat")
        .collect::<Vec<_>>();
    files.sort_by(|a, b| a.file_name().unwrap().cmp(b.file_name().unwrap()));
    files
}

// build script's entry point
fn main() {
    lalrpop::process_root().unwrap();

    let mut test_file = File::create("./tests/generated_from_inputs.rs").unwrap();

    // write test file header, put `use`, `const` etc there
    write_header(&mut test_file);

    let good_inputs = read_latte_inputs("./inputs/good/");
    for entry in good_inputs {
        write_good_test(&mut test_file, &entry);
    }

    let bad_inputs = read_latte_inputs("./inputs/bad/");
    for entry in bad_inputs {
        write_bad_test(&mut test_file, &entry);
    }
}

// TODO: once actual code generation is implemented, run the code and compare the output
fn write_good_test(test_file: &mut File, path: &Path) {
    let path_str = path.to_string_lossy();
    let test_name = path.file_stem().unwrap().to_string_lossy();

    write!(
        test_file,
        include_str!("tests/good_test.rs.template"),
        name = test_name,
        path = path_str
    )
    .unwrap();
}

fn write_bad_test(test_file: &mut File, path: &Path) {
    let path_str = path.to_string_lossy();
    let test_name = path.file_stem().unwrap().to_string_lossy();

    write!(
        test_file,
        include_str!("tests/bad_test.rs.template"),
        name = test_name,
        path = path_str
    )
    .unwrap();
}

fn write_header(test_file: &mut File) {
    write!(test_file, include_str!("tests/test_header.rs.template")).unwrap();
}
