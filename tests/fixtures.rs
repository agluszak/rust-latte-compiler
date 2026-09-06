use inkwell::context::Context;
use libtest_mimic::{Arguments, Failed, Trial};
use rust_latte_compiler::compile;
use rust_latte_compiler::input::Input;
use rust_latte_compiler::link_runtime;
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

#[path = "common/mod.rs"]
mod common;

fn main() {
    let mut trials = fixture_trials("inputs/good", "good", test_good);
    trials.extend(fixture_trials("inputs/bad", "bad", test_bad));
    libtest_mimic::run(&Arguments::from_args(), trials).exit();
}

fn fixture_trials(
    directory: &str,
    kind: &'static str,
    test: fn(&Path) -> Result<(), Failed>,
) -> Vec<Trial> {
    latte_files(directory)
        .into_iter()
        .map(|path| {
            let name = fixture_name(&path);
            Trial::test(format!("{kind}::{name}"), move || test(&path))
        })
        .collect()
}

fn latte_files(directory: &str) -> Vec<PathBuf> {
    let mut paths = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("failed to read {directory}: {error}"))
        .map(|entry| {
            entry
                .expect("failed to read fixture directory entry")
                .path()
        })
        .filter(|path| path.is_file() && path.extension() == Some(OsStr::new("lat")))
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn fixture_name(path: &Path) -> String {
    path.file_stem()
        .expect("fixture has no file name")
        .to_string_lossy()
        .into_owned()
}

/// Deliberate backend snapshots. Ordinary good fixtures are checked by
/// execution against their expected output, not by IR text.
const SNAPSHOT_FIXTURES: &[&str] = &["concatenation", "expression_cfg_sequencing"];

fn test_good(path: &Path) -> Result<(), Failed> {
    let source = fs::read_to_string(path)?;
    let filename = path.to_string_lossy().into_owned();
    let input = Input::new(source, filename);
    let name = fixture_name(path);
    let context = Context::create();
    let module = compile(&context, &input.text, &name)
        .map_err(|reports| format!("compilation failed with {} reports", reports.len()))?;
    module.verify().map_err(|error| error.to_string())?;

    if SNAPSHOT_FIXTURES.contains(&name.as_str()) {
        let ir = module.print_to_string().to_string();
        let snapshots_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots");
        insta::with_settings!({
            input_file => path,
            snapshot_path => snapshots_dir,
            prepend_module_to_snapshot => false,
            description => &input.text,
            omit_expression => true
        }, {
            insta::assert_snapshot!(format!("generated_from_inputs__good_{name}"), ir);
        });
    }

    let expected_output = read_optional(&path.with_extension("output"))?;
    let program_input = read_optional(&path.with_extension("input"))?;
    let expected_exit_code = read_optional(&path.with_extension("exitcode"))?
        .trim()
        .parse::<i32>()
        .unwrap_or(0);
    link_runtime(&module)?;
    module.verify().map_err(|error| error.to_string())?;
    let bitcode = tempfile::NamedTempFile::new()?;
    if !module.write_bitcode_to_path(bitcode.path()) {
        return Err("failed to write fixture bitcode".into());
    }

    let output = common::run_lli(bitcode.path(), &program_input);
    if output.timed_out {
        return Err("lli timed out after 10s".into());
    }

    if output.code != Some(expected_exit_code) {
        return Err(format!(
            "lli exited with {:?}: {}",
            output.code,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    if output.stdout != expected_output.as_bytes() {
        return Err(format!(
            "unexpected output\nexpected: {:?}\nactual:   {:?}",
            expected_output,
            String::from_utf8_lossy(&output.stdout)
        )
        .into());
    }

    Ok(())
}

fn test_bad(path: &Path) -> Result<(), Failed> {
    let source = fs::read_to_string(path)?;
    let filename = path.to_string_lossy().into_owned();
    let input = Input::new(source, filename);
    let name = fixture_name(path);
    let context = Context::create();
    let reports = match compile(&context, &input.text, &name) {
        Ok(_) => return Err("bad fixture compiled successfully".into()),
        Err(reports) => reports,
    };
    if reports.is_empty() {
        return Err("compilation failed without a diagnostic".into());
    }

    let mut output = BufWriter::new(Vec::new());
    for report in reports {
        report.write(&input, &mut output)?;
    }
    let output = String::from_utf8(output.into_inner()?)?;
    let snapshots_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots");
    insta::with_settings!({
        input_file => path,
        snapshot_path => snapshots_dir,
        prepend_module_to_snapshot => false,
        description => &input.text,
        omit_expression => true
    }, {
        insta::assert_snapshot!(format!("generated_from_inputs__bad_{name}"), output);
    });

    Ok(())
}

fn read_optional(path: &Path) -> Result<String, Failed> {
    if path.is_file() {
        Ok(fs::read_to_string(path)?)
    } else {
        Ok(String::new())
    }
}
