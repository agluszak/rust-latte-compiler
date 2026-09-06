use std::io::Write;
use std::process::Command;

#[test]
fn cli_writes_ll_and_bc_and_reports_ok_on_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("smoke.lat");
    let mut source = std::fs::File::create(&source_path).unwrap();
    writeln!(source, "int main() {{ return 0; }}").unwrap();
    drop(source);

    let output = Command::new(env!("CARGO_BIN_EXE_latc"))
        .arg(&source_path)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "latc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.lines().next().is_some_and(|line| line == "OK"),
        "first stderr line must be OK, got: {stderr:?}"
    );
    assert!(
        !String::from_utf8(output.stdout).unwrap().contains("OK"),
        "status must go to stderr, not stdout"
    );
    assert!(
        dir.path().join("smoke.ll").is_file(),
        "expected .ll artifact"
    );
    assert!(
        dir.path().join("smoke.bc").is_file(),
        "expected .bc artifact"
    );
}

#[test]
fn cli_reports_error_on_stderr_for_bad_program() {
    let dir = tempfile::tempdir().unwrap();
    let source_path = dir.path().join("bad.lat");
    let mut source = std::fs::File::create(&source_path).unwrap();
    writeln!(source, "int main() {{ return; }}").unwrap();
    drop(source);

    let output = Command::new(env!("CARGO_BIN_EXE_latc"))
        .arg(&source_path)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.lines().next().is_some_and(|line| line == "ERROR"),
        "first stderr line must be ERROR, got: {stderr:?}"
    );
}
