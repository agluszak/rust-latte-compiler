//! Shared `lli` execution helper for integration tests.
//!
//! The child inherits no pipes: stdin comes from a pre-written file and
//! stdout/stderr go to files, so a chatty program cannot deadlock against a
//! full pipe buffer while the harness waits for its exit.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const LLI_TIMEOUT: Duration = Duration::from_secs(10);

#[allow(dead_code)]
pub struct Outcome {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub code: Option<i32>,
    pub timed_out: bool,
}

pub fn run_lli(bitcode: &Path, stdin_data: &str) -> Outcome {
    let dir = tempfile::tempdir().expect("failed to create lli scratch dir");
    let stdin_path = dir.path().join("stdin");
    let stdout_path = dir.path().join("stdout");
    let stderr_path = dir.path().join("stderr");
    fs::write(&stdin_path, stdin_data).expect("failed to write lli stdin");
    let stdin_file = fs::File::open(&stdin_path).expect("failed to open lli stdin");
    let stdout_file = fs::File::create(&stdout_path).expect("failed to create lli stdout");
    let stderr_file = fs::File::create(&stderr_path).expect("failed to create lli stderr");

    let mut child = Command::new("lli")
        .arg(bitcode)
        .stdin(Stdio::from(stdin_file))
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .expect("failed to launch lli");

    let deadline = Instant::now() + LLI_TIMEOUT;
    loop {
        match child.try_wait().expect("failed to poll lli") {
            Some(status) => {
                return Outcome {
                    stdout: fs::read(&stdout_path).unwrap_or_default(),
                    stderr: fs::read(&stderr_path).unwrap_or_default(),
                    code: status.code(),
                    timed_out: false,
                };
            }
            None => {
                if Instant::now() >= deadline {
                    child.kill().expect("failed to kill timed-out lli");
                    let status = child.wait().expect("failed to reap timed-out lli");
                    return Outcome {
                        stdout: fs::read(&stdout_path).unwrap_or_default(),
                        stderr: fs::read(&stderr_path).unwrap_or_default(),
                        code: status.code(),
                        timed_out: true,
                    };
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}
