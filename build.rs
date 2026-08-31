extern crate lalrpop;

use std::process::Command;

// build script's entry point
fn main() {
    lalrpop::process_root().unwrap();

    let clang = "clang";

    // Compile runtime.c
    let status = Command::new(clang)
        .args([
            "-emit-llvm",
            "-c",
            "./lib/runtime.c",
            "-o",
            "./lib/runtime.bc",
        ])
        .status()
        .unwrap();

    assert!(status.success());

    let status = Command::new(clang)
        .args([
            "-emit-llvm",
            "-S",
            "./lib/runtime.c",
            "-o",
            "./lib/runtime.ll",
        ])
        .status()
        .unwrap();

    assert!(status.success());
}
