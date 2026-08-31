extern crate lalrpop;

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    lalrpop::process_root().unwrap();
    build_runtime();
}

fn build_runtime() {
    println!("cargo:rerun-if-changed=lib/runtime.c");

    let output =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set")).join("runtime.bc");
    let status = Command::new("clang")
        .args(["-emit-llvm", "-c", "lib/runtime.c", "-o"])
        .arg(&output)
        .status()
        .expect("failed to run clang");

    assert!(status.success(), "failed to compile Latte runtime");
}
