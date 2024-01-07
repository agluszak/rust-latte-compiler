all:
	cargo build --release
	cp target/release/latc latc_llvm

clean:
	cargo clean
	rm -f latc_llvm

.PHONY: all clean