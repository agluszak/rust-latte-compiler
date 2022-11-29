all:
	cargo build --release
	cp target/release/latc latc

clean:
	cargo clean
	rm -f latc

.PHONY: all clean