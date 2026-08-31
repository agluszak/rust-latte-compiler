{
  description = "Latte compiler development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/26.05";

    # nixpkgs 26.05 no longer ships LLVM 14. Keep the rest of the
    # environment on 26.05 and source LLVM 14.0.6 from nixpkgs 24.05.
    nixpkgs-llvm14.url = "github:NixOS/nixpkgs/24.05";
  };

  outputs = { nixpkgs, nixpkgs-llvm14, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      llvm14Pkgs = nixpkgs-llvm14.legacyPackages.${system};
      llvm14 = llvm14Pkgs.llvmPackages_14;
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.cargo
          pkgs.rustc
          pkgs.gnumake
          llvm14.clang
          llvm14.llvm
        ];

        LLVM_SYS_140_PREFIX = "${llvm14.llvm.dev}";

        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          llvm14Pkgs.libffi
          llvm14Pkgs.libxml2
          llvm14Pkgs.ncurses
          llvm14Pkgs.zlib
          pkgs.stdenv.cc.cc.lib
        ];

        # llvm-sys links LLVM 14 statically by default with this Inkwell version.
        buildInputs = [
          llvm14Pkgs.libffi
          llvm14Pkgs.libxml2
          llvm14Pkgs.ncurses
          llvm14Pkgs.zlib
        ];
      };
    };
}
