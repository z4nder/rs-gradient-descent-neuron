{
  description = "ia-explore — Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            rustToolchain
            pkgs.cargo-watch
            pkgs.cargo-expand
            pkgs.cargo-edit
            pkgs.pkg-config
          ];

          buildInputs = [
            pkgs.fontconfig.dev
          ];

          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          RUST_BACKTRACE = "1";

          shellHook = ''
            echo "ia-explore dev shell"
            echo "rust $(rustc --version)"
            echo "cargo $(cargo --version)"
          '';
        };
      }
    );
}
