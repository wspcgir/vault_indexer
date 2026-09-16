{
  description = "Obsidian hybrid search and indexing tool";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem = { pkgs, ... }:
        let
          python = pkgs.python3;
          vault-indexer = python.pkgs.buildPythonApplication {
            pname = "vault-indexer";
            version = "0.1.0";
            src = ./.;

            format = "pyproject";

            nativeBuildInputs = with python.pkgs; [
              setuptools
            ];

            propagatedBuildInputs = with python.pkgs; [
              lancedb
              pyyaml
              requests
              rank-bm25
              numpy
            ];

            doCheck = false;

            meta = with pkgs.lib; {
              description = "Obsidian Hybrid Semantic & Frontmatter Search Engine";
              mainProgram = "vault-indexer";
            };
          };
        in
        {
          packages.default = vault-indexer;

          apps.default = {
            type = "app";
            program = "${vault-indexer}/bin/vault-indexer";
          };

          devShells.default = pkgs.mkShell {
            inputsFrom = [ vault-indexer ];
            packages = [ python.pkgs.pip ];
          };
        };
    };
}