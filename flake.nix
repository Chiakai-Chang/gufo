{
  description = "strix-halo.cpp - Strix Engine for AMD Strix Halo (gfx1151 GPU + XDNA2 NPU)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      # Strix Halo is a Linux x86-64-only target (see README non-goals).
      forAllSystems = nixpkgs.lib.genAttrs [ "x86_64-linux" ];
      version = self.shortRev or self.dirtyShortRev or "dirty";

      pkgs = forAllSystems (system: import nixpkgs { inherit system; });

      strixPackages = forAllSystems (
        system:
        pkgs.${system}.callPackage ./.devops/nix/scope.nix { inherit version; }
      );

      # Offline model-conversion toolchain. Python-only; never a transitive
      # dependency of the server (see docs/QUANTIZATION.md "Offline Toolchain").
      # Unified python313 + torchWithRocm: every Strix Halo box ships ROCm, so
      # the single toolchain serves CPU flows and the --device cuda
      # calibration forward alike. gfx1151 verified on this host.
      pythonTools = system:
        let pt = pkgs.${system}.python313; in
        pt.withPackages (ps: [
          ps.torchWithRocm
          ps.transformers
          ps.safetensors
          ps.huggingface-hub
          ps.numpy
          ps.scipy
          ps.zstandard
        ]);
    in
    {
      packages = forAllSystems (
        system:
        let
          base = strixPackages.${system}.strix;
        in
        {
          # The only package we ship: ROCm/HIP compiled for gfx1151 + XRT NPU
          # shim. The driver derivations (xrt, xrt-plugin-amdxdna) are internal
          # build inputs in scope.nix and are not exposed as flake packages.
          default = base.override {
            rocmSupport = true;
            rocmGpuTargets = [ "gfx1151" ];
          };
        }
      );

      devShells = forAllSystems (
        system:
        {
          # Unified toolchain (CPU + ROCm torch): one default shell serves all
          # offline CPU flows and the --device cuda calibration forward.
          default = pkgs.${system}.mkShell {
            inputsFrom = [ self.packages.${system}.default ];
            packages = [ (pythonTools system) ];
            env = {
              ROCM_PATH = "${pkgs.${system}.rocmPackages.clr}";
              XRT_PATH = "${strixPackages.${system}.xrt}/opt/xilinx/xrt";
            };
          };
        }
      );

      checks = forAllSystems (
        system:
        let
          pkgsSys = pkgs.${system};
        in
        {
          format = pkgsSys.runCommand "check-format" {
            nativeBuildInputs = [ pkgsSys.clang-tools ];
            src = self;
          } ''
            cd "$src"
            clang-format --dry-run --Werror src/main.cpp src/server/main.cpp
            mkdir -p $out
            echo "PASS: Formatting check clean" > $out/result.txt
          '';

          static-analysis = pkgsSys.runCommand "check-static-analysis" {
            nativeBuildInputs = [
              pkgsSys.stdenv.cc
              pkgsSys.clang-tools
              pkgsSys.cmake
              pkgsSys.ninja
              pkgsSys.python3
            ];
            src = self;
          } ''
            export HOME=$TMPDIR
            mkdir -p build && cd build
            cmake "$src" -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=OFF
            clang-tidy -p . "$src"/src/main.cpp "$src"/src/server/main.cpp
            mkdir -p $out
            echo "PASS: clang-tidy static analysis clean" > $out/result.txt
          '';

          dependency-inventory = pkgsSys.runCommand "check-dependency-inventory" {
            nativeBuildInputs = [ pkgsSys.python3 ];
            src = self;
          } ''
            cd "$src"
            mkdir -p $out
            python3 tools/check-dependencies.py --json-report $out/dependency-inventory.json
            echo "PASS: Dependency inventory clean" > $out/result.txt
          '';

          docs = pkgsSys.runCommand "check-docs" {
            nativeBuildInputs = [ pkgsSys.python3 ];
            src = self;
          } ''
            cd "$src"
            mkdir -p $out
            python3 tools/check-docs.py --root "$src"
            echo "PASS: Documentation check clean" > $out/result.txt
          '';
        }
      );
    };
}
