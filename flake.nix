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
            nativeBuildInputs = [ pkgsSys.clang-tools pkgsSys.findutils ];
            src = self;
          } ''
            cd "$src"
            find src tests -not -path "*/fixtures/*" \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -exec clang-format --dry-run --Werror {} +
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
              pkgsSys.findutils
            ];
            src = self;
          } ''
            export HOME=$TMPDIR
            mkdir -p build && cd build
            cmake "$src" -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=OFF
            find "$src"/src "$src"/tests/diagnostics -name "*.cpp" -exec clang-tidy -p . {} +
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

          # Canonical PR umbrella check (all quality gates)
          pr = pkgsSys.runCommand "check-pr" {
            nativeBuildInputs = [
              pkgsSys.stdenv.cc
              pkgsSys.clang-tools
              pkgsSys.cmake
              pkgsSys.ninja
              pkgsSys.python3
              pkgsSys.findutils
            ];
            src = self;
          } ''
            export HOME=$TMPDIR
            set -euo pipefail

            echo "=== [PR Gate 1/7] Format Validation (clang-format) ==="
            cd "$src"
            find src tests -not -path "*/fixtures/*" \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -exec clang-format --dry-run --Werror {} +
            echo "PASS: Formatting check clean"

            echo "=== [PR Gate 2/7] Static Analysis (clang-tidy) ==="
            mkdir -p "$TMPDIR/build-static" && cd "$TMPDIR/build-static"
            cmake "$src" -GNinja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_TESTING=OFF -DENGINE_ENABLE_HIP=OFF -DENGINE_ENABLE_XRT=OFF
            find "$src"/src "$src"/tests/diagnostics -name "*.cpp" -exec clang-tidy -p . {} +
            echo "PASS: Static analysis clean"

            echo "=== [PR Gate 3/7] Dependency and License Inventory Consistency ==="
            cd "$src"
            python3 tools/check-dependencies.py --notices "$src"/THIRD_PARTY_NOTICES.md --package-nix "$src"/.devops/nix/package.nix
            echo "PASS: Dependency inventory consistent"

            echo "=== [PR Gate 4/7] Documentation and Local Link Validation ==="
            python3 tools/check-docs.py --root "$src"
            echo "PASS: Documentation links and syntax valid"

            echo "=== [PR Gate 5/7] Build and CTest Test Suites ==="
            mkdir -p "$TMPDIR/build-test" && cd "$TMPDIR/build-test"
            cmake "$src" -GNinja -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON -DSTRIX_ENABLE_WARNINGS=ON -DSTRIX_ENABLE_SANITIZERS=OFF
            ninja
            ctest --output-on-failure
            echo "PASS: CTest test suite passed"

            echo "=== [PR Gate 6/7] Anti-CUDA Boundary Scanner ==="
            python3 "$src"/tools/check-no-cuda.py \
              --source "$src"/src \
              --source "$src"/models \
              --source "$src"/CMakeLists.txt \
              --binary "${self.packages.${system}.default}/bin/strix" \
              --binary "${self.packages.${system}.default}/bin/strix-server" \
              --self-test "$src"/tests/static/fixtures
            echo "PASS: Anti-CUDA boundary scan clean"

            echo "=== [PR Gate 7/7] Installed strix-server --version & --help Smoke ==="
            "${self.packages.${system}.default}/bin/strix-server" --version
            "${self.packages.${system}.default}/bin/strix-server" --help
            echo "PASS: strix-server CLI version and help smoke passed"

            mkdir -p $out/bin
            cp "${self.packages.${system}.default}/bin/strix" $out/bin/strix
            cp "${self.packages.${system}.default}/bin/strix-server" $out/bin/strix-server

            cat <<EOF > $out/pr-summary.txt
PR Check Summary
Status: ALL GATES PASSED
Package: ${self.packages.${system}.default.name}
Revision: ${version}
System: ${system}
Composed Gates:
  1. Format Validation (clang-format)
  2. Static Analysis (clang-tidy)
  3. Dependency and License Inventory (THIRD_PARTY_NOTICES.md)
  4. Documentation & Local Link Validation (check-docs.py)
  5. CTest Test Suites (CPU & Static Tests)
  6. Anti-CUDA Boundary Scan (Sources & Binaries)
  7. Installed strix-server CLI Smoke (--version, --help)
EOF
          '';
        }
      );
    };
}
