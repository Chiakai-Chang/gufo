{
  description = "strix.cpp - Strix Engine for AMD Strix Halo (gfx1151 GPU + XDNA2 NPU)";

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

      # Offline model-conversion toolchain. Python only; never a transitive
      # dependency of the server (see docs/QUANTIZATION.md "Offline Toolchain").
      pythonTools = system: (pkgs.${system}.python3.withPackages (ps: [
        ps.torch
        ps.transformers
        ps.safetensors
        ps.huggingface-hub
        ps.numpy
        ps.scipy
        ps.zstandard
      ]));
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
        let
          py = pythonTools system;
        in
        {
          default = pkgs.${system}.mkShell {
            inputsFrom = [ self.packages.${system}.default ];
            packages = [ py ];
          };
          rocm = pkgs.${system}.mkShell {
            inputsFrom = [ self.packages.${system}.rocm-gfx1151 ];
          };
        }
      );
    };
}
