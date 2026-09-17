{
  lib,
  newScope,
  version,
}:

lib.makeScope newScope (self: {
  aie-smoke = self.callPackage ./aie-smoke.nix {
    inherit (self)
      aiebu
      llvm-aie
      mlir-aie
      xrt
      ;
  };
  aiebu = self.callPackage ./aiebu.nix { };
  gufo = self.callPackage ./package.nix {
    inherit version;
    inherit (self) aie-smoke;
  };
  xrt = self.callPackage ./xrt.nix { };
  xrt-plugin-amdxdna = self.callPackage ./xrt-plugin-amdxdna.nix {
    inherit (self) xrt;
  };
  llvm-aie = self.callPackage ./llvm-aie.nix { };
  mlir-aie = self.callPackage ./mlir-aie.nix { };
  hrx-system = self.callPackage ./hrx-system.nix { };
  hyperloom = self.callPackage ./hyperloom.nix { };
  mkServe = self.callPackage ./mk-serve.nix { };
})
