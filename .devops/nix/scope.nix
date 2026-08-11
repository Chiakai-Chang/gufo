{
  lib,
  newScope,
  version,
}:

lib.makeScope newScope (self: {
  strix = self.callPackage ./package.nix { inherit version; };
  xrt = self.callPackage ./xrt.nix { };
  xrt-plugin-amdxdna = self.callPackage ./xrt-plugin-amdxdna.nix {
    inherit (self) xrt;
  };
})
