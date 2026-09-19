{
  lib,
  newScope,
  version,
}:

lib.makeScope newScope (self: {
  gufo = self.callPackage ./package.nix { inherit version; };
  mkServe = self.callPackage ./mk-serve.nix { };
})
