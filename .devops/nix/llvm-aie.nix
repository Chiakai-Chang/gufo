{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  python312Packages,
  zlib,
}:

python312Packages.buildPythonPackage rec {
  pname = "llvm-aie";
  version = "21.0.0.2026080301+c9c5ecb7";
  format = "wheel";

  src = fetchurl {
    url = "https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-${version}-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-WmwnxVFyRAQKTcNODB+pdvK9tuFLKaNVQiNvZuibE3U=";
  };

  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [
    stdenv.cc.cc.lib
    zlib
  ];

  dontStrip = true;
  doCheck = false;
  pythonImportsCheck = [ ];

  meta = {
    description = "Peano LLVM compiler for AMD AI Engine targets";
    homepage = "https://github.com/Xilinx/llvm-aie";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
