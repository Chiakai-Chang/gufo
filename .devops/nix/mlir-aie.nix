{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  python312Packages,
  zlib,
}:

python312Packages.buildPythonPackage rec {
  pname = "mlir-aie";
  version = "1.4.1";
  format = "wheel";

  src = fetchurl {
    url = "https://github.com/Xilinx/mlir-aie/releases/download/v${version}/mlir_aie-${version}-cp312-cp312-manylinux_2_35_x86_64.whl";
    hash = "sha256-GU+XzzFc8XHKV0m5GhhIB24EangL4z97i5YRe8b4pUA=";
  };

  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [
    stdenv.cc.cc.lib
    zlib
  ];
  dependencies = with python312Packages; [
    aiofiles
    cloudpickle
    ml-dtypes
    numpy
    rich
  ];

  dontStrip = true;
  doCheck = false;
  postInstall = ''
    site_packages="$out/${python312Packages.python.sitePackages}"
    ln -s "$site_packages/mlir_aie/python/aie" "$site_packages/aie"
  '';
  pythonImportsCheck = [ "aie.iron" ];
  pythonRuntimeDepsCheck = false;

  meta = {
    description = "IRON and MLIR-AIE toolchain for AMD Ryzen AI NPUs";
    homepage = "https://github.com/Xilinx/mlir-aie";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
