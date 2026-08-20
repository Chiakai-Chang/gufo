{
  lib,
  stdenv,
  fetchgit,
  cmake,
  ninja,
  pkg-config,
  boost,
}:

stdenv.mkDerivation {
  pname = "aiebu";
  version = "1.0-20260819";

  src = fetchgit {
    url = "https://github.com/Xilinx/aiebu.git";
    rev = "27a302c5840773e79c79f0f2fc8a1832d6ab1774";
    hash = "sha256-t6PIhfK+eQeBBgpq0gedqkJqUOYyR3Iv5uZxMKntIU0=";
    fetchSubmodules = true;
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ];
  buildInputs = [ boost ];

  cmakeFlags = [
    "-DAIEBU_BUILD_TESTS=OFF"
    "-DAIEBU_CTRLCODE_CODEGEN=OFF"
    "-DAIEBU_INSTALL_STATIC_LIBRARY=OFF"
    "-DAIEBU_PYTHON=OFF"
    "-DAIEBU_UPSTREAM=ON"
  ];

  installPhase = ''
    runHook preInstall
    install -Dm755 src/cpp/utils/asm/aiebu-asm "$out/bin/aiebu-asm"
    runHook postInstall
  '';

  meta = {
    description = "Assembler for AMD AI Engine control-code ELF files";
    homepage = "https://github.com/Xilinx/aiebu";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
