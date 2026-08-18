{
  lib,
  stdenv,
  fetchFromGitHub,
  fetchzip,
  cmake,
  ninja,
  pkg-config,
  python3,
  libbacktrace,
  rocmPackages,
}:

let
  # Fixed-output dependency sources pinned in hrx-system MODULE.cmake.lock
  flatcc-src = fetchzip {
    url = "https://github.com/dvidelabs/flatcc/archive/9362cd00f0007d8cbee7bff86e90fb4b6b227ff3.tar.gz";
    sha256 = "sha256-umZ9TvNYDZtF/mNwQUGuhAGve0kPw7uXkaaQX0EzkBY=";
  };

  hsa-headers-src = fetchzip {
    url = "https://github.com/iree-org/hsa-runtime-headers/archive/4285513114a70f7cf4830c89279c8cfa57b901bb.tar.gz";
    sha256 = "sha256-Px1Erkg6T6z9KTuL3fc9dYR28zvuebxe7wgLfRIPhEQ=";
  };

  hip-headers-src = fetchzip {
    url = "https://github.com/iree-org/hip-build-deps/archive/c64ec391ab8eaf4c9872660b24a71fda8026b438.tar.gz";
    sha256 = "sha256-L571fRWGoH+ezaR/3JvvKIKfe1azG4ajerC/b4jVnKs=";
  };

  clangStdenv = stdenv.override {
    cc = rocmPackages.llvm.clang;
  };
in
clangStdenv.mkDerivation {
  pname = "hrx-system";
  version = "unstable-2026-08";

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "hrx-system";
    rev = "8c274e544a9d411eb05e1eb5629101dd44d2379e";
    hash = "sha256-ixzyD1tJ+sJxSU/8halBJrGsWZEq2EDKxMan5/SXDjs=";
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    python3
  ];

  buildInputs = [
    libbacktrace
    rocmPackages.clr
    rocmPackages.rocm-runtime
    rocmPackages.aqlprofile
    rocmPackages.rocm-device-libs
  ];

  cmakeFlags = [
    "-DFETCHCONTENT_SOURCE_DIR_FLATCC=${flatcc-src}"
    "-DFETCHCONTENT_SOURCE_DIR_HSA_RUNTIME_HEADERS=${hsa-headers-src}"
    "-DFETCHCONTENT_SOURCE_DIR_HIP_API_HEADERS=${hip-headers-src}"
    "-DIREE_DEPENDENCY_MODE=pinned"
    "-DIREE_ROCM_DEPENDENCY_MODE=pinned"
    "-DCMAKE_PREFIX_PATH=${rocmPackages.aqlprofile}"
    "-DIREE_ROCM_PATH=${rocmPackages.clr}"
    "-DIREE_BUILD_TESTS=OFF"
    "-DIREE_BUILD_BENCHMARKS=OFF"
    "-DIREE_HAL_DRIVER_AMDGPU=ON"
    "-DIREE_HAL_DRIVER_HIP=ON"
    "-DLIBHRX_BUILD=ON"
    "-DLOOM_BUILD=OFF"
  ];

  meta = with lib; {
    description = "HRX (Hip Runtime Extended) system runtime components and AMDGPU/HIP driver";
    homepage = "https://github.com/ROCm/hrx-system";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
