{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  python3,
  libuuid,
  rocmPackages,
  xrt,
  xrt-plugin-amdxdna,
  config,
  version,

  # Overridable feature flags
  rocmSupport ? config.rocmSupport or false,
  rocmGpuTargets ? (lib.optionals rocmSupport rocmPackages.clr.gpuTargets),
  # Wire the XRT NPU shim + amdxdna plugin into the build.
  xrtSupport ? true,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "strix";
  inherit version;
  src = lib.cleanSource ../..;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    python3
  ]
  ++ lib.optional rocmSupport rocmPackages.clr;

  buildInputs = lib.optionals rocmSupport [
    rocmPackages.clr
    rocmPackages.hipblas
    rocmPackages.rocblas
  ]
  ++ lib.optionals xrtSupport [
    xrt
    xrt-plugin-amdxdna
    libuuid
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  ]
  ++ lib.optional rocmSupport "-DENGINE_ENABLE_HIP=ON"
  ++ lib.optional rocmSupport "-DCMAKE_HIP_COMPILER=${rocmPackages.llvm.clang}/bin/clang"
  ++ lib.optional rocmSupport "-DGPU_TARGETS=${lib.concatStringsSep ";" rocmGpuTargets}"
  ++ lib.optional xrtSupport "-DENGINE_ENABLE_XRT=ON";

  env = lib.optionalAttrs rocmSupport {
    ROCM_PATH = "${rocmPackages.clr}";
  }
  // lib.optionalAttrs xrtSupport {
    XRT_PATH = "${xrt}/opt/xilinx/xrt";
    # Combined NPU lib dir so XRT can discover the amdxdna plugin at runtime.
    LD_LIBRARY_PATH = "${xrt}/opt/xilinx/xrt/lib:${xrt-plugin-amdxdna}/opt/xilinx/xrt/lib";
  };

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin
    cp strix $out/bin/strix
    cp strix-server $out/bin/strix-server
    if [ -f qwen_gpu_ops_test ]; then
      cp qwen_gpu_ops_test $out/bin/qwen_gpu_ops_test
    fi
    chmod +x $out/bin/*

    runHook postInstall
  '';

  doCheck = true;
  checkPhase = ''
    runHook preCheck

    # Run anti-CUDA boundary scanner on built binaries, fixtures, and project sources
    python3 ../tools/check-no-cuda.py \
      --source ../src \
      --source ../models \
      --source ../CMakeLists.txt \
      --binary strix \
      --binary strix-server \
      --self-test ../tests/static/fixtures

    runHook postCheck
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck

    # Scan installed package binaries and verify no CUDA libraries/symbols
    python3 ../tools/check-no-cuda.py \
      --binary $out/bin/strix \
      --binary $out/bin/strix-server

    # Scan output directory references for any forbidden CUDA strings or paths
    if grep -rnwi "$out" -e "libcudart" -e "libcublas" -e "cuda_runtime" 2>/dev/null; then
      echo "ERROR: CUDA string/library found in output closure!" >&2
      exit 1
    fi

    runHook postInstallCheck
  '';

  passthru = {
    inherit rocmPackages xrt xrt-plugin-amdxdna;
    toolchain = {
      targetPlatform = "x86_64-linux";
      targetGpu = "gfx1151";
      targetNpu = "XDNA2/AIE2P";
      cxxCompiler = stdenv.cc.name;
      rocmVersion = rocmPackages.clr.version;
      hipClangVersion = rocmPackages.llvm.clang.version;
      xrtCommit = xrt.src.rev;
      xrtPluginCommit = xrt-plugin-amdxdna.src.rev;
      xrtPluginVersion = xrt-plugin-amdxdna.pluginVersion;
    };
  };

  meta = with lib; {
    description = "Strix Engine — local inference runtime for AMD Strix Halo (gfx1151 GPU + XDNA2 NPU)";
    homepage = "https://github.com/";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "strix";
  };
})
