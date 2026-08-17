{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
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
    chmod +x $out/bin/strix $out/bin/strix-server

    runHook postInstall
  '';

  meta = with lib; {
    description = "Strix Engine — local inference runtime for AMD Strix Halo (gfx1151 GPU + XDNA2 NPU)";
    homepage = "https://github.com/";
    license = licenses.mit;
    platforms = platforms.unix;
    mainProgram = "strix";
  };
})
