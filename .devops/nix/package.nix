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

let
  sourceRoot = ../..;
  productionSource = lib.cleanSourceWith {
    src = sourceRoot;
    filter =
      path: _type:
      let
        root = toString sourceRoot;
        pathString = toString path;
        relativePath = lib.removePrefix "${root}/" pathString;
      in
      pathString == root
      || relativePath == "CMakeLists.txt"
      || relativePath == "cmake"
      || lib.hasPrefix "cmake/" relativePath
      || relativePath == "src"
      || lib.hasPrefix "src/" relativePath;
  };
in
stdenv.mkDerivation (finalAttrs: {
  pname = "strix";
  inherit version;
  src = productionSource;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ]
  ++ lib.optional rocmSupport rocmPackages.clr;

  buildInputs = lib.optionals rocmSupport [
    rocmPackages.clr
    rocmPackages.hipblas
    rocmPackages.hipblaslt
    rocmPackages.rocblas
  ]
  ++ lib.optionals xrtSupport [
    xrt
    xrt-plugin-amdxdna
    libuuid
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DBUILD_TESTING=OFF"
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
    if [ -f strix-bench ]; then
      cp strix-bench $out/bin/strix-bench
    fi
    chmod +x $out/bin/*

    runHook postInstall
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    runHook preInstallCheck

    $out/bin/strix-server --version
    $out/bin/strix-server --help >/dev/null

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
