{
  lib,
  runCommand,
  python312,
  aiebu,
  llvm-aie,
  mlir-aie,
  xrt,
}:

let
  pythonEnv = python312.withPackages (_: [
    llvm-aie
    mlir-aie
  ]);
  sitePackages = python312.sitePackages;
  mlirAieRoot = "${mlir-aie}/${sitePackages}/mlir_aie";
  peanoRoot = "${llvm-aie}/${sitePackages}/llvm-aie";
in
runCommand "gufo-aie-smoke-program" {
  nativeBuildInputs = [
    aiebu
    pythonEnv
    xrt
  ];
  src = ../../src/core/xdna2/programs/smoke/smoke.py;
  passthru = {
    inherit aiebu mlir-aie llvm-aie;
    target = "npu2";
  };
  meta = {
    description = "Deterministic AIE2P add-one smoke program for Strix Halo";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
} ''
  export MLIR_AIE_INSTALL_DIR=${mlirAieRoot}
  export PEANO_INSTALL_DIR=${peanoRoot}
  export PATH="$MLIR_AIE_INSTALL_DIR/bin:$PATH"
  export PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python:''${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib:''${LD_LIBRARY_PATH:-}"

  mkdir -p "$out"
  python "$src" --output-dir "$out"

  test -s "$out/smoke.xclbin"
  test -s "$out/smoke.insts.bin"
  test -s "$out/smoke.insts.elf"
  test -s "$out/smoke.pdi"

  xclbinutil \
    --dump-section AIE_PARTITION:JSON:"$out/smoke.aie-partition.json" \
    --input "$out/smoke.xclbin" \
    >/dev/null
  partition_columns="$(
    python - "$out/smoke.aie-partition.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as partition_file:
    metadata = json.load(partition_file)
print(int(metadata["aie_partition"]["partition"]["column_width"]))
PY
  )"
  test "$partition_columns" -gt 0

  xclbin_sha="$(sha256sum "$out/smoke.xclbin" | cut -d ' ' -f 1)"
  insts_sha="$(sha256sum "$out/smoke.insts.bin" | cut -d ' ' -f 1)"
  elf_sha="$(sha256sum "$out/smoke.insts.elf" | cut -d ' ' -f 1)"
  pdi_sha="$(sha256sum "$out/smoke.pdi" | cut -d ' ' -f 1)"
  program_sha="$(
    cat "$out/smoke.xclbin" "$out/smoke.insts.elf" | sha256sum | cut -d ' ' -f 1
  )"

  (
    cd "$out"
    sha256sum smoke.xclbin smoke.insts.bin smoke.insts.elf smoke.pdi \
      >SHA256SUMS
  )

  mkdir -p "$out/include/gufo"
  cat >"$out/include/gufo/aie_smoke_manifest.h" <<EOF
#ifndef GUFO_AIE_SMOKE_MANIFEST_H_
#define GUFO_AIE_SMOKE_MANIFEST_H_

#include <string_view>

namespace gufo::xdna2::generated {

inline constexpr std::string_view kTarget = "npu2";
inline constexpr std::string_view kAbi = "xrt-elf-v1";
inline constexpr unsigned int kPartitionColumns = $partition_columns;
inline constexpr std::string_view kProgramSha256 = "$program_sha";
inline constexpr std::string_view kXclbinSha256 = "$xclbin_sha";
inline constexpr std::string_view kInstructionSha256 = "$insts_sha";
inline constexpr std::string_view kElfSha256 = "$elf_sha";
inline constexpr std::string_view kPdiSha256 = "$pdi_sha";

}  // namespace gufo::xdna2::generated

#endif  // GUFO_AIE_SMOKE_MANIFEST_H_
EOF

  cat >"$out/manifest.json" <<EOF
{
  "schemaVersion": "1.0.0",
  "target": "npu2",
  "abi": "xrt-elf-v1",
  "partitionColumns": $partition_columns,
  "programSha256": "$program_sha",
  "xclbinSha256": "$xclbin_sha",
  "instructionSha256": "$insts_sha",
  "elfSha256": "$elf_sha",
  "pdiSha256": "$pdi_sha"
}
EOF
''
