# Supported Strix Halo Toolchain Matrix

Status: active specification, updated 2026-08-17

## Objective

This document defines the single supported Linux, compiler, ROCm, XRT, `amdxdna`, firmware, and AIE toolchain matrix for Strix-Halo.cpp.

Strix-Halo.cpp exclusively targets **AMD Strix Halo** systems on **Linux x86-64**. All build-time and runtime dependencies are pinned immutably through Nix. Direct host builds, ambient `/usr` path lookups, and unpinned dependencies are unsupported.

---

## Machine Architecture Targets

| Target Component | Identifier | Architectural Specifications |
| --- | --- | --- |
| **System Architecture** | `x86_64-linux` | AMD Zen 5 CPU architecture, unified LPDDR5X memory (up to 128 GiB) |
| **Target GPU** | `gfx1151` | AMD RDNA 3.5 Graphics (AMD Radeon 8060S Graphics), 40 Compute Units |
| **Target NPU** | `XDNA2` / `AIE2P` | AMD XDNA 2 Neural Processing Unit, 4×8 spatial AIE tile array (PCI ID `17f0:10` / `17f0:11`) |

---

## Pinned Nix Build & Development Toolchain

All userspace toolchain dependencies are locked via [`flake.lock`](file:///home/mixer/strix-halo.cpp/flake.lock) and machine-checkable via `nix eval .#default.toolchain --json`.

| Dependency | Version / Revision | Source / Derivation | Purpose |
| --- | --- | --- | --- |
| **Nixpkgs Flake Input** | `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | `github:NixOS/nixpkgs/nixos-unstable` (`narHash: sha256-RzPPiWeUtuvymnpuEWsdtzli5w4kjZs49FqEs3/1u+I=`) | Pinned base package repository |
| **Host C/C++ Compiler** | GCC `15.3.0` (`gcc-wrapper-15.3.0`) | `pkgs.stdenv.cc` | C++20 host compilation (`-std=c++20`) |
| **Build Tools** | CMake `4.3.4`, Ninja `1.13.2` | `pkgs.cmake`, `pkgs.ninja` | Build generation and compilation runner |
| **ROCm CLR** | ROCm `7.2.3` | `rocmPackages.clr` (`7.2.3`) | AMD ROCm Common Language Runtime |
| **Device HIP Compiler** | AMD Clang `22.0.0` (`rocm-7.2.3`) | `rocmPackages.llvm.clang` / `hipClang` | Native device code generation for `gfx1151` |
| **ROCm Libraries** | `hipblas` `7.2.3`, `rocblas` `7.2.3` | `rocmPackages.hipblas`, `rocmPackages.rocblas` | BLAS baselines and runtime utilities |
| **XRT Userspace Runtime** | Git commit `8661761775a266b11992a3bd6eb08209d88aa845` | `github:Xilinx/XRT` (`sha256-JrqJIGJoQiXTwXjZpqAXVaHx+6i09B1qtqkJzoRPZKw=`) | AMD XRT NPU runtime shim (`libxrt_coreutil.so`) |
| **AMD XDNA Driver Plugin** | Git commit `4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305` (Plugin `2.21.0`) | `github:amd/xdna-driver` (`sha256-YUiM9u9vtISttbThTt9fTtBB/w3d5UatGyVK5mAgWNM=`) | Userspace driver plugin (`libxrt_driver_xdna.so`) |
| **Offline Python Toolchain** | Python `3.13` (`python313`) | `python313.withPackages` | Offline quantization and logit evaluation (`torchWithRocm`, `transformers`, `safetensors`, `numpy`, `scipy`, `zstandard`) |
| **AIE Microkernel Toolchain** | Ahead-of-Time MLIR-AIE / IRON | `github:amd/IRON` & `github:Xilinx/aie_api` | Ahead-of-time AIE2P microkernel generation (`mmul_8_4`) |

---

## Host Validation Prerequisites (Kernel & Firmware)

> [!IMPORTANT]
> Kernel drivers, system firmware, and device nodes (`/dev/kfd`, `/dev/dri/renderD*`, `/dev/accel/accel*`) are host-managed kernel interfaces and are not outputs of the Nix build store. They must be validated against the host environment.

| Component | Tested Host Baseline | Verification Command | Notes |
| --- | --- | --- | --- |
| **Linux Kernel** | `Linux 7.1.8+` (x86_64) | `uname -r` | Must include `amdgpu` and `amdxdna` drivers |
| **GPU Driver** | `amdgpu` with KFD support | `ls -l /dev/kfd /dev/dri/renderD*` | User must belong to `video` / `render` group |
| **NPU Driver** | `amdxdna` (Linux DRM Accel subsystem) | `ls -l /dev/accel/accel*` | Exposes XDNA2 NPU device node |
| **GPU Firmware** | GC `11.5.1` (`linux-firmware-20260810-zstd`) | `ls /lib/firmware/amdgpu/gc_11_5_1_*` | IMU, ME, MEC, MES, PFP, and RLC firmware binaries |
| **NPU Firmware** | XDNA2 `17f0:10` / `17f0:11` (`npu_7.sbin.zst` / `npu.sbin.1.1.2.64/65`) | `ls /lib/firmware/amdnpu/17f0_*` | NPU scheduler firmware |

---

## Machine-Readable Validation

The pinned toolchain metadata is exposed directly through Nix evaluation:

```sh
# Query machine-readable toolchain metadata
nix eval .#default.toolchain --json
```

Output:
```json
{
  "cxxCompiler": "gcc-wrapper-15.3.0",
  "hipClangVersion": "22",
  "rocmVersion": "7.2.3",
  "targetGpu": "gfx1151",
  "targetNpu": "XDNA2/AIE2P",
  "targetPlatform": "x86_64-linux",
  "xrtCommit": "8661761775a266b11992a3bd6eb08209d88aa845",
  "xrtPluginCommit": "4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305",
  "xrtPluginVersion": "2.21.0"
}
```

Verify build and flake integrity:

```sh
# Verify flake lock resolution
nix flake metadata --json --no-write-lock-file

# Verify development environment tools
nix develop -c bash -c 'c++ --version && cmake --version && ninja --version && hipcc --version'

# Build default target package
nix build .#default
```
