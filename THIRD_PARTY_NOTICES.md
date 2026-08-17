# Third-Party Notices and Inventory

This document records the complete inventory of third-party software, libraries, drivers,
and system components used, linked, or required by **Strix-Halo.cpp**.

For design policy details regarding licensing boundaries, see [docs/LICENSING.md](docs/LICENSING.md).

---

## Inventory Summary

| Component Name | Relationship | License (SPDX) | Pinned Revision / Version | Upstream Source / Location |
| --- | --- | --- | --- | --- |
| **XRT** | Linked | `Apache-2.0` | `8661761775a266b11992a3bd6eb08209d88aa845` | [Xilinx/XRT](https://github.com/Xilinx/XRT) |
| **xdna-driver** | Linked / Loaded | `Apache-2.0` | `4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305` | [amd/xdna-driver](https://github.com/amd/xdna-driver) |
| **ROCm / HIP** | Linked / Toolchain | `MIT OR Apache-2.0 WITH LLVM-exception` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/clr](https://github.com/ROCm/clr) |
| **hipBLAS** | Linked | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/hipBLAS](https://github.com/ROCm/hipBLAS) |
| **rocBLAS** | Linked | `MIT` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) |
| **libuuid** | Linked | `BSD-3-Clause` / `LGPL-2.1-or-later` | Nixpkgs `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` | [util-linux](https://git.kernel.org/pub/scm/utils/util-linux/util-linux.git) |
| **`amdxdna` Kernel Driver** | System (Kernel) | `GPL-2.0-only` | System Kernel (`amdxdna.ko`) | [amd/xdna-driver](https://github.com/amd/xdna-driver) |
| **`amdxdna` UAPI Headers** | System / Header | `GPL-2.0 WITH Linux-syscall-note` | `4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305` | [amd/xdna-driver](https://github.com/amd/xdna-driver) |
| **AMD NPU Firmware** | System (Firmware) | Proprietary Binary (`LICENSE.amdnpu`) | System Firmware (`linux-firmware`) | Host OS Distribution / AMD |

---

## 1. Distributed Userspace Dependencies

### 1.1 XRT (Xilinx Runtime)

- **Component Name**: XRT (Xilinx Runtime for AMD Ryzen AI NPU)
- **Upstream URL**: https://github.com/Xilinx/XRT
- **Pinned Revision**: Commit `8661761775a266b11992a3bd6eb08209d88aa845` (`unstable-2026-06-04`)
- **Component Used**: Userspace runtime libraries (`libxrt_core`, `libxrt_coreutil`, C/C++ headers)
- **SPDX License Identifier**: `Apache-2.0`
- **Copyright / Notice Source**:
  - Copyright (C) 2016-2022 Xilinx, Inc.
  - Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
  - Additional upstream copyright holders preserved in project [NOTICE](NOTICE) file.
- **Relationship**: Linked (Dynamic runtime dependency in Nix derivation `.devops/nix/xrt.nix`)
- **Corresponding-Source Location**: https://github.com/Xilinx/XRT/tree/8661761775a266b11992a3bd6eb08209d88aa845

### 1.2 xdna-driver (AMD XDNA XRT Userspace Shim)

- **Component Name**: xdna-driver (AMD XDNA Driver Shim & Plugin for XRT)
- **Upstream URL**: https://github.com/amd/xdna-driver
- **Pinned Revision**: Commit `4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305` (Branch `1.7`, plugin version `2.21.0`, `1.7-unstable-2026-07-22`)
- **Component Used**: Userspace driver plugin (`libxrt_driver_xdna.so`, shim headers under `src/shim`)
- **SPDX License Identifier**: `Apache-2.0`
- **Copyright / Notice Source**: Copyright (C) 2022-2025, Advanced Micro Devices, Inc. All rights reserved. (Upstream repository checked: no separate root NOTICE file exists at this revision).
- **Relationship**: Linked / Loaded (Dynamic XRT plugin loaded at runtime in `.devops/nix/xrt-plugin-amdxdna.nix`)
- **Corresponding-Source Location**: https://github.com/amd/xdna-driver/tree/4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305

### 1.3 ROCm / HIP (AMD ROCm Compute Language Runtime)

- **Component Name**: ROCm / HIP (AMD ROCm Compute Language Runtime)
- **Upstream URL**: https://github.com/ROCm/clr
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.clr` / `rocmPackages.llvm.clang`)
- **Component Used**: HIP runtime, headers, host/device headers, and Clang compiler for `gfx1151`
- **SPDX License Identifier**: `MIT OR Apache-2.0 WITH LLVM-exception`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked / Toolchain (Build toolchain and runtime library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/clr (via Nix derivation `rocmPackages.clr`)

### 1.4 hipBLAS

- **Component Name**: hipBLAS (AMD ROCm BLAS Marshalling Library)
- **Upstream URL**: https://github.com/ROCm/hipBLAS
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.hipblas`)
- **Component Used**: Dynamic BLAS abstraction library and interface headers
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked (Dynamic library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/hipBLAS (via Nix derivation `rocmPackages.hipblas`)

### 1.5 rocBLAS

- **Component Name**: rocBLAS (AMD ROCm Basic Linear Algebra Subprograms)
- **Upstream URL**: https://github.com/ROCm/rocBLAS
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`rocmPackages.rocblas`)
- **Component Used**: GPU BLAS execution kernels optimized for `gfx1151`
- **SPDX License Identifier**: `MIT`
- **Copyright / Notice Source**: Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **Relationship**: Linked (Dynamic library dependency)
- **Corresponding-Source Location**: https://github.com/ROCm/rocBLAS (via Nix derivation `rocmPackages.rocblas`)

### 1.6 libuuid (util-linux)

- **Component Name**: libuuid (util-linux UUID library)
- **Upstream URL**: https://git.kernel.org/pub/scm/utils/util-linux/util-linux.git
- **Pinned Revision**: Nixpkgs `nixos-unstable` lock revision `2fcb964de67fcf60b43471c55d5d99e61a9ccb5a` (`libuuid`)
- **Component Used**: Dynamic library for Universally Unique Identifier (UUID) generation
- **SPDX License Identifier**: `BSD-3-Clause` / `LGPL-2.1-or-later`
- **Copyright / Notice Source**: Copyright (C) 1996, 1997, 1998 Theodore Ts'o.
- **Relationship**: Linked (Dynamic runtime library dependency required by XRT and strix)
- **Corresponding-Source Location**: https://git.kernel.org/pub/scm/utils/util-linux/util-linux.git

---

## 2. System Boundary Dependencies (Non-Distributed)

The following components are required from the host environment or system kernel. They are **not** bundled, copied, or distributed by Strix-Halo.cpp.

### 2.1 Upstream Linux `amdxdna` Kernel Driver

- **Component Name**: Upstream Linux `amdxdna` Kernel Driver (`amdxdna.ko`)
- **Upstream URL**: https://github.com/amd/xdna-driver (kernel driver subdirectory) / Linux kernel mainline
- **Pinned Revision**: Host Linux kernel driver (`amdxdna`)
- **Component Used**: System device driver providing character device node `/dev/accel/accel*`
- **SPDX License Identifier**: `GPL-2.0-only`
- **Copyright / Notice Source**: Copyright (C) Advanced Micro Devices, Inc.
- **Relationship**: System (Non-distributed system kernel driver. Strix-Halo.cpp runs as an independent userspace process communicating via the standard kernel UAPI boundary without copying kernel implementation code).
- **Corresponding-Source Location**: Host OS Linux kernel package / distribution kernel source tree

### 2.2 `amdxdna` Kernel UAPI Headers

- **Component Name**: `amdxdna` Kernel User-Space API Headers (`amdxdna_accel.h`)
- **Upstream URL**: https://github.com/amd/xdna-driver
- **Pinned Revision**: Commit `4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305` (`src/driver/amdxdna/amdxdna_accel.h`)
- **Component Used**: Header file defining ioctl numbers and kernel-userspace interface structs
- **SPDX License Identifier**: `GPL-2.0 WITH Linux-syscall-note`
- **Copyright / Notice Source**: Copyright (C) 2023-2025 Advanced Micro Devices, Inc.
- **Relationship**: System / Header (The Linux-syscall-note explicitly permits non-GPL userspace applications to include these UAPI header definitions without triggering GPL copyleft on the userspace application).
- **Corresponding-Source Location**: https://github.com/amd/xdna-driver/tree/4e5aed38f3b74a5a9a2c7a6222eaff1a8be54305

### 2.3 AMD NPU Firmware Binaries

- **Component Name**: AMD XDNA / NPU Firmware Artifacts
- **Upstream URL**: https://github.com/amd/xdna-driver / Host OS distribution firmware package (`linux-firmware`)
- **Pinned Revision**: Host system firmware files (e.g. `/lib/firmware/amdgpu/` or `/lib/firmware/amd/`)
- **Component Used**: Binary firmware blobs loaded into hardware NPU tiles by the kernel driver
- **SPDX License Identifier**: Proprietary Redistribution License (`LICENSE.amdnpu`)
- **Copyright / Notice Source**: Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.
- **Relationship**: System (Host system dependency. Strix-Halo.cpp does **not** bundle or redistribute NPU firmware binaries; firmware must be provided by the host Linux distribution).
- **Corresponding-Source Location**: Host OS `linux-firmware` package / AMD hardware driver packages
