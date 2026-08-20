# Copyright (C) 2026 Strix Engine contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build the deterministic XDNA2 add-one smoke program."""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, Out
from aie.iron.algorithms import transform
from aie.iron.device import from_name


@iron.jit
def add_one(
    input_tensor: In,
    output_tensor: Out,
    *,
    problem_size: CompileTime[int] = 1024,
    tile_size: CompileTime[int] = 32,
):
    tensor_type = np.ndarray[(problem_size,), np.dtype[np.int32]]
    return transform(lambda value: value + 1, tensor_type, tile_size=tile_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    iron.set_current_device(from_name("npu2"))

    program = add_one.specialize(problem_size=1024, tile_size=32)
    program.compile(
        xclbin_path=args.output_dir / "smoke.xclbin",
        inst_path=args.output_dir / "smoke.insts.bin",
        elf_path=args.output_dir / "smoke.insts.elf",
        pdi_path=args.output_dir / "smoke.pdi",
    )


if __name__ == "__main__":
    main()
