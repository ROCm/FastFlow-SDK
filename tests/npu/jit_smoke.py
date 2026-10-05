#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""SMOKE: the compiler plus the IRON JIT still produce a design that runs.

Compiles the smallest complete design -- a pure-DMA memcpy, DDR -> MemTile ->
DDR, with no compute kernel -- and dispatches it. That covers the whole chain:
IRON -> MLIR -> aiecc -> bootgen -> xclbin -> HRX -> NPU.

Run it with `python dev.py test-npu-jit [--build-only]`, or directly once
fastflow-sdk-compiler is installed.

For the programming abstractions (ObjectFifo, Worker, tiling, C++ kernels) and
many more designs, see the iron-tools/MLIR-AIE programming guide
<https://xilinx.github.io/mlir-aie/dev/> and programming examples
<https://github.com/Xilinx/mlir-aie/tree/main/programming_examples>.
"""

import argparse
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import NPU2, AnyShimTile

LINE = 1024
SIZE = 4096


@iron.jit
def memcpy_passthrough(src: In, dst: Out, *, n: CompileTime[int]):
    line_ty = np.ndarray[(LINE,), np.dtype[np.int32]]
    vec_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_in = ObjectFifo(line_ty, name="in")
    of_out = of_in.cons().forward()

    def sequence(a_in, c_out, in_handle, out_handle):
        in_handle.fill(a_in)
        out_handle.drain(c_out, wait=True)

    rt = Runtime(
        sequence,
        [
            vec_ty,
            vec_ty,
            of_in.prod(tile=AnyShimTile),
            of_out.cons(tile=AnyShimTile),
        ],
    )

    return Program(NPU2(), rt).resolve_program()


def build_only(size: int) -> int:
    xclbin, insts = memcpy_passthrough.specialize(n=size).compile()
    for label, path in (("xclbin", xclbin), ("insts", insts)):
        if path is None or not Path(path).is_file() or Path(path).stat().st_size == 0:
            print(f"FAIL: compiler produced no usable {label} ({path})")
            return 1
    print(f"PASS: compiled a {size} int32 memcpy design")
    print(f"  xclbin: {xclbin} ({Path(xclbin).stat().st_size} bytes)")
    print(f"  insts:  {insts} ({Path(insts).stat().st_size} bytes)")
    return 0


def build_and_run(size: int) -> int:
    src = iron.arange(1, size + 1, dtype=np.int32, device="npu")
    dst = iron.zeros_like(src)

    memcpy_passthrough(src, dst, n=size)

    if np.array_equal(dst.numpy(), src.numpy()):
        print(f"PASS: {size} int32 elements copied through the NPU")
        return 0

    differing = int(np.count_nonzero(dst.numpy() != src.numpy()))
    print(f"FAIL: {differing}/{size} elements differ")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="compile the design but do not dispatch it (no NPU required)",
    )
    parser.add_argument(
        "-n", "--size", type=int, default=SIZE, help=f"elements (default: {SIZE})"
    )
    options = parser.parse_args()
    if options.size <= 0 or options.size % LINE:
        parser.error(f"--size must be a positive multiple of {LINE}")
    if options.build_only:
        return build_only(options.size)
    return build_and_run(options.size)


if __name__ == "__main__":
    sys.exit(main())
