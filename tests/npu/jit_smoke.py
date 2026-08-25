#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""SMOKE: the compiler plus the IRON JIT still produce a design that runs.

Compiles the smallest complete design -- a pure-DMA memcpy, DDR -> MemTile ->
DDR, with no compute kernel -- and dispatches it. That covers the whole chain:
IRON -> MLIR -> aiecc -> bootgen -> xclbin -> HRX -> NPU. Nothing here is
Peano-specific, so a Peano or aie_api regression shows up as a compile failure
rather than a wrong result.

--build-only stops at the xclbin so a CPU-only runner can gate the compiler and
the JIT without an NPU.

Run it with `python dev.py test-npu-jit [--build-only]`, or directly once
rocm-npu-compiler is installed.

For the programming abstractions (ObjectFifo, Worker, tiling, C++ kernels) and
many more designs, see the iron-tools/MLIR-AIE programming guide
<https://xilinx.github.io/mlir-aie/dev/> and programming examples
<https://github.com/Xilinx/mlir-aie/tree/main/programming_examples>.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import rocm_npu_compiler
except ImportError:
    # Source build: dev.py test-npu-jit exports the toolchain location.
    pass
else:
    # Compiler wheels from 0.1.0.dev202608050023 and earlier do not export their
    # own tool locations to the in-process JIT. Newer ones do, and setdefault
    # keeps any caller's values.
    _prefix = Path(rocm_npu_compiler.compiler_prefix())
    os.environ.setdefault("PEANO_INSTALL_DIR", str(_prefix / "peano" / "llvm-aie"))
    os.environ.setdefault("AIEOPT_DIR", str(_prefix / "third_party" / "aie_api"))
    # Found on PATH, so this also works when the virtual environment is not activated.
    os.environ["PATH"] = os.pathsep.join(
        [str(_prefix / "bin"), os.environ.get("PATH", "")]
    )

try:
    import rocm_npu_runtime
except ImportError:
    # Source build, or --build-only against a compiler-only install.
    pass
else:
    # HRX is opt-in, and its discovery does not look inside the runtime wheel.
    os.environ.setdefault("NPU_RUNTIME", "hrx")
    _runtime_root = Path(rocm_npu_runtime.__file__).parent
    for _pattern in ("lib/libhrx.so*", "bin/hrx.dll", "lib/hrx.dll"):
        _found = sorted(_runtime_root.glob(_pattern))
        if _found:
            os.environ.setdefault("HRX_LIBHRX", str(_found[0]))
            break

import numpy as np  # noqa: E402  (must follow the environment setup above)

import aie.iron as iron  # noqa: E402
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime  # noqa: E402
from aie.iron.device import NPU2  # noqa: E402

LINE = 1024
SIZE = 4096


@iron.jit
def memcpy_passthrough(src: In, dst: Out, *, n: CompileTime[int]):
    line_ty = np.ndarray[(LINE,), np.dtype[np.int32]]
    vec_ty = np.ndarray[(n,), np.dtype[np.int32]]

    of_in = ObjectFifo(line_ty, name="in")
    of_out = of_in.cons().forward(name="out")

    rt = Runtime()
    with rt.sequence(vec_ty, vec_ty) as (a_in, c_out):
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

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
    src = iron.arange(size, dtype=np.int32)
    dst = iron.zeros(size, dtype=np.int32)

    memcpy_passthrough(src, dst, n=size)

    if np.array_equal(np.asarray(src), np.asarray(dst)):
        print(f"PASS: {size} int32 elements copied through the NPU")
        return 0

    differing = int(np.count_nonzero(np.asarray(src) != np.asarray(dst)))
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
