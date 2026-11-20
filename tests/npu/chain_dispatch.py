#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""ON-DEVICE e2e: chain dispatches of a built iron-tools design through the hrx HAL.

Several dispatches of vector_scalar_add's ``x + 1`` kernel go out as one batched
submit (``HRXHostRuntime.run_chain``, one ERT_CMD_CHAIN), each consuming the
previous one's output. Stage ``k`` of a chain from ``in[i] = i + 1`` must hold
``i + k + 2``: a stage that read its input before the previous stage wrote it
would not. Chaining is an hrx runlist feature, not compiler output, so this
runs under `dev.py test-npu --chain` and does not gate CI.

Run from the design directory after `make all NPU2=1`.
"""

import argparse
import sys

import numpy as np
from aie.utils.hostruntime.hrxruntime.hostruntime import HRXHostRuntime
from aie.utils.hostruntime.hrxruntime.tensor import HRXTensor
from aie.utils.npukernel import NPUKernel


def run_chain(rt, handle, src, depth):
    """Chain ``depth`` dispatches from ``src``; return the mismatching stages."""
    stages = [HRXTensor((src.size,), dtype=np.int32) for _ in range(depth)]
    previous = HRXTensor(src, dtype=np.int32)
    links = []
    for stage in stages:
        links.append((handle, [previous, stage]))
        previous = stage
    rt.run_chain(links)

    failures = []
    for k, stage in enumerate(stages):
        got = stage.numpy()
        ref = src + k + 1
        if not np.array_equal(got, ref):
            bad = int(np.argmax(got != ref))
            count = int(np.count_nonzero(got != ref))
            failures.append(
                f"depth {depth} stage {k}: {count}/{src.size} mismatches; first at "
                f"{bad}: got {got[bad]}, expected {ref[bad]}"
            )
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-x", "--xclbin", default="build/final.xclbin")
    p.add_argument("-i", "--instr", default="build/insts.bin")
    p.add_argument("-k", "--kernel", default="MLIR_AIE")
    p.add_argument("-n", "--size", type=int, default=1024)
    p.add_argument(
        "--depth",
        type=int,
        action="append",
        help="links in a chain; repeat for several chains (default: 2 and 8)",
    )
    opts = p.parse_args()
    if opts.depth is not None and any(depth <= 0 for depth in opts.depth):
        p.error("--depth must be positive")

    rt = HRXHostRuntime()
    handle = rt.load(NPUKernel(opts.xclbin, opts.instr, kernel_name=opts.kernel))
    src = np.arange(1, opts.size + 1, dtype=np.int32)  # in[i] = i + 1

    failures = []
    for depth in opts.depth or (2, 8):
        result = run_chain(rt, handle, src, depth)
        failures += result
        if not result:
            print(f"chain of {depth}: stage k == in + k + 1")
    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
