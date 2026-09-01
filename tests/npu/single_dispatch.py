#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""ON-DEVICE e2e: dispatch a built iron-tools design once through the hrx HAL.

This is what iron-tools actually emits for vector_scalar_add: one xclbin, one TXN
instruction stream, one ERT_START_NPU dispatch. Chaining several dispatches into an
ERT_CMD_CHAIN is an hrx runlist feature, not compiler output, so it is exercised
separately (`dev.py test-npu --chain`) and does not gate this.

Run from the design directory after `make all NPU2=1`.
"""

import argparse
import sys

import numpy as np
from aie.utils.hostruntime.hrxruntime.hostruntime import HRXHostRuntime
from aie.utils.hostruntime.hrxruntime.tensor import HRXTensor
from aie.utils.npukernel import NPUKernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-x", "--xclbin", default="build/final.xclbin")
    p.add_argument("-i", "--instr", default="build/insts.bin")
    p.add_argument("-k", "--kernel", default="MLIR_AIE")
    p.add_argument("-n", "--size", type=int, default=1024)
    opts = p.parse_args()

    rt = HRXHostRuntime()
    handle = rt.load(NPUKernel(opts.xclbin, opts.instr, kernel_name=opts.kernel))

    src = np.arange(1, opts.size + 1, dtype=np.int32)  # in[i] = i + 1
    in_a = HRXTensor(src, dtype=np.int32)
    out = HRXTensor((opts.size,), dtype=np.int32)  # zero-filled

    rt.run(handle, [in_a, out])

    got = out.numpy()
    ref = src + 1
    if not np.array_equal(got, ref):
        bad = int(np.argmax(got != ref))
        count = int(np.count_nonzero(got != ref))
        print(
            f"FAIL: {count}/{opts.size} mismatches; first at {bad}: "
            f"got {got[bad]}, expected {ref[bad]}"
        )
        # An all-zero output means the dispatch reported success without writing.
        if not got.any():
            print("      output is entirely zero: the kernel did not write its result")
        return 1

    print(f"single dispatch: out == in + 1  (first 4: {got[:4]})")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
