#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build IRON GEMM and dequant operators and stage them for FastFlowLM.

Derived from amd/IRON's Gemma 4 GEMM/dequant example:
https://github.com/amd/IRON/tree/devel/iron/applications/gemma4_flm

    python build.py <FastFlowLM xclbins> <out dir>

Writes:

    <out>/xclbins/Gemma4-E2B-IT-NPU2/  engine xclbins, with GEMM and dequant replaced
    <out>/xclbins/.../iron/            GEMM and dequant instruction sequences
"""

import argparse
import shutil
from pathlib import Path

import aie.utils as aie_utils
from aie.iron.device import NPU2
from iron.common import AIEContext
from iron.operators.flm import GEMM, DequantBFP
from iron.operators.flm.gemm.design import Epilogue

MODEL = "Gemma4-E2B-IT-NPU2"

# The longest prompt chunk, flm's --prefill-chunk-len. The engine starts a
# chunk up to 127 tokens early and pads it to a multiple of 256 rows. Each row
# count M needs its own GEMM instruction sequence.
CHUNK = 512
WIDTHS = range(256, (CHUNK + 127 + 255) // 256 * 256 + 1, 256)

# (K, N, gelu) of every projection GEMM of Gemma 4 E2B.
GEMMS = [
    (1536, 2048, False),  # q, sliding window
    (1536, 4096, False),  # q, global
    (1536, 256, False),  # k and v, sliding window
    (1536, 512, False),  # k and v, global
    (2048, 1536, False),  # o, sliding window
    (4096, 1536, False),  # o, global
    (1536, 6144, True),  # gate
    (1536, 6144, False),  # up
    (6144, 1536, False),  # down
    (1536, 12288, True),  # gate, wide MLP
    (1536, 12288, False),  # up, wide MLP
    (12288, 1536, False),  # down
]

# (K, N) of every runtime dequantization step. A layer with its own KV cache
# dequantizes q, k, and v in one dispatch.
DEQUANT = [
    (1536, 2048),
    (1536, 4096),
    (1536, 2048 + 2 * 256),
    (1536, 4096 + 2 * 512),
    (2048, 1536),
    (4096, 1536),
    (6144, 1536),
    (12288, 1536),
]
UPGATE = [(1536, 6144), (1536, 12288)]
UPGATE_RUN = 512


def gemm(m, k, n, gelu, context):
    """Create one GEMM with arithmetic matching the FastFlowLM engine."""
    return GEMM(
        M=m,
        K=k,
        N=n,
        epilogue=Epilogue.GELU if gelu else Epilogue.NONE,
        tile_n=64,
        k_tile=256,
        rounding="floor",
        gelu="bf16_steps",
        context=context,
    )


def create_operators(context):
    dequants = [DequantBFP(K=k, N=n, context=context) for k, n in DEQUANT]
    dequants += [
        DequantBFP(
            K=k,
            N=n,
            run_out_features=UPGATE_RUN,
            run_period_out_features=2 * UPGATE_RUN,
            context=context,
        )
        for k, n in UPGATE
    ]
    gemms = [gemm(m, k, n, gelu, context) for m in WIDTHS for k, n, gelu in GEMMS]
    return dequants, gemms


def compile_all(operators):
    for index, operator in enumerate(operators, 1):
        print(f"[{index}/{len(operators)}] {operator.name}", flush=True)
        operator.compile()


def one_xclbin(operators):
    """Return the xclbin shared by all operators in one family."""
    xclbins = {operator.xclbin_artifact.filename for operator in operators}
    if len(xclbins) != 1:
        raise RuntimeError(f"expected one configuration, got {sorted(xclbins)}")
    return xclbins.pop()


def stage(engine_xclbins, out, dequants, gemms):
    target = out / "xclbins" / MODEL
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(Path(engine_xclbins) / MODEL, target, symlinks=True)
    sequences = target / "iron"
    sequences.mkdir()

    for name, xclbin in {
        "mm": one_xclbin(gemms),
        "dequant": one_xclbin(dequants),
    }.items():
        shutil.copyfile(xclbin, target / f"{name}.xclbin")

    for operator in dequants:
        shutil.copyfile(
            operator.insts_artifact.filename,
            sequences / f"dequant_K{operator.K}_N{operator.N}.bin",
        )
    for operator in gemms:
        gelu = "_gelu" if operator.epilogue == Epilogue.GELU else ""
        shutil.copyfile(
            operator.insts_artifact.filename,
            sequences / f"gemm_M{operator.M}_K{operator.K}_N{operator.N}{gelu}.bin",
        )
    print(f"staged {target}")


def main(engine_xclbins, out, *, compile_only=False, smoke=False):
    aie_utils.set_current_device(NPU2())
    context = AIEContext(build_dir=out / "ops")
    dequants, gemms = create_operators(context)
    selected = [dequants[0], gemms[0]] if smoke else [*dequants, *gemms]
    compile_all(selected)
    if not compile_only:
        if smoke:
            raise ValueError("--smoke requires --compile-only")
        stage(engine_xclbins, out, dequants, gemms)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_xclbins")
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="compile operators without staging FastFlowLM assets",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="compile one GEMM and one dequant operator for CI",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.engine_xclbins,
        args.out.resolve(),
        compile_only=args.compile_only,
        smoke=args.smoke,
    )
