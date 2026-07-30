# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared IRON target and toolchain policy."""

import sys
from pathlib import Path

from build_tools.dev import paths


NPU_TARGETS = {
    "npu2": {"device": "npu2", "devgen": 4, "arch": "AIE2P (Strix)"},
    "npu1": {"device": "npu", "devgen": 3, "arch": "AIE2IPU (Phoenix)"},
}


def iron_ops_suite():
    suite_dir = str(paths.ROOT / "tests" / "iron-ops")
    if suite_dir not in sys.path:
        sys.path.insert(0, suite_dir)
    import suite

    return suite


def run_e2e_module():
    tests_dir = str(paths.ROOT / "tests" / "e2e")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import run_e2e

    return run_e2e


def iron_env(compiler_prefix=None):
    run_e2e = run_e2e_module()
    prefix = (
        Path(compiler_prefix).resolve()
        if compiler_prefix
        else run_e2e.compiler_prefix_from_env()
    )
    try:
        return run_e2e.toolchain_env(
            run_e2e.require_xclbinutil(prefix), prefix
        )
    except run_e2e.E2EError as error:
        sys.exit(str(error))


def require_txn_devgen(insts, target):
    want = NPU_TARGETS[target]["devgen"]
    data = insts.read_bytes()
    if len(data) < 3:
        sys.exit(f"{insts}: too small for a TXN header")
    got = data[2]
    if got != want:
        by_generation = {
            value["devgen"]: f"{key} / {value['arch']}"
            for key, value in NPU_TARGETS.items()
        }
        sys.exit(
            f"{insts}: TXN DevGen={got} "
            f"({by_generation.get(got, 'unknown')}) but --device={target} "
            f"needs DevGen={want} ({by_generation[want]})"
        )
    print(f"insts.bin: TXN DevGen={got} (matches --device={target})")
