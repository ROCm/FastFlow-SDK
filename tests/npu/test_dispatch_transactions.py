# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path

import pytest

from build_tools.dispatch_transactions import generate_dispatch_transaction
from build_tools.npu_test_package import (
    verify_tree,
    write_dispatch_case,
    write_manifest,
)

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]

ROOT = Path(__file__).resolve().parents[2]
IRON_PYTHON = ROOT / "build" / "iron-tools" / "python"
LAYER_LIBRARY = (
    ROOT
    / "out"
    / "iron-ops"
    / "flm.layer"
    / "FLM_DecodeLayer_gemma4_e2b_global_npu2.dll"
)


def _require_dispatch_library():
    if not LAYER_LIBRARY.is_file() or not IRON_PYTHON.is_dir():
        pytest.skip("flm.layer dispatch library is not built")
    location = str(IRON_PYTHON)
    if location not in sys.path:
        sys.path.insert(0, location)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [location, os.environ.get("PYTHONPATH", "")]
    )


def test_layer_library_emits_a_distinct_transaction_per_scalar_set():
    _require_dispatch_library()
    first = generate_dispatch_transaction(
        LAYER_LIBRARY,
        ["context_len", "max_l"],
        {"context_len": 5, "max_l": 1024},
    )
    second = generate_dispatch_transaction(
        LAYER_LIBRARY,
        ["context_len", "max_l"],
        {"context_len": 700, "max_l": 1024},
    )
    assert first and second
    assert first != second


def test_dispatch_case_records_each_generated_transaction(tmp_path):
    import numpy as np

    values = np.arange(4, dtype=np.int32)
    generated = {
        "name": "flm.layer",
        "buffers": [
            {
                "name": "input",
                "direction": "input",
                "dtype": "int32",
                "shape": [4],
                "data": values,
            },
            {
                "name": "output",
                "direction": "output",
                "dtype": "int32",
                "shape": [4],
                "expected": values,
                "rtol": 0.0,
                "atol": 0.0,
            },
        ],
        "dispatches": [
            {
                "name": "context_5",
                "transaction": b"txn-5",
                "expected": {"output": values},
            },
            {
                "name": "context_700",
                "transaction": b"txn-700",
                "expected": {"output": values + 1},
            },
        ],
    }
    case = write_dispatch_case(tmp_path, generated, b"xclbin", {"name": "flm.layer"})
    write_manifest(tmp_path, [case], "test-revision", "Windows", "AMD64")
    assert (tmp_path / case["dispatches"][0]["transaction"]).read_bytes() == b"txn-5"
    assert (tmp_path / case["dispatches"][1]["transaction"]).read_bytes() == b"txn-700"
    assert not (tmp_path / "cases" / "flm.layer" / "kernel.xadx").exists()
    verify_tree(tmp_path, {"flm.layer"}, "test-revision", "Windows")
