# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest

from tests.support.hrx import FakeBuffer, FakeExecutable
from tests.support.modules import load_module

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]


ROOT = Path(__file__).resolve().parents[2]
NPU_TEST_DIR = ROOT / "tests" / "npu"
MODULE_PATH = NPU_TEST_DIR / "run_test_package.py"


@pytest.fixture(scope="module", autouse=True)
def _load_package_runner():
    global np, runner
    import numpy as np

    runner = load_module("npu_package_runner", MODULE_PATH)


class FakeRuntime:
    instances = []

    def __init__(self, correct=True, capabilities=0):
        self.correct = correct
        self.capabilities = capabilities
        self.loaded = []
        self.export_names = []
        self.buffers = []
        self.dispatches = []
        self.constants = []
        self.transactions = []
        self.synchronize_count = 0
        self.closed = False
        self.executable = None
        self.executables = []
        self.__class__.instances.append(self)

    def load_xadx(self, data):
        self.loaded.append(data)
        self.executable = FakeExecutable(
            export_ordinal=9,
            on_lookup=self.export_names.append,
        )
        self.executables.append(self.executable)
        return self.executable

    def allocate_buffer(self, shape, dtype):
        buffer = FakeBuffer(shape, dtype)
        self.buffers.append(buffer)
        return buffer

    def load_xclbin_transaction(self, xclbin, transaction, entry_name="MLIR_AIE"):
        self.transactions.append((xclbin, transaction, entry_name))
        executable = FakeExecutable(
            export_ordinal=9,
            on_lookup=self.export_names.append,
        )
        self.executables.append(executable)
        return executable

    def dispatch(self, executable, export_ordinal, buffers, constants=b""):
        self.dispatches.append((executable, export_ordinal, buffers))
        self.constants.append(constants)
        if constants:
            offset = int.from_bytes(constants, "little", signed=True)
            buffers[1].array[:] = buffers[0].array[
                offset : offset + buffers[1].array.size
            ]
        else:
            buffers[1].array[:] = buffers[0].array + (1 if self.correct else 2)

    def synchronize(self):
        self.synchronize_count += 1

    def amdxdna_capabilities(self):
        return self.capabilities

    def close(self):
        self.closed = True


@pytest.fixture
def package_root(tmp_path):
    FakeRuntime.instances.clear()
    (tmp_path / "kernel.xadx").write_bytes(b"XADX")
    source = np.arange(-4, 4, dtype=np.int32)
    expected = source + 1
    source.tofile(tmp_path / "input.bin")
    expected.tofile(tmp_path / "expected.bin")
    manifest = {
        "cases": [
            {
                "name": "vector_scalar_add",
                "xadx": "kernel.xadx",
                "export": "MLIR_AIE",
                "buffers": [
                    {
                        "name": "input",
                        "direction": "input",
                        "dtype": "int32",
                        "shape": [8],
                        "input": "input.bin",
                    },
                    {
                        "name": "output",
                        "direction": "output",
                        "dtype": "int32",
                        "shape": [8],
                        "expected": "expected.bin",
                        "rtol": 0.0,
                        "atol": 0.0,
                    },
                ],
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_runs_package_with_fake_runtime(package_root, capsys):
    failures = runner.run_package(package_root, runtime_factory=FakeRuntime)

    assert failures == 0
    output = capsys.readouterr().out
    assert "[npu-test] lifecycle 1/5 vector_scalar_add: preparing\n" in output
    assert "[npu-test] resident 10/10 vector_scalar_add: complete\n" in output
    assert output.endswith("vector_scalar_add: PASS\n")
    runtime = FakeRuntime.instances[-1]
    assert runtime.loaded == [b"XADX"] * 6
    assert runtime.export_names == ["MLIR_AIE"] * 6
    assert runtime.dispatches[0][1] == 9
    assert len(runtime.dispatches) == 15
    assert runtime.constants == [b""] * 15
    assert runtime.synchronize_count == 15
    assert runtime.closed
    assert all(executable.closed for executable in runtime.executables)
    assert all(buffer.closed for buffer in runtime.buffers)
    assert [buffer.flush_count for buffer in runtime.buffers] == [
        1,
        2,
    ] * 5 + [1, 11]
    assert [buffer.invalidate_count for buffer in runtime.buffers] == [
        0,
        1,
    ] * 5 + [0, 10]


def test_reports_output_mismatch_and_returns_failure(package_root, capsys):
    failures = runner.run_package(
        package_root,
        runtime_factory=lambda: FakeRuntime(correct=False),
    )

    assert failures == 1
    output = capsys.readouterr().out
    assert "vector_scalar_add: FAIL" in output
    assert "8/8 mismatches" in output


def test_skips_case_with_unsupported_required_feature(package_root, capsys):
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["required_features"] = ["amdxdna-full-elf-load-pdi"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def unexpected_runtime():
        raise AssertionError("a skipped case must not initialize HRX")

    failures = runner.run_package(
        package_root,
        runtime_factory=unexpected_runtime,
        supported_features=set(),
    )

    assert failures == 0
    assert capsys.readouterr().out == (
        "vector_scalar_add: SKIP (missing features: amdxdna-full-elf-load-pdi)\n"
    )


@pytest.mark.parametrize(("capabilities", "expected"), ((0, "SKIP"), (1, "PASS")))
def test_uses_hrx_amdxdna_capability(package_root, capsys, capabilities, expected):
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["required_features"] = ["amdxdna-full-elf-load-pdi"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    failures = runner.run_package(
        package_root,
        runtime_factory=lambda: FakeRuntime(capabilities=capabilities),
    )

    assert failures == 0
    assert f"vector_scalar_add: {expected}" in capsys.readouterr().out
    assert FakeRuntime.instances[-1].closed


def test_capability_query_failure_is_not_treated_as_unsupported(package_root):
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["required_features"] = ["amdxdna-full-elf-load-pdi"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    runtime = FakeRuntime()
    runtime.amdxdna_capabilities = lambda: (_ for _ in ()).throw(
        RuntimeError("capability query failed")
    )
    with pytest.raises(RuntimeError, match="capability query failed"):
        runner.run_package(package_root, runtime_factory=lambda: runtime)
    assert runtime.closed


def test_alternates_dispatch_constants_and_expected_outputs(tmp_path, capsys):
    FakeRuntime.instances.clear()
    (tmp_path / "kernel.xadx").write_bytes(b"XADX")
    source = np.arange(16, dtype=np.int32)
    source.tofile(tmp_path / "input.bin")
    source[:8].tofile(tmp_path / "expected-0.bin")
    source[8:].tofile(tmp_path / "expected-8.bin")
    (tmp_path / "constants-0.bin").write_bytes((0).to_bytes(4, "little", signed=True))
    (tmp_path / "constants-8.bin").write_bytes((8).to_bytes(4, "little", signed=True))
    manifest = {
        "cases": [
            {
                "name": "dynamic_copy",
                "xadx": "kernel.xadx",
                "export": "MLIR_AIE",
                "buffers": [
                    {
                        "name": "input",
                        "direction": "input",
                        "dtype": "int32",
                        "shape": [16],
                        "input": "input.bin",
                    },
                    {
                        "name": "output",
                        "direction": "output",
                        "dtype": "int32",
                        "shape": [8],
                        "expected": "expected-0.bin",
                        "rtol": 0.0,
                        "atol": 0.0,
                    },
                ],
                "dispatches": [
                    {
                        "name": "offset_0",
                        "constants": "constants-0.bin",
                        "expected": {"output": "expected-0.bin"},
                    },
                    {
                        "name": "offset_8",
                        "constants": "constants-8.bin",
                        "expected": {"output": "expected-8.bin"},
                    },
                ],
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    failures = runner.run_package(
        tmp_path,
        runtime_factory=FakeRuntime,
        lifecycle_rounds=4,
        resident_rounds=4,
    )

    assert failures == 0
    output = capsys.readouterr().out
    assert "[npu-test] lifecycle 4/4 dynamic_copy: complete\n" in output
    assert "[npu-test] resident 4/4 dynamic_copy: complete\n" in output
    assert output.endswith("dynamic_copy: PASS\n")
    runtime = FakeRuntime.instances[-1]
    assert [
        int.from_bytes(value, "little", signed=True) for value in runtime.constants
    ] == [
        0,
        8,
        0,
        8,
        0,
        8,
        0,
        8,
    ]


def test_submits_one_generated_transaction_per_dispatch(tmp_path, capsys):
    FakeRuntime.instances.clear()
    xclbin = b"xclbin-bytes"
    first = b"txn-context-5"
    second = b"txn-context-700"
    source = np.arange(4, dtype=np.int32)
    (tmp_path / "kernel.xclbin").write_bytes(xclbin)
    source.tofile(tmp_path / "input.bin")
    (source + 1).tofile(tmp_path / "expected.bin")
    (tmp_path / "txn-5.bin").write_bytes(first)
    (tmp_path / "txn-700.bin").write_bytes(second)
    (source + 1).tofile(tmp_path / "expected-5.bin")
    (source + 1).tofile(tmp_path / "expected-700.bin")
    manifest = {
        "cases": [
            {
                "name": "flm.layer",
                "xclbin": "kernel.xclbin",
                "export": "MLIR_AIE",
                "buffers": [
                    {
                        "name": "input",
                        "direction": "input",
                        "dtype": "int32",
                        "shape": [4],
                        "input": "input.bin",
                    },
                    {
                        "name": "output",
                        "direction": "output",
                        "dtype": "int32",
                        "shape": [4],
                        "expected": "expected.bin",
                        "rtol": 0.0,
                        "atol": 0.0,
                    },
                ],
                "dispatches": [
                    {
                        "name": "context_5",
                        "transaction": "txn-5.bin",
                        "expected": {"output": "expected-5.bin"},
                    },
                    {
                        "name": "context_700",
                        "transaction": "txn-700.bin",
                        "expected": {"output": "expected-700.bin"},
                    },
                ],
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    failures = runner.run_package(
        tmp_path,
        runtime_factory=FakeRuntime,
        lifecycle_rounds=2,
        resident_rounds=0,
    )

    assert failures == 0
    assert capsys.readouterr().out.endswith("flm.layer: PASS\n")
    submitted = [
        transaction
        for _xclbin, transaction, _name in FakeRuntime.instances[-1].transactions
    ]
    assert first in submitted
    assert second in submitted


def test_bfloat16_comparison_decodes_raw_storage():
    expected = np.array([0x3F80, 0x4000], dtype=np.uint16)
    actual = np.array([0x3F81, 0x4000], dtype=np.uint16)
    assert runner._compare(actual, expected, "bfloat16", rtol=0.01, atol=0.0) is None
    assert runner._compare(actual, expected, "bfloat16", rtol=0.0, atol=0.0) is not None


def test_inout_slice_uses_relative_l2_and_keeps_the_rest(tmp_path):
    class LeaveRuntime(FakeRuntime):
        def dispatch(self, executable, export_ordinal, buffers, constants=b""):
            self.dispatches.append((executable, export_ordinal, buffers))

    FakeRuntime.instances.clear()
    values = np.arange(4, dtype=np.int32)
    values.tofile(tmp_path / "x.bin")
    values.tofile(tmp_path / "x-expected.bin")
    (tmp_path / "kernel.xclbin").write_bytes(b"xclbin")
    (tmp_path / "txn.bin").write_bytes(b"txn")
    changed = values.copy()
    changed[:2] = (100, 100)
    changed.tofile(tmp_path / "dispatch-x.bin")
    manifest = {
        "cases": [
            {
                "name": "flm.layer-5",
                "xclbin": "kernel.xclbin",
                "export": "MLIR_AIE",
                "buffers": [
                    {
                        "name": "x",
                        "direction": "inout",
                        "dtype": "int32",
                        "shape": [4],
                        "input": "x.bin",
                        "expected": "x-expected.bin",
                    }
                ],
                "dispatches": [
                    {
                        "name": "context_5",
                        "transaction": "txn.bin",
                        "expected": {"x": "dispatch-x.bin"},
                        "compare_slices": {"x": [[0, 2]]},
                        "rel_l2": {"x": 0.01},
                    }
                ],
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        runner.run_package(
            tmp_path,
            runtime_factory=LeaveRuntime,
            lifecycle_rounds=1,
            resident_rounds=0,
        )
        == 1
    )
