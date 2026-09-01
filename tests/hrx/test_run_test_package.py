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

    def __init__(self, correct=True):
        self.correct = correct
        self.loaded = []
        self.export_names = []
        self.buffers = []
        self.dispatches = []
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

    def dispatch(self, executable, export_ordinal, buffers):
        self.dispatches.append((executable, export_ordinal, buffers))
        buffers[1].array[:] = buffers[0].array + (1 if self.correct else 2)

    def synchronize(self):
        self.synchronize_count += 1

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
    assert capsys.readouterr().out == "vector_scalar_add: PASS\n"
    runtime = FakeRuntime.instances[-1]
    assert runtime.loaded == [b"XADX"] * 6
    assert runtime.export_names == ["MLIR_AIE"] * 6
    assert runtime.dispatches[0][1] == 9
    assert len(runtime.dispatches) == 15
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


def test_bfloat16_comparison_decodes_raw_storage():
    expected = np.array([0x3F80, 0x4000], dtype=np.uint16)
    actual = np.array([0x3F81, 0x4000], dtype=np.uint16)
    assert runner._compare(actual, expected, "bfloat16", rtol=0.01, atol=0.0) is None
    assert runner._compare(actual, expected, "bfloat16", rtol=0.0, atol=0.0) is not None
