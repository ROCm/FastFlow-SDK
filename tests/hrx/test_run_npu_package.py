# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]

NPU_TESTS = Path(__file__).resolve().parents[1] / "npu"
sys.path.insert(0, str(NPU_TESTS))
import run_test_package as runner  # noqa: E402


class FakeExecutable:
    def __init__(self, runtime, name):
        self.runtime = runtime
        self.name = name

    def lookup_export(self, name):
        self.runtime.events.append(("lookup", self.name, name))
        return 7

    def close(self):
        self.runtime.events.append(("executable-close", self.name))


class FakeBuffer:
    def __init__(self, runtime, shape, dtype):
        self.runtime = runtime
        self.array = np.zeros(shape, dtype=dtype)

    def flush(self):
        self.runtime.events.append(("flush", self.array.size))

    def invalidate(self):
        self.runtime.events.append(("invalidate", self.array.size))

    def close(self):
        self.runtime.events.append(("buffer-close", self.array.size))


class FakeRuntime:
    last = None
    wrong_output = False
    fail_dispatch = False

    def __init__(self):
        FakeRuntime.last = self
        self.events = []

    def load_xadx(self, data):
        name = data.decode()
        self.events.append(("load", name))
        return FakeExecutable(self, name)

    def allocate_buffer(self, shape, dtype):
        return FakeBuffer(self, shape, dtype)

    def dispatch(self, executable, ordinal, buffers):
        self.events.append(("dispatch", executable.name))
        if self.fail_dispatch:
            raise RuntimeError("injected dispatch failure")
        if not bool(np.all(buffers[-1].array == 0)):
            raise AssertionError("output was not reset before dispatch")
        buffers[-1].array[:] = buffers[0].array + (2 if self.wrong_output else 1)

    def synchronize(self):
        self.events.append(("synchronize",))

    def close(self):
        self.events.append(("runtime-close",))


class TestRunNpuPackage:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        FakeRuntime.wrong_output = False
        FakeRuntime.fail_dispatch = False
        self.root = tmp_path
        cases = []
        for name in ("alpha", "beta"):
            (self.root / f"{name}.xadx").write_bytes(name.encode())
            source = np.arange(4, dtype=np.int32)
            expected = source + 1
            source.tofile(self.root / f"{name}-input.bin")
            expected.tofile(self.root / f"{name}-expected.bin")
            cases.append(
                {
                    "name": name,
                    "xadx": f"{name}.xadx",
                    "export": "MLIR_AIE",
                    "buffers": [
                        {
                            "name": "input",
                            "direction": "input",
                            "dtype": "<i4",
                            "shape": [4],
                            "input": f"{name}-input.bin",
                        },
                        {
                            "name": "output",
                            "direction": "output",
                            "dtype": "<i4",
                            "shape": [4],
                            "expected": f"{name}-expected.bin",
                        },
                    ],
                }
            )
        (self.root / "manifest.json").write_text(
            json.dumps({"cases": cases}), encoding="utf-8"
        )

    def test_lifecycle_and_resident_dispatch_counts_and_order(self):
        failed = runner.run_package(
            self.root,
            runtime_factory=FakeRuntime,
            stdout=io.StringIO(),
            lifecycle_rounds=2,
            resident_rounds=3,
        )
        assert failed == 0
        dispatches = [
            event[1] for event in FakeRuntime.last.events if event[0] == "dispatch"
        ]
        assert dispatches == [
            "alpha",
            "beta",
            "alpha",
            "beta",
            "alpha",
            "beta",
            "alpha",
            "beta",
            "alpha",
            "beta",
        ]

    def test_selected_case_only(self):
        runner.run_package(
            self.root,
            ["beta"],
            runtime_factory=FakeRuntime,
            stdout=io.StringIO(),
            lifecycle_rounds=1,
            resident_rounds=1,
        )
        dispatches = [
            event[1] for event in FakeRuntime.last.events if event[0] == "dispatch"
        ]
        assert dispatches == ["beta", "beta"]

    def test_cleanup_occurs_after_dispatch_failure(self):
        FakeRuntime.fail_dispatch = True
        output = io.StringIO()
        failed = runner.run_package(
            self.root,
            ["alpha"],
            runtime_factory=FakeRuntime,
            stdout=output,
            lifecycle_rounds=1,
            resident_rounds=0,
        )
        assert failed == 1
        assert "injected dispatch failure" in output.getvalue()
        event_names = [event[0] for event in FakeRuntime.last.events]
        assert "buffer-close" in event_names
        assert "executable-close" in event_names
        assert event_names[-1] == "runtime-close"

    def test_numeric_mismatch_is_reported(self):
        FakeRuntime.wrong_output = True
        output = io.StringIO()
        failed = runner.run_package(
            self.root,
            ["alpha"],
            runtime_factory=FakeRuntime,
            stdout=output,
            lifecycle_rounds=1,
            resident_rounds=0,
        )
        assert failed == 1
        assert "mismatches" in output.getvalue()

    def test_unknown_selected_case_is_rejected(self):
        with pytest.raises(ValueError):
            runner.run_package(self.root, ["missing"], runtime_factory=FakeRuntime)
