# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NPU_TEST_DIR = ROOT / "tests" / "npu"
sys.path.insert(0, str(NPU_TEST_DIR))
MODULE_PATH = NPU_TEST_DIR / "run_test_package.py"
SPEC = importlib.util.spec_from_file_location("npu_package_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeExecutable:
    def __init__(self, runtime):
        self.runtime = runtime
        self.closed = False

    def lookup_export(self, name):
        self.runtime.export_names.append(name)
        return 9

    def close(self):
        self.closed = True


class FakeBuffer:
    def __init__(self, shape, dtype):
        self.array = np.empty(shape, dtype=dtype)
        self.flush_count = 0
        self.invalidate_count = 0
        self.closed = False

    def flush(self):
        self.flush_count += 1

    def invalidate(self):
        self.invalidate_count += 1

    def close(self):
        self.closed = True


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
        self.__class__.instances.append(self)

    def load_xadx(self, data):
        self.loaded.append(data)
        self.executable = FakeExecutable(self)
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


class PackageRunnerTest(unittest.TestCase):
    def setUp(self):
        FakeRuntime.instances.clear()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.package_root = Path(self.temporary_directory.name)
        (self.package_root / "kernel.xadx").write_bytes(b"XADX")
        source = np.arange(-4, 4, dtype=np.int32)
        expected = source + 1
        source.tofile(self.package_root / "input.bin")
        expected.tofile(self.package_root / "expected.bin")
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
        (self.package_root / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_runs_package_with_fake_runtime(self):
        output = io.StringIO()
        failures = runner.run_package(
            self.package_root, runtime_factory=FakeRuntime, stdout=output
        )

        self.assertEqual(failures, 0)
        self.assertEqual(output.getvalue(), "vector_scalar_add: PASS\n")
        runtime = FakeRuntime.instances[-1]
        self.assertEqual(runtime.loaded, [b"XADX"])
        self.assertEqual(runtime.export_names, ["MLIR_AIE"])
        self.assertEqual(runtime.dispatches[0][1], 9)
        self.assertEqual(runtime.synchronize_count, 1)
        self.assertTrue(runtime.closed)
        self.assertTrue(runtime.executable.closed)
        self.assertTrue(all(buffer.closed for buffer in runtime.buffers))
        self.assertEqual([buffer.flush_count for buffer in runtime.buffers], [1, 1])
        self.assertEqual(
            [buffer.invalidate_count for buffer in runtime.buffers], [0, 1]
        )

    def test_reports_output_mismatch_and_returns_failure(self):
        output = io.StringIO()
        failures = runner.run_package(
            self.package_root,
            runtime_factory=lambda: FakeRuntime(correct=False),
            stdout=output,
        )

        self.assertEqual(failures, 1)
        self.assertIn("vector_scalar_add: FAIL", output.getvalue())
        self.assertIn("8/8 mismatches", output.getvalue())

    def test_bfloat16_comparison_decodes_raw_storage(self):
        expected = np.array([0x3F80, 0x4000], dtype=np.uint16)
        actual = np.array([0x3F81, 0x4000], dtype=np.uint16)
        self.assertIsNone(
            runner._compare(
                actual, expected, "bfloat16", rtol=0.01, atol=0.0
            )
        )
        self.assertIsNotNone(
            runner._compare(
                actual, expected, "bfloat16", rtol=0.0, atol=0.0
            )
        )


if __name__ == "__main__":
    unittest.main()

