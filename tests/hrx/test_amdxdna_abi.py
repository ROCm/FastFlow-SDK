# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ctypes
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ABI_PATH = ROOT / "tests" / "npu" / "hrx_api.py"
SPEC = importlib.util.spec_from_file_location("iron_hrx_abi", ABI_PATH)
abi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(abi)


class HrxAmdxdnaAbiTest(unittest.TestCase):
    def test_x64_public_record_layout(self):
        self.assertEqual(ctypes.sizeof(ctypes.c_void_p), 8)
        self.assertEqual(ctypes.sizeof(abi.HrxConstByteSpan), 16)
        self.assertEqual(ctypes.sizeof(abi.HrxAmdxdnaExecutableRun), 40)
        self.assertEqual(ctypes.sizeof(abi.HrxAmdxdnaExecutableEntryPoint), 72)
        self.assertEqual(ctypes.sizeof(abi.HrxAmdxdnaExecutableCreateParams), 48)
        self.assertEqual(abi.HrxAmdxdnaExecutableRun.transaction.offset, 8)
        self.assertEqual(abi.HrxAmdxdnaExecutableEntryPoint.runs.offset, 56)
        self.assertEqual(abi.HrxAmdxdnaExecutableCreateParams.xclbins.offset, 16)
        self.assertEqual(abi.HrxAmdxdnaExecutableCreateParams.entry_points.offset, 32)

    def test_builder_populates_records_and_retains_storage(self):
        xclbin = b"xclbin-bytes"
        transaction = b"transaction-bytes"
        description = abi.AmdxdnaExecutableDescription(
            xclbin, transaction, "MLIR_AIE"
        )

        params = description.params
        self.assertEqual(params.record_length, ctypes.sizeof(params))
        self.assertEqual(params.abi_version, 0)
        self.assertEqual(params.xclbin_count, 1)
        self.assertEqual(params.entry_point_count, 1)
        self.assertEqual(
            ctypes.string_at(params.xclbins[0].data, params.xclbins[0].data_length),
            xclbin,
        )

        entry = params.entry_points[0]
        self.assertEqual(entry.record_length, ctypes.sizeof(entry))
        self.assertEqual(entry.name.data[: entry.name.size], b"MLIR_AIE")
        self.assertEqual(entry.context_mode, abi.HRX_AMDXDNA_CONTEXT_MODE_CREATE)
        self.assertEqual(entry.run_count, 1)

        run = entry.runs[0]
        self.assertEqual(run.record_length, ctypes.sizeof(run))
        self.assertEqual(
            ctypes.string_at(run.transaction.data, run.transaction.data_length),
            transaction,
        )

    def test_builder_rejects_empty_borrowed_data(self):
        with self.assertRaises(ValueError):
            abi.AmdxdnaExecutableDescription(b"", b"txn", "MLIR_AIE")
        with self.assertRaises(ValueError):
            abi.AmdxdnaExecutableDescription(b"xclbin", b"", "MLIR_AIE")
        with self.assertRaises(ValueError):
            abi.AmdxdnaExecutableDescription(b"xclbin", b"txn", "")


if __name__ == "__main__":
    unittest.main()
