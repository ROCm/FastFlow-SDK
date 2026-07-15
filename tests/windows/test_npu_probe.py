# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_tools.windows_npu import (
    device_is_healthy,
    driver_firmware_packages,
    parse_device_query,
)

class WindowsFirmwarePackageTest(unittest.TestCase):
    def test_reads_release_and_hash_without_xrt(self):
        # Real AMD firmware headers place Release immediately after a hex hash.
        content = b"deadbeefRelease 1.1.2.64\0payload"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            driver = root / "ipustack.sys"
            driver.write_bytes(b"driver")
            firmware = root / "ipu_3_pasid.sbin"
            firmware.write_bytes(content)

            packages = driver_firmware_packages(driver)

        self.assertEqual(len(packages), 1)
        self.assertEqual(packages[0]["name"], "ipu_3_pasid.sbin")
        self.assertEqual(packages[0]["release"], "1.1.2.64")
        self.assertEqual(packages[0]["sha256"], hashlib.sha256(content).hexdigest())

    def test_reports_unknown_for_unversioned_firmware(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            driver = root / "ipustack.sys"
            driver.write_bytes(b"driver")
            (root / "ipu_pasid.sbin").write_bytes(b"no release metadata")

            packages = driver_firmware_packages(driver)

        self.assertEqual(packages[0]["release"], "unknown")


class WindowsDeviceQueryTest(unittest.TestCase):
    def test_preserves_multiple_mcdm_devices(self):
        devices = [
            {"device_id": "PCI\\VEN_1022&DEV_17F0", "status": "OK", "problem": "0"},
            {"device_id": "PCI\\VEN_1022&DEV_1502", "status": "OK", "problem": "CM_PROB_NONE"},
        ]
        self.assertEqual(parse_device_query(json.dumps(devices)), devices)
        self.assertTrue(all(device_is_healthy(device) for device in devices))

    def test_rejects_empty_or_non_object_results(self):
        for output in ("[]", "null", "[1]"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_device_query(output)

    def test_unhealthy_pnp_state_is_reported(self):
        self.assertFalse(device_is_healthy({"status": "Error", "problem": "22"}))


if __name__ == "__main__":
    unittest.main()
