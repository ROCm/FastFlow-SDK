# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import errno
import os
import stat
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import dev  # noqa: E402


class LinuxNpuProbeDiagnosticTest(unittest.TestCase):
    def _mock_devnode_context(self, kernel_release: str, srcversion: str):
        sys_node = Path("/sys/class/accel/accel0")
        module_dir = Path("/sys/module/amdxdna")
        st = SimpleNamespace(st_mode=stat.S_IFCHR | 0o660, st_rdev=os.makedev(261, 0))

        def read_sysfs(path: Path):
            text = str(path)
            if text == "/sys/class/accel/accel0/dev":
                return "261:0"
            if text == "/sys/class/accel/accel0/device/iommu_group/type":
                return "identity"
            if text == "/sys/module/amdxdna/srcversion":
                return srcversion
            return None

        return (
            mock.patch.object(dev, "_sys_accel_nodes", return_value=[sys_node]),
            mock.patch.object(dev, "_read_sysfs_text", side_effect=read_sysfs),
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(Path, "stat", return_value=st),
            mock.patch.object(dev.os, "access", return_value=True),
            mock.patch.object(
                dev.os, "uname", return_value=SimpleNamespace(release=kernel_release)
            ),
        )

    def test_known_ubuntu_617_sva_regression_tuple(self):
        self.assertTrue(
            dev._known_amdxdna_sva_regression(
                "6.17.0-22-generic", "2DBDA75956CAFA9D029EA89"
            )
        )
        self.assertFalse(
            dev._known_amdxdna_sva_regression(
                "6.17.0-20-generic", "2DBDA75956CAFA9D029EA89"
            )
        )

    def test_healthy_linux_stack_accepts_successful_accel_open(self):
        patches = self._mock_devnode_context(
            "7.0.0-28-generic", "E4FC5AFFF53D21BC6AFDAE3"
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            mock.patch.object(dev.os, "open", return_value=7) as open_mock,
            mock.patch.object(dev.os, "close") as close_mock,
            mock.patch("builtins.print"),
        ):
            rc, usable = dev._probe_npu_devnodes()

        self.assertEqual(rc, 0)
        self.assertEqual(usable, [Path("/dev/accel/accel0")])
        open_mock.assert_called_once_with(Path("/dev/accel/accel0"), mock.ANY)
        close_mock.assert_called_once_with(7)

    def test_known_bad_stack_preopen_reports_sva_failure(self):
        patches = self._mock_devnode_context(
            "6.17.0-22-generic", "2DBDA75956CAFA9D029EA89"
        )
        open_error = OSError(errno.EOPNOTSUPP, os.strerror(errno.EOPNOTSUPP))
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            mock.patch.object(dev.os, "open", side_effect=open_error),
            mock.patch("builtins.print") as print_mock,
        ):
            rc, usable = dev._probe_npu_devnodes()

        self.assertEqual(rc, 1)
        self.assertEqual(usable, [])
        output = "\n".join(str(call.args[0]) for call in print_mock.call_args_list)
        self.assertIn("open(O_RDWR) failed", output)
        self.assertIn("SVA bind device failed", output)
        self.assertIn("known-bad Ubuntu 6.17.0-22", output)


if __name__ == "__main__":
    unittest.main()
