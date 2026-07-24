# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from build_tools import rocm_npu_windows_wheel_cli as cli
from build_tools.rocm_npu_windows_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
)


class RocmNpuWindowsWheelCliTest(unittest.TestCase):
    def test_package_uses_explicit_product_layout_without_dev_module(self):
        wheels = {
            RUNTIME_DISTRIBUTION: Path("runtime.whl"),
            FASTFLOWLM_DISTRIBUTION: Path("fastflowlm.whl"),
        }
        with mock.patch.object(cli, "_verify_fresh_installs") as verify, \
             mock.patch.object(cli, "stage_runtime_bundle", return_value=Path("bundle")) as stage, \
             mock.patch.object(
                 cli, "build_runtime_fastflowlm_wheels", return_value=wheels
             ) as build:
            cli.package_wheels("1.2.3")
        verify.assert_called_once_with()
        stage.assert_called_once_with(
            cli.ROOT,
            cli.FASTFLOWLM_INSTALL,
            cli.HRX_INSTALL,
            cli.FASTFLOWLM_BUNDLE_STAGE,
        )
        build.assert_called_once_with(
            Path("bundle"), cli.WHEEL_STAGE, cli.FASTFLOWLM_ARTIFACTS, "1.2.3"
        )
        self.assertNotIn("dev", cli.__dict__)

    def test_probe_passes_installed_package_library_directly(self):
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / "hrx.dll"
            runtime = SimpleNamespace(libhrx_path=lambda: library)
            with mock.patch.dict(sys.modules, {"rocm_npu_runtime": runtime}), \
                 mock.patch.object(cli, "probe_runtime", return_value=0) as probe:
                self.assertEqual(cli.probe_runtime_package(), 0)
            probe.assert_called_once_with(library)


if __name__ == "__main__":
    unittest.main()
