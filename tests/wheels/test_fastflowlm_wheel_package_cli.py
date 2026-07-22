# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import builtins
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import dev


class FastFlowLMWheelPackageCliTest(unittest.TestCase):
    def test_wheel_package_requires_installed_wheels(self):
        real_import = builtins.__import__

        def import_without_wheels(name, *args, **kwargs):
            if name in ("rocm_npu_runtime", "rocm_npu_fastflowlm"):
                raise ImportError(f"No module named {name}")
            return real_import(name, *args, **kwargs)

        args = SimpleNamespace(
            bundle=None,
            wheel_package=True,
            inference_iterations=1,
        )
        with mock.patch("builtins.__import__", side_effect=import_without_wheels):
            with self.assertRaises(SystemExit) as error:
                dev.cmd_test_fastflowlm(args)

        message = str(error.exception)
        self.assertIn("test-fastflowlm --wheel-package requires", message)
        self.assertIn("rocm-npu-runtime", message)
        self.assertIn("rocm-npu-fastflowlm", message)
        self.assertIn("verify-fastflowlm-wheels", message)

    def test_wheel_package_sets_installed_package_smoke_env(self):
        args = SimpleNamespace(
            bundle=None,
            wheel_package=True,
            inference_iterations=3,
        )
        modules = {
            "rocm_npu_runtime": types.ModuleType("rocm_npu_runtime"),
            "rocm_npu_fastflowlm": types.ModuleType("rocm_npu_fastflowlm"),
        }
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(dev, "sh", return_value=SimpleNamespace(returncode=7)) as sh,
            self.assertRaises(SystemExit) as error,
        ):
            dev.cmd_test_fastflowlm(args)

        self.assertEqual(error.exception.code, 7)
        command = sh.call_args.args[0]
        env = sh.call_args.kwargs["env"]
        self.assertEqual(command[0], sys.executable)
        self.assertIn("run_fastflowlm_hrx_smoke.py", str(command[1]))
        self.assertIn("--inference-iterations", command)
        self.assertIn("3", command)
        self.assertEqual(env["FASTFLOWLM_INSTALLED_PACKAGE"], "1")


if __name__ == "__main__":
    unittest.main()
