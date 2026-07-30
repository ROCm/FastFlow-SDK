# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import builtins
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from build_tools import fastflowlm_package
from build_tools.rocm_npu_linux_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
)
from build_tools.dev import testing


class FastFlowLMWheelPackageCliTest(unittest.TestCase):
    def test_verify_resolves_runtime_as_fastflowlm_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            wheel_directory = Path(temporary)
            runtime_wheel = wheel_directory / "runtime.whl"
            fastflowlm_wheel = wheel_directory / "fastflowlm.whl"
            wheels = {
                RUNTIME_DISTRIBUTION: runtime_wheel,
                FASTFLOWLM_DISTRIBUTION: fastflowlm_wheel,
            }
            args = SimpleNamespace(
                dir=str(wheel_directory),
                venv=str(wheel_directory / "venv"),
            )
            with (
                mock.patch.object(
                    fastflowlm_package,
                    "find_rocm_npu_wheels",
                    return_value=wheels,
                ),
                mock.patch.object(fastflowlm_package, "verify_wheel_checksums"),
                mock.patch.object(
                    fastflowlm_package,
                    "verify_wheels_do_not_have_text_path_references",
                ),
                mock.patch.object(
                    fastflowlm_package, "validate_directory_target"
                ),
                mock.patch.object(
                    fastflowlm_package,
                    "_venv_python",
                    return_value=Path("python"),
                ),
                mock.patch.object(fastflowlm_package, "sh") as sh,
            ):
                fastflowlm_package.cmd_verify_fastflowlm_wheels(args)

        install = sh.call_args_list[1].args[0]
        self.assertIn("--find-links", install)
        self.assertIn(wheel_directory, install)
        self.assertIn(fastflowlm_wheel, install)
        self.assertNotIn(runtime_wheel, install)

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
                testing.cmd_test_fastflowlm(args)

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
            mock.patch.object(
                testing.environment,
                "sh",
                return_value=SimpleNamespace(returncode=7),
            ) as sh,
            self.assertRaises(SystemExit) as error,
        ):
            testing.cmd_test_fastflowlm(args)

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
