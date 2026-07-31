# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from build_tools import fastflowlm_package
from build_tools.rocm_npu_linux_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
)
from build_tools.dev import testing

pytestmark = [pytest.mark.compiler_stack, pytest.mark.fastflowlm]


class TestFastFlowLMWheelPackageCli:
    def test_verify_resolves_runtime_as_fastflowlm_dependency(self, tmp_path):
        wheel_directory = tmp_path
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
            mock.patch.object(fastflowlm_package, "validate_directory_target"),
            mock.patch.object(
                fastflowlm_package,
                "_venv_python",
                return_value=Path("python"),
            ),
            mock.patch.object(fastflowlm_package, "sh") as sh,
        ):
            fastflowlm_package.cmd_verify_fastflowlm_wheels(args)

        install = sh.call_args_list[1].args[0]
        assert "--find-links" in install
        assert wheel_directory in install
        assert fastflowlm_wheel in install
        assert runtime_wheel not in install

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
            with pytest.raises(SystemExit) as error:
                testing.cmd_test_fastflowlm(args)

        message = str(error.value)
        assert "test-fastflowlm --wheel-package requires" in message
        assert "rocm-npu-runtime" in message
        assert "rocm-npu-fastflowlm" in message
        assert "verify-fastflowlm-wheels" in message

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
            pytest.raises(SystemExit) as error,
        ):
            testing.cmd_test_fastflowlm(args)

        assert error.value.code == 7
        command = sh.call_args.args[0]
        env = sh.call_args.kwargs["env"]
        assert command[0] == sys.executable
        assert "run_fastflowlm_hrx_smoke.py" in str(command[1])
        assert "--inference-iterations" in command
        assert "3" in command
        assert env["FASTFLOWLM_INSTALLED_PACKAGE"] == "1"
