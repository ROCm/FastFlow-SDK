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
from build_tools.dev import host_testing
from build_tools.rocm_npu_linux_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
)

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


class TestFastFlowLMWheelPackageCli:
    def test_package_dispatches_to_platform_wheel_implementation(self):
        wheels = {
            RUNTIME_DISTRIBUTION: Path("runtime.whl"),
            FASTFLOWLM_DISTRIBUTION: Path("fastflowlm.whl"),
        }
        implementation = SimpleNamespace(
            build_runtime_fastflowlm_wheels=mock.Mock(return_value=wheels)
        )
        with (
            mock.patch.object(fastflowlm_package, "IS_WINDOWS", True),
            mock.patch.object(
                fastflowlm_package,
                "_wheel_implementation",
                return_value=implementation,
            ),
            mock.patch.object(fastflowlm_package, "_verify_fastflowlm_fresh_installs"),
            mock.patch.object(
                fastflowlm_package,
                "stage_runtime_bundle",
                return_value=Path("bundle"),
            ),
        ):
            assert fastflowlm_package.package_fastflowlm_wheels("1.2.3") == wheels

        implementation.build_runtime_fastflowlm_wheels.assert_called_once_with(
            Path("bundle"),
            fastflowlm_package.WINDOWS_WHEEL_STAGE,
            fastflowlm_package.FASTFLOWLM_ARTIFACTS,
            "1.2.3",
        )

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
        implementation = SimpleNamespace(
            find_rocm_npu_wheels=mock.Mock(return_value=wheels),
            verify_wheels_do_not_have_text_path_references=mock.Mock(),
        )
        with (
            mock.patch.object(
                fastflowlm_package,
                "_wheel_implementation",
                return_value=implementation,
            ),
            mock.patch.object(fastflowlm_package, "verify_wheel_checksums"),
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
        implementation.verify_wheels_do_not_have_text_path_references.assert_called_once()

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
                host_testing.cmd_test_fastflowlm(args)

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
                host_testing.environment,
                "sh",
                return_value=SimpleNamespace(returncode=7),
            ) as sh,
            pytest.raises(SystemExit) as error,
        ):
            host_testing.cmd_test_fastflowlm(args)

        assert error.value.code == 7
        command = sh.call_args.args[0]
        env = sh.call_args.kwargs["env"]
        assert command[0] == sys.executable
        assert "run_fastflowlm_hrx_smoke.py" in str(command[1])
        assert "--inference-iterations" in command
        assert "3" in command
        assert env["FASTFLOWLM_INSTALLED_PACKAGE"] == "1"
