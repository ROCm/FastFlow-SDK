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
from build_tools.artifact import write_sha256_sidecar
from build_tools.dev import host_testing
from build_tools.fastflow_sdk_linux_wheels import (
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
            find_fastflow_sdk_wheels=mock.Mock(return_value=wheels),
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
            mock.patch.object(
                fastflowlm_package, "verify_hrx_package_split"
            ) as hrx_split,
        ):
            fastflowlm_package.cmd_verify_fastflowlm_wheels(args)

        install = sh.call_args_list[1].args[0]
        assert "--find-links" in install
        assert wheel_directory in install
        assert fastflowlm_wheel in install
        assert runtime_wheel not in install
        implementation.verify_wheels_do_not_have_text_path_references.assert_called_once()
        hrx_split.assert_called_once_with(wheel_directory, wheels)

    def test_wheel_package_requires_installed_wheels(self):
        real_import = builtins.__import__

        def import_without_wheels(name, *args, **kwargs):
            if name in ("fastflow_sdk_runtime", "fastflow_sdk_fastflowlm"):
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
        assert "fastflow-sdk-runtime" in message
        assert "fastflow-sdk-fastflowlm" in message
        assert "verify-fastflowlm-wheels" in message

    def test_wheel_package_sets_installed_package_smoke_env(self):
        args = SimpleNamespace(
            bundle=None,
            wheel_package=True,
            inference_iterations=3,
        )
        modules = {
            "fastflow_sdk_runtime": types.ModuleType("fastflow_sdk_runtime"),
            "fastflow_sdk_fastflowlm": types.ModuleType("fastflow_sdk_fastflowlm"),
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

    def test_find_archive_ignores_compiler_product(self, tmp_path):
        runtime = tmp_path / "fastflow-sdk-linux-x86_64.tar.zst"
        compiler = tmp_path / "fastflow-sdk-compiler-linux-x86_64.tar.zst"
        runtime.write_bytes(b"runtime")
        compiler.write_bytes(b"compiler")
        write_sha256_sidecar(runtime)
        write_sha256_sidecar(compiler)

        assert fastflowlm_package._find_fastflowlm_archive(tmp_path) == runtime
