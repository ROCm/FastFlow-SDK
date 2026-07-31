# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from build_tools import rocm_npu_windows_wheel_cli as cli
from build_tools.rocm_npu_windows_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
)

pytestmark = [pytest.mark.compiler_stack, pytest.mark.fastflowlm]


class TestRocmNpuWindowsWheelCli:
    def test_package_uses_explicit_product_layout_without_dev_module(self):
        wheels = {
            RUNTIME_DISTRIBUTION: Path("runtime.whl"),
            FASTFLOWLM_DISTRIBUTION: Path("fastflowlm.whl"),
        }
        with (
            mock.patch.object(cli, "_verify_fresh_installs") as verify,
            mock.patch.object(
                cli, "stage_runtime_bundle", return_value=Path("bundle")
            ) as stage,
            mock.patch.object(
                cli, "build_runtime_fastflowlm_wheels", return_value=wheels
            ) as build,
        ):
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
        assert "dev" not in cli.__dict__

    def test_probe_passes_installed_package_library_directly(self, tmp_path):
        library = tmp_path / "hrx.dll"
        runtime = SimpleNamespace(libhrx_path=lambda: library)
        with (
            mock.patch.dict(sys.modules, {"rocm_npu_runtime": runtime}),
            mock.patch.object(cli, "probe_runtime", return_value=0) as probe,
        ):
            assert cli.probe_runtime_package() == 0
        probe.assert_called_once_with(library)

    def test_verify_installs_fastflowlm_and_resolves_runtime_transitively(
        self, tmp_path
    ):
        directory = tmp_path
        venv = directory / "venv"
        runtime_wheel = directory / "runtime.whl"
        fastflowlm_wheel = directory / "fastflowlm.whl"
        wheels = {
            RUNTIME_DISTRIBUTION: runtime_wheel,
            FASTFLOWLM_DISTRIBUTION: fastflowlm_wheel,
        }
        with (
            mock.patch.object(cli, "find_rocm_npu_wheels", return_value=wheels),
            mock.patch.object(cli, "verify_wheel_checksums"),
            mock.patch.object(cli, "verify_wheels_do_not_have_text_path_references"),
            mock.patch.object(cli, "validate_directory_target"),
            mock.patch.object(cli.subprocess, "run") as run,
        ):
            cli.verify_wheels(directory, venv)

        install = run.call_args_list[1].args[0]
        assert "--find-links" in install
        assert directory in install
        assert fastflowlm_wheel in install
        assert runtime_wheel not in install
