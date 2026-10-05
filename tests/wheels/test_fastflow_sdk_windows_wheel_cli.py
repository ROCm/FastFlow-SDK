# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path
from unittest import mock

import pytest

from build_tools import fastflow_sdk_windows_wheel_cli as cli
from build_tools.fastflow_sdk_windows_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
)

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


class TestFastflowSdkWindowsWheelCli:
    def test_package_dispatches_to_product_orchestration_without_dev_module(self):
        wheels = {
            RUNTIME_DISTRIBUTION: Path("runtime.whl"),
            FASTFLOWLM_DISTRIBUTION: Path("fastflowlm.whl"),
        }
        with (
            mock.patch.object(cli, "_is_windows_host", return_value=True),
            mock.patch.object(
                sys,
                "argv",
                ["fastflow_sdk_windows_wheel_cli.py", "package", "--version", "1.2.3"],
            ),
            mock.patch.object(
                cli, "package_fastflowlm_wheels", return_value=wheels
            ) as package,
        ):
            assert cli.main() == 0

        package.assert_called_once_with("1.2.3")
        assert "dev" not in cli.__dict__

    def test_verify_dispatches_paths_to_product_orchestration(self, tmp_path):
        wheel_directory = tmp_path / "wheels"
        venv = tmp_path / "venv"
        with (
            mock.patch.object(cli, "_is_windows_host", return_value=True),
            mock.patch.object(
                sys,
                "argv",
                [
                    "fastflow_sdk_windows_wheel_cli.py",
                    "verify",
                    "--dir",
                    str(wheel_directory),
                    "--venv",
                    str(venv),
                ],
            ),
            mock.patch.object(cli, "verify_fastflowlm_wheels") as verify,
        ):
            assert cli.main() == 0

        verify.assert_called_once_with(wheel_directory, venv)

    def test_probe_and_smoke_preserve_exit_codes(self):
        with (
            mock.patch.object(cli, "_is_windows_host", return_value=True),
            mock.patch.object(sys, "argv", ["cli", "probe"]),
            mock.patch.object(cli, "probe_runtime_package", return_value=3),
        ):
            assert cli.main() == 3
        with (
            mock.patch.object(cli, "_is_windows_host", return_value=True),
            mock.patch.object(
                sys, "argv", ["cli", "smoke", "--inference-iterations", "4"]
            ),
            mock.patch.object(cli, "smoke_fastflowlm", return_value=5) as smoke,
        ):
            assert cli.main() == 5
        smoke.assert_called_once_with(4)
