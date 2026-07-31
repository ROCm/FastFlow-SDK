# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path, PureWindowsPath
from unittest import mock

import pytest

from build_tools.dev import environment

pytestmark = pytest.mark.presubmit


class TestEnvironment:
    def test_venv_tools_follow_explicit_platform_conventions(self):
        assert (
            environment.venv_py(Path("test-venv"), windows=True)
            == Path("test-venv") / "Scripts" / "python.exe"
        )
        assert (
            environment.uv_bin(Path("test-venv"), windows=False)
            == Path("test-venv") / "bin" / "uv"
        )

    def test_cmake_path_uses_forward_slashes(self):
        assert (
            environment.cmake_path(PureWindowsPath(r"C:\source\project"))
            == "C:/source/project"
        )

    def test_windows_path_lookup_and_replacement_are_case_insensitive(self):
        original = {"PATH": "first", "path": "second", "OTHER": "value"}
        assert environment.env_path(original, windows=True) == "first"
        environment.set_env_path(original, "replacement", windows=True)
        assert original == {"Path": "replacement", "OTHER": "value"}

    def test_prepend_path_ignores_empty_entries(self):
        selected = {"PATH": "existing"}
        environment.prepend_env_path(
            selected, ["first", "", Path("second")], windows=False
        )
        assert selected["PATH"] == os.pathsep.join(("first", "second", "existing"))

    def test_python_install_uses_uv_and_refreshes_selected_packages(self, tmp_path):
        uv = tmp_path / "tools" / "uv"
        uv.parent.mkdir()
        uv.touch()
        with (
            mock.patch.object(environment, "uv_bin", return_value=uv),
            mock.patch.object(environment.subprocess, "run") as run,
        ):
            environment.python_install(
                Path("python"),
                "example",
                refresh_packages=("one", "two"),
            )

        command = run.call_args.args[0]
        assert command == [
            str(uv),
            "pip",
            "install",
            "--python",
            "python",
            "--refresh-package",
            "one",
            "--refresh-package",
            "two",
            "example",
        ]
        assert run.call_args.kwargs["check"]

    def test_python_install_falls_back_to_pip(self, tmp_path):
        with (
            mock.patch.object(
                environment,
                "uv_bin",
                return_value=tmp_path / "missing-uv",
            ),
            mock.patch.object(environment.subprocess, "run") as run,
        ):
            environment.python_install("python", "--upgrade", "example")

        assert run.call_args.args[0] == [
            "python",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            "--upgrade",
            "example",
        ]

    def test_python_install_removes_stale_ssl_cert_file(self, tmp_path):
        missing = str(tmp_path / "missing-rns-certificate.pem")
        with (
            mock.patch.object(
                environment,
                "uv_bin",
                return_value=tmp_path / "missing-uv",
            ),
            mock.patch.dict(
                environment.os.environ,
                {"SSL_CERT_FILE": missing},
                clear=True,
            ),
            mock.patch.object(environment.subprocess, "run") as run,
        ):
            environment.python_install("python", "example")

        assert "SSL_CERT_FILE" not in run.call_args.kwargs["env"]

    def test_python_install_keeps_existing_ssl_cert_file(self, tmp_path):
        certificate = tmp_path / "certificate.pem"
        certificate.touch()
        with (
            mock.patch.object(
                environment, "uv_bin", return_value=tmp_path / "missing-uv"
            ),
            mock.patch.dict(
                environment.os.environ,
                {"SSL_CERT_FILE": str(certificate)},
                clear=True,
            ),
            mock.patch.object(environment.subprocess, "run") as run,
        ):
            environment.python_install("python", "example")

        assert run.call_args.kwargs["env"]["SSL_CERT_FILE"] == str(certificate)

    def test_rustc_version_parses_release_suffix(self):
        result = mock.Mock(returncode=0, stdout="rustc 1.82.0-nightly\n")
        with (
            mock.patch.object(environment, "which", return_value="rustc"),
            mock.patch.object(environment.subprocess, "run", return_value=result),
        ):
            assert environment.rustc_version() == (1, 82, 0)

    def test_cmake_source_override_is_empty_when_unset(self):
        with mock.patch.dict(environment.os.environ, {}, clear=True):
            assert environment.cmake_source_override("TEST_SOURCE") == []
