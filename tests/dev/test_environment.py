# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

from build_tools.dev import environment


class EnvironmentTest(unittest.TestCase):
    def test_venv_tools_follow_explicit_platform_conventions(self):
        self.assertEqual(
            environment.venv_py(Path("test-venv"), windows=True),
            Path("test-venv") / "Scripts" / "python.exe",
        )
        self.assertEqual(
            environment.uv_bin(Path("test-venv"), windows=False),
            Path("test-venv") / "bin" / "uv",
        )

    def test_cmake_path_uses_forward_slashes(self):
        self.assertEqual(
            environment.cmake_path(PureWindowsPath(r"C:\source\project")),
            "C:/source/project",
        )

    def test_windows_path_lookup_and_replacement_are_case_insensitive(self):
        original = {"PATH": "first", "path": "second", "OTHER": "value"}
        self.assertEqual(environment.env_path(original, windows=True), "first")
        environment.set_env_path(original, "replacement", windows=True)
        self.assertEqual(original, {"Path": "replacement", "OTHER": "value"})

    def test_prepend_path_ignores_empty_entries(self):
        selected = {"PATH": "existing"}
        environment.prepend_env_path(
            selected, ["first", "", Path("second")], windows=False
        )
        self.assertEqual(
            selected["PATH"],
            os.pathsep.join(("first", "second", "existing")),
        )

    def test_python_install_uses_uv_and_refreshes_selected_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary)
            uv = venv / "tools" / "uv"
            uv.parent.mkdir()
            uv.touch()
            with mock.patch.object(environment, "uv_bin", return_value=uv), \
                    mock.patch.object(environment.subprocess, "run") as run:
                environment.python_install(
                    Path("python"),
                    "example",
                    refresh_packages=("one", "two"),
                )

        command = run.call_args.args[0]
        self.assertEqual(
            command,
            [
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
            ],
        )
        self.assertTrue(run.call_args.kwargs["check"])

    def test_python_install_falls_back_to_pip(self):
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(
                    environment,
                    "uv_bin",
                    return_value=Path(temporary) / "missing-uv",
                ), \
                mock.patch.object(environment.subprocess, "run") as run:
            environment.python_install("python", "--upgrade", "example")

        self.assertEqual(
            run.call_args.args[0],
            [
                "python",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-compile",
                "--upgrade",
                "example",
            ],
        )

    def test_python_install_removes_stale_ssl_cert_file(self):
        missing = str(Path(tempfile.gettempdir()) / "missing-rns-certificate.pem")
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(
                    environment,
                    "uv_bin",
                    return_value=Path(temporary) / "missing-uv",
                ), \
                mock.patch.dict(
                    environment.os.environ,
                    {"SSL_CERT_FILE": missing},
                    clear=True,
                ), \
                mock.patch.object(environment.subprocess, "run") as run:
            environment.python_install("python", "example")

        self.assertNotIn("SSL_CERT_FILE", run.call_args.kwargs["env"])

    def test_python_install_keeps_existing_ssl_cert_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            certificate = root / "certificate.pem"
            certificate.touch()
            with mock.patch.object(
                    environment, "uv_bin", return_value=root / "missing-uv"
                ), \
                    mock.patch.dict(
                        environment.os.environ,
                        {"SSL_CERT_FILE": str(certificate)},
                        clear=True,
                    ), \
                    mock.patch.object(environment.subprocess, "run") as run:
                environment.python_install("python", "example")

        self.assertEqual(
            run.call_args.kwargs["env"]["SSL_CERT_FILE"],
            str(certificate),
        )

    def test_rustc_version_parses_release_suffix(self):
        result = mock.Mock(returncode=0, stdout="rustc 1.82.0-nightly\n")
        with mock.patch.object(environment, "which", return_value="rustc"), \
                mock.patch.object(
                    environment.subprocess, "run", return_value=result
                ):
            self.assertEqual(environment.rustc_version(), (1, 82, 0))

    def test_cmake_source_override_is_empty_when_unset(self):
        with mock.patch.dict(environment.os.environ, {}, clear=True):
            self.assertEqual(environment.cmake_source_override("TEST_SOURCE"), [])


if __name__ == "__main__":
    unittest.main()
