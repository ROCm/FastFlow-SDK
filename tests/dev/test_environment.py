# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import time
from pathlib import Path, PureWindowsPath
from unittest import mock

import pytest

from build_tools import environment

pytestmark = pytest.mark.presubmit


class TestEnvironment:
    def test_environment_is_a_shared_foundation(self):
        assert environment.__name__ == "build_tools.environment"
        assert not (
            environment.paths.ROOT / "build_tools" / "dev" / "environment.py"
        ).exists()

    def test_shell_commands_have_a_generous_default_timeout(self):
        with mock.patch.object(environment, "run") as run:
            environment.sh(["tool", "argument"])

        assert run.call_args.kwargs["timeout"] == (
            environment.DEFAULT_COMMAND_TIMEOUT_SECONDS
        )

    def test_managed_run_starts_a_process_group(self):
        process = mock.Mock(returncode=0)
        with (
            mock.patch.object(
                environment,
                "_managed_process_kwargs",
                return_value={"start_new_session": True},
            ),
            mock.patch.object(
                environment.subprocess,
                "Popen",
                return_value=process,
            ) as popen,
        ):
            result = environment.run(
                [Path("tool"), "argument"],
                capture_output=True,
                text=True,
                timeout=17,
            )

        assert popen.call_args.args == (["tool", "argument"],)
        assert popen.call_args.kwargs["cwd"] is None
        assert popen.call_args.kwargs["env"] is None
        assert popen.call_args.kwargs["text"] is True
        assert popen.call_args.kwargs["start_new_session"] is True
        assert popen.call_args.kwargs["stdout"] is not None
        assert popen.call_args.kwargs["stderr"] is not None
        process.wait.assert_called_once_with(timeout=17)
        assert (result.returncode, result.stdout, result.stderr) == (
            0,
            "",
            "",
        )

    def test_managed_run_terminates_the_process_tree_on_timeout(self):
        timeout = environment.subprocess.TimeoutExpired(["tool"], 17)
        process = mock.Mock()
        process.wait.side_effect = timeout

        def start(*_args, **kwargs):
            kwargs["stdout"].write(b"partial")
            kwargs["stderr"].write(b"diagnostic")
            return process

        with (
            mock.patch.object(
                environment.subprocess,
                "Popen",
                side_effect=start,
            ),
            mock.patch.object(environment, "_terminate_process_tree") as terminate,
            pytest.raises(environment.subprocess.TimeoutExpired) as raised,
        ):
            environment.run(["tool"], capture_output=True, text=True, timeout=17)

        terminate.assert_called_once_with(process)
        assert raised.value.output == "partial"
        assert raised.value.stderr == "diagnostic"

    def test_managed_run_terminates_the_process_tree_on_interruption(self):
        process = mock.Mock()
        process.wait.side_effect = KeyboardInterrupt
        with (
            mock.patch.object(
                environment.subprocess,
                "Popen",
                return_value=process,
            ),
            mock.patch.object(environment, "_terminate_process_tree") as terminate,
            pytest.raises(KeyboardInterrupt),
        ):
            environment.run(["tool"])

        terminate.assert_called_once_with(process)

    def test_managed_run_timeout_stops_descendant_writes(self, tmp_path):
        heartbeat = tmp_path / "heartbeat"
        child = (
            "import pathlib,time;"
            f"p=pathlib.Path({str(heartbeat)!r});"
            "[(p.open('a',encoding='utf-8').write('x'),time.sleep(.01))"
            " for _ in range(1000)]"
        )
        parent = (
            "import pathlib,subprocess,sys,time;"
            f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
            f"p=pathlib.Path({str(heartbeat)!r});"
            "deadline=time.monotonic()+2;"
            "\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
            "time.sleep(10)"
        )

        with pytest.raises(environment.subprocess.TimeoutExpired):
            environment.run(
                [sys.executable, "-c", parent],
                capture_output=True,
                timeout=0.2,
            )

        assert heartbeat.is_file()
        size = heartbeat.stat().st_size
        deadline = time.monotonic() + 0.75
        while time.monotonic() < deadline:
            time.sleep(0.01)
            assert heartbeat.stat().st_size == size

    def test_process_tree_cleanup_never_waits_without_a_bound(self):
        process = mock.Mock(pid=1234)
        process.poll.return_value = None
        process.wait.side_effect = [
            environment.subprocess.TimeoutExpired(["tool"], 5),
            environment.subprocess.TimeoutExpired(["tool"], 5),
        ]
        with (
            mock.patch.object(environment.paths, "is_windows", return_value=True),
            mock.patch.object(environment.subprocess, "run"),
            pytest.raises(RuntimeError, match="failed to terminate process tree"),
        ):
            environment._terminate_process_tree(process, grace_seconds=5)

        assert process.wait.call_args_list == [
            mock.call(timeout=5),
            mock.call(timeout=5),
        ]

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
            mock.patch.object(environment, "run") as run,
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
        assert run.call_args.kwargs["timeout"] == (
            environment.DEFAULT_COMMAND_TIMEOUT_SECONDS
        )

    def test_python_install_falls_back_to_pip(self, tmp_path):
        with (
            mock.patch.object(
                environment,
                "uv_bin",
                return_value=tmp_path / "missing-uv",
            ),
            mock.patch.object(environment, "run") as run,
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
            mock.patch.object(environment, "run") as run,
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
            mock.patch.object(environment, "run") as run,
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
