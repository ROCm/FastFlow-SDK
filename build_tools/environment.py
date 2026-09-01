# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared process, tool-discovery, and subprocess-environment helpers."""

import locale
import os
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

from build_tools import paths

UV_VERSION = "0.11.33"
DEFAULT_COMMAND_TIMEOUT_SECONDS = 2 * 60 * 60

_WINDOWS_BUILD_ENV = None
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _selected_windows(windows: bool | None) -> bool:
    return paths.is_windows() if windows is None else windows


def _managed_process_kwargs() -> dict[str, object]:
    if paths.is_windows():
        return {"creationflags": _CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_process_tree(
    process: subprocess.Popen, *, grace_seconds: float = 5.0
) -> None:
    if paths.is_windows():
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                creationflags=_CREATE_NO_WINDOW,
                timeout=grace_seconds,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"failed to terminate process tree rooted at PID {process.pid}"
            ) from error


def _read_captured_output(stream, *, text: bool, encoding: str, errors: str):
    if stream is None:
        return None
    stream.flush()
    stream.seek(0)
    data = stream.read()
    return data.decode(encoding, errors=errors) if text else data


def run(
    cmd,
    cwd=None,
    env=None,
    check=True,
    timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS,
    **kwargs,
):
    """Run a bounded command in a process group and clean up its descendants."""
    command = [str(item) for item in cmd]
    capture_output = kwargs.pop("capture_output", False)
    if capture_output:
        if "stdout" in kwargs or "stderr" in kwargs:
            raise ValueError(
                "stdout and stderr arguments may not be used with capture_output"
            )
    elif (
        kwargs.get("stdout") == subprocess.PIPE
        or kwargs.get("stderr") == subprocess.PIPE
    ):
        raise ValueError("use capture_output=True instead of PIPE")

    text_mode = bool(kwargs.get("text") or kwargs.get("universal_newlines"))
    encoding = kwargs.get("encoding") or (
        "utf-8" if sys.flags.utf8_mode else locale.getencoding()
    )
    errors = kwargs.get("errors") or "strict"
    with (
        tempfile.TemporaryFile() if capture_output else nullcontext() as stdout_file,
        tempfile.TemporaryFile() if capture_output else nullcontext() as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            **(
                {"stdout": stdout_file, "stderr": stderr_file} if capture_output else {}
            ),
            **kwargs,
            **_managed_process_kwargs(),
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _terminate_process_tree(process)
            stdout = _read_captured_output(
                stdout_file, text=text_mode, encoding=encoding, errors=errors
            )
            stderr = _read_captured_output(
                stderr_file, text=text_mode, encoding=encoding, errors=errors
            )
            error.output = stdout
            error.stderr = stderr
            raise
        except BaseException:
            _terminate_process_tree(process)
            raise
        stdout = _read_captured_output(
            stdout_file, text=text_mode, encoding=encoding, errors=errors
        )
        stderr = _read_captured_output(
            stderr_file, text=text_mode, encoding=encoding, errors=errors
        )
    result = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        stderr,
    )
    if check:
        result.check_returncode()
    return result


def sh(
    cmd,
    cwd=None,
    env=None,
    check=True,
    timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS,
):
    print(f"+ {' '.join(map(str, cmd))}", flush=True)
    return run(
        cmd,
        cwd=cwd,
        env=env,
        check=check,
        timeout=timeout,
    )


def venv_py(venv: Path | None = None, *, windows: bool | None = None) -> Path:
    return paths.venv_bin(venv, windows=windows) / paths.executable_name(
        "python", windows=windows
    )


def uv_bin(venv: Path | None = None, *, windows: bool | None = None) -> Path:
    return paths.venv_bin(venv, windows=windows) / paths.executable_name(
        "uv", windows=windows
    )


def python_install(python, *arguments, refresh_packages=()):
    """Installs with the shared uv cache, with pip as a standalone fallback."""
    uv = uv_bin()
    if uv.is_file():
        refresh = [
            item
            for package in refresh_packages
            for item in ("--refresh-package", package)
        ]
        command = [uv, "pip", "install", "--python", python, *refresh, *arguments]
    else:
        command = [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            *arguments,
        ]
    env = dict(os.environ)
    ssl_cert_file = env.get("SSL_CERT_FILE")
    if ssl_cert_file and not Path(ssl_cert_file).is_file():
        env.pop("SSL_CERT_FILE")
    run(
        [str(item) for item in command],
        check=True,
        env=env,
        timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )


def cmake_bin(venv: Path | None = None, *, windows: bool | None = None) -> str:
    """cmake from the venv (pip, >=3.30) if set up, else system cmake."""
    candidate = paths.venv_bin(venv, windows=windows) / paths.executable_name(
        "cmake", windows=windows
    )
    return str(candidate) if candidate.exists() else "cmake"


def cmake_path(p: Path) -> str:
    r"""Return a path in the form CMake parses correctly on every platform."""
    return p.as_posix()


def cmake_source_override(name: str):
    """Forwards an explicitly prepared FetchContent source into CMake."""
    value = os.environ.get(name)
    if not value:
        return []
    source = Path(value).resolve()
    if not source.is_dir():
        sys.exit(f"{name} is not a directory: {source}")
    return [f"-D{name}={cmake_path(source)}"]


def cmake_compiler_flags(cc: str | None, cxx: str | None) -> list[str]:
    """Return CMake cache flags for an explicitly selected compiler pair."""
    if not cc or not cxx:
        return []
    return [
        f"-DCMAKE_C_COMPILER={cc}",
        f"-DCMAKE_CXX_COMPILER={cxx}",
    ]


def cmake_ccache_flags() -> list[str]:
    """Return CMake compiler-launcher flags when ccache is available."""
    if not shutil.which("ccache"):
        return []
    return [
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON",
    ]


def env_path(env=None, *, windows: bool | None = None):
    selected = env or os.environ
    if _selected_windows(windows):
        return (
            selected.get("Path") or selected.get("PATH") or selected.get("path") or ""
        )
    return selected.get("PATH", "")


def set_env_path(env, value, *, windows: bool | None = None):
    if _selected_windows(windows):
        for key in list(env):
            if key.lower() == "path":
                del env[key]
        env["Path"] = value
    else:
        env["PATH"] = value


def prepend_env_path(env, entries, *, windows: bool | None = None):
    parts = [str(entry) for entry in entries if str(entry)]
    current = env_path(env, windows=windows)
    if current:
        parts.append(current)
    set_env_path(env, os.pathsep.join(parts), windows=windows)


def which(exe, env=None, *, windows: bool | None = None):
    return shutil.which(exe, path=env_path(env, windows=windows))


def windows_vcvars64_path(*, windows: bool | None = None):
    if not _selected_windows(windows):
        return None
    for key in ("RNS_VCVARS64", "VCVARS64"):
        value = os.environ.get(key)
        if value and Path(value).is_file():
            return Path(value)
    candidates = []
    for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if not root:
            continue
        base = Path(root) / "Microsoft Visual Studio"
        if base.exists():
            candidates.extend(base.glob("*/*/VC/Auxiliary/Build/vcvars64.bat"))
    return sorted(candidates, reverse=True)[0] if candidates else None


def windows_msvc_env(*, windows: bool | None = None):
    """Return an environment with the installed VS x64 toolchain loaded."""
    global _WINDOWS_BUILD_ENV
    selected_windows = _selected_windows(windows)
    if not selected_windows:
        return dict(os.environ)
    if _WINDOWS_BUILD_ENV is not None:
        return dict(_WINDOWS_BUILD_ENV)

    env = dict(os.environ)
    vcvars = windows_vcvars64_path(windows=selected_windows)
    if not which("cl", env, windows=selected_windows) and vcvars:
        # A wrapper avoids cmd.exe /c quoting edge cases for VS paths with spaces.
        with tempfile.TemporaryDirectory(prefix="rns-vs-env-") as temporary:
            wrapper = Path(temporary) / "load-vs.cmd"
            wrapper.write_text(
                f'@echo off\r\ncall "{vcvars}" >nul\r\nset\r\n',
                encoding="utf-8",
            )
            result = subprocess.run([str(wrapper)], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.lower() == "path":
                    set_env_path(env, value, windows=selected_windows)
                else:
                    env[key] = value
        else:
            print(
                f"warning: failed to load {vcvars}: {result.stderr.strip()}",
                file=sys.stderr,
            )

    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        prepend_env_path(env, [cargo_bin], windows=selected_windows)
    if not env.get("ASM") and which("ml64", env, windows=selected_windows):
        env["ASM"] = "ml64"
    _WINDOWS_BUILD_ENV = env
    return dict(env)


def cmake_env(venv: Path | None = None, *, windows: bool | None = None):
    """PATH with the venv bin first so pip-provided cmake/ninja are used."""
    selected_windows = _selected_windows(windows)
    env = (
        windows_msvc_env(windows=selected_windows)
        if selected_windows
        else dict(os.environ)
    )
    prepend_env_path(
        env,
        [paths.venv_bin(venv, windows=selected_windows)],
        windows=selected_windows,
    )
    return env


def compilers(*, windows: bool | None = None):
    """Return the C and C++ compilers used for builds."""
    cc, cxx = os.environ.get("CC"), os.environ.get("CXX")
    if cc and cxx:
        return cc, cxx
    if _selected_windows(windows):
        return None, None
    if shutil.which("gcc") and shutil.which("g++"):
        return "gcc", "g++"
    if shutil.which("clang") and shutil.which("clang++"):
        return "clang", "clang++"
    return None, None


def rustc_version(env=None, *, windows: bool | None = None):
    rustc = which("rustc", env, windows=windows)
    if not rustc:
        return None
    result = subprocess.run(
        [rustc, "--version"], capture_output=True, text=True, env=env
    )
    if result.returncode != 0:
        return None
    parts = result.stdout.split()
    if len(parts) < 2:
        return None
    numbers = parts[1].split("-", 1)[0].split(".")
    try:
        return tuple(int(number) for number in numbers[:3])
    except ValueError:
        return None


def have_header(hdr, language="c"):
    if hdr == "Python.h":
        include = sysconfig.get_path("include")
        return bool(include) and (Path(include) / "Python.h").exists()
    if language == "c++":
        compiler = (
            os.environ.get("CXX")
            or shutil.which("clang++")
            or shutil.which("c++")
            or shutil.which("g++")
        )
    else:
        compiler = (
            os.environ.get("CC")
            or shutil.which("clang")
            or shutil.which("cc")
            or shutil.which("gcc")
        )
    if not compiler:
        return False
    result = subprocess.run(
        [compiler, "-fsyntax-only", "-x", language, "-"],
        input=f"#include <{hdr}>\nint main(){{return 0;}}\n".encode(),
        capture_output=True,
    )
    return result.returncode == 0


def have_perl_module(module):
    perl = shutil.which("perl")
    if not perl:
        return False
    return (
        subprocess.run([perl, f"-M{module}", "-e", "1"], capture_output=True).returncode
        == 0
    )
