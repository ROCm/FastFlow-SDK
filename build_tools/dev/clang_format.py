# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Format first-party C/C++ with the pinned clang-format wheel."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from build_tools import paths
from build_tools.dev import format_tools

CXX_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp"}
FIRST_PARTY_PREFIXES = (
    "build_tools/native/",
    "examples/iron/",
    "tools/fastflow/",
    "tools/fastflow-sdk-qualification/native/",
    "tests/hal-contract/",
)
PINNED_VERSION = format_tools.requirement_pin("clang-format")
install_command = format_tools.install_command


def first_party_cxx_files() -> list[str]:
    tracked = subprocess.check_output(
        ["git", "ls-files"],
        cwd=paths.ROOT,
        text=True,
    ).splitlines()
    files = []
    for path in tracked:
        posix = path.replace("\\", "/")
        if Path(posix).suffix not in CXX_SUFFIXES:
            continue
        if posix.startswith(FIRST_PARTY_PREFIXES):
            files.append(posix)
    return files


def clang_format_executable() -> str | None:
    try:
        import clang_format
    except ImportError:
        clang_format = None
    if clang_format is not None:
        bundled = Path(clang_format.get_executable("clang-format"))
        if bundled.is_file():
            return str(bundled)
    return shutil.which("clang-format")


def resolve_pinned_executable() -> tuple[str | None, str | None]:
    executable = clang_format_executable()
    if executable is None:
        return None, f"clang-format {PINNED_VERSION} is not installed"
    version = subprocess.check_output([executable, "--version"], text=True)
    if PINNED_VERSION not in version:
        return None, f"expected clang-format {PINNED_VERSION}, got: {version.strip()}"
    return executable, None


def run_clang_format(*, check: bool) -> int:
    executable, error = resolve_pinned_executable()
    if error:
        return format_tools.fail(error)
    files = first_party_cxx_files()
    if not files:
        print("no first-party C/C++ files to format", file=sys.stderr)
        return 1
    command = [executable]
    if check:
        command.extend(["--dry-run", "--Werror"])
    else:
        command.append("-i")
    command.extend(files)
    status = subprocess.call(command, cwd=paths.ROOT)
    if status == 0:
        word = "already formatted" if check else "formatted"
        print(f"{len(files)} C/C++ files {word}", flush=True)
    return status


def cmd_clang_format(args):
    return run_clang_format(check=args.check)
