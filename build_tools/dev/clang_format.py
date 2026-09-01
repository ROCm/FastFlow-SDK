# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Format first-party C/C++ with the pinned clang-format wheel."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from build_tools import paths

CXX_SUFFIXES = {".c", ".cc", ".cpp", ".h", ".hpp"}
FIRST_PARTY_PREFIXES = (
    "tools/rocm-npu/",
    "tools/rocm-npu-qualification/native/",
    "tests/hal-contract/",
)
TEST_REQUIREMENTS = paths.ROOT / "requirements" / "test.txt"


def requirement_pin(package: str) -> str:
    prefix = f"{package}=="
    for line in TEST_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        entry = line.partition("#")[0].strip()
        if entry.startswith(prefix):
            return entry.removeprefix(prefix)
    raise ValueError(f"requirements/test.txt does not pin {package}")


PINNED_VERSION = requirement_pin("clang-format")


def install_command() -> str:
    return "python -m pip install --require-hashes -r requirements/test.lock"


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


def _fail(message: str) -> int:
    print(f"{message}\nInstall with:\n  {install_command()}", file=sys.stderr)
    return 1


def run_clang_format(*, check: bool) -> int:
    executable, error = resolve_pinned_executable()
    if error:
        return _fail(error)
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
    return subprocess.call(command, cwd=paths.ROOT)


def cmd_clang_format(args):
    return run_clang_format(check=args.check)
