# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""First-party Python lint/format and the parent format command."""

from __future__ import annotations

import subprocess
import sys

from build_tools import paths

TEST_REQUIREMENTS = paths.ROOT / "requirements" / "test.txt"


def requirement_pin(package: str) -> str:
    prefix = f"{package}=="
    for line in TEST_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        entry = line.partition("#")[0].strip()
        if entry.startswith(prefix):
            return entry.removeprefix(prefix)
    raise ValueError(f"requirements/test.txt does not pin {package}")


PINNED_RUFF_VERSION = requirement_pin("ruff")


def install_command() -> str:
    return "python -m pip install --require-hashes -r requirements/test.lock"


def fail(message: str) -> int:
    print(f"{message}\nInstall with:\n  {install_command()}", file=sys.stderr)
    return 1


def resolve_pinned_ruff() -> tuple[list[str] | None, str]:
    command = [sys.executable, "-m", "ruff"]
    try:
        output = subprocess.check_output([*command, "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None, f"ruff {PINNED_RUFF_VERSION} is not installed"
    if PINNED_RUFF_VERSION not in output:
        return None, f"expected ruff {PINNED_RUFF_VERSION}, got: {output.strip()}"
    return command, output.strip()


def run_ruff(*, check: bool) -> int:
    command, detail = resolve_pinned_ruff()
    if command is None:
        return fail(detail)
    lint = [*command, "check"]
    fmt = [*command, "format"]
    if check:
        fmt.append("--check")
    else:
        lint.append("--fix")
    lint_status = subprocess.call(lint, cwd=paths.ROOT)
    format_status = subprocess.call(fmt, cwd=paths.ROOT)
    return lint_status or format_status


def cmd_ruff(args):
    return run_ruff(check=args.check)


def cmd_format(args):
    from build_tools.dev import clang_format

    print("Python (ruff)", flush=True)
    ruff_status = run_ruff(check=args.check)
    print("C/C++ (clang-format)", flush=True)
    cxx_status = clang_format.run_clang_format(check=args.check)
    return ruff_status or cxx_status
