# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Repository package-version resolution and validation."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

PACKAGE_VERSION_ENV = "ROCM_NPU_PACKAGE_VERSION"
_VERSION_RE = re.compile(
    r"^[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+|\.post[0-9]+|\.dev[0-9]+)?"
    r"(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?$"
)


def validate_version(version: str) -> str:
    """Validates the supported, normalized subset of PEP 440 versions."""
    if not _VERSION_RE.fullmatch(version):
        raise RuntimeError(f"invalid PEP 440 package version for rocm-npu: {version}")
    return version


def default_package_version(
    root: Path, *, environment_variable: str = PACKAGE_VERSION_ENV
) -> str:
    """Returns an explicit package version or a revision-derived development version."""
    version = os.environ.get(environment_variable)
    if version:
        return validate_version(version)
    revision = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"0.0.0.dev0+g{revision}"
