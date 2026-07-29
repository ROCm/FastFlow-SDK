# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared repository layout and host-path conventions."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "projects"
BUILD = ROOT / "build"
OUT = ROOT / "out"
VENV = ROOT / ".venv"
PATCHES = ROOT / "patches"
FETCH_CACHE = ROOT / ".cache" / "find-links"


def is_windows() -> bool:
    return os.name == "nt"


def venv_bin(venv: Path | None = None, *, windows: bool | None = None) -> Path:
    selected_venv = VENV if venv is None else venv
    selected_windows = is_windows() if windows is None else windows
    return selected_venv / ("Scripts" if selected_windows else "bin")


def executable_name(name: str, *, windows: bool | None = None) -> str:
    selected_windows = is_windows() if windows is None else windows
    return f"{name}.exe" if selected_windows else name
