# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Windows packaged-engine loading contract."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.fastflowlm, pytest.mark.windows]

_REPO = Path(__file__).resolve().parents[2]


def _require_flm() -> Path:
    if os.name != "nt":
        pytest.skip("Windows engine DLL loading")
    for path in (
        _REPO / "build" / "FastFlowLM-install" / "bin" / "flm.exe",
        _REPO / "build" / "FastFlowLM" / "flm.exe",
    ):
        if path.is_file() and (path.parent / "gemma4e_npu.dll").is_file():
            return path
    pytest.skip("flm.exe with its engine DLLs is not built")


def test_installed_dlls_load():
    bin_dir = _require_flm().parent
    cookie = os.add_dll_directory(str(bin_dir))
    failures: list[str] = []
    try:
        for dll in sorted(bin_dir.glob("*.dll")):
            try:
                ctypes.WinDLL(str(dll))
            except OSError as exc:
                failures.append(f"{dll.name}: {exc}")
    finally:
        cookie.close()
    assert failures == []
