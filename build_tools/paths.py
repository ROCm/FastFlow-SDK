# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared repository layout and host-path conventions."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "projects"
BUILD = ROOT / "build"
OUT = ROOT / "out"
VENV = ROOT / ".venv"
PATCHES = ROOT / "patches"
FETCH_CACHE = ROOT / ".cache" / "find-links"

# Stable build layout shared by developer commands, product packaging, and tests.
LLVM_BUILD = Path(os.environ.get("RNS_LLVM_BUILD_DIR") or (BUILD / "llvm"))
LLVM_AIE_BUILD = Path(os.environ.get("RNS_LLVM_AIE_BUILD_DIR") or (BUILD / "llvm-aie"))
LLVM_AIE_INSTALL = Path(
    os.environ.get("RNS_LLVM_AIE_INSTALL_DIR") or (BUILD / "llvm-aie-install")
)
IRON_TOOLS_BUILD = BUILD / "iron-tools"
IRON_TOOLS_INSTALL = BUILD / "iron-tools-install"
TOOLS_BIN = ROOT / "tools-bin"
HRX_CMAKE_BUILD = PROJ / "hrx" / "build" / "cmake"
HRX_INSTALL = BUILD / "hrx-install"
HRX_PACKAGE_SMOKE_BUILD = BUILD / "hrx-package-smoke"
ROCM_NPU_CLI_BUILD = BUILD / "rocm-npu-cli"
ROCM_NPU_CLI_INSTALL = BUILD / "rocm-npu-cli-install"
FASTFLOWLM_BUILD = BUILD / "FastFlowLM"
FASTFLOWLM_INSTALL = BUILD / "FastFlowLM-install"
# The XRT-backed FastFlowLM build is kept fully separate from the default HRX
# build so the two never share CMake state or install trees (see
# fastflowlm_build.py); it exists only as a diagnostic comparison target and is
# never packaged into the shipped HRX wheels.
FASTFLOWLM_XRT_BUILD = BUILD / "FastFlowLM-xrt"
FASTFLOWLM_XRT_INSTALL = BUILD / "FastFlowLM-xrt-install"
FASTFLOWLM_MODEL_CACHE = BUILD / "FastFlowLM-models"


def is_windows() -> bool:
    return os.name == "nt"


def venv_bin(venv: Path | None = None, *, windows: bool | None = None) -> Path:
    selected_venv = VENV if venv is None else venv
    selected_windows = is_windows() if windows is None else windows
    return selected_venv / ("Scripts" if selected_windows else "bin")


def executable_name(name: str, *, windows: bool | None = None) -> str:
    selected_windows = is_windows() if windows is None else windows
    return f"{name}.exe" if selected_windows else name
