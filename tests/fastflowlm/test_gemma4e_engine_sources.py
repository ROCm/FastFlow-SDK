# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pytest

from build_tools.fastflow_sdk_archive import (
    stage_gemma4e_engine_sources,
    verify_installed_engine_sdk,
)

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def test_installed_prefix_receives_gemma4e_sources(tmp_path: Path):
    source = tmp_path / "FastFlowLM"
    detail = source / "src" / "detail"
    (detail / "gemma4e_npu").mkdir(parents=True)
    (detail / "CMakeLists.txt").write_text(
        "add_library(gemma4e_npu SHARED)\n", encoding="utf-8"
    )
    (detail / "gemma4e_npu" / "gemma4e_npu.cpp").write_text(
        "void gemma() {}\n", encoding="utf-8"
    )
    include = source / "src" / "include"
    include.mkdir()
    (include / "flm_override.hpp").write_text("#pragma once\n", encoding="utf-8")
    libs = source / "src" / "lib" / "hrx"
    libs.mkdir(parents=True)
    if os.name == "nt":
        (libs / "q4_npu_eXpress.lib").write_bytes(b"q4")
        (libs / "mha.lib").write_bytes(b"mha")
    else:
        (libs / "libq4_npu_eXpress.so").write_bytes(b"q4")
        (libs / "libmha.so").write_bytes(b"mha")

    bundle = tmp_path / "prefix"
    bundle.mkdir()
    stage_gemma4e_engine_sources(source, bundle, Path(__file__).resolve().parents[2])

    engine = bundle / "share" / "fastflow-sdk" / "gemma4e"
    assert (engine / "CMakeLists.txt").is_file()
    cmake_text = (engine / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "find_package(hrx CONFIG REQUIRED)" in cmake_text
    assert "FLM_IRON_INCLUDE_DIRS" in cmake_text
    assert (engine / "src" / "detail" / "gemma4e_npu" / "gemma4e_npu.cpp").is_file()
    assert (engine / "src" / "include" / "flm_override.hpp").is_file()
    if os.name == "nt":
        assert (bundle / "lib" / "flm" / "q4_npu_eXpress.lib").read_bytes() == b"q4"
        assert (bundle / "lib" / "flm" / "mha.lib").read_bytes() == b"mha"
    else:
        assert (bundle / "lib" / "flm" / "libq4_npu_eXpress.so").is_file()
        assert (bundle / "lib" / "flm" / "libmha.so").is_file()


def test_installed_engine_library_is_not_replaced(tmp_path: Path):
    source = tmp_path / "FastFlowLM"
    detail = source / "src" / "detail"
    (detail / "gemma4e_npu").mkdir(parents=True)
    (detail / "CMakeLists.txt").write_text(
        "add_library(gemma4e_npu SHARED)\n", encoding="utf-8"
    )
    (detail / "gemma4e_npu" / "gemma4e_npu.cpp").write_text(
        "void gemma() {}\n", encoding="utf-8"
    )
    include = source / "src" / "include"
    include.mkdir()
    (include / "flm_override.hpp").write_text("#pragma once\n", encoding="utf-8")
    libs = source / "src" / "lib" / "hrx"
    libs.mkdir(parents=True)
    bundle = tmp_path / "prefix"
    installed = bundle / "lib" / "flm"
    installed.mkdir(parents=True)
    if os.name == "nt":
        (libs / "q4_npu_eXpress.lib").write_bytes(b"from-git")
        (libs / "mha.lib").write_bytes(b"from-git")
        (installed / "q4_npu_eXpress.lib").write_bytes(b"from-install")
    else:
        (libs / "libq4_npu_eXpress.so").write_bytes(b"from-git")
        (libs / "libmha.so").write_bytes(b"from-git")
        (installed / "libq4_npu_eXpress.so").write_bytes(b"from-install")

    stage_gemma4e_engine_sources(source, bundle, Path(__file__).resolve().parents[2])

    if os.name == "nt":
        assert (installed / "q4_npu_eXpress.lib").read_bytes() == b"from-install"
        assert (installed / "mha.lib").read_bytes() == b"from-git"
    else:
        assert (installed / "libq4_npu_eXpress.so").read_bytes() == b"from-install"
        assert (installed / "libmha.so").read_bytes() == b"from-git"


def test_engine_sdk_check_rejects_a_prefix_without_sources(tmp_path: Path):
    bundle = tmp_path / "prefix"
    bundle.mkdir()
    try:
        verify_installed_engine_sdk(bundle)
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("expected a prefix without engine sources to fail")
    if "Gemma 4 engine sources missing" not in message:
        raise AssertionError(message)
