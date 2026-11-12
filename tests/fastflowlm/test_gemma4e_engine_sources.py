# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path

import pytest

from build_tools.fastflow_sdk_archive import (
    stage_gemma4e_extension_sources,
    verify_installed_gemma4e_sdk,
)

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def test_installed_prefix_receives_gemma4e_extension_sources(tmp_path: Path):
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
    stage_gemma4e_extension_sources(source, bundle, Path(__file__).resolve().parents[2])

    engine = bundle / "share" / "fastflow-sdk" / "gemma4e"
    assert (engine / "CMakeLists.txt").is_file()
    cmake_text = (engine / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "find_package(hrx CONFIG REQUIRED)" in cmake_text
    assert "find_package(FastFlowLMModelExtension CONFIG REQUIRED)" in cmake_text
    assert "gemma4e_open_extension" in cmake_text
    assert "gemma4e_npu/*.cpp" in cmake_text
    assert "add_subdirectory(src/detail)" not in cmake_text
    assert "CXX_VISIBILITY_PRESET hidden" in cmake_text
    assert "-Wl,-Bsymbolic-functions" in cmake_text
    assert "CMAKE_POSITION_INDEPENDENT_CODE ON" in cmake_text
    assert "gemma4e_iron_engine" not in cmake_text
    assert "FLM_IRON_INCLUDE_DIRS" in cmake_text
    assert "q4_npu_eXpress.dll" in cmake_text
    assert "mha.dll" in cmake_text
    assert 'FLM_EXTENSION_ENGINE_HEADER ""' in cmake_text
    assert "frontend/gemma4_open_frontend.cpp" in cmake_text
    assert "FLM_GEMMA4_XCLBIN_DIR" in cmake_text
    assert "xclbins/Gemma4-E2B-IT-NPU2" in cmake_text
    assert "contains no .xclbin files" in cmake_text
    assert (engine / "model_extension.cpp").is_file()
    assert (engine / "model-list.json").is_file()
    assert (engine / "src" / "detail" / "gemma4e_npu" / "gemma4e_npu.cpp").is_file()
    assert (engine / "src" / "include" / "flm_override.hpp").is_file()
    assert not (engine / "src" / "frontend").exists()
    if os.name == "nt":
        assert (bundle / "lib" / "flm" / "q4_npu_eXpress.lib").read_bytes() == b"q4"
        assert (bundle / "lib" / "flm" / "mha.lib").read_bytes() == b"mha"
    else:
        assert (bundle / "lib" / "flm" / "libq4_npu_eXpress.so").is_file()
        assert (bundle / "lib" / "flm" / "libmha.so").is_file()


def test_gemma4e_extension_catalog_selects_custom_engine_without_folder_collision():
    catalog_path = (
        Path(__file__).resolve().parents[2]
        / "cmake"
        / "gemma4e-extension"
        / "model-list.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entry = catalog["models"]["gemma4-open-it"]["e2b"]

    assert entry["name"] == "Gemma4-Open-E2B-IT-NPU2"
    assert entry["name"] != entry["xclbin_folder"]
    assert entry["details"]["family"] == "gemma4-open"
    assert entry["backend"] == "flm"
    assert entry["backend_provider"] == "gemma4-open"
    assert entry["frontend_provider"] == "gemma4-open"
    assert entry["vlm"] is True
    assert entry["asr"] is True
    assert "vision_weight.q4nx" in entry["files"]
    assert "audio_weight.q4nx" in entry["files"]
    provider = catalog_path.parent / "model_extension.cpp"
    provider_source = provider.read_text(encoding="utf-8")
    assert "config_.model_name = xclbin_folder" in provider_source
    assert "prefill(ids, native)" in provider_source
    assert 'config_._json_config["is_vlm"] = false' not in provider_source
    assert 'config_._json_config["is_audio"] = false' not in provider_source


def _write_open_frontend_tree(source: Path) -> None:
    frontend = source / "src" / "test" / "model_extension_gemma4e"
    frontend.mkdir(parents=True)
    (frontend / "gemma4_open_frontend.cpp").write_text(
        "void frontend() {}\n", encoding="utf-8"
    )
    for relative in (
        "chat/chat_continue.cpp",
        "image/image_reader.cpp",
        "image_process_utils/imageproc.cpp",
        "image_process_utils/imageprocAVX512.cpp",
        "audio/audio_reader.cpp",
        "audio/audio_decoder.cpp",
        "audio/audio_resampler.cpp",
        "audio_process_utils/audioproc.cpp",
        "audio_process_utils/audioprocAVX512.cpp",
    ):
        path = source / "src" / "common" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("void media() {}\n", encoding="utf-8")
    cmake = source / "src" / "cmake"
    cmake.mkdir()
    (cmake / "FlmImageCodecs.cmake").write_text("# image\n", encoding="utf-8")
    (cmake / "FlmAudioCodecs.cmake").write_text("# audio\n", encoding="utf-8")
    tokenizers = source / "third_party" / "tokenizers-cpp"
    tokenizers.mkdir(parents=True)
    (tokenizers / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n", encoding="utf-8"
    )


def test_shipped_tree_includes_the_open_frontend(tmp_path: Path):
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
    _write_open_frontend_tree(source)

    bundle = tmp_path / "prefix"
    bundle.mkdir()
    stage_gemma4e_extension_sources(source, bundle, Path(__file__).resolve().parents[2])

    engine = bundle / "share" / "fastflow-sdk" / "gemma4e"
    assert (engine / "src" / "frontend" / "gemma4_open_frontend.cpp").is_file()
    assert (engine / "src" / "common" / "audio" / "audio_reader.cpp").is_file()
    assert (engine / "src" / "cmake" / "FlmImageCodecs.cmake").is_file()
    assert (engine / "src" / "cmake" / "FlmAudioCodecs.cmake").is_file()
    assert (engine / "third_party" / "tokenizers-cpp" / "CMakeLists.txt").is_file()


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

    stage_gemma4e_extension_sources(source, bundle, Path(__file__).resolve().parents[2])

    if os.name == "nt":
        assert (installed / "q4_npu_eXpress.lib").read_bytes() == b"from-install"
        assert (installed / "mha.lib").read_bytes() == b"from-git"
    else:
        assert (installed / "libq4_npu_eXpress.so").read_bytes() == b"from-install"
        assert (installed / "libmha.so").read_bytes() == b"from-git"


def test_gemma4e_sdk_check_rejects_a_prefix_without_sources(tmp_path: Path):
    bundle = tmp_path / "prefix"
    (bundle / "include" / "AutoModel").mkdir(parents=True)
    (bundle / "include" / "AutoModel" / "model_extension.h").write_text(
        "#pragma once\n", encoding="utf-8"
    )
    package = bundle / "lib" / "cmake" / "FastFlowLMModelExtension"
    package.mkdir(parents=True)
    (package / "FastFlowLMModelExtensionConfig.cmake").write_text("", encoding="utf-8")
    try:
        verify_installed_gemma4e_sdk(bundle)
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("expected a prefix without engine sources to fail")
    if "Gemma 4 extension sources missing" not in message:
        raise AssertionError(message)
