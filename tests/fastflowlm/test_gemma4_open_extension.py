# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest

from build_tools import gemma4_open_extension
from build_tools.fastflow_sdk_archive import _linux_elf_files
from build_tools.gemma4_open_extension import (
    EXTENSION_RELATIVE_DIR,
    build_gemma4_open_extension,
    verify_gemma4_open_extension,
)

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]

ROOT = Path(__file__).resolve().parents[2]


def _write_extension(bundle: Path) -> Path:
    output = bundle / EXTENSION_RELATIVE_DIR
    (output / "lib").mkdir(parents=True)
    (output / "lib" / "libgemma4e_open_extension.so").write_bytes(b"elf")
    xclbins = output / "xclbins" / "Gemma4-E2B-IT-NPU2"
    xclbins.mkdir(parents=True)
    (xclbins / "attn.xclbin").write_bytes(b"xclbin")
    (output / "catalog").mkdir()
    (output / "catalog" / "model-list.json").write_text(
        (ROOT / "cmake" / "gemma4e-extension" / "model-list.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    (output / "extension.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "gemma4-open",
                "version": "1.0.0",
                "library": "lib/libgemma4e_open_extension.so",
                "asset_root": ".",
                "catalog": "catalog/model-list.json",
            }
        ),
        encoding="utf-8",
    )
    return output


def test_verify_accepts_a_complete_extension(tmp_path: Path):
    _write_extension(tmp_path)
    verify_gemma4_open_extension(tmp_path)


def test_verify_rejects_an_extension_without_xclbins(tmp_path: Path):
    output = _write_extension(tmp_path)
    (output / "xclbins" / "Gemma4-E2B-IT-NPU2" / "attn.xclbin").unlink()
    with pytest.raises(RuntimeError, match="no Gemma4-E2B-IT-NPU2 xclbins"):
        verify_gemma4_open_extension(tmp_path)


def test_build_configures_the_staged_sources_against_the_bundle(
    monkeypatch, tmp_path: Path
):
    root = tmp_path / "root"
    (root / "build").mkdir(parents=True)
    bundle = tmp_path / "bundle"
    source = bundle / "share" / "fastflow-sdk" / "gemma4e"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("project(x)\n", encoding="utf-8")
    commands = []

    def fake_sh(command, **kwargs):
        commands.append([str(item) for item in command])
        if "--build" in command:
            _write_extension(bundle)

    monkeypatch.setattr(gemma4_open_extension.environment, "sh", fake_sh)
    monkeypatch.setattr(
        gemma4_open_extension,
        "dependency_paths",
        lambda _: {
            "toolchain": str(tmp_path / "vcpkg.cmake"),
            "triplet": "x64-test",
            "install_root": str(tmp_path / "vcpkg-installed"),
        },
    )

    manifest = build_gemma4_open_extension(root, bundle)

    configure = commands[0]
    assert f"-S{source.as_posix()}" in "".join(configure)
    assert f"-DFASTFLOW_SDK_ROOT={bundle.as_posix()}" in configure
    assert (
        f"-DFLM_EXTENSION_BUNDLE_DIR={(bundle / EXTENSION_RELATIVE_DIR).as_posix()}"
        in configure
    )
    assert "-DVCPKG_TARGET_TRIPLET=x64-test" in configure
    assert "--build" in commands[1]
    assert manifest == bundle / EXTENSION_RELATIVE_DIR / "extension.json"


def test_linux_closure_includes_extension_libraries(tmp_path: Path):
    _write_extension(tmp_path)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "flm").write_bytes(b"elf")
    names = [path.name for path in _linux_elf_files(tmp_path)]
    assert names == ["flm", "libgemma4e_open_extension.so"]
