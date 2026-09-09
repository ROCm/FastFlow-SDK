# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from build_tools.dev.fastflowlm_build import (
    _host_xrt_include_dir,
    _host_xrt_lib_dir,
    _host_xrt_pkgconfig_dir,
    _host_xrt_prefix_candidates,
    _is_host_xrt_prefix,
    _missing_host_xrt_message,
    preset_cache_variables,
)

pytestmark = pytest.mark.presubmit


def test_host_xrt_pkgconfig_dir_uses_real_host_pc(tmp_path):
    host = tmp_path / "xrt"
    pc = host / "share" / "pkgconfig" / "xrt.pc"
    pc.parent.mkdir(parents=True)
    pc.write_text("Name: XRT\nVersion: 2.23.0\n", encoding="utf-8")
    assert _host_xrt_pkgconfig_dir(host) == pc.parent


def test_host_xrt_pkgconfig_dir_does_not_invent_a_stub(tmp_path):
    host = tmp_path / "xrt"
    host.mkdir()
    assert _host_xrt_pkgconfig_dir(host) is None
    assert not list(host.rglob("xrt.pc"))


def test_windows_xrt_prefix_candidates_match_fastflowlm_defaults():
    candidates = _host_xrt_prefix_candidates(windows=True)
    assert Path(r"C:\dev\xrt-prefix") in candidates
    assert Path(r"C:\dev\xrtNPUfromDLL") in candidates
    assert Path("/opt/xilinx/xrt") not in candidates
    linux = _host_xrt_prefix_candidates(windows=False)
    assert Path("/opt/xilinx/xrt") in linux
    assert Path(r"C:\dev\xrt-prefix") not in linux


def test_missing_host_xrt_message_mentions_windows_defaults():
    windows = _missing_host_xrt_message(windows=True)
    assert "C:\\dev\\xrt-prefix" in windows
    assert "/opt/xilinx/xrt" not in windows
    linux = _missing_host_xrt_message(windows=False)
    assert "/opt/xilinx/xrt" in linux
    assert "C:\\dev\\xrt-prefix" not in linux


def test_host_xrt_lib_dir_uses_fhs_lib_when_present(tmp_path):
    prefix = tmp_path / "xrt"
    lib = prefix / "lib"
    lib.mkdir(parents=True)
    (lib / "libxrt_coreutil.so").write_bytes(b"")
    assert _host_xrt_lib_dir(prefix) == lib


def test_host_xrt_lib_dir_uses_dll_root_layout(tmp_path):
    prefix = tmp_path / "xrtNPUfromDLL"
    prefix.mkdir()
    (prefix / "xrt_coreutil.lib").write_bytes(b"")
    (prefix / "xrt_coreutil.dll").write_bytes(b"")
    assert _is_host_xrt_prefix(prefix)
    assert _host_xrt_lib_dir(prefix) == prefix


def test_host_xrt_include_dir_accepts_xrt_source_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "build_tools.dev.fastflowlm_build._WINDOWS_XRT_INCLUDE_DIRS",
        (),
    )
    include = tmp_path / "XRT" / "src" / "runtime_src" / "core" / "include"
    (include / "xrt" / "experimental").mkdir(parents=True)
    (include / "xrt" / "xrt_bo.h").write_text("", encoding="utf-8")
    (include / "xrt" / "experimental" / "xrt_ext.h").write_text("", encoding="utf-8")
    prefix = tmp_path / "xrtNPUfromDLL"
    prefix.mkdir()
    (prefix / "xrt_coreutil.dll").write_bytes(b"")
    assert _host_xrt_include_dir(prefix, windows=True) is None
    found = _host_xrt_include_dir(tmp_path / "XRT", windows=True)
    assert found == include.resolve()


def test_preset_cache_variables_reads_common_default(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "CMakePresets.json").write_text(
        """{
  "version": 3,
  "configurePresets": [
    {
      "name": "common-default",
      "hidden": true,
      "cacheVariables": {
        "FLM_VERSION": "1.0.5",
        "NPU_VERSION": "32.0.203.304"
      }
    }
  ]
}
""",
        encoding="utf-8",
    )
    assert preset_cache_variables(source) == {
        "FLM_VERSION": "1.0.5",
        "NPU_VERSION": "32.0.203.304",
    }
