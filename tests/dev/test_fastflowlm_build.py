# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from build_tools.dev.fastflowlm_build import (
    _host_xrt_include_dir,
    _host_xrt_lib_dir,
    _host_xrt_pkgconfig_dir,
    _host_xrt_prefix,
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


def test_windows_xrt_candidates_use_vendor_roots_not_local_dev():
    environ = {
        "ProgramFiles": r"C:\Program Files",
        "PATH": "",
        "XILINX_XRT": "",
    }
    candidates = _host_xrt_prefix_candidates(windows=True, environ=environ)
    assert Path(r"C:\Program Files\Xilinx\XRT") in candidates
    assert Path(r"C:\Program Files\AMD\XRT") in candidates
    assert Path(r"C:\Xilinx\XRT") in candidates
    assert Path(r"C:\dev\xrt-prefix") not in candidates
    assert Path(r"C:\dev\xrtNPUfromDLL") not in candidates
    linux = _host_xrt_prefix_candidates(windows=False, environ=environ)
    assert Path("/opt/xilinx/xrt") in linux
    assert Path(r"C:\Program Files\Xilinx\XRT") not in linux


def test_missing_host_xrt_message_does_not_pin_local_dev():
    windows = _missing_host_xrt_message(windows=True)
    assert "XILINX_XRT" in windows
    assert "PATH" in windows
    assert "C:\\dev\\" not in windows
    linux = _missing_host_xrt_message(windows=False)
    assert "/opt/xilinx/xrt" in linux
    assert "C:\\dev\\" not in linux


def test_host_xrt_prefix_prefers_xilinx_xrt_env(tmp_path):
    env_root = tmp_path / "from-env"
    (env_root / "lib").mkdir(parents=True)
    (env_root / "lib" / "xrt_coreutil.dll").write_bytes(b"")
    path_bin = tmp_path / "from-path" / "bin"
    path_bin.mkdir(parents=True)
    (path_bin / "xrt_coreutil.dll").write_bytes(b"")
    found = _host_xrt_prefix(
        windows=True,
        environ={
            "XILINX_XRT": str(env_root),
            "PATH": str(path_bin),
            "ProgramFiles": str(tmp_path / "unused-pf"),
        },
    )
    assert found == env_root.resolve()


def test_host_xrt_prefix_discovers_dll_on_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "build_tools.dev.fastflowlm_build._windows_install_roots",
        lambda environ: [],
    )
    bindir = tmp_path / "Xilinx" / "XRT" / "bin"
    bindir.mkdir(parents=True)
    (bindir / "xrt_coreutil.dll").write_bytes(b"")
    found = _host_xrt_prefix(
        windows=True,
        environ={
            "XILINX_XRT": "",
            "PATH": str(bindir),
        },
    )
    assert found == (tmp_path / "Xilinx" / "XRT").resolve()


def test_host_xrt_lib_dir_uses_fhs_lib_when_present(tmp_path):
    prefix = tmp_path / "xrt"
    lib = prefix / "lib"
    lib.mkdir(parents=True)
    (lib / "libxrt_coreutil.so").write_bytes(b"")
    assert _host_xrt_lib_dir(prefix) == lib


def test_host_xrt_lib_dir_uses_dll_root_layout(tmp_path):
    prefix = tmp_path / "xrt-runtime"
    prefix.mkdir()
    (prefix / "xrt_coreutil.lib").write_bytes(b"")
    (prefix / "xrt_coreutil.dll").write_bytes(b"")
    assert _is_host_xrt_prefix(prefix)
    assert _host_xrt_lib_dir(prefix) == prefix


def test_host_xrt_include_dir_reads_prefix_source_tree(tmp_path):
    include = tmp_path / "XRT" / "src" / "runtime_src" / "core" / "include"
    (include / "xrt" / "experimental").mkdir(parents=True)
    (include / "xrt" / "xrt_bo.h").write_text("", encoding="utf-8")
    (include / "xrt" / "experimental" / "xrt_ext.h").write_text("", encoding="utf-8")
    runtime = tmp_path / "xrt-runtime"
    runtime.mkdir()
    (runtime / "xrt_coreutil.dll").write_bytes(b"")
    assert _host_xrt_include_dir(runtime, windows=True) is None
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
