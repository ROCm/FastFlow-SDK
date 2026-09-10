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
    _windows_install_roots,
    preset_cache_variables,
)
from rocm_npu_qualification import host_xrt, runner

pytestmark = pytest.mark.presubmit


def _write_xrt_headers(include: Path) -> None:
    (include / "xrt" / "experimental").mkdir(parents=True)
    (include / "xrt" / "xrt_bo.h").write_text("", encoding="utf-8")
    (include / "xrt" / "experimental" / "xrt_ext.h").write_text("", encoding="utf-8")


def test_fastflowlm_and_qualify_share_host_xrt_discovery():
    assert _host_xrt_prefix_candidates is host_xrt.host_xrt_prefix_candidates
    assert runner._host_xrt_prefix_candidates is host_xrt.host_xrt_prefix_candidates
    assert _windows_install_roots is host_xrt.windows_install_roots
    assert runner._windows_install_roots is host_xrt.windows_install_roots


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
    program_files = Path(environ["ProgramFiles"])
    # Join the same way production does so this stays valid on POSIX CI
    # (Path(r"C:\Program Files\Xilinx\XRT") is a single name there).
    assert program_files / "Xilinx" / "XRT" in candidates
    assert program_files / "AMD" / "XRT" in candidates
    assert Path(r"C:\Xilinx\XRT") in candidates
    assert Path(r"C:\dev\xrt-prefix") not in candidates
    assert Path(r"C:\dev\xrtNPUfromDLL") not in candidates
    linux = _host_xrt_prefix_candidates(windows=False, environ=environ)
    assert Path("/opt/xilinx/xrt") in linux
    assert program_files / "Xilinx" / "XRT" not in linux


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
    monkeypatch.setattr(host_xrt, "windows_install_roots", lambda environ: [])
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


def test_host_xrt_prefix_accepts_headers_only_sdk(tmp_path):
    sdk = tmp_path / "sdk"
    _write_xrt_headers(sdk / "include")
    found = _host_xrt_prefix(
        windows=True,
        environ={
            "XILINX_XRT": str(sdk),
            "PATH": "",
            "ProgramFiles": str(tmp_path / "unused-pf"),
        },
    )
    assert found == sdk.resolve()
    assert _is_host_xrt_prefix(sdk)


def test_qualify_host_xrt_skips_headers_only_vendor_root_for_path_loader(
    tmp_path, monkeypatch
):
    sdk = tmp_path / "Program Files" / "Xilinx" / "XRT"
    _write_xrt_headers(sdk / "include")
    monkeypatch.setattr(host_xrt, "windows_install_roots", lambda environ: [sdk])
    path_bin = tmp_path / "path-xrt" / "bin"
    path_bin.mkdir(parents=True)
    (path_bin / "xrt_coreutil.dll").write_bytes(b"")
    environ = {
        "XILINX_XRT": "",
        "PATH": str(path_bin),
    }
    found = host_xrt.host_xrt_prefix(windows=True, require_loader=True, environ=environ)
    assert found == (tmp_path / "path-xrt").resolve()
    build = _host_xrt_prefix(windows=True, environ=environ)
    assert build == sdk.resolve()


def test_qualify_host_xrt_env_headers_only_is_unavailable(tmp_path):
    sdk = tmp_path / "sdk"
    _write_xrt_headers(sdk / "include")
    path_bin = tmp_path / "path-xrt" / "bin"
    path_bin.mkdir(parents=True)
    (path_bin / "xrt_coreutil.dll").write_bytes(b"")
    found = host_xrt.host_xrt_prefix(
        windows=True,
        require_loader=True,
        environ={
            "XILINX_XRT": str(sdk),
            "PATH": str(path_bin),
            "ProgramFiles": str(tmp_path / "unused-pf"),
        },
    )
    assert found is None
    assert (
        runner._host_xrt_prefix(
            windows=True,
            environ={
                "XILINX_XRT": str(sdk),
                "PATH": str(path_bin),
                "ProgramFiles": str(tmp_path / "unused-pf"),
            },
        )
        is None
    )


def test_host_xrt_library_dirs_put_windows_bin_before_lib(tmp_path):
    prefix = tmp_path / "xrt"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    (prefix / "bin" / "xrt_coreutil.dll").write_bytes(b"")
    (prefix / "lib" / "xrt_coreutil.lib").write_bytes(b"")
    windows_dirs = host_xrt.host_xrt_library_dirs(prefix, windows=True)
    assert windows_dirs[0] == str((prefix / "bin").resolve())
    assert str((prefix / "lib").resolve()) in windows_dirs
    linux_dirs = host_xrt.host_xrt_library_dirs(prefix, windows=False)
    assert linux_dirs[0] == str((prefix / "lib").resolve())


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
