# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Windows loader contract for the model engines.

The import table says which DLLs are delay-loaded. Loading each DLL from the
directory beside flm.exe checks that the install still opens. One serve with
that engine's DLL first on PATH checks that the delay-load hook selects it.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

import pytest

from build_tools.fastflow_sdk_archive import (
    WINDOWS_ENGINE_NORMAL_IMPORTS,
    WINDOWS_MODEL_ENGINE_DELAY_IMPORTS,
)
from build_tools.native_dependencies import pe_import_sets

pytestmark = [pytest.mark.fastflowlm, pytest.mark.windows]

_REPO = Path(__file__).resolve().parents[2]

_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010
_CREATE_NO_WINDOW = 0x08000000


def _flm_executable() -> Path | None:
    candidates = [
        _REPO / "build" / "FastFlowLM-install" / "bin" / "flm.exe",
        _REPO / "build" / "FastFlowLM" / "flm.exe",
    ]
    for path in candidates:
        if path.is_file() and (path.parent / "gemma4e_npu.dll").is_file():
            return path
    return None


def _require_flm() -> Path:
    if os.name != "nt":
        pytest.skip("Windows delay-load")
    flm = _flm_executable()
    if flm is None:
        pytest.skip("flm.exe with its engine DLLs is not built")
    return flm


def _loaded_module(pid: int, module_name: str) -> str | None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    psapi.EnumProcessModules.restype = wintypes.BOOL
    psapi.EnumProcessModules.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD
    psapi.GetModuleFileNameExW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    handle = kernel32.OpenProcess(
        _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, pid
    )
    if not handle:
        return None
    try:
        modules = (ctypes.c_void_p * 1024)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModules(
            handle, modules, ctypes.sizeof(modules), ctypes.byref(needed)
        ):
            return None
        count = needed.value // ctypes.sizeof(ctypes.c_void_p)
        buffer = ctypes.create_unicode_buffer(32768)
        target = module_name.casefold()
        for index in range(count):
            if not psapi.GetModuleFileNameExW(
                handle, modules[index], buffer, len(buffer)
            ):
                continue
            if Path(buffer.value).name.casefold() == target:
                return buffer.value
        return None
    finally:
        kernel32.CloseHandle(handle)


def test_flm_delay_loads_model_engines_only():
    flm = _require_flm()
    regular, delay = pe_import_sets(flm)
    regular_set = set(regular)
    delay_set = set(delay)
    assert delay_set == WINDOWS_MODEL_ENGINE_DELAY_IMPORTS
    assert WINDOWS_ENGINE_NORMAL_IMPORTS <= regular_set
    assert WINDOWS_MODEL_ENGINE_DELAY_IMPORTS.isdisjoint(regular_set)
    assert "q4_npu_express.dll" not in delay_set
    assert "llama_npu.dll" not in delay_set


def test_installed_dlls_load():
    flm = _require_flm()
    bin_dir = flm.parent
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


def test_path_selects_a_rebuilt_engine(tmp_path: Path):
    flm = _require_flm()
    model_root = Path(os.environ.get("FLM_MODEL_PATH", Path.home() / ".cache" / "flm"))
    weights = model_root / "models" / "Gemma4-E2B-IT-NPU2" / "model.q4nx"
    share = flm.parent.parent / "share" / "flm"
    if not weights.is_file() or not (share / "xclbins").is_dir():
        pytest.skip("Gemma 4 E2B weights or xclbins are not installed")

    source = flm.parent / "gemma4e_npu.dll"
    replacement = tmp_path / "gemma4e_npu.dll"
    shutil.copy2(source, replacement)
    env = os.environ.copy()
    env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
    env["FLM_MODEL_PATH"] = str(model_root)
    env["FLM_XCLBIN_PATH"] = str(share)
    env["FLM_MODELINFO_PATH"] = str(share / "model_info.json")
    process = subprocess.Popen(
        [
            str(flm),
            "serve",
            "gemma4-it:e2b",
            "--host",
            "127.0.0.1",
            "--port",
            "11491",
        ],
        cwd=flm.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_CREATE_NO_WINDOW,
    )
    mapped = None
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline and process.poll() is None:
            mapped = _loaded_module(process.pid, "gemma4e_npu.dll")
            if mapped:
                break
            time.sleep(0.3)
        stderr = ""
        if process.poll() is not None and process.stderr is not None:
            stderr = process.stderr.read().decode("utf-8", errors="replace")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
    assert mapped is not None, stderr[-2000:]
    assert Path(mapped).resolve() == replacement.resolve()
