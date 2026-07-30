# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared HRX runtime discovery, loader setup, and probe helpers."""

import contextlib
import ctypes
import os
from pathlib import Path

from build_tools.dev import build, paths


def libhrx_path(bundle=None, runtime_package=False):
    if runtime_package:
        try:
            import rocm_npu_runtime

            return Path(rocm_npu_runtime.libhrx_path()).resolve()
        except (ImportError, FileNotFoundError) as error:
            print(f"  [MISS] installed rocm-npu-runtime package: {error}")
            return None
    if bundle:
        relative = "bin/hrx.dll" if paths.is_windows() else "lib/libhrx.so"
        candidate = (Path(bundle) / relative).resolve()
        return candidate if candidate.is_file() else None
    relative = "bin/hrx.dll" if paths.is_windows() else "lib/libhrx.so"
    candidate = build.HRX_INSTALL / relative
    return candidate if candidate.is_file() else None


def add_hrx_install_env(env, install=None):
    """Expose an installed HRX package to ctypes and native dependencies."""
    selected_install = Path(install or build.HRX_INSTALL)
    lib_dir = selected_install / ("bin" if paths.is_windows() else "lib")
    env["HRX_INSTALL"] = str(selected_install)
    env["LIBHRX_DIR"] = str(lib_dir)
    loader_var = "PATH" if paths.is_windows() else "LD_LIBRARY_PATH"
    env[loader_var] = (
        str(lib_dir) + os.pathsep + env.get(loader_var, "")
    )
    return env


def hrx_check(lib, status, what):
    """Return ``(ok, message)`` for and consume an ``hrx_status_t``."""
    if status is None:
        return True, "ok"
    message = ctypes.c_char_p()
    length = ctypes.c_size_t()
    lib.hrx_status_ignore(
        lib.hrx_status_to_string(
            status, ctypes.byref(message), ctypes.byref(length)
        )
    )
    code = lib.hrx_status_code(status)
    text = (
        message.value.decode("utf-8", "replace") if message.value else "?"
    )
    if message.value:
        lib.hrx_status_free_message(message)
    lib.hrx_status_ignore(status)
    return False, f"{what} failed (code {code}): {text}"


def bind_libhrx(lib):
    void = ctypes.c_void_p
    pointer = ctypes.POINTER
    signatures = {
        "hrx_status_code": (ctypes.c_int, [void]),
        "hrx_status_ignore": (None, [void]),
        "hrx_status_free_message": (None, [ctypes.c_char_p]),
        "hrx_status_to_string": (
            void,
            [void, pointer(ctypes.c_char_p), pointer(ctypes.c_size_t)],
        ),
        "hrx_gpu_initialize": (void, [ctypes.c_uint32]),
        "hrx_gpu_shutdown": (void, []),
        "hrx_gpu_device_count": (void, [pointer(ctypes.c_int)]),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = result, arguments


def probe_libhrx(bundle=None, runtime_package=False):
    lib_path = libhrx_path(bundle, runtime_package)
    library_name = "hrx.dll" if paths.is_windows() else "libhrx.so"
    if not lib_path:
        print(f"  [MISS] {library_name} - run: python dev.py build hrx")
        return 1
    print(f"  [ok ] {library_name}: {lib_path}")
    with contextlib.ExitStack() as stack:
        if paths.is_windows() and hasattr(os, "add_dll_directory"):
            stack.enter_context(os.add_dll_directory(str(lib_path.parent)))
        load_args = (
            {} if paths.is_windows() else {"mode": ctypes.RTLD_GLOBAL}
        )
        try:
            library = ctypes.CDLL(str(lib_path), **load_args)
        except OSError as error:
            print(f"  [MISS] loading {library_name} failed: {error}")
            return 1
        bind_libhrx(library)
        ok, message = hrx_check(
            library,
            library.hrx_gpu_initialize(0),
            "hrx_gpu_initialize",
        )
        if not ok:
            print(f"  [MISS] {message}")
            return 1
        print("  [ok ] hrx_gpu_initialize")
        rc = 0
        count = ctypes.c_int(0)
        ok, message = hrx_check(
            library,
            library.hrx_gpu_device_count(ctypes.byref(count)),
            "hrx_gpu_device_count",
        )
        if not ok:
            print(f"  [MISS] {message}")
            rc = 1
        else:
            print(
                f"  [{'ok ' if count.value else 'MISS'}] "
                f"hrx gpu devices: {count.value}"
            )
            if count.value == 0:
                rc = 1
        library.hrx_status_ignore(library.hrx_gpu_shutdown())
        return rc
