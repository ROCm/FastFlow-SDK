# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compiler-owned native runtime and relocation policy."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from build_tools.native_dependencies import (
    elf_rpath,
    is_loadable_elf,
    set_elf_rpath,
)


def relocate_linux_compiler_rpaths(product: Path) -> dict[str, str]:
    """Replaces build-tree ELF search paths with product-relative locations."""
    product = Path(product)
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to relocate the Linux compiler product")
    library_roots = tuple(
        path
        for path in (product / "lib", product / "lib" / "aie")
        if path.is_dir()
    )
    rewritten = {}
    for path in sorted(
        item
        for item in product.rglob("*")
        if item.is_file() and is_loadable_elf(item)
    ):
        current = elf_rpath(path, patchelf=patchelf)
        entries = current.split(":") if current else []
        relative_entries = [entry for entry in entries if not entry.startswith("/")]
        for library_root in library_roots:
            relative = os.path.relpath(library_root, path.parent).replace(os.sep, "/")
            relative_entries.append(
                "$ORIGIN" if relative == "." else f"$ORIGIN/{relative}"
            )
        relocated = ":".join(dict.fromkeys(relative_entries))
        if relocated == current:
            continue
        set_elf_rpath(path, relocated, patchelf=patchelf)
        rewritten[path.relative_to(product).as_posix()] = relocated
    return rewritten


def windows_system_runtime(name: str) -> bool:
    """Returns whether a compiler dependency is provided by Windows or CPython."""
    name = name.casefold()
    # Stable-ABI extensions resolve the implementation-neutral CPython DLL.
    if name == "python3.dll":
        return True
    if name.startswith(("api-ms-", "ext-ms-")):
        return True
    return (
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / name
    ).is_file()


def linux_system_runtime(name: str) -> bool:
    """Returns whether manylinux owns a compiler runtime library."""
    return (
        name.startswith(("ld-linux", "linux-vdso"))
        or name
        in {
            "libc.so.6",
            "libdl.so.2",
            "libgcc_s.so.1",
            "libm.so.6",
            "libpthread.so.0",
            "librt.so.1",
            "libstdc++.so.6",
            "libutil.so.1",
            "libz.so.1",
        }
    )
