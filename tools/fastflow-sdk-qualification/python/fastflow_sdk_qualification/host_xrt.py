# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host XRT install discovery for qualify ``--xrt-flm`` and ``build fastflowlm-xrt``."""

from __future__ import annotations

import os
from pathlib import Path

_XRT_LOADER_NAMES = (
    "xrt_coreutil.dll",
    "xrt_core.dll",
    "libxrt_coreutil.so",
    "libxrt_coreutil.so.2",
    "libxrt_coreutil.so.1",
)

_XRT_RUNTIME_NAMES = _XRT_LOADER_NAMES + (
    "xrt_coreutil.lib",
    "xbutil.exe",
    "xbutil2.exe",
    "xbutil",
    "xbutil2",
)


def directory_has_xrt_loader(path: Path) -> bool:
    """True when *path* contains XRT shared libraries the process can load."""
    if not path.is_dir():
        return False
    return any((path / name).is_file() for name in _XRT_LOADER_NAMES)


def directory_has_xrt_runtime(path: Path) -> bool:
    """True when *path* contains XRT link or loader libraries."""
    if not path.is_dir():
        return False
    return any((path / name).is_file() for name in _XRT_RUNTIME_NAMES)


def has_xrt_headers(include_dir: Path) -> bool:
    return (include_dir / "xrt" / "experimental" / "xrt_ext.h").is_file() and (
        include_dir / "xrt" / "xrt_bo.h"
    ).is_file()


def is_host_xrt_loader_prefix(prefix: Path) -> bool:
    """True when *prefix* contains XRT shared libraries the process can load."""
    if not prefix.is_dir():
        return False
    return (
        directory_has_xrt_loader(prefix)
        or directory_has_xrt_loader(prefix / "lib")
        or directory_has_xrt_loader(prefix / "bin")
    )


def is_host_xrt_prefix(prefix: Path) -> bool:
    """True when *prefix* contains XRT runtime libraries or SDK headers."""
    if not prefix.is_dir():
        return False
    return (
        directory_has_xrt_runtime(prefix)
        or directory_has_xrt_runtime(prefix / "lib")
        or directory_has_xrt_runtime(prefix / "bin")
        or has_xrt_headers(prefix / "include")
        or has_xrt_headers(prefix / "src" / "runtime_src" / "core" / "include")
    )


def prefix_from_runtime_dir(directory: Path) -> Path:
    """Map a directory that holds XRT DLLs/SOs to an install prefix."""
    if directory.name.lower() in {"bin", "lib"}:
        return directory.parent
    return directory


def windows_install_roots(environ) -> list[Path]:
    """Vendor-default Windows XRT locations (not machine-local scratch paths)."""
    roots: list[Path] = []
    seen: set[Path] = set()
    bases = []
    for key in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        value = (environ.get(key) or "").strip()
        if value:
            bases.append(Path(value))
    for base in bases:
        for relative in (Path("Xilinx") / "XRT", Path("AMD") / "XRT"):
            candidate = base / relative
            if candidate in seen:
                continue
            seen.add(candidate)
            roots.append(candidate)
    xilinx = Path(r"C:\Xilinx\XRT")
    if xilinx not in seen:
        roots.append(xilinx)
    return roots


def path_runtime_dirs(environ) -> list[Path]:
    directories: list[Path] = []
    seen: set[Path] = set()
    for entry in (environ.get("PATH") or "").split(os.pathsep):
        if not entry.strip():
            continue
        directory = Path(entry.strip())
        if directory in seen or not directory_has_xrt_loader(directory):
            continue
        seen.add(directory)
        directories.append(directory)
    return directories


def host_xrt_prefix_candidates(*, windows=None, environ=None):
    """Ordered host XRT prefix candidates (env override first)."""
    selected_windows = os.name == "nt" if windows is None else windows
    selected_environ = os.environ if environ is None else environ
    candidates = []
    env_root = (selected_environ.get("XILINX_XRT") or "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    if selected_windows:
        candidates.extend(windows_install_roots(selected_environ))
    else:
        candidates.append(Path("/opt/xilinx/xrt"))
    candidates.extend(
        prefix_from_runtime_dir(directory)
        for directory in path_runtime_dirs(selected_environ)
    )
    return candidates


def host_xrt_prefix(*, windows=None, environ=None, require_loader=False):
    """Return the host XRT install prefix, or ``None`` if it is not present.

    Discovery is ``$XILINX_XRT``, the platform default install root, then
    ``PATH`` entries that hold XRT loader libraries. ``require_loader=True``
    requires loadable shared libraries. An explicit ``$XILINX_XRT`` with
    ``require_loader`` is authoritative: a tree that cannot load returns
    ``None``.
    """
    selected_windows = os.name == "nt" if windows is None else windows
    selected_environ = os.environ if environ is None else environ
    predicate = is_host_xrt_loader_prefix if require_loader else is_host_xrt_prefix
    env_root = (selected_environ.get("XILINX_XRT") or "").strip()
    if require_loader and env_root:
        prefix = Path(env_root).expanduser()
        try:
            resolved = prefix.resolve()
        except OSError:
            resolved = prefix
        return resolved if predicate(prefix) else None
    seen: set[Path] = set()
    for prefix in host_xrt_prefix_candidates(
        windows=selected_windows, environ=selected_environ
    ):
        try:
            resolved = prefix.resolve()
        except OSError:
            resolved = prefix
        if resolved in seen:
            continue
        seen.add(resolved)
        if predicate(prefix):
            return resolved
    return None


def host_xrt_library_dirs(prefix: Path, *, windows=None) -> list[str]:
    """Loader directories under a host XRT prefix, if they exist."""
    selected_windows = os.name == "nt" if windows is None else windows
    relatives = (
        ("bin", "lib", "")
        if selected_windows
        else ("lib", os.path.join("lib", "x86_64-linux-gnu"), "bin", "")
    )
    directories: list[str] = []
    seen: set[str] = set()
    for relative in relatives:
        path = prefix if relative == "" else prefix / relative
        if not path.is_dir():
            continue
        if relative == "bin" and not directory_has_xrt_loader(path):
            continue
        if relative == "" and not directory_has_xrt_loader(path):
            continue
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        directories.append(resolved)
    return directories
