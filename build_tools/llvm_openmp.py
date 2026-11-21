# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Builds LLVM's OpenMP runtime for the Linux FastFlowLM payload.

FastFlowLM's engine libraries are compiled with GCC and import libgomp. libomp
implements the same ``GOMP_*`` ABI under Apache-2.0 WITH LLVM-exception, so the
packager relinks those libraries against a bundled libomp instead of leaving
the OpenMP runtime to the host.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

from build_tools import environment
from build_tools.artifact import reset_directory
from build_tools.native_dependencies import elf_soname, require_patchelf

LIBRARY = "libomp.so"
GNU_LIBRARY = "libgomp.so.1"
LICENSE = "LICENSE.TXT"
MARKER = "llvm-openmp.json"
COMPONENT_NAME = "llvm-openmp"
COMPONENT_LICENSE = "Apache-2.0 WITH LLVM-exception"
BUNDLED_LICENSE = "llvm-openmp-LICENSE.TXT"
BUILD_JOBS = os.environ.get("RNS_BUILD_JOBS") or str(os.cpu_count() or 8)

# The parts of llvm-project that the runtimes build reads for openmp alone.
SPARSE_PATHS = (
    "/openmp/",
    "/runtimes/",
    "/cmake/",
    "/llvm/cmake/",
    "/llvm/utils/llvm-lit/",
    "/third-party/",
)
CMAKE_OPTIONS = (
    "-DLLVM_ENABLE_RUNTIMES=openmp",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DOPENMP_ENABLE_LIBOMPTARGET=OFF",
    "-DOPENMP_ENABLE_OMPT_TOOLS=OFF",
    "-DLIBOMP_OMPD_SUPPORT=OFF",
    "-DLIBOMP_ARCHER_SUPPORT=OFF",
    "-DLIBOMP_USE_HWLOC=OFF",
    "-DLIBOMP_INSTALL_ALIASES=OFF",
    "-DLLVM_INCLUDE_TESTS=OFF",
    "-DLLVM_INCLUDE_DOCS=OFF",
)
# Version nodes and symbol prefixes that belong to the OpenMP runtime ABI.
_OPENMP_VERSION_PREFIXES = ("GOMP_", "OMP_", "GOACC_")
_OPENMP_SYMBOL_PREFIXES = ("GOMP_", "GOACC_", "omp_", "kmp_", "__kmpc_")
_HEX = re.compile(r"[0-9a-fA-F]{8,16}")


def pinned_source(root: Path) -> dict[str, str]:
    """Returns the llvm-project remote and commit pinned in versions.toml."""
    versions = tomllib.loads(
        (root / "manifests" / "versions.toml").read_text(encoding="utf-8")
    )
    llvm = versions["llvm"]
    return {"repo": f"https://{llvm['repo']}.git", "commit": llvm["commit"]}


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def openmp_source(root: Path, source_dir: Path) -> Path:
    """Returns an llvm-project tree at the pinned commit that contains openmp.

    A projects/llvm-project checkout at the pin is used as is. Otherwise only
    the directories in ``SPARSE_PATHS`` are fetched into ``source_dir``.
    """
    commit = pinned_source(root)["commit"]
    for candidate in (root / "projects" / "llvm-project", source_dir):
        if (candidate / "openmp" / "CMakeLists.txt").is_file() and _git_head(
            candidate
        ) == commit:
            return candidate
    reset_directory(source_dir, root / "build")
    git = ["git", "-C", str(source_dir)]
    environment.sh(["git", "init", "-q", str(source_dir)])
    environment.sh([*git, "remote", "add", "origin", pinned_source(root)["repo"]])
    environment.sh([*git, "sparse-checkout", "set", "--no-cone", *SPARSE_PATHS])
    environment.sh(
        [*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit]
    )
    environment.sh(
        [*git, "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD"]
    )
    return source_dir


def build(root: Path, source: Path, build_dir: Path, install_dir: Path) -> Path:
    """Configures, builds, and installs libomp, then records its provenance."""
    env = environment.cmake_env()
    cc, cxx = environment.compilers()
    reset_directory(build_dir, root / "build")
    reset_directory(install_dir, root / "build")
    cmake = environment.cmake_bin()
    environment.sh(
        [
            cmake,
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(source / "runtimes"),
            "-B",
            environment.cmake_path(build_dir),
            f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(install_dir)}",
            *CMAKE_OPTIONS,
            *environment.cmake_compiler_flags(cc, cxx),
        ],
        env=env,
    )
    environment.sh(
        [
            cmake,
            "--build",
            environment.cmake_path(build_dir),
            "--target",
            "install",
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    library = install_dir / "lib" / LIBRARY
    soname = elf_soname(library, patchelf=require_patchelf())
    if soname != LIBRARY:
        raise RuntimeError(f"{library} has SONAME {soname!r}, expected {LIBRARY}")
    shutil.copy2(source / "openmp" / LICENSE, install_dir / LICENSE)
    marker = {
        **pinned_source(root),
        "cmake_options": list(CMAKE_OPTIONS),
        "compiler": cxx or "default",
    }
    (install_dir / MARKER).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return install_dir


def load_install(root: Path, install_dir: Path) -> dict[str, object]:
    """Returns the installed libomp, its license, and provenance at the pin."""
    marker_path = install_dir / MARKER
    library = install_dir / "lib" / LIBRARY
    license_file = install_dir / LICENSE
    if not (marker_path.is_file() and library.is_file() and license_file.is_file()):
        raise RuntimeError(
            f"LLVM OpenMP runtime is not installed under {install_dir}; "
            "run: python dev.py build llvm-openmp"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    commit = pinned_source(root)["commit"]
    if marker.get("commit") != commit:
        raise RuntimeError(
            f"LLVM OpenMP runtime was built from {marker.get('commit')}, not the "
            f"pinned llvm-project {commit}; run: python dev.py build llvm-openmp"
        )
    return {"library": library, "license": license_file, "marker": marker}


def component(marker: dict[str, object]) -> dict[str, str]:
    """Returns the package-manifest component record for the bundled runtime."""
    return {
        "name": COMPONENT_NAME,
        "version": str(marker["commit"]),
        "upstream": str(marker["repo"]).removesuffix(".git"),
        "license": COMPONENT_LICENSE,
    }


def parse_dynamic_symbols(
    objdump_output: str,
) -> tuple[set[tuple[str, str | None]], set[tuple[str, str | None]]]:
    """Splits ``objdump -T`` output into (defined, undefined) name/version pairs."""
    defined: set[tuple[str, str | None]] = set()
    undefined: set[tuple[str, str | None]] = set()
    for line in objdump_output.splitlines():
        parts = line.split()
        if len(parts) < 4 or not _HEX.fullmatch(parts[0]):
            continue
        name = parts[-1]
        version: str | None = parts[-2].strip("()")
        if _HEX.fullmatch(version) or version == "Base":
            version = None
        (undefined if "*UND*" in parts else defined).add((name, version))
    return defined, undefined


def _dynamic_symbols(path: Path, objdump: str):
    output = subprocess.run(
        [objdump, "-T", "--wide", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_dynamic_symbols(output)


def missing_openmp_symbols(
    provided: set[tuple[str, str | None]],
    required: set[tuple[str, str | None]],
) -> list[str]:
    """Returns OpenMP imports in ``required`` that ``provided`` does not define."""
    provided_names = {name for name, _ in provided}
    missing = []
    for name, version in sorted(required, key=lambda item: (item[0], item[1] or "")):
        is_openmp = (
            version.startswith(_OPENMP_VERSION_PREFIXES)
            if version
            else name.startswith(_OPENMP_SYMBOL_PREFIXES)
        )
        if not is_openmp:
            continue
        if (name, version) in provided or (version is None and name in provided_names):
            continue
        missing.append(f"{name}@{version}" if version else name)
    return missing


def verify_openmp_imports(library: Path, consumers: list[Path]) -> None:
    """Requires ``library`` to define every OpenMP symbol ``consumers`` import."""
    objdump = shutil.which("objdump")
    if not objdump:
        raise RuntimeError("objdump is required to verify the bundled OpenMP runtime")
    provided, _ = _dynamic_symbols(library, objdump)
    failures = []
    for consumer in consumers:
        _, required = _dynamic_symbols(consumer, objdump)
        missing = missing_openmp_symbols(provided, required)
        if missing:
            failures.append(f"{consumer.name}: {', '.join(missing)}")
    if failures:
        raise RuntimeError(
            f"{library.name} does not define OpenMP symbols the payload imports:\n  "
            + "\n  ".join(failures)
        )
