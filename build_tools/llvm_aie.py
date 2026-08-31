# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Peano LLVM-AIE source-build install validation."""

from __future__ import annotations

import json
import platform
import tomllib
from pathlib import Path

from build_tools import paths
from build_tools.artifact import git_revision

RUNTIME_TARGETS = (
    "aie2-none-unknown-elf",
    "aie2p-none-unknown-elf",
    "aie2ps-none-unknown-elf",
)
BUILD_STAMP = ".rns-llvm-aie-build.json"
WINDOWS_COMPILER_CHOICES = ("cl", "clang-cl", "env")


def manifest() -> dict[str, object]:
    with open(paths.ROOT / "manifests" / "versions.toml", "rb") as stream:
        return tomllib.load(stream)["peano"]


def normalize_windows_compiler(windows_compiler: str) -> str:
    compiler = windows_compiler.casefold()
    if compiler not in WINDOWS_COMPILER_CHOICES:
        raise RuntimeError(
            "llvm-aie Windows compiler must be one of: "
            + ", ".join(WINDOWS_COMPILER_CHOICES)
        )
    return compiler


def expected_build_state(
    windows_compiler: str | None = None,
) -> dict[str, object]:
    state = {
        "schema_version": 1,
        "source_revision": git_revision(paths.PROJ / "llvm-aie"),
        "cache_recipe": manifest().get("cache_recipe"),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }
    if paths.is_windows() and windows_compiler is not None:
        state["windows_compiler"] = normalize_windows_compiler(windows_compiler)
    return state


def write_build_stamp(
    install_dir: Path | None = None,
    *,
    windows_compiler: str | None = None,
) -> None:
    root = paths.LLVM_AIE_INSTALL if install_dir is None else Path(install_dir)
    stamp = root / BUILD_STAMP
    stamp.write_text(
        json.dumps(
            expected_build_state(windows_compiler),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def verify_install(
    install_dir: Path | None = None,
    *,
    windows_compiler: str | None = None,
) -> Path:
    """Validate the source-built Peano install tree consumed by IRON."""
    root = paths.LLVM_AIE_INSTALL if install_dir is None else Path(install_dir)
    required = [
        Path("bin") / paths.executable_name("clang"),
        Path("bin") / paths.executable_name("llc"),
        Path("bin") / paths.executable_name("ld.lld"),
        Path("bin") / paths.executable_name("llvm-ar"),
    ]
    for target in RUNTIME_TARGETS:
        required.extend(
            [
                Path("include") / target / "stdint.h",
                Path("lib") / target / "libc.a",
            ]
        )
    required.append(Path(BUILD_STAMP))
    missing = [
        relative.as_posix() for relative in required if not (root / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Peano llvm-aie install is incomplete: {root}\n  missing: "
            + "\n  ".join(missing)
        )
    recorded = json.loads((root / BUILD_STAMP).read_text(encoding="utf-8"))
    expected = expected_build_state(windows_compiler)
    mismatches = {
        key: {"expected": value, "recorded": recorded.get(key)}
        for key, value in expected.items()
        if recorded.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Peano llvm-aie install stamp is stale: {root / BUILD_STAMP}\n"
            f"  mismatches: {mismatches}\n"
            f"  recorded: {recorded}"
        )
    return root
