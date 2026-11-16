# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the stock ``gemma4-open`` model extension into a runtime bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path

from build_tools import environment
from build_tools.artifact import reset_directory
from build_tools.fastflowlm_deps import dependency_paths

EXTENSION_NAME = "gemma4-open"
EXTENSION_RELATIVE_DIR = Path("share") / "fastflow-sdk" / "extensions" / EXTENSION_NAME
BUILD_JOBS = os.environ.get("RNS_BUILD_JOBS") or str(os.cpu_count() or 8)


def extension_library_dir(bundle: Path) -> Path:
    return bundle / EXTENSION_RELATIVE_DIR / "lib"


def build_gemma4_open_extension(root: Path, bundle: Path) -> Path:
    """Builds the staged ``share/fastflow-sdk/gemma4e`` sources against ``bundle``.

    The extension carries its own Gemma 4 engine, compiled from the patched
    FastFlowLM open sources, so ``gemma4-open-it`` models run without the
    prebuilt engine installed beside ``flm``. Returns ``extension.json``.
    """
    source = bundle / "share" / "fastflow-sdk" / "gemma4e"
    if not (source / "CMakeLists.txt").is_file():
        raise RuntimeError(f"staged Gemma 4 extension sources missing: {source}")
    build_dir = root / "build" / "gemma4-open-extension"
    output = bundle / EXTENSION_RELATIVE_DIR
    reset_directory(build_dir, root / "build")
    dependencies = dependency_paths(root)
    env = environment.cmake_env()
    cc, cxx = environment.compilers()
    environment.sh(
        [
            environment.cmake_bin(),
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(source),
            "-B",
            environment.cmake_path(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DFASTFLOW_SDK_ROOT={environment.cmake_path(bundle)}",
            f"-DFLM_ENGINE_LIB_DIR={environment.cmake_path(bundle / 'lib' / 'flm')}",
            f"-DFLM_EXTENSION_BUNDLE_DIR={environment.cmake_path(output)}",
            f"-DCMAKE_TOOLCHAIN_FILE={environment.cmake_path(Path(dependencies['toolchain']))}",
            f"-DVCPKG_TARGET_TRIPLET={dependencies['triplet']}",
            f"-DVCPKG_INSTALLED_DIR={environment.cmake_path(Path(dependencies['install_root']))}",
            f"-DVCPKG_MANIFEST_DIR={environment.cmake_path(root / '.github' / 'vcpkg' / 'fastflowlm')}",
            "-DVCPKG_MANIFEST_INSTALL=OFF",
            *environment.cmake_compiler_flags(cc, cxx),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(build_dir),
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    manifest = output / "extension.json"
    verify_gemma4_open_extension(bundle)
    return manifest


def verify_gemma4_open_extension(bundle: Path) -> None:
    """Requires a loadable ``gemma4-open`` manifest, plugin, catalog, and xclbins."""
    output = bundle / EXTENSION_RELATIVE_DIR
    manifest_path = output / "extension.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"gemma4-open extension manifest invalid: {manifest_path}"
        ) from error
    if manifest.get("name") != EXTENSION_NAME:
        raise RuntimeError(f"{manifest_path} does not describe {EXTENSION_NAME}")
    for key in ("library", "catalog"):
        value = manifest.get(key)
        if not isinstance(value, str) or not (output / value).is_file():
            raise RuntimeError(f"{manifest_path} {key} is missing: {value}")
    catalog = json.loads((output / manifest["catalog"]).read_text(encoding="utf-8"))
    if "gemma4-open-it" not in catalog.get("models", {}):
        raise RuntimeError(f"{manifest_path} catalog lacks gemma4-open-it")
    if not any((output / "xclbins" / "Gemma4-E2B-IT-NPU2").glob("*.xclbin")):
        raise RuntimeError(f"{output} contains no Gemma4-E2B-IT-NPU2 xclbins")
