# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""FastFlowLM application build command."""

import json
import os
import shutil
import sys
from pathlib import Path

from build_tools import environment, paths
from build_tools.artifact import directory_state, git_source_state
from build_tools.dev import patches
from build_tools.fastflowlm_deps import dependency_errors, dependency_paths

BUILD_JOBS = os.environ.get("RNS_BUILD_JOBS") or str(os.cpu_count() or 8)
FRESH_BUILD_MARKER = ".rns-fresh-build.json"


def _require_fastflowlm_deps(env):
    configured = dependency_paths(paths.ROOT)
    missing = dependency_errors(
        configured,
        cargo_available=bool(environment.which("cargo", env)),
        rustc_version=environment.rustc_version(env),
    )
    if missing:
        sys.exit("missing FastFlowLM prerequisite(s):\n  - " + "\n  - ".join(missing))
    return configured


def _prepare_fresh_cmake_tree(path):
    if path.resolve() != paths.FASTFLOWLM_BUILD.resolve():
        raise RuntimeError(f"refusing to clear unexpected build directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def cmd_build_fastflowlm(args):
    source = paths.PROJ / "FastFlowLM" / "src"
    if not (source / "CMakeLists.txt").exists():
        sys.exit(
            "missing projects/FastFlowLM "
            "(run: git submodule update --init --recursive "
            "projects/FastFlowLM)"
        )
    env = environment.cmake_env()
    dependencies = _require_fastflowlm_deps(env)
    if not paths.is_windows() and not environment.have_header("libdrm/drm.h"):
        sys.exit(
            "missing libdrm/drm.h: FastFlowLM's amdxdna headers require "
            "libdrm (Ubuntu: sudo apt-get install libdrm-dev; "
            "Fedora: sudo dnf install libdrm-devel)"
        )
    patches.apply_patches({"FastFlowLM"})
    fresh = getattr(args, "fresh", False)
    if fresh:
        patches.validate_declared_patch_state("FastFlowLM", paths.PROJ / "FastFlowLM")
        _prepare_fresh_cmake_tree(paths.FASTFLOWLM_BUILD)
        shutil.rmtree(source / "build", ignore_errors=True)
    hrx_config = paths.HRX_INSTALL / "lib" / "cmake" / "hrx" / "hrx-config.cmake"
    if not hrx_config.exists():
        sys.exit(
            f"missing installed HRX package ({hrx_config}) - "
            "run: python dev.py build hrx"
        )
    cc, cxx = environment.compilers()
    environment.sh(
        [
            environment.cmake_bin(),
            *(["--fresh"] if fresh else []),
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(source),
            "-B",
            environment.cmake_path(paths.FASTFLOWLM_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DFLM_USE_HRX=ON",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            "-DFLM_VERSION=0.9.45",
            "-DNPU_VERSION=32.0.203.304",
            f"-DCMAKE_PREFIX_PATH={environment.cmake_path(paths.HRX_INSTALL)}",
            f"-DCMAKE_TOOLCHAIN_FILE={environment.cmake_path(Path(dependencies['toolchain']))}",
            f"-DVCPKG_TARGET_TRIPLET={dependencies['triplet']}",
            f"-DVCPKG_INSTALLED_DIR={environment.cmake_path(Path(dependencies['install_root']))}",
            f"-DVCPKG_MANIFEST_DIR={environment.cmake_path(paths.ROOT / '.github' / 'vcpkg' / 'fastflowlm')}",
            "-DVCPKG_MANIFEST_INSTALL=OFF",
            f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(paths.FASTFLOWLM_INSTALL)}",
            f"-DCMAKE_XCLBIN_PREFIX={environment.cmake_path(source)}",
            *environment.cmake_compiler_flags(cc, cxx),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(paths.FASTFLOWLM_BUILD),
            *(["--clean-first"] if fresh else []),
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    shutil.rmtree(paths.FASTFLOWLM_INSTALL, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--install",
            environment.cmake_path(paths.FASTFLOWLM_BUILD),
        ],
        env=env,
    )
    installed = paths.FASTFLOWLM_INSTALL / "bin" / paths.executable_name("flm")
    if not installed.exists():
        sys.exit(f"FastFlowLM install did not produce {installed}")
    if fresh:
        marker = {
            "source": git_source_state(paths.PROJ / "FastFlowLM"),
            "hrx": json.loads(
                (paths.HRX_INSTALL / FRESH_BUILD_MARKER).read_text(encoding="utf-8")
            ),
            "install": directory_state(
                paths.FASTFLOWLM_INSTALL,
                exclude_names=(FRESH_BUILD_MARKER,),
            ),
        }
        (paths.FASTFLOWLM_INSTALL / FRESH_BUILD_MARKER).write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("build fastflowlm: ok ->", installed)
