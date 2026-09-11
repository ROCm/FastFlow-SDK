# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""HRX runtime, CTS, and standalone native CLI build commands."""

import json
import os
import shutil
import sys

from build_tools import environment, hrx_runtime, paths
from build_tools.artifact import directory_state, git_source_state
from build_tools.dev import patches
from build_tools.package_version import default_package_version

BUILD_JOBS = os.environ.get("RNS_BUILD_JOBS") or str(os.cpu_count() or 8)
FRESH_BUILD_MARKER = ".rns-fresh-build.json"


def _hrx_test_cmake_options(include_tests, include_cts):
    enabled = "ON" if include_tests or include_cts else "OFF"
    # IREE's async CTS test libraries include benchmark support whenever tests
    # are enabled, even when only non-benchmark targets are built. Several host
    # tests also link the local-sync HAL driver, so that driver must be on for
    # the same configure.
    return (
        f"-DIREE_BUILD_TESTS={enabled}",
        f"-DIREE_BUILD_BENCHMARKS={enabled}",
        f"-DIREE_HAL_DRIVER_LOCAL_SYNC={enabled}",
    )


def _prepare_fresh_cmake_tree(path):
    """Clear generated CMake state in an owned runtime build tree."""
    allowed = {paths.HRX_CMAKE_BUILD.resolve(), paths.ROCM_NPU_CLI_BUILD.resolve()}
    if path.resolve() not in allowed:
        raise RuntimeError(f"refusing to clear unexpected build directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _build_hrx_cts_binaries(env):
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(paths.HRX_CMAKE_BUILD),
            *[
                f"--target=iree_hal_drivers_amdxdna_cts_{test}"
                for test in hrx_runtime.HRX_CTS_TESTS
            ],
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )


def cmd_build_hrx(args):
    source = paths.PROJ / "hrx"
    if not (source / "CMakeLists.txt").exists():
        sys.exit("missing projects/hrx (run: git submodule update --init projects/hrx)")
    patches.apply_patches({"hrx"})
    fresh = getattr(args, "fresh", False)
    include_tests = getattr(args, "include_tests", False)
    include_cts = getattr(args, "include_cts", False)
    if fresh:
        patches.validate_declared_patch_state("hrx", source)
        _prepare_fresh_cmake_tree(paths.HRX_CMAKE_BUILD)
    cc, cxx = environment.compilers()
    env = environment.cmake_env()
    environment.sh(
        [
            environment.cmake_bin(),
            *(["--fresh"] if fresh else []),
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(source),
            "-B",
            environment.cmake_path(paths.HRX_CMAKE_BUILD),
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            "-DIREE_HAL_DRIVER_AMDGPU=OFF",
            "-DIREE_HAL_DRIVER_AMDXDNA=ON",
            "-DIREE_HAL_DRIVER_AMDXDNA_BUILD_QUALIFICATION_TOOLS=ON",
            *_hrx_test_cmake_options(include_tests, include_cts),
            "-DLIBHRX_BUILD_CTS=OFF",
            "-DHRX_INSTALL_TESTS=OFF",
            "-DLIBHRX_BUILD_PASSTHROUGH=OFF",
            "-DLIBHRX_BUILD_HIP_BINDING=OFF",
            "-DLOOM_BUILD=OFF",
            "-DIREE_HAL_DRIVER_NULL=OFF",
            *environment.cmake_source_override("FETCHCONTENT_SOURCE_DIR_FLATCC"),
            *environment.cmake_compiler_flags(cc, cxx),
            *environment.cmake_ccache_flags(),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(paths.HRX_CMAKE_BUILD),
            *(["--clean-first"] if fresh else []),
            "--target",
            "iree_hal_drivers_amdxdna_registration_registration",
            "iree_hal_drivers_amdxdna_resource_pressure_test_tool",
            "hrx",
            "flatcc_runtime",
            # Public dist installs hrx-info; a targeted build otherwise skips it.
            "libhrx_tools_hrx-info",
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    if include_tests:
        environment.sh(
            [
                environment.cmake_bin(),
                "--build",
                environment.cmake_path(paths.HRX_CMAKE_BUILD),
                *[
                    f"--target={name}"
                    for name in hrx_runtime.hrx_amdxdna_host_test_targets()
                ],
                "-j",
                BUILD_JOBS,
            ],
            env=env,
        )
        environment.sh(
            [
                "ctest",
                "--test-dir",
                environment.cmake_path(paths.HRX_CMAKE_BUILD),
                "-R",
                hrx_runtime.hrx_amdxdna_host_test_regex(),
                "--output-on-failure",
            ],
            env=env,
        )
    shutil.rmtree(paths.HRX_INSTALL, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--install",
            environment.cmake_path(paths.HRX_CMAKE_BUILD),
            "--prefix",
            environment.cmake_path(paths.HRX_INSTALL),
            "--component",
            "HrxPublicDist",
        ],
        env=env,
    )
    _smoke_installed_hrx(source, cc, cxx, env)
    if fresh:
        marker = {
            "source": git_source_state(source),
            "install": directory_state(
                paths.HRX_INSTALL, exclude_names=(FRESH_BUILD_MARKER,)
            ),
        }
        (paths.HRX_INSTALL / FRESH_BUILD_MARKER).write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
    if include_cts:
        _build_hrx_cts_binaries(env)
    summary = "minimal amdxdna HAL + installed package"
    if include_tests:
        summary += f" + {len(hrx_runtime.hrx_amdxdna_host_tests())} host unit tests"
    if include_cts:
        summary += f" + on-device CTS ({', '.join(hrx_runtime.HRX_CTS_TESTS)})"
    print(f"build hrx ({summary}): ok ->", paths.HRX_INSTALL)


def _smoke_installed_hrx(source, cc, cxx, env):
    shutil.rmtree(paths.HRX_PACKAGE_SMOKE_BUILD, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(source / "libhrx" / "cts" / "package_smoke"),
            "-B",
            environment.cmake_path(paths.HRX_PACKAGE_SMOKE_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={environment.cmake_path(paths.HRX_INSTALL)}",
            *environment.cmake_compiler_flags(cc, cxx),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(paths.HRX_PACKAGE_SMOKE_BUILD),
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    smoke_env = dict(env)
    loader = "PATH" if paths.is_windows() else "LD_LIBRARY_PATH"
    lib_dir = paths.HRX_INSTALL / ("bin" if paths.is_windows() else "lib")
    smoke_env[loader] = str(lib_dir) + os.pathsep + smoke_env.get(loader, "")
    for name in ("hrx_package_smoke", "hrx_package_smoke_cxx"):
        environment.sh(
            [paths.HRX_PACKAGE_SMOKE_BUILD / paths.executable_name(name)],
            env=smoke_env,
        )


def cmd_build_rocm_npu_cli(args):
    """Builds the standalone C++ runtime CLI without linking HRX."""
    env = environment.cmake_env()
    fresh = getattr(args, "fresh", False)
    if fresh:
        _prepare_fresh_cmake_tree(paths.ROCM_NPU_CLI_BUILD)
    version = default_package_version(paths.ROOT)
    source_revision = git_source_state(paths.ROOT)["revision"]
    cc, cxx = environment.compilers()
    environment.sh(
        [
            environment.cmake_bin(),
            *(["--fresh"] if fresh else []),
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(paths.ROOT / "tools" / "rocm-npu"),
            "-B",
            environment.cmake_path(paths.ROCM_NPU_CLI_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DROCM_NPU_VERSION={version}",
            f"-DROCM_NPU_SOURCE_REVISION={source_revision}",
            f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(paths.ROCM_NPU_CLI_INSTALL)}",
            *environment.cmake_compiler_flags(cc, cxx),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(paths.ROCM_NPU_CLI_BUILD),
            *(["--clean-first"] if fresh else []),
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    shutil.rmtree(paths.ROCM_NPU_CLI_INSTALL, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--install",
            environment.cmake_path(paths.ROCM_NPU_CLI_BUILD),
        ],
        env=env,
    )
    installed = paths.ROCM_NPU_CLI_INSTALL / "bin" / paths.executable_name("rocm-npu")
    if not installed.exists():
        sys.exit(f"rocm-npu CLI install did not produce {installed}")
    if fresh:
        marker = {
            "source": git_source_state(paths.ROOT),
            "hrx_headers": git_source_state(paths.PROJ / "hrx"),
            "version": version,
            "compilers": {"c": str(cc), "cxx": str(cxx)},
            "install": directory_state(
                paths.ROCM_NPU_CLI_INSTALL,
                exclude_names=(FRESH_BUILD_MARKER,),
            ),
        }
        (paths.ROCM_NPU_CLI_INSTALL / FRESH_BUILD_MARKER).write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("build rocm-npu-cli: ok ->", installed)
