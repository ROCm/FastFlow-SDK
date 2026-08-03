# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CMake configuration and developer build commands."""

import json
import os
import shutil
import sys
from pathlib import Path

from build_tools.dev import environment, patches, paths
from build_tools.fastflowlm_artifact import directory_state, git_source_state
from build_tools.fastflowlm_deps import (
    dependency_errors,
    dependency_paths,
)
from build_tools.rocm_npu_wheel_common import default_package_version

LLVM_BUILD = Path(
    os.environ.get("RNS_LLVM_BUILD_DIR") or (paths.BUILD / "llvm")
)
IRON_TOOLS_BUILD = paths.BUILD / "iron-tools"
IRON_TOOLS_INSTALL = paths.BUILD / "iron-tools-install"
HRX_CMAKE_BUILD = paths.PROJ / "hrx" / "build" / "cmake"
HRX_INSTALL = paths.BUILD / "hrx-install"
HRX_PACKAGE_SMOKE_BUILD = paths.BUILD / "hrx-package-smoke"
FASTFLOWLM_BUILD = paths.BUILD / "FastFlowLM"
FASTFLOWLM_INSTALL = paths.BUILD / "FastFlowLM-install"
FASTFLOWLM_MODEL_CACHE = paths.BUILD / "FastFlowLM-models"
ROCM_NPU_CLI_BUILD = paths.BUILD / "rocm-npu-cli"
ROCM_NPU_CLI_INSTALL = paths.BUILD / "rocm-npu-cli-install"
TOOLS_BIN = paths.ROOT / "tools-bin"

NPROC = str(os.cpu_count() or 8)
BUILD_JOBS = os.environ.get("RNS_BUILD_JOBS") or NPROC
FRESH_BUILD_MARKER = ".rns-fresh-build.json"
HRX_CTS_TESTS = [
    "core_tests",
    "buffer_tests",
    "command_buffer_tests",
    "queue_tests",
]
HRX_CTS_DIR = (
    "runtime",
    "src",
    "iree",
    "hal",
    "drivers",
    "amdxdna",
    "cts",
)
LLVM_BUILD_TARGETS = [
    "llvm-libraries",
    "mlir-libraries",
    "llvm-headers",
    "mlir-headers",
    "llvm-tblgen",
    "mlir-tblgen",
    "mlir-linalg-ods-yaml-gen",
    "mlir-pdll",
    "MLIRPythonModules",
    "mlir-python-sources",
    "FileCheck",
    "not",
    "count",
    "llvm-config",
    "llvm-objcopy",
]


def _cc_flags(cc: str | None, cxx: str | None) -> list[str]:
    if not cc or not cxx:
        return []
    return [
        f"-DCMAKE_C_COMPILER={cc}",
        f"-DCMAKE_CXX_COMPILER={cxx}",
    ]


def _ccache_flags():
    if not shutil.which("ccache"):
        return []
    return [
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON",
    ]


def _require_fastflowlm_deps(env):
    configured = dependency_paths(paths.ROOT)
    missing = dependency_errors(
        configured,
        cargo_available=bool(environment.which("cargo", env)),
        rustc_version=environment.rustc_version(env),
    )
    if missing:
        sys.exit(
            "missing FastFlowLM prerequisite(s):\n  - "
            + "\n  - ".join(missing)
        )
    return configured


def _prepare_fresh_cmake_tree(path):
    """Clear generated CMake and FetchContent state in a known build tree."""
    allowed = {
        HRX_CMAKE_BUILD.resolve(),
        FASTFLOWLM_BUILD.resolve(),
        ROCM_NPU_CLI_BUILD.resolve(),
    }
    if path.resolve() not in allowed:
        raise RuntimeError(
            f"refusing to clear unexpected build directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def cmd_build_llvm(args):
    """Build LLVM and MLIR from the pinned amd-llvm source."""
    paths.OUT.mkdir(exist_ok=True)
    cc, cxx = environment.compilers()
    crt = (
        [
            "-DCMAKE_MSVC_RUNTIME_LIBRARY="
            "MultiThreaded$<$<CONFIG:Debug>:Debug>"
        ]
        if paths.is_windows()
        else []
    )
    configure = [
        environment.cmake_bin(),
        "-G",
        "Ninja",
        "-S",
        environment.cmake_path(paths.PROJ / "llvm-project" / "llvm"),
        "-B",
        environment.cmake_path(LLVM_BUILD),
        "-DLLVM_ENABLE_PROJECTS=mlir",
        "-DLLVM_TARGETS_TO_BUILD=host",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_ENABLE_RTTI=OFF",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        "-DMLIR_ENABLE_PYTHON_STABLE_ABI=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_INSTALL_UTILS=ON",
        *_ccache_flags(),
        "-DLLVM_INCLUDE_EXAMPLES=OFF",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        "-DLLVM_INCLUDE_TESTS=OFF",
        "-DMLIR_INCLUDE_TESTS=OFF",
        *_cc_flags(cc, cxx),
        *crt,
        f"-DPython3_EXECUTABLE={environment.cmake_path(environment.venv_py())}",
        f"-DPython_EXECUTABLE={environment.cmake_path(environment.venv_py())}",
    ]
    if (
        not paths.is_windows()
        and "clang" in (cc or "")
        and shutil.which("ld.lld")
    ):
        configure.append("-DLLVM_USE_LINKER=lld")
    environment.sh(configure, env=environment.cmake_env())
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(LLVM_BUILD),
            "-j",
            NPROC,
            "--target",
            *LLVM_BUILD_TARGETS,
        ],
        env=environment.cmake_env(),
    )
    print("build llvm: ok ->", LLVM_BUILD / "lib" / "cmake")


def build_iron_tools(python_executable, build_dir, install_dir):
    if not patches.eudsl_cache_ok():
        sys.exit(
            "eudsl not prefetched/verified — run: python dev.py fetch  "
            "(or: python dev.py setup)"
        )
    source = paths.PROJ / "iron-tools"
    python_executable = Path(os.path.abspath(python_executable))
    build_dir = Path(build_dir)
    install_dir = Path(install_dir)
    if not python_executable.is_file():
        raise RuntimeError(
            "compiler binding Python executable missing: "
            f"{python_executable}"
        )
    cc, cxx = environment.compilers()
    env = environment.cmake_env()
    env["PIP_FIND_LINKS"] = str(paths.FETCH_CACHE)
    env["PIP_NO_INDEX"] = "1"
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
            *_cc_flags(cc, cxx),
            *_ccache_flags(),
            f"-DLLVM_DIR={environment.cmake_path(LLVM_BUILD / 'lib' / 'cmake' / 'llvm')}",
            f"-DMLIR_DIR={environment.cmake_path(LLVM_BUILD / 'lib' / 'cmake' / 'mlir')}",
            f"-DCMAKE_MODULE_PATH={environment.cmake_path(source / 'cmake' / 'modulesXilinx')}",
            f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(build_dir)}",
            "-DAIE_RUNTIME_TARGETS=x86_64",
            "-DLLVM_ENABLE_ASSERTIONS=ON",
            "-DLLVM_ENABLE_RTTI=OFF",
            "-DAIE_ENABLE_BINDINGS_PYTHON=ON",
            "-DAIE_ENABLE_XRT_PYTHON_BINDINGS=OFF",
            "-DAIE_ENABLE_PYTHON_PASSES=OFF",
            "-DMLIR_ENABLE_PYTHON_STABLE_ABI=ON",
            f"-DPython3_EXECUTABLE={environment.cmake_path(python_executable)}",
            f"-DPython_EXECUTABLE={environment.cmake_path(python_executable)}",
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(build_dir),
            "-j",
            NPROC,
        ],
        env=env,
    )
    if install_dir.parent.resolve() != paths.BUILD.resolve():
        raise RuntimeError(
            f"compiler binding install must be under {paths.BUILD}: {install_dir}"
        )
    if install_dir.exists():
        shutil.rmtree(install_dir)
    install_dir.mkdir(parents=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--install",
            environment.cmake_path(build_dir),
            "--prefix",
            environment.cmake_path(install_dir),
            "--config",
            "Release",
        ],
        env=env,
    )
    print("build iron-tools: ok ->", build_dir / "bin")


def cmd_build_iron_tools(args):
    build_iron_tools(
        environment.venv_py(),
        IRON_TOOLS_BUILD,
        IRON_TOOLS_INSTALL,
    )


def cmd_build_hrx_xclbinutil(args):
    source = (
        paths.PROJ
        / "iron-tools"
        / "third_party"
        / "hrx-xclbinutil"
    )
    if not (source / "CMakeLists.txt").exists():
        sys.exit(
            "missing hrx-xclbinutil (vendored in mlir-aie); run: "
            "python dev.py submodules iron-tools (recursive init pulls "
            "projects/iron-tools/third_party/hrx-xclbinutil)"
        )
    build_dir = source / "bld"
    env = environment.cmake_env()
    environment.sh(
        [
            environment.cmake_bin(),
            "-G",
            "Ninja",
            "-B",
            environment.cmake_path(build_dir),
            "-S",
            environment.cmake_path(source),
            "-DCMAKE_BUILD_TYPE=Release",
            *_ccache_flags(),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(build_dir),
            "--target",
            "hrx-xclbinutil",
            "pdi_transform_test",
            "-j",
            NPROC,
        ],
        env=env,
    )
    TOOLS_BIN.mkdir(exist_ok=True)
    executable = paths.executable_name("hrx-xclbinutil")
    built = build_dir / "tools" / executable
    link = TOOLS_BIN / paths.executable_name("xclbinutil")
    if link.exists() or link.is_symlink():
        link.unlink()
    if paths.is_windows():
        shutil.copy2(built, link)
    else:
        link.symlink_to(built)
    print("build hrx-xclbinutil: ok ->", link)


def cmd_build_hrx(args):
    source = paths.PROJ / "hrx"
    if not (source / "CMakeLists.txt").exists():
        sys.exit(
            "missing projects/hrx "
            "(run: git submodule update --init projects/hrx)"
        )
    patches.apply_patches({"hrx"})
    fresh = getattr(args, "fresh", False)
    if fresh:
        patches.validate_declared_patch_state("hrx", source)
        _prepare_fresh_cmake_tree(HRX_CMAKE_BUILD)
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
            environment.cmake_path(HRX_CMAKE_BUILD),
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            "-DIREE_HAL_DRIVER_AMDGPU=OFF",
            "-DIREE_HAL_DRIVER_AMDXDNA=ON",
            "-DIREE_BUILD_TESTS=OFF",
            "-DIREE_BUILD_BENCHMARKS=OFF",
            "-DLIBHRX_BUILD_CTS=OFF",
            "-DHRX_INSTALL_TESTS=OFF",
            "-DLIBHRX_BUILD_PASSTHROUGH=OFF",
            "-DLIBHRX_BUILD_HIP_BINDING=OFF",
            "-DLIBHRX_BUILD_TOOLS=OFF",
            "-DLOOM_BUILD=OFF",
            "-DIREE_HAL_DRIVER_LOCAL_SYNC=OFF",
            "-DIREE_HAL_DRIVER_NULL=OFF",
            *environment.cmake_source_override(
                "FETCHCONTENT_SOURCE_DIR_FLATCC"
            ),
            *_cc_flags(cc, cxx),
            *_ccache_flags(),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(HRX_CMAKE_BUILD),
            *(["--clean-first"] if fresh else []),
            "--target",
            "iree_hal_drivers_amdxdna_registration_registration",
            "hrx",
            "flatcc_runtime",
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    shutil.rmtree(HRX_INSTALL, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--install",
            environment.cmake_path(HRX_CMAKE_BUILD),
            "--prefix",
            environment.cmake_path(HRX_INSTALL),
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
                HRX_INSTALL, exclude_names=(FRESH_BUILD_MARKER,)
            ),
        }
        (HRX_INSTALL / FRESH_BUILD_MARKER).write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        "build hrx (minimal amdxdna HAL + installed package): ok ->",
        HRX_INSTALL,
    )


def _smoke_installed_hrx(source, cc, cxx, env):
    shutil.rmtree(HRX_PACKAGE_SMOKE_BUILD, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(source / "libhrx" / "cts" / "package_smoke"),
            "-B",
            environment.cmake_path(HRX_PACKAGE_SMOKE_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={environment.cmake_path(HRX_INSTALL)}",
            *_cc_flags(cc, cxx),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(HRX_PACKAGE_SMOKE_BUILD),
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    smoke_env = dict(env)
    loader = "PATH" if paths.is_windows() else "LD_LIBRARY_PATH"
    lib_dir = HRX_INSTALL / ("bin" if paths.is_windows() else "lib")
    smoke_env[loader] = (
        str(lib_dir) + os.pathsep + smoke_env.get(loader, "")
    )
    for name in ("hrx_package_smoke", "hrx_package_smoke_cxx"):
        environment.sh(
            [
                HRX_PACKAGE_SMOKE_BUILD
                / paths.executable_name(name)
            ],
            env=smoke_env,
        )


def cmd_build_hrx_cts(args):
    if not (HRX_CMAKE_BUILD / "CMakeCache.txt").exists():
        sys.exit("hrx not configured — run: python dev.py build hrx")
    env = environment.cmake_env()
    environment.sh(
        [
            environment.cmake_bin(),
            "-S",
            environment.cmake_path(paths.PROJ / "hrx"),
            "-B",
            environment.cmake_path(HRX_CMAKE_BUILD),
            "-DIREE_BUILD_TESTS=ON",
            "-DIREE_BUILD_BENCHMARKS=ON",
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(HRX_CMAKE_BUILD),
            *[
                f"--target=iree_hal_drivers_amdxdna_cts_{test}"
                for test in HRX_CTS_TESTS
            ],
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    print(f"build hrx-cts: ok ({', '.join(HRX_CTS_TESTS)})")


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
        patches.validate_declared_patch_state(
            "FastFlowLM", paths.PROJ / "FastFlowLM"
        )
        _prepare_fresh_cmake_tree(FASTFLOWLM_BUILD)
        shutil.rmtree(source / "build", ignore_errors=True)
    hrx_config = (
        HRX_INSTALL / "lib" / "cmake" / "hrx" / "hrx-config.cmake"
    )
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
            environment.cmake_path(FASTFLOWLM_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_LIBDIR=lib",
            "-DFLM_VERSION=0.9.45",
            "-DNPU_VERSION=32.0.203.304",
            f"-DCMAKE_PREFIX_PATH={environment.cmake_path(HRX_INSTALL)}",
            f"-DCMAKE_TOOLCHAIN_FILE={environment.cmake_path(Path(dependencies['toolchain']))}",
            f"-DVCPKG_TARGET_TRIPLET={dependencies['triplet']}",
            f"-DVCPKG_INSTALLED_DIR={environment.cmake_path(Path(dependencies['install_root']))}",
            f"-DVCPKG_MANIFEST_DIR={environment.cmake_path(paths.ROOT / '.github' / 'vcpkg' / 'fastflowlm')}",
            "-DVCPKG_MANIFEST_INSTALL=OFF",
            f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(FASTFLOWLM_INSTALL)}",
            f"-DCMAKE_XCLBIN_PREFIX={environment.cmake_path(source)}",
            *_cc_flags(cc, cxx),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(FASTFLOWLM_BUILD),
            *(["--clean-first"] if fresh else []),
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    shutil.rmtree(FASTFLOWLM_INSTALL, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--install",
            environment.cmake_path(FASTFLOWLM_BUILD),
        ],
        env=env,
    )
    installed = FASTFLOWLM_INSTALL / "bin" / paths.executable_name("flm")
    if not installed.exists():
        sys.exit(f"FastFlowLM install did not produce {installed}")
    if fresh:
        marker = {
            "source": git_source_state(paths.PROJ / "FastFlowLM"),
            "hrx": json.loads(
                (HRX_INSTALL / FRESH_BUILD_MARKER).read_text(
                    encoding="utf-8"
                )
            ),
            "install": directory_state(
                FASTFLOWLM_INSTALL,
                exclude_names=(FRESH_BUILD_MARKER,),
            ),
        }
        (FASTFLOWLM_INSTALL / FRESH_BUILD_MARKER).write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("build fastflowlm: ok ->", installed)


def cmd_build_rocm_npu_cli(args):
    """Builds the standalone C++ runtime CLI without linking HRX."""
    env = environment.cmake_env()
    fresh = getattr(args, "fresh", False)
    if fresh:
        _prepare_fresh_cmake_tree(ROCM_NPU_CLI_BUILD)
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
            environment.cmake_path(ROCM_NPU_CLI_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DROCM_NPU_VERSION={version}",
            f"-DROCM_NPU_SOURCE_REVISION={source_revision}",
            f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(ROCM_NPU_CLI_INSTALL)}",
            *_cc_flags(cc, cxx),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(ROCM_NPU_CLI_BUILD),
            *(["--clean-first"] if fresh else []),
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    shutil.rmtree(ROCM_NPU_CLI_INSTALL, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--install",
            environment.cmake_path(ROCM_NPU_CLI_BUILD),
        ],
        env=env,
    )
    installed = (
        ROCM_NPU_CLI_INSTALL / "bin" / paths.executable_name("rocm-npu")
    )
    if not installed.exists():
        sys.exit(f"rocm-npu CLI install did not produce {installed}")
    if fresh:
        marker = {
            "source": git_source_state(paths.ROOT),
            "hrx_headers": git_source_state(paths.PROJ / "hrx"),
            "version": version,
            "compilers": {"c": str(cc), "cxx": str(cxx)},
            "install": directory_state(
                ROCM_NPU_CLI_INSTALL,
                exclude_names=(FRESH_BUILD_MARKER,),
            ),
        }
        (ROCM_NPU_CLI_INSTALL / FRESH_BUILD_MARKER).write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("build rocm-npu-cli: ok ->", installed)


def cmd_build_all(args):
    cmd_build_rocm_npu_cli(args)
    cmd_build_hrx_xclbinutil(args)
    cmd_build_llvm(args)
    cmd_build_iron_tools(args)
    cmd_build_hrx(args)
