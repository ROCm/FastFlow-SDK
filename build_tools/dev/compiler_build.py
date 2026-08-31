# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compiler-stack build commands for LLVM, IRON tools, and xclbinutil."""

import os
import shutil
import sys
from pathlib import Path

from build_tools import environment, llvm_aie, paths
from build_tools.artifact import reset_directory
from build_tools.dev import patches

NPROC = str(os.cpu_count() or 8)
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


def _reset_owned_directory(path: Path) -> None:
    allowed_roots = [paths.BUILD]
    runner_temp = os.environ.get("RUNNER_TEMP")
    if runner_temp:
        allowed_roots.append(Path(runner_temp))
    last_error = None
    for root in allowed_roots:
        try:
            reset_directory(path, root)
            return
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(
        f"unexpected build directory outside owned roots: {path}"
    ) from last_error


def _llvm_aie_compilers(
    windows_compiler: str = "cl",
) -> tuple[str | None, str | None]:
    if not paths.is_windows():
        return environment.compilers()
    compiler = llvm_aie.normalize_windows_compiler(windows_compiler)
    if compiler == "cl":
        return "cl", "cl"
    if compiler == "clang-cl":
        return "clang-cl", "clang-cl"
    if compiler == "env":
        return environment.compilers()
    raise AssertionError(f"unhandled llvm-aie Windows compiler: {compiler}")


def cmd_build_llvm(args):
    """Build LLVM and MLIR from the pinned amd-llvm source."""
    paths.OUT.mkdir(exist_ok=True)
    cc, cxx = environment.compilers()
    crt = (
        ["-DCMAKE_MSVC_RUNTIME_LIBRARY=" "MultiThreaded$<$<CONFIG:Debug>:Debug>"]
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
        environment.cmake_path(paths.LLVM_BUILD),
        "-DLLVM_ENABLE_PROJECTS=mlir",
        "-DLLVM_TARGETS_TO_BUILD=host",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_ENABLE_RTTI=OFF",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        "-DMLIR_ENABLE_PYTHON_STABLE_ABI=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_INSTALL_UTILS=ON",
        *environment.cmake_ccache_flags(),
        "-DLLVM_INCLUDE_EXAMPLES=OFF",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        "-DLLVM_INCLUDE_TESTS=OFF",
        "-DMLIR_INCLUDE_TESTS=OFF",
        *environment.cmake_compiler_flags(cc, cxx),
        *crt,
        f"-DPython3_EXECUTABLE={environment.cmake_path(environment.venv_py())}",
        f"-DPython_EXECUTABLE={environment.cmake_path(environment.venv_py())}",
    ]
    if not paths.is_windows() and "clang" in (cc or "") and shutil.which("ld.lld"):
        configure.append("-DLLVM_USE_LINKER=lld")
    environment.sh(configure, env=environment.cmake_env())
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(paths.LLVM_BUILD),
            "-j",
            NPROC,
            "--target",
            *LLVM_BUILD_TARGETS,
        ],
        env=environment.cmake_env(),
    )
    print("build llvm: ok ->", paths.LLVM_BUILD / "lib" / "cmake")


def cmd_build_llvm_aie(args):
    """Build Peano LLVM-AIE from the pinned source submodule."""
    source = paths.PROJ / "llvm-aie"
    llvm_source = source / "llvm"
    cache = source / "clang" / "cmake" / "caches" / "Peano-AIE.cmake"
    windows_compiler = getattr(args, "llvm_aie_windows_compiler", "cl")
    if paths.is_windows():
        windows_compiler = llvm_aie.normalize_windows_compiler(windows_compiler)
    if not cache.is_file() or not (llvm_source / "CMakeLists.txt").is_file():
        sys.exit(
            "missing llvm-aie source checkout; run: "
            "python dev.py submodules llvm-aie"
        )
    if getattr(args, "fresh", False):
        _reset_owned_directory(paths.LLVM_AIE_BUILD)
        _reset_owned_directory(paths.LLVM_AIE_INSTALL)
    else:
        try:
            verified = llvm_aie.verify_install(
                paths.LLVM_AIE_INSTALL,
                windows_compiler=windows_compiler,
            )
            print("build llvm-aie: cached ->", verified)
            return
        except RuntimeError:
            if paths.LLVM_AIE_INSTALL.exists():
                _reset_owned_directory(paths.LLVM_AIE_INSTALL)

    paths.LLVM_AIE_BUILD.mkdir(parents=True, exist_ok=True)
    paths.LLVM_AIE_INSTALL.parent.mkdir(parents=True, exist_ok=True)
    cc, cxx = _llvm_aie_compilers(windows_compiler)
    configure = [
        environment.cmake_bin(),
        "-G",
        "Ninja",
        "-S",
        environment.cmake_path(llvm_source),
        "-B",
        environment.cmake_path(paths.LLVM_AIE_BUILD),
        "-C",
        environment.cmake_path(cache),
        "-DLLVM_BUILD_BENCHMARKS=OFF",
        "-DLLVM_BUILD_EXAMPLES=OFF",
        "-DLLVM_BUILD_TESTS=OFF",
        "-DLLVM_BUILD_TOOLS=ON",
        "-DLLVM_BUILD_UTILS=ON",
        "-DLLVM_CCACHE_BUILD=ON",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_ENABLE_RTTI=ON",
        "-DLLVM_ENABLE_ZSTD=OFF",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        "-DLLVM_INCLUDE_EXAMPLES=OFF",
        "-DLLVM_INCLUDE_TESTS=OFF",
        "-DLLVM_INCLUDE_TOOLS=ON",
        "-DLLVM_INCLUDE_UTILS=ON",
        "-DLLVM_INSTALL_UTILS=ON",
        "-DLLVM_VERSION_SUFFIX=",
        "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
        f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(paths.LLVM_AIE_INSTALL)}",
        f"-DPython3_EXECUTABLE={environment.cmake_path(environment.venv_py())}",
        f"-DPython_EXECUTABLE={environment.cmake_path(environment.venv_py())}",
        "-DCMAKE_BUILD_TYPE=Release",
        *environment.cmake_ccache_flags(),
        *environment.cmake_compiler_flags(cc, cxx),
    ]
    if paths.is_windows():
        configure.extend(
            [
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
                "-DCMAKE_C_FLAGS=/MT",
                "-DCMAKE_CXX_FLAGS=/MT",
                "-DLLVM_USE_CRT_MINSIZEREL=MT",
                "-DLLVM_USE_CRT_RELEASE=MT",
                "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW",
            ]
        )
    else:
        configure.extend(
            [
                "-DLLVM_BUILD_LLVM_DYLIB=ON",
                "-DLLVM_LINK_LLVM_DYLIB=ON",
            ]
        )
        if "clang" in (cc or "") and shutil.which("ld.lld"):
            configure.append("-DLLVM_USE_LINKER=lld")
    environment.sh(configure, env=environment.cmake_env())
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(paths.LLVM_AIE_BUILD),
            "-j",
            NPROC,
            "--target",
            "install-distribution",
        ],
        env=environment.cmake_env(),
    )
    llvm_aie.write_build_stamp(
        paths.LLVM_AIE_INSTALL,
        windows_compiler=windows_compiler,
    )
    verified = llvm_aie.verify_install(
        paths.LLVM_AIE_INSTALL,
        windows_compiler=windows_compiler,
    )
    print("build llvm-aie: ok ->", verified)


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
            "compiler binding Python executable missing: " f"{python_executable}"
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
            *environment.cmake_compiler_flags(cc, cxx),
            *environment.cmake_ccache_flags(),
            f"-DLLVM_DIR={environment.cmake_path(paths.LLVM_BUILD / 'lib' / 'cmake' / 'llvm')}",
            f"-DMLIR_DIR={environment.cmake_path(paths.LLVM_BUILD / 'lib' / 'cmake' / 'mlir')}",
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
        paths.IRON_TOOLS_BUILD,
        paths.IRON_TOOLS_INSTALL,
    )


def cmd_build_hrx_xclbinutil(args):
    source = paths.PROJ / "iron-tools" / "third_party" / "hrx-xclbinutil"
    if not (source / "CMakeLists.txt").exists():
        sys.exit(
            "missing hrx-xclbinutil (vendored in-tree under mlir-aie); run: "
            "python dev.py submodules iron-tools"
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
            *environment.cmake_ccache_flags(),
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
    paths.TOOLS_BIN.mkdir(exist_ok=True)
    executable = paths.executable_name("hrx-xclbinutil")
    built = build_dir / "tools" / executable
    link = paths.TOOLS_BIN / paths.executable_name("xclbinutil")
    if link.exists() or link.is_symlink():
        link.unlink()
    if paths.is_windows():
        shutil.copy2(built, link)
    else:
        link.symlink_to(built)
    print("build hrx-xclbinutil: ok ->", link)
