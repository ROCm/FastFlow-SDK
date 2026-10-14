# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compiler-stack build commands for LLVM, AIEBU, IRON tools, and xclbinutil."""

import os
import shutil
import sys
from pathlib import Path

from build_tools import environment, paths
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
    "llvm-nm",
    "llvm-objcopy",
]


def cmd_build_llvm(args):
    """Build LLVM and MLIR from the pinned amd-llvm source."""
    paths.OUT.mkdir(exist_ok=True)
    cc, cxx = environment.compilers()
    crt = (
        ["-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>"]
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


def cmd_build_aiebu(args):
    """Builds the pinned assembler used by aiecc sequence compilation."""
    source = paths.PROJ / "aiebu"
    if not (source / "CMakeLists.txt").is_file():
        sys.exit("missing AIEBU checkout; run: python dev.py submodules aiebu")
    cc, cxx = environment.compilers()
    env = environment.cmake_env()
    environment.sh(
        [
            environment.cmake_bin(),
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(source),
            "-B",
            environment.cmake_path(paths.AIEBU_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(paths.AIEBU_INSTALL)}",
            "-DAIEBU_BUILD_TESTS=OFF",
            *environment.cmake_compiler_flags(cc, cxx),
            *environment.cmake_ccache_flags(),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(paths.AIEBU_BUILD),
            "-j",
            NPROC,
            "--target",
            "aiebu-asm",
        ],
        env=env,
    )
    if paths.AIEBU_INSTALL.exists():
        shutil.rmtree(paths.AIEBU_INSTALL)
    install_bin = paths.AIEBU_INSTALL / "bin"
    install_bin.mkdir(parents=True)
    executable = "aiebu-asm.exe" if paths.is_windows() else "aiebu-asm"
    built_assembler = paths.AIEBU_BUILD / "src" / "cpp" / "utils" / "asm" / executable
    if not built_assembler.is_file():
        raise RuntimeError(
            f"AIEBU assembler build output is missing: {built_assembler}"
        )
    # Upstream's install target includes sibling tools that are intentionally
    # not built here. Stage the one requested target instead of making the
    # compiler product depend on unrelated AIEBU executables.
    shutil.copy2(built_assembler, install_bin / executable)
    print("build aiebu: ok ->", paths.AIEBU_INSTALL / "bin")


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
            f"compiler binding Python executable missing: {python_executable}"
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
