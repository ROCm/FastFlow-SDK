#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""rocm-npu-staging developer command router.

Builds the NPU stack together (everything except llvm-aie, which comes
from the Peano wheel). Mirrors the command-router shape of hrx's dev.py.

  python dev.py doctor [PROFILE]      # quick validation: compiler-stack (default), fastflowlm, or all
                                     #   --deep: replay every patch series in a temporary index
  python dev.py setup [--profile P]   # venv + python deps (incl. pip cmake/ninja) + fetch
                                     #   --iron-ops-tests: + what iron-ops' own tests need
                                     #     (torch, pytest); the NPU job only
  python dev.py fetch                 # apply temp patches + prefetch eudsl (verified) — offline-ready
  python dev.py build llvm            # LLVM+MLIR from projects/llvm-project (amd-llvm pin)
  python dev.py build iron-tools      # aie-opt/aie-translate against build/llvm
  python dev.py build hrx-xclbinutil  # the xclbin packager
  python dev.py build hrx             # the amdxdna HAL (IREE runtime)
  python dev.py build hrx-cts         # amdxdna HAL conformance binaries (need an NPU to run)
  python dev.py build fastflowlm      # HRX amdxdna + FastFlowLM app only
  python dev.py package-fastflowlm    # relocatable .zip/.tar.zst + manifest
  python dev.py package-fastflowlm-wheels # rocm-npu-runtime + rocm-npu-fastflowlm wheels
  python dev.py verify-fastflowlm-wheels  # pip install wheels into a clean venv and smoke
  python dev.py unpack-fastflowlm     # checksum, extract, and verify the bundle
  python dev.py package-compiler-product # relocatable rocm-npu-compiler product artifact
  python dev.py verify-compiler-product  # verify compiler product and relocation smoke
  python dev.py package-compiler-wheels  # compiler meta/core/CPython abi3 wheels
  python dev.py verify-compiler-wheels   # clean pip install, native closure, and e2e
  python dev.py build all             # everything above, in dependency order
  python dev.py test [DESIGN] [-- ...] # e2e build test (no NPU run); Linux + Windows
  python dev.py test-iron-ops [OP]    # build the iron-ops operators' xclbins (no NPU run)
                                     #   Linux + Windows; suite: tests/iron-ops/suite.py
  python dev.py submodules NAME...    # init (and optionally reset/clean) submodules
  python dev.py package-npu-prereqs   # stage + tar HRX and its CTS binaries (no NPU)
  python dev.py package-npu-test-package # prebuild XADX kernels + numeric cases (no NPU)
  python dev.py unpack-npu-prereqs    # extract that archive (codec-agnostic; no NPU)
  python dev.py verify-npu-prereqs    # assert that archive is self-contained + imports (no NPU)
  python dev.py npu-probe             # print NPU driver/fw versions + check libhrx open
  python dev.py test-hrx-cts [SUITE]  # amdxdna HAL conformance suites (needs an NPU)
  python dev.py test-npu-package      # ON-DEVICE execute-only XADX + numeric suite
  python dev.py test-npu [DESIGN]     # ON-DEVICE compile+run developer path (needs compiler)
                                     #   --device npu2|npu1   --chain (opt-in chained testbench)
  python dev.py test-fastflowlm       # pull qwen3:0.6b + run a 1k HRX benchmark
  python dev.py test-npu-iron-ops [OP] # ON-DEVICE e2e: an iron-ops operator, numerics checked (needs an NPU)
  python dev.py deps                  # print the dependency overview

Build-only: running on the NPU needs the amdxdna driver + an NPU (not here).
"""
import argparse, contextlib, ctypes, errno, json, os, platform, re, shutil, stat, subprocess, sys, sysconfig, tarfile
from pathlib import Path

from build_tools.fastflowlm_deps import (
    dependency_errors as fastflowlm_dependency_errors,
    dependency_paths as fastflowlm_dependency_paths,
)
from build_tools.fastflowlm_artifact import (
    create_archive as create_fastflowlm_archive,
    directory_state,
    extract_archive as extract_fastflowlm_archive,
    git_source_state,
    pe_imports,
    sha256_file,
    stage_runtime_bundle,
    validate_directory_target,
    validate_source_policy,
    verify_fastflowlm_fresh_installs,
    verify_bundle as verify_fastflowlm_bundle,
)
from build_tools import npu_test_package
from build_tools.rocm_npu_compiler_wheels import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    _is_loadable_elf,
    _linux_system_runtime,
    _windows_system_runtime,
    build_compiler_wheel_family,
    compiler_core_shared_library_names,
    compiler_product_report,
    relocate_linux_compiler_rpaths,
    verify_compiler_binding_stable_abi,
    verify_compiler_wheel_path_references,
    verify_linux_compiler_wheels,
    verify_windows_compiler_wheels,
)
from build_tools.rocm_npu_linux_wheels import (
    FASTFLOWLM_LINUX_WHEEL_TAG,
    FASTFLOWLM_DISTRIBUTION,
    LINUX_WHEEL_TAG,
    RUNTIME_DISTRIBUTION,
    auditwheel_policy,
    build_runtime_fastflowlm_wheels,
    find_rocm_npu_wheels,
    validate_auditwheel_policy,
    verify_wheels_do_not_have_text_path_references,
)
from build_tools.rocm_npu_wheel_common import (
    COMPILER_PACKAGE,
    PACKAGE_VERSION_ENV,
    default_package_version,
    verify_wheel_checksums,
)
from build_tools.windows_npu import (
    probe_runtime as probe_windows_npu_runtime,
)
from build_tools.dev import doctor as dev_doctor
from build_tools.dev import environment as dev_environment
from build_tools.dev import patches as dev_patches
from build_tools.dev import setup as dev_setup
from build_tools.dev.environment import (
    _cmake_source_override,
    _python_install,
    cmake_bin,
    cmake_env,
    cmake_path,
    compilers as _compilers,
    have_header as _have_header,
    rustc_version as _rustc_version,
    sh,
    venv_py,
    which as _which,
)
from build_tools.dev.patches import (
    apply_patches,
    eudsl_cache_ok as _eudsl_cache_ok,
    validate_declared_patch_state as _validate_declared_patch_state,
)
from build_tools.dev.profiles import (
    doctor_profile_names,
    setup_profile_names,
)

ROOT = Path(__file__).resolve().parent
PROJ = ROOT / "projects"
BUILD = ROOT / "build"
OUT = ROOT / "out"
VENV = ROOT / ".venv"
# LLVM's build tree is the deepest one we create. On Windows the SDK's rc.exe is not
# long-path aware, and CMake mangles the ABSOLUTE path of out-of-tree .rc sources into
# the object dir (…/foo.dir/C_/home/runner/…/llvm/resources/…), which pushes some
# targets past MAX_PATH (260) -> "fatal error RC1109: error creating …". Allow the LLVM
# build tree to be relocated somewhere short; the Windows CI points this at the
# job-scoped $RUNNER_TEMP. Unset -> build/llvm.
LLVM_BUILD = Path(os.environ.get("RNS_LLVM_BUILD_DIR") or (BUILD / "llvm"))
HRX_CMAKE_BUILD = PROJ / "hrx" / "build" / "cmake"
HRX_INSTALL = BUILD / "hrx-install"
HRX_PACKAGE_SMOKE_BUILD = BUILD / "hrx-package-smoke"
IRON_TOOLS_INSTALL = BUILD / "iron-tools-install"
FASTFLOWLM_BUILD = BUILD / "FastFlowLM"
FASTFLOWLM_INSTALL = BUILD / "FastFlowLM-install"
FASTFLOWLM_MODEL_CACHE = BUILD / "FastFlowLM-models"
FASTFLOWLM_BUNDLE_STAGE = BUILD / "FastFlowLM-bundle"
FASTFLOWLM_ARTIFACTS = BUILD / "artifacts"
FASTFLOWLM_EXTRACTED = BUILD / "FastFlowLM-extracted"
FASTFLOWLM_WHEEL_STAGE = BUILD / "rocm-npu-wheel-stage"
FASTFLOWLM_WHEEL_VENV = BUILD / "rocm-npu-wheel-venv"
COMPILER_PRODUCT_STAGE = BUILD / "rocm-npu-compiler-product"
COMPILER_PRODUCT_EXTRACTED = BUILD / "rocm-npu-compiler-extracted"
COMPILER_PRODUCT_RELOCATED = BUILD / "rocm-npu-compiler-relocated"
COMPILER_WHEEL_STAGE = BUILD / "rocm-npu-compiler-wheel-stage"
COMPILER_WHEEL_VENV = BUILD / "rocm-npu-compiler-wheel-venv"
NPU_TEST_PACKAGE_STAGE = BUILD / "npu-test-package"
NPU_TEST_PACKAGE_EXTRACTED = BUILD / "npu-test-package-extracted"
ARCHIVE_FIXED_MTIME = 946684800  # 2000-01-01T00:00:00Z
COMPILER_PRODUCT_ZSTD_LEVEL = 6
COMPILER_PRODUCT_ZSTD_WINDOW_LOG = 27  # 128 MiB: finds duplicate large LLVM binaries.
FRESH_BUILD_MARKER = ".rns-fresh-build.json"
FETCH_CACHE = ROOT / ".cache" / "find-links"   # prefetched, hash-verified sdists (gitignored)
NPROC = str(os.cpu_count() or 8)
BUILD_JOBS = os.environ.get("RNS_BUILD_JOBS") or NPROC
IS_WINDOWS = os.name == "nt"                    # venv layout, .exe suffix, and MSVC differ on Windows
_VBIN = "Scripts" if IS_WINDOWS else "bin"      # venv exe dir: POSIX bin/, Windows Scripts/
_EXE = ".exe" if IS_WINDOWS else ""

# Peano (AIE codegen) — pulled from the wheel, not built from llvm-aie source.
PEANO_WHEEL = "llvm-aie==21.0.0.2026061501+acc2a72c"
PEANO_INDEX = "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"

# amdxdna HAL conformance suites (gtest binaries). They open a real device.
HRX_CTS_TESTS = ["core_tests", "buffer_tests", "command_buffer_tests", "queue_tests"]
HRX_CTS_DIR = ("runtime", "src", "iree", "hal", "drivers", "amdxdna", "cts")
# The amdxdna NPU is a DRM accel node: /dev/accel/accel0, not /dev/accel0.
ACCEL_DIR = Path("/dev/accel")
SYS_ACCEL_DIR = Path("/sys/class/accel")
AMDXDNA_MODULE_DIR = Path("/sys/module/amdxdna")
KNOWN_AMDXDNA_SVA_REGRESSION_SRCVERSIONS = {
    "2DBDA75956CAFA9D029EA89",
}
# NPU targets for `test-npu`. The @iron.jit script accepts these device names
# directly; `devgen` is the XAie_TxnHeader.DevGen the design must then carry
# (aie-rt xaiegbl_defs.h). libhrx cannot report the generation -- its ARCHITECTURE
# property is a gfxNNNN string -- so the target is declared, not detected.
NPU_TARGETS = {
    "npu2": {"device": "npu2", "devgen": 4, "arch": "AIE2P (Strix)"},
    "npu1": {"device": "npu",  "devgen": 3, "arch": "AIE2IPU (Phoenix)"},
}



def cmd_fetch(args, submodule_names=None):
    return dev_patches.cmd_fetch(args, submodule_names)


def cmd_doctor(args):
    return dev_doctor.cmd_doctor(args)


def cmd_setup(args):
    return dev_setup.cmd_setup(args)


def _cc_flags(cc: str | None, cxx: str | None) -> list[str]:
    """CMake compiler flags. Empty when _compilers() found none — on Windows with CC/CXX
    unset, so CMake picks from the VS Developer environment rather than us hardcoding a
    compiler it may not have. On POSIX these pin gcc/clang exactly as before."""
    return [f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}"] if cc and cxx else []


def _require_fastflowlm_deps(env):
    paths = fastflowlm_dependency_paths(ROOT)
    missing = fastflowlm_dependency_errors(
        paths, cargo_available=bool(_which("cargo", env)),
        rustc_version=_rustc_version(env))
    if missing:
        sys.exit("missing FastFlowLM prerequisite(s):\n  - " + "\n  - ".join(missing))
    return paths


def _ccache_flags():
    """Route compiles through ccache when it is on PATH; nothing when it is not.

    COMPILER_LAUNCHER rather than LLVM_CCACHE_BUILD, because that knob only exists in LLVM's
    build and left iron-tools/hrx/hrx-xclbinutil uncached — ~3,800 objects in all, of which
    LLVM is 3,265. Staying optional is deliberate: a plain dev box has no ccache, and
    LLVM_CCACHE_BUILD=ON is a FATAL_ERROR when it cannot find one.

    Caching only pays off if the cache OUTLIVES THE JOB. The CI runners are ephemeral, so the
    workflows persist CCACHE_DIR through actions/cache; without that this is a no-op (which is
    what it silently was: CCACHE_DIR pointed into the workspace and nothing preserved it)."""
    if not shutil.which("ccache"):
        return []
    return [
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
        # PCH off whenever ccache is in play. LLVM's HandleLLVMOptions warns that ccache +
        # precompiled headers on a non-Clang compiler "may lead to false positives" -- i.e. a
        # cache hit that yields a WRONG object. LLVM keeps PCH off by default anyway, so this
        # costs nothing, and a silently miscompiled build is not worth a marginal speedup.
        "-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON",
    ]


# What this stack actually consumes out of the LLVM build, and nothing else.
#
# `cmake --build` with no --target builds LLVM's default `all`: 133 binaries, 2.8 GB of
# bin/ — mlir-opt (309 MB), mlir-rewrite, mlir-reduce, mlir-transform-opt, mlir-lsp-server,
# mlir-query, llc, opt, lli, ... — and NOT ONE of them is ever run here. iron-tools links
# the MLIR *libraries* and builds its own aie-opt/aie-translate. Ninja's own graph puts
# `all` at 114 link steps against 9 for the list below, and the 105 it drops are the huge
# ones. Measured: this list yields 11 binaries / 21 MB of bin/ instead of 133 / 2.8 GB.
#
#   {llvm,mlir}-libraries  the static libs iron-tools links. find_package(MLIR) resolves
#                          IMPORTED targets to lib/*.a, so these must exist ON DISK — which
#                          is why there is no smaller subset: we cannot know which .a the
#                          downstream link will reach for.
#   {llvm,mlir}-headers    the tablegen'd *.inc under the build tree's include dirs
#   *tblgen, mlir-pdll,    build-time generators that MLIR's cmake functions invoke in the
#   mlir-linalg-ods-*      downstream (iron-tools) build
#   MLIRPythonModules,     AIE_ENABLE_BINDINGS_PYTHON builds on MLIR's python packages
#   mlir-python-sources
#   FileCheck, not, count  iron-tools' lit suites name these as DEPENDS
#   llvm-config            queried by downstream cmake
LLVM_BUILD_TARGETS = [
    "llvm-libraries", "mlir-libraries", "llvm-headers", "mlir-headers",
    "llvm-tblgen", "mlir-tblgen", "mlir-linalg-ods-yaml-gen", "mlir-pdll",
    "MLIRPythonModules", "mlir-python-sources",
    "FileCheck", "not", "count", "llvm-config",
]


def cmd_build_llvm(args):
    """LLVM+MLIR from the amd-llvm pin (mlir-aie's flags)."""
    OUT.mkdir(exist_ok=True)
    cc, cxx = _compilers()
    # Static MSVC runtime, matching what iron-tools expects: it does
    # `set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded...")` unconditionally, and lld-link
    # rejects a mixed link ("/failifmismatch: ... 'RuntimeLibrary'"). This is the same CRT
    # upstream builds its own LLVM with (utils/mlir_wheels/setup.py), so we track a
    # configuration mlir-aie actually tests. LLVM_USE_CRT_RELEASE is gone in LLVM 23;
    # CMAKE_MSVC_RUNTIME_LIBRARY is the only knob (honoured: CMP0091 is NEW here).
    #
    # The `$<$<CONFIG:Debug>:Debug>` is CMake's idiom for "…Debug only in a Debug config":
    # it selects the debug static CRT (/MTd) there and the release one (/MT) everywhere
    # else. We always configure Release so it evaluates to plain "MultiThreaded", but it is
    # the same expression iron-tools uses for itself, so the two agree in *any* config.
    crt = ["-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>"] if IS_WINDOWS else []
    cfg = [
        cmake_bin(), "-G", "Ninja",
        "-S", cmake_path(PROJ / "llvm-project" / "llvm"), "-B", cmake_path(LLVM_BUILD),
        "-DLLVM_ENABLE_PROJECTS=mlir", "-DLLVM_TARGETS_TO_BUILD=host",
        "-DLLVM_ENABLE_ASSERTIONS=ON", "-DLLVM_ENABLE_RTTI=OFF",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        "-DMLIR_ENABLE_PYTHON_STABLE_ABI=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_INSTALL_UTILS=ON", *_ccache_flags(),
        # Nothing in this stack consumes the LLVM/MLIR examples or benchmarks, and the
        # default target builds them. Dropping them cuts build time, and it removes
        # mlir/examples (gated on LLVM_INCLUDE_EXAMPLES) whose long target names are what
        # first pushed the Windows rc.exe output path past MAX_PATH. Utils (FileCheck,
        # lit, ...) are separate and still built via LLVM_INSTALL_UTILS above.
        "-DLLVM_INCLUDE_EXAMPLES=OFF", "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        # We never run LLVM's or MLIR's own test suites, and iron-tools needs no test dialect
        # (checked: no MLIRTest*/TestDialect reference in its cmake). FileCheck/not/count are
        # gated on LLVM_INCLUDE_UTILS, NOT on this, so iron-tools' own lit suites keep the
        # DEPENDS they name.
        "-DLLVM_INCLUDE_TESTS=OFF", "-DMLIR_INCLUDE_TESTS=OFF",
        *_cc_flags(cc, cxx), *crt,
        f"-DPython3_EXECUTABLE={cmake_path(venv_py())}",
        f"-DPython_EXECUTABLE={cmake_path(venv_py())}",
    ]
    # clang+lld; gcc uses its default linker. Skipped on Windows: "clang" matches
    # "clang-cl" there, but the MSVC-style driver picks its own linker (link/lld-link).
    if not IS_WINDOWS and "clang" in (cc or "") and shutil.which("ld.lld"):
        cfg.append("-DLLVM_USE_LINKER=lld")
    sh(cfg, env=cmake_env())
    sh([cmake_bin(), "--build", cmake_path(LLVM_BUILD), "-j", NPROC,
        "--target", *LLVM_BUILD_TARGETS], env=cmake_env())
    print("build llvm: ok ->", LLVM_BUILD / "lib" / "cmake")


def _build_iron_tools(python_executable, bdir, install_dir):
    if not _eudsl_cache_ok():
        sys.exit("eudsl not prefetched/verified — run: python dev.py fetch  (or: python dev.py setup)")
    src = PROJ / "iron-tools"
    # Keep a virtualenv's python path intact. On POSIX it is commonly a symlink;
    # resolving it selects the base interpreter, whose site-packages do not
    # contain the build dependencies installed in the virtualenv.
    python_executable = Path(os.path.abspath(python_executable))
    bdir = Path(bdir)
    install_dir = Path(install_dir)
    if not python_executable.is_file():
        raise RuntimeError(f"compiler binding Python executable missing: {python_executable}")
    cc, cxx = _compilers()   # match the compiler used for build/llvm
    env = cmake_env()
    # Network-free eudsl vendoring: the patch points mlir-aie's find-links at the local
    # prefetch cache (no remote URL), PIP_FIND_LINKS gives the cache as an absolute path
    # (cwd-independent), and PIP_NO_INDEX stops any PyPI query. The build resolves the
    # sha256-verified sdist entirely offline.
    env["PIP_FIND_LINKS"] = str(FETCH_CACHE)
    env["PIP_NO_INDEX"] = "1"
    cfg = [
        cmake_bin(), "-G", "Ninja",
        "-S", cmake_path(src), "-B", cmake_path(bdir), "-DCMAKE_BUILD_TYPE=Release",
        *_cc_flags(cc, cxx), *_ccache_flags(),
        f"-DLLVM_DIR={cmake_path(LLVM_BUILD/'lib'/'cmake'/'llvm')}",
        f"-DMLIR_DIR={cmake_path(LLVM_BUILD/'lib'/'cmake'/'mlir')}",
        f"-DCMAKE_MODULE_PATH={cmake_path(src/'cmake'/'modulesXilinx')}",
        # install the xaiengine runtime_lib into the build tree (co-located with bin/python),
        # not the default /usr/local (which needs root and breaks an unprivileged build).
        f"-DCMAKE_INSTALL_PREFIX={cmake_path(bdir)}",
        "-DAIE_RUNTIME_TARGETS=x86_64", "-DLLVM_ENABLE_ASSERTIONS=ON", "-DLLVM_ENABLE_RTTI=OFF",
        "-DAIE_ENABLE_BINDINGS_PYTHON=ON",
        # This compiler product uses HRX exclusively. Do not let a host XRT
        # installation silently add unsupported _xrt bindings or dependencies.
        "-DAIE_ENABLE_XRT_PYTHON_BINDINGS=OFF",
        # The omnibus Python-passes mode requires RTTI and creates extension
        # aliases itself. This product uses the standard MLIR-declared modules.
        "-DAIE_ENABLE_PYTHON_PASSES=OFF",
        "-DMLIR_ENABLE_PYTHON_STABLE_ABI=ON",
        # MLIR uses both CMake's FindPython3 and FindPython modules. Pin both;
        # otherwise a generic Python lookup can escape the virtual environment.
        f"-DPython3_EXECUTABLE={cmake_path(python_executable)}",
        f"-DPython_EXECUTABLE={cmake_path(python_executable)}",
    ]
    sh(cfg, env=env)
    sh([cmake_bin(), "--build", cmake_path(bdir), "-j", NPROC], env=env)
    if install_dir.parent.resolve() != BUILD.resolve():
        raise RuntimeError(f"compiler binding install must be under {BUILD}: {install_dir}")
    if install_dir.exists():
        shutil.rmtree(install_dir)
    install_dir.mkdir(parents=True)
    sh([cmake_bin(), "--install", cmake_path(bdir),
        "--prefix", cmake_path(install_dir), "--config", "Release"], env=env)
    print("build iron-tools: ok ->", bdir / "bin")


def cmd_build_iron_tools(args):
    """aie-opt/aie-translate against build/llvm (NOT the wheel)."""
    _build_iron_tools(venv_py(), BUILD / "iron-tools", IRON_TOOLS_INSTALL)


def cmd_build_hrx_xclbinutil(args):
    d = PROJ / "hrx-xclbinutil"
    sh([cmake_bin(), "-G", "Ninja", "-B", cmake_path(d / "bld"), "-S", cmake_path(d),
        "-DCMAKE_BUILD_TYPE=Release", *_ccache_flags()], env=cmake_env())
    sh([cmake_bin(), "--build", cmake_path(d / "bld"), "--target", "hrx-xclbinutil",
        "-j", NPROC], env=cmake_env())
    # expose as `xclbinutil` for aiecc.py
    tb = ROOT / "tools-bin"; tb.mkdir(exist_ok=True)
    built = d / "bld" / "tools" / f"hrx-xclbinutil{_EXE}"
    link = tb / f"xclbinutil{_EXE}"
    if link.exists() or link.is_symlink():
        link.unlink()
    if IS_WINDOWS:
        shutil.copy2(built, link)   # symlinks need Developer Mode/admin on Windows
    else:
        link.symlink_to(built)
    print("build hrx-xclbinutil: ok ->", link)


def _prepare_fresh_cmake_tree(path):
    """Clears all generated CMake and FetchContent state."""
    allowed = {HRX_CMAKE_BUILD.resolve(), FASTFLOWLM_BUILD.resolve()}
    if path.resolve() not in allowed:
        raise RuntimeError(f"refusing to clear unexpected build directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def cmd_build_hrx(args):
    d = PROJ / "hrx"
    cc, cxx = _compilers()
    env = cmake_env()
    if not (d / "CMakeLists.txt").exists():
        sys.exit("missing projects/hrx (run: git submodule update --init projects/hrx)")
    apply_patches({"hrx"})
    fresh = getattr(args, "fresh", False)
    if fresh:
        _validate_declared_patch_state("hrx", d)
        _prepare_fresh_cmake_tree(HRX_CMAKE_BUILD)

    # Configure HRX directly with Ninja. This keeps Windows off the Visual Studio
    # generator default and lets staging apply its temporary HRX patches first.
    sh([
        cmake_bin(), *(["--fresh"] if fresh else []), "-G", "Ninja",
        "-S", cmake_path(d), "-B", cmake_path(HRX_CMAKE_BUILD),
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        # Runtime artifacts use one deterministic layout on every distro.
        # GNUInstallDirs otherwise selects lib64 on some Linux hosts.
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DIREE_HAL_DRIVER_AMDGPU=OFF",
        "-DIREE_HAL_DRIVER_AMDXDNA=ON",
        # Keep the runtime-only build free of test-only downloads. `build
        # hrx-cts` explicitly reconfigures with tests enabled before requesting
        # those targets.
        "-DIREE_BUILD_TESTS=OFF",
        "-DIREE_BUILD_BENCHMARKS=OFF",
        "-DLIBHRX_BUILD_CTS=OFF",
        "-DHRX_INSTALL_TESTS=OFF",
        "-DLIBHRX_BUILD_PASSTHROUGH=OFF",
        "-DLIBHRX_BUILD_HIP_BINDING=OFF",
        "-DLIBHRX_BUILD_TOOLS=OFF",
        "-DLOOM_BUILD=OFF",
        # libhrx currently links local-task support unconditionally; keep it enabled
        # until HRX makes that dependency conditional.
        "-DIREE_HAL_DRIVER_LOCAL_SYNC=OFF",
        "-DIREE_HAL_DRIVER_NULL=OFF",
        *_cmake_source_override("FETCHCONTENT_SOURCE_DIR_FLATCC"),
        *_cc_flags(cc, cxx), *_ccache_flags(),
    ], env=env)
    sh([
        cmake_bin(), "--build", cmake_path(HRX_CMAKE_BUILD),
        *(["--clean-first"] if fresh else []),
        "--target", "iree_hal_drivers_amdxdna_registration_registration", "hrx", "flatcc_runtime",
        "-j", BUILD_JOBS,
    ], env=env)
    shutil.rmtree(HRX_INSTALL, ignore_errors=True)
    sh([
        cmake_bin(), "--install", cmake_path(HRX_CMAKE_BUILD),
        "--prefix", cmake_path(HRX_INSTALL),
        "--component", "HrxPublicDist",
    ], env=env)

    # Exercise the installed package from an unrelated build tree. This catches
    # absolute build paths, missing headers, and incomplete imported targets.
    shutil.rmtree(HRX_PACKAGE_SMOKE_BUILD, ignore_errors=True)
    package_smoke = d / "libhrx" / "cts" / "package_smoke"
    sh([
        cmake_bin(), "-G", "Ninja",
        "-S", cmake_path(package_smoke), "-B", cmake_path(HRX_PACKAGE_SMOKE_BUILD),
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={cmake_path(HRX_INSTALL)}",
        *_cc_flags(cc, cxx),
    ], env=env)
    sh([cmake_bin(), "--build", cmake_path(HRX_PACKAGE_SMOKE_BUILD), "-j", BUILD_JOBS],
       env=env)
    smoke_env = dict(env)
    if IS_WINDOWS:
        smoke_env["PATH"] = str(HRX_INSTALL / "bin") + os.pathsep + smoke_env.get("PATH", "")
    else:
        smoke_env["LD_LIBRARY_PATH"] = (str(HRX_INSTALL / "lib") + os.pathsep +
                                         smoke_env.get("LD_LIBRARY_PATH", ""))
    sh([HRX_PACKAGE_SMOKE_BUILD / f"hrx_package_smoke{_EXE}"], env=smoke_env)
    sh([HRX_PACKAGE_SMOKE_BUILD / f"hrx_package_smoke_cxx{_EXE}"], env=smoke_env)
    if fresh:
        marker = {
            "source": git_source_state(d),
            "install": directory_state(HRX_INSTALL,
                                       exclude_names=(FRESH_BUILD_MARKER,)),
        }
        (HRX_INSTALL / FRESH_BUILD_MARKER).write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
    print("build hrx (minimal amdxdna HAL + installed package): ok ->", HRX_INSTALL)


def cmd_build_hrx_cts(args):
    """The amdxdna HAL conformance binaries. Separate from `build hrx` (and from
    `build all`) because they are only useful on a machine with an NPU. Enabling
    tests here keeps the normal runtime build from fetching GoogleTest."""
    if not (HRX_CMAKE_BUILD / "CMakeCache.txt").exists():
        sys.exit("hrx not configured — run: python dev.py build hrx")
    sh([cmake_bin(), "-S", cmake_path(PROJ / "hrx"),
        "-B", cmake_path(HRX_CMAKE_BUILD),
        "-DIREE_BUILD_TESTS=ON", "-DIREE_BUILD_BENCHMARKS=ON"],
       env=cmake_env())
    sh([cmake_bin(), "--build", cmake_path(HRX_CMAKE_BUILD),
        *[f"--target=iree_hal_drivers_amdxdna_cts_{t}" for t in HRX_CTS_TESTS], "-j", BUILD_JOBS],
       env=cmake_env())
    print(f"build hrx-cts: ok ({', '.join(HRX_CTS_TESTS)})")


def cmd_build_fastflowlm(args):
    """Build the pinned FastFlowLM app against the installed HRX package.
    This intentionally skips LLVM, iron-tools, and hrx-xclbinutil."""
    flm_src = PROJ / "FastFlowLM" / "src"
    if not (flm_src / "CMakeLists.txt").exists():
        sys.exit("missing projects/FastFlowLM (run: git submodule update --init --recursive projects/FastFlowLM)")

    env = cmake_env()
    dependencies = _require_fastflowlm_deps(env)
    if not IS_WINDOWS:
        if not _have_header("libdrm/drm.h"):
            sys.exit("missing libdrm/drm.h: FastFlowLM's amdxdna headers require libdrm "
                     "(Ubuntu: sudo apt-get install libdrm-dev; Fedora: sudo dnf install libdrm-devel)")

    apply_patches({"FastFlowLM"})
    fresh = getattr(args, "fresh", False)
    if fresh:
        _validate_declared_patch_state("FastFlowLM", PROJ / "FastFlowLM")
        _prepare_fresh_cmake_tree(FASTFLOWLM_BUILD)
        shutil.rmtree(flm_src / "build", ignore_errors=True)
    hrx_config = HRX_INSTALL / "lib" / "cmake" / "hrx" / "hrx-config.cmake"
    if not hrx_config.exists():
        sys.exit(f"missing installed HRX package ({hrx_config}) - run: python dev.py build hrx")

    cc, cxx = _compilers()
    cfg = [
        cmake_bin(), *(["--fresh"] if fresh else []), "-G", "Ninja",
        "-S", cmake_path(flm_src), "-B", cmake_path(FASTFLOWLM_BUILD),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DFLM_VERSION=0.9.45",
        "-DNPU_VERSION=32.0.203.304",
        f"-DCMAKE_PREFIX_PATH={cmake_path(HRX_INSTALL)}",
        f"-DCMAKE_TOOLCHAIN_FILE={cmake_path(Path(dependencies['toolchain']))}",
        f"-DVCPKG_TARGET_TRIPLET={dependencies['triplet']}",
        f"-DVCPKG_INSTALLED_DIR={cmake_path(Path(dependencies['install_root']))}",
        f"-DVCPKG_MANIFEST_DIR={cmake_path(ROOT / '.github' / 'vcpkg' / 'fastflowlm')}",
        "-DVCPKG_MANIFEST_INSTALL=OFF",
        f"-DCMAKE_INSTALL_PREFIX={cmake_path(FASTFLOWLM_INSTALL)}",
        f"-DCMAKE_XCLBIN_PREFIX={cmake_path(flm_src)}",
        *_cc_flags(cc, cxx),
    ]
    sh(cfg, env=env)
    sh([cmake_bin(), "--build", cmake_path(FASTFLOWLM_BUILD),
        *(["--clean-first"] if fresh else []), "-j", BUILD_JOBS], env=env)
    shutil.rmtree(FASTFLOWLM_INSTALL, ignore_errors=True)
    sh([cmake_bin(), "--install", cmake_path(FASTFLOWLM_BUILD)], env=env)
    installed_flm = FASTFLOWLM_INSTALL / "bin" / f"flm{_EXE}"
    if not installed_flm.exists():
        sys.exit(f"FastFlowLM install did not produce {installed_flm}")
    if fresh:
        marker = {
            "source": git_source_state(PROJ / "FastFlowLM"),
            "hrx": json.loads((HRX_INSTALL / FRESH_BUILD_MARKER).read_text(
                encoding="utf-8")),
            "install": directory_state(FASTFLOWLM_INSTALL,
                                       exclude_names=(FRESH_BUILD_MARKER,)),
        }
        (FASTFLOWLM_INSTALL / FRESH_BUILD_MARKER).write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
    print("build fastflowlm: ok ->", installed_flm)


def _verify_fastflowlm_fresh_installs():
    try:
        return verify_fastflowlm_fresh_installs(
            hrx_install=HRX_INSTALL,
            hrx_source=PROJ / "hrx",
            fastflowlm_install=FASTFLOWLM_INSTALL,
            fastflowlm_source=PROJ / "FastFlowLM",
            marker_name=FRESH_BUILD_MARKER,
        )
    except RuntimeError as error:
        sys.exit(str(error))


def cmd_package_fastflowlm(args):
    """Create and independently re-extract the relocatable FLM + HRX runtime."""
    dev_environment.require_fastflowlm_linux_packaging_tools(archive=True)
    _verify_fastflowlm_fresh_installs()
    bundle = stage_runtime_bundle(ROOT, FASTFLOWLM_INSTALL, HRX_INSTALL,
                                  FASTFLOWLM_BUNDLE_STAGE)
    archive = create_fastflowlm_archive(bundle, FASTFLOWLM_ARTIFACTS)
    extracted = extract_fastflowlm_archive(
        archive, FASTFLOWLM_EXTRACTED, allowed_root=BUILD)
    print("package fastflowlm: ok")
    print("  archive:  ", archive)
    print("  extracted:", extracted)


def cmd_package_fastflowlm_wheels(args):
    """Build split rocm-npu-runtime and rocm-npu-fastflowlm Linux wheels."""
    if IS_WINDOWS:
        sys.exit("rocm-npu runtime/FastFlowLM wheel packaging is Linux-only for now")
    dev_environment.require_fastflowlm_linux_packaging_tools()
    _verify_fastflowlm_fresh_installs()
    version = args.version or default_package_version(ROOT)
    bundle = stage_runtime_bundle(ROOT, FASTFLOWLM_INSTALL, HRX_INSTALL,
                                  FASTFLOWLM_BUNDLE_STAGE)
    wheels = build_runtime_fastflowlm_wheels(
        bundle, FASTFLOWLM_WHEEL_STAGE, FASTFLOWLM_ARTIFACTS, version)
    print("package fastflowlm wheels: ok")
    for product in (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION):
        print(f"  {product}: {wheels[product]}")


def _python_executable_tool(name):
    tool = shutil.which(name)
    if tool:
        return tool
    candidate = Path(sys.executable).resolve().parent / (f"{name}.exe" if IS_WINDOWS else name)
    return str(candidate) if candidate.is_file() else None


def cmd_audit_fastflowlm_wheels(args):
    if IS_WINDOWS:
        sys.exit("audit-fastflowlm-wheels is Linux-only")
    wheels = find_rocm_npu_wheels(Path(args.dir))
    verify_wheel_checksums(wheels)
    expected_tags = {
        RUNTIME_DISTRIBUTION: LINUX_WHEEL_TAG,
        FASTFLOWLM_DISTRIBUTION: FASTFLOWLM_LINUX_WHEEL_TAG,
    }
    for product in (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION):
        print(f"== auditwheel analyze: {wheels[product].name} ==")
        if not wheels[product].name.endswith(f"-{expected_tags[product]}.whl"):
            raise RuntimeError(
                f"{product} wheel does not use expected tag {expected_tags[product]}: "
                f"{wheels[product].name}"
            )
        policy = auditwheel_policy(wheels[product])
        print(f"auditwheel overall policy: {policy}")
        validate_auditwheel_policy(product, policy)


def _venv_python(venv_dir):
    return Path(venv_dir) / ("Scripts" if IS_WINDOWS else "bin") / f"python{_EXE}"


def cmd_verify_fastflowlm_wheels(args):
    wheel_directory = Path(args.dir)
    wheels = find_rocm_npu_wheels(wheel_directory)
    verify_wheel_checksums(wheels)
    verify_wheels_do_not_have_text_path_references(wheels, (ROOT,))

    venv_dir = Path(args.venv)
    validate_directory_target(venv_dir, BUILD)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    sh([sys.executable, "-m", "venv", venv_dir])
    py = _venv_python(venv_dir)
    sh(
        [
            py,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links",
            wheel_directory,
            wheels[FASTFLOWLM_DISTRIBUTION],
        ]
    )
    sh([py, "-m", "pip", "check"])
    smoke = (
        "import os, sysconfig; "
        "from pathlib import Path; "
        "import rocm_npu_runtime as r, rocm_npu_fastflowlm as f; "
        "print('runtime', r.self_test(load=True)); "
        "print('flm', f.flm_path()); "
        "script = Path(sysconfig.get_path('scripts')) / ('flm.exe' if os.name == 'nt' else 'flm'); "
        "print('flm script', script); "
        "assert script.is_file(), script; "
        "print('config', f.config_path()); "
        "print('xclbins', f.xclbin_path())"
    )
    sh([py, "-c", smoke])
    print(f"verify fastflowlm wheels: ok -> {venv_dir}")


def _find_fastflowlm_archive(directory):
    directory = Path(directory)
    archives = sorted([*directory.glob("fastflowlm-hrx-*.zip"),
                       *directory.glob("fastflowlm-hrx-*.tar.zst")])
    if len(archives) != 1:
        sys.exit(f"expected exactly one FastFlowLM artifact in {directory}, found {len(archives)}")
    archive = archives[0]
    checksum_path = archive.with_name(archive.name + ".sha256")
    if not checksum_path.is_file():
        sys.exit(f"artifact checksum missing: {checksum_path}")
    fields = checksum_path.read_text(encoding="ascii").strip().split()
    if (len(fields) != 2 or fields[1] != archive.name or
            not re.fullmatch(r"[0-9a-f]{64}", fields[0])):
        sys.exit(f"malformed artifact checksum: {checksum_path}")
    if sha256_file(archive) != fields[0]:
        sys.exit(f"artifact checksum mismatch: {archive}")
    return archive


def cmd_unpack_fastflowlm(args):
    archive = _find_fastflowlm_archive(args.dir)
    bundle = extract_fastflowlm_archive(
        archive, FASTFLOWLM_EXTRACTED, allowed_root=BUILD)
    print("unpack fastflowlm: ok ->", bundle)


def cmd_verify_fastflowlm(args):
    location = Path(args.dir)
    if not location.is_dir():
        sys.exit(f"extracted FastFlowLM bundle directory not found: {location}")
    candidates = ([location] if (location / "manifest.json").is_file() else
                  [path for path in location.iterdir()
                   if path.is_dir() and (path / "manifest.json").is_file()])
    if len(candidates) != 1:
        sys.exit(f"expected exactly one extracted FastFlowLM bundle in {location}")
    manifest = verify_fastflowlm_bundle(candidates[0])
    try:
        validate_source_policy(manifest, ROOT)
    except RuntimeError as error:
        sys.exit(str(error))
    current_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    source_revision = (manifest.get("sources", {})
                       .get("rocm-npu-staging", {}).get("revision"))
    if source_revision != current_revision:
        sys.exit("FastFlowLM artifact source revision does not match this checkout")
    for record in manifest.get("patches", []):
        patch_path = ROOT / record["path"]
        if (not patch_path.is_file() or patch_path.stat().st_size != record["size"] or
                sha256_file(patch_path) != record["sha256"]):
            sys.exit(f"FastFlowLM artifact patch identity mismatch: {record['path']}")
    print(f"verify fastflowlm: ok ({len(manifest['files'])} files) -> {candidates[0]}")


def cmd_build_all(args):
    cmd_build_hrx_xclbinutil(args)
    cmd_build_llvm(args)
    cmd_build_iron_tools(args)
    cmd_build_hrx(args)


def cmd_test(args):
    """e2e: build a design's xclbin + HAL PDI artifact-contract check. Runs on Linux and
    Windows. DESIGN is an iron-tools programming_examples/basic design; the default and
    its -d device come from `[iron_tools] e2e_design`/`e2e_device` in the manifest. Others
    must be Peano-capable and use the `jit_xclbin` rule — `jit_xclbin_elf` designs also
    emit insts.elf, which needs aiebu-asm (absent here) — and pass extra aieargs after
    `--`."""
    # run_e2e.py derives every tool location from its own sys.executable, so it must run
    # under the venv python (which owns cmake, ninja and the Peano wheel).
    if not venv_py().exists():
        sys.exit(f"no venv at {VENV} (run: python dev.py setup)")
    cmd = [venv_py(), ROOT / "tests" / "e2e" / "run_e2e.py"]
    if getattr(args, "compiler_prefix", None):
        cmd += ["--compiler-prefix", args.compiler_prefix]
    if args.design:
        cmd.append(args.design)
    if args.design_args:
        cmd += ["--", *args.design_args]
    # No env= here: run_e2e.py builds the IRON toolchain env itself (see the note above
    # build_env's former home). run_e2e.py already prints "E2E FAIL: ..."; propagate its
    # status rather than let sh()'s check=True bury it under a CalledProcessError.
    sys.exit(sh(cmd, check=False).returncode)


def _require_fastflowlm_wheel_packages():
    missing = []
    for module in ("rocm_npu_runtime", "rocm_npu_fastflowlm"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        executable = FASTFLOWLM_WHEEL_VENV / _VBIN / f"python{_EXE}"
        sys.exit(
            "test-fastflowlm --wheel-package requires running dev.py with a "
            "Python environment where the rocm-npu-runtime and "
            "rocm-npu-fastflowlm wheels are installed; missing "
            f"{', '.join(missing)}. Run `python dev.py verify-fastflowlm-wheels` "
            f"first, then use `{executable} dev.py test-fastflowlm --wheel-package`."
        )


def cmd_test_fastflowlm(args):
    """Run the on-device FastFlowLM HRX smoke test."""
    script = ROOT / "tests" / "fastflowlm" / "run_fastflowlm_hrx_smoke.py"
    env = cmake_env()
    if args.bundle:
        env["FASTFLOWLM_BUNDLE"] = str(Path(args.bundle).resolve())
    elif args.wheel_package:
        _require_fastflowlm_wheel_packages()
        env["FASTFLOWLM_INSTALLED_PACKAGE"] = "1"
    else:
        _add_hrx_install_env(env)
    env.setdefault("FLM_MODEL_PATH", str(FASTFLOWLM_MODEL_CACHE))
    command = [
        sys.executable,
        script,
        "--inference-iterations",
        str(args.inference_iterations),
    ]
    sys.exit(sh(command, env=env, check=False).returncode)


def _iron_ops_suite():
    """The operator table, from its single definition in tests/iron-ops/suite.py.

    Imported by path rather than restated here (and rather than living in versions.toml,
    which is the pin authority, not a test matrix): dev.py routes the gates, suite.py says
    what they cover, and adding an operator touches only suite.py."""
    sys.path.insert(0, str(ROOT / "tests" / "iron-ops"))
    import suite
    return suite


def cmd_test_iron_ops(args):
    """Build the iron-ops operators through the source-built toolchain (no NPU run). Runs on
    Linux and Windows; this is the CPU-side gate.

    Distinct from `dev.py test`, which builds an iron-tools programming example: that proves
    the emit->Peano->bootgen->xclbin chain, while this proves the SOURCE-built aie package
    still satisfies the IRON operator library, which upstream develops against the mlir_aie
    WHEEL (v1.3.4) rather than our nightly source pin. That is exactly where API drift shows
    up, and it shows up here as a compile failure rather than an hour later on the NPU
    runner. It also compiles each operator's C++ kernel (aie_kernels/*.cc) through Peano,
    which the iron-tools examples' jit path does not do.

    Builds every operator in tests/iron-ops/suite.py; OPERATOR builds just one, and --device
    overrides the NPU generation the suite pins for it."""
    if not venv_py().exists():
        sys.exit(f"no venv at {VENV} (run: python dev.py setup)")
    cmd = [venv_py(), ROOT / "tests" / "iron-ops" / "build_operator.py"]
    if getattr(args, "operator", None):
        cmd += ["--operator", args.operator]
    if getattr(args, "device", None):
        cmd += ["--device", args.device]
    # env= here, unlike cmd_test: run_e2e.py only hands the toolchain env to the design
    # subprocesses it spawns, while build_operator.py imports `aie` in-process, so PYTHONPATH
    # and PEANO_INSTALL_DIR must already be set when the interpreter starts. _iron_env() is
    # run_e2e.toolchain_env(), so both commands still resolve the toolchain the same way.
    sys.exit(sh(cmd, env=_iron_ops_env(getattr(args, "compiler_prefix", None)),
                check=False).returncode)


def cmd_smoke(args):
    """Build-level smoke tests (no NPU, no design build):
      - hrx-xclbinutil: its ctest 'smoke' (package a MEM_TOPOLOGY section into an
        .xclbin, --info, dump-section round-trip);
      - iron-tools: aie-opt reports its version and parses a trivial NPU device.
    Cheaper than the full e2e, which additionally builds a design through Peano."""
    rc = 0
    xbld = PROJ / "hrx-xclbinutil" / "bld"
    if (xbld / "CMakeCache.txt").exists():
        print("== hrx-xclbinutil smoke (ctest) ==")
        rc |= subprocess.run(["ctest", "--test-dir", str(xbld), "--output-on-failure"],
                             env=cmake_env()).returncode
    else:
        print("smoke: hrx-xclbinutil not built (run: python dev.py build hrx-xclbinutil)"); rc = 1
    aieopt = BUILD / "iron-tools" / "bin" / f"aie-opt{_EXE}"
    if aieopt.exists():
        print("== iron-tools smoke (aie-opt) ==")
        sh([aieopt, "--version"])
        p = subprocess.run([str(aieopt)], input=b"module { aie.device(npu2) {} }\n",
                           capture_output=True)
        ok = p.returncode == 0
        print(f"  aie-opt parses aie.device(npu2): {'ok' if ok else 'FAIL'}")
        if not ok:
            print(p.stderr.decode()[:300]); rc = 1
    else:
        print("smoke: aie-opt not built (run: python dev.py build iron-tools)"); rc = 1
    print("smoke: PASS" if rc == 0 else "smoke: FAIL")
    return rc


def _require_txn_devgen(insts, target):
    """Fail if insts.bin targets a different AIE generation than the NPU we dispatch on.

    A mismatched stream is accepted by the driver and then never completes -- the
    firmware watchdog reports it as `ert state 8` (ERT_CMD_STATE_TIMEOUT) a minute later,
    with nothing pointing at the device. XAie_TxnHeader: Major, Minor, DevGen, ..."""
    want = NPU_TARGETS[target]["devgen"]
    data = insts.read_bytes()
    if len(data) < 3:
        sys.exit(f"{insts}: too small for a TXN header")
    got = data[2]
    if got != want:
        by_gen = {v["devgen"]: f"{k} / {v['arch']}" for k, v in NPU_TARGETS.items()}
        sys.exit(f"{insts}: TXN DevGen={got} ({by_gen.get(got, 'unknown')}) but --device="
                 f"{target} needs DevGen={want} ({by_gen[want]})")
    print(f"insts.bin: TXN DevGen={got} (matches --device={target})")


def _run_e2e_module():
    """Import the shared E2E design/toolchain implementation."""
    tests_dir = str(ROOT / "tests" / "e2e")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import run_e2e
    return run_e2e


def _iron_env(compiler_prefix=None):
    """The IRON toolchain env, from its single definition in tests/e2e/run_e2e.py.

    That module owns PATH/PYTHONPATH/PEANO_INSTALL_DIR/AIEOPT_DIR and the "is iron-tools
    built?" check, and it puts tools-bin first on PATH so aiecc's bare `xclbinutil`
    resolves to hrx-xclbinutil. Imported by path rather than duplicated here: the old
    build_env() copy drifted, which is why it was removed."""
    run_e2e = _run_e2e_module()
    prefix = Path(compiler_prefix).resolve() if compiler_prefix else run_e2e.compiler_prefix_from_env()
    try:
        return run_e2e.toolchain_env(run_e2e.require_xclbinutil(prefix), prefix)
    except run_e2e.E2EError as e:
        sys.exit(str(e))


def _iron_ops_env(compiler_prefix=None):
    """Layer the iron-ops source package over the selected compiler product."""
    env = _iron_env(compiler_prefix)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PROJ / "iron-ops"), env["PYTHONPATH"]])
    return env


def _libhrx_path(bundle=None, runtime_package=False):
    if runtime_package:
        try:
            import rocm_npu_runtime
            return Path(rocm_npu_runtime.libhrx_path()).resolve()
        except (ImportError, FileNotFoundError) as error:
            print(f"  [MISS] installed rocm-npu-runtime package: {error}")
            return None
    if bundle:
        candidate = (Path(bundle) / ("bin/hrx.dll" if IS_WINDOWS else "lib/libhrx.so")).resolve()
        return candidate if candidate.is_file() else None
    candidate = HRX_INSTALL / ("bin/hrx.dll" if IS_WINDOWS else "lib/libhrx.so")
    return candidate if candidate.is_file() else None


def _add_hrx_install_env(env, install=HRX_INSTALL):
    """Expose an installed HRX package to ctypes and native dependencies."""
    install = Path(install)
    lib_dir = install / ("bin" if IS_WINDOWS else "lib")
    env["HRX_INSTALL"] = str(install)
    env["LIBHRX_DIR"] = str(lib_dir)
    loader_var = "PATH" if IS_WINDOWS else "LD_LIBRARY_PATH"
    env[loader_var] = str(lib_dir) + os.pathsep + env.get(loader_var, "")
    return env


def _hrx_check(lib, status, what):
    """(ok, message) for an hrx_status_t. Consumes the status.

    hrx_status_is_ok is a header inline (status == NULL) and is not exported, so a
    c_void_p restype of None is the OK sentinel."""
    if status is None:
        return True, "ok"
    msg = ctypes.c_char_p()
    length = ctypes.c_size_t()
    lib.hrx_status_ignore(lib.hrx_status_to_string(status, ctypes.byref(msg), ctypes.byref(length)))
    code = lib.hrx_status_code(status)
    text = msg.value.decode("utf-8", "replace") if msg.value else "?"
    if msg.value:
        lib.hrx_status_free_message(msg)
    lib.hrx_status_ignore(status)
    return False, f"{what} failed (code {code}): {text}"


def _bind_libhrx(lib):
    v, p, u32, i32p = ctypes.c_void_p, ctypes.POINTER, ctypes.c_uint32, ctypes.POINTER(ctypes.c_int)
    sigs = {
        "hrx_status_code": (ctypes.c_int, [v]),
        "hrx_status_ignore": (None, [v]),
        "hrx_status_free_message": (None, [ctypes.c_char_p]),
        "hrx_status_to_string": (v, [v, p(ctypes.c_char_p), p(ctypes.c_size_t)]),
        "hrx_gpu_initialize": (v, [u32]),
        "hrx_gpu_shutdown": (v, []),
        "hrx_gpu_device_count": (v, [i32p]),
    }
    for name, (res, argtypes) in sigs.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, argtypes


# Submodules with nested submodules of their own, and those a patch touches.
RECURSIVE_SUBMODULES = ("iron-tools", "FastFlowLM")


def cmd_submodules(args):
    """Initialize the submodules a job needs, uniformly across Linux and Windows.

    --reset restores tracked files (`update --force`). --clean additionally removes
    untracked and ignored files from the named submodules: actions/checkout cleans the
    repo root but not submodule work trees, so on a reused self-hosted runner a
    previously patched tree survives, and the files patch 0003 creates are gitignored --
    a leftover copy makes `git apply` fail with "already exists". `-x` is what removes
    them, so `clean -fd` is not enough. Nested submodules carry a .git and are skipped.

    --shallow (--depth 1 --jobs 4) is always safe here and is the single biggest win:
    llvm-project alone carries >4 GB of history, and nothing reads it. Patches apply to
    the work tree, `doctor` compares pins with rev-parse, and `git apply --reverse
    --check` needs only the checkout. Fetching a pinned SHA at depth 1 works because the
    forges we clone from serve it."""
    force = ["--force"] if args.reset else []
    shallow = ["--depth", "1", "--jobs", "4"] if args.shallow else []
    update = ["git", "submodule", "update", "--init", *force]
    # One call for the flat submodules so --jobs can fetch them in parallel; the ones with
    # nested submodules need --recursive, which we do not want to apply to the others.
    flat = [n for n in args.names if n not in RECURSIVE_SUBMODULES]
    if flat:
        sh([*update, *shallow, *[f"projects/{n}" for n in flat]], cwd=ROOT)
    for name in args.names:
        if name in RECURSIVE_SUBMODULES:
            sh([*update, "--recursive", *shallow, f"projects/{name}"], cwd=ROOT)
    for name in args.clean:
        if name not in args.names:
            sys.exit(f"--clean {name}: not among the initialized submodules")
        sh(["git", "-C", str(PROJ / name), "clean", "-qfdx"])
    print(f"submodules: ok ({', '.join(args.names)})")


# The relocatable compiler product tools. aiecc drives aie-opt/aie-translate/bootgen;
# xclbinutil is added separately. aie-lsp-server and aie-visualize are editor tools and
# are not shipped.
NPU_PREREQ_TOOLS = ["aiecc", "aie-opt", "aie-translate", "bootgen",
                    "target_model", "target_model_rtti", "register_database"]
COMPILER_PRODUCT_LICENSES = (
    "rocm-npu-staging-LICENSE",
    "iron-tools-LICENSE",
    "hrx-xclbinutil-LICENSE",
    "llvm-LICENSE.txt",
    "aie-api-LICENSE",
    "aie-runtime-LICENSE.txt",
    "bootgen-LICENSE",
)
NPU_PREREQ_DIR = BUILD / "artifacts"


def _prereq_codec():
    """Codec for the run-prereqs archive: zstd when it is on PATH, gzip otherwise.

    Measured on the real payload (1.2 GB staged): tarfile's default gzip -9 takes 98 s and
    yields 324 MB; zstd -6 -T0 takes 3 s and yields 266 MB — faster AND smaller, so there is
    no trade to weigh. gzip stays as the fallback and drops to level 6 (35 s, 327 MB): level
    9 costs 3x the time for 1% of the size.

    Windows uses gzip so the NPU consumer needs no separately installed zstd executable.
    Linux probes zstd and records the selected codec in the filename; extraction trusts the
    archive extension rather than making an independent choice.
    """
    return "zst" if not IS_WINDOWS and shutil.which("zstd") else "gz"


def _prereq_archive():
    """Where package-npu-prereqs writes: named for the codec this machine can actually use."""
    return NPU_PREREQ_DIR / f"npu-prereqs.tar.{_prereq_codec()}"


def _find_prereq_archive(where):
    """The archive present under `where`, whichever codec produced it — read side, so trust
    the file on disk over what this machine would have written. None if there is none."""
    for ext in ("zst", "gz"):
        p = where / f"npu-prereqs.tar.{ext}"
        if p.exists():
            return p
    return None


@contextlib.contextmanager
def _prereq_tar(path):
    """Read the archive. tarfile handles .gz natively; .zst goes through the zstd CLI, as the
    stdlib has no zstd codec before Python 3.14 and this targets 3.12. Streamed ('r|'), so a
    handle is a single forward pass — reopen for a second pass."""
    if path.suffix != ".zst":
        with tarfile.open(path) as tar:
            yield tar
        return
    # The producer had zstd and we do not. Only reachable if the two ends stopped sharing a
    # container image; say so, rather than dying on a bare FileNotFoundError from Popen.
    if not shutil.which("zstd"):
        sys.exit(f"{path.name} needs zstd to read, and zstd is not on PATH — install it, or "
                 f"re-package where zstd is absent (dev.py then writes .tar.gz)")
    proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            yield tar
    finally:
        proc.stdout.close()
        # SIGPIPE (-13) is expected when the reader stopped early; anything else is a failure.
        if proc.wait() not in (0, -13):
            sys.exit(f"zstd -dc {path} failed (exit {proc.returncode})")


def _tool(path):
    """`path`, or `path.exe` on Windows. Fails loudly rather than silently skipping."""
    for cand in (path, path.with_suffix(".exe")):
        if cand.exists():
            return cand
    sys.exit(f"missing from the build tree: {path}")


def _stage(src, dst):
    """Copy dereferenced: the archive must hold real files, never symlinks.

    MLIR's AddMLIRPython.cmake materializes the aie Python package as absolute symlinks
    into projects/llvm-project, projects/iron-tools and build/llvm. build/llvm is not
    shipped and the consumer checks out no llvm-project, so archiving the links produces
    an artifact that only works on the machine that built it. copytree/copy2 follow
    symlinks by default and raise on a dangling one."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=True)


def _strip(stage, cts_rel):
    """Strip the ELF binaries: they are run-only and the artifact crosses a slow link.

    Never strip a static archive because it discards the symbol table. python/ holds
    executable .py scripts, so select by location and suffix rather than by the
    executable bit."""
    if IS_WINDOWS:
        return
    strip = shutil.which("strip")
    if not strip:
        print("package: no strip(1) on PATH; shipping unstripped binaries")
        return
    targets = []
    for f in stage.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(stage).as_posix()
        in_bindir = "/bin/" in f"/{rel}" or rel.startswith("tools-bin/") or cts_rel in rel
        if in_bindir or f.suffix == ".so" or ".so." in f.name:
            targets.append(f)
    if targets:
        subprocess.run([strip, "-s", *[str(t) for t in targets]], check=True)
        print(f"package: stripped {len(targets)} binaries")


def _platform_machine(machine=None):
    machine = (machine or platform.machine() or "unknown").lower()
    return "x86_64" if machine == "amd64" else machine


def _compiler_product_name(system=None, machine=None):
    return f"rocm-npu-compiler-{(system or platform.system()).lower()}-{_platform_machine(machine)}"


def _reset_build_child(path):
    resolved = Path(path).resolve()
    build_root = BUILD.resolve()
    if resolved == build_root or build_root not in resolved.parents:
        raise RuntimeError(f"refusing to remove directory outside {build_root}: {resolved}")
    if Path(path).is_symlink():
        Path(path).unlink()
    elif Path(path).exists():
        shutil.rmtree(path)
    Path(path).mkdir(parents=True)


def _assert_no_symlinks(root):
    links = [p for p in [root, *root.rglob("*")] if p.is_symlink()]
    if links:
        raise RuntimeError("compiler product contains symlinks:\n  " +
                           "\n  ".join(str(p) for p in links[:20]))


_BUILD_METADATA_NAMES = {
    "__pycache__", "CMakeFiles", "CMakeCache.txt", "cmake_install.cmake",
    "CTestTestfile.cmake", "build.ninja", "rules.ninja",
    ".ninja_deps", ".ninja_log", "install_manifest.txt",
}
_BUILD_INTERMEDIATE_SUFFIXES = {".obj", ".pdb", ".ilk", ".pyc", ".pyo"}


def _is_build_metadata(path):
    path = Path(path)
    return (any(
                part == "CMakeFiles"
                or part.startswith("objects-")
                or part.endswith("Tmp")
                for part in path.parts
            ) or
            path.name in _BUILD_METADATA_NAMES or
            path.suffix.lower() in _BUILD_INTERMEDIATE_SUFFIXES)


def _stage_runtime_tree(src, dst):
    """Copy a mixed runtime build tree without CMake/Ninja intermediates."""
    shutil.copytree(
        src, dst, symlinks=False, dirs_exist_ok=True,
        ignore=lambda _directory, names: [
            name for name in names if _is_build_metadata(Path(name))
        ])


def _peano_prefix():
    peano = Path(sysconfig.get_paths()["purelib"]) / "llvm-aie"
    if not peano.is_dir():
        raise RuntimeError(
            f"Peano is not installed in this Python environment: {peano}\n"
            "run: python dev.py setup")
    return peano


def _stage_compiler_openssl_license(product, copied):
    if not any(
        name.casefold().startswith(("libssl", "libcrypto"))
        for name in copied
    ):
        return
    openssl_root = Path(os.environ.get("OPENSSL_ROOT_DIR", "/usr"))
    license_candidates = (
        openssl_root / "share" / "openssl" / "copyright",
        openssl_root.parent / "share" / "openssl" / "copyright",
        Path("/usr/share/licenses/openssl/LICENSE.txt"),
        Path("/usr/share/doc/openssl/copyright"),
    )
    license_path = next(
        (path for path in license_candidates if path.is_file()), None
    )
    if license_path:
        _stage(
            license_path,
            product / "licenses" / "openssl-copyright",
        )


def _stage_compiler_linux_native_dependencies(product):
    """Bundles non-manylinux shared libraries required by the compiler product."""
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to close Linux compiler dependencies")
    roots = []
    openssl_root = os.environ.get("OPENSSL_ROOT_DIR")
    if openssl_root:
        root = Path(openssl_root)
        roots.extend((root, root / "lib", root / "lib64"))
    roots.extend(
        Path(path)
        for path in (
            "/usr/local/lib64",
            "/usr/local/lib",
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/lib64",
            "/lib/x86_64-linux-gnu",
        )
    )
    candidates = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.glob("*.so*"):
            if path.is_file():
                candidates.setdefault(path.name, path)

    provided = {
        path.name
        for path in product.rglob("*")
        if path.is_file() and _is_loadable_elf(path)
    }
    copied = {}
    queue = [
        path for path in product.rglob("*")
        if path.is_file() and _is_loadable_elf(path)
    ]
    while queue:
        path = queue.pop(0)
        needed_libraries = subprocess.run(
            [patchelf, "--print-needed", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for needed in needed_libraries:
            if needed in provided or _linux_system_runtime(needed):
                continue
            source = candidates.get(needed)
            if source is None:
                relative = path.relative_to(product).as_posix()
                raise RuntimeError(
                    "compiler product dependency is not available from standard "
                    f"Linux library roots: {relative} -> {needed}"
                )
            destination = product / "lib" / needed
            _stage(source, destination)
            provided.add(needed)
            copied[needed] = str(source)
            queue.append(destination)
    _stage_compiler_openssl_license(product, copied)
    return copied


def _stage_compiler_native_dependencies(product):
    """Closes non-system native imports required by the compiler product."""
    if not IS_WINDOWS:
        return _stage_compiler_linux_native_dependencies(product)

    def is_pe(path):
        try:
            return path.read_bytes()[:2] == b"MZ"
        except OSError:
            return False

    candidate_roots = []
    for variable in ("OPENSSL_ROOT_DIR", "VCPKG_INSTALLED_DIR"):
        value = os.environ.get(variable)
        if value and Path(value).is_dir():
            candidate_roots.append(Path(value))
    candidates = {
        path.name.casefold(): path
        for root in candidate_roots
        for path in root.rglob("*.dll")
        if path.is_file()
    }
    provided = {
        path.name.casefold()
        for path in product.rglob("*")
        if path.is_file() and is_pe(path)
    }
    copied = {}
    queue = [
        path for path in product.rglob("*")
        if path.is_file() and is_pe(path)
    ]
    while queue:
        path = queue.pop(0)
        for needed in pe_imports(path):
            folded = needed.casefold()
            if re.fullmatch(r"python3[0-9]{1,2}\.dll", folded):
                relative = path.relative_to(product).as_posix()
                raise RuntimeError(
                    "compiler product imports a versioned CPython DLL instead of "
                    f"the stable-ABI python3.dll: {relative} -> {needed}"
                )
            if folded in provided or _windows_system_runtime(needed):
                continue
            source = candidates.get(folded)
            if source is None:
                relative = path.relative_to(product).as_posix()
                raise RuntimeError(
                    "compiler product dependency is not available from its "
                    f"configured native dependency roots: {relative} -> {needed}"
                )
            destination = product / "bin" / source.name
            _stage(source, destination)
            provided.add(folded)
            copied[needed] = str(source)
            queue.append(destination)

    _stage_compiler_openssl_license(product, copied)
    return copied


def _verify_compiler_native_dependencies(product):
    if not IS_WINDOWS:
        return {}

    def is_pe(path):
        try:
            return path.read_bytes()[:2] == b"MZ"
        except OSError:
            return False

    binaries = [
        path for path in product.rglob("*")
        if path.is_file() and is_pe(path)
    ]
    provided = {path.name.casefold() for path in binaries}
    graph = {}
    missing = []
    versioned_python_imports = []
    for path in binaries:
        imports = pe_imports(path)
        relative = path.relative_to(product).as_posix()
        graph[relative] = imports
        for name in imports:
            folded = name.casefold()
            if re.fullmatch(r"python3[0-9]{1,2}\.dll", folded):
                versioned_python_imports.append(f"{relative} -> {name}")
            if (
                folded not in provided
                and not _windows_system_runtime(name)
            ):
                missing.append(f"{relative} -> {name}")
    if versioned_python_imports:
        raise RuntimeError(
            "compiler product imports a versioned CPython DLL instead of "
            "the stable-ABI python3.dll:\n  "
            + "\n  ".join(sorted(versioned_python_imports))
        )
    if missing:
        raise RuntimeError(
            "compiler product Windows dependency closure is incomplete:\n  "
            + "\n  ".join(sorted(missing))
        )
    return graph


def _manifest_file_record(path, root=ROOT):
    return {
        "path": path.relative_to(root).as_posix() if path.is_relative_to(root)
                else str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_product_manifest(bundle, manifest):
    manifest_path = bundle / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    files = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        files.append({"path": path.relative_to(bundle).as_posix(),
                      "size": path.stat().st_size,
                      "sha256": sha256_file(path)})
    manifest = dict(manifest, files=files)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    # _create_compiler_product_archive verifies this inventory immediately
    # before archiving. Avoid hashing the full product twice back-to-back here.
    return manifest


def _stage_compiler_product():
    it_build = BUILD / "iron-tools"
    it = IRON_TOOLS_INSTALL
    if not it.is_dir():
        raise RuntimeError("iron-tools is not installed - run: python dev.py build iron-tools")

    _reset_build_child(COMPILER_PRODUCT_STAGE)
    product = COMPILER_PRODUCT_STAGE / _compiler_product_name()
    product.mkdir(parents=True)

    required_tools = []
    for tool_name in NPU_PREREQ_TOOLS:
        tool = _tool(it_build / "bin" / tool_name)
        relative = Path("bin") / tool.name
        _stage(tool, product / relative)
        required_tools.append(relative.as_posix())

    xclbinutil = _tool(ROOT / "tools-bin" / "xclbinutil")
    relative = Path("bin") / xclbinutil.name
    _stage(xclbinutil, product / relative)
    required_tools.append(relative.as_posix())

    installed_python = it / "python"
    if not installed_python.is_dir():
        raise RuntimeError(f"missing from iron-tools install: {installed_python}")
    _stage_runtime_tree(installed_python, product / "python")
    installed_include = it / "include"
    if not installed_include.is_dir():
        raise RuntimeError(f"missing from iron-tools install: {installed_include}")
    _stage(installed_include, product / "include")
    build_include = it_build / "include"
    if not build_include.is_dir():
        raise RuntimeError(f"missing from iron-tools build: {build_include}")
    _stage_runtime_tree(build_include, product / "include")
    installed_lib = it / "lib"
    if not installed_lib.is_dir():
        raise RuntimeError(f"missing from iron-tools install: {installed_lib}")
    _stage_runtime_tree(installed_lib, product / "lib")
    # These data/header trees are generated directly into the configured build
    # prefix by upstream install rules. They contain runtime inputs, not CMake
    # object state; the Python package above must always come from the clean
    # `cmake --install` tree.
    for dirname in ("runtime_lib", "aie_runtime_lib"):
        source = it_build / dirname
        if not source.is_dir():
            raise RuntimeError(f"missing from iron-tools build: {source}")
        _stage_runtime_tree(source, product / dirname)

    aie_api = PROJ / "iron-tools" / "third_party" / "aie_api"
    if not aie_api.is_dir():
        raise RuntimeError(f"missing aie_api checkout: {aie_api}")
    _stage(aie_api, product / "third_party" / "aie_api")
    _stage(_peano_prefix(), product / "peano" / "llvm-aie")

    native_dependencies = _stage_compiler_native_dependencies(product)
    relocated_rpaths = (
        {}
        if IS_WINDOWS
        else relocate_linux_compiler_rpaths(product)
    )

    license_sources = (
        (ROOT / "LICENSE", "rocm-npu-staging-LICENSE"),
        (PROJ / "iron-tools" / "LICENSE", "iron-tools-LICENSE"),
        # hrx-xclbinutil declares Apache-2.0 in its checked-in CMakeLists.txt,
        # but its trimmed source checkout does not carry a standalone LICENSE.
        # Stage this repository's identical Apache-2.0 terms under the
        # component-specific name so the product remains license-complete.
        (ROOT / "LICENSE", "hrx-xclbinutil-LICENSE"),
        (PROJ / "llvm-project" / "llvm" / "LICENSE.TXT", "llvm-LICENSE.txt"),
        (
            PROJ / "iron-tools" / "third_party" / "aie_api" / "LICENSE",
            "aie-api-LICENSE",
        ),
        (
            PROJ / "iron-tools" / "third_party" / "aie-rt" / "license.txt",
            "aie-runtime-LICENSE.txt",
        ),
        (
            PROJ / "iron-tools" / "third_party" / "bootgen" / "LICENSE",
            "bootgen-LICENSE",
        ),
    )
    for source, name in license_sources:
        if source.is_file():
            _stage(source, product / "licenses" / name)

    _assert_no_symlinks(product)
    manifest = _write_product_manifest(product, {
        "schema_version": 1,
        "product": "rocm-npu-compiler",
        "artifact": product.name,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "required_tools": required_tools,
        "peano": {"wheel": PEANO_WHEEL, "index": PEANO_INDEX,
                  "path": "peano/llvm-aie"},
        "sources": {
            "rocm-npu-staging": git_source_state(ROOT),
            "llvm-project": git_source_state(PROJ / "llvm-project"),
            "iron-tools": git_source_state(PROJ / "iron-tools"),
            "hrx-xclbinutil": git_source_state(PROJ / "hrx-xclbinutil"),
        },
        "dependencies": {
            "requirements": [
                _manifest_file_record(ROOT / "requirements.txt"),
                _manifest_file_record(ROOT / "requirements.lock"),
            ],
            "native_runtime_files": native_dependencies,
            "relocated_elf_rpaths": relocated_rpaths,
        },
        "toolchain": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "build_image": os.environ.get("RNS_BUILD_IMAGE"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
        },
    })
    return product, manifest


def _compiler_product_root(location):
    location = Path(location)
    if (location / "manifest.json").is_file():
        return location
    if not location.is_dir():
        raise RuntimeError(f"compiler product directory not found: {location}")
    candidates = [p for p in location.iterdir()
                  if p.is_dir() and (p / "manifest.json").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one compiler product under {location}")
    return candidates[0]


def _write_compiler_product_tar_members(product, output):
    for path in sorted([product, *product.rglob("*")]):
        relative = (product.name if path == product else
                    f"{product.name}/{path.relative_to(product).as_posix()}")
        info = tarfile.TarInfo(relative)
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = ARCHIVE_FIXED_MTIME
        if path.is_file():
            info.type = tarfile.REGTYPE
            info.size = path.stat().st_size
            mode = stat.S_IMODE(path.stat().st_mode)
            info.mode = 0o755 if mode & 0o111 else 0o644
            with path.open("rb") as stream:
                output.addfile(info, stream)
        elif path.is_dir() and not path.is_symlink():
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            output.addfile(info)
        else:
            raise RuntimeError(f"unsupported compiler product filesystem entry: {path}")


def _write_compiler_product_tar_gz(product, archive):
    with tarfile.open(archive, "w:gz", compresslevel=6,
                      format=tarfile.PAX_FORMAT) as output:
        _write_compiler_product_tar_members(product, output)


def _write_compiler_product_tar_zst(product, archive):
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError(
            "the zstandard Python package is required to package the Windows "
            "compiler product; run: python dev.py setup"
        ) from error
    parameters = zstandard.ZstdCompressionParameters.from_level(
        COMPILER_PRODUCT_ZSTD_LEVEL,
        enable_ldm=True,
        window_log=COMPILER_PRODUCT_ZSTD_WINDOW_LOG,
        threads=-1,
        write_checksum=True,
    )
    compressor = zstandard.ZstdCompressor(compression_params=parameters)
    with archive.open("wb") as raw_stream:
        with compressor.stream_writer(raw_stream, closefd=False) as compressed_stream:
            with tarfile.open(fileobj=compressed_stream, mode="w|",
                              format=tarfile.PAX_FORMAT) as output:
                _write_compiler_product_tar_members(product, output)


def _create_compiler_product_archive(product, output_directory):
    _verify_compiler_product_inventory(product)
    output_directory.mkdir(parents=True, exist_ok=True)
    archive = output_directory / f"{product.name}{'.tar.zst' if IS_WINDOWS else '.tar.gz'}"
    for stale in output_directory.glob(f"{product.name}.*"):
        if stale != archive and stale != archive.with_name(archive.name + ".sha256"):
            stale.unlink()
    if IS_WINDOWS:
        _write_compiler_product_tar_zst(product, archive)
    else:
        _write_compiler_product_tar_gz(product, archive)
    archive.with_name(archive.name + ".sha256").write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="ascii")
    return archive


def _find_compiler_product_archive(directory):
    directory = Path(directory)
    archives = sorted([*directory.glob("rocm-npu-compiler-*.zip"),
                       *directory.glob("rocm-npu-compiler-*.tar.gz"),
                       *directory.glob("rocm-npu-compiler-*.tar.zst")])
    if len(archives) != 1:
        raise RuntimeError(
            f"expected exactly one rocm-npu-compiler artifact in {directory}, "
            f"found {len(archives)}")
    archive = archives[0]
    checksum_path = archive.with_name(archive.name + ".sha256")
    if not checksum_path.is_file():
        raise RuntimeError(f"artifact checksum missing: {checksum_path}")
    fields = checksum_path.read_text(encoding="ascii").strip().split()
    if (len(fields) != 2 or fields[1] != archive.name or
            not re.fullmatch(r"[0-9a-f]{64}", fields[0])):
        raise RuntimeError(f"malformed artifact checksum: {checksum_path}")
    if sha256_file(archive) != fields[0]:
        raise RuntimeError(f"artifact checksum mismatch: {archive}")
    return archive


def _verify_compiler_product_inventory(product):
    manifest = verify_fastflowlm_bundle(product)
    if manifest.get("product") != "rocm-npu-compiler":
        raise RuntimeError(f"not a rocm-npu-compiler product: {product}")
    artifact = str(manifest.get("artifact", ""))
    if "staging" in artifact:
        raise RuntimeError(f"compiler product artifact name must not include staging: {artifact}")
    build_metadata = [
        record["path"] for record in manifest.get("files", [])
        if _is_build_metadata(record["path"])
    ]
    if build_metadata:
        raise RuntimeError(
            "compiler product contains build-system metadata: "
            f"{build_metadata[0]}")
    _assert_no_symlinks(product)

    required_files = [
        "python/aie/ir.py",
        "python/aie/compiler/aiecc/main.py",
        "include/aie_api/aie.hpp",
        "aie_runtime_lib/AIE2P/vec_math.h",
        *manifest.get("required_tools", []),
    ]
    for relative in required_files:
        if not (product / relative).is_file():
            raise RuntimeError(f"compiler product is missing file: {relative}")
    for relative in ("peano/llvm-aie", "third_party/aie_api", "lib"):
        if not (product / relative).is_dir():
            raise RuntimeError(f"compiler product is missing directory: {relative}")
    for name in COMPILER_PRODUCT_LICENSES:
        if not (product / "licenses" / name).is_file():
            raise RuntimeError(f"compiler product is missing license: {name}")
    _verify_compiler_native_dependencies(product)
    return manifest


def _compiler_product_env(prefix):
    sys.path.insert(0, str(ROOT / "tests" / "e2e"))
    import run_e2e
    try:
        return run_e2e.toolchain_env(run_e2e.require_xclbinutil(prefix), prefix)
    except run_e2e.E2EError as e:
        raise RuntimeError(str(e)) from e


def _smoke_compiler_product(prefix):
    manifest = _verify_compiler_product_inventory(prefix)
    env = _compiler_product_env(prefix)
    aieopt = prefix / "bin" / f"aie-opt{_EXE}"
    sh([aieopt, "--version"], env=env)
    p = subprocess.run([str(aieopt)], input=b"module { aie.device(npu2) {} }\n",
                       env=env, capture_output=True)
    if p.returncode != 0:
        detail = p.stderr.decode(errors="replace")[:1000]
        raise RuntimeError(f"aie-opt relocation smoke failed: {detail}")
    p = subprocess.run(
        [str(venv_py()), "-c", "import aie.dialects.transform.interpreter"],
        env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "compiler Python extension import failed: "
            f"{p.stderr[-1000:]}")
    return manifest


def _relocate_compiler_product(product):
    _reset_build_child(COMPILER_PRODUCT_RELOCATED)
    relocated = COMPILER_PRODUCT_RELOCATED / product.name
    _stage(product, relocated)
    _smoke_compiler_product(relocated)
    env = dict(os.environ, ROCM_NPU_COMPILER_PREFIX=str(relocated))
    rc = sh([venv_py(), ROOT / "tests" / "e2e" / "run_e2e.py",
             "--compiler-prefix", relocated], env=env, check=False).returncode
    if rc != 0:
        raise RuntimeError(f"relocated compiler product e2e failed: {relocated}")
    return relocated


def cmd_package_compiler_product(args):
    """Stage and archive the source-built compiler product under a relocatable prefix."""
    try:
        product, manifest = _stage_compiler_product()
        archive = _create_compiler_product_archive(product, FASTFLOWLM_ARTIFACTS)
    except RuntimeError as error:
        sys.exit(str(error))
    print("package compiler product: ok")
    print("  product: ", product)
    print("  archive: ", archive)
    print(f"  files:   {len(manifest['files'])}")


def _compiler_binding_root(value):
    root = Path(value) if value else IRON_TOOLS_INSTALL / "python"
    if not (root / "aie").is_dir():
        raise RuntimeError(f"compiler binding aie package missing: {root}")
    return root


def _compiler_wheel_files(directory):
    directory = Path(directory)
    prefixes = tuple(
        distribution.replace("-", "_") + "-"
        for distribution in (
            COMPILER_DISTRIBUTION,
            COMPILER_CORE_DISTRIBUTION,
            COMPILER_BINDINGS_DISTRIBUTION,
        )
    )
    wheels = sorted(
        path for path in directory.glob("*.whl")
        if path.name.startswith(prefixes)
    )
    counts = {
        distribution: len([
            wheel for wheel in wheels
            if wheel.name.startswith(distribution.replace("-", "_") + "-")
        ])
        for distribution in (
            COMPILER_DISTRIBUTION,
            COMPILER_CORE_DISTRIBUTION,
            COMPILER_BINDINGS_DISTRIBUTION,
        )
    }
    if counts[COMPILER_DISTRIBUTION] != 1 or counts[COMPILER_CORE_DISTRIBUTION] != 1:
        raise RuntimeError(
            f"expected one compiler meta and core wheel in {directory}: {counts}"
        )
    if counts[COMPILER_BINDINGS_DISTRIBUTION] != 1:
        raise RuntimeError(
            f"expected one compiler bindings wheel in {directory}: {counts}"
        )
    return wheels


def cmd_package_compiler_wheels(args):
    version = args.version or default_package_version(ROOT)
    try:
        if args.product:
            product = _compiler_product_root(args.product)
            _verify_compiler_product_inventory(product)
        else:
            product, _ = _stage_compiler_product()
        binding_root = _compiler_binding_root(args.binding)
        platform_tag = "win_amd64" if IS_WINDOWS else "manylinux_2_28_x86_64"
        report = compiler_product_report(product)
        report_path = FASTFLOWLM_ARTIFACTS / "rocm-npu-compiler-wheel-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        wheels = build_compiler_wheel_family(
            product,
            binding_root,
            COMPILER_WHEEL_STAGE,
            FASTFLOWLM_ARTIFACTS,
            version=version,
            platform_tag=platform_tag,
            include_meta=not args.no_meta,
        )
        verify_compiler_binding_stable_abi(
            wheels[COMPILER_BINDINGS_DISTRIBUTION]
        )
    except RuntimeError as error:
        sys.exit(str(error))
    for name, wheel in wheels.items():
        print(f"{name}: {wheel}")
    print("compiler wheel report:", report_path)


def cmd_audit_compiler_wheels(args):
    if IS_WINDOWS:
        sys.exit("audit-compiler-wheels is Linux-only")
    wheels = _compiler_wheel_files(args.dir)
    verify_linux_compiler_wheels(wheels)
    core_wheels = [
        wheel for wheel in wheels
        if wheel.name.startswith(
            COMPILER_CORE_DISTRIBUTION.replace("-", "_") + "-"
        )
    ]
    if len(core_wheels) != 1:
        sys.exit("audit-compiler-wheels requires exactly one compiler core wheel")
    core_libraries = compiler_core_shared_library_names(core_wheels[0])
    for wheel in wheels:
        if wheel.name.startswith(COMPILER_DISTRIBUTION.replace("-", "_") + "-"):
            continue
        distribution = (
            COMPILER_CORE_DISTRIBUTION
            if wheel.name.startswith(
                COMPILER_CORE_DISTRIBUTION.replace("-", "_") + "-"
            )
            else COMPILER_BINDINGS_DISTRIBUTION
        )
        policy = auditwheel_policy(
            wheel,
            exclude=(
                core_libraries
                if distribution == COMPILER_BINDINGS_DISTRIBUTION
                else frozenset()
            ),
        )
        validate_auditwheel_policy(distribution, policy)
        if "manylinux_2_28_x86_64" in wheel.name and not policy.startswith(
            "manylinux_"
        ):
            raise RuntimeError(
                f"{wheel.name} claims manylinux_2_28 but auditwheel reports {policy}"
            )
        print(f"{wheel.name}: {policy}")


def _compiler_installed_prefix(python):
    result = subprocess.run(
        [
            python,
            "-c",
            "import rocm_npu_compiler as c; print(c.compiler_prefix())",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _verify_installed_compiler_bindings(python, env=None):
    expected_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    subprocess.run(
        [
            python,
            ROOT / "tests" / "compiler" / "verify_compiler_bindings.py",
            "--expected-python",
            expected_python,
        ],
        check=True,
        env=env,
    )


def _compiler_e2e_env(prefix, env=None):
    result = dict(os.environ if env is None else env)
    result["ROCM_NPU_COMPILER_PREFIX"] = str(prefix)
    result["ROCM_NPU_CMAKE"] = cmake_bin()
    return result


def _hardlink_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _compiler_site_packages(python):
    result = subprocess.run(
        [
            python,
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _export_compiler_consumer_root(python, destination):
    """Hard-links the installed compiler packages for cross-minor ABI checks."""
    destination = Path(destination)
    validate_directory_target(destination, BUILD)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    site_packages = _compiler_site_packages(python)
    names = {
        "aie",
        COMPILER_DISTRIBUTION.replace("-", "_"),
        COMPILER_CORE_DISTRIBUTION.replace("-", "_"),
        COMPILER_BINDINGS_DISTRIBUTION.replace("-", "_"),
    }
    sources = [
        path for path in site_packages.iterdir()
        if path.name in names
        or any(path.name.startswith(name + "-") and path.name.endswith(".dist-info")
               for name in names)
    ]
    if not any(path.name == "aie" for path in sources):
        raise RuntimeError(f"installed compiler bindings missing from {site_packages}")
    for source in sources:
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                copy_function=_hardlink_or_copy,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        else:
            _hardlink_or_copy(source, target)
    return destination


def _verify_compiler_wheel_consumer(args):
    if not args.consumer_root:
        raise RuntimeError("--consumer-only requires --consumer-root")
    consumer_root = Path(args.consumer_root)
    if not (consumer_root / "aie").is_dir():
        raise RuntimeError(
            f"shared compiler consumer root is missing or incomplete: {consumer_root}"
        )
    venv_dir = Path(args.venv)
    validate_directory_target(venv_dir, BUILD)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    python = venv_dir / _VBIN / f"python{_EXE}"
    _python_install(python, *COMPILER_BINDINGS_RUNTIME_REQUIREMENTS)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(consumer_root.resolve()), env.get("PYTHONPATH", "")]
    )
    _verify_installed_compiler_bindings(python, env=env)
    subprocess.run(
        [
            python,
            "-c",
            "import aie.dialects.transform.interpreter; "
            "import rocm_npu_compiler as c; "
            "print(c.compiler_prefix()); print(c.native_library_dir())",
        ],
        check=True,
        env=env,
    )
    if getattr(args, "consumer_e2e", False):
        prefix = consumer_root / COMPILER_PACKAGE
        subprocess.run(
            [
                python,
                ROOT / "tests" / "e2e" / "run_e2e.py",
                "--compiler-prefix",
                prefix,
            ],
            check=True,
            env=_compiler_e2e_env(prefix, env),
        )
    print(f"verify compiler wheel consumer: ok -> {venv_dir}")


def cmd_verify_compiler_wheels(args):
    if getattr(args, "consumer_only", False):
        try:
            _verify_compiler_wheel_consumer(args)
        except RuntimeError as error:
            sys.exit(str(error))
        return
    wheels = _compiler_wheel_files(args.dir)
    version = args.version
    if not version:
        meta = next(
            wheel for wheel in wheels
            if wheel.name.startswith(COMPILER_DISTRIBUTION.replace("-", "_") + "-")
        )
        version = meta.name[
            len(COMPILER_DISTRIBUTION.replace("-", "_")) + 1:
        ].split("-py3-none-any.whl", 1)[0]
    verify_wheel_checksums({
        f"compiler-{index}": wheel for index, wheel in enumerate(wheels)
    })
    verify_compiler_wheel_path_references(wheels, (ROOT, BUILD))
    if IS_WINDOWS:
        graph = verify_windows_compiler_wheels(wheels)
    else:
        graph = verify_linux_compiler_wheels(wheels)
    venv_dir = Path(args.venv)
    validate_directory_target(venv_dir, BUILD)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    python = venv_dir / _VBIN / f"python{_EXE}"
    _python_install(
        python,
        "--find-links",
        str(Path(args.dir).resolve()),
        f"{COMPILER_DISTRIBUTION}=={version}",
        refresh_packages=(
            COMPILER_DISTRIBUTION,
            COMPILER_CORE_DISTRIBUTION,
            COMPILER_BINDINGS_DISTRIBUTION,
        ),
    )
    subprocess.run([python, "-m", "pip", "check"], check=True)
    _verify_installed_compiler_bindings(python)
    subprocess.run(
        [
            python,
            "-c",
            "import aie.dialects.transform.interpreter; "
            "import rocm_npu_compiler as c; "
            "print(c.compiler_prefix()); print(c.native_library_dir())",
        ],
        check=True,
    )
    scripts = venv_dir / _VBIN
    for tool, argument in (("aie-opt", "--version"), ("aiecc", "--help")):
        subprocess.run([scripts / f"{tool}{_EXE}", argument], check=True)
    prefix = _compiler_installed_prefix(python)
    subprocess.run(
        [
            python,
            ROOT / "tests" / "e2e" / "run_e2e.py",
            "--compiler-prefix",
            prefix,
        ],
        check=True,
        env=_compiler_e2e_env(prefix),
    )
    if getattr(args, "consumer_root", None):
        exported = _export_compiler_consumer_root(python, args.consumer_root)
        print("compiler wheel consumer root:", exported)
    print(
        f"verify compiler wheels: ok ({len(graph)} native dependency records) "
        f"-> {venv_dir}"
    )


def cmd_unpack_compiler_product(args):
    try:
        archive = _find_compiler_product_archive(args.dir)
        output_directory = Path(args.output_dir)
        product = extract_fastflowlm_archive(
            archive, output_directory, allowed_root=output_directory.parent)
    except RuntimeError as error:
        sys.exit(str(error))
    print("unpack compiler product: ok ->", product)


def cmd_verify_compiler_product(args):
    try:
        product = _compiler_product_root(args.dir)
        manifest = _smoke_compiler_product(product)
        if args.relocate:
            relocated = _relocate_compiler_product(product)
            print("  relocated:", relocated)
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"verify compiler product: ok ({len(manifest['files'])} files) -> {product}")


def _stage_npu_runtime_prereqs(stage, hrx_install, cts_src):
    """Stage the runtime-only payload not already in the compiler product."""
    _stage(hrx_install, stage / "build/hrx-install")
    cts_rel = "/".join(HRX_CTS_DIR)
    for test_name in HRX_CTS_TESTS:
        binary = _tool(cts_src / test_name)
        _stage(binary, stage / "projects/hrx/build/cmake" / cts_rel / binary.name)


def cmd_package_npu_prereqs(args):
    """Stage and archive NPU runtime prerequisites not in the compiler product.

    Not `tar -h`: staging with a dereferencing copy fails loudly on a dangling symlink,
    where tar would silently drop it."""
    hrxb = PROJ / "hrx" / "build" / "cmake"
    cts_src = hrxb.joinpath(*HRX_CTS_DIR)
    cts_rel = "/".join(HRX_CTS_DIR)
    stage = BUILD / "npu-stage"
    if stage.exists():
        shutil.rmtree(stage)

    # The NPU job downloads the separately verified compiler product for aiecc,
    # Python bindings, headers, Peano, and xclbinutil. Duplicating that tree here
    # made the "runtime prerequisites" artifact hundreds of megabytes.
    _stage_npu_runtime_prereqs(stage, HRX_INSTALL, cts_src)

    _strip(stage, cts_rel)

    archive = _prereq_archive()
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Drop an archive of the other codec: _find_prereq_archive prefers .zst, so a stale one
    # left by an earlier run would be picked over the .gz we are writing now.
    for other in NPU_PREREQ_DIR.glob("npu-prereqs.tar.*"):
        if other != archive:
            other.unlink()
    if archive.suffix == ".zst":
        proc = subprocess.Popen(["zstd", "-6", "-T0", "-q", "-f", "-o", str(archive)],
                                stdin=subprocess.PIPE)
        with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
            tar.add(stage, arcname=".")
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit(f"zstd failed (exit {proc.returncode})")
    else:
        with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
            tar.add(stage, arcname=".")
    shutil.rmtree(stage)
    mb = archive.stat().st_size / 1e6
    print(f"package-npu-prereqs: ok -> {archive} ({mb:.0f} MB)")


def _rehearse(python, pythonpath, hrx_install):
    """Load packaged HRX through the compiler-independent NPU test binding.

    Check the packaged runtime library explicitly before importing the backend so a
    packaging omission is reported directly. The hardware job executes prebuilt XADX
    packages and intentionally has no aie/MLIR/Peano Python environment.

    ``hrx_install`` is the installed package produced and package-smoked by
    ``build hrx``."""
    libhrx_dir = hrx_install / ("bin" if IS_WINDOWS else "lib")
    env = _add_hrx_install_env(
        dict(os.environ, PYTHONPATH=str(pythonpath)), hrx_install
    )

    library_name = "hrx.dll" if IS_WINDOWS else "libhrx.so"
    libhrx = libhrx_dir / library_name
    if not libhrx.is_file():
        sys.exit(f"artifact is missing {libhrx} (needed by discovery.find_libhrx())")
    for header in ("hrx_runtime.h", "hrx_amdxdna.h"):
        if not (hrx_install / "include" / "hrx" / header).is_file():
            sys.exit(f"artifact is missing installed HRX header: {header}")

    # 1b. dlopen it here: hrxruntime does this at import and the probe discards the reason,
    #     so a libhrx that gained a shared-library dependency the image lacks would be
    #     reported as "libhrx.so could not be located".
    if not IS_WINDOWS:
        try:
            ctypes.CDLL(str(libhrx), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            sys.exit(f"{libhrx} is present but will not load: {e}")

    # Import and load exactly the tiny binding used by the hardware runner.
    script = (
        "from hrx_api import load_libhrx\n"
        "lib = load_libhrx()\n"
        "print('libhrx:', lib._name)\n"
    )
    sh([python, "-c", script], env=env)


def cmd_verify_npu_prereqs(args):
    """Assert the run-prerequisites are self-contained, off-device.

    --extracted checks a tree the consumer already unpacked; the default checks the
    archive and then rehearses npu_e2e's imports from it."""
    cts_rel = "/".join(HRX_CTS_DIR)
    cts_root = f"projects/hrx/build/cmake/{cts_rel}"
    required = [
        ("build/hrx-install/bin/hrx.dll"
         if IS_WINDOWS else
         "build/hrx-install/lib/libhrx.so"),
        "build/hrx-install/include/hrx/hrx_runtime.h",
        "build/hrx-install/include/hrx/hrx_amdxdna.h",
        *(f"{cts_root}/{name}{_EXE}" for name in HRX_CTS_TESTS),
    ]
    if args.extracted:
        # `find`/rglob over a missing root yields nothing, so assert the roots exist
        # before concluding anything from the absence of symlinks.
        roots = [BUILD / "hrx-install", PROJ.joinpath(
            "hrx", "build", "cmake", *HRX_CTS_DIR)]
        for r in roots:
            if not r.is_dir():
                sys.exit(f"{r} missing: did the artifact extract?")
        links = [p for r in roots for p in r.rglob("*") if p.is_symlink()]
        if links:
            sys.exit("artifact has symlinks (staged without dereferencing?):\n  " +
                     "\n  ".join(str(p) for p in links[:20]))
        for rel in required:
            if not (ROOT / rel).is_file():
                sys.exit(f"missing from the artifact: {rel}")
        print(f"verify-npu-prereqs: ok (runtime-only; no symlinks under {len(roots)} roots)")
        return 0

    archive = _find_prereq_archive(NPU_PREREQ_DIR)
    if archive is None:
        sys.exit(f"no npu-prereqs.tar.* under {NPU_PREREQ_DIR} — run: python dev.py package-npu-prereqs")
    with _prereq_tar(archive) as tar:
        members = tar.getmembers()
        links = [m.name for m in members if m.issym() or m.islnk()]
        if links:
            sys.exit(f"archive holds {len(links)} link(s); it is not self-contained:\n  " +
                     "\n  ".join(links[:20]))
        names = {m.name.lstrip("./") for m in members}
        for rel in required:
            if rel not in names:
                sys.exit(f"missing from the archive: {rel}")
    rehearsal = BUILD / "rehearsal"
    if rehearsal.exists():
        shutil.rmtree(rehearsal)
    rehearsal.mkdir(parents=True)
    with _prereq_tar(archive) as tar:   # streamed: a second pass needs a second handle
        tar.extractall(rehearsal, filter="data")
    runtime_python = ROOT / "tests" / "npu"
    _rehearse(venv_py(), runtime_python, rehearsal / "build" / "hrx-install")
    shutil.rmtree(rehearsal)
    print(f"verify-npu-prereqs: ok ({len(members)} entries, no links)")
    return 0


def cmd_unpack_npu_prereqs(args):
    """Extract the artifact the build job published, into the repo root.

    Extraction lives here rather than as a `tar -xf` in the workflow for the same reason
    packaging does: the two ends must agree on the codec, and only one of them gets to
    choose. Keeping both in this module means they read the same _prereq_codec()."""
    src = Path(args.dir)
    archive = _find_prereq_archive(src)
    if archive is None:
        sys.exit(f"no npu-prereqs.tar.* under {src} — did the download-artifact step run?")
    with _prereq_tar(archive) as tar:
        tar.extractall(ROOT, filter="data")
    print(f"unpack-npu-prereqs: ok ({archive.name} -> {ROOT})")


def _npu_test_modules():
    module_dir = ROOT / "tests" / "npu"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    import hrx_api
    import test_cases
    return hrx_api, test_cases


def _single_artifact(directory, pattern, description):
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise RuntimeError(
            f"{directory}: expected one {description} ({pattern}), found {len(matches)}: {names}"
        )
    return matches[0]


def cmd_package_npu_test_package(args):
    """Build execute-only XADX kernels and deterministic numeric cases for NPU CI."""
    hrx_api, test_cases = _npu_test_modules()
    libhrx_env = _add_hrx_install_env(os.environ.copy())
    serializer = hrx_api.HrxApi(
        hrx_api.load_libhrx(libhrx_env, "nt" if IS_WINDOWS else "posix")
    )

    if NPU_TEST_PACKAGE_STAGE.exists():
        shutil.rmtree(NPU_TEST_PACKAGE_STAGE)
    NPU_TEST_PACKAGE_STAGE.mkdir(parents=True)
    cases = []

    # The on-device single-dispatch gate uses vector_scalar_add, which differs
    # from the matrix_scalar_add compiler/HAL contract gate run earlier.
    design = "vector_scalar_add"
    design_dir = PROJ / "iron-tools" / "programming_examples" / "basic" / design
    run_e2e = _run_e2e_module()
    artifacts = run_e2e.build_design(
        design_dir, NPU_TARGETS["npu2"]["device"], [], _iron_env()
    )
    _require_txn_devgen(artifacts.insts, "npu2")
    cases.append(
        npu_test_package.write_case(
            NPU_TEST_PACKAGE_STAGE,
            test_cases.generate_case(design),
            serializer.serialize_xadx(
                artifacts.xclbin.read_bytes(), artifacts.insts.read_bytes(), "MLIR_AIE"
            ),
            {"kind": "iron-tools-design", "name": design, "device": "npu2"},
        )
    )

    suite = _iron_ops_suite()
    for operator in suite.OPERATORS:
        build_dir = OUT / "iron-ops" / operator.name
        xclbin = _single_artifact(build_dir, "*.xclbin", "operator xclbin")
        insts = _single_artifact(build_dir, "*.bin", "operator transaction")
        _require_txn_devgen(insts, operator.device)
        cases.append(
            npu_test_package.write_case(
                NPU_TEST_PACKAGE_STAGE,
                test_cases.generate_case(operator.name),
                serializer.serialize_xadx(
                    xclbin.read_bytes(), insts.read_bytes(), "MLIR_AIE"
                ),
                {
                    "kind": "iron-ops",
                    "name": operator.name,
                    "device": operator.device,
                    "build_kwargs": operator.build_kwargs,
                },
            )
        )

    npu_test_package.write_manifest(
        NPU_TEST_PACKAGE_STAGE,
        cases,
        _source_revision(),
        platform.system().lower(),
        platform.machine().lower(),
    )
    _verify_npu_test_package(NPU_TEST_PACKAGE_STAGE)
    archive = npu_test_package.create_archive(
        NPU_TEST_PACKAGE_STAGE, NPU_PREREQ_DIR, _prereq_codec()
    )
    print(
        f"package-npu-test-package: ok ({len(cases)} XADX cases) -> "
        f"{archive} ({archive.stat().st_size / 1e6:.1f} MB)"
    )


def _source_revision():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_npu_test_package(package):
    expected_cases = ["vector_scalar_add"] + [
        operator.name for operator in _iron_ops_suite().OPERATORS
    ]
    return npu_test_package.verify_tree(
        package, expected_cases, _source_revision(), platform.system().lower()
    )


def cmd_verify_npu_test_package(args):
    if args.extracted:
        package = NPU_TEST_PACKAGE_EXTRACTED
        try:
            manifest = _verify_npu_test_package(package)
        except RuntimeError as error:
            sys.exit(str(error))
        print(f"verify-npu-test-package: ok ({len(manifest['cases'])} XADX cases)")
        return

    archive = npu_test_package.find_archive(NPU_PREREQ_DIR)
    if archive is None:
        sys.exit(
            f"no npu-test-package.tar.* under {NPU_PREREQ_DIR} — "
            "run: python dev.py package-npu-test-package"
        )
    try:
        npu_test_package.extract_archive(archive, NPU_TEST_PACKAGE_EXTRACTED)
        manifest = _verify_npu_test_package(NPU_TEST_PACKAGE_EXTRACTED)
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"verify-npu-test-package: ok ({len(manifest['cases'])} XADX cases)")


def cmd_unpack_npu_test_package(args):
    archive = npu_test_package.find_archive(args.dir)
    if archive is None:
        sys.exit(f"no npu-test-package.tar.* under {args.dir}")
    try:
        npu_test_package.extract_archive(archive, NPU_TEST_PACKAGE_EXTRACTED)
    except RuntimeError as error:
        sys.exit(str(error))
    print(
        f"unpack-npu-test-package: ok ({archive.name} -> {NPU_TEST_PACKAGE_EXTRACTED})"
    )


def cmd_test_npu_package(args):
    """Run prebuilt XADX cases directly through HRX, with no compiler imports."""
    try:
        _verify_npu_test_package(NPU_TEST_PACKAGE_EXTRACTED)
    except RuntimeError as error:
        sys.exit(str(error))
    env = _add_hrx_install_env(os.environ.copy())
    command = [
        venv_py(),
        ROOT / "tests" / "npu" / "run_test_package.py",
        NPU_TEST_PACKAGE_EXTRACTED,
    ]
    for case in args.cases or []:
        command += ["--case", case]
    sys.exit(sh(command, env=env, check=False).returncode)


def _read_sysfs_text(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return text if text else None


def _print_probe_file(label: str, path: Path, indent="  "):
    value = _read_sysfs_text(path)
    if value is None:
        print(f"{indent}[info] {label}: unavailable ({path})")
    else:
        print(f"{indent}[info] {label}: {value}")


def _modinfo_field(field: str):
    if not shutil.which("modinfo"):
        return None
    p = subprocess.run(["modinfo", "-F", field, "amdxdna"],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       text=True, check=False)
    if p.returncode != 0:
        return None
    value = p.stdout.strip()
    return value if value else None


def _sys_accel_nodes():
    return sorted(SYS_ACCEL_DIR.glob("accel*")) if SYS_ACCEL_DIR.is_dir() else []


def _known_amdxdna_sva_regression(kernel_release: str, srcversion: str | None) -> bool:
    return (
        kernel_release.startswith("6.17.0-22")
        and srcversion in KNOWN_AMDXDNA_SVA_REGRESSION_SRCVERSIONS
    )


def _accel_open_failure_notes(err: OSError, sys_node: Path):
    notes = []
    if err.errno in (errno.EACCES, errno.EPERM):
        notes.append("runner host must grant read/write access before this job runs")
        return notes

    if err.errno == errno.EOPNOTSUPP:
        notes.append(
            "amdxdna refused userspace open with EOPNOTSUPP; check kernel log for "
            "'amdxdna_drm_open: SVA bind device failed, ret -95'"
        )
        group_type = _read_sysfs_text(sys_node / "device" / "iommu_group" / "type")
        if group_type:
            notes.append(f"IOMMU group type: {group_type}")
        try:
            kernel_release = os.uname().release
        except AttributeError:
            kernel_release = ""
        srcversion = _read_sysfs_text(AMDXDNA_MODULE_DIR / "srcversion")
        if _known_amdxdna_sva_regression(kernel_release, srcversion):
            notes.append(
                "known-bad Ubuntu 6.17.0-22 amdxdna/SVA stack; use 6.17.0-20 "
                "or a fixed newer kernel for NPU tests"
            )
        else:
            notes.append(
                "this is an amdxdna/IOMMU SVA setup failure before HRX dispatch starts"
            )
    return notes


def _probe_accel_open(target: Path, sys_node: Path) -> bool:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(target, flags)
    except OSError as e:
        print(f"  [MISS] {target} exists but open(O_RDWR) failed: {e}")
        for note in _accel_open_failure_notes(e, sys_node):
            print(f"         {note}")
        return False
    os.close(fd)
    return True


def _probe_npu_devnodes():
    rc = 0
    usable = []
    sys_nodes = _sys_accel_nodes()
    if not sys_nodes:
        print(f"  [MISS] no {SYS_ACCEL_DIR}/accel* sysfs node")
        return 1, usable

    for sys_node in sys_nodes:
        dev_text = _read_sysfs_text(sys_node / "dev")
        target = ACCEL_DIR / sys_node.name
        if not dev_text or ":" not in dev_text:
            print(f"  [MISS] {sys_node}/dev unavailable; cannot validate {target}")
            rc = 1
            continue
        major_s, minor_s = dev_text.split(":", 1)
        try:
            major, minor = int(major_s), int(minor_s)
        except ValueError:
            print(f"  [MISS] {sys_node}/dev has unexpected value {dev_text!r}")
            rc = 1
            continue

        if target.exists():
            try:
                st = target.stat()
            except OSError as e:
                print(f"  [MISS] cannot stat {target}: {e}")
                rc = 1
                continue
            if not stat.S_ISCHR(st.st_mode):
                print(f"  [MISS] {target} exists but is not a character device")
                rc = 1
                continue
            if os.major(st.st_rdev) != major or os.minor(st.st_rdev) != minor:
                have = f"{os.major(st.st_rdev)}:{os.minor(st.st_rdev)}"
                print(f"  [MISS] {target} has dev {have}, expected {dev_text}")
                rc = 1
                continue
            if not (os.access(target, os.R_OK) and os.access(target, os.W_OK)):
                print(f"  [MISS] {target} exists but is not read+write for this user")
                print("         runner host must grant read/write access before this job runs")
                rc = 1
                continue
            if not _probe_accel_open(target, sys_node):
                rc = 1
                continue
            print(f"  [ok ] {target} ({dev_text})")
            usable.append(target)
            continue

        print(f"  [MISS] missing {target} for {sys_node} ({dev_text})")
        print("         runner host must provide this devtmpfs/udev node or pass it through")
        print(f"         root repair example: install -d -m 755 {ACCEL_DIR}")
        print(f"                              mknod -m 666 {target} c {major} {minor}")
        rc = 1

    return rc, usable


def _print_npu_driver_info():
    try:
        kernel = " ".join(os.uname())
    except AttributeError:
        kernel = sys.platform
    print("== amdxdna driver / firmware ==")
    print(f"  [info] kernel: {kernel}")

    if AMDXDNA_MODULE_DIR.is_dir():
        _print_probe_file("amdxdna module version", AMDXDNA_MODULE_DIR / "version")
        _print_probe_file("amdxdna module srcversion", AMDXDNA_MODULE_DIR / "srcversion")
        _print_probe_file("amdxdna module initstate", AMDXDNA_MODULE_DIR / "initstate")
    else:
        print(f"  [MISS] no {AMDXDNA_MODULE_DIR} - is the amdxdna driver loaded?")

    for field in ["filename", "version", "srcversion"]:
        value = _modinfo_field(field)
        if value:
            print(f"  [info] amdxdna modinfo {field}: {value}")

    fw_files = _modinfo_field("firmware")
    if fw_files:
        print("  [info] amdxdna modinfo firmware files:")
        for line in fw_files.splitlines():
            print(f"    {line}")

    sys_nodes = _sys_accel_nodes()
    if not sys_nodes:
        print(f"  [MISS] no {SYS_ACCEL_DIR}/accel* sysfs node")
        return

    for n in sys_nodes:
        print(f"  [info] {n}:")
        try:
            print(f"    device path: {(n / 'device').resolve(strict=False)}")
        except OSError:
            pass
        _print_probe_file("firmware version", n / "device" / "fw_version", indent="    ")
        _print_probe_file("vbnv", n / "device" / "vbnv", indent="    ")
        _print_probe_file("vendor", n / "device" / "vendor", indent="    ")
        _print_probe_file("device", n / "device" / "device", indent="    ")
        _print_probe_file("revision", n / "device" / "revision", indent="    ")


def _probe_libhrx(bundle=None, runtime_package=False):
    lib_path = _libhrx_path(bundle, runtime_package)
    library_name = "hrx.dll" if IS_WINDOWS else "libhrx.so"
    if not lib_path:
        print(f"  [MISS] {library_name} - run: python dev.py build hrx")
        return 1
    print(f"  [ok ] {library_name}: {lib_path}")

    with contextlib.ExitStack() as stack:
        if IS_WINDOWS and hasattr(os, "add_dll_directory"):
            stack.enter_context(os.add_dll_directory(str(lib_path.parent)))
        load_args = {} if IS_WINDOWS else {"mode": ctypes.RTLD_GLOBAL}
        try:
            lib = ctypes.CDLL(str(lib_path), **load_args)
        except OSError as e:
            print(f"  [MISS] loading {library_name} failed: {e}")
            return 1
        _bind_libhrx(lib)
        ok, msg = _hrx_check(lib, lib.hrx_gpu_initialize(0), "hrx_gpu_initialize")
        if not ok:
            print(f"  [MISS] {msg}")
            return 1
        print("  [ok ] hrx_gpu_initialize")

        rc = 0
        count = ctypes.c_int(0)
        ok, msg = _hrx_check(
            lib, lib.hrx_gpu_device_count(ctypes.byref(count)), "hrx_gpu_device_count"
        )
        if not ok:
            print(f"  [MISS] {msg}")
            rc = 1
        else:
            print(f"  [{'ok ' if count.value else 'MISS'}] hrx gpu devices: {count.value}")
            if count.value == 0:
                rc = 1
        lib.hrx_status_ignore(lib.hrx_gpu_shutdown())
        return rc


def cmd_npu_probe(args):
    """Print NPU driver/firmware versions and check that libhrx can open the NPU.

    Run before on-device tests so a missing or busy device is not reported as a
    dispatch failure. Needs no build artifact beyond libhrx."""
    if IS_WINDOWS:
        library = _libhrx_path(args.bundle, args.runtime_package)
        if not library:
            print("npu-probe: hrx.dll was not found")
            return 1
        return probe_windows_npu_runtime(library)

    rc = 0
    _print_npu_driver_info()

    print("== NPU device / libhrx probe ==")
    devnode_rc, devnodes = _probe_npu_devnodes()
    rc |= devnode_rc

    if not devnodes:
        print("  [MISS] skipping hrx_gpu_initialize because no usable NPU devnode was found")
        print("npu-probe: FAIL")
        return 1
    rc |= _probe_libhrx(args.bundle, args.runtime_package)

    print("npu-probe: " + ("ok" if rc == 0 else "FAIL"))
    return rc


def _gtest_counts(output):
    """(passed, skipped) from a gtest run's summary lines."""
    import re
    passed = skipped = 0
    for line in output.splitlines():
        m = re.match(r"\[\s+PASSED\s+\] (\d+) test", line)
        if m:
            passed = int(m.group(1))
        m = re.match(r"\[\s+SKIPPED\s+\] (\d+) test", line)
        if m:
            skipped = int(m.group(1))
    return passed, skipped


def cmd_test_hrx_cts(args):
    """Run the amdxdna HAL conformance suites on the NPU (REQUIRES an NPU). These open a
    real device, so they localise a dispatch failure to the HAL rather than the IRON
    design: core_tests covers driver/event/semaphore (incl. chained command buffers).

    A suite whose cases all self-skip still exits 0 -- gtest does not count a skip as a
    failure -- so report passed/skipped per suite and fail a suite that ran nothing. The
    amdxdna backend skips every fill/copy/update case until it has native blit support,
    which is why command_buffer_tests is mostly skips today."""
    cts = PROJ.joinpath("hrx", "build", "cmake", *HRX_CTS_DIR)
    names = [args.suite] if getattr(args, "suite", None) else HRX_CTS_TESTS
    executables = {}
    for name in names:
        candidate = cts / f"{name}{_EXE}"
        if not candidate.is_file():
            sys.exit(f"CTS binary missing ({candidate}) — run: python dev.py build hrx-cts")
        executables[name] = candidate
    failed, counts = [], {}
    for n in names:
        print(f"\n=== hrx CTS: {n} ===", flush=True)
        r = subprocess.run([str(executables[n])], capture_output=True, text=True)
        print(r.stdout, end="")
        if r.stderr:
            print(r.stderr, end="", file=sys.stderr)
        counts[n] = _gtest_counts(r.stdout)
        if r.returncode != 0:
            failed.append(n)
        elif counts[n][0] == 0:
            print(f"{n}: every case skipped; the suite asserts nothing")
            failed.append(n)

    print("\nhrx CTS summary:")
    for n in names:
        p, s = counts.get(n, (0, 0))
        print(f"  {n:24s} {p:3d} passed  {s:3d} skipped")
    total_p = sum(c[0] for c in counts.values())
    total_s = sum(c[1] for c in counts.values())
    if failed:
        print(f"\nhrx CTS: FAIL ({', '.join(failed)})")
        return 1
    print(f"\nhrx CTS: PASS ({total_p} passed, {total_s} skipped across {len(names)} suites)")
    return 0


def cmd_test_npu(args):
    """ON-DEVICE e2e (REQUIRES an NPU, e.g. a gfx1151/Strix box): build an iron-tools
    design and dispatch it once through the hrx HAL, checking the result. That single
    ERT_START_NPU dispatch is what the compiler emits; it is the gate.

    --chain instead runs patch 0003's chained testbench, which dispatches the same
    kernel several times through one recorded runlist. On Linux driver/firmware
    stacks known not to handle parent ERT_CMD_CHAIN submissions, hrx now detects the
    stack and falls back to serial dispatch. Chaining remains an hrx runlist feature
    rather than compiler output, so it is opt-in and does not gate CI.

    Needs `build iron-tools` + `build hrx` (which builds libhrx.so) first."""
    design = getattr(args, "design", None) or "vector_scalar_add"
    ddir = PROJ / "iron-tools" / "programming_examples" / "basic" / design
    if not ddir.is_dir():
        sys.exit(f"{design}: no such design under programming_examples/basic")
    # The hrx IRON host-runtime is added by patch 0003 and registered in the built
    # package by its python/CMakeLists.txt hunk; without both, IRON_RUNTIME=hrx raises.
    _require_hrx_runtime(getattr(args, "compiler_prefix", None))

    env = _iron_env(getattr(args, "compiler_prefix", None))
    env["IRON_RUNTIME"] = "hrx"
    _add_hrx_install_env(env)
    target = getattr(args, "device", None) or "npu2"
    run_e2e = _run_e2e_module()
    try:
        artifacts = run_e2e.build_design(
            ddir, NPU_TARGETS[target]["device"], [], env
        )
    except run_e2e.E2EError as e:
        sys.exit(f"test-npu design build failed: {e}")
    # A wrong target is accepted by the driver and then never completes, so inspect
    # the emitted transaction rather than trusting the command-line argument.
    _require_txn_devgen(artifacts.insts, target)

    if getattr(args, "chain", False):
        if not (ddir / "test_chain_hrx.py").exists():
            sys.exit(f"{design}: no chained testbench (patch 0003 only adds one for vector_scalar_add)")
        testbench = [venv_py(), "test_chain_hrx.py"]
    else:
        testbench = [venv_py(), ROOT / "tests" / "npu" / "single_dispatch.py"]
    sh(testbench + ["--xclbin", str(artifacts.xclbin), "--instr", str(artifacts.insts),
                    "--kernel", "MLIR_AIE"], cwd=ddir, env=env)
    print(f"test-npu: ok ({design} ran on the NPU via hrx)")


def _compiler_python_root(compiler_prefix=None):
    value = compiler_prefix or os.environ.get("ROCM_NPU_COMPILER_PREFIX")
    prefix = Path(value).resolve() if value else None
    return prefix / "python" if prefix else BUILD / "iron-tools" / "python"


def _require_hrx_runtime(compiler_prefix=None):
    """The two things IRON_RUNTIME=hrx needs, named separately so the failure says which."""
    python_root = _compiler_python_root(compiler_prefix)
    if not (python_root / "aie" / "utils" / "hostruntime" / "hrxruntime"
            / "discovery.py").exists():
        sys.exit("hrxruntime missing from the built aie package — "
                 "is patches/iron-tools/0003 applied? (run: python dev.py fetch, then build iron-tools)")
    if not _libhrx_path():
        library_name = "hrx.dll" if IS_WINDOWS else "libhrx.so"
        sys.exit(f"{library_name} not found — run: python dev.py build hrx")


def cmd_test_npu_iron_ops(args):
    """ON-DEVICE e2e (REQUIRES an NPU): run an iron-ops operator's own test suite on the
    NPU through the hrx HAL, with its numerics checked against the CPU reference.

    Where `test-npu` proves ONE hand-written dispatch reaches the device, this proves the
    IRON operator library -- the layer a model actually calls -- computes the right answer
    on it. The operator is compiled here (the design is small; no LLVM is needed) and run
    with IRON_RUNTIME=hrx, so the buffers are HRXTensors and dispatch goes through libhrx,
    not XRT. That path only exists because of patches/iron-ops/0001.

    We run the operators' OWN tests rather than a testbench of our own: they already own the
    reference implementations and the tolerances, and they stay current as iron-ops grows.
    `-m "not extensive"` is upstream's small tier; suite.py's `hrx_pytest_filter`
    narrows it to out-of-box HRX smoke cases. --iterations 1 because the 5 that conftest
    defaults to exist to fill its perf CSV, and this is a correctness gate.

    Covers every operator in tests/iron-ops/suite.py; OPERATOR runs just one."""
    suite = _iron_ops_suite()
    try:
        operators = suite.select(getattr(args, "operator", None))
    except KeyError as e:
        sys.exit(e.args[0])   # KeyError stringifies to its repr; the message is the argument
    _require_hrx_runtime(getattr(args, "compiler_prefix", None))
    if not (VENV / _VBIN / f"pytest{_EXE}").exists():
        sys.exit("pytest not in the venv — the NPU job installs the iron-ops test deps: "
                 "python dev.py setup --iron-ops-tests")

    ops_dir = PROJ / "iron-ops"
    test_dirs = []
    for op in operators:
        test_dir = ops_dir / "iron" / "operators" / op.name
        if not (test_dir / "test.py").exists():
            sys.exit(f"{op.name}: no test.py under projects/iron-ops/iron/operators")
        test_dirs.append(str(test_dir))

    env = _iron_ops_env(getattr(args, "compiler_prefix", None))
    env["IRON_RUNTIME"] = "hrx"
    _add_hrx_install_env(env)
    filters = [op.hrx_pytest_filter for op in operators if op.hrx_pytest_filter]
    pytest_filter = ["-k", " or ".join(f"({f})" for f in filters)] if filters else []
    # cwd is the iron-ops checkout: pytest.ini and conftest.py live there, and conftest picks
    # the device off the LIVE runtime -- it is what skips cases this NPU does not support.
    rc = subprocess.run(
        [str(venv_py()), "-m", "pytest", *test_dirs, "-m", "not extensive",
         *pytest_filter, "--iterations", "1", "--csv-output", str(OUT / "iron-ops.csv")],
        cwd=ops_dir, env=env).returncode
    names = ", ".join(op.name for op in operators)
    if rc != 0:
        sys.exit(f"test-npu-iron-ops: FAIL ({names})")
    print(f"test-npu-iron-ops: ok ({names} ran on the NPU via hrx, numerics verified)")


def cmd_deps(args):
    """Print the in-repo dependency manifest (self-contained)."""
    print((ROOT / "requirements.txt").read_text())


def argparse_positive_int(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


_BUILD_HANDLERS = {
    "llvm": cmd_build_llvm,
    "iron-tools": cmd_build_iron_tools,
    "hrx-xclbinutil": cmd_build_hrx_xclbinutil,
    "hrx": cmd_build_hrx,
    "hrx-cts": cmd_build_hrx_cts,
    "fastflowlm": cmd_build_fastflowlm,
    "all": cmd_build_all,
}


def _dispatch_build(args):
    return _BUILD_HANDLERS[args.what](args)


def _add_command(subparsers, name, handler, *, exit_on_return=False):
    parser = subparsers.add_parser(name)
    parser.set_defaults(
        _handler=handler,
        _exit_on_return=exit_on_return,
    )
    return parser


def dispatch(args):
    """Dispatches using metadata owned by the selected argparse subparser."""
    result = args._handler(args)
    if args._exit_on_return:
        raise SystemExit(result)
    return result


def create_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = _add_command(sub, "doctor", cmd_doctor, exit_on_return=True)
    d.add_argument("profile", nargs="?", choices=doctor_profile_names(),
                   default="compiler-stack", help="dependency closure to validate")
    d.add_argument(
        "--deep",
        action="store_true",
        help="replay every patch series in a temporary Git index",
    )
    s = _add_command(sub, "setup", cmd_setup)
    s.add_argument("--profile", choices=setup_profile_names(),
                   default="compiler-stack", help="dependency closure to prepare")
    s.add_argument("--iron-ops-tests", action="store_true",
                   help="also install what iron-ops' own tests need (torch, pytest); "
                        "the NPU job does, the build jobs do not")
    _add_command(sub, "fetch", cmd_fetch)
    b = _add_command(sub, "build", _dispatch_build); b.add_argument(
        "what", choices=sorted(_BUILD_HANDLERS)
    )
    b.add_argument("--fresh", action="store_true",
                   help="clear generated CMake state and outputs before configuring")
    t = _add_command(sub, "test", cmd_test); t.add_argument("design", nargs="?")
    t.add_argument("--compiler-prefix",
                   help="relocated rocm-npu-compiler product prefix to test")
    t.add_argument("design_args", nargs="*", help="extra aieargs for the design script")
    tf = _add_command(sub, "test-fastflowlm", cmd_test_fastflowlm)
    tf.add_argument("--bundle", help="run only from an extracted runtime bundle")
    tf.add_argument("--wheel-package", action="store_true",
                    help="run from installed rocm-npu-runtime/fastflowlm wheels")
    tf.add_argument(
        "--inference-iterations",
        type=argparse_positive_int,
        default=1,
        metavar="N",
        help="run N independent two-turn FLM inference processes before the benchmark",
    )
    to = _add_command(sub, "test-iron-ops", cmd_test_iron_ops); to.add_argument("operator", nargs="?",
        help="operator to build (default: every one in tests/iron-ops/suite.py)")
    to.add_argument("--compiler-prefix",
                    help="relocated rocm-npu-compiler product prefix to test")
    to.add_argument("--device", choices=sorted(NPU_TARGETS),
                    help="override the NPU generation the suite pins for the operator")
    _add_command(sub, "smoke", cmd_smoke, exit_on_return=True)
    sm = _add_command(sub, "submodules", cmd_submodules); sm.add_argument("names", nargs="+")
    sm.add_argument("--reset", action="store_true", help="force-reset tracked files")
    sm.add_argument("--clean", nargs="*", default=[],
                    help="also remove untracked+ignored files (needed where a patch adds them)")
    sm.add_argument("--shallow", action="store_true", help="--depth 1 --jobs 4")
    _add_command(sub, "package-npu-prereqs", cmd_package_npu_prereqs)
    _add_command(sub, "package-npu-test-package", cmd_package_npu_test_package)
    _add_command(sub, "package-fastflowlm", cmd_package_fastflowlm)
    pfw = _add_command(
        sub, "package-fastflowlm-wheels", cmd_package_fastflowlm_wheels
    )
    pfw.add_argument("--version", help=f"PEP 440 package version (or ${PACKAGE_VERSION_ENV})")
    afw = _add_command(
        sub, "audit-fastflowlm-wheels", cmd_audit_fastflowlm_wheels
    )
    afw.add_argument("--dir", default=str(FASTFLOWLM_ARTIFACTS),
                     help="directory containing the produced rocm-npu wheels")
    vfw = _add_command(
        sub, "verify-fastflowlm-wheels", cmd_verify_fastflowlm_wheels
    )
    vfw.add_argument("--dir", default=str(FASTFLOWLM_ARTIFACTS),
                     help="directory containing the produced rocm-npu wheels")
    vfw.add_argument("--venv", default=str(FASTFLOWLM_WHEEL_VENV),
                     help="fresh venv to create for wheel install testing")
    _add_command(
        sub, "package-compiler-product", cmd_package_compiler_product
    )
    pcw = _add_command(
        sub, "package-compiler-wheels", cmd_package_compiler_wheels
    )
    pcw.add_argument("--version",
                     help=f"PEP 440 package version (or ${PACKAGE_VERSION_ENV})")
    pcw.add_argument(
        "--product",
        help="reuse an existing validated compiler product root or parent directory",
    )
    pcw.add_argument(
        "--binding",
        metavar="PYTHON_ROOT",
        help="installed stable-ABI iron-tools Python root",
    )
    pcw.add_argument(
        "--no-meta",
        action="store_true",
        help="omit the platform-neutral meta wheel (Windows release job)",
    )
    acw = _add_command(
        sub, "audit-compiler-wheels", cmd_audit_compiler_wheels
    )
    acw.add_argument("--dir", default=str(FASTFLOWLM_ARTIFACTS))
    vcw = _add_command(
        sub, "verify-compiler-wheels", cmd_verify_compiler_wheels
    )
    vcw.add_argument("--dir", default=str(FASTFLOWLM_ARTIFACTS))
    vcw.add_argument("--version")
    vcw.add_argument("--venv", default=str(COMPILER_WHEEL_VENV))
    vcw.add_argument(
        "--consumer-root",
        help="export or reuse a hard-linked compiler tree for cross-minor ABI checks",
    )
    vcw.add_argument(
        "--consumer-only",
        action="store_true",
        help="skip static/core/e2e checks and validate bindings against --consumer-root",
    )
    vcw.add_argument(
        "--consumer-e2e",
        action="store_true",
        help="also run the compiler e2e with a shared cross-minor consumer tree",
    )
    uf = _add_command(sub, "unpack-fastflowlm", cmd_unpack_fastflowlm)
    uf.add_argument("--dir", default=".artifact", help="directory containing the downloaded archive")
    vf = _add_command(sub, "verify-fastflowlm", cmd_verify_fastflowlm)
    vf.add_argument("--dir", default=str(FASTFLOWLM_EXTRACTED),
                    help="extracted bundle root or its parent directory")
    uc = _add_command(
        sub, "unpack-compiler-product", cmd_unpack_compiler_product
    )
    uc.add_argument("--dir", default=".artifact", help="directory containing the downloaded archive")
    uc.add_argument("--output-dir", default=str(COMPILER_PRODUCT_EXTRACTED),
                    help="physical extraction root (use a short path on Windows)")
    vc = _add_command(
        sub, "verify-compiler-product", cmd_verify_compiler_product
    )
    vc.add_argument("--dir", default=str(COMPILER_PRODUCT_STAGE),
                    help="compiler product root or its parent directory")
    vc.add_argument("--relocate", action="store_true",
                    help="copy the product to another prefix and run the off-device e2e")
    u = _add_command(sub, "unpack-npu-prereqs", cmd_unpack_npu_prereqs)
    u.add_argument("--dir", default=".artifact", help="where download-artifact put it")
    v = _add_command(
        sub,
        "verify-npu-prereqs",
        cmd_verify_npu_prereqs,
        exit_on_return=True,
    )
    v.add_argument("--extracted", action="store_true",
                   help="check an already-unpacked tree instead of the archive")
    unt = _add_command(
        sub, "unpack-npu-test-package", cmd_unpack_npu_test_package
    )
    unt.add_argument("--dir", default=".artifact", help="where download-artifact put it")
    vnt = _add_command(
        sub, "verify-npu-test-package", cmd_verify_npu_test_package
    )
    vnt.add_argument("--extracted", action="store_true",
                     help="check the unpacked package instead of the archive")
    rnt = _add_command(sub, "test-npu-package", cmd_test_npu_package)
    rnt.add_argument("--case", dest="cases", action="append",
                     help="run only this packaged case; may be repeated")
    np = _add_command(
        sub, "npu-probe", cmd_npu_probe, exit_on_return=True
    )
    np.add_argument("--bundle", help="probe libHRX from an extracted runtime bundle")
    np.add_argument("--runtime-package", action="store_true",
                    help="probe libHRX from an installed rocm-npu-runtime wheel")
    c = _add_command(
        sub, "test-hrx-cts", cmd_test_hrx_cts, exit_on_return=True
    ); c.add_argument("suite", nargs="?", choices=HRX_CTS_TESTS)
    tno = _add_command(
        sub, "test-npu-iron-ops", cmd_test_npu_iron_ops
    ); tno.add_argument("operator", nargs="?",
        help="operator to run on the NPU (default: every one in tests/iron-ops/suite.py)")
    tno.add_argument("--compiler-prefix",
                     help="relocated rocm-npu-compiler product prefix to test")
    tn = _add_command(sub, "test-npu", cmd_test_npu); tn.add_argument("design", nargs="?")
    tn.add_argument("--compiler-prefix",
                    help="relocated rocm-npu-compiler product prefix to test")
    tn.add_argument("--device", choices=sorted(NPU_TARGETS), default="npu2",
                    help="NPU generation to build the design for (default: npu2/Strix)")
    tn.add_argument("--chain", action="store_true",
                    help="run the chained testbench instead of the single-dispatch gate")
    _add_command(sub, "deps", cmd_deps)
    return p


def main():
    dispatch(create_parser().parse_args())


if __name__ == "__main__":
    main()
