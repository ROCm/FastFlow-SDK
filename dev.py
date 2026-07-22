#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""rocm-npu-staging developer command router.

Builds the NPU stack together (everything except llvm-aie, which comes
from the Peano wheel). Mirrors the command-router shape of hrx's dev.py.

  python dev.py doctor [PROFILE]      # validate compiler-stack (default), fastflowlm, or all
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
  python dev.py unpack-fastflowlm     # checksum, extract, and verify the bundle
  python dev.py package-compiler-product # relocatable rocm-npu-compiler product artifact
  python dev.py verify-compiler-product  # verify compiler product and relocation smoke
  python dev.py build all             # everything above, in dependency order
  python dev.py test [DESIGN] [-- ...] # e2e build test (no NPU run); Linux + Windows
  python dev.py test-iron-ops [OP]    # build the iron-ops operators' xclbins (no NPU run)
                                     #   Linux + Windows; suite: tests/iron-ops/suite.py
  python dev.py submodules NAME...    # init (and optionally reset/clean) submodules
  python dev.py package-npu-prereqs   # stage + tar what the NPU job needs (no NPU)
  python dev.py unpack-npu-prereqs    # extract that archive (codec-agnostic; no NPU)
  python dev.py verify-npu-prereqs    # assert that archive is self-contained + imports (no NPU)
  python dev.py npu-probe             # print NPU driver/fw versions + check libhrx open
  python dev.py test-hrx-cts [SUITE]  # amdxdna HAL conformance suites (needs an NPU)
  python dev.py test-npu [DESIGN]     # ON-DEVICE e2e: one dispatch of a design via hrx (needs an NPU)
                                     #   --device npu2|npu1   --chain (opt-in chained testbench)
  python dev.py test-fastflowlm       # pull qwen3:0.6b + run a 1k HRX benchmark
  python dev.py test-npu-iron-ops [OP] # ON-DEVICE e2e: an iron-ops operator, numerics checked (needs an NPU)
  python dev.py deps                  # print the dependency overview

Build-only: running on the NPU needs the amdxdna driver + an NPU (not here).
"""
import argparse, contextlib, ctypes, hashlib, json, os, platform, re, shutil, stat, subprocess, sys, sysconfig, tarfile, tempfile, zipfile
from pathlib import Path

from build_tools.fastflowlm_deps import (
    MIN_RUSTC as FASTFLOWLM_MIN_RUSTC,
    dependency_errors as fastflowlm_dependency_errors,
    dependency_paths as fastflowlm_dependency_paths,
    vcpkg_commit as fastflowlm_vcpkg_commit,
)
from build_tools.fastflowlm_artifact import (
    create_archive as create_fastflowlm_archive,
    directory_state,
    extract_archive as extract_fastflowlm_archive,
    git_source_state,
    sha256_file,
    stage_runtime_bundle,
    validate_source_policy,
    verify_bundle as verify_fastflowlm_bundle,
)
from build_tools.windows_npu import (
    device_is_healthy,
    driver_firmware_packages,
    parse_device_query,
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
FASTFLOWLM_BUILD = BUILD / "FastFlowLM"
FASTFLOWLM_INSTALL = BUILD / "FastFlowLM-install"
FASTFLOWLM_MODEL_CACHE = BUILD / "FastFlowLM-models"
FASTFLOWLM_BUNDLE_STAGE = BUILD / "FastFlowLM-bundle"
FASTFLOWLM_ARTIFACTS = BUILD / "artifacts"
FASTFLOWLM_EXTRACTED = BUILD / "FastFlowLM-extracted"
COMPILER_PRODUCT_STAGE = BUILD / "rocm-npu-compiler-product"
COMPILER_PRODUCT_EXTRACTED = BUILD / "rocm-npu-compiler-extracted"
COMPILER_PRODUCT_RELOCATED = BUILD / "rocm-npu-compiler-relocated"
ARCHIVE_FIXED_MTIME = 946684800  # 2000-01-01T00:00:00Z
ARCHIVE_FIXED_ZIP_DATE = (2000, 1, 1, 0, 0, 0)
FRESH_BUILD_MARKER = ".rns-fresh-build.json"
PATCHES = ROOT / "patches"                      # temporary submodule patches (see patches/README.md)
FETCH_CACHE = ROOT / ".cache" / "find-links"   # prefetched, hash-verified sdists (gitignored)
NPROC = str(os.cpu_count() or 8)
BUILD_JOBS = os.environ.get("RNS_BUILD_JOBS") or NPROC
IS_WINDOWS = os.name == "nt"                    # venv layout, .exe suffix, and MSVC differ on Windows
_VBIN = "Scripts" if IS_WINDOWS else "bin"      # venv exe dir: POSIX bin/, Windows Scripts/
_EXE = ".exe" if IS_WINDOWS else ""
_WINDOWS_BUILD_ENV = None

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
# NPU targets for `test-npu`. `make` selects the device through makefile-common's
# NPU2 switch; `devgen` is the XAie_TxnHeader.DevGen the design must then carry
# (aie-rt xaiegbl_defs.h). libhrx cannot report the generation -- its ARCHITECTURE
# property is a gfxNNNN string -- so the target is declared, not detected.
NPU_TARGETS = {
    "npu2": {"make": ["NPU2=1"], "devgen": 4, "arch": "AIE2P (Strix)"},
    "npu1": {"make": [],         "devgen": 3, "arch": "AIE2IPU (Phoenix)"},
}


def sh(cmd, cwd=None, env=None, check=True):
    print(f"+ {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=check)


def venv_py():
    return VENV / _VBIN / f"python{_EXE}"


def cmake_bin():
    """cmake from the venv (pip, >=3.30) if set up, else system cmake."""
    c = VENV / _VBIN / f"cmake{_EXE}"
    return str(c) if c.exists() else "cmake"


def cmake_path(p: Path) -> str:
    r"""A path in the form CMake always parses correctly (forward slashes; no-op on POSIX).

    CMake accepts '/' on every platform, and it re-emits some cache values verbatim into
    generated CMake code — CMAKE_MODULE_PATH, for instance, is forwarded into the
    try_compile scratch project. A Windows backslash there is read as an escape:
        set(CMAKE_MODULE_PATH "C:\home\...")  ->  Invalid character escape '\h'
    which fails the compiler-ABI check before any of our code is compiled.
    """
    return p.as_posix()


def _cmake_source_override(name: str):
    """Forwards an explicitly prepared FetchContent source into CMake."""
    value = os.environ.get(name)
    if not value:
        return []
    source = Path(value).resolve()
    if not source.is_dir():
        sys.exit(f"{name} is not a directory: {source}")
    return [f"-D{name}={cmake_path(source)}"]


def _env_path(env=None):
    e = env or os.environ
    if IS_WINDOWS:
        return e.get("Path") or e.get("PATH") or e.get("path") or ""
    return e.get("PATH", "")


def _set_env_path(env, value):
    if IS_WINDOWS:
        for key in list(env):
            if key.lower() == "path":
                del env[key]
        env["Path"] = value
    else:
        env["PATH"] = value


def _prepend_env_path(env, entries):
    parts = [str(entry) for entry in entries if str(entry)]
    current = _env_path(env)
    if current:
        parts.append(current)
    _set_env_path(env, os.pathsep.join(parts))


def _which(exe, env=None):
    return shutil.which(exe, path=_env_path(env))


def _windows_vcvars64_path():
    if not IS_WINDOWS:
        return None
    for key in ("RNS_VCVARS64", "VCVARS64"):
        value = os.environ.get(key)
        if value and Path(value).is_file():
            return Path(value)
    candidates = []
    for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if not root:
            continue
        base = Path(root) / "Microsoft Visual Studio"
        if base.exists():
            candidates.extend(base.glob("*/*/VC/Auxiliary/Build/vcvars64.bat"))
    return sorted(candidates, reverse=True)[0] if candidates else None


def _windows_msvc_env():
    """Return an environment with the installed VS x64 toolchain loaded."""
    global _WINDOWS_BUILD_ENV
    if not IS_WINDOWS:
        return dict(os.environ)
    if _WINDOWS_BUILD_ENV is not None:
        return dict(_WINDOWS_BUILD_ENV)

    env = dict(os.environ)
    vcvars = _windows_vcvars64_path()
    if not _which("cl", env) and vcvars:
        # A wrapper avoids cmd.exe /c quoting edge cases for VS paths with spaces.
        with tempfile.TemporaryDirectory(prefix="rns-vs-env-") as td:
            wrapper = Path(td) / "load-vs.cmd"
            wrapper.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\nset\r\n', encoding="utf-8")
            result = subprocess.run([str(wrapper)], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.lower() == "path":
                    _set_env_path(env, value)
                else:
                    env[key] = value
        else:
            print(f"warning: failed to load {vcvars}: {result.stderr.strip()}", file=sys.stderr)

    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        _prepend_env_path(env, [cargo_bin])
    if not env.get("ASM") and _which("ml64", env):
        env["ASM"] = "ml64"
    _WINDOWS_BUILD_ENV = env
    return dict(env)


def cmake_env():
    """PATH with the venv bin first so the pip-provided cmake/ninja are used."""
    e = _windows_msvc_env() if IS_WINDOWS else dict(os.environ)
    _prepend_env_path(e, [VENV / _VBIN])
    return e


# System (apt) prerequisites NOT installed by pip (cmake/ninja come from pip).
_SYS_HEADERS = [
    ("Python.h", "python3-dev"),              # MLIR Python bindings
    ("openssl/opensslv.h", "libssl-dev"),     # iron-tools' bootgen find_package(OpenSSL)
]


def _compilers():
    """(cc, cxx) for the build. Canonical toolchain is gcc — matching TheRock, whose
    manylinux image pins a gcc-toolset to build amd-llvm. Override locally with the
    standard CC/CXX env vars (e.g. `CC=clang CXX=clang++ python dev.py build llvm`).
    Falls back to clang only if gcc is unavailable."""
    cc, cxx = os.environ.get("CC"), os.environ.get("CXX")
    if cc and cxx:
        return cc, cxx
    if IS_WINDOWS:
        # Windows wants clang-cl (see check_system): MSVC cl rejects the GNU-only syntax
        # in hrx-xclbinutil's vendored sources. CC/CXX are handled above; with neither set,
        # return None rather than hardcode -DCMAKE_*_COMPILER, and let CMake pick from the
        # VS Developer environment. doctor flags a missing clang-cl.
        return None, None
    if shutil.which("gcc") and shutil.which("g++"):
        return "gcc", "g++"
    if shutil.which("clang") and shutil.which("clang++"):
        return "clang", "clang++"
    return None, None


def _rustc_version(env=None):
    rustc = _which("rustc", env)
    if not rustc:
        return None
    r = subprocess.run([rustc, "--version"], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return None
    parts = r.stdout.split()
    if len(parts) < 2:
        return None
    nums = parts[1].split("-", 1)[0].split(".")
    try:
        return tuple(int(n) for n in nums[:3])
    except ValueError:
        return None


def _have_header(hdr, language="c"):
    if hdr == "Python.h":  # lives under the python include dir, not a default search path
        import sysconfig
        inc = sysconfig.get_path("include")
        return bool(inc) and (Path(inc) / "Python.h").exists()
    if language == "c++":
        compiler = (os.environ.get("CXX") or shutil.which("clang++") or
                    shutil.which("c++") or shutil.which("g++"))
    else:
        compiler = (os.environ.get("CC") or shutil.which("clang") or
                    shutil.which("cc") or shutil.which("gcc"))
    if not compiler:
        return False
    p = subprocess.run([compiler, "-fsyntax-only", "-x", language, "-"],
                       input=f"#include <{hdr}>\nint main(){{return 0;}}\n".encode(),
                       capture_output=True)
    return p.returncode == 0


def _have_perl_module(module):
    perl = shutil.which("perl")
    if not perl:
        return False
    return subprocess.run(
        [perl, f"-M{module}", "-e", "1"], capture_output=True
    ).returncode == 0


def _check_fastflowlm_linux_host(verbose):
    """Checks host tools/interfaces needed before vcpkg can build FLM deps."""
    missing = []
    cc, cxx = _compilers()
    checks = [
        ("bash", bool(shutil.which("bash")), "bash"),
        ("git", bool(shutil.which("git")), "git"),
        ("make", bool(shutil.which("make")), "build-essential"),
        ("nasm", bool(shutil.which("nasm")), "nasm"),
        ("pkg-config", bool(shutil.which("pkg-config")), "pkg-config"),
        ("tar", bool(shutil.which("tar")), "tar"),
        ("unzip", bool(shutil.which("unzip")), "unzip"),
        ("curl or wget", bool(shutil.which("curl") or shutil.which("wget")), "curl"),
        ("Perl IPC::Cmd", _have_perl_module("IPC::Cmd"), "libipc-cmd-perl"),
        ("<linux/types.h>", _have_header("linux/types.h"), "linux-libc-dev"),
        ("<libdrm/drm.h>", _have_header("libdrm/drm.h"), "libdrm-dev"),
    ]
    if verbose:
        print(f"  [{'ok ' if cc else 'MISS'}] C/C++ compiler: "
              f"{(cc + '/' + cxx) if cc else 'need gcc/g++ (or set CC/CXX)'}")
    if not cc:
        missing.append("build-essential")
    for label, ok, package in checks:
        if verbose:
            print(f"  [{'ok ' if ok else 'MISS'}] {label}")
        if not ok and package not in missing:
            missing.append(package)
    return not missing, missing


def _check_compiler_stack_system(verbose):
    """Return (ok, missing_apt_packages) for the LLVM/IRON compiler stack."""
    missing = []
    cc, cxx = _compilers()
    if IS_WINDOWS:
        # Windows deps come from the VS toolchain + choco/vcpkg, not apt; _compilers()
        # returns None by design, so probe the toolchain directly. clang-cl is REQUIRED,
        # not merely preferred: MSVC cl rejects the GNU-only syntax in hrx-xclbinutil's
        # vendored sources (__attribute__((aligned))/((visibility)), __builtin_bswap64).
        # Finding only cl is a fail here rather than a confusing compile error later.
        # Relax this to "cl or clang-cl" once those five sites are made portable.
        env = _windows_msvc_env()
        has_clang_cl = bool(_which("clang-cl", env))
        has_git = bool(_which("git", env))
        if verbose:
            print(f"  [{'ok ' if has_clang_cl else 'MISS'}] clang-cl (required; MSVC cl cannot build hrx-xclbinutil)")
            if not has_clang_cl and shutil.which("cl"):
                print("         found MSVC cl but not clang-cl — install the VS 'C++ Clang tools' component")
            print(f"  [{'ok ' if has_git else 'MISS'}] git")
            print("  [note] set CC=CXX=clang-cl so the builds dev.py drives pin the compiler")
            print("  [note] OpenSSL (bootgen) comes from choco/vcpkg on Windows; no apt package list")
        return (has_clang_cl and has_git, [])   # apt package names are not meaningful on Windows
    if verbose:
        print(f"  [{'ok ' if cc else 'MISS'}] C/C++ compiler: {(cc + '/' + cxx) if cc else 'need gcc (or set CC/CXX)'} (canonical: gcc; set CC/CXX for clang)")
        print(f"  [{'ok ' if shutil.which('ld.lld') else '--  '}] ld.lld (optional; used with clang)")
    if not cc:
        missing.append("build-essential")  # provides gcc/g++
    for probe, pkg in [("git", "git"), ("make", "build-essential")]:
        ok = shutil.which(probe) is not None
        if verbose:
            print(f"  [{'ok ' if ok else 'MISS'}] {probe}")
        if not ok and pkg not in missing:
            missing.append(pkg)
    for hdr, pkg in _SYS_HEADERS:
        ok = _have_header(hdr)
        if verbose:
            print(f"  [{'ok ' if ok else 'MISS'}] <{hdr}>")
        if not ok and pkg not in missing:
            missing.append(pkg)
    return (not missing, missing)


def _check_fastflowlm_system(verbose):
    """Return (ok, missing_apt_packages) for the minimal HRX/FLM build."""
    missing = []
    env = _windows_msvc_env() if IS_WINDOWS else dict(os.environ)
    paths = fastflowlm_dependency_paths(ROOT)
    native_errors = fastflowlm_dependency_errors(
        paths,
        cargo_available=bool(_which("cargo", env)),
        rustc_version=_rustc_version(env),
    )
    if IS_WINDOWS:
        compiler = _which("cl", env) or _which("clang-cl", env)
        has_git = bool(_which("git", env))
        if verbose:
            print(f"  [{'ok ' if compiler else 'MISS'}] C/C++ compiler: {Path(compiler).name if compiler else 'need MSVC cl or clang-cl'}")
            print(f"  [{'ok ' if has_git else 'MISS'}] git")
            for error in native_errors:
                print(f"  [MISS] {error}")
            if not native_errors:
                print(f"  [ok ] Rust/Cargo >= {'.'.join(map(str, FASTFLOWLM_MIN_RUSTC))}")
                print(f"  [ok ] vcpkg FastFlowLM dependencies: {paths['triplet_root']}")
        return (bool(compiler) and has_git and not native_errors, [])

    host_ok, missing = _check_fastflowlm_linux_host(verbose)
    if verbose:
        for error in native_errors:
            print(f"  [MISS] {error}")
        if not native_errors:
            print(f"  [ok ] Rust/Cargo >= {'.'.join(map(str, FASTFLOWLM_MIN_RUSTC))}")
            print(f"  [ok ] vcpkg FastFlowLM dependencies: {paths['triplet_root']}")
    return (host_ok and not native_errors, missing)


def check_system(profile="compiler-stack", verbose=True):
    """Return (ok, missing_apt_packages) for a doctor profile."""
    if profile == "fastflowlm":
        return _check_fastflowlm_system(verbose)
    if profile == "compiler-stack":
        return _check_compiler_stack_system(verbose)
    if profile == "all":
        if verbose:
            print("  compiler-stack:")
        compiler_ok, compiler_missing = _check_compiler_stack_system(verbose)
        if verbose:
            print("  fastflowlm:")
        flm_ok, flm_missing = _check_fastflowlm_system(verbose)
        return compiler_ok and flm_ok, sorted(set(compiler_missing + flm_missing))
    raise ValueError(f"unknown doctor profile: {profile}")


# NOTE: the IRON toolchain env (PATH/PYTHONPATH/PEANO_INSTALL_DIR/AIEOPT_DIR + the
# "is iron-tools built?" check) lives in tests/e2e/run_e2e.py, its only consumer. It used
# to be duplicated here as build_env(); the copy drifted — it looked for `aie-opt` with no
# `.exe` and put `.venv/bin` (not Scripts) on PATH, so `dev.py test` aborted on Windows
# before run_e2e.py ever ran. One definition, in the place that uses it.


# ---- commands ----

def _manifest():
    import tomllib
    return tomllib.load(open(ROOT / "manifests" / "versions.toml", "rb"))


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _eudsl_sdist():
    """(cache_path, pin) for the pinned eudsl sdist. cache_path may not exist yet."""
    pin = _manifest().get("eudsl", {})
    if not pin:
        return None, {}
    fname = f"{pin['package'].replace('-', '_')}-{pin['version']}.tar.gz"
    return FETCH_CACHE / fname, pin


def _eudsl_cache_ok():
    """True iff the pinned eudsl sdist is in the cache and its sha256 matches."""
    path, pin = _eudsl_sdist()
    return bool(pin) and path.exists() and _sha256(path) == pin.get("sha256")


def _patch_state(patch, sub):
    """Return (state, apply options) for a patch against submodule dir `sub`.

    Patches are normalized to LF. Some upstream files are tracked as CRLF, so if
    strict matching fails, retry context matching without treating line-ending
    whitespace as semantic. Non-whitespace source drift still fails.
    """
    for options in ((), ("--ignore-space-change",)):
        rev = subprocess.run(
            ["git", "-C", str(sub), "apply", "--reverse", "--check", *options, str(patch)],
            capture_output=True,
        )
        if rev.returncode == 0:
            return "applied", options
        fwd = subprocess.run(
            ["git", "-C", str(sub), "apply", "--check", *options, str(patch)],
            capture_output=True,
        )
        if fwd.returncode == 0:
            return "appliable", options
    return "broken", ()


def apply_patches(submodule_names=None):
    """Apply the temporary submodule patches (idempotent). These live only in
    rocm-npu-staging and are upstreamable as-is; see patches/README.md."""
    if not PATCHES.exists():
        return
    selected = set(submodule_names) if submodule_names is not None else None
    for sub_dir in sorted(p for p in PATCHES.iterdir() if p.is_dir()):
        if selected is not None and sub_dir.name not in selected:
            continue
        sub = PROJ / sub_dir.name
        if not sub.exists():
            sys.exit(f"submodule missing for patches/{sub_dir.name} (run: git submodule update --init projects/{sub_dir.name})")
        patches = sorted(sub_dir.glob("*.patch"))
        states = [_patch_state(patch, sub) for patch in patches]
        if any(state == "broken" for state, _ in states):
            try:
                exact = git_source_state(sub) == _declared_patch_expected_state(
                    sub_dir.name, sub)
            except RuntimeError:
                exact = False
            if exact:
                for patch in patches:
                    print(f"patch: {sub_dir.name}/{patch.name} already applied")
                continue
        for patch in patches:
            st, options = _patch_state(patch, sub)
            if st == "applied":
                print(f"patch: {sub_dir.name}/{patch.name} already applied")
            elif st == "appliable":
                sh(["git", "-C", str(sub), "apply", *options, str(patch)])
                print(f"patch: applied {sub_dir.name}/{patch.name}")
            else:
                sys.exit(f"patch does not apply cleanly: {patch} (submodule moved? regenerate it)")


def _declared_patch_expected_state(component, source):
    """Builds the expected source state from the pin and ordered patch series."""
    BUILD.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(
        prefix=f"patch-state-{component}-", dir=BUILD))
    worktree = temporary_parent / "source"
    try:
        subprocess.run(
            ["git", "-C", str(source), "worktree", "add", "--quiet",
             "--detach", str(worktree), "HEAD"], check=True)
        for patch in sorted((PATCHES / component).glob("*.patch")):
            state, options = _patch_state(patch, worktree)
            if state == "appliable":
                subprocess.run(
                    ["git", "-C", str(worktree), "apply", *options, str(patch)],
                    check=True)
            elif state != "applied":
                raise RuntimeError(f"declared patch does not apply: {patch}")
        return git_source_state(worktree)
    finally:
        subprocess.run(
            ["git", "-C", str(source), "worktree", "remove", "--force",
             str(worktree)], capture_output=True)
        shutil.rmtree(temporary_parent, ignore_errors=True)


def _validate_declared_patch_state(component, source):
    """Proves the source tree equals its pin plus the ordered patch sequence."""
    expected = _declared_patch_expected_state(component, source)
    actual = git_source_state(source)
    if actual != expected:
        expected_files = {item["path"]: item for item in expected["changed_files"]}
        actual_files = {item["path"]: item for item in actual["changed_files"]}
        differing = sorted(
            path for path in set(expected_files) | set(actual_files)
            if expected_files.get(path) != actual_files.get(path))
        raise RuntimeError(
            f"{component} source differs from its pin plus declared patches: "
            f"{', '.join(differing) or 'tracked diff metadata'}")


def cmd_fetch(args, submodule_names=None):
    """Prepare sources for an offline build (the single network step): apply temporary
    patches + prefetch the build-time-fetched eudsl sdist into a gitignored,
    hash-verified cache. Idempotent."""
    apply_patches(submodule_names)
    if submodule_names is not None and "iron-tools" not in submodule_names:
        print("fetch: selected source patches are ready (no compiler-stack prefetch needed)")
        return
    path, pin = _eudsl_sdist()
    if not pin:
        sys.exit("manifests/versions.toml has no [eudsl] section")
    if _eudsl_cache_ok():
        print(f"fetch: eudsl already cached + verified ({path.name})")
        return
    FETCH_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"fetch: downloading {pin['package']}=={pin['version']}")
    sh([str(venv_py()), "-m", "pip", "download", f"{pin['package']}=={pin['version']}",
        "-f", pin["find_links"], "--no-deps", "--no-binary", ":all:", "-d", str(FETCH_CACHE)])
    got = _sha256(path) if path.exists() else "<missing>"
    if got != pin["sha256"]:
        path.unlink(missing_ok=True)
        sys.exit(f"fetch: sha256 mismatch for {path.name}\n  expected {pin['sha256']}\n  got      {got}")
    print(f"fetch: ok -> {path} (sha256 verified)")


PIN_SECTION_TO_SUBMODULE = {
    "llvm": "llvm-project",
    "iron_tools": "iron-tools",
    "hrx": "hrx",
    "hrx_xclbinutil": "hrx-xclbinutil",
    "fastflowlm": "FastFlowLM",
    "iron_ops": "iron-ops",
}
DOCTOR_PROFILE_SUBMODULES = {
    "compiler-stack": {"llvm-project", "iron-tools", "hrx", "hrx-xclbinutil", "iron-ops"},
    "fastflowlm": {"hrx", "FastFlowLM"},
    "all": set(PIN_SECTION_TO_SUBMODULE.values()),
}
SETUP_PROFILE_SUBMODULES = {
    "compiler-stack": DOCTOR_PROFILE_SUBMODULES["compiler-stack"],
    "fastflowlm": DOCTOR_PROFILE_SUBMODULES["fastflowlm"],
    "npu-e2e": {"hrx", "iron-tools", "iron-ops"},
}


def check_pins(submodules):
    """Verify selected submodule checkouts against manifests/versions.toml."""
    man = _manifest()
    problems = []
    for sec, sub in PIN_SECTION_TO_SUBMODULE.items():
        if sub not in submodules:
            continue
        pinned = man.get(sec, {}).get("commit")
        if not pinned:
            continue
        sub_dir = PROJ / sub
        if not (sub_dir / ".git").exists():   # not checked out — rev-parse here would walk up to the superproject
            print(f"  [ABSENT] {sub}: required by this doctor profile")
            problems.append(sub)
            continue
        r = subprocess.run(["git", "-C", str(sub_dir), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        head = r.stdout.strip()
        match = head == pinned
        print(f"  [{'ok  ' if match else 'DRIFT'}] {sub}: manifest {pinned[:12]} vs HEAD {(head[:12] or '?')}")
        if not match:
            problems.append(sub)
    return problems


def cmd_doctor(args):
    """Check the selected build path's prerequisites, pins, patches, and fetches."""
    profile = args.profile
    selected_submodules = DOCTOR_PROFILE_SUBMODULES[profile]
    print(f"doctor profile: {profile}")
    print("system prerequisites (cmake + ninja come from pip via `setup`):")
    system_ok, missing = check_system(profile)
    print("\nsubmodule pins vs manifests/versions.toml:")
    pin_bad = check_pins(selected_submodules)
    print("\ntemporary submodule patches (patches/, applied by builds or `dev.py fetch`):")
    patch_bad = []
    for sub_dir in sorted(p for p in PATCHES.iterdir() if p.is_dir()) if PATCHES.exists() else []:
        if sub_dir.name not in selected_submodules:
            continue
        submodule_dir = PROJ / sub_dir.name
        if not (submodule_dir / ".git").exists():
            print(f"  [ABSENT] {sub_dir.name}: cannot check patches")
            continue
        patches = sorted(sub_dir.glob("*.patch"))
        states = [_patch_state(patch, submodule_dir)[0] for patch in patches]
        try:
            exact = git_source_state(
                submodule_dir) == _declared_patch_expected_state(
                    sub_dir.name, submodule_dir)
            series_valid = True
        except RuntimeError:
            exact = False
            series_valid = False
        if exact:
            states = ["applied"] * len(patches)
        elif series_valid:
            # A later patch may overlap an earlier patch's reverse context.
            # The clean ordered replay above is authoritative; "pending" means
            # setup/fetch still needs to apply that valid sequence.
            states = ["pending" if state == "broken" else state
                      for state in states]
        for patch, st in zip(patches, states):
            print(f"  [{'ok   ' if st == 'applied' else st.upper()}] {sub_dir.name}/{patch.name}")
            if st == "broken":
                patch_bad.append(patch.name)

    needs_eudsl = profile in ("compiler-stack", "all")
    eudsl_ok = True
    ver_ok = True
    if needs_eudsl:
        print("\nbuild-time prefetch (run `dev.py fetch` to populate):")
        eudsl_ok = _eudsl_cache_ok()
        print(f"  [{'ok  ' if eudsl_ok else 'MISS '}] eudsl sdist cached + sha256 verified")
        # Cross-check the two declarations so a pin change cannot silently use stale input.
        eudsl_ver = _manifest().get("eudsl", {}).get("version", "")
        patch_f = PATCHES / "iron-tools" / "0001-eudsl-for-amd-llvm-23.patch"
        pm = re.search(r"\+eudsl-python-extras==(\S+)", patch_f.read_text()) if patch_f.exists() else None
        ver_ok = bool(eudsl_ver) and pm is not None and pm.group(1) == eudsl_ver
        print(f"  [{'ok  ' if ver_ok else 'DRIFT'}] eudsl version: patch=={pm.group(1) if pm else '?'} == manifest=={eudsl_ver or '?'}")
    rc = 0
    if not ver_ok:
        print("\neudsl version DRIFT: patches/iron-tools/0001-… and versions.toml [eudsl] disagree — sync them")
        rc = 1
    if patch_bad:
        print(f"\nPATCH does not apply: {', '.join(patch_bad)} — regenerate against the current submodule")
        rc = 1
    if not system_ok:
        if missing:
            print("\nMISSING system deps — install with:\n  sudo apt-get install -y " + " ".join(sorted(set(missing))))
        else:
            print("\nMISSING system prerequisites shown above")
        rc = 1
    if pin_bad:
        print(f"\nSUBMODULE PIN problems in: {', '.join(pin_bad)} — initialize it or update its manifest pin")
        rc = 1
    if needs_eudsl and not eudsl_ok:
        print("\neudsl not prefetched — run: python dev.py fetch")
        rc = 1
    if rc == 0:
        print(f"\nall good: {profile} prerequisites, pins, and patches are ready.")
    return rc


def _install_fastflowlm_deps():
    """Installs the manifest-defined native dependencies into the managed cache."""
    if not IS_WINDOWS:
        host_ok, missing = _check_fastflowlm_linux_host(verbose=False)
        if not host_ok:
            sys.exit(
                "FastFlowLM host build prerequisites are missing; install: "
                "sudo apt-get install -y " + " ".join(sorted(set(missing)))
            )
    paths = fastflowlm_dependency_paths(ROOT)
    source = Path(paths["source"])
    install_root = Path(paths["install_root"])
    binary_cache = ROOT / ".cache" / "vcpkg-bincache"
    downloads = ROOT / ".cache" / "vcpkg-downloads"
    for path in (source.parent, install_root, binary_cache, downloads):
        path.mkdir(parents=True, exist_ok=True)

    source_changed = False
    if not (source / ".git").is_dir():
        sh(["git", "init", str(source)])
        sh(["git", "-C", str(source), "remote", "add", "origin",
            "https://github.com/microsoft/vcpkg.git"])
        source_changed = True
    else:
        sh(["git", "-C", str(source), "remote", "set-url", "origin",
            "https://github.com/microsoft/vcpkg.git"])
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    vcpkg_commit = fastflowlm_vcpkg_commit(ROOT)
    if head != vcpkg_commit:
        sh(["git", "-C", str(source), "fetch", "--depth", "1", "origin",
            vcpkg_commit])
        sh(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"])
        source_changed = True
    executable = Path(paths["executable"])
    if source_changed or not executable.is_file():
        if IS_WINDOWS:
            sh(["cmd.exe", "/d", "/c", str(source / "bootstrap-vcpkg.bat"),
                "-disableMetrics"])
        else:
            sh([str(source / "bootstrap-vcpkg.sh"), "-disableMetrics"])

    env = os.environ.copy()
    env["VCPKG_BINARY_SOURCES"] = (
        f"clear;files,{cmake_path(binary_cache)},readwrite"
    )
    env["VCPKG_DOWNLOADS"] = str(downloads)
    # Some managed Windows runners expose different TLS/revocation behavior
    # through WinHTTP and system curl. This transport adapter tries both and
    # preserves vcpkg's required SHA-512 verification; dependency policy and
    # package resolution remain entirely owned by vcpkg.
    powershell = shutil.which("powershell.exe") if IS_WINDOWS else None
    downloader = ROOT / ".github" / "scripts" / "vcpkg-download.ps1"
    if powershell and downloader.is_file():
        template = (
            f'"{powershell}" -NoProfile -NonInteractive -ExecutionPolicy Bypass '
            f'-File "{downloader}" -Url {{url}} -Sha512 {{sha512}} '
            '-Destination {dst}'
        )
        env["X_VCPKG_ASSET_SOURCES"] = f"clear;x-script,{template}"

    sh([
        executable, "install",
        "--triplet", paths["triplet"],
        f"--overlay-triplets={ROOT / '.github' / 'vcpkg' / 'triplets'}",
        f"--x-manifest-root={ROOT / '.github' / 'vcpkg' / 'fastflowlm'}",
        f"--x-install-root={install_root}",
    ], env=env)
    marker = Path(paths["marker"])
    marker_tmp = marker.with_suffix(".tmp")
    marker_tmp.write_text(
        json.dumps(paths["policy"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker_tmp.replace(marker)
    print("setup: FastFlowLM native dependencies ->",
          paths["triplet_root"])


def cmd_setup(args):
    """Prepare the selected dependency closure in the repository-local cache.

    Compiler-stack Python deps come from the hash-pinned requirements.lock;
    FastFlowLM installs its pinned CMake/Ninja tooling and the same
    manifest-defined native dependency graph on Linux and Windows.

    requirements.txt is the human-edited input; regenerate the lock after editing it:
      .venv/bin/uv pip compile requirements.txt --generate-hashes --emit-index-url \\
          --emit-find-links --python-version 3.12 -o requirements.lock
    iron-tools/LLVM/bootgen/aie_api/aie-rt build from source; eudsl is prefetched (fetch)."""
    OUT.mkdir(exist_ok=True)
    if args.profile == "compiler-stack":
        ok, missing = check_system(verbose=False)
        if not ok:
            if missing:
                print("WARNING: missing system deps (apt): " + " ".join(sorted(missing)))
            else:
                print("WARNING: compiler-stack system prerequisites are missing")
            print("  install them, then re-run; see `python dev.py doctor compiler-stack`")
    if not venv_py().exists():
        sh([sys.executable, "-m", "venv", str(VENV)])
    pip = [str(venv_py()), "-m", "pip", "install"]
    if args.profile != "fastflowlm":
        sh(pip + ["--upgrade", "pip"])
    lock = ROOT / "requirements.lock"
    if args.profile == "fastflowlm":
        src = ROOT / "requirements-fastflowlm.lock"
    # requirements.lock is hash-pinned for LINUX wheels; on Windows the platform-specific
    # wheels (llvm-aie, cmake, ...) have different sha256s, so those hashes can't match.
    # Use the unpinned requirements.txt there — same version pins + find-links, no hashes.
    elif IS_WINDOWS:
        src = ROOT / "requirements.txt"
    else:
        src = lock if lock.exists() else ROOT / "requirements.txt"   # lock = reproducible (hashes)
    pip_args = ["--require-hashes"] if args.profile == "fastflowlm" else []
    sh(pip + pip_args + ["-r", str(src)])
    cmd_fetch(args, SETUP_PROFILE_SUBMODULES[args.profile])
    if args.profile != "fastflowlm":
        install_iron_ops(pip, with_tests=getattr(args, "iron_ops_tests", False))
    if args.profile == "fastflowlm":
        _install_fastflowlm_deps()
    print(f"setup: ok ({args.profile} dependencies are ready)")


def install_iron_ops(pip, with_tests=False):
    """Put `iron` (the operator library) on the venv's path, WITHOUT its dependencies.

    --no-deps is the whole point. iron-ops' pyproject pins the mlir_aie and llvm-aie WHEELS;
    installing those would put a second, wheel-built `aie` package in site-packages, where it
    would shadow the source-built one this stack exists to test (for an installed
    distribution, site-packages beats PYTHONPATH). Its real deps are ours to pin, and
    requirements.txt does.

    Editable: the checkout IS the source of truth -- its aie_kernels/*.cc are what the
    operator build compiles -- and it keeps the venv valid when the submodule pin moves.

    with_tests adds what iron-ops' OWN tests need (torch, for the golden references), which
    only the NPU job runs. The build jobs deliberately do without: torch is a ~200 MB wheel
    they would download to import and never call. Compiling an operator does not need it --
    but only because patches/iron-ops/0002 made iron.operators import lazily. Before that,
    `import iron.operators.elementwise_add.op` eagerly imported mha, and mha imports torch."""
    if not (PROJ / "iron-ops" / "pyproject.toml").exists():
        print("setup: projects/iron-ops not checked out; skipping the `iron` install")
        return
    sh(pip + ["--no-deps", "-e", str(PROJ / "iron-ops")])
    if with_tests:
        # Same lock-vs-txt split as the base deps above: the lock's hashes are for the Linux
        # wheels, and the NPU job is Linux, so this is the reproducible path in practice.
        lock = ROOT / "requirements-iron-ops.lock"
        src = ROOT / "requirements-iron-ops.txt" if IS_WINDOWS or not lock.exists() else lock
        sh(pip + ["-r", str(src)])


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
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON", "-DCMAKE_BUILD_TYPE=Release",
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
    ]
    # clang+lld; gcc uses its default linker. Skipped on Windows: "clang" matches
    # "clang-cl" there, but the MSVC-style driver picks its own linker (link/lld-link).
    if not IS_WINDOWS and "clang" in (cc or "") and shutil.which("ld.lld"):
        cfg.append("-DLLVM_USE_LINKER=lld")
    sh(cfg, env=cmake_env())
    sh([cmake_bin(), "--build", cmake_path(LLVM_BUILD), "-j", NPROC,
        "--target", *LLVM_BUILD_TARGETS], env=cmake_env())
    print("build llvm: ok ->", LLVM_BUILD / "lib" / "cmake")


def cmd_build_iron_tools(args):
    """aie-opt/aie-translate against build/llvm (NOT the wheel)."""
    if not _eudsl_cache_ok():
        sys.exit("eudsl not prefetched/verified — run: python dev.py fetch  (or: python dev.py setup)")
    src = PROJ / "iron-tools"
    bdir = BUILD / "iron-tools"
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
        f"-DPython3_EXECUTABLE={cmake_path(venv_py())}",
    ]
    sh(cfg, env=env)
    sh([cmake_bin(), "--build", cmake_path(bdir), "-j", NPROC], env=env)
    print("build iron-tools: ok ->", bdir / "bin")


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
    ], env=cmake_env())
    sh([
        cmake_bin(), "--build", cmake_path(HRX_CMAKE_BUILD),
        *(["--clean-first"] if fresh else []),
        "--target", "iree_hal_drivers_amdxdna_registration_registration", "hrx", "flatcc_runtime",
        "-j", BUILD_JOBS,
    ], env=cmake_env())
    shutil.rmtree(HRX_INSTALL, ignore_errors=True)
    sh([
        cmake_bin(), "--install", cmake_path(HRX_CMAKE_BUILD),
        "--prefix", cmake_path(HRX_INSTALL),
        "--component", "HrxPublicDist",
    ], env=cmake_env())

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
    ], env=cmake_env())
    sh([cmake_bin(), "--build", cmake_path(HRX_PACKAGE_SMOKE_BUILD), "-j", BUILD_JOBS],
       env=cmake_env())
    smoke_env = cmake_env()
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


def cmd_package_fastflowlm(args):
    """Create and independently re-extract the relocatable FLM + HRX runtime."""
    if not (FASTFLOWLM_INSTALL / "bin" / f"flm{_EXE}").is_file():
        sys.exit("FastFlowLM is not installed - run: python dev.py build fastflowlm")
    markers = {}
    for name, install, source in (
            ("HRX", HRX_INSTALL, PROJ / "hrx"),
            ("FastFlowLM", FASTFLOWLM_INSTALL, PROJ / "FastFlowLM")):
        marker = install / FRESH_BUILD_MARKER
        if not marker.is_file():
            sys.exit(f"{name} was not built with --fresh")
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("source") != git_source_state(source):
            sys.exit(f"{name} sources changed after the fresh build; rebuild with --fresh")
        if recorded.get("install") != directory_state(
                install, exclude_names=(FRESH_BUILD_MARKER,)):
            sys.exit(f"{name} install changed after the fresh build; rebuild with --fresh")
        markers[name] = recorded
    if markers["FastFlowLM"].get("hrx") != markers["HRX"]:
        sys.exit("FastFlowLM was built against a different HRX install; rebuild with --fresh")
    bundle = stage_runtime_bundle(ROOT, FASTFLOWLM_INSTALL, HRX_INSTALL,
                                  FASTFLOWLM_BUNDLE_STAGE)
    archive = create_fastflowlm_archive(bundle, FASTFLOWLM_ARTIFACTS)
    extracted = extract_fastflowlm_archive(
        archive, FASTFLOWLM_EXTRACTED, allowed_root=BUILD)
    print("package fastflowlm: ok")
    print("  archive:  ", archive)
    print("  extracted:", extracted)


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


def cmd_test_fastflowlm(args):
    """Run the on-device FastFlowLM HRX smoke test."""
    script = ROOT / "tests" / "fastflowlm" / "run_fastflowlm_hrx_smoke.py"
    env = cmake_env()
    if args.bundle:
        env["FASTFLOWLM_BUNDLE"] = str(Path(args.bundle).resolve())
    else:
        env.setdefault("HRX_BUILD", str(HRX_CMAKE_BUILD))
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
    sys.exit(sh(cmd, env=_iron_env(getattr(args, "compiler_prefix", None)),
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


def _iron_env(compiler_prefix=None):
    """The IRON toolchain env, from its single definition in tests/e2e/run_e2e.py.

    That module owns PATH/PYTHONPATH/PEANO_INSTALL_DIR/AIEOPT_DIR and the "is iron-tools
    built?" check, and it puts tools-bin first on PATH so aiecc's bare `xclbinutil`
    resolves to hrx-xclbinutil. Imported by path rather than duplicated here: the old
    build_env() copy drifted, which is why it was removed."""
    sys.path.insert(0, str(ROOT / "tests" / "e2e"))
    import run_e2e
    prefix = Path(compiler_prefix).resolve() if compiler_prefix else run_e2e.compiler_prefix_from_env()
    try:
        return run_e2e.toolchain_env(run_e2e.require_xclbinutil(prefix), prefix)
    except run_e2e.E2EError as e:
        sys.exit(str(e))


def _libhrx_path(bundle=None):
    if bundle:
        candidate = (Path(bundle) / ("bin/hrx.dll" if IS_WINDOWS else "lib/libhrx.so")).resolve()
        return candidate if candidate.is_file() else None
    pattern = "hrx.dll" if IS_WINDOWS else "libhrx*.so"
    hits = sorted((PROJ / "hrx" / "build" / "cmake").rglob(pattern))
    return hits[0] if hits else None


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


# The run-prerequisites npu_e2e needs, and nothing else. aiecc drives
# aie-opt/aie-translate/bootgen; xclbinutil packages the xclbin. aie-lsp-server and
# aie-visualize are editor tools and are not shipped.
NPU_PREREQ_TOOLS = ["aiecc", "aie-opt", "aie-translate", "bootgen",
                    "target_model", "target_model_rtti", "register_database"]
NPU_PREREQ_DIR = BUILD / "artifacts"


def _prereq_codec():
    """Codec for the run-prereqs archive: zstd when it is on PATH, gzip otherwise.

    Measured on the real payload (1.2 GB staged): tarfile's default gzip -9 takes 98 s and
    yields 324 MB; zstd -6 -T0 takes 3 s and yields 266 MB — faster AND smaller, so there is
    no trade to weigh. gzip stays as the fallback and drops to level 6 (35 s, 327 MB): level
    9 costs 3x the time for 1% of the size.

    Probing rather than hardcoding is what keeps the two ends in agreement: the build job and
    the NPU job run the same container image, so this resolves identically on both, and both
    go through this module rather than naming a codec in YAML.
    """
    return "zst" if shutil.which("zstd") else "gz"


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

    Never strip a static archive -- it discards the symbol table, and libflatcc_runtime.a
    is linked by build_xadx_helper.sh. python/ holds executable .py scripts, so select by
    location and suffix rather than by the executable bit."""
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


def _peano_prefix():
    peano = Path(sysconfig.get_paths()["purelib"]) / "llvm-aie"
    if not peano.is_dir():
        raise RuntimeError(
            f"Peano is not installed in this Python environment: {peano}\n"
            "run: python dev.py setup")
    return peano


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
    return verify_fastflowlm_bundle(bundle)


def _stage_compiler_product():
    it = BUILD / "iron-tools"
    if not it.is_dir():
        raise RuntimeError("iron-tools is not installed - run: python dev.py build iron-tools")

    _reset_build_child(COMPILER_PRODUCT_STAGE)
    product = COMPILER_PRODUCT_STAGE / _compiler_product_name()
    product.mkdir(parents=True)

    required_tools = []
    for tool_name in NPU_PREREQ_TOOLS:
        tool = _tool(it / "bin" / tool_name)
        relative = Path("bin") / tool.name
        _stage(tool, product / relative)
        required_tools.append(relative.as_posix())

    xclbinutil = _tool(ROOT / "tools-bin" / "xclbinutil")
    relative = Path("bin") / xclbinutil.name
    _stage(xclbinutil, product / relative)
    required_tools.append(relative.as_posix())

    for dirname in ("python", "lib", "runtime_lib", "include", "aie_runtime_lib"):
        source = it / dirname
        if not source.is_dir():
            raise RuntimeError(f"missing from iron-tools install: {source}")
        _stage(source, product / dirname)

    aie_api = PROJ / "iron-tools" / "third_party" / "aie_api"
    if not aie_api.is_dir():
        raise RuntimeError(f"missing aie_api checkout: {aie_api}")
    _stage(aie_api, product / "third_party" / "aie_api")
    _stage(_peano_prefix(), product / "peano" / "llvm-aie")

    license_sources = (
        (ROOT / "LICENSE", "rocm-npu-staging-LICENSE"),
        (PROJ / "iron-tools" / "LICENSE", "iron-tools-LICENSE"),
        (PROJ / "hrx-xclbinutil" / "LICENSE", "hrx-xclbinutil-LICENSE"),
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


def _write_compiler_product_zip(product, archive):
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as output:
        for path in sorted(p for p in product.rglob("*") if p.is_file()):
            relative = f"{product.name}/{path.relative_to(product).as_posix()}"
            info = zipfile.ZipInfo(relative, date_time=ARCHIVE_FIXED_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, path.read_bytes(), compresslevel=9)


def _write_compiler_product_tar_gz(product, archive):
    with tarfile.open(archive, "w:gz", compresslevel=6,
                      format=tarfile.PAX_FORMAT) as output:
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


def _create_compiler_product_archive(product, output_directory):
    _verify_compiler_product_inventory(product)
    output_directory.mkdir(parents=True, exist_ok=True)
    archive = output_directory / f"{product.name}{'.zip' if IS_WINDOWS else '.tar.gz'}"
    for stale in output_directory.glob(f"{product.name}.*"):
        if stale != archive and stale != archive.with_name(archive.name + ".sha256"):
            stale.unlink()
    if IS_WINDOWS:
        _write_compiler_product_zip(product, archive)
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


def cmd_unpack_compiler_product(args):
    try:
        archive = _find_compiler_product_archive(args.dir)
        product = extract_fastflowlm_archive(
            archive, COMPILER_PRODUCT_EXTRACTED, allowed_root=BUILD)
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


def cmd_package_npu_prereqs(args):
    """Stage exactly what npu_e2e executes and tar it (build/artifacts/npu-prereqs.tar.{zst,gz}
    — see _prereq_codec; npu_e2e reads it back via `dev.py unpack-npu-prereqs`).

    Not `tar -h`: staging with a dereferencing copy fails loudly on a dangling symlink,
    where tar would silently drop it."""
    it, hrxb = BUILD / "iron-tools", PROJ / "hrx" / "build" / "cmake"
    cts_src = hrxb.joinpath(*HRX_CTS_DIR)
    cts_rel = "/".join(HRX_CTS_DIR)
    stage = BUILD / "npu-stage"
    if stage.exists():
        shutil.rmtree(stage)

    for t in NPU_PREREQ_TOOLS:
        tool = _tool(it / "bin" / t)
        _stage(tool, stage / "build" / "iron-tools" / "bin" / tool.name)
    # include/ + aie_runtime_lib/ are the kernel-compile headers: iron-ops compiles its own
    # aie_kernels/*.cc on the NPU runner (`-I <root>/include -I <root>/aie_runtime_lib/AIE2P`,
    # see IRON's KernelCompilationRule), where the iron-tools examples ship a prebuilt design
    # and never needed them. 22 MB on a 1.2 GB stage.
    for d in ("python", "lib", "runtime_lib", "include", "aie_runtime_lib"):
        _stage(it / d, stage / "build" / "iron-tools" / d)

    # hrx: the runtime library plus exactly what build_xadx_helper.sh compiles and links
    # against (flatcc runtime + its headers + the generated runtime headers). The rest of
    # that build tree is build-time only.
    _stage(hrxb / "libhrx", stage / "projects/hrx/build/cmake/libhrx")
    flatcc = [f for f in hrxb.glob("*flatcc_runtime*") if f.is_file()]
    if not flatcc:
        sys.exit(f"flatcc_runtime not built in {hrxb} — run: python dev.py build hrx")
    for f in flatcc:
        _stage(f, stage / "projects/hrx/build/cmake" / f.name)
    _stage(hrxb / "_deps/flatcc-src/include",
           stage / "projects/hrx/build/cmake/_deps/flatcc-src/include")
    for h in (hrxb / "runtime" / "src").rglob("*.h"):
        _stage(h, stage / "projects/hrx/build/cmake" / h.relative_to(hrxb))

    # amdxdna HAL conformance suites; the binaries only, not the CMake scratch beside them.
    for t in HRX_CTS_TESTS:
        binary = _tool(cts_src / t)
        _stage(binary, stage / "projects/hrx/build/cmake" / cts_rel / binary.name)

    xclbinutil = _tool(ROOT / "tools-bin" / "xclbinutil")
    _stage(xclbinutil, stage / "tools-bin" / xclbinutil.name)

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


def _rehearse(python, pythonpath, hrx_dir, hrx_build):
    """Import the aie package the way the design build does, with the HRX backend forced.

    Three distinct failures all reach `import aie.iron` as the same message -- a missing
    libhrx, an hrxruntime the CMake SOURCES list never installed, and an xadx helper that
    will not compile. aie/utils/__init__.py catches every exception from its hrx probe and
    reports only "libhrx.so could not be located", and discovery.ensure_xadx_helper()
    swallows the compiler's CalledProcessError and returns None. So check each layer
    explicitly, cheapest and most specific first, before letting the import speak.

    HRX_DIR is the source checkout (the artifact ships only projects/hrx/build);
    HRX_BUILD is the artifact."""
    env = dict(os.environ, PYTHONPATH=str(pythonpath), HRX_DIR=str(hrx_dir),
               HRX_BUILD=str(hrx_build))

    # 1. The files discovery.py resolves by path. Names a packaging omission outright.
    for rel, why in [("libhrx/src/libhrx/libhrx.so", "discovery.find_libhrx()"),
                     ("libflatcc_runtime.a", "discovery.find_hrx_build()")]:
        if not (hrx_build / rel).is_file():
            sys.exit(f"artifact is missing {rel} under {hrx_build} (needed by {why})")
    if not (hrx_dir / "libhrx" / "include" / "hrx_runtime.h").is_file():
        sys.exit(f"{hrx_dir} is not an hrx source checkout (needed by build_xadx_helper.sh)")

    # 1b. dlopen it here: hrxruntime does this at import and the probe discards the reason,
    #     so a libhrx that gained a shared-library dependency the image lacks would be
    #     reported as "libhrx.so could not be located".
    libhrx = hrx_build / "libhrx/src/libhrx/libhrx.so"
    if not IS_WINDOWS:
        try:
            ctypes.CDLL(str(libhrx), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            sys.exit(f"{libhrx} is present but will not load: {e}")

    # 2. Build the xadx helper ourselves so the compiler's error is visible; hrxruntime
    #    would build it on import and discard the reason.
    pkg = pythonpath / "aie/utils/hostruntime/hrxruntime"
    helper, source = pkg / "build_xadx_helper.sh", pkg / "libironhrx_xadx.so"
    if not helper.is_file():
        sys.exit(f"artifact is missing {helper} (is patch 0003's CMakeLists hunk applied?)")
    if not source.exists() and not IS_WINDOWS:   # the helper is a POSIX shell script
        sh(["bash", str(helper)], env=dict(env, HRX_DIR=str(hrx_dir), HRX_BUILD=str(hrx_build)))

    # 3. Now the probe can only fail for reasons the layers above did not cover, and
    #    IRON_RUNTIME=auto lets aie.utils import so its DEBUG log reaches us.
    script = (
        "import logging, sys\n"
        "logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)\n"
        "import aie.utils as u\n"
        "from aie.utils.hostruntime.hrxruntime import discovery\n"
        "print('hrx_dir  :', discovery.find_hrx_dir())\n"
        "print('hrx_build:', discovery.find_hrx_build())\n"
        "print('libhrx   :', discovery.find_libhrx())\n"
        "print('xadx     :', discovery.find_xadx_helper())\n"
        "assert u.has_hrx, 'hrx backend unavailable; see the DEBUG line above for why'\n"
    )
    sh([python, "-c", script], env=dict(env, IRON_RUNTIME="auto"))

    # 4. Finally the real thing: the design build's import, with the backend forced.
    sh([python, "-c", "import aie.ir, aie.iron\nprint('rehearsal ok:', aie.ir.__file__)\n"],
       env=dict(env, IRON_RUNTIME="hrx"))


def cmd_verify_npu_prereqs(args):
    """Assert the run-prerequisites are self-contained, off-device.

    --extracted checks a tree the consumer already unpacked; the default checks the
    archive and then rehearses npu_e2e's imports from it."""
    hrxruntime = "build/iron-tools/python/aie/utils/hostruntime/hrxruntime"
    required = ["build/iron-tools/python/aie/ir.py",
                f"{hrxruntime}/discovery.py",
                # build_xadx_helper.sh compiles iron_hrx_xadx.cpp against these; discovery
                # resolves libhrx.so and the flatcc archive by path, so a packaging omission
                # here surfaces as an unrelated "libhrx.so could not be located".
                f"{hrxruntime}/build_xadx_helper.sh",
                f"{hrxruntime}/iron_hrx_xadx.cpp",
                "projects/hrx/build/cmake/libhrx/src/libhrx/libhrx.so",
                "projects/hrx/build/cmake/libflatcc_runtime.a",
                # The kernel-compile headers test-npu-iron-ops compiles aie_kernels/*.cc
                # against. One header from each of the two -I roots: without them Peano
                # fails on the NPU runner with a missing-include error that says nothing
                # about packaging. See cmd_package_npu_prereqs.
                "build/iron-tools/include/aie_api/aie.hpp",
                "build/iron-tools/aie_runtime_lib/AIE2P/vec_math.h"]
    if args.extracted:
        # `find`/rglob over a missing root yields nothing, so assert the roots exist
        # before concluding anything from the absence of symlinks.
        roots = [BUILD / "iron-tools", PROJ / "hrx" / "build", ROOT / "tools-bin"]
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
        print(f"verify-npu-prereqs: ok (no symlinks under {len(roots)} roots)")
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
    _rehearse(venv_py(), rehearsal / "build" / "iron-tools" / "python",
              PROJ / "hrx", rehearsal / "projects" / "hrx" / "build" / "cmake")
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


def _windows_npu_info():
    """Returns every amdxdna MCDM device without opening or dispatching it."""
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not powershell:
        return None, "PowerShell was not found"
    script = r"""
$ErrorActionPreference = 'Stop'
$candidates = @(Get-PnpDevice -PresentOnly |
  Where-Object {
    $_.Class -eq 'ComputeAccelerator' -and
    $_.InstanceId -like 'PCI\VEN_1022*'
  })
$devices = @($candidates | ForEach-Object {
$pnp = $_
$deviceId = $pnp.InstanceId
$service = (Get-PnpDeviceProperty -InstanceId $deviceId `
  -KeyName 'DEVPKEY_Device_Service').Data
if ($service -ne 'IpuMcdmDriver') { return }
$driverVersion = (Get-PnpDeviceProperty -InstanceId $deviceId `
  -KeyName 'DEVPKEY_Device_DriverVersion').Data
$driverDateValue = (Get-PnpDeviceProperty -InstanceId $deviceId `
  -KeyName 'DEVPKEY_Device_DriverDate').Data
$driverDate = if ($driverDateValue -is [DateTime]) {
  $driverDateValue.ToString('yyyy-MM-dd')
} else {
  [string]$driverDateValue
}
$driverProvider = (Get-PnpDeviceProperty -InstanceId $deviceId `
  -KeyName 'DEVPKEY_Device_DriverProvider').Data
$infName = (Get-PnpDeviceProperty -InstanceId $deviceId `
  -KeyName 'DEVPKEY_Device_DriverInfPath').Data
$systemDriver = Get-CimInstance Win32_SystemDriver -Filter "Name='$service'"
$driverPath = if ($systemDriver) { $systemDriver.PathName.Trim('"') } else { $null }
if ($driverPath -and $driverPath.StartsWith('\SystemRoot\')) {
  $driverPath = Join-Path $env:SystemRoot $driverPath.Substring(12)
}
[PSCustomObject]@{
  device_name = $pnp.FriendlyName
  device_id = $deviceId
  driver_version = [string]$driverVersion
  driver_date = $driverDate
  driver_provider = [string]$driverProvider
  inf_name = [string]$infName
  status = [string]$pnp.Status
  problem = [string]$pnp.Problem
  service = [string]$service
  driver_path = [string]$driverPath
}
})
if ($devices.Count -eq 0) { throw 'amdxdna MCDM device was not found' }
ConvertTo-Json -InputObject $devices -Compress
"""
    p = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if p.returncode != 0:
        detail = p.stderr.strip() or p.stdout.strip() or f"exit code {p.returncode}"
        return None, detail
    try:
        return parse_device_query(p.stdout), None
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"could not parse PowerShell device query: {e}"


def _windows_driver_firmware_packages(driver_path):
    return driver_firmware_packages(driver_path)


def _print_windows_npu_driver_info():
    print("== amdxdna driver / firmware ==")
    devices, error = _windows_npu_info()
    if error:
        print(f"  [MISS] Windows NPU device query failed: {error}")
        return 1

    all_healthy = True
    for index, info in enumerate(devices):
        prefix = f"device {index}: " if len(devices) > 1 else "device: "
        print(f"  [ok ] {prefix}{info['device_name']}")
        print(f"  [info] device id: {info['device_id']}")
        print(f"  [info] driver: {info['driver_version']} ({info['driver_provider']}, "
              f"{info['driver_date']}, {info['inf_name']})")
        print(f"  [info] service: {info['service']} ({info['driver_path']})")
        healthy = device_is_healthy(info)
        all_healthy &= healthy
        state = "ok " if healthy else "MISS"
        print(f"  [{state}] PnP status: {info['status']} / {info['problem']}")

        packages = _windows_driver_firmware_packages(info.get("driver_path"))
        if packages:
            print("  [info] driver-package firmware candidates:")
            for package in packages:
                if error := package.get("error"):
                    print(f"    {package['name']}: unreadable ({error})")
                else:
                    print(f"    {package['name']}: release {package['release']}, "
                          f"sha256 {package['sha256']}")
            print("  [note] Windows PnP does not expose which firmware candidate the KMD loaded")
        else:
            print("  [note] no firmware package metadata found beside the active KMD")
    return 0 if all_healthy else 1


def _probe_libhrx(bundle=None):
    lib_path = _libhrx_path(bundle)
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
        rc = _print_windows_npu_driver_info()
        print("== NPU device / libhrx probe ==")
        rc |= _probe_libhrx(args.bundle)
        print("npu-probe: " + ("ok" if rc == 0 else "FAIL"))
        return rc

    rc = 0
    _print_npu_driver_info()

    print("== NPU device / libhrx probe ==")
    devnode_rc, devnodes = _probe_npu_devnodes()
    rc |= devnode_rc

    if not devnodes:
        print("  [MISS] skipping hrx_gpu_initialize because no usable NPU devnode was found")
        print("npu-probe: FAIL")
        return 1
    rc |= _probe_libhrx(args.bundle)

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
    missing = [n for n in names if not (cts / n).exists()]
    if missing:
        sys.exit(f"CTS binaries missing ({', '.join(missing)}) — run: python dev.py build hrx")
    failed, counts = [], {}
    for n in names:
        print(f"\n=== hrx CTS: {n} ===", flush=True)
        r = subprocess.run([str(cts / n)], capture_output=True, text=True)
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
    env["HRX_DIR"] = str(PROJ / "hrx")
    env["HRX_BUILD"] = str(PROJ / "hrx" / "build" / "cmake")
    env["IRON_RUNTIME"] = "hrx"
    # makefile-common reads NPU2, not DEVICE (`devicename ?= $(if $(filter 1,$(NPU2)),
    # npu2,npu)`). Building for the wrong device is accepted by the driver and then never
    # completes, so check the emitted TXN's DevGen rather than trusting the make flags.
    target = getattr(args, "device", None) or "npu2"
    sh(["make", "all", *NPU_TARGETS[target]["make"]], cwd=ddir, env=env)
    _require_txn_devgen(ddir / "build" / "insts.bin", target)

    if getattr(args, "chain", False):
        if not (ddir / "test_chain_hrx.py").exists():
            sys.exit(f"{design}: no chained testbench (patch 0003 only adds one for vector_scalar_add)")
        testbench = [venv_py(), "test_chain_hrx.py"]
    else:
        testbench = [venv_py(), ROOT / "tests" / "npu" / "single_dispatch.py"]
    sh(testbench + ["--xclbin", "build/final.xclbin", "--instr", "build/insts.bin",
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
    if not list((PROJ / "hrx" / "build" / "cmake").rglob("libhrx*.so")):
        sys.exit("libhrx.so not found — run: python dev.py build hrx")


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

    env = _iron_env(getattr(args, "compiler_prefix", None))
    env["HRX_DIR"] = str(PROJ / "hrx")
    env["HRX_BUILD"] = str(PROJ / "hrx" / "build" / "cmake")
    env["IRON_RUNTIME"] = "hrx"
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("profile", nargs="?", choices=sorted(DOCTOR_PROFILE_SUBMODULES),
                   default="compiler-stack", help="dependency closure to validate")
    s = sub.add_parser("setup")
    s.add_argument("--profile", choices=sorted(SETUP_PROFILE_SUBMODULES),
                   default="compiler-stack", help="dependency closure to prepare")
    s.add_argument("--iron-ops-tests", action="store_true",
                   help="also install what iron-ops' own tests need (torch, pytest); "
                        "the NPU job does, the build jobs do not")
    sub.add_parser("fetch")
    b = sub.add_parser("build"); b.add_argument("what",
        choices=["llvm", "iron-tools", "hrx-xclbinutil", "hrx", "hrx-cts", "fastflowlm", "all"])
    b.add_argument("--fresh", action="store_true",
                   help="clear generated CMake state and outputs before configuring")
    t = sub.add_parser("test"); t.add_argument("design", nargs="?")
    t.add_argument("--compiler-prefix",
                   help="relocated rocm-npu-compiler product prefix to test")
    t.add_argument("design_args", nargs="*", help="extra aieargs for the design script")
    tf = sub.add_parser("test-fastflowlm")
    tf.add_argument("--bundle", help="run only from an extracted runtime bundle")
    tf.add_argument(
        "--inference-iterations",
        type=argparse_positive_int,
        default=1,
        metavar="N",
        help="run N independent FLM inference processes before the benchmark",
    )
    to = sub.add_parser("test-iron-ops"); to.add_argument("operator", nargs="?",
        help="operator to build (default: every one in tests/iron-ops/suite.py)")
    to.add_argument("--compiler-prefix",
                    help="relocated rocm-npu-compiler product prefix to test")
    to.add_argument("--device", choices=sorted(NPU_TARGETS),
                    help="override the NPU generation the suite pins for the operator")
    sub.add_parser("smoke")
    sm = sub.add_parser("submodules"); sm.add_argument("names", nargs="+")
    sm.add_argument("--reset", action="store_true", help="force-reset tracked files")
    sm.add_argument("--clean", nargs="*", default=[],
                    help="also remove untracked+ignored files (needed where a patch adds them)")
    sm.add_argument("--shallow", action="store_true", help="--depth 1 --jobs 4")
    sub.add_parser("package-npu-prereqs")
    sub.add_parser("package-fastflowlm")
    sub.add_parser("package-compiler-product")
    uf = sub.add_parser("unpack-fastflowlm")
    uf.add_argument("--dir", default=".artifact", help="directory containing the downloaded archive")
    vf = sub.add_parser("verify-fastflowlm")
    vf.add_argument("--dir", default=str(FASTFLOWLM_EXTRACTED),
                    help="extracted bundle root or its parent directory")
    uc = sub.add_parser("unpack-compiler-product")
    uc.add_argument("--dir", default=".artifact", help="directory containing the downloaded archive")
    vc = sub.add_parser("verify-compiler-product")
    vc.add_argument("--dir", default=str(COMPILER_PRODUCT_STAGE),
                    help="compiler product root or its parent directory")
    vc.add_argument("--relocate", action="store_true",
                    help="copy the product to another prefix and run the off-device e2e")
    u = sub.add_parser("unpack-npu-prereqs")
    u.add_argument("--dir", default=".artifact", help="where download-artifact put it")
    v = sub.add_parser("verify-npu-prereqs")
    v.add_argument("--extracted", action="store_true",
                   help="check an already-unpacked tree instead of the archive")
    np = sub.add_parser("npu-probe")
    np.add_argument("--bundle", help="probe libHRX from an extracted runtime bundle")
    c = sub.add_parser("test-hrx-cts"); c.add_argument("suite", nargs="?", choices=HRX_CTS_TESTS)
    tno = sub.add_parser("test-npu-iron-ops"); tno.add_argument("operator", nargs="?",
        help="operator to run on the NPU (default: every one in tests/iron-ops/suite.py)")
    tno.add_argument("--compiler-prefix",
                     help="relocated rocm-npu-compiler product prefix to test")
    tn = sub.add_parser("test-npu"); tn.add_argument("design", nargs="?")
    tn.add_argument("--compiler-prefix",
                    help="relocated rocm-npu-compiler product prefix to test")
    tn.add_argument("--device", choices=sorted(NPU_TARGETS), default="npu2",
                    help="NPU generation to build the design for (default: npu2/Strix)")
    tn.add_argument("--chain", action="store_true",
                    help="run the chained testbench instead of the single-dispatch gate")
    sub.add_parser("deps")
    a = p.parse_args()
    if a.cmd == "doctor": sys.exit(cmd_doctor(a))
    elif a.cmd == "setup": cmd_setup(a)
    elif a.cmd == "fetch": cmd_fetch(a)
    elif a.cmd == "deps": cmd_deps(a)
    elif a.cmd == "test": cmd_test(a)
    elif a.cmd == "test-fastflowlm": cmd_test_fastflowlm(a)
    elif a.cmd == "test-iron-ops": cmd_test_iron_ops(a)
    elif a.cmd == "test-npu-iron-ops": cmd_test_npu_iron_ops(a)
    elif a.cmd == "smoke": sys.exit(cmd_smoke(a))
    elif a.cmd == "submodules": cmd_submodules(a)
    elif a.cmd == "package-npu-prereqs": cmd_package_npu_prereqs(a)
    elif a.cmd == "package-fastflowlm": cmd_package_fastflowlm(a)
    elif a.cmd == "package-compiler-product": cmd_package_compiler_product(a)
    elif a.cmd == "unpack-fastflowlm": cmd_unpack_fastflowlm(a)
    elif a.cmd == "verify-fastflowlm": cmd_verify_fastflowlm(a)
    elif a.cmd == "unpack-compiler-product": cmd_unpack_compiler_product(a)
    elif a.cmd == "verify-compiler-product": cmd_verify_compiler_product(a)
    elif a.cmd == "unpack-npu-prereqs": cmd_unpack_npu_prereqs(a)
    elif a.cmd == "verify-npu-prereqs": sys.exit(cmd_verify_npu_prereqs(a))
    elif a.cmd == "npu-probe": sys.exit(cmd_npu_probe(a))
    elif a.cmd == "test-hrx-cts": sys.exit(cmd_test_hrx_cts(a))
    elif a.cmd == "test-npu": cmd_test_npu(a)
    elif a.cmd == "build":
        {"llvm": cmd_build_llvm, "iron-tools": cmd_build_iron_tools,
         "hrx-xclbinutil": cmd_build_hrx_xclbinutil, "hrx": cmd_build_hrx,
         "hrx-cts": cmd_build_hrx_cts, "fastflowlm": cmd_build_fastflowlm,
         "all": cmd_build_all}[a.what](a)


if __name__ == "__main__":
    main()
