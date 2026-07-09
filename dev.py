#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""rocm-npu-staging developer command router.

Builds the NPU stack together (everything except llvm-aie, which comes
from the Peano wheel). Mirrors the command-router shape of hrx's dev.py.

  python dev.py doctor                # check system (apt) prereqs + pins + prefetch
  python dev.py setup                 # venv + python deps (incl. pip cmake/ninja) + fetch
  python dev.py fetch                 # apply temp patches + prefetch eudsl (verified) — offline-ready
  python dev.py build llvm            # LLVM+MLIR from projects/llvm-project (amd-llvm pin)
  python dev.py build iron-tools      # aie-opt/aie-translate against build/llvm
  python dev.py build hrx-xclbinutil  # the xclbin packager
  python dev.py build hrx             # the amdxdna HAL (IREE runtime)
  python dev.py build all             # everything above, in dependency order
  python dev.py test [DESIGN] [-- ...] # e2e build test (no NPU run); Linux + Windows
  python dev.py deps                  # print the dependency overview

Build-only: running on the NPU needs the amdxdna driver + an NPU (not here).
"""
import argparse, os, subprocess, sys, shutil
from pathlib import Path

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
PATCHES = ROOT / "patches"                      # temporary submodule patches (see patches/README.md)
FETCH_CACHE = ROOT / ".cache" / "find-links"   # prefetched, hash-verified sdists (gitignored)
NPROC = str(os.cpu_count() or 8)
IS_WINDOWS = os.name == "nt"                    # venv layout, .exe suffix, and MSVC differ on Windows
_VBIN = "Scripts" if IS_WINDOWS else "bin"      # venv exe dir: POSIX bin/, Windows Scripts/
_EXE = ".exe" if IS_WINDOWS else ""

# Peano (AIE codegen) — pulled from the wheel, not built from llvm-aie source.
PEANO_WHEEL = "llvm-aie==21.0.0.2026061501+acc2a72c"
PEANO_INDEX = "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"


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


def cmake_env():
    """PATH with the venv bin first so the pip-provided cmake/ninja are used."""
    e = dict(os.environ)
    e["PATH"] = os.pathsep.join([str(VENV / _VBIN), e.get("PATH", "")])
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


def _have_header(hdr):
    if hdr == "Python.h":  # lives under the python include dir, not a default search path
        import sysconfig
        inc = sysconfig.get_path("include")
        return bool(inc) and (Path(inc) / "Python.h").exists()
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return False
    p = subprocess.run([cc, "-fsyntax-only", "-x", "c", "-"],
                       input=f"#include <{hdr}>\nint main(){{return 0;}}\n".encode(),
                       capture_output=True)
    return p.returncode == 0


def check_system(verbose=True):
    """Return (ok, missing_apt_packages). Reports tool/header presence."""
    missing = []
    cc, cxx = _compilers()
    if IS_WINDOWS:
        # Windows deps come from the VS toolchain + choco/vcpkg, not apt; _compilers()
        # returns None by design, so probe the toolchain directly. clang-cl is REQUIRED,
        # not merely preferred: MSVC cl rejects the GNU-only syntax in hrx-xclbinutil's
        # vendored sources (__attribute__((aligned))/((visibility)), __builtin_bswap64).
        # Finding only cl is a fail here rather than a confusing compile error later.
        # Relax this to "cl or clang-cl" once those five sites are made portable.
        has_clang_cl = bool(shutil.which("clang-cl"))
        has_git = bool(shutil.which("git"))
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
    """'applied' | 'appliable' | 'broken' for a patch against submodule dir `sub`."""
    rev = subprocess.run(["git", "-C", str(sub), "apply", "--reverse", "--check", str(patch)],
                         capture_output=True)
    if rev.returncode == 0:
        return "applied"
    fwd = subprocess.run(["git", "-C", str(sub), "apply", "--check", str(patch)],
                         capture_output=True)
    return "appliable" if fwd.returncode == 0 else "broken"


def apply_patches():
    """Apply the temporary submodule patches (idempotent). These live only in
    rocm-npu-staging and are upstreamable as-is; see patches/README.md."""
    if not PATCHES.exists():
        return
    for sub_dir in sorted(p for p in PATCHES.iterdir() if p.is_dir()):
        sub = PROJ / sub_dir.name
        for patch in sorted(sub_dir.glob("*.patch")):
            st = _patch_state(patch, sub)
            if st == "applied":
                print(f"patch: {sub_dir.name}/{patch.name} already applied")
            elif st == "appliable":
                sh(["git", "-C", str(sub), "apply", str(patch)])
                print(f"patch: applied {sub_dir.name}/{patch.name}")
            else:
                sys.exit(f"patch does not apply cleanly: {patch} (submodule moved? regenerate it)")


def cmd_fetch(args):
    """Prepare sources for an offline build (the single network step): apply temporary
    patches + prefetch the build-time-fetched eudsl sdist into a gitignored,
    hash-verified cache. Idempotent."""
    apply_patches()
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


def check_pins():
    """Verify each submodule's checked-out commit == manifests/versions.toml (the
    manifest is load-bearing: drift is reported)."""
    man = _manifest()
    sec_to_sub = {"llvm": "llvm-project", "iron_tools": "iron-tools", "hrx": "hrx",
                  "hrx_xclbinutil": "hrx-xclbinutil"}
    drift = []
    for sec, sub in sec_to_sub.items():
        pinned = man.get(sec, {}).get("commit")
        if not pinned:
            continue
        sub_dir = PROJ / sub
        if not (sub_dir / ".git").exists():   # not checked out — rev-parse here would walk up to the superproject
            print(f"  [absent] {sub}: not checked out (skipped)")
            continue
        r = subprocess.run(["git", "-C", str(sub_dir), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        head = r.stdout.strip()
        match = head == pinned
        print(f"  [{'ok  ' if match else 'DRIFT'}] {sub}: manifest {pinned[:12]} vs HEAD {(head[:12] or '?')}")
        if not match:
            drift.append(sub)
    return drift


def cmd_doctor(args):
    """Check system (apt) prereqs + that submodule checkouts match the manifest."""
    print("system prerequisites (cmake + ninja come from pip via `setup`):")
    _, missing = check_system()
    print("\nsubmodule pins vs manifests/versions.toml:")
    drift = check_pins()
    print("\ntemporary submodule patches (patches/, applied by `dev.py fetch`):")
    patch_bad = []
    for sub_dir in sorted(p for p in PATCHES.iterdir() if p.is_dir()) if PATCHES.exists() else []:
        for patch in sorted(sub_dir.glob("*.patch")):
            st = _patch_state(patch, PROJ / sub_dir.name)
            print(f"  [{'ok   ' if st == 'applied' else st.upper()}] {sub_dir.name}/{patch.name}")
            if st == "broken":
                patch_bad.append(patch.name)

    print("\nbuild-time prefetch (run `dev.py fetch` to populate):")
    eudsl_ok = _eudsl_cache_ok()
    print(f"  [{'ok  ' if eudsl_ok else 'MISS '}] eudsl sdist cached + sha256 verified")
    # cross-check: the eudsl version lives in BOTH the patch and versions.toml — guard drift
    import re
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
    if missing:
        print("\nMISSING system deps — install with:\n  sudo apt-get install -y " + " ".join(sorted(missing)))
        rc = 1
    if drift:
        print(f"\nPIN DRIFT in: {', '.join(drift)} — `git -C projects/<x> checkout <manifest commit>` or update versions.toml")
        rc = 1
    if not eudsl_ok:
        print("\neudsl not prefetched — run: python dev.py fetch")
        rc = 1
    if rc == 0:
        print("\nall good: prerequisites present, pins match the manifest, build-time pkgs prefetched.")
    return rc


def cmd_setup(args):
    """venv + all Python deps from the HASH-PINNED requirements.lock (reproducible).
    requirements.txt is the human-edited input; regenerate the lock after editing it:
      .venv/bin/uv pip compile requirements.txt --generate-hashes --emit-index-url \\
          --emit-find-links --python-version 3.12 -o requirements.lock
    iron-tools/LLVM/bootgen/aie_api/aie-rt build from source; eudsl is prefetched (fetch)."""
    OUT.mkdir(exist_ok=True)
    ok, missing = check_system(verbose=False)
    if missing:
        print("WARNING: missing system deps (apt): " + " ".join(sorted(missing)))
        print("  install them, then re-run; see `python dev.py doctor`")
    if not venv_py().exists():
        sh([sys.executable, "-m", "venv", str(VENV)])
    pip = [str(venv_py()), "-m", "pip", "install"]
    sh(pip + ["--upgrade", "pip"])
    lock = ROOT / "requirements.lock"
    # requirements.lock is hash-pinned for LINUX wheels; on Windows the platform-specific
    # wheels (llvm-aie, cmake, ...) have different sha256s, so those hashes can't match.
    # Use the unpinned requirements.txt there — same version pins + find-links, no hashes.
    if IS_WINDOWS:
        src = ROOT / "requirements.txt"
    else:
        src = lock if lock.exists() else ROOT / "requirements.txt"   # lock = reproducible (hashes)
    sh(pip + ["-r", str(src)])
    cmd_fetch(args)   # group all network here: prefetch build-time pkgs so `build` is offline
    print("setup: ok (deps + prefetched build-time pkgs; everything else builds from source)")


def _cc_flags(cc: str | None, cxx: str | None) -> list[str]:
    """CMake compiler flags. Empty when _compilers() found none — on Windows with CC/CXX
    unset, so CMake picks from the VS Developer environment rather than us hardcoding a
    compiler it may not have. On POSIX these pin gcc/clang exactly as before."""
    return [f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}"] if cc and cxx else []


def cmd_build_llvm(args):
    """LLVM+MLIR from the amd-llvm pin (mlir-aie's flags)."""
    OUT.mkdir(exist_ok=True)
    cc, cxx = _compilers()
    # ccache is OPTIONAL: LLVM_CCACHE_BUILD=ON is a FATAL_ERROR if ccache isn't found, so
    # only enable it when ccache is on PATH (Linux CI has it; a Windows runner may not).
    ccache = ["-DLLVM_CCACHE_BUILD=ON"] if shutil.which("ccache") else []
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
        "-DLLVM_INSTALL_UTILS=ON", *ccache,
        # Nothing in this stack consumes the LLVM/MLIR examples or benchmarks, and the
        # default target builds them. Dropping them cuts build time, and it removes
        # mlir/examples (gated on LLVM_INCLUDE_EXAMPLES) whose long target names are what
        # first pushed the Windows rc.exe output path past MAX_PATH. Utils (FileCheck,
        # lit, ...) are separate and still built via LLVM_INSTALL_UTILS above.
        "-DLLVM_INCLUDE_EXAMPLES=OFF", "-DLLVM_INCLUDE_BENCHMARKS=OFF",
        *_cc_flags(cc, cxx), *crt,
        f"-DPython3_EXECUTABLE={cmake_path(venv_py())}",
    ]
    # clang+lld; gcc uses its default linker. Skipped on Windows: "clang" matches
    # "clang-cl" there, but the MSVC-style driver picks its own linker (link/lld-link).
    if not IS_WINDOWS and "clang" in (cc or "") and shutil.which("ld.lld"):
        cfg.append("-DLLVM_USE_LINKER=lld")
    sh(cfg, env=cmake_env())
    sh([cmake_bin(), "--build", cmake_path(LLVM_BUILD), "-j", NPROC], env=cmake_env())
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
        *_cc_flags(cc, cxx),
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
        "-DCMAKE_BUILD_TYPE=Release"], env=cmake_env())
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


def cmd_build_hrx(args):
    d = PROJ / "hrx"
    cc, cxx = _compilers()
    # hrx's `dev.py cmake configure` forwards these to cmake verbatim and picks no
    # generator of its own, so on Windows CMake defaults to the Visual Studio generator.
    # That is wrong three ways: (1) MSBuild ignores CC/CXX, silently building with `cl`
    # instead of the clang-cl the rest of the stack uses; (2) it is multi-config, so an
    # unqualified `--build` produces Debug; (3) `--build -j N` under MSBuild races on the
    # per-target .pdb ("error C1041 ... please use /FS"). Ninja + an explicit build type
    # fixes all three, and matches hrx's own README (`cmake -S . -B ... -GNinja`).
    gen = ["-GNinja", "-DCMAKE_BUILD_TYPE=Release", *_cc_flags(cc, cxx)] if IS_WINDOWS else []
    sh([sys.executable, "dev.py", "setup"], cwd=d)
    sh([sys.executable, "dev.py", "cmake", "setup"], cwd=d)
    sh([sys.executable, "dev.py", "cmake", "configure", *gen,
        "-DIREE_HAL_DRIVER_AMDXDNA=ON", "-DIREE_BUILD_TESTS=OFF"], cwd=d)
    # cmake_bin()/cmake_env(): the pip cmake lives in the venv and is not on PATH, so a
    # bare `cmake` here finds the system one on Linux and nothing at all on Windows.
    sh([cmake_bin(), "--build", cmake_path(d / "build" / "cmake"),
        "--target", "iree_hal_drivers_amdxdna_registration_registration", "-j", NPROC],
       env=cmake_env())
    print("build hrx (amdxdna HAL): ok")


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
    if args.design:
        cmd.append(args.design)
    if args.design_args:
        cmd += ["--", *args.design_args]
    # No env= here: run_e2e.py builds the IRON toolchain env itself (see the note above
    # build_env's former home). run_e2e.py already prints "E2E FAIL: ..."; propagate its
    # status rather than let sh()'s check=True bury it under a CalledProcessError.
    sys.exit(sh(cmd, check=False).returncode)


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


def cmd_deps(args):
    """Print the in-repo dependency manifest (self-contained)."""
    print((ROOT / "requirements.txt").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    sub.add_parser("setup")
    sub.add_parser("fetch")
    b = sub.add_parser("build"); b.add_argument("what",
        choices=["llvm", "iron-tools", "hrx-xclbinutil", "hrx", "all"])
    t = sub.add_parser("test"); t.add_argument("design", nargs="?")
    t.add_argument("design_args", nargs="*", help="extra aieargs for the design script")
    sub.add_parser("smoke")
    sub.add_parser("deps")
    a = p.parse_args()
    if a.cmd == "doctor": sys.exit(cmd_doctor(a))
    elif a.cmd == "setup": cmd_setup(a)
    elif a.cmd == "fetch": cmd_fetch(a)
    elif a.cmd == "deps": cmd_deps(a)
    elif a.cmd == "test": cmd_test(a)
    elif a.cmd == "smoke": sys.exit(cmd_smoke(a))
    elif a.cmd == "build":
        {"llvm": cmd_build_llvm, "iron-tools": cmd_build_iron_tools,
         "hrx-xclbinutil": cmd_build_hrx_xclbinutil, "hrx": cmd_build_hrx,
         "all": cmd_build_all}[a.what](a)


if __name__ == "__main__":
    main()
