#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""rocm-npu-staging developer command router.

Builds the NPU stack together (everything except llvm-aie, which comes
from the Peano wheel). Mirrors the command-router shape of hrx's dev.py.

  python dev.py doctor                # check system (apt) prereqs + pins + prefetch
  python dev.py setup                 # venv + python deps (incl. pip cmake/ninja) + fetch
                                     #   --iron-ops-tests: + what iron-ops' own tests need
                                     #     (torch, pytest); the NPU job only
  python dev.py fetch                 # apply temp patches + prefetch eudsl (verified) — offline-ready
  python dev.py build llvm            # LLVM+MLIR from projects/llvm-project (amd-llvm pin)
  python dev.py build iron-tools      # aie-opt/aie-translate against build/llvm
  python dev.py build hrx-xclbinutil  # the xclbin packager
  python dev.py build hrx             # the amdxdna HAL (IREE runtime)
  python dev.py build hrx-cts         # amdxdna HAL conformance binaries (need an NPU to run)
  python dev.py build fastflowlm      # HRX amdxdna + FastFlowLM app only
  python dev.py build all             # everything above, in dependency order
  python dev.py test [DESIGN] [-- ...] # e2e build test (no NPU run); Linux + Windows
  python dev.py test-iron-ops [OP]    # build the iron-ops operators' xclbins (no NPU run)
                                     #   Linux + Windows; suite: tests/iron-ops/suite.py
  python dev.py submodules NAME...    # init (and optionally reset/clean) submodules
  python dev.py package-npu-prereqs   # stage + tar what the NPU job needs (no NPU)
  python dev.py unpack-npu-prereqs    # extract that archive (codec-agnostic; no NPU)
  python dev.py verify-npu-prereqs    # assert that archive is self-contained + imports (no NPU)
  python dev.py npu-probe             # Linux: print NPU driver/fw versions + check libhrx open
  python dev.py test-hrx-cts [SUITE]  # amdxdna HAL conformance suites (needs an NPU)
  python dev.py test-npu [DESIGN]     # ON-DEVICE e2e: one dispatch of a design via hrx (needs an NPU)
                                     #   --device npu2|npu1   --chain (opt-in chained testbench)
  python dev.py test-fastflowlm       # pull qwen3:0.6b + run a 1k HRX benchmark
  python dev.py test-npu-iron-ops [OP] # ON-DEVICE e2e: an iron-ops operator, numerics checked (needs an NPU)
  python dev.py deps                  # print the dependency overview

Build-only: running on the NPU needs the amdxdna driver + an NPU (not here).
"""
import argparse, contextlib, ctypes, os, shutil, stat, subprocess, sys, tarfile
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
HRX_CMAKE_BUILD = PROJ / "hrx" / "build" / "cmake"
FASTFLOWLM_BUILD = BUILD / "FastFlowLM"
FASTFLOWLM_INSTALL = BUILD / "FastFlowLM-install"
FASTFLOWLM_MODEL_CACHE = BUILD / "FastFlowLM-models"
PATCHES = ROOT / "patches"                      # temporary submodule patches (see patches/README.md)
FETCH_CACHE = ROOT / ".cache" / "find-links"   # prefetched, hash-verified sdists (gitignored)
NPROC = str(os.cpu_count() or 8)
BUILD_JOBS = os.environ.get("RNS_BUILD_JOBS") or NPROC
FASTFLOWLM_MIN_RUSTC = (1, 80, 0)
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


def _rustc_version():
    rustc = shutil.which("rustc")
    if not rustc:
        return None
    r = subprocess.run([rustc, "--version"], capture_output=True, text=True)
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
                  "hrx_xclbinutil": "hrx-xclbinutil", "fastflowlm": "FastFlowLM",
                  "iron_ops": "iron-ops"}
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
    install_iron_ops(pip, with_tests=getattr(args, "iron_ops_tests", False))
    print("setup: ok (deps + prefetched build-time pkgs; everything else builds from source)")


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


def cmd_build_hrx(args):
    d = PROJ / "hrx"
    cc, cxx = _compilers()
    if not (d / "CMakeLists.txt").exists():
        sys.exit("missing projects/hrx (run: git submodule update --init projects/hrx)")
    apply_patches({"hrx"})

    # Configure HRX directly with Ninja. This keeps Windows off the Visual Studio
    # generator default and lets staging apply its temporary HRX patches first.
    sh([
        cmake_bin(), "-G", "Ninja",
        "-S", cmake_path(d), "-B", cmake_path(HRX_CMAKE_BUILD),
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DIREE_HAL_DRIVER_AMDGPU=OFF",
        "-DIREE_HAL_DRIVER_AMDXDNA=ON",
        # Declare test targets so `build hrx-cts` can compile them later, but
        # the normal HRX build below only asks CMake for the runtime targets.
        "-DIREE_BUILD_TESTS=ON",
        "-DLIBHRX_BUILD_CTS=OFF",
        "-DHRX_INSTALL_TESTS=OFF",
        "-DLIBHRX_BUILD_PASSTHROUGH=OFF",
        "-DLOOM_BUILD=OFF",
        # libhrx currently links local-task support unconditionally; keep it enabled
        # until HRX makes that dependency conditional.
        "-DIREE_HAL_DRIVER_LOCAL_SYNC=OFF",
        "-DIREE_HAL_DRIVER_NULL=OFF",
        *_cc_flags(cc, cxx), *_ccache_flags(),
    ], env=cmake_env())
    sh([
        cmake_bin(), "--build", cmake_path(HRX_CMAKE_BUILD),
        "--target", "iree_hal_drivers_amdxdna_registration_registration", "hrx", "flatcc_runtime",
        "-j", BUILD_JOBS,
    ], env=cmake_env())
    print("build hrx (minimal amdxdna HAL + libhrx): ok ->", HRX_CMAKE_BUILD)


def cmd_build_hrx_cts(args):
    """The amdxdna HAL conformance binaries. Separate from `build hrx` (and from
    `build all`) because they are only useful on a machine with an NPU; `build hrx`
    configures with IREE_BUILD_TESTS=ON, which merely declares the targets."""
    if not (HRX_CMAKE_BUILD / "CMakeCache.txt").exists():
        sys.exit("hrx not configured — run: python dev.py build hrx")
    sh([cmake_bin(), "--build", cmake_path(HRX_CMAKE_BUILD),
        *[f"--target=iree_hal_drivers_amdxdna_cts_{t}" for t in HRX_CTS_TESTS], "-j", BUILD_JOBS],
       env=cmake_env())
    print(f"build hrx-cts: ok ({', '.join(HRX_CTS_TESTS)})")


def cmd_build_fastflowlm(args):
    """Build the pinned FastFlowLM app against `dev.py build hrx` output.
    This intentionally skips LLVM, iron-tools, and hrx-xclbinutil."""
    if IS_WINDOWS:
        sys.exit("build fastflowlm is currently wired for the Linux HRX smoke path")
    flm_src = PROJ / "FastFlowLM" / "src"
    if not (flm_src / "CMakeLists.txt").exists():
        sys.exit("missing projects/FastFlowLM (run: git submodule update --init --recursive projects/FastFlowLM)")
    if not _have_header("libdrm/drm.h"):
        sys.exit("missing libdrm/drm.h: FastFlowLM's amdxdna headers require libdrm "
                 "(Ubuntu: sudo apt-get install libdrm-dev; Fedora: sudo dnf install libdrm-devel)")
    if not shutil.which("cargo"):
        sys.exit("missing cargo: FastFlowLM's tokenizers-cpp build requires Rust/Cargo "
                 "(install a current toolchain with rustup)")
    rustc_version = _rustc_version()
    if rustc_version is None or rustc_version < FASTFLOWLM_MIN_RUSTC:
        got = ".".join(str(n) for n in rustc_version) if rustc_version else "missing/unknown"
        need = ".".join(str(n) for n in FASTFLOWLM_MIN_RUSTC)
        sys.exit(f"rustc >= {need} required for FastFlowLM's tokenizers-cpp build "
                 f"(found {got}; install/update a current toolchain with rustup)")
    apply_patches({"FastFlowLM"})
    if not (HRX_CMAKE_BUILD / "libhrx" / "src" / "libhrx" / "libhrx.so").exists():
        sys.exit("missing libhrx.so (run: python dev.py build hrx)")
    if not (HRX_CMAKE_BUILD / "libflatcc_runtime.a").exists():
        sys.exit("missing libflatcc_runtime.a (run: python dev.py build hrx)")

    cc, cxx = _compilers()
    sh([
        cmake_bin(), "-G", "Ninja",
        "-S", cmake_path(flm_src), "-B", cmake_path(FASTFLOWLM_BUILD),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DFLM_VERSION=0.9.45",
        "-DNPU_VERSION=32.0.203.304",
        f"-DHRX_DIR={cmake_path(PROJ / 'hrx')}",
        f"-DHRX_BUILD={cmake_path(HRX_CMAKE_BUILD)}",
        f"-DHRX_LIB={cmake_path(HRX_CMAKE_BUILD / 'libhrx' / 'src' / 'libhrx' / 'libhrx.so')}",
        f"-DHRX_FLATCC_LIB={cmake_path(HRX_CMAKE_BUILD / 'libflatcc_runtime.a')}",
        f"-DCMAKE_INSTALL_PREFIX={cmake_path(FASTFLOWLM_INSTALL)}",
        f"-DCMAKE_XCLBIN_PREFIX={cmake_path(flm_src)}",
        *_cc_flags(cc, cxx),
    ], env=cmake_env())
    sh([cmake_bin(), "--build", cmake_path(FASTFLOWLM_BUILD), "-j", BUILD_JOBS], env=cmake_env())
    print("build fastflowlm: ok ->", flm_src / "build" / f"flm{_EXE}")


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


def cmd_test_fastflowlm(args):
    """Run the on-device FastFlowLM HRX smoke test."""
    script = ROOT / "tests" / "fastflowlm" / "run_fastflowlm_hrx_smoke.sh"
    env = cmake_env()
    env.setdefault("HRX_BUILD", str(HRX_CMAKE_BUILD))
    env.setdefault("FLM_MODEL_PATH", str(FASTFLOWLM_MODEL_CACHE))
    sys.exit(sh(["bash", script], env=env, check=False).returncode)
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
    sys.exit(sh(cmd, env=_iron_env(), check=False).returncode)


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


def _iron_env():
    """The IRON toolchain env, from its single definition in tests/e2e/run_e2e.py.

    That module owns PATH/PYTHONPATH/PEANO_INSTALL_DIR/AIEOPT_DIR and the "is iron-tools
    built?" check, and it puts tools-bin first on PATH so aiecc's bare `xclbinutil`
    resolves to hrx-xclbinutil. Imported by path rather than duplicated here: the old
    build_env() copy drifted, which is why it was removed."""
    sys.path.insert(0, str(ROOT / "tests" / "e2e"))
    import run_e2e
    try:
        return run_e2e.toolchain_env(run_e2e.require_xclbinutil())
    except run_e2e.E2EError as e:
        sys.exit(str(e))


def _libhrx_path():
    hits = sorted((PROJ / "hrx" / "build" / "cmake").rglob("libhrx*.so"))
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


def cmd_npu_probe(args):
    """Print Linux NPU driver/firmware versions and check that libhrx can open the
    NPU. Run before test-npu/test-hrx-cts so a missing or busy device is not
    reported as a dispatch failure. Needs no artifact beyond libhrx.so."""
    if IS_WINDOWS:
        print("npu-probe: unsupported on Windows")
        print("  This probe reads Linux amdxdna sysfs and DRM accel nodes:")
        print(f"  - {AMDXDNA_MODULE_DIR}")
        print(f"  - {SYS_ACCEL_DIR}")
        print(f"  - {ACCEL_DIR}")
        print("  Windows support would need a separate XRT/driver query path.")
        return 1

    rc = 0
    _print_npu_driver_info()

    print("== NPU device / libhrx probe ==")
    devnode_rc, devnodes = _probe_npu_devnodes()
    rc |= devnode_rc

    lib_path = _libhrx_path()
    if not lib_path:
        print("  [MISS] libhrx.so — run: python dev.py build hrx")
        return 1
    print(f"  [ok ] libhrx: {lib_path}")
    if not devnodes:
        print("  [MISS] skipping hrx_gpu_initialize because no usable NPU devnode was found")
        print("npu-probe: FAIL")
        return 1

    lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
    _bind_libhrx(lib)
    ok, msg = _hrx_check(lib, lib.hrx_gpu_initialize(0), "hrx_gpu_initialize")
    if not ok:
        print(f"  [MISS] {msg}")
        return 1
    print("  [ok ] hrx_gpu_initialize")

    count = ctypes.c_int(0)
    ok, msg = _hrx_check(lib, lib.hrx_gpu_device_count(ctypes.byref(count)), "hrx_gpu_device_count")
    if not ok:
        print(f"  [MISS] {msg}")
        rc = 1
    else:
        print(f"  [{'ok ' if count.value else 'MISS'}] hrx gpu devices: {count.value}")
        if count.value == 0:
            rc = 1
    lib.hrx_status_ignore(lib.hrx_gpu_shutdown())

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
    _require_hrx_runtime()

    env = _iron_env()
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


def _require_hrx_runtime():
    """The two things IRON_RUNTIME=hrx needs, named separately so the failure says which."""
    if not (BUILD / "iron-tools" / "python" / "aie" / "utils" / "hostruntime"
            / "hrxruntime" / "discovery.py").exists():
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
    _require_hrx_runtime()
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

    env = _iron_env()
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    s = sub.add_parser("setup")
    s.add_argument("--iron-ops-tests", action="store_true",
                   help="also install what iron-ops' own tests need (torch, pytest); "
                        "the NPU job does, the build jobs do not")
    sub.add_parser("fetch")
    b = sub.add_parser("build"); b.add_argument("what",
        choices=["llvm", "iron-tools", "hrx-xclbinutil", "hrx", "hrx-cts", "fastflowlm", "all"])
    t = sub.add_parser("test"); t.add_argument("design", nargs="?")
    t.add_argument("design_args", nargs="*", help="extra aieargs for the design script")
    sub.add_parser("test-fastflowlm")
    to = sub.add_parser("test-iron-ops"); to.add_argument("operator", nargs="?",
        help="operator to build (default: every one in tests/iron-ops/suite.py)")
    to.add_argument("--device", choices=sorted(NPU_TARGETS),
                    help="override the NPU generation the suite pins for the operator")
    sub.add_parser("smoke")
    sm = sub.add_parser("submodules"); sm.add_argument("names", nargs="+")
    sm.add_argument("--reset", action="store_true", help="force-reset tracked files")
    sm.add_argument("--clean", nargs="*", default=[],
                    help="also remove untracked+ignored files (needed where a patch adds them)")
    sm.add_argument("--shallow", action="store_true", help="--depth 1 --jobs 4")
    sub.add_parser("package-npu-prereqs")
    u = sub.add_parser("unpack-npu-prereqs")
    u.add_argument("--dir", default=".artifact", help="where download-artifact put it")
    v = sub.add_parser("verify-npu-prereqs")
    v.add_argument("--extracted", action="store_true",
                   help="check an already-unpacked tree instead of the archive")
    sub.add_parser("npu-probe")
    c = sub.add_parser("test-hrx-cts"); c.add_argument("suite", nargs="?", choices=HRX_CTS_TESTS)
    tno = sub.add_parser("test-npu-iron-ops"); tno.add_argument("operator", nargs="?",
        help="operator to run on the NPU (default: every one in tests/iron-ops/suite.py)")
    tn = sub.add_parser("test-npu"); tn.add_argument("design", nargs="?")
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
