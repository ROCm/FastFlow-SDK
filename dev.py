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
  python dev.py test [DESIGN]         # e2e build test (no NPU run)
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
LLVM_BUILD = BUILD / "llvm"
PATCHES = ROOT / "patches"                      # temporary submodule patches (see patches/README.md)
FETCH_CACHE = ROOT / ".cache" / "find-links"   # prefetched, hash-verified sdists (gitignored)
NPROC = str(os.cpu_count() or 8)

# Peano (AIE codegen) — pulled from the wheel, not built from llvm-aie source.
PEANO_WHEEL = "llvm-aie==21.0.0.2026061501+acc2a72c"
PEANO_INDEX = "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"


def sh(cmd, cwd=None, env=None, check=True):
    print(f"+ {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=check)


def venv_py():
    return VENV / "bin" / "python"


def site_packages():
    """Resolve the venv's site-packages without hardcoding the python version."""
    import glob
    hits = sorted(glob.glob(str(VENV / "lib" / "python3*" / "site-packages")))
    if hits:
        return Path(hits[-1])
    return VENV / "lib" / "python3" / "site-packages"  # fallback


def cmake_bin():
    """cmake from the venv (pip, >=3.30) if set up, else system cmake."""
    c = VENV / "bin" / "cmake"
    return str(c) if c.exists() else "cmake"


def cmake_env():
    """PATH with the venv bin first so the pip-provided cmake/ninja are used."""
    e = dict(os.environ)
    e["PATH"] = os.pathsep.join([str(VENV / "bin"), e.get("PATH", "")])
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


def build_env():
    """Env for the IRON flow: the SOURCE iron-tools toolchain (built against ROCm
    amd-llvm) + Peano (wheel) + hrx-xclbinutil. No wheel fallback for iron-tools."""
    e = dict(os.environ)
    src_bin = BUILD / "iron-tools" / "bin"
    if not (src_bin / "aie-opt").exists():
        sys.exit("iron-tools not built from source — run: python dev.py build iron-tools")
    e["PATH"] = os.pathsep.join(
        [str(ROOT / "tools-bin"), str(src_bin), str(VENV / "bin"), e.get("PATH", "")])
    e["PYTHONPATH"] = str(BUILD / "iron-tools" / "python")
    e["PEANO_INSTALL_DIR"] = str(site_packages() / "llvm-aie")   # Peano: the only wheel
    e["AIEOPT_DIR"] = str(PROJ / "iron-tools" / "third_party" / "aie_api")
    return e


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
    sec_to_sub = {"llvm": "llvm-project", "iron_tools": "iron-tools", "hrx": "hrx"}
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
    src = lock if lock.exists() else ROOT / "requirements.txt"   # lock = reproducible (hashes)
    sh(pip + ["-r", str(src)])
    cmd_fetch(args)   # group all network here: prefetch build-time pkgs so `build` is offline
    print("setup: ok (deps + prefetched build-time pkgs; everything else builds from source)")


def cmd_build_llvm(args):
    """LLVM+MLIR from the amd-llvm pin (mlir-aie's flags)."""
    OUT.mkdir(exist_ok=True)
    cc, cxx = _compilers()
    cfg = [
        cmake_bin(), "-G", "Ninja", "-S", PROJ / "llvm-project" / "llvm", "-B", LLVM_BUILD,
        "-DLLVM_ENABLE_PROJECTS=mlir", "-DLLVM_TARGETS_TO_BUILD=host",
        "-DLLVM_ENABLE_ASSERTIONS=ON", "-DLLVM_ENABLE_RTTI=OFF",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON", "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_INSTALL_UTILS=ON", "-DLLVM_CCACHE_BUILD=ON",
        f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DPython3_EXECUTABLE={venv_py()}",
    ]
    if "clang" in (cc or "") and shutil.which("ld.lld"):   # clang+lld; gcc uses its default linker
        cfg.append("-DLLVM_USE_LINKER=lld")
    sh(cfg, env=cmake_env())
    sh([cmake_bin(), "--build", LLVM_BUILD, "-j", NPROC], env=cmake_env())
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
        cmake_bin(), "-G", "Ninja", "-S", src, "-B", bdir, "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DLLVM_DIR={LLVM_BUILD/'lib'/'cmake'/'llvm'}",
        f"-DMLIR_DIR={LLVM_BUILD/'lib'/'cmake'/'mlir'}",
        f"-DCMAKE_MODULE_PATH={src/'cmake'/'modulesXilinx'}",
        # install the xaiengine runtime_lib into the build tree (co-located with bin/python),
        # not the default /usr/local (which needs root and breaks an unprivileged build).
        f"-DCMAKE_INSTALL_PREFIX={bdir}",
        "-DAIE_RUNTIME_TARGETS=x86_64", "-DLLVM_ENABLE_ASSERTIONS=ON", "-DLLVM_ENABLE_RTTI=OFF",
        "-DAIE_ENABLE_BINDINGS_PYTHON=ON",
        f"-DPython3_EXECUTABLE={venv_py()}",
    ]
    sh(cfg, env=env)
    sh([cmake_bin(), "--build", bdir, "-j", NPROC], env=env)
    print("build iron-tools: ok ->", bdir / "bin")


def cmd_build_hrx_xclbinutil(args):
    d = PROJ / "hrx-xclbinutil"
    sh([cmake_bin(), "-G", "Ninja", "-B", d / "bld", "-S", d, "-DCMAKE_BUILD_TYPE=Release"], env=cmake_env())
    sh([cmake_bin(), "--build", d / "bld", "--target", "hrx-xclbinutil", "-j", NPROC], env=cmake_env())
    # expose as `xclbinutil` for aiecc.py
    tb = ROOT / "tools-bin"; tb.mkdir(exist_ok=True)
    link = tb / "xclbinutil"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(d / "bld" / "tools" / "hrx-xclbinutil")
    print("build hrx-xclbinutil: ok ->", link)


def cmd_build_hrx(args):
    d = PROJ / "hrx"
    sh([sys.executable, "dev.py", "setup"], cwd=d)
    sh([sys.executable, "dev.py", "cmake", "setup"], cwd=d)
    sh([sys.executable, "dev.py", "cmake", "configure",
        "-DIREE_HAL_DRIVER_AMDXDNA=ON", "-DIREE_BUILD_TESTS=OFF"], cwd=d)
    sh(["cmake", "--build", d / "build" / "cmake",
        "--target", "iree_hal_drivers_amdxdna_registration_registration", "-j", NPROC])
    print("build hrx (amdxdna HAL): ok")


def cmd_build_all(args):
    cmd_build_hrx_xclbinutil(args)
    cmd_build_llvm(args)
    cmd_build_iron_tools(args)
    cmd_build_hrx(args)


def cmd_test(args):
    """e2e: build a design's xclbin + HAL PDI artifact-contract check. DESIGN is an
    iron-tools programming_examples/basic design; defaults to matrix_scalar_add (the
    validated one). Others must be Peano-capable and use the `jit_xclbin` rule —
    `jit_xclbin_elf` designs also emit insts.elf, which needs aiebu-asm (absent here)."""
    cmd = ["bash", ROOT / "tests" / "e2e" / "run_e2e.sh"]
    if getattr(args, "design", None):
        cmd.append(args.design)
    sh(cmd, env=build_env())


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
    aieopt = BUILD / "iron-tools" / "bin" / "aie-opt"
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
