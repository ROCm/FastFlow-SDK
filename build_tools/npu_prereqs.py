# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NPU runtime prerequisite artifact packaging orchestration."""

import contextlib
import ctypes
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from build_tools import hrx_runtime
from build_tools.dev import build, environment, paths

ROOT = paths.ROOT
PROJ = paths.PROJ
BUILD = paths.BUILD
HRX_INSTALL = build.HRX_INSTALL
NPU_PREREQ_DIR = paths.BUILD / "artifacts"
IS_WINDOWS = paths.is_windows()
_EXE = ".exe" if IS_WINDOWS else ""
sh = environment.sh
venv_py = environment.venv_py
_add_hrx_install_env = hrx_runtime.add_hrx_install_env



def archive_codec():
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
    return NPU_PREREQ_DIR / f"npu-prereqs.tar.{archive_codec()}"


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
    """Return the platform tool path, failing when it is absent."""
    for candidate in (path, path.with_suffix(".exe")):
        if candidate.exists():
            return candidate
    sys.exit(f"missing from the build tree: {path}")


def _stage(source, destination):
    """Copy dereferenced so the artifact never contains source-tree symlinks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(
            source, destination, symlinks=False, dirs_exist_ok=True
        )
    else:
        shutil.copy2(source, destination, follow_symlinks=True)


def _strip(stage, cts_rel):
    """Strip runnable payload binaries when the host provides strip.

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


def _stage_npu_runtime_prereqs(stage, hrx_install, cts_src):
    """Stage the runtime-only payload not already in the compiler product."""
    _stage(hrx_install, stage / "build/hrx-install")
    cts_rel = "/".join(build.HRX_CTS_DIR)
    for test_name in build.HRX_CTS_TESTS:
        binary = _tool(cts_src / test_name)
        _stage(binary, stage / "projects/hrx/build/cmake" / cts_rel / binary.name)


def cmd_package_npu_prereqs(args):
    """Stage and archive NPU runtime prerequisites not in the compiler product.

    Not `tar -h`: staging with a dereferencing copy fails loudly on a dangling symlink,
    where tar would silently drop it."""
    hrxb = PROJ / "hrx" / "build" / "cmake"
    cts_src = hrxb.joinpath(*build.HRX_CTS_DIR)
    cts_rel = "/".join(build.HRX_CTS_DIR)
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
    cts_rel = "/".join(build.HRX_CTS_DIR)
    cts_root = f"projects/hrx/build/cmake/{cts_rel}"
    required = [
        ("build/hrx-install/bin/hrx.dll"
         if IS_WINDOWS else
         "build/hrx-install/lib/libhrx.so"),
        "build/hrx-install/include/hrx/hrx_runtime.h",
        "build/hrx-install/include/hrx/hrx_amdxdna.h",
        *(f"{cts_root}/{name}{_EXE}" for name in build.HRX_CTS_TESTS),
    ]
    if args.extracted:
        # `find`/rglob over a missing root yields nothing, so assert the roots exist
        # before concluding anything from the absence of symlinks.
        roots = [BUILD / "hrx-install", PROJ.joinpath(
            "hrx", "build", "cmake", *build.HRX_CTS_DIR)]
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
    choose. Keeping both in this module means they read the same archive_codec()."""
    src = Path(args.dir)
    archive = _find_prereq_archive(src)
    if archive is None:
        sys.exit(f"no npu-prereqs.tar.* under {src} — did the download-artifact step run?")
    with _prereq_tar(archive) as tar:
        tar.extractall(ROOT, filter="data")
    print(f"unpack-npu-prereqs: ok ({archive.name} -> {ROOT})")
