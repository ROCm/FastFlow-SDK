# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Locked native dependency policy for FastFlowLM on Linux and Windows."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

from build_tools import environment, paths


VCPKG_TRIPLETS = {
    "Linux": "x64-linux-rel",
    "Windows": "x64-windows-rel",
}
MIN_RUSTC = (1, 80, 0)
POLICY_MARKER = ".rns-vcpkg-policy.json"


def check_fastflowlm_linux_host(verbose: bool) -> tuple[bool, list[str]]:
    """Checks host tools and interfaces needed to build FastFlowLM dependencies."""
    missing = []
    cc, cxx = environment.compilers(windows=False)
    checks = [
        ("bash", bool(shutil.which("bash")), "bash"),
        ("git", bool(shutil.which("git")), "git"),
        ("make", bool(shutil.which("make")), "build-essential"),
        ("nasm", bool(shutil.which("nasm")), "nasm"),
        ("pkg-config", bool(shutil.which("pkg-config")), "pkg-config"),
        ("tar", bool(shutil.which("tar")), "tar"),
        ("unzip", bool(shutil.which("unzip")), "unzip"),
        (
            "curl or wget",
            bool(shutil.which("curl") or shutil.which("wget")),
            "curl",
        ),
        (
            "Perl IPC::Cmd",
            environment.have_perl_module("IPC::Cmd"),
            "libipc-cmd-perl",
        ),
        (
            "<linux/types.h>",
            environment.have_header("linux/types.h"),
            "linux-libc-dev",
        ),
        (
            "<libdrm/drm.h>",
            environment.have_header("libdrm/drm.h"),
            "libdrm-dev",
        ),
    ]
    if verbose:
        print(
            f"  [{'ok ' if cc else 'MISS'}] C/C++ compiler: "
            f"{(cc + '/' + cxx) if cc else 'need gcc/g++ (or set CC/CXX)'}"
        )
    if not cc:
        missing.append("build-essential")
    for label, ok, package in checks:
        if verbose:
            print(f"  [{'ok ' if ok else 'MISS'}] {label}")
        if not ok and package not in missing:
            missing.append(package)
    return not missing, missing


def check_fastflowlm_linux_packaging_tools(
    *, archive: bool = False, verbose: bool = False
) -> tuple[bool, list[str]]:
    """Checks Linux-only tools needed to package FastFlowLM artifacts."""
    checks = [("patchelf", bool(shutil.which("patchelf")), "patchelf")]
    if archive:
        checks.append(("zstd", bool(shutil.which("zstd")), "zstd"))
    missing = []
    for label, ok, package in checks:
        if verbose:
            print(f"  [{'ok ' if ok else 'MISS'}] {label}")
        if not ok and package not in missing:
            missing.append(package)
    return not missing, missing


def require_fastflowlm_linux_packaging_tools(
    *, archive: bool = False
) -> None:
    if paths.is_windows():
        return
    ok, missing = check_fastflowlm_linux_packaging_tools(archive=archive)
    if not ok:
        sys.exit(
            "FastFlowLM Linux packaging prerequisite(s) are missing; install: "
            "sudo apt-get install -y " + " ".join(sorted(set(missing)))
        )


def vcpkg_commit(repo_root: Path) -> str:
    """Returns the curated-registry baseline that also pins the vcpkg checkout."""
    manifest = json.loads(
        (repo_root / ".github/vcpkg/fastflowlm/vcpkg.json").read_text())
    commit = manifest.get("builtin-baseline", "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("FastFlowLM vcpkg manifest has no valid builtin-baseline")
    return commit


def vcpkg_triplet(system_name: str | None = None) -> str:
    system_name = system_name or platform.system()
    try:
        return VCPKG_TRIPLETS[system_name]
    except KeyError as exc:
        raise RuntimeError(f"FastFlowLM vcpkg is unsupported on {system_name}") from exc


def dependency_policy(repo_root: Path, system_name: str | None = None) -> dict[str, str]:
    """Returns the complete native-dependency policy and its stable identity."""
    system_name = system_name or platform.system()
    triplet = vcpkg_triplet(system_name)
    manifest_path = repo_root / ".github/vcpkg/fastflowlm/vcpkg.json"
    triplet_path = repo_root / ".github/vcpkg/triplets" / f"{triplet}.cmake"
    manifest = manifest_path.read_bytes()
    triplet_contents = triplet_path.read_bytes()
    identity = hashlib.sha256(
        b"rocm-npu-staging-fastflowlm-vcpkg-v1\0" + manifest + b"\0" +
        triplet_contents
    ).hexdigest()
    return {
        "identity": identity,
        "system": system_name,
        "triplet": triplet,
        "vcpkg_commit": vcpkg_commit(repo_root),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "triplet_sha256": hashlib.sha256(triplet_contents).hexdigest(),
    }


def dependency_paths(repo_root: Path, system_name: str | None = None) -> dict[str, Path | str]:
    """Returns the repository-managed, policy-isolated vcpkg installation."""
    policy = dependency_policy(repo_root, system_name)
    system_name = policy["system"]
    triplet = policy["triplet"]
    source = repo_root / ".cache" / "vcpkg-src"
    install_root = (
        repo_root / ".cache" / "vcpkg-installed" / policy["identity"][:16]
    )
    return {
        "system": system_name,
        "triplet": triplet,
        "policy": policy,
        "source": source,
        "executable": source / ("vcpkg.exe" if system_name == "Windows" else "vcpkg"),
        "toolchain": source / "scripts" / "buildsystems" / "vcpkg.cmake",
        "install_root": install_root,
        "triplet_root": install_root / triplet,
        "bin": install_root / triplet / "bin",
        "marker": install_root / POLICY_MARKER,
    }


def dependency_errors(
    paths: dict[str, Path | str], *, cargo_available: bool,
    rustc_version: tuple[int, ...] | None,
) -> list[str]:
    """Checks the capabilities supplied by the locked vcpkg installation."""
    missing = []
    if not cargo_available:
        missing.append("Rust/Cargo (install with rustup; rustc >= 1.80 required)")
    if rustc_version is None or rustc_version < MIN_RUSTC:
        got = ".".join(str(n) for n in rustc_version) if rustc_version else "missing/unknown"
        need = ".".join(str(n) for n in MIN_RUSTC)
        missing.append(f"rustc >= {need} (found {got})")

    required_paths = (
        Path(paths["executable"]),
        Path(paths["toolchain"]),
        Path(paths["triplet_root"]),
        Path(paths["install_root"]) / "vcpkg" / "status",
        Path(paths["marker"]),
    )
    if any(not path.exists() for path in required_paths):
        missing.append(
            f"complete managed vcpkg dependency tree ({paths['triplet_root']}); run "
            "`python dev.py setup --profile fastflowlm`"
        )
        return missing

    try:
        marker = json.loads(Path(paths["marker"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        marker = None
    if marker != paths["policy"]:
        missing.append(
            f"vcpkg policy marker mismatch ({paths['marker']}); run "
            "`python dev.py setup --profile fastflowlm`"
        )
    return missing
