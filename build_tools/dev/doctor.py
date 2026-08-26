# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Developer host, source-pin, patch, and prefetch diagnostics."""

import os
import re
import shutil
import time
from pathlib import Path

from build_tools import environment, paths
from build_tools.dev import patches
from build_tools.dev.profiles import profile
from build_tools.fastflowlm_deps import (
    MIN_RUSTC as FASTFLOWLM_MIN_RUSTC,
    check_fastflowlm_linux_host,
    check_fastflowlm_linux_packaging_tools,
    dependency_errors as fastflowlm_dependency_errors,
    dependency_paths as fastflowlm_dependency_paths,
)

_SYS_HEADERS = [
    ("Python.h", "python3-dev"),
    ("openssl/opensslv.h", "libssl-dev"),
]

EUDSL_FIND_LINKS_PATCH = "0001-eudsl-offline-find-links.patch"


def eudsl_version_from_patch(text: str) -> str | None:
    """Version iron-tools installs after the eudsl find-links patch.

    A ``+eudsl-python-extras==`` hunk is an override. If the patch only rewrites
    ``-f``, the unchanged context line is mlir-aie's pin — that is in-sync with
    ``[eudsl]``.
    """
    plus = re.search(r"^\+eudsl-python-extras==(\S+)", text, re.M)
    if plus:
        return plus.group(1)
    context = re.search(r"^ eudsl-python-extras==(\S+)", text, re.M)
    if context:
        return context.group(1)
    return None


def check_compiler_stack_system(verbose, *, windows: bool | None = None):
    """Return (ok, missing_apt_packages) for the LLVM/IRON compiler stack."""
    missing = []
    selected_windows = paths.is_windows() if windows is None else windows
    cc, cxx = environment.compilers(windows=selected_windows)
    if selected_windows:
        env = environment.windows_msvc_env(windows=True)
        clang_cl = None
        for candidate in (cxx, cc, "clang-cl"):
            if not candidate:
                continue
            resolved = environment.which(candidate, env, windows=True)
            if resolved and Path(resolved).name.lower() in {
                "clang-cl",
                "clang-cl.exe",
            }:
                clang_cl = resolved
                break
        has_clang_cl = bool(clang_cl)
        has_git = bool(environment.which("git", env, windows=True))
        if verbose:
            print(
                f"  [{'ok ' if has_clang_cl else 'MISS'}] clang-cl "
                "(required; MSVC cl cannot build hrx-xclbinutil)"
            )
            if not has_clang_cl and shutil.which("cl"):
                print(
                    "         found MSVC cl but not clang-cl — install the VS "
                    "'C++ Clang tools' component"
                )
            print(f"  [{'ok ' if has_git else 'MISS'}] git")
            print("  [note] set CC=CXX=clang-cl so the builds dev.py drives pin the compiler")
            print(
                "  [note] OpenSSL (bootgen) comes from choco/vcpkg on Windows; "
                "no apt package list"
            )
        return (has_clang_cl and has_git, [])
    if verbose:
        print(
            f"  [{'ok ' if cc else 'MISS'}] C/C++ compiler: "
            f"{(cc + '/' + cxx) if cc else 'need gcc (or set CC/CXX)'} "
            "(canonical: gcc; set CC/CXX for clang)"
        )
        print(
            f"  [{'ok ' if shutil.which('ld.lld') else '--  '}] "
            "ld.lld (optional; used with clang)"
        )
    if not cc:
        missing.append("build-essential")
    for probe, package in [("git", "git"), ("make", "build-essential")]:
        ok = shutil.which(probe) is not None
        if verbose:
            print(f"  [{'ok ' if ok else 'MISS'}] {probe}")
        if not ok and package not in missing:
            missing.append(package)
    for header, package in _SYS_HEADERS:
        ok = environment.have_header(header)
        if verbose:
            print(f"  [{'ok ' if ok else 'MISS'}] <{header}>")
        if not ok and package not in missing:
            missing.append(package)
    return (not missing, missing)


def check_fastflowlm_system(verbose, *, windows: bool | None = None):
    """Return (ok, missing_apt_packages) for the minimal HRX/FLM build."""
    missing = []
    selected_windows = paths.is_windows() if windows is None else windows
    env = (
        environment.windows_msvc_env(windows=True)
        if selected_windows
        else dict(os.environ)
    )
    dependency_config = fastflowlm_dependency_paths(paths.ROOT)
    native_errors = fastflowlm_dependency_errors(
        dependency_config,
        cargo_available=bool(
            environment.which("cargo", env, windows=selected_windows)
        ),
        rustc_version=environment.rustc_version(
            env, windows=selected_windows
        ),
    )
    if selected_windows:
        compiler = (
            environment.which("cl", env, windows=True)
            or environment.which("clang-cl", env, windows=True)
        )
        has_git = bool(environment.which("git", env, windows=True))
        if verbose:
            print(
                f"  [{'ok ' if compiler else 'MISS'}] C/C++ compiler: "
                f"{Path(compiler).name if compiler else 'need MSVC cl or clang-cl'}"
            )
            print(f"  [{'ok ' if has_git else 'MISS'}] git")
            for error in native_errors:
                print(f"  [MISS] {error}")
            if not native_errors:
                print(
                    f"  [ok ] Rust/Cargo >= "
                    f"{'.'.join(map(str, FASTFLOWLM_MIN_RUSTC))}"
                )
                print(
                    "  [ok ] vcpkg FastFlowLM dependencies: "
                    f"{dependency_config['triplet_root']}"
                )
        return (bool(compiler) and has_git and not native_errors, [])

    host_ok, missing = check_fastflowlm_linux_host(verbose)
    if verbose:
        for error in native_errors:
            print(f"  [MISS] {error}")
        if not native_errors:
            print(
                f"  [ok ] Rust/Cargo >= "
                f"{'.'.join(map(str, FASTFLOWLM_MIN_RUSTC))}"
            )
            print(
                "  [ok ] vcpkg FastFlowLM dependencies: "
                f"{dependency_config['triplet_root']}"
            )
    return (host_ok and not native_errors, missing)


def check_fastflowlm_packaging_system(
    verbose, *, windows: bool | None = None
):
    """Checks the FastFlowLM build and artifact-packaging prerequisites."""
    selected_windows = paths.is_windows() if windows is None else windows
    build_ok, build_missing = check_fastflowlm_system(
        verbose, windows=selected_windows
    )
    if selected_windows:
        return build_ok, build_missing
    if verbose:
        print("  packaging:")
    package_ok, package_missing = (
        check_fastflowlm_linux_packaging_tools(
            archive=True,
            verbose=verbose,
        )
    )
    return (
        build_ok and package_ok,
        sorted(set(build_missing + package_missing)),
    )


def check_system(profile_name="compiler-stack", verbose=True):
    """Return (ok, missing_apt_packages) for a doctor profile."""
    try:
        selected_profile = profile(profile_name)
    except KeyError as error:
        raise ValueError(
            f"unknown doctor profile: {profile_name}"
        ) from error
    if not selected_profile.doctor:
        raise ValueError(f"unknown doctor profile: {profile_name}")

    checks = {
        "compiler-stack": check_compiler_stack_system,
        "fastflowlm": check_fastflowlm_system,
        "fastflowlm-packaging": check_fastflowlm_packaging_system,
    }
    if len(selected_profile.system_checks) == 1:
        return checks[selected_profile.system_checks[0]](verbose)

    all_ok = True
    missing = []
    for check_name in selected_profile.system_checks:
        if verbose and len(selected_profile.system_checks) > 1:
            print(f"  {check_name}:")
        ok, check_missing = checks[check_name](verbose)
        all_ok &= ok
        missing.extend(check_missing)
    return all_ok, sorted(set(missing))


def cmd_doctor(args):
    """Check the selected build path's prerequisites, pins, patches, and fetches."""
    started = time.perf_counter()
    profile_name = args.profile
    deep = getattr(args, "deep", False)
    selected_profile = profile(profile_name)
    selected_submodules = set(selected_profile.submodules)
    mode = "deep" if deep else "quick"
    print(f"doctor profile: {profile_name} ({mode} patch validation)")
    print("system prerequisites (cmake + ninja come from pip via `setup`):")
    section_started = time.perf_counter()
    system_ok, missing = check_system(profile_name)
    print(
        f"  [time] system prerequisites: "
        f"{time.perf_counter() - section_started:.2f}s"
    )
    print("\nsubmodule pins vs manifests/versions.toml:")
    section_started = time.perf_counter()
    pin_bad = patches.check_pins(selected_submodules)
    print(f"  [time] submodule pins: {time.perf_counter() - section_started:.2f}s")
    print(
        "\ntemporary submodule patches "
        "(patches/, applied by builds or `dev.py fetch`):"
    )
    patch_bad = []
    patch_dirs = (
        sorted(path for path in paths.PATCHES.iterdir() if path.is_dir())
        if paths.PATCHES.exists()
        else []
    )
    for sub_dir in patch_dirs:
        if sub_dir.name not in selected_submodules:
            continue
        submodule_dir = paths.PROJ / sub_dir.name
        if not (submodule_dir / ".git").exists():
            print(f"  [ABSENT] {sub_dir.name}: cannot check patches")
            continue
        patch_files = sorted(sub_dir.glob("*.patch"))
        component_started = time.perf_counter()
        states = patches.doctor_patch_states(
            sub_dir.name, submodule_dir, patch_files, deep=deep
        )
        for patch_file, state in zip(patch_files, states):
            print(
                f"  [{'ok   ' if state == 'applied' else state.upper()}] "
                f"{sub_dir.name}/{patch_file.name}"
            )
            if state == "broken":
                patch_bad.append(patch_file.name)
        print(
            f"  [time] {sub_dir.name} patches: "
            f"{time.perf_counter() - component_started:.2f}s"
        )

    needs_eudsl = selected_profile.needs_eudsl
    eudsl_ok = True
    ver_ok = True
    if needs_eudsl:
        print("\nbuild-time prefetch (run `dev.py fetch` to populate):")
        section_started = time.perf_counter()
        eudsl_ok = patches.eudsl_cache_ok()
        print(
            f"  [{'ok  ' if eudsl_ok else 'MISS '}] "
            "eudsl sdist cached + sha256 verified"
        )
        eudsl_ver = patches.manifest().get("eudsl", {}).get("version", "")
        patch_file = paths.PATCHES / "iron-tools" / EUDSL_FIND_LINKS_PATCH
        installed = (
            eudsl_version_from_patch(patch_file.read_text())
            if patch_file.exists()
            else None
        )
        ver_ok = bool(eudsl_ver) and installed == eudsl_ver
        print(
            f"  [{'ok  ' if ver_ok else 'DRIFT'}] eudsl version: "
            f"patch=={installed or '?'} == "
            f"manifest=={eudsl_ver or '?'}"
        )
        print(
            f"  [time] build-time prefetch: "
            f"{time.perf_counter() - section_started:.2f}s"
        )
    rc = 0
    if not ver_ok:
        print(
            "\neudsl version DRIFT: patches/iron-tools/"
            f"{EUDSL_FIND_LINKS_PATCH} and versions.toml "
            "[eudsl] disagree — sync them"
        )
        rc = 1
    if patch_bad:
        print(
            f"\nPATCH does not apply: {', '.join(patch_bad)} — regenerate "
            "against the current submodule"
        )
        rc = 1
    if not system_ok:
        if missing:
            print(
                "\nMISSING system deps — install with:\n  "
                "sudo apt-get install -y " + " ".join(sorted(set(missing)))
            )
        else:
            print("\nMISSING system prerequisites shown above")
        rc = 1
    if pin_bad:
        print(
            f"\nSUBMODULE PIN problems in: {', '.join(pin_bad)} — initialize "
            "it or update its manifest pin"
        )
        rc = 1
    if needs_eudsl and not eudsl_ok:
        print("\neudsl not prefetched — run: python dev.py fetch")
        rc = 1
    if rc == 0:
        print(
            f"\nall good: {profile_name} prerequisites, pins, and patches are ready."
        )
    print(f"doctor elapsed: {time.perf_counter() - started:.2f}s")
    return rc
