# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Repository-local dependency setup orchestration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from build_tools import environment, paths
from build_tools.dev import doctor
from build_tools.dev import patches
from build_tools.dev.profiles import profile
from build_tools.fastflowlm_deps import (
    check_fastflowlm_linux_host,
    dependency_paths as fastflowlm_dependency_paths,
    vcpkg_commit as fastflowlm_vcpkg_commit,
)


@dataclass(frozen=True)
class SetupPlan:
    """Resolved, side-effect-free setup policy for one invocation."""

    profile_name: str
    requirements: Path
    require_hashes: bool
    submodules: frozenset[str]
    check_system: bool
    install_iron_ops: bool
    install_iron_ops_tests: bool
    install_fastflowlm_native: bool


def requirements_file(
    profile_name: str,
    *,
    root: Path | None = None,
    is_windows: bool | None = None,
) -> tuple[Path, bool]:
    """Returns the profile requirements path and whether hashes are required."""
    selected_root = paths.ROOT if root is None else root
    selected_windows = paths.is_windows() if is_windows is None else is_windows
    requirements = profile(profile_name).requirements
    if requirements is None:
        raise ValueError(f"profile has no setup requirements: {profile_name}")
    source = selected_root / requirements.filename(windows=selected_windows)
    if (
        not selected_windows
        and not source.exists()
        and requirements.fallback_if_missing
    ):
        source = selected_root / requirements.fallback_if_missing
    return source, requirements.require_hashes


def build_setup_plan(
    profile_name: str,
    *,
    iron_ops_tests: bool = False,
    root: Path | None = None,
    is_windows: bool | None = None,
) -> SetupPlan:
    """Resolves setup policy before performing any filesystem mutation."""
    spec = profile(profile_name)
    requirements, require_hashes = requirements_file(
        profile_name,
        root=root,
        is_windows=is_windows,
    )
    return SetupPlan(
        profile_name=profile_name,
        requirements=requirements,
        require_hashes=require_hashes,
        submodules=spec.submodules,
        check_system=spec.check_system_before_setup,
        install_iron_ops=spec.install_iron_ops,
        install_iron_ops_tests=iron_ops_tests and spec.install_iron_ops,
        install_fastflowlm_native=spec.install_fastflowlm_native,
    )


def install_iron_ops(pip: list[str], with_tests: bool = False) -> None:
    """Installs the source checkout without its conflicting wheel dependencies."""
    if not (paths.PROJ / "iron-ops" / "pyproject.toml").exists():
        print("setup: projects/iron-ops not checked out; skipping the `iron` install")
        return
    environment.sh(pip + ["--no-deps", "-e", str(paths.PROJ / "iron-ops")])
    if with_tests:
        lock = paths.ROOT / "requirements/iron-ops-test.lock"
        source = (
            paths.ROOT / "requirements/iron-ops-test.txt"
            if paths.is_windows() or not lock.exists()
            else lock
        )
        environment.sh(pip + ["-r", str(source)])


def _install_fastflowlm_deps() -> None:
    """Installs the manifest-defined native dependencies into the managed cache."""
    if not paths.is_windows():
        host_ok, missing = check_fastflowlm_linux_host(verbose=False)
        if not host_ok:
            sys.exit(
                "FastFlowLM host build prerequisites are missing; install: "
                "sudo apt-get install -y " + " ".join(sorted(set(missing)))
            )
    dependency_config = fastflowlm_dependency_paths(paths.ROOT)
    source = Path(dependency_config["source"])
    install_root = Path(dependency_config["install_root"])
    binary_cache = paths.ROOT / ".cache" / "vcpkg-bincache"
    downloads = paths.ROOT / ".cache" / "vcpkg-downloads"
    for path in (source.parent, install_root, binary_cache, downloads):
        path.mkdir(parents=True, exist_ok=True)

    source_changed = False
    if not (source / ".git").is_dir():
        environment.sh(["git", "init", str(source)])
        environment.sh(
            [
                "git",
                "-C",
                str(source),
                "remote",
                "add",
                "origin",
                "https://github.com/microsoft/vcpkg.git",
            ]
        )
        source_changed = True
    else:
        environment.sh(
            [
                "git",
                "-C",
                str(source),
                "remote",
                "set-url",
                "origin",
                "https://github.com/microsoft/vcpkg.git",
            ]
        )
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    vcpkg_commit = fastflowlm_vcpkg_commit(paths.ROOT)
    if head != vcpkg_commit:
        environment.sh(
            ["git", "-C", str(source), "fetch", "--depth", "1", "origin", vcpkg_commit]
        )
        environment.sh(
            ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"]
        )
        source_changed = True
    executable = Path(dependency_config["executable"])
    if source_changed or not executable.is_file():
        if paths.is_windows():
            environment.sh(
                [
                    "cmd.exe",
                    "/d",
                    "/c",
                    str(source / "bootstrap-vcpkg.bat"),
                    "-disableMetrics",
                ]
            )
        else:
            environment.sh([str(source / "bootstrap-vcpkg.sh"), "-disableMetrics"])

    command_env = os.environ.copy()
    command_env["VCPKG_BINARY_SOURCES"] = (
        f"clear;files,{environment.cmake_path(binary_cache)},readwrite"
    )
    command_env["VCPKG_DOWNLOADS"] = str(downloads)
    powershell = shutil.which("powershell.exe") if paths.is_windows() else None
    downloader = paths.ROOT / ".github" / "scripts" / "vcpkg-download.ps1"
    if powershell and downloader.is_file():
        template = (
            f'"{powershell}" -NoProfile -NonInteractive -ExecutionPolicy Bypass '
            f'-File "{downloader}" -Url {{url}} -Sha512 {{sha512}} '
            "-Destination {dst}"
        )
        command_env["X_VCPKG_ASSET_SOURCES"] = f"clear;x-script,{template}"

    environment.sh(
        [
            executable,
            "install",
            "--triplet",
            dependency_config["triplet"],
            f"--overlay-triplets={paths.ROOT / '.github' / 'vcpkg' / 'triplets'}",
            f"--x-manifest-root={paths.ROOT / '.github' / 'vcpkg' / 'fastflowlm'}",
            f"--x-install-root={install_root}",
        ],
        env=command_env,
    )
    marker = Path(dependency_config["marker"])
    marker_tmp = marker.with_suffix(".tmp")
    marker_tmp.write_text(
        json.dumps(dependency_config["policy"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker_tmp.replace(marker)
    print(
        "setup: FastFlowLM native dependencies ->",
        dependency_config["triplet_root"],
    )


def cmd_setup(args) -> None:
    """Prepares the selected dependency closure in repository-local caches."""
    paths.OUT.mkdir(exist_ok=True)
    plan = build_setup_plan(
        args.profile,
        iron_ops_tests=getattr(args, "iron_ops_tests", False),
    )
    if plan.check_system:
        ok, missing = doctor.check_system(plan.profile_name, verbose=False)
        if not ok:
            if missing:
                print("WARNING: missing system deps (apt): " + " ".join(sorted(missing)))
            else:
                print("WARNING: compiler-stack system prerequisites are missing")
            print(
                "  install them, then re-run; "
                "see `python dev.py doctor compiler-stack`"
            )
    if not environment.venv_py().exists():
        environment.sh([sys.executable, "-m", "venv", str(paths.VENV)])

    environment.sh(
        [
            str(environment.venv_py()),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            f"uv=={environment.UV_VERSION}",
        ]
    )
    pip = [
        str(environment.uv_bin()),
        "pip",
        "install",
        "--python",
        str(environment.venv_py()),
    ]
    pip_args = ["--require-hashes"] if plan.require_hashes else []
    environment.sh(pip + pip_args + ["-r", str(plan.requirements)])
    patches.cmd_fetch(args, set(plan.submodules))
    if plan.install_iron_ops:
        install_iron_ops(
            pip,
            with_tests=plan.install_iron_ops_tests,
        )
    if plan.install_fastflowlm_native:
        _install_fastflowlm_deps()
    print(f"setup: ok ({args.profile} dependencies are ready)")
