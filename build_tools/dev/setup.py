# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Repository-local dependency setup orchestration."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from build_tools import environment, fastflowlm_deps, paths
from build_tools.dev import doctor, patches
from build_tools.dev.profiles import profile


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
                print(
                    "WARNING: missing system deps (apt): " + " ".join(sorted(missing))
                )
            else:
                print("WARNING: compiler-stack system prerequisites are missing")
            print(
                "  install them, then re-run; see `python dev.py doctor compiler-stack`"
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
    environment.sh(pip + doctor.format_pip_packages())
    patches.cmd_fetch(args, set(plan.submodules))
    if plan.install_iron_ops:
        install_iron_ops(
            pip,
            with_tests=plan.install_iron_ops_tests,
        )
    if plan.install_fastflowlm_native:
        try:
            fastflowlm_deps.install_managed_vcpkg(paths.ROOT)
        except RuntimeError as error:
            sys.exit(str(error))
    print(f"setup: ok ({args.profile} dependencies are ready)")
