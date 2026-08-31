# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Resolve a release meta-wheel against every compiler Python x platform."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import shutil
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from build_tools import release_policy
from build_tools.environment import sh
from build_tools.rocm_npu_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    OPS_DISTRIBUTION,
)

PLATFORMS = {
    "linux": ("manylinux_2_28_x86_64",),
    "windows": ("win_amd64",),
}
PRODUCTS = ("compiler", "ops")


def python_tag(python_minor: str) -> str:
    return "cp" + python_minor.replace(".", "")


def bindings_runtime_pins() -> dict[str, str]:
    pins = {}
    for requirement_text in COMPILER_BINDINGS_RUNTIME_REQUIREMENTS:
        requirement = Requirement(requirement_text)
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise RuntimeError(
                "COMPILER_BINDINGS_RUNTIME_REQUIREMENTS must be == pins: "
                f"{requirement_text}"
            )
        pins[canonicalize_name(requirement.name)] = specifiers[0].version
    return pins


def expected_distributions(product: str, version: str) -> dict[str, str]:
    compiler_family = {
        COMPILER_DISTRIBUTION: version,
        COMPILER_CORE_DISTRIBUTION: version,
        COMPILER_BINDINGS_DISTRIBUTION: version,
    }
    if product == "compiler":
        return {**compiler_family, **bindings_runtime_pins()}
    if product == "ops":
        return {OPS_DISTRIBUTION: version, **compiler_family}
    raise RuntimeError(f"unknown release resolve product: {product}")


def installed_versions(target: Path) -> dict[str, str]:
    return {
        canonicalize_name(distribution.metadata["Name"]): distribution.version
        for distribution in metadata.distributions(path=[str(target)])
    }


def verify_installed(target: Path, expected: dict[str, str]) -> None:
    installed = installed_versions(target)
    if not expected.items() <= installed.items():
        raise RuntimeError(
            f"release resolve closed fewer packages than required in {target}: "
            f"expected {expected}, installed {installed}"
        )


def pip_install_command(
    *,
    python: str | Path,
    wheel: Path,
    find_links: Path,
    python_minor: str,
    platforms: tuple[str, ...],
    target: Path,
) -> list[str]:
    tag = python_tag(python_minor)
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--find-links",
        str(find_links),
        "--only-binary=:all:",
    ]
    for platform in platforms:
        command.extend(["--platform", platform])
    command.extend(
        [
            "--implementation",
            "cp",
            "--python-version",
            python_minor,
            "--abi",
            tag,
            "--abi",
            "abi3",
            "--target",
            str(target),
            str(wheel),
        ]
    )
    return command


def resolve_release_wheel(
    *,
    wheel: Path,
    find_links: Path,
    version: str,
    product: str,
    work_dir: Path,
    python: str | Path = sys.executable,
    python_versions: tuple[str, ...] | None = None,
) -> None:
    wheel = Path(wheel)
    find_links = Path(find_links)
    work_dir = Path(work_dir)
    if not wheel.is_file():
        raise RuntimeError(f"release wheel missing: {wheel}")
    expected = expected_distributions(product, version)
    minors = (
        python_versions
        if python_versions is not None
        else release_policy.compiler_python_versions()
    )
    for platform_name, platforms in PLATFORMS.items():
        for python_minor in minors:
            target = work_dir / (
                f"{product}-release-install-{platform_name}-"
                f"{python_tag(python_minor)}"
            )
            if target.exists():
                shutil.rmtree(target)
            try:
                sh(
                    pip_install_command(
                        python=python,
                        wheel=wheel,
                        find_links=find_links,
                        python_minor=python_minor,
                        platforms=platforms,
                        target=target,
                    )
                )
                verify_installed(target, expected)
            finally:
                shutil.rmtree(target, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--find-links", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--product", choices=PRODUCTS, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    resolve_release_wheel(
        wheel=args.wheel,
        find_links=args.find_links,
        version=args.version,
        product=args.product,
        work_dir=args.work_dir,
    )


if __name__ == "__main__":
    main()
