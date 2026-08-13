# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared filesystem mechanics for staging wheel payload trees."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from build_tools.artifact import file_inventory


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        resolved = source.resolve()
        if not resolved.is_file():
            raise RuntimeError(
                f"wheel payload symlink does not resolve to a file: {source}"
            )
        source = resolved
    shutil.copy2(source, destination)


def copy_relative_file(
    source_root: Path, relative: str, destination_root: Path
) -> None:
    source = source_root / relative
    if not source.is_file():
        raise RuntimeError(f"required wheel payload file missing: {source}")
    copy_file(source, destination_root / relative)


def copy_relative_tree(
    source_root: Path, relative: str, destination_root: Path
) -> None:
    source = source_root / relative
    if not source.is_dir():
        raise RuntimeError(f"required wheel payload directory missing: {source}")
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        copy_file(path, destination_root / path.relative_to(source_root))


def copy_license(source_bundle: Path, relative: str, package_root: Path) -> None:
    source = source_bundle / relative
    if not source.exists():
        raise RuntimeError(f"required wheel license metadata missing: {source}")
    destination = package_root / relative
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        copy_file(source, destination)


def copy_license_as(
    source_bundle: Path,
    relative: str,
    package_root: Path,
    destination_relative: str,
) -> None:
    source = source_bundle / relative
    if not source.is_file():
        raise RuntimeError(f"required wheel license metadata missing: {source}")
    copy_file(source, package_root / destination_relative)


def _dependency_license_packages(
    manifest: dict[str, object], shipped_names: set[str]
) -> tuple[set[str], set[str]]:
    shipped_names = {name.casefold() for name in shipped_names}
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, dict):
        return set(), set()
    vcpkg = (
        dependencies.get("vcpkg")
        if dependencies.get("provider") == "vcpkg+system"
        else dependencies
    )
    system = (
        dependencies.get("system")
        if dependencies.get("provider") == "vcpkg+system"
        else {}
    )

    vcpkg_packages = set()
    if isinstance(vcpkg, dict):
        mapping = vcpkg.get("runtime_file_packages")
        if isinstance(mapping, dict):
            vcpkg_packages.update(
                str(package)
                for name, package in mapping.items()
                if str(name).casefold() in shipped_names
            )

    system_packages = set()
    if isinstance(system, dict):
        mapping = system.get("library_packages")
        if isinstance(mapping, dict):
            system_packages.update(
                str(package)
                for name, package in mapping.items()
                if str(name).casefold() in shipped_names
            )
    return vcpkg_packages, system_packages


def copy_dependency_licenses(
    source_bundle: Path,
    manifest: dict[str, object],
    package_root: Path,
    shipped_names: set[str],
) -> None:
    vcpkg_packages, system_packages = _dependency_license_packages(
        manifest, shipped_names
    )
    for package in sorted(vcpkg_packages):
        source = source_bundle / "licenses" / "vcpkg" / package
        if not source.is_dir():
            raise RuntimeError(
                f"shipped vcpkg dependency {package} is missing license material: "
                f"{source}"
            )
        shutil.copytree(
            source,
            package_root / "licenses" / "vcpkg" / package,
            dirs_exist_ok=True,
        )
    for package in sorted(system_packages):
        source = source_bundle / "licenses" / "system" / package
        if not source.is_dir():
            raise RuntimeError(
                f"shipped system dependency {package} is missing license material: "
                f"{source}"
            )
        shutil.copytree(
            source,
            package_root / "licenses" / "system" / package,
            dirs_exist_ok=True,
        )


def write_file(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    path.chmod(mode)


def write_package_manifest(
    package_root: Path, manifest: dict[str, object]
) -> dict[str, object]:
    manifest_path = package_root / "manifest.json"
    manifest = dict(
        manifest,
        files=file_inventory(package_root, exclude_paths=(manifest_path,)),
    )
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest
