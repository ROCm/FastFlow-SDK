# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build Linux fastflow runtime and FastFlowLM wheels from staged artifacts."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from build_tools.fastflow_sdk_wheel_common import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    find_distribution_wheels,
    find_wheel_path_references,
    remove_distribution_wheels,
)
from build_tools.fastflow_sdk_wheel_common import (
    build_wheel as _build_wheel,
)
from build_tools.fastflowlm_artifact import (
    _is_linux_system_runtime,
    verify_bundle,
)
from build_tools.manylinux_audit import validate_manylinux_policy
from build_tools.native_dependencies import (
    elf_needed,
    elf_rpath,
    elf_soname,
    is_elf,
    set_elf_rpath,
)
from build_tools.package_version import validate_version
from build_tools.runtime_wheel_staging import (
    prepare_runtime_wheel_trees,
    write_runtime_wheel_manifests,
)
from build_tools.wheel_payload import (
    copy_dependency_licenses as _copy_dependency_licenses,
)
from build_tools.wheel_payload import (
    copy_file as _copy_file,
)
from build_tools.wheel_payload import (
    copy_relative_file as _copy_relative_file,
)
from build_tools.wheel_payload import (
    copy_relative_tree as _copy_relative_tree,
)

LINUX_WHEEL_TAG = "py3-none-manylinux_2_28_x86_64"
FASTFLOWLM_LINUX_WHEEL_TAG = "py3-none-linux_x86_64"


def validate_auditwheel_policy(distribution: str, policy: str) -> str:
    """Checks runtime-product audit policy against its claimed wheel tag."""
    if distribution == RUNTIME_DISTRIBUTION:
        try:
            return validate_manylinux_policy(policy, maximum_glibc=(2, 28))
        except RuntimeError as error:
            raise RuntimeError(
                f"{distribution} is not manylinux_2_28 compatible: {policy}"
            ) from error
    if distribution == FASTFLOWLM_DISTRIBUTION:
        if policy != "linux_x86_64" and not re.fullmatch(
            r"manylinux_\d+_\d+_x86_64", policy
        ):
            raise RuntimeError(
                f"{distribution} has an unexpected auditwheel policy: {policy}"
            )
        return policy
    raise RuntimeError(f"unexpected Linux runtime wheel distribution: {distribution}")


def wheel_tag(system: str | None = None, machine: str | None = None) -> str:
    system = (system or platform.system()).lower()
    if system != "linux":
        raise RuntimeError(f"fastflow Linux wheels are unsupported on {system}")
    machine = (machine or platform.machine()).lower()
    if machine not in ("amd64", "x86_64"):
        raise RuntimeError(
            f"fastflow Linux wheels require an x86-64 host, got {machine}"
        )
    return LINUX_WHEEL_TAG


def fastflowlm_wheel_tag(system: str | None = None, machine: str | None = None) -> str:
    wheel_tag(system, machine)
    return FASTFLOWLM_LINUX_WHEEL_TAG


def _elf_files(package_root: Path) -> list[Path]:
    return sorted(
        path for path in package_root.rglob("*") if path.is_file() and is_elf(path)
    )


def _copy_hrx_soname(source_bundle: Path, runtime_root: Path) -> str:
    """Copies only the runtime SONAME, omitting linker and version aliases."""
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to prepare Linux fastflow wheels")
    libraries = sorted((source_bundle / "lib").glob("libhrx.so*"))
    if not libraries:
        raise RuntimeError(
            f"staged runtime bundle contains no libhrx.so* under "
            f"{source_bundle / 'lib'}"
        )
    sonames = {
        soname
        for library in libraries
        if (soname := elf_soname(library, patchelf=patchelf))
    }
    if len(sonames) != 1:
        raise RuntimeError(
            "staged HRX libraries must expose exactly one ELF SONAME, found: "
            + ", ".join(sorted(sonames))
        )
    soname = next(iter(sonames))
    source = next(
        (library for library in libraries if library.name == soname),
        libraries[0],
    )
    _copy_file(source, runtime_root / "lib" / soname)
    return soname


def _provided_library_filenames(package_root: Path) -> set[str]:
    return {path.name for path in _elf_files(package_root)}


def _source_linux_library_candidates(
    source_bundle: Path, patchelf: str
) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    for path in sorted((source_bundle / "lib").rglob("*.so*")):
        if not path.is_file() or not is_elf(path):
            continue
        candidates.setdefault(path.name, path)
        soname = elf_soname(path, patchelf=patchelf)
        if soname:
            candidates.setdefault(soname, path)
    return candidates


def _copy_native_closure(
    source_bundle: Path, package_root: Path, external_roots: tuple[Path, ...] = ()
) -> set[str]:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to prepare Linux fastflow wheels")
    candidates = _source_linux_library_candidates(source_bundle, patchelf)
    license_names = set()
    queue = _elf_files(package_root)
    while queue:
        path = queue.pop(0)
        for needed in elf_needed(path, patchelf=patchelf):
            provided = _provided_library_filenames(package_root)
            for external_root in external_roots:
                provided.update(_provided_library_filenames(external_root))
            if needed in provided:
                continue
            if _is_linux_system_runtime(needed):
                continue
            source = candidates.get(needed)
            if not source:
                relative = path.relative_to(package_root).as_posix()
                raise RuntimeError(
                    f"{package_root.name} dependency is not present in the staged "
                    f"FastFlowLM bundle: {relative} -> {needed}"
                )
            destination = package_root / "lib" / needed
            if destination.exists():
                continue
            _copy_file(source, destination)
            license_names.update({source.name, needed})
            queue.append(destination)
    return license_names


def _copy_runtime_native_closure(source_bundle: Path, runtime_root: Path) -> set[str]:
    return _copy_native_closure(source_bundle, runtime_root)


def _copy_fastflowlm_native_closure(
    source_bundle: Path, fastflowlm_root: Path, runtime_root: Path
) -> set[str]:
    return _copy_native_closure(source_bundle, fastflowlm_root, (runtime_root,))


def _rpath_for(path: Path, package_root: Path) -> str:
    relative = path.relative_to(package_root)
    if package_root.name == RUNTIME_PACKAGE:
        return "$ORIGIN"
    if relative.parts[0] == "bin":
        return (
            "$ORIGIN/../lib/flm:$ORIGIN/../lib:$ORIGIN/../../fastflow_sdk_runtime/lib"
        )
    if len(relative.parts) >= 3 and relative.parts[0:2] == ("lib", "flm"):
        return "$ORIGIN:$ORIGIN/..:$ORIGIN/../../../fastflow_sdk_runtime/lib"
    if relative.parts[0] == "lib":
        return "$ORIGIN:$ORIGIN/flm:$ORIGIN/../../fastflow_sdk_runtime/lib"
    return "$ORIGIN"


def _strip_debug(package_roots: list[Path]) -> dict[str, int]:
    strip = shutil.which("strip")
    if not strip:
        raise RuntimeError("strip is required to prepare Linux fastflow wheels")
    targets = [
        path for package_root in package_roots for path in _elf_files(package_root)
    ]
    if targets:
        subprocess.run(
            [strip, "--strip-debug", *[str(path) for path in targets]], check=True
        )
    return {"stripped": len(targets)}


def _patch_and_verify_linux_products(package_roots: list[Path]) -> dict[str, list[str]]:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to prepare Linux fastflow wheels")
    package_elf_files = {
        package_root: _elf_files(package_root) for package_root in package_roots
    }
    if not any(package_elf_files.values()):
        raise RuntimeError("fastflow wheel payload contains no ELF files")

    for package_root, elf_files in package_elf_files.items():
        for path in elf_files:
            set_elf_rpath(path, _rpath_for(path, package_root), patchelf=patchelf)
            absolute_entries = [
                entry
                for entry in elf_rpath(path, patchelf=patchelf).split(":")
                if entry.startswith("/")
            ]
            if absolute_entries:
                relative = (
                    f"{package_root.name}/{path.relative_to(package_root).as_posix()}"
                )
                raise RuntimeError(
                    "fastflow wheel ELF RPATH/RUNPATH contains absolute paths: "
                    f"{relative} -> {':'.join(absolute_entries)}"
                )

    provided_names = {
        package_root: _provided_library_filenames(package_root)
        for package_root in package_roots
    }
    runtime_names = set()
    for package_root, names in provided_names.items():
        if package_root.name == RUNTIME_PACKAGE:
            runtime_names = names
            break

    graph = {}
    missing = []
    for package_root, elf_files in package_elf_files.items():
        allowed_names = set(provided_names[package_root])
        if package_root.name == FASTFLOWLM_PACKAGE:
            allowed_names.update(runtime_names)
        for path in elf_files:
            needed = elf_needed(path, patchelf=patchelf)
            relative = (
                f"{package_root.name}/{path.relative_to(package_root).as_posix()}"
            )
            graph[relative] = sorted(needed)
            missing.extend(
                f"{relative} -> {name}"
                for name in needed
                if name not in allowed_names and not _is_linux_system_runtime(name)
            )
    if missing:
        raise RuntimeError(
            "fastflow wheel Linux dependency closure is incomplete "
            "within package boundaries:\n  " + "\n  ".join(sorted(missing))
        )
    return graph


def stage_wheel_products(
    source_bundle: Path, destination: Path, version: str
) -> dict[str, Path]:
    """Stage split package roots from the combined FastFlowLM+HRX bundle."""
    if os.name == "nt":
        raise RuntimeError(
            "fastflow runtime/FastFlowLM wheel staging is Linux-only for now"
        )
    version = validate_version(version)
    source_manifest = verify_bundle(source_bundle)
    trees = prepare_runtime_wheel_trees(source_bundle, destination)
    runtime_tree = trees.runtime_tree
    fastflowlm_tree = trees.fastflowlm_tree
    runtime_root = trees.runtime_root
    fastflowlm_root = trees.fastflowlm_root

    _copy_hrx_soname(source_bundle, runtime_root)
    _copy_relative_file(
        source_bundle, "bin/hrx-amdxdna-resource-pressure", runtime_root
    )
    _copy_relative_file(source_bundle, "bin/fastflow", runtime_root)
    runtime_license_names = _copy_runtime_native_closure(source_bundle, runtime_root)

    _copy_relative_file(source_bundle, "bin/flm", fastflowlm_root)
    flm_lib_dir = source_bundle / "lib" / "flm"
    if flm_lib_dir.is_dir():
        _copy_relative_tree(source_bundle, "lib/flm", fastflowlm_root)
    fastflowlm_license_names = _copy_fastflowlm_native_closure(
        source_bundle, fastflowlm_root, runtime_root
    )

    runtime_names = {path.name for path in runtime_root.rglob("*") if path.is_file()}
    runtime_names.update(runtime_license_names)
    fastflowlm_names = {
        path.name for path in fastflowlm_root.rglob("*") if path.is_file()
    }
    fastflowlm_names.update(fastflowlm_license_names)
    _copy_dependency_licenses(
        source_bundle, source_manifest, runtime_root, runtime_names
    )
    _copy_dependency_licenses(
        source_bundle, source_manifest, fastflowlm_root, fastflowlm_names
    )

    strip_report = _strip_debug([runtime_root, fastflowlm_root])
    binary_dependencies = _patch_and_verify_linux_products(
        [runtime_root, fastflowlm_root]
    )

    write_runtime_wheel_manifests(
        trees,
        source_manifest,
        version=version,
        runtime_wheel_tag=wheel_tag(),
        fastflowlm_wheel_tag=fastflowlm_wheel_tag(),
        strip_report=strip_report,
        binary_dependencies=binary_dependencies,
    )
    return {
        RUNTIME_DISTRIBUTION: runtime_tree,
        FASTFLOWLM_DISTRIBUTION: fastflowlm_tree,
    }


def build_wheel(
    tree_root: Path,
    distribution: str,
    version: str,
    output_directory: Path,
    *,
    summary: str,
    requires: tuple[str, ...] = (),
    console_scripts: dict[str, str] | None = None,
    tag: str | None = None,
) -> Path:
    return _build_wheel(
        tree_root,
        distribution,
        version,
        output_directory,
        summary=summary,
        tag=tag or wheel_tag(),
        requires=requires,
        console_scripts=console_scripts,
    )


def build_runtime_fastflowlm_wheels(
    source_bundle: Path, stage_directory: Path, output_directory: Path, version: str
) -> dict[str, Path]:
    products = stage_wheel_products(source_bundle, stage_directory, version)
    remove_distribution_wheels(
        output_directory, (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION)
    )
    runtime_wheel = build_wheel(
        products[RUNTIME_DISTRIBUTION],
        RUNTIME_DISTRIBUTION,
        version,
        output_directory,
        summary="FastFlow SDK HRX amdxdna runtime",
        console_scripts={"fastflow": f"{RUNTIME_PACKAGE}:main"},
    )
    fastflowlm_wheel = build_wheel(
        products[FASTFLOWLM_DISTRIBUTION],
        FASTFLOWLM_DISTRIBUTION,
        version,
        output_directory,
        summary="FastFlow SDK FastFlowLM runtime assets",
        tag=fastflowlm_wheel_tag(),
        requires=(f"{RUNTIME_DISTRIBUTION} =={version}",),
        console_scripts={"flm": f"{FASTFLOWLM_PACKAGE}:main"},
    )
    return {
        RUNTIME_DISTRIBUTION: runtime_wheel,
        FASTFLOWLM_DISTRIBUTION: fastflowlm_wheel,
    }


def find_fastflow_sdk_wheels(directory: Path) -> dict[str, Path]:
    return find_distribution_wheels(
        directory, (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION)
    )


def _zip_member_is_elf(data: bytes) -> bool:
    return data.startswith(b"\x7fELF")


def verify_wheels_do_not_have_text_path_references(
    wheels: dict[str, Path], forbidden_paths: tuple[Path, ...]
) -> None:
    """Rejects local paths in text payloads and audits opaque native payloads."""
    references = find_wheel_path_references(wheels, forbidden_paths)
    text_references = []
    native_references = []
    for reference in references:
        with zipfile.ZipFile(reference.wheel) as archive:
            with archive.open(reference.member) as member:
                is_native = _zip_member_is_elf(member.read(4))
        (native_references if is_native else text_references).append(reference)
    if native_references:
        print(
            "wheel path audit: "
            f"{len(native_references)} native member(s) contain local path strings; "
            "native relocation is enforced by ELF RPATH/RUNPATH and runtime tests"
        )
    if text_references:
        details = "\n  ".join(
            f"{reference.wheel.name}:{reference.member} -> {reference.forbidden_path}"
            for reference in text_references
        )
        raise RuntimeError(
            "wheel text payload contains local checkout/build paths:\n  " + details
        )
