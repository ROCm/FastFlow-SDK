# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build Windows rocm-npu runtime and FastFlowLM wheels."""

from __future__ import annotations

import os
import platform
import zipfile
from pathlib import Path

from build_tools.fastflowlm_artifact import verify_bundle
from build_tools.native_dependencies import is_pe, pe_imports
from build_tools.rocm_npu_wheel_common import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    _copy_dependency_licenses,
    _copy_file,
    _copy_license,
    _copy_relative_file,
    find_distribution_wheels,
    find_wheel_path_references,
    remove_distribution_wheels,
    validate_version,
)
from build_tools.rocm_npu_wheel_common import (
    build_wheel as _build_wheel,
)
from build_tools.runtime_wheel_staging import (
    prepare_runtime_wheel_trees,
    write_runtime_wheel_manifests,
)

WINDOWS_WHEEL_TAG = "py3-none-win_amd64"


def wheel_tag(system: str | None = None, machine: str | None = None) -> str:
    system = (system or platform.system()).lower()
    if system != "windows":
        raise RuntimeError(f"rocm-npu wheels are unsupported on {system}")
    machine = (machine or platform.machine()).lower()
    if machine not in ("amd64", "x86_64"):
        raise RuntimeError(
            f"rocm-npu Windows wheels require an x86-64 host, got {machine}"
        )
    return WINDOWS_WHEEL_TAG


def _pe_files(package_root: Path) -> list[Path]:
    return sorted(path for path in package_root.rglob("*")
                  if path.is_file() and is_pe(path))


def _is_windows_system_runtime(name: str) -> bool:
    name = name.lower()
    if name.startswith(("api-ms-", "ext-ms-")):
        return True
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return (system32 / name).is_file()


def _copy_windows_runtime_closure(source_bundle: Path, runtime_root: Path) -> None:
    candidates = {
        path.name.lower(): path
        for path in sorted((source_bundle / "bin").glob("*.dll"))
        if path.is_file()
    }
    queue = _pe_files(runtime_root)
    while queue:
        path = queue.pop(0)
        for needed in pe_imports(path):
            destination = runtime_root / "bin" / needed
            if destination.is_file() or _is_windows_system_runtime(needed):
                continue
            source = candidates.get(needed)
            if not source:
                relative = path.relative_to(runtime_root).as_posix()
                raise RuntimeError(
                    f"{RUNTIME_PACKAGE} dependency is not present in the staged "
                    f"FastFlowLM bundle: {relative} -> {needed}")
            _copy_file(source, destination)
            queue.append(destination)


def _copy_windows_fastflowlm_payload(source_bundle: Path, fastflowlm_root: Path,
                                     runtime_root: Path) -> None:
    runtime_names = {
        path.name.lower() for path in (runtime_root / "bin").glob("*.dll")
    }
    _copy_relative_file(source_bundle, "bin/flm.exe", fastflowlm_root)
    for source in sorted((source_bundle / "bin").glob("*.dll")):
        if source.name.lower() not in runtime_names:
            _copy_file(source, fastflowlm_root / "bin" / source.name)


def _copy_microsoft_runtime_notice(
    source_bundle: Path,
    source_manifest: dict[str, object],
    package_root: Path,
) -> None:
    dependencies = source_manifest.get("dependencies")
    if not isinstance(dependencies, dict):
        return
    vcpkg = (
        dependencies.get("vcpkg")
        if dependencies.get("provider") == "vcpkg+system"
        else dependencies
    )
    if not isinstance(vcpkg, dict):
        return
    runtime_files = vcpkg.get("microsoft_runtime_files")
    if not isinstance(runtime_files, list):
        return
    shipped_names = {
        path.name.casefold() for path in package_root.rglob("*") if path.is_file()
    }
    if not any(str(name).casefold() in shipped_names for name in runtime_files):
        return
    _copy_license(
        source_bundle,
        "licenses/Microsoft-Visual-Cpp-Runtime-NOTICE.txt",
        package_root,
    )


def _verify_windows_products(package_roots: list[Path]) -> dict[str, list[str]]:
    package_files = {
        package_root: {path.name.lower() for path in _pe_files(package_root)}
        for package_root in package_roots
    }
    runtime_names = next(
        (names for root, names in package_files.items()
         if root.name == RUNTIME_PACKAGE),
        set(),
    )
    graph = {}
    missing = []
    for package_root in package_roots:
        allowed_names = set(package_files[package_root])
        if package_root.name == FASTFLOWLM_PACKAGE:
            allowed_names.update(runtime_names)
        for path in _pe_files(package_root):
            imports = pe_imports(path)
            relative = f"{package_root.name}/{path.relative_to(package_root).as_posix()}"
            graph[relative] = imports
            missing.extend(
                f"{relative} -> {name}" for name in imports
                if name not in allowed_names and not _is_windows_system_runtime(name)
            )
    if missing:
        raise RuntimeError(
            "rocm-npu wheel Windows dependency closure is incomplete within "
            "package boundaries:\n  " + "\n  ".join(sorted(missing)))
    return graph


def stage_wheel_products(source_bundle: Path, destination: Path,
                         version: str) -> dict[str, Path]:
    """Stage split package roots from the combined FastFlowLM+HRX bundle."""
    if os.name != "nt":
        raise RuntimeError("Windows rocm-npu wheel staging requires a Windows host")
    version = validate_version(version)
    source_manifest = verify_bundle(source_bundle)
    trees = prepare_runtime_wheel_trees(source_bundle, destination)
    runtime_tree = trees.runtime_tree
    fastflowlm_tree = trees.fastflowlm_tree
    runtime_root = trees.runtime_root
    fastflowlm_root = trees.fastflowlm_root

    _copy_relative_file(source_bundle, "bin/hrx.dll", runtime_root)
    _copy_relative_file(source_bundle, "bin/rocm-npu.exe", runtime_root)
    _copy_windows_runtime_closure(source_bundle, runtime_root)
    _copy_windows_fastflowlm_payload(
        source_bundle, fastflowlm_root, runtime_root)
    runtime_names = {path.name for path in runtime_root.rglob("*") if path.is_file()}
    fastflowlm_names = {path.name for path in fastflowlm_root.rglob("*") if path.is_file()}
    _copy_dependency_licenses(source_bundle, source_manifest, runtime_root, runtime_names)
    _copy_dependency_licenses(source_bundle, source_manifest, fastflowlm_root, fastflowlm_names)
    _copy_microsoft_runtime_notice(source_bundle, source_manifest, runtime_root)
    _copy_microsoft_runtime_notice(source_bundle, source_manifest, fastflowlm_root)

    binary_dependencies = _verify_windows_products(
        [runtime_root, fastflowlm_root])

    write_runtime_wheel_manifests(
        trees,
        source_manifest,
        version=version,
        runtime_wheel_tag=wheel_tag(),
        fastflowlm_wheel_tag=wheel_tag(),
        strip_report={"stripped": 0},
        binary_dependencies=binary_dependencies,
    )
    return {
        RUNTIME_DISTRIBUTION: runtime_tree,
        FASTFLOWLM_DISTRIBUTION: fastflowlm_tree,
    }


def build_wheel(tree_root: Path, distribution: str, version: str, output_directory: Path,
                *, summary: str, requires: tuple[str, ...] = (),
                console_scripts: dict[str, str] | None = None,
                tag: str | None = None) -> Path:
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


def build_runtime_fastflowlm_wheels(source_bundle: Path, stage_directory: Path,
                                    output_directory: Path,
                                    version: str) -> dict[str, Path]:
    products = stage_wheel_products(source_bundle, stage_directory, version)
    remove_distribution_wheels(
        output_directory, (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION)
    )
    runtime_wheel = build_wheel(
        products[RUNTIME_DISTRIBUTION],
        RUNTIME_DISTRIBUTION,
        version,
        output_directory,
        summary="ROCm NPU HRX amdxdna runtime",
        console_scripts={"rocm-npu": f"{RUNTIME_PACKAGE}:main"},
    )
    fastflowlm_wheel = build_wheel(
        products[FASTFLOWLM_DISTRIBUTION],
        FASTFLOWLM_DISTRIBUTION,
        version,
        output_directory,
        summary="ROCm NPU FastFlowLM runtime assets",
        requires=(f"{RUNTIME_DISTRIBUTION} =={version}",),
        console_scripts={"flm": f"{FASTFLOWLM_PACKAGE}:main"},
    )
    return {
        RUNTIME_DISTRIBUTION: runtime_wheel,
        FASTFLOWLM_DISTRIBUTION: fastflowlm_wheel,
    }


def find_rocm_npu_wheels(directory: Path) -> dict[str, Path]:
    return find_distribution_wheels(
        directory, (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION)
    )


def verify_wheels_do_not_have_text_path_references(
    wheels: dict[str, Path], forbidden_paths: tuple[Path, ...]
) -> None:
    """Rejects local paths in text payloads and audits opaque native payloads.

    Native strings are not a sound relocation signal: compilers and third-party
    libraries record source paths in diagnostics and several debug formats. PE
    import closure and clean-install runtime tests enforce native relocation.
    """
    references = find_wheel_path_references(wheels, forbidden_paths)
    text_references = []
    native_references = []
    for reference in references:
        with zipfile.ZipFile(reference.wheel) as archive:
            with archive.open(reference.member) as member:
                is_native = member.read(2) == b"MZ"
        (native_references if is_native else text_references).append(reference)
    if native_references:
        print(
            "wheel path audit: "
            f"{len(native_references)} native member(s) contain local path strings; "
            "native relocation is enforced by PE import and runtime tests"
        )
    if text_references:
        details = "\n  ".join(
            f"{reference.wheel.name}:{reference.member} -> "
            f"{reference.forbidden_path}"
            for reference in text_references
        )
        raise RuntimeError(
            "wheel text payload contains local checkout/build paths:\n  "
            + details
        )
