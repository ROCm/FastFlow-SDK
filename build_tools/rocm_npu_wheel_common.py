# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Platform-neutral mechanics for building and verifying ROCm NPU wheels."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote


FIXED_ZIP_DATE = (2000, 1, 1, 0, 0, 0)
PACKAGE_VERSION_ENV = "ROCM_NPU_PACKAGE_VERSION"
RUNTIME_DISTRIBUTION = "rocm-npu-runtime"
FASTFLOWLM_DISTRIBUTION = "rocm-npu-fastflowlm"
RUNTIME_PACKAGE = "rocm_npu_runtime"
FASTFLOWLM_PACKAGE = "rocm_npu_fastflowlm"
LICENSE_EXPRESSIONS = {
    RUNTIME_DISTRIBUTION: "MIT AND Apache-2.0",
    FASTFLOWLM_DISTRIBUTION:
        "MIT AND LicenseRef-FastFlowLM-Proprietary-Kernel-Terms",
}
_VERSION_RE = re.compile(
    r"^[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+|\.post[0-9]+|\.dev[0-9]+)?"
    r"(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?$"
)


@dataclass(frozen=True)
class WheelPathReference:
    """A forbidden local path embedded in a wheel member."""

    wheel: Path
    member: str
    forbidden_path: Path


def sha256_file(path: Path) -> str:
    """Returns the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_artifact_name(system: str | None = None, machine: str | None = None) -> str:
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower().replace("amd64", "x86_64")
    return f"rocm-npu-runtime-{system}-{machine}"


def fastflowlm_artifact_name(system: str | None = None, machine: str | None = None) -> str:
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower().replace("amd64", "x86_64")
    return f"rocm-npu-fastflowlm-{system}-{machine}"


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        resolved = source.resolve()
        if not resolved.is_file():
            raise RuntimeError(f"wheel payload symlink does not resolve to a file: {source}")
        source = resolved
    shutil.copy2(source, destination)


def _copy_relative_file(source_root: Path, relative: str, destination_root: Path) -> None:
    source = source_root / relative
    if not source.is_file():
        raise RuntimeError(f"required wheel payload file missing: {source}")
    _copy_file(source, destination_root / relative)


def _copy_relative_tree(source_root: Path, relative: str, destination_root: Path) -> None:
    source = source_root / relative
    if not source.is_dir():
        raise RuntimeError(f"required wheel payload directory missing: {source}")
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        _copy_file(path, destination_root / path.relative_to(source_root))


def _copy_license(source_bundle: Path, relative: str, package_root: Path) -> None:
    source = source_bundle / relative
    if not source.exists():
        raise RuntimeError(f"required wheel license metadata missing: {source}")
    destination = package_root / relative
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        _copy_file(source, destination)


def _copy_license_as(source_bundle: Path, relative: str, package_root: Path,
                     destination_relative: str) -> None:
    source = source_bundle / relative
    if not source.is_file():
        raise RuntimeError(f"required wheel license metadata missing: {source}")
    _copy_file(source, package_root / destination_relative)


def _dependency_license_packages(manifest: dict[str, object], shipped_names: set[str]) -> tuple[set[str], set[str]]:
    shipped_names = {name.casefold() for name in shipped_names}
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, dict):
        return set(), set()
    vcpkg = dependencies.get("vcpkg") if dependencies.get("provider") == "vcpkg+system" else dependencies
    system = dependencies.get("system") if dependencies.get("provider") == "vcpkg+system" else {}

    vcpkg_packages = set()
    if isinstance(vcpkg, dict):
        mapping = vcpkg.get("runtime_file_packages")
        if isinstance(mapping, dict):
            vcpkg_packages.update(str(package) for name, package in mapping.items()
                                  if str(name).casefold() in shipped_names)

    system_packages = set()
    if isinstance(system, dict):
        mapping = system.get("library_packages")
        if isinstance(mapping, dict):
            system_packages.update(str(package) for name, package in mapping.items()
                                   if str(name).casefold() in shipped_names)
    return vcpkg_packages, system_packages


def _copy_dependency_licenses(source_bundle: Path, manifest: dict[str, object],
                              package_root: Path, shipped_names: set[str]) -> None:
    vcpkg_packages, system_packages = _dependency_license_packages(manifest, shipped_names)
    for package in sorted(vcpkg_packages):
        source = source_bundle / "licenses" / "vcpkg" / package
        if not source.is_dir():
            raise RuntimeError(
                f"shipped vcpkg dependency {package} is missing license material: "
                f"{source}"
            )
        shutil.copytree(source, package_root / "licenses" / "vcpkg" / package,
                        dirs_exist_ok=True)
    for package in sorted(system_packages):
        source = source_bundle / "licenses" / "system" / package
        if not source.is_dir():
            raise RuntimeError(
                f"shipped system dependency {package} is missing license material: "
                f"{source}"
            )
        shutil.copytree(source, package_root / "licenses" / "system" / package,
                        dirs_exist_ok=True)


def _write_file(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def _file_records(package_root: Path) -> list[dict[str, object]]:
    records = []
    manifest = package_root / "manifest.json"
    for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
        if path == manifest:
            continue
        records.append({
            "path": path.relative_to(package_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def _write_package_manifest(package_root: Path, manifest: dict[str, object]) -> dict[str, object]:
    manifest = dict(manifest, files=_file_records(package_root))
    (package_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_version(version: str) -> str:
    """Validates the supported, normalized subset of PEP 440 versions."""
    if not _VERSION_RE.fullmatch(version):
        raise RuntimeError(
            f"invalid PEP 440 package version for rocm-npu wheels: {version}"
        )
    return version


def default_package_version(
    root: Path, *, environment_variable: str = PACKAGE_VERSION_ENV
) -> str:
    """Returns an explicit package version or a revision-derived development version."""
    version = os.environ.get(environment_variable)
    if version:
        return validate_version(version)
    revision = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"0.0.0.dev0+g{revision}"


def _wheel_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii")


def _record_csv(rows: list[tuple[str, str, str]]) -> str:
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue()


def _zip_info(name: str, mode: int = 0o100644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = mode << 16
    return info


def _write_wheel_entry(
    output: zipfile.ZipFile,
    records: list[tuple[str, str, str]],
    name: str,
    data: bytes,
    mode: int = 0o100644,
) -> None:
    output.writestr(_zip_info(name, mode), data, compresslevel=9)
    records.append((name, _wheel_hash(data), str(len(data))))


def _wheel_file_mode(relative: str, source_mode: int) -> int:
    parts = PurePosixPath(relative).parts
    is_package_bin_payload = len(parts) >= 3 and parts[1] == "bin"
    if source_mode & 0o111 or is_package_bin_payload:
        return 0o100755
    return 0o100644


def _package_manifest(tree_root: Path) -> dict[str, object]:
    candidates = sorted(tree_root.glob("*/manifest.json"))
    if not candidates:
        return {}
    if len(candidates) != 1:
        raise RuntimeError(
            f"wheel tree must contain at most one package manifest: {tree_root}"
        )
    manifest = json.loads(candidates[0].read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError(f"wheel package manifest is not an object: {candidates[0]}")
    return manifest


def _license_entries(tree_root: Path) -> list[tuple[str, bytes]]:
    entries: dict[str, bytes] = {}
    for path in sorted(item for item in tree_root.rglob("*") if item.is_file()):
        relative = path.relative_to(tree_root)
        if len(relative.parts) < 3 or relative.parts[1] != "licenses":
            continue
        name = PurePosixPath(*relative.parts[2:]).as_posix()
        data = path.read_bytes()
        previous = entries.setdefault(name, data)
        if previous != data:
            raise RuntimeError(
                f"wheel license destination has conflicting contents: {name}"
            )
    return sorted(entries.items())


def _component_purl(kind: str, name: str, version: str) -> str:
    suffix = f"@{quote(version, safe='.+-')}" if version else ""
    return f"pkg:{kind}/{quote(name, safe='._-')}{suffix}"


def _dependency_components(
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, dict):
        return []
    files = manifest.get("files")
    shipped_names = {
        PurePosixPath(str(record.get("path", ""))).name.casefold()
        for record in files if isinstance(record, dict)
    } if isinstance(files, list) else set()

    provider = dependencies.get("provider")
    vcpkg = dependencies.get("vcpkg") if provider == "vcpkg+system" else dependencies
    system = dependencies.get("system") if provider == "vcpkg+system" else {}
    components: list[dict[str, object]] = []

    if isinstance(vcpkg, dict):
        runtime_files = vcpkg.get("runtime_file_packages")
        selected = {
            str(package)
            for name, package in runtime_files.items()
            if str(name).casefold() in shipped_names
        } if isinstance(runtime_files, dict) else set()
        resolved = vcpkg.get("resolved_packages")
        records = resolved if isinstance(resolved, list) else []
        versions = {
            str(record.get("package")): str(record.get("version", ""))
            for record in records
            if isinstance(record, dict) and record.get("package")
        }
        for name in sorted(selected):
            version = versions.get(name, "")
            component: dict[str, object] = {
                "type": "library",
                "bom-ref": f"vcpkg:{name}@{version}",
                "name": name,
                "properties": [{"name": "rocm-npu:provider", "value": "vcpkg"}],
            }
            if version:
                component["version"] = version
                component["purl"] = _component_purl("generic/vcpkg", name, version)
            components.append(component)

        microsoft_files = vcpkg.get("microsoft_runtime_files")
        shipped_microsoft = sorted(
            str(name) for name in microsoft_files
            if str(name).casefold() in shipped_names
        ) if isinstance(microsoft_files, list) else []
        if shipped_microsoft:
            components.append({
                "type": "framework",
                "bom-ref": "microsoft:visual-cpp-runtime",
                "name": "Microsoft Visual C++ Runtime",
                "supplier": {"name": "Microsoft Corporation"},
                "licenses": [{
                    "license": {
                        "name": "Microsoft Software License Terms",
                        "url": str(vcpkg.get(
                            "microsoft_runtime_license",
                            "https://visualstudio.microsoft.com/license-terms/",
                        )),
                    },
                }],
                "externalReferences": [{
                    "type": "license",
                    "url": str(vcpkg.get(
                        "microsoft_runtime_license",
                        "https://visualstudio.microsoft.com/license-terms/",
                    )),
                }],
                "properties": [{
                    "name": "rocm-npu:runtime-files",
                    "value": ",".join(shipped_microsoft),
                }],
            })

    if isinstance(system, dict):
        library_packages = system.get("library_packages")
        selected = {
            str(package)
            for name, package in library_packages.items()
            if str(name).casefold() in shipped_names
        } if isinstance(library_packages, dict) else set()
        packages = system.get("packages")
        records = packages if isinstance(packages, list) else []
        versions = {
            str(record.get("name")): str(record.get("version", ""))
            for record in records
            if isinstance(record, dict) and record.get("name")
        }
        for name in sorted(selected):
            version = versions.get(name, "")
            provider_name = str(system.get("provider", "system"))
            component = {
                "type": "library",
                "bom-ref": f"system:{name}@{version}",
                "name": name,
                "properties": [{
                    "name": "rocm-npu:provider",
                    "value": provider_name,
                }],
            }
            if version:
                component["version"] = version
                purl_type = {"dpkg": "deb", "rpm": "rpm"}.get(provider_name)
                if purl_type:
                    component["purl"] = _component_purl(purl_type, name, version)
            components.append(component)
    return components


def _required_distribution_components(
    manifest: dict[str, object],
) -> list[dict[str, object]]:
    requires = manifest.get("requires")
    if not isinstance(requires, list):
        return []
    components = []
    for requirement in requires:
        match = re.fullmatch(
            r"\s*([A-Za-z0-9_.-]+)\s*==\s*([A-Za-z0-9_.+!-]+)\s*",
            str(requirement),
        )
        if not match:
            raise RuntimeError(
                f"wheel SBOM requires an exact distribution dependency: {requirement}"
            )
        name, version = match.groups()
        purl = _component_purl("pypi", name, version)
        component: dict[str, object] = {
            "type": "library",
            "bom-ref": purl,
            "name": name,
            "version": version,
            "purl": purl,
            "supplier": {"name": "Advanced Micro Devices, Inc."},
        }
        license_expression = LICENSE_EXPRESSIONS.get(name)
        if license_expression:
            component["licenses"] = [{"expression": license_expression}]
        components.append(component)
    return components


def _file_components(manifest: dict[str, object]) -> list[dict[str, object]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        return []
    components = []
    for record in files:
        if not isinstance(record, dict):
            raise RuntimeError("wheel manifest file record is not an object")
        path = str(record.get("path", ""))
        digest = str(record.get("sha256", ""))
        size = record.get("size")
        if (
            not path
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(size, int)
            or size < 0
        ):
            raise RuntimeError(f"invalid wheel manifest file record: {record}")
        components.append({
            "type": "file",
            "bom-ref": f"file:{path}",
            "name": path,
            "hashes": [{"alg": "SHA-256", "content": digest}],
            "properties": [{
                "name": "rocm-npu:size",
                "value": str(size),
            }],
        })
    return components


def _source_pedigree(manifest: dict[str, object]) -> dict[str, object] | None:
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        return None
    ancestors = []
    for name, state in sorted(sources.items()):
        if not isinstance(state, dict) or not state.get("revision"):
            continue
        revision = str(state["revision"])
        properties = [{
            "name": "rocm-npu:source-dirty",
            "value": str(bool(state.get("dirty"))).lower(),
        }]
        if state.get("tracked_diff_sha256"):
            properties.append({
                "name": "rocm-npu:tracked-diff-sha256",
                "value": str(state["tracked_diff_sha256"]),
            })
        ancestors.append({
            "type": "application" if name == "rocm-npu-staging" else "library",
            "bom-ref": f"source:{name}@{revision}",
            "name": name,
            "version": revision,
            "properties": properties,
        })
    return {"ancestors": ancestors} if ancestors else None


def _tool_components(manifest: dict[str, object]) -> list[dict[str, str]]:
    toolchain = manifest.get("toolchain")
    if not isinstance(toolchain, dict):
        return []
    tools = []
    for name in (
        "python", "git", "cmake", "ninja", "rustc", "cargo",
        "zstd", "patchelf", "ccache",
    ):
        value = toolchain.get(name)
        if value:
            tools.append({"name": name, "version": str(value)})
    return tools


def _verify_sbom_file_inventory(
    sbom: dict[str, object],
    manifest: dict[str, object],
) -> None:
    files = manifest.get("files")
    expected = {
        str(record["path"]): str(record["sha256"])
        for record in files
        if isinstance(record, dict) and "path" in record and "sha256" in record
    } if isinstance(files, list) else {}
    actual = {}
    components = sbom.get("components")
    for component in components if isinstance(components, list) else []:
        if not isinstance(component, dict) or component.get("type") != "file":
            continue
        hashes = component.get("hashes")
        sha256 = next(
            (
                str(item.get("content"))
                for item in hashes
                if isinstance(item, dict) and item.get("alg") == "SHA-256"
            ),
            "",
        ) if isinstance(hashes, list) else ""
        actual[str(component.get("name", ""))] = sha256
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatched = sorted(
            path for path in set(actual).intersection(expected)
            if actual[path] != expected[path]
        )
        raise RuntimeError(
            "CycloneDX file inventory differs from wheel manifest: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )


def _cyclonedx_sbom(
    distribution: str,
    version: str,
    tag: str,
    manifest: dict[str, object],
    license_expression: str,
) -> bytes:
    dependency_components = [
        *_required_distribution_components(manifest),
        *_dependency_components(manifest),
    ]
    file_components = _file_components(manifest)
    components = [*dependency_components, *file_components]
    root_ref = f"pkg:pypi/{distribution}@{version}"
    identity = json.dumps(
        [distribution, version, tag, manifest], sort_keys=True, separators=(",", ":")
    )
    root_component: dict[str, object] = {
        "type": "application",
        "bom-ref": root_ref,
        "name": distribution,
        "version": version,
        "purl": root_ref,
        "supplier": {"name": "Advanced Micro Devices, Inc."},
        "licenses": [{"expression": license_expression}],
        "properties": [{
            "name": "rocm-npu:wheel-tag",
            "value": tag,
        }],
    }
    pedigree = _source_pedigree(manifest)
    if pedigree:
        root_component["pedigree"] = pedigree
    metadata: dict[str, object] = {"component": root_component}
    tools = _tool_components(manifest)
    if tools:
        metadata["tools"] = tools
    sbom: dict[str, object] = {
        "$schema": "https://cyclonedx.org/schema/bom-1.6.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, identity)}",
        "version": 1,
        "metadata": metadata,
        "dependencies": [{
            "ref": root_ref,
            "dependsOn": [str(component["bom-ref"]) for component in components],
        }],
    }
    if components:
        sbom["components"] = components
    _verify_sbom_file_inventory(sbom, manifest)
    return (json.dumps(sbom, indent=2, sort_keys=True) + "\n").encode("utf-8")


def wheel_filename(distribution: str, version: str, tag: str) -> str:
    normalized_name = distribution.replace("-", "_")
    normalized_version = version.replace("-", "_")
    return f"{normalized_name}-{normalized_version}-{tag}.whl"


def build_wheel(
    tree_root: Path,
    distribution: str,
    version: str,
    output_directory: Path,
    *,
    summary: str,
    tag: str,
    requires: tuple[str, ...] = (),
    console_scripts: dict[str, str] | None = None,
    license_expression: str | None = None,
) -> Path:
    """Builds a deterministic wheel from an already staged package tree."""
    version = validate_version(version)
    output_directory.mkdir(parents=True, exist_ok=True)
    dist_info = (
        f"{distribution.replace('-', '_')}-{version.replace('-', '_')}.dist-info"
    )
    license_expression = license_expression or LICENSE_EXPRESSIONS.get(distribution)
    if not license_expression:
        raise RuntimeError(
            f"wheel license expression is required for distribution: {distribution}"
        )
    license_entries = _license_entries(tree_root)
    package_manifest = _package_manifest(tree_root)
    manifest_requires = package_manifest.get("requires")
    if manifest_requires is not None and list(requires) != manifest_requires:
        raise RuntimeError(
            "wheel package manifest requirements differ from METADATA requirements"
        )
    package_manifest = dict(package_manifest, requires=list(requires))
    metadata_lines = [
        "Metadata-Version: 2.4",
        f"Name: {distribution}",
        f"Version: {version}",
        f"Summary: {summary}",
        "Requires-Python: >=3.11",
        f"License-Expression: {license_expression}",
        "Description-Content-Type: text/markdown",
    ]
    metadata_lines.extend(f"License-File: {name}" for name, _ in license_entries)
    metadata_lines.extend(f"Requires-Dist: {requirement}" for requirement in requires)
    metadata = "\n".join(metadata_lines) + f"\n\n{summary}\n"
    wheel_metadata = (
        "\n".join(
            [
                "Wheel-Version: 1.0",
                "Generator: rocm-npu-wheel-common",
                "Root-Is-Purelib: false",
                f"Tag: {tag}",
            ]
        )
        + "\n"
    )

    wheel = output_directory / wheel_filename(distribution, version, tag)
    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(
        wheel, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as output:
        for path in sorted(item for item in tree_root.rglob("*") if item.is_file()):
            relative = path.relative_to(tree_root).as_posix()
            mode = stat.S_IMODE(path.stat().st_mode)
            _write_wheel_entry(
                output,
                records,
                relative,
                path.read_bytes(),
                _wheel_file_mode(relative, mode),
            )
        for name, data in license_entries:
            _write_wheel_entry(
                output, records, f"{dist_info}/licenses/{name}", data
            )
        _write_wheel_entry(
            output,
            records,
            f"{dist_info}/sboms/cyclonedx.json",
            _cyclonedx_sbom(
                distribution,
                version,
                tag,
                package_manifest,
                license_expression,
            ),
        )
        _write_wheel_entry(
            output, records, f"{dist_info}/METADATA", metadata.encode("utf-8")
        )
        if console_scripts:
            entry_points = "[console_scripts]\n" + "".join(
                f"{name} = {target}\n"
                for name, target in sorted(console_scripts.items())
            )
            _write_wheel_entry(
                output,
                records,
                f"{dist_info}/entry_points.txt",
                entry_points.encode("utf-8"),
            )
        _write_wheel_entry(
            output, records, f"{dist_info}/WHEEL", wheel_metadata.encode("utf-8")
        )
        record_name = f"{dist_info}/RECORD"
        record = _record_csv([*records, (record_name, "", "")]).encode("utf-8")
        output.writestr(_zip_info(record_name), record, compresslevel=9)
    wheel.with_name(wheel.name + ".sha256").write_text(
        f"{sha256_file(wheel)}  {wheel.name}\n", encoding="ascii"
    )
    return wheel


def find_distribution_wheels(
    directory: Path, distributions: tuple[str, ...]
) -> dict[str, Path]:
    """Finds exactly one wheel for each requested distribution."""
    directory = Path(directory)
    wheels: dict[str, Path] = {}
    counts = {}
    for distribution in distributions:
        matches = sorted(directory.glob(f"{distribution.replace('-', '_')}-*.whl"))
        counts[distribution] = len(matches)
        if len(matches) == 1:
            wheels[distribution] = matches[0]
    if len(wheels) != len(distributions):
        detail = ", ".join(f"{name}={counts[name]}" for name in distributions)
        raise RuntimeError(
            f"expected one wheel per distribution in {directory}; found {detail}"
        )
    return wheels


def remove_distribution_wheels(
    directory: Path, distributions: tuple[str, ...]
) -> None:
    """Removes stale wheels and checksum sidecars for the requested products."""
    directory = Path(directory)
    for distribution in distributions:
        prefix = distribution.replace("-", "_")
        for pattern in (f"{prefix}-*.whl", f"{prefix}-*.whl.sha256"):
            for artifact in directory.glob(pattern):
                artifact.unlink()


def verify_wheel_checksums(wheels: dict[str, Path]) -> None:
    """Verifies every wheel's required SHA-256 sidecar."""
    for wheel in wheels.values():
        checksum = wheel.with_name(wheel.name + ".sha256")
        if not checksum.is_file():
            raise RuntimeError(f"wheel checksum missing: {checksum}")
        fields = checksum.read_text(encoding="ascii").strip().split()
        if len(fields) != 2 or fields[1] != wheel.name:
            raise RuntimeError(f"malformed wheel checksum: {checksum}")
        if sha256_file(wheel) != fields[0]:
            raise RuntimeError(f"wheel checksum mismatch: {wheel}")


def path_needles(path: Path) -> tuple[bytes, ...]:
    """Returns case-normalized UTF-8 and UTF-16 spellings of a local path."""
    resolved = str(path.resolve())
    spellings = {
        resolved,
        resolved.replace("\\", "/"),
        resolved.replace("/", "\\"),
    }
    spellings.update(spelling.casefold() for spelling in tuple(spellings))
    return tuple(
        dict.fromkeys(
            encoded
            for spelling in sorted(spellings)
            for encoded in (spelling.encode("utf-8"), spelling.encode("utf-16le"))
        )
    )


def _stream_path_reference(
    stream, needles: list[tuple[Path, tuple[bytes, ...]]]
) -> Path | None:
    folded_needles = [
        (path, tuple(needle.lower() for needle in variants))
        for path, variants in needles
    ]
    overlap = max(
        (len(needle) - 1 for _, variants in folded_needles for needle in variants),
        default=0,
    )
    carry = b""
    while chunk := stream.read(1024 * 1024):
        data = carry + chunk
        folded = data.lower()
        for path, variants in folded_needles:
            if any(needle in folded for needle in variants):
                return path
        carry = data[-overlap:] if overlap else b""
    return None


def find_wheel_path_references(
    wheels: dict[str, Path], forbidden_paths: tuple[Path, ...]
) -> list[WheelPathReference]:
    """Returns local path references in text or native wheel payloads."""
    references = []
    needles = [(path.resolve(), path_needles(path)) for path in forbidden_paths]
    for wheel in wheels.values():
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                with archive.open(member) as stream:
                    forbidden_path = _stream_path_reference(stream, needles)
                if forbidden_path is not None:
                    references.append(
                        WheelPathReference(
                            wheel=wheel,
                            member=member.filename,
                            forbidden_path=forbidden_path,
                        )
                    )
    return references
