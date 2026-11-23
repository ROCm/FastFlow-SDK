# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Repack one release's component wheels into the single PyPI ``fastflow-sdk`` wheel.

PyPI gets one wheel per platform instead of the component wheels, so one
``pip install fastflow-sdk`` resolves without the project's package index. The
wheel is a lossless merge of the qualified release wheels: every payload file
keeps its bytes and file mode, so the PyPI install runs exactly what the
release qualified. Only the ``fastflow-sdk`` dist-info is new.
"""

from __future__ import annotations

import configparser
import csv
import email.parser
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from email.message import Message
from pathlib import Path

from build_tools import paths
from build_tools.artifact import write_sha256_sidecar
from build_tools.fastflow_sdk_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    FASTFLOWLM_DISTRIBUTION,
    OPS_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
    SDK_DISTRIBUTION,
    _record_csv,
    _wheel_hash,
    _zip_info,
    wheel_filename,
)
from build_tools.manylinux_audit import auditwheel_policy, validate_manylinux_policy
from build_tools.package_version import validate_version
from build_tools.validate_fastflow_cli import run_reports

PYPI_COMPONENTS = (
    SDK_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
    FASTFLOWLM_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_BINDINGS_DISTRIBUTION,
    OPS_DISTRIBUTION,
)
# Read back by fastflow_sdk_qualification.installation.bundled_component_versions.
SDK_COMPONENTS_FILE = "fastflow-sdk-components.json"
PYPI_PLATFORMS = ("linux", "windows")
PYPI_WHEEL_ARTIFACTS = paths.BUILD / "pypi-wheel"
PYPI_WHEEL_VENV = paths.BUILD / "pypi-wheel-venv"
PYPI_WHEEL_REPORTS = paths.BUILD / "pypi-wheel-reports"
PYPI_SUMMARY = (
    "AMD FastFlow SDK for Ryzen AI NPUs: FastFlowLM, the IRON compiler and "
    "operators, and the HRX runtime"
)
PYPI_PROJECT_URLS = (
    ("Documentation", "https://rocm.github.io/FastFlow-SDK/"),
    ("Source", "https://github.com/ROCm/FastFlow-SDK"),
    ("Releases", "https://github.com/ROCm/FastFlow-SDK/releases"),
    ("Issues", "https://github.com/ROCm/FastFlow-SDK/issues"),
)
PYPI_DESCRIPTION = Path(__file__).with_name("fastflow_sdk_pypi_description.md")
_COMPONENT_DIST_INFO_FILES = frozenset(
    ("METADATA", "WHEEL", "RECORD", "entry_points.txt")
)
_REQUIREMENT = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*)")
_PYTHON_BOUND = re.compile(r"(>=|<)\s*(\d+(?:\.\d+)*)")
_MANYLINUX = re.compile(r"manylinux_(\d+)_(\d+)_x86_64")


@dataclass(frozen=True)
class ComponentWheel:
    """A verified release wheel and the metadata the merge needs from it."""

    distribution: str
    path: Path
    sha256: str
    version: str
    tag: tuple[str, str, str]
    dist_info: str
    metadata: Message
    entry_points: dict[str, dict[str, str]]
    records: dict[str, tuple[str, int]]


def _write_entry(
    output: zipfile.ZipFile,
    records: list[tuple[str, str, str]],
    name: str,
    data: bytes,
    mode: int = 0o100644,
    compression_level: int = 9,
) -> None:
    info = _zip_info(name, mode)
    # zipfile records the building OS; installers such as uv ignore the Unix
    # mode bits of entries marked as MS-DOS, which drops executable bits.
    info.create_system = 3
    output.writestr(info, data, compresslevel=compression_level)
    records.append((name, _wheel_hash(data), str(len(data))))


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_wheel_name(path: Path) -> tuple[str, str, tuple[str, str, str]]:
    parts = path.name.removesuffix(".whl").split("-")
    if len(parts) != 5:
        raise RuntimeError(f"unsupported wheel filename: {path.name}")
    name, version, python_tag, abi_tag, platform_tag = parts
    if any("." in tag for tag in (python_tag, abi_tag, platform_tag)):
        raise RuntimeError(f"compressed wheel tag sets are unsupported: {path.name}")
    return _normalize(name), version, (python_tag, abi_tag, platform_tag)


def _platform_matches(platform_tag: str, platform: str) -> bool:
    if platform_tag == "any":
        return True
    if platform == "linux":
        return platform_tag.startswith(("manylinux_", "linux_"))
    return platform_tag == "win_amd64"


def read_sha256sums(path: Path) -> dict[str, str]:
    """Reads a ``sha256sum`` listing keyed by file name."""
    digests = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        name = name.lstrip("*")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or not name:
            raise RuntimeError(f"invalid SHA256SUMS line in {path}: {line!r}")
        digests[name] = digest
    return digests


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_component_wheels(release_dir: Path, platform: str) -> dict[str, Path]:
    """Finds exactly one release wheel per PyPI component for ``platform``."""
    if platform not in PYPI_PLATFORMS:
        raise RuntimeError(f"unsupported PyPI wheel platform: {platform}")
    selected: dict[str, list[Path]] = {name: [] for name in PYPI_COMPONENTS}
    for wheel in sorted(Path(release_dir).glob("*.whl")):
        name, _, tag = _parse_wheel_name(wheel)
        if name in selected and _platform_matches(tag[2], platform):
            selected[name].append(wheel)
    counts = {name: len(wheels) for name, wheels in selected.items()}
    if any(count != 1 for count in counts.values()):
        detail = ", ".join(f"{name}={count}" for name, count in counts.items())
        raise RuntimeError(
            f"expected one {platform} wheel per component in {release_dir}; "
            f"found {detail}"
        )
    return {name: wheels[0] for name, wheels in selected.items()}


def _read_records(
    archive: zipfile.ZipFile, record_name: str, wheel: Path
) -> dict[str, tuple[str, int]]:
    rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
    records = {}
    for row in rows:
        if len(row) != 3:
            raise RuntimeError(f"invalid RECORD row in {wheel.name}: {row!r}")
        name, digest, size = row
        if name == record_name:
            continue
        if name in records or not digest.startswith("sha256=") or not size.isdigit():
            raise RuntimeError(f"invalid RECORD row in {wheel.name}: {row!r}")
        records[name] = (digest, int(size))
    members = {
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and info.filename != record_name
    }
    if members != set(records):
        unrecorded = sorted(members - set(records))
        missing = sorted(set(records) - members)
        raise RuntimeError(
            f"RECORD does not match the members of {wheel.name}: "
            f"unrecorded={unrecorded[:5]}, missing={missing[:5]}"
        )
    return records


def _read_entry_points(text: str) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser(interpolation=None, delimiters=("=",))
    parser.optionxform = str
    parser.read_string(text)
    return {section: dict(parser[section]) for section in parser.sections()}


def load_component_wheel(
    distribution: str, wheel: Path, digests: dict[str, str]
) -> ComponentWheel:
    """Verifies a release wheel against the release digests and its RECORD."""
    expected = digests.get(wheel.name)
    if expected is None:
        raise RuntimeError(f"release SHA256SUMS does not list {wheel.name}")
    actual = _sha256_file(wheel)
    if actual != expected:
        raise RuntimeError(
            f"release wheel digest mismatch: {wheel.name}: "
            f"expected {expected}, got {actual}"
        )
    name, version, tag = _parse_wheel_name(wheel)
    if name != distribution:
        raise RuntimeError(f"{wheel.name} is not a {distribution} wheel")
    dist_info = f"{distribution.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        records = _read_records(archive, f"{dist_info}/RECORD", wheel)
        for member in records:
            top = member.split("/", 1)[0]
            if top.endswith(".data"):
                raise RuntimeError(
                    f"wheel data directories are unsupported: {wheel.name}: {member}"
                )
            if top.endswith(".dist-info") and top != dist_info:
                raise RuntimeError(f"unexpected dist-info in {wheel.name}: {member}")
            if top == dist_info:
                relative = member.removeprefix(f"{dist_info}/")
                if relative not in _COMPONENT_DIST_INFO_FILES and not (
                    relative.startswith("licenses/")
                    or relative == "sboms/cyclonedx.json"
                ):
                    raise RuntimeError(
                        f"unsupported dist-info file in {wheel.name}: {member}"
                    )
                data = archive.read(member)
                if (_wheel_hash(data), len(data)) != records[member]:
                    raise RuntimeError(f"{wheel.name} RECORD does not match {member}")
        parser = email.parser.Parser()
        metadata = parser.parsestr(archive.read(f"{dist_info}/METADATA").decode())
        wheel_metadata = parser.parsestr(archive.read(f"{dist_info}/WHEEL").decode())
        entry_points_name = f"{dist_info}/entry_points.txt"
        entry_points = (
            _read_entry_points(archive.read(entry_points_name).decode("utf-8"))
            if entry_points_name in records
            else {}
        )
    if _normalize(metadata["Name"]) != distribution or metadata["Version"] != version:
        raise RuntimeError(f"METADATA name or version mismatch in {wheel.name}")
    if wheel_metadata.get_all("Tag") != ["-".join(tag)]:
        raise RuntimeError(f"WHEEL tag does not match the filename of {wheel.name}")
    return ComponentWheel(
        distribution=distribution,
        path=wheel,
        sha256=actual,
        version=version,
        tag=tag,
        dist_info=dist_info,
        metadata=metadata,
        entry_points=entry_points,
        records=records,
    )


def merge_requirements(components: list[ComponentWheel], version: str) -> list[str]:
    """Drops the internal component pins and unions the external requirements."""
    internal = set(PYPI_COMPONENTS)
    external: dict[str, str] = {}
    for component in components:
        for requirement in component.metadata.get_all("Requires-Dist") or ():
            match = _REQUIREMENT.fullmatch(requirement.strip())
            if not match:
                raise RuntimeError(
                    f"unsupported requirement in {component.distribution}: "
                    f"{requirement!r}"
                )
            name = _normalize(match.group(1))
            if name in internal:
                if match.group(2).replace(" ", "") != f"=={version}":
                    raise RuntimeError(
                        f"{component.distribution} must pin {name} =={version}: "
                        f"{requirement!r}"
                    )
                continue
            if name.startswith("fastflow-sdk"):
                raise RuntimeError(
                    f"{component.distribution} requires {name}, which the PyPI "
                    "wheel does not bundle"
                )
            previous = external.setdefault(name, requirement)
            if previous != requirement:
                raise RuntimeError(
                    f"conflicting requirements for {name}: {previous!r} and "
                    f"{requirement!r}"
                )
    return [external[name] for name in sorted(external)]


def merge_requires_python(components: list[ComponentWheel]) -> str:
    """Intersects ``>=``/``<`` Requires-Python ranges."""
    lower: tuple[int, ...] | None = None
    upper: tuple[int, ...] | None = None
    for component in components:
        specifier = component.metadata.get("Requires-Python", "")
        for clause in filter(None, (part.strip() for part in specifier.split(","))):
            match = _PYTHON_BOUND.fullmatch(clause)
            if not match:
                raise RuntimeError(
                    f"unsupported Requires-Python in {component.distribution}: "
                    f"{specifier!r}"
                )
            bound = tuple(int(part) for part in match.group(2).split("."))
            if match.group(1) == ">=":
                lower = bound if lower is None else max(lower, bound)
            else:
                upper = bound if upper is None else min(upper, bound)
    if lower is None:
        raise RuntimeError("components do not declare a minimum Python version")
    if upper is not None and upper <= lower:
        raise RuntimeError("components have no common Python version")
    clauses = [">=" + ".".join(map(str, lower))]
    if upper is not None:
        clauses.append("<" + ".".join(map(str, upper)))
    return ",".join(clauses)


def _and_terms(expression: str) -> list[str]:
    terms, depth, start = [], 0, 0
    for index, character in enumerate(expression):
        depth += {"(": 1, ")": -1}.get(character, 0)
        if depth == 0 and expression.startswith(" AND ", index):
            terms.append(expression[start:index].strip())
            start = index + len(" AND ")
    terms.append(expression[start:].strip())
    for term in terms:
        if " OR " in term and not (term.startswith("(") and term.endswith(")")):
            raise RuntimeError(f"unsupported license expression: {expression!r}")
    return terms


def merge_license_expression(components: list[ComponentWheel]) -> str:
    """Joins every component's license terms, since the wheel ships them all."""
    terms: list[str] = []
    for component in components:
        expression = component.metadata.get("License-Expression")
        if not expression:
            raise RuntimeError(f"{component.distribution} has no License-Expression")
        for term in _and_terms(expression):
            if term not in terms:
                terms.append(term)
    return " AND ".join(terms)


def merge_entry_points(components: list[ComponentWheel]) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for component in components:
        for section, entries in component.entry_points.items():
            target = merged.setdefault(section, {})
            for name, value in entries.items():
                previous = target.setdefault(name, value)
                if previous != value:
                    raise RuntimeError(
                        f"conflicting {section} entry point {name}: "
                        f"{previous!r} and {value!r}"
                    )
    return merged


def merged_wheel_tag(components: list[ComponentWheel], platform: str) -> str:
    """Returns the tag satisfying every component's interpreter and platform."""
    interpreters = {
        component.tag[:2]
        for component in components
        if component.tag[:2] != ("py3", "none")
    }
    if len(interpreters) > 1:
        raise RuntimeError(f"components need different interpreters: {interpreters}")
    python_tag, abi_tag = interpreters.pop() if interpreters else ("py3", "none")
    platform_tags = {
        component.tag[2] for component in components if component.tag[2] != "any"
    }
    if platform == "windows":
        if platform_tags != {"win_amd64"}:
            raise RuntimeError(f"unexpected Windows platform tags: {platform_tags}")
        return f"{python_tag}-{abi_tag}-win_amd64"
    glibc = []
    for tag in sorted(platform_tags):
        match = _MANYLINUX.fullmatch(tag)
        if not match:
            offenders = ", ".join(
                sorted(
                    component.path.name
                    for component in components
                    if component.tag[2] == tag
                )
            )
            raise RuntimeError(
                f"PyPI accepts only manylinux Linux wheels; {offenders} is tagged {tag}"
            )
        glibc.append((int(match.group(1)), int(match.group(2))))
    if not glibc:
        raise RuntimeError("no Linux platform wheel among the components")
    major, minor = max(glibc)
    return f"{python_tag}-{abi_tag}-manylinux_{major}_{minor}_x86_64"


def _sdk_metadata(
    version: str,
    components: list[ComponentWheel],
    license_files: list[str],
) -> bytes:
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {SDK_DISTRIBUTION}",
        f"Version: {version}",
        f"Summary: {PYPI_SUMMARY}",
        "Author: Advanced Micro Devices, Inc.",
        f"Requires-Python: {merge_requires_python(components)}",
        f"License-Expression: {merge_license_expression(components)}",
        "Description-Content-Type: text/markdown",
    ]
    lines.extend(f"Project-URL: {label}, {url}" for label, url in PYPI_PROJECT_URLS)
    lines.extend(f"License-File: {name}" for name in license_files)
    lines.extend(
        f"Requires-Dist: {requirement}"
        for requirement in merge_requirements(components, version)
    )
    description = PYPI_DESCRIPTION.read_text(encoding="utf-8")
    return ("\n".join(lines) + "\n\n" + description).encode("utf-8")


def _entry_points_text(entry_points: dict[str, dict[str, str]]) -> bytes:
    sections = []
    for section in sorted(entry_points):
        body = "".join(
            f"{name} = {value}\n"
            for name, value in sorted(entry_points[section].items())
        )
        sections.append(f"[{section}]\n{body}")
    return "\n".join(sections).encode("utf-8")


def _components_record(
    version: str, tag: str, components: list[ComponentWheel]
) -> bytes:
    record = {
        "schema_version": 1,
        "distribution": SDK_DISTRIBUTION,
        "version": version,
        "tag": tag,
        "components": [
            {
                "distribution": component.distribution,
                "version": component.version,
                "wheel": component.path.name,
                "sha256": component.sha256,
            }
            for component in components
        ],
    }
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _payload_owners(components: list[ComponentWheel]) -> dict[str, ComponentWheel]:
    owners: dict[str, ComponentWheel] = {}
    for component in components:
        for member in component.records:
            if member.startswith(f"{component.dist_info}/"):
                continue
            previous = owners.setdefault(member, component)
            if previous is not component:
                raise RuntimeError(
                    f"{member} is shipped by both {previous.distribution} and "
                    f"{component.distribution}"
                )
    return owners


def _verify_output(
    wheel: Path,
    dist_info: str,
    owners: dict[str, ComponentWheel],
    modes: dict[str, int],
) -> None:
    """Re-reads the output and checks it against its RECORD and the components."""
    with zipfile.ZipFile(wheel) as archive:
        records = _read_records(archive, f"{dist_info}/RECORD", wheel)
        infos = {info.filename: info for info in archive.infolist()}
        for member, (digest, size) in records.items():
            data = archive.read(member)
            if _wheel_hash(data) != digest or len(data) != size:
                raise RuntimeError(f"output RECORD mismatch: {member}")
            owner = owners.get(member)
            if owner is None:
                if not member.startswith(f"{dist_info}/"):
                    raise RuntimeError(f"output has an unexpected file: {member}")
                continue
            if owner.records[member] != (digest, size):
                raise RuntimeError(
                    f"output differs from {owner.distribution}: {member}"
                )
            info = infos[member]
            if info.external_attr >> 16 != modes[member] or info.create_system != 3:
                raise RuntimeError(f"output file mode differs: {member}")
    missing = sorted(set(owners) - set(records))
    if missing:
        raise RuntimeError(f"output is missing component files: {missing[:5]}")


def build_pypi_wheel(
    release_dir: Path,
    platform: str,
    output_directory: Path,
    *,
    sha256sums: Path | None = None,
    compression_level: int = 9,
) -> Path:
    """Merges the component wheels of one release into the PyPI wheel."""
    release_dir = Path(release_dir)
    digests = read_sha256sums(sha256sums or release_dir / "SHA256SUMS")
    wheels = select_component_wheels(release_dir, platform)
    components = [
        load_component_wheel(distribution, wheels[distribution], digests)
        for distribution in PYPI_COMPONENTS
    ]
    versions = {component.version for component in components}
    if len(versions) != 1:
        raise RuntimeError(f"components have different versions: {sorted(versions)}")
    version = validate_version(versions.pop())
    tag = merged_wheel_tag(components, platform)
    entry_points = merge_entry_points(components)
    owners = _payload_owners(components)
    dist_info = f"{SDK_DISTRIBUTION.replace('-', '_')}-{version}.dist-info"

    licenses: list[tuple[str, ComponentWheel, str]] = []
    sboms: list[tuple[str, ComponentWheel, str]] = []
    for component in components:
        prefix = f"{component.dist_info}/"
        for member in sorted(component.records):
            relative = member.removeprefix(prefix)
            if relative.startswith("licenses/"):
                name = f"{component.distribution}/{relative.removeprefix('licenses/')}"
                licenses.append((name, component, member))
            elif relative == "sboms/cyclonedx.json":
                sboms.append(
                    (f"{component.distribution}.cyclonedx.json", component, member)
                )
    metadata = _sdk_metadata(version, components, [name for name, _, _ in licenses])

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    wheel = output_directory / wheel_filename(SDK_DISTRIBUTION, version, tag)
    records: list[tuple[str, str, str]] = []
    modes: dict[str, int] = {}
    sources = {
        component.distribution: zipfile.ZipFile(component.path)
        for component in components
    }
    try:
        with zipfile.ZipFile(
            wheel,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compression_level,
        ) as output:
            for member in sorted(owners):
                owner = owners[member]
                source = sources[owner.distribution]
                info = source.getinfo(member)
                data = source.read(info)
                if (_wheel_hash(data), len(data)) != owner.records[member]:
                    raise RuntimeError(
                        f"{owner.path.name} RECORD does not match {member}"
                    )
                modes[member] = (info.external_attr >> 16) or 0o100644
                _write_entry(
                    output, records, member, data, modes[member], compression_level
                )
            for name, owner, member in licenses:
                _write_entry(
                    output,
                    records,
                    f"{dist_info}/licenses/{name}",
                    sources[owner.distribution].read(member),
                    compression_level=compression_level,
                )
            for name, owner, member in sboms:
                _write_entry(
                    output,
                    records,
                    f"{dist_info}/sboms/{name}",
                    sources[owner.distribution].read(member),
                    compression_level=compression_level,
                )
            dist_info_files = [
                (SDK_COMPONENTS_FILE, _components_record(version, tag, components)),
                ("METADATA", metadata),
                (
                    "WHEEL",
                    (
                        "Wheel-Version: 1.0\n"
                        "Generator: fastflow-sdk-pypi-wheel\n"
                        "Root-Is-Purelib: false\n"
                        f"Tag: {tag}\n"
                    ).encode(),
                ),
            ]
            if entry_points:
                dist_info_files.append(
                    ("entry_points.txt", _entry_points_text(entry_points))
                )
            for name, data in dist_info_files:
                _write_entry(
                    output,
                    records,
                    f"{dist_info}/{name}",
                    data,
                    compression_level=compression_level,
                )
            record_name = f"{dist_info}/RECORD"
            records.sort(key=lambda row: row[0])
            record = _record_csv([*records, (record_name, "", "")]).encode("utf-8")
            _write_entry(
                output, [], record_name, record, compression_level=compression_level
            )
    except BaseException:
        wheel.unlink(missing_ok=True)
        raise
    finally:
        for source in sources.values():
            source.close()
    _verify_output(wheel, dist_info, owners, modes)
    write_sha256_sidecar(wheel)
    return wheel


def bundled_shared_libraries(wheel: Path) -> frozenset[str]:
    """Returns the shared libraries the wheel ships itself.

    The component audits exclude libraries provided by a sibling component
    wheel, which the components load through relative RPATHs or a preload
    (``fastflow_sdk_compiler.activate_native_libraries``). In the merged wheel
    those siblings are part of the same wheel, so they are not host
    dependencies.
    """
    with zipfile.ZipFile(wheel) as archive:
        return frozenset(
            Path(name).name
            for name in archive.namelist()
            if re.search(r"\.so(\.\d+)*$", name)
        )


def audit_pypi_wheel(wheel: Path) -> str:
    """Requires auditwheel to confirm the Linux wheel's claimed manylinux tag."""
    match = _MANYLINUX.search(wheel.name)
    if not match:
        raise RuntimeError(f"not a manylinux wheel: {wheel.name}")
    claimed = (int(match.group(1)), int(match.group(2)))
    policy = auditwheel_policy(wheel, exclude=bundled_shared_libraries(wheel))
    try:
        return validate_manylinux_policy(policy, maximum_glibc=claimed)
    except RuntimeError as error:
        raise RuntimeError(f"{wheel.name} claims {match.group()}: {error}") from error


SMOKE_IMPORTS = (
    "fastflow_sdk_runtime",
    "fastflow_sdk_fastflowlm",
    "fastflow_sdk_qualification",
    "fastflow_sdk_compiler",
    "fastflow_sdk_compiler_bindings",
    "fastflow_sdk_ops",
    "iron",
    "aie.iron",
)
_SMOKE_PROBE = """
import importlib, json, sys, sysconfig
from importlib import metadata
for module in sys.argv[1:]:
    importlib.import_module(module)
from aie.ir import Context
Context()
import fastflow_sdk_ops
kernel = fastflow_sdk_ops.kernels_root() / "eltwise" / "add.cc"
if not kernel.is_file():
    sys.exit(f"missing IRON kernel source: {kernel}")
import fastflow_sdk_qualification
import fastflow_sdk_runtime
fastflow_sdk_runtime.self_test(load=True)
names = sorted(
    name
    for name in {d.metadata["Name"] for d in metadata.distributions()}
    if name.lower().startswith("fastflow")
)
print(json.dumps({
    "distributions": names,
    "platlib": sysconfig.get_path("platlib"),
    "qualification": fastflow_sdk_qualification.__version__,
}))
"""


def _venv_executable(venv: Path, name: str) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def _run(command: list) -> str:
    command = [str(part) for part in command]
    shown = " ".join("<script>" if "\n" in part else part for part in command)
    print(f"+ {shown}", flush=True)
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{Path(command[0]).name} exited {result.returncode}:\n"
            f"{result.stdout[-4000:]}{result.stderr[-4000:]}"
        )
    return result.stdout


def verify_pypi_wheel_install(
    wheel: Path,
    venv: Path,
    report_dir: Path,
    *,
    require_hardware: bool = False,
) -> None:
    """Installs the PyPI wheel into a clean venv and checks the installed SDK."""
    _, version, _ = _parse_wheel_name(wheel)
    dist_info = f"{SDK_DISTRIBUTION.replace('-', '_')}-{version}.dist-info"
    if venv.exists():
        shutil.rmtree(venv)
    _run([sys.executable, "-m", "venv", venv])
    python = _venv_executable(venv, "python")
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--only-binary=:all:",
            wheel,
        ]
    )
    _run([python, "-m", "pip", "check"])
    probe = json.loads(
        _run([python, "-c", _SMOKE_PROBE, *SMOKE_IMPORTS]).splitlines()[-1]
    )
    if probe["distributions"] != [SDK_DISTRIBUTION]:
        raise RuntimeError(
            f"expected only {SDK_DISTRIBUTION} installed: {probe['distributions']}"
        )
    if probe["qualification"] != version:
        raise RuntimeError(f"qualification reports version {probe['qualification']}")

    with zipfile.ZipFile(wheel) as archive:
        entry_points = _read_entry_points(
            archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        )
        executables = [
            info.filename
            for info in archive.infolist()
            if (info.external_attr >> 16) & 0o111
        ]
    for script in entry_points.get("console_scripts", {}):
        if not _venv_executable(venv, script).is_file():
            raise RuntimeError(f"console script was not installed: {script}")
    if os.name != "nt":
        for member in executables:
            if not os.access(Path(probe["platlib"]) / member, os.X_OK):
                raise RuntimeError(f"installed file lost its executable bit: {member}")
    _run([_venv_executable(venv, "aie-opt"), "--version"])
    _run([_venv_executable(venv, "xclbinutil"), "--version"])

    fastflow = _venv_executable(venv, "fastflow")
    report = json.loads(_run([fastflow, "qualify", "--list-profiles", "--json"]))
    expected = {
        name: version
        for name in (
            RUNTIME_DISTRIBUTION,
            FASTFLOWLM_DISTRIBUTION,
            QUALIFICATION_DISTRIBUTION,
        )
    }
    if report.get("status") != "pass" or report.get("packages") != expected:
        raise RuntimeError(f"fastflow qualify --list-profiles reported {report}")
    run_reports(fastflow, report_dir, require_hardware=require_hardware, prefix="pypi-")
    if require_hardware:
        _run([_venv_executable(venv, "flm"), "version"])
        _run([python, paths.ROOT / "examples" / "hello_npu.py"])


def cmd_package_pypi_wheel(args):
    try:
        wheel = build_pypi_wheel(
            Path(args.release_dir),
            args.platform,
            Path(args.dir),
            sha256sums=Path(args.sha256sums) if args.sha256sums else None,
        )
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"package pypi wheel: ok -> {wheel}")


def cmd_audit_pypi_wheel(args):
    try:
        policy = audit_pypi_wheel(Path(args.wheel))
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"audit pypi wheel: ok ({policy})")


def host_pypi_wheel(path: Path) -> Path:
    """Returns ``path``, or the one wheel in directory ``path`` for this host."""
    if not path.is_dir():
        return path
    pattern = "*-win_amd64.whl" if os.name == "nt" else "*-manylinux_*_x86_64.whl"
    wheels = sorted(path.glob(f"{SDK_DISTRIBUTION.replace('-', '_')}-{pattern}"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"expected one {pattern} wheel in {path}, found {len(wheels)}"
        )
    return wheels[0]


def cmd_verify_pypi_wheel(args):
    try:
        verify_pypi_wheel_install(
            host_pypi_wheel(Path(args.wheel)),
            Path(args.venv),
            Path(args.report_dir),
            require_hardware=args.require_hardware,
        )
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"verify pypi wheel: ok -> {args.venv}")
