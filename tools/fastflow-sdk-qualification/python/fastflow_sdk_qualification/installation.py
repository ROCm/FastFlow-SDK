# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate plugin context and bind installed packages to release candidates."""

from __future__ import annotations

import base64
import configparser
import csv
import email.parser
import hashlib
import importlib
import io
import json
import os
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Any

from .candidate import CandidateError, normalize_candidate_manifest
from .errors import QualificationError
from .hashing import sha256_file
from .numeric_runtime import NUMERIC_DISTRIBUTIONS, NUMERIC_RUNTIME

PLUGIN_API = "1"
PACKAGE_DISTRIBUTIONS = (
    "fastflow-sdk-runtime",
    "fastflow-sdk-fastflowlm",
    "fastflow-sdk-qualification",
)
PACKAGE_MODULES = {
    "fastflow-sdk-runtime": "fastflow_sdk_runtime",
    "fastflow-sdk-fastflowlm": "fastflow_sdk_fastflowlm",
    "fastflow-sdk-qualification": "fastflow_sdk_qualification",
    "numpy": "numpy",
    "ml-dtypes": "ml_dtypes",
}


def package_versions() -> dict[str, str]:
    versions = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError as error:
            raise QualificationError(
                f"required package is not installed: {distribution}"
            ) from error
    return versions


def validate_protocol() -> tuple[Path, str, bool]:
    api = os.environ.get("FASTFLOW_SDK_PLUGIN_API")
    if api != PLUGIN_API:
        raise QualificationError(
            f"unsupported FASTFLOW_SDK_PLUGIN_API {api!r}; expected {PLUGIN_API}"
        )
    executable = Path(os.environ.get("FASTFLOW_SDK_CORE_EXECUTABLE", ""))
    if not executable.is_absolute() or not executable.is_file():
        raise QualificationError(
            "FASTFLOW_SDK_CORE_EXECUTABLE is not an executable file"
        )
    core_version = os.environ.get("FASTFLOW_SDK_CORE_VERSION", "")
    if not core_version:
        raise QualificationError("FASTFLOW_SDK_CORE_VERSION is required")
    json_output = os.environ.get("FASTFLOW_SDK_OUTPUT_FORMAT") == "json"
    return executable, core_version, json_output


def normalize_candidate(manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        return normalize_candidate_manifest(manifest)
    except CandidateError as error:
        raise QualificationError(str(error)) from error


def verify_installed_packages(
    candidate: dict[str, Any], installed: dict[str, str]
) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        module_name = PACKAGE_MODULES[distribution]
        package_root = Path(importlib.import_module(module_name).__file__).parent
        manifest_path = package_root / "manifest.json"
        if not manifest_path.is_file():
            raise QualificationError(
                f"installed package manifest is missing: {distribution}"
            )
        manifest_bytes = manifest_path.read_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        try:
            package_manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as error:
            raise QualificationError(
                f"installed package manifest is invalid: {distribution}"
            ) from error
        files = package_manifest.get("files")
        if not isinstance(files, list):
            raise QualificationError(
                f"installed package file inventory is missing: {distribution}"
            )
        for record in files:
            if not isinstance(record, dict):
                raise QualificationError(
                    f"installed package file record is invalid: {distribution}"
                )
            relative = record.get("path")
            digest = record.get("sha256")
            if (
                not isinstance(relative, str)
                or "\\" in relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(digest, str)
            ):
                raise QualificationError(
                    f"installed package file record is unsafe: {distribution}"
                )
            path = package_root / relative
            if not path.is_file() or sha256_file(path) != digest:
                raise QualificationError(
                    "installed package file verification failed: "
                    f"{distribution}/{relative}"
                )
        candidates = candidate["packages"][distribution]
        matches = [
            identity
            for identity in candidates
            if identity["version"] == installed[distribution]
            and identity["package_manifest_sha256"] == manifest_digest
        ]
        if len(matches) != 1:
            raise QualificationError(
                f"installed package identity does not match candidate: {distribution}"
            )
        selected[distribution] = matches[0]
    return selected


def verify_candidate_wheels(
    selected: dict[str, dict[str, str]], manifest_path: Path
) -> dict[str, Path]:
    root = manifest_path.resolve().parent
    wheels: dict[str, Path] = {}
    for distribution, identity in selected.items():
        matches = [
            path
            for path in root.rglob(identity["filename"])
            if path.is_file() and not path.is_symlink()
        ]
        if len(matches) != 1:
            raise QualificationError(
                "candidate wheel must exist exactly once beside the manifest: "
                f"{identity['filename']}"
            )
        if sha256_file(matches[0]) != identity["sha256"]:
            raise QualificationError(
                f"candidate wheel SHA256 does not match: {distribution}"
            )
        wheels[distribution] = matches[0]
    return wheels


def verify_recorded_install_tree(
    distribution: str,
    package_roots: set[Path] | Path,
    metadata_root: Path,
    recorded_paths: set[Path],
) -> None:
    recorded = {path.resolve() for path in recorded_paths}
    if isinstance(package_roots, Path):
        package_roots = {package_roots}
    installer_files = {"INSTALLER", "REQUESTED", "direct_url.json"}
    for root in (*sorted(package_roots), metadata_root):
        if not root.is_dir():
            raise QualificationError(
                f"installed package directory is missing: {distribution}/{root.name}"
            )
        for path in root.rglob("*"):
            if path.is_symlink():
                raise QualificationError(
                    "installed package has an unsafe symlink: "
                    f"{distribution}/{path.relative_to(root).as_posix()}"
                )
            if not path.is_file() or path.resolve() in recorded:
                continue
            generated_pyc = (
                path.parent.name == "__pycache__"
                and path.suffix == ".pyc"
                and (path.parent.parent / f"{path.name.split('.', 1)[0]}.py").resolve()
                in recorded
            )
            installer_metadata = (
                root == metadata_root
                and path.parent == metadata_root
                and path.name in installer_files
            )
            if not generated_pyc and not installer_metadata:
                raise QualificationError(
                    "installed package has an unrecorded file: "
                    f"{distribution}/{path.relative_to(root).as_posix()}"
                )


def _installed_record_name(member_name: str) -> str:
    """Map wheel .data payloads to their installed site-packages RECORD path."""
    parts = member_name.replace("\\", "/").split("/")
    if (
        len(parts) >= 3
        and parts[0].endswith(".data")
        and parts[1]
        in {
            "purelib",
            "platlib",
        }
    ):
        return "/".join(parts[2:])
    return member_name


def _verify_recorded_payload(
    distribution: str, member_name: str, values: list[str], path: Path
) -> None:
    if not path.is_file() or path.is_symlink():
        raise QualificationError(
            f"installed wheel payload is missing or unsafe: "
            f"{distribution}/{member_name}"
        )
    if len(values) != 2 or not values[0] or not values[1]:
        raise QualificationError(
            f"candidate wheel payload is not hash-bound: {distribution}/{member_name}"
        )
    try:
        algorithm, encoded = values[0].split("=", 1)
        expected_size = int(values[1])
        digest = hashlib.new(algorithm, path.read_bytes()).digest()
    except (ValueError, OSError) as error:
        raise QualificationError(
            f"candidate wheel payload record is invalid: {distribution}/{member_name}"
        ) from error
    actual = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    if path.stat().st_size != expected_size or actual != encoded:
        raise QualificationError(
            f"installed wheel payload hash does not match candidate: "
            f"{distribution}/{member_name}"
        )


def verify_installed_wheel_metadata(wheels: dict[str, Path | tuple[Path, str]]) -> None:
    for distribution, wheel_value in wheels.items():
        wheel, expected_record_sha256 = (
            wheel_value if isinstance(wheel_value, tuple) else (wheel_value, None)
        )
        installed_distribution = metadata.distribution(distribution)
        with zipfile.ZipFile(wheel) as archive:
            record_members = [
                member
                for member in archive.infolist()
                if member.filename.endswith(".dist-info/RECORD")
            ]
            if len(record_members) != 1:
                raise QualificationError(
                    f"candidate wheel RECORD is missing: {distribution}"
                )
            if (
                expected_record_sha256 is not None
                and hashlib.sha256(archive.read(record_members[0])).hexdigest()
                != expected_record_sha256
            ):
                raise QualificationError(
                    f"candidate wheel RECORD digest does not match: {distribution}"
                )
            for member in archive.infolist():
                if (
                    member.is_dir()
                    or member.filename.endswith(".dist-info/RECORD")
                    or ".dist-info/" not in member.filename
                ):
                    continue
                installed_path = Path(
                    installed_distribution.locate_file(member.filename)
                )
                if (
                    not installed_path.is_file()
                    or hashlib.sha256(installed_path.read_bytes()).digest()
                    != hashlib.sha256(archive.read(member)).digest()
                ):
                    raise QualificationError(
                        "installed wheel metadata does not match candidate: "
                        f"{distribution}/{Path(member.filename).name}"
                    )
            record_member = record_members[0]
            candidate_rows = {
                _installed_record_name(row[0]): row[1:]
                for row in csv.reader(
                    io.StringIO(archive.read(record_member).decode("utf-8"))
                )
                if row
            }
            entry_points = configparser.ConfigParser()
            entry_point_members = [
                member
                for member in archive.infolist()
                if member.filename.endswith(".dist-info/entry_points.txt")
            ]
            if entry_point_members:
                entry_points.read_string(
                    archive.read(entry_point_members[0]).decode("utf-8")
                )
            script_names = (
                set(entry_points["console_scripts"])
                if entry_points.has_section("console_scripts")
                else set()
            )
        installed_record = Path(
            installed_distribution.locate_file(record_member.filename)
        )
        if not installed_record.is_file():
            raise QualificationError(
                f"installed wheel RECORD is missing: {distribution}"
            )
        installed_rows = {
            row[0]: row[1:]
            for row in csv.reader(
                io.StringIO(installed_record.read_text(encoding="utf-8"))
            )
            if row
        }
        installed_record_name = _installed_record_name(record_member.filename)
        for member_name, values in candidate_rows.items():
            if member_name == installed_record_name:
                continue
            if installed_rows.get(member_name) != values:
                raise QualificationError(
                    "installed wheel RECORD does not match candidate: "
                    f"{distribution}/{member_name}"
                )
            _verify_recorded_payload(
                distribution,
                member_name,
                values,
                Path(installed_distribution.locate_file(member_name)),
            )
        installer_files = {"INSTALLER", "REQUESTED", "direct_url.json"}
        for member_name in installed_rows.keys() - candidate_rows.keys():
            values = installed_rows[member_name]
            normalized = member_name.replace("\\", "/")
            basename = normalized.rsplit("/", 1)[-1]
            allowed = (
                (".dist-info/" in normalized and basename in installer_files)
                or (
                    "/__pycache__/" in normalized
                    and basename.endswith(".pyc")
                    and (
                        normalized.split("/__pycache__/", 1)[0]
                        + "/"
                        + basename.split(".", 1)[0]
                        + ".py"
                    )
                    in candidate_rows
                )
                or (
                    ("/bin/" in normalized or "/Scripts/" in normalized)
                    and any(
                        basename in (name, f"{name}.exe", f"{name}-script.py")
                        for name in script_names
                    )
                )
            )
            if not allowed:
                raise QualificationError(
                    "installed wheel RECORD has an unexpected file: "
                    f"{distribution}/{member_name}"
                )
            if values and values[0]:
                algorithm, encoded = values[0].split("=", 1)
                installed_path = Path(installed_distribution.locate_file(member_name))
                actual = (
                    base64.urlsafe_b64encode(
                        hashlib.new(algorithm, installed_path.read_bytes()).digest()
                    )
                    .rstrip(b"=")
                    .decode()
                )
                if actual != encoded:
                    raise QualificationError(
                        "installed wheel RECORD hash is invalid: "
                        f"{distribution}/{member_name}"
                    )
        recorded_paths = {
            Path(installed_distribution.locate_file(member_name))
            for member_name in installed_rows
        }
        package_roots = {
            Path(installed_distribution.locate_file(normalized.split("/", 1)[0]))
            for member_name in candidate_rows
            for normalized in [member_name.replace("\\", "/")]
            if "/" in normalized
            and not normalized.startswith("../")
            and ".dist-info/" not in normalized
        }
        package_roots.add(
            Path(installed_distribution.locate_file(PACKAGE_MODULES[distribution]))
        )
        verify_recorded_install_tree(
            distribution,
            package_roots,
            installed_record.parent,
            recorded_paths,
        )


def numeric_runtime_identities() -> dict[str, dict[str, Any]]:
    """Validate and describe the installed numeric wheel payloads."""
    identities = {}
    for distribution in NUMERIC_DISTRIBUTIONS:
        try:
            installed_distribution = metadata.distribution(distribution)
        except metadata.PackageNotFoundError as error:
            raise QualificationError(
                f"required package is not installed: {distribution}"
            ) from error
        installed_version = installed_distribution.version
        if installed_version != NUMERIC_RUNTIME[distribution]:
            raise QualificationError(
                f"installed numeric runtime version is invalid: {distribution}"
            )
        wheel_text = installed_distribution.read_text("WHEEL")
        if wheel_text is None:
            raise QualificationError(
                f"installed wheel metadata is missing: {distribution}"
            )
        wheel_metadata = email.parser.Parser().parsestr(wheel_text)
        installed_tags = sorted(set(wheel_metadata.get_all("Tag") or ()))
        if not installed_tags:
            raise QualificationError(
                f"installed numeric runtime has no compatibility tags: {distribution}"
            )
        files = list(installed_distribution.files or ())
        records = [
            item for item in files if item.as_posix().endswith(".dist-info/RECORD")
        ]
        if len(records) != 1:
            raise QualificationError(
                f"installed numeric runtime RECORD is missing: {distribution}"
            )
        record_path = Path(installed_distribution.locate_file(records[0]))
        try:
            record_bytes = record_path.read_bytes()
        except OSError as error:
            raise QualificationError(
                f"installed numeric runtime RECORD cannot be read: {distribution}"
            ) from error
        payload = []
        recorded_paths = {
            Path(installed_distribution.locate_file(item)) for item in files
        }
        for item in sorted(files, key=lambda value: value.as_posix()):
            name = item.as_posix()
            if item == records[0]:
                continue
            if item.hash is None or item.size is None:
                if name.endswith(".pyc"):
                    continue
                raise QualificationError(
                    f"installed numeric runtime has an unbound file: "
                    f"{distribution}/{name}"
                )
            path = Path(installed_distribution.locate_file(item))
            if not path.is_file() or path.is_symlink():
                raise QualificationError(
                    f"installed numeric runtime payload is missing or unsafe: "
                    f"{distribution}/{name}"
                )
            digest = hashlib.new(item.hash.mode, path.read_bytes()).digest()
            encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            if encoded != item.hash.value or path.stat().st_size != item.size:
                raise QualificationError(
                    f"installed numeric runtime payload hash is invalid: "
                    f"{distribution}/{name}"
                )
            payload.append((name, f"{item.hash.mode}={item.hash.value}", item.size))
        package_roots = {
            Path(installed_distribution.locate_file(name.split("/", 1)[0]))
            for item in files
            for name in [item.as_posix()]
            if "/" in name and not name.startswith("../") and ".dist-info/" not in name
        }
        verify_recorded_install_tree(
            distribution,
            package_roots,
            record_path.parent,
            recorded_paths,
        )
        payload_bytes = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        identities[distribution] = {
            "version": installed_version,
            "compatibility_tags": installed_tags,
            "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        }
    return identities


def verify_numeric_runtime(candidate_manifest: dict[str, Any]) -> None:
    if candidate_manifest["numeric_runtime"] != NUMERIC_RUNTIME:
        raise QualificationError(
            "candidate numeric runtime versions do not match the qualification contract"
        )
    numeric_runtime_identities()


def validate_versions(
    installed: dict[str, str],
    core_version: str,
    candidate_manifest: dict[str, Any] | None,
    candidate_path: Path | None = None,
) -> None:
    mismatched = {
        name: version for name, version in installed.items() if version != core_version
    }
    if mismatched:
        raise QualificationError(
            f"installed package versions do not match core {core_version}: {mismatched}"
        )
    if candidate_manifest is None:
        return
    if candidate_path is None:
        raise QualificationError("candidate manifest path is required")
    for name, version in installed.items():
        expected = {
            identity["version"] for identity in candidate_manifest["packages"][name]
        }
        if version not in expected:
            raise QualificationError(
                f"candidate manifest versions for {name} are {sorted(expected)!r}, "
                f"installed version is {version!r}"
            )
    selected = verify_installed_packages(candidate_manifest, installed)
    wheels = verify_candidate_wheels(selected, candidate_path)
    verify_installed_wheel_metadata(wheels)
    verify_numeric_runtime(candidate_manifest)
