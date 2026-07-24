# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Platform-neutral mechanics for building and verifying ROCm NPU wheels."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import re
import stat
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


FIXED_ZIP_DATE = (2000, 1, 1, 0, 0, 0)
PACKAGE_VERSION_ENV = "ROCM_NPU_PACKAGE_VERSION"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> Path:
    """Builds a deterministic wheel from an already staged package tree."""
    version = validate_version(version)
    output_directory.mkdir(parents=True, exist_ok=True)
    dist_info = (
        f"{distribution.replace('-', '_')}-{version.replace('-', '_')}.dist-info"
    )
    metadata_lines = [
        "Metadata-Version: 2.1",
        f"Name: {distribution}",
        f"Version: {version}",
        f"Summary: {summary}",
        "Requires-Python: >=3.11",
        "License: See included license files",
    ]
    metadata_lines.extend(f"Requires-Dist: {requirement}" for requirement in requires)
    metadata = "\n".join(metadata_lines) + "\n"
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
        f"{_sha256_file(wheel)}  {wheel.name}\n", encoding="ascii"
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
        if _sha256_file(wheel) != fields[0]:
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
