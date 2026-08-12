# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Cross-product artifact identity and filesystem primitives."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    """Returns the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_sidecar_path(path: Path) -> Path:
    """Returns the conventional checksum sidecar path for an artifact."""
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def write_sha256_sidecar(path: Path, *, digest: str | None = None) -> Path:
    """Writes a deterministic GNU-style SHA-256 sidecar."""
    path = Path(path)
    sidecar = sha256_sidecar_path(path)
    digest = digest or sha256_file(path)
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return sidecar


def verify_sha256_sidecar(path: Path, *, kind: str = "artifact") -> str:
    """Verifies a required SHA-256 sidecar and returns the artifact digest."""
    path = Path(path)
    sidecar = sha256_sidecar_path(path)
    if not sidecar.is_file():
        raise RuntimeError(f"{kind} checksum missing: {sidecar}")
    try:
        fields = sidecar.read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"malformed {kind} checksum: {sidecar}") from error
    if (
        len(fields) != 2
        or fields[1] != path.name
        or len(fields[0]) != 64
        or any(character not in "0123456789abcdef" for character in fields[0])
    ):
        raise RuntimeError(f"malformed {kind} checksum: {sidecar}")
    digest = sha256_file(path)
    if digest != fields[0]:
        raise RuntimeError(f"{kind} checksum mismatch: {path}")
    return digest


def git_revision(path: Path) -> str:
    """Returns the checked-out Git revision for a repository."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(path),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_source_state(path: Path) -> dict[str, object]:
    """Describes the exact tracked and non-ignored source state used to build."""
    path = Path(path)
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout
    tracked_output = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACDMRTUXB", "-z", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout
    untracked_output = subprocess.run(
        ["git", "ls-files", "-o", "--exclude-standard", "-z"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout
    changed_paths = sorted(
        {
            item.decode("utf-8", errors="surrogateescape")
            for output in (tracked_output, untracked_output)
            for item in output.split(b"\0")
            if item
        }
    )
    changed_files = []
    for relative in changed_paths:
        source = path / relative
        record: dict[str, object] = {"path": relative.replace("\\", "/")}
        if source.is_file():
            record.update({"size": source.stat().st_size, "sha256": sha256_file(source)})
        else:
            record["deleted"] = True
        changed_files.append(record)
    return {
        "revision": git_revision(path),
        "dirty": bool(changed_files),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "changed_files": changed_files,
    }


def file_record(root: Path, path: Path) -> dict[str, object]:
    """Constructs a portable identity record for one file under ``root``."""
    path = Path(path)
    return {
        "path": path.relative_to(Path(root)).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def file_inventory(
    root: Path,
    *,
    exclude_paths: Iterable[Path] = (),
    exclude_names: Iterable[str] = (),
) -> list[dict[str, object]]:
    """Constructs a deterministic identity inventory for a directory tree."""
    root = Path(root)
    excluded = {Path(path) for path in exclude_paths}
    excluded_names = set(exclude_names)
    return [
        file_record(root, path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path not in excluded and path.name not in excluded_names
    ]


def directory_state(
    path: Path, *, exclude_names: tuple[str, ...] = ()
) -> dict[str, object]:
    """Returns a digest over the deterministic file inventory of a directory."""
    files = file_inventory(path, exclude_names=exclude_names)
    digest = hashlib.sha256()
    for record in files:
        digest.update(json.dumps(record, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "file_count": len(files)}


def verify_file_inventory(
    root: Path,
    records: object,
    *,
    exclude_paths: Iterable[Path] = (),
) -> None:
    """Validates and verifies a complete portable file inventory."""
    root = Path(root)
    if not isinstance(records, list):
        raise RuntimeError("file inventory must be a list")  # noqa: TRY004
    expected: dict[str, Mapping[str, object]] = {}
    for entry in records:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError(  # noqa: TRY004
                "file inventory contains an invalid file record"
            )
        relative = PurePosixPath(entry["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or str(relative) != entry["path"]
        ):
            raise RuntimeError(f"file inventory contains an unsafe path: {entry['path']}")
        if entry["path"] in expected:
            raise RuntimeError(f"file inventory contains a duplicate path: {entry['path']}")
        if (
            not isinstance(entry.get("size"), int)
            or entry["size"] < 0
            or not isinstance(entry.get("sha256"), str)
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
        ):
            raise RuntimeError(f"file inventory contains invalid metadata: {entry['path']}")
        expected[entry["path"]] = entry

    excluded = {Path(path) for path in exclude_paths}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path not in excluded
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"file inventory mismatch; missing={missing}, unexpected={unexpected}"
        )
    for relative, entry in expected.items():
        path = actual[relative]
        if (
            path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise RuntimeError(f"file verification failed: {relative}")


def validate_directory_target(path: Path, allowed_root: Path) -> None:
    """Rejects a destructive target outside or equal to its allowed root."""
    resolved = Path(path).resolve()
    root = Path(allowed_root).resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"refusing to remove directory outside {root}: {resolved}")


def reset_directory(path: Path, allowed_root: Path) -> None:
    """Safely replaces a directory below an allowed root with an empty one."""
    path = Path(path)
    validate_directory_target(path, allowed_root)
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
