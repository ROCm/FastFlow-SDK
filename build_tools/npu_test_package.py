# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Artifact helpers for compiler-free NPU XADX test packages."""

import contextlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from build_tools.fastflowlm_artifact import sha256_file


ARCHIVE_STEM = "npu-test-package.tar"


def find_archive(directory):
    directory = Path(directory)
    for extension in ("zst", "gz"):
        candidate = directory / f"{ARCHIVE_STEM}.{extension}"
        if candidate.is_file():
            return candidate
    return None


def write_case(stage, generated, xadx, source):
    """Writes one generated numeric case and returns its manifest record."""
    import numpy as np

    stage = Path(stage)
    case_dir = stage / "cases" / generated["name"]
    case_dir.mkdir(parents=True)
    xadx_path = case_dir / "kernel.xadx"
    xadx_path.write_bytes(xadx)
    buffers = []
    for descriptor in generated["buffers"]:
        record = {
            key: (list(value) if key == "shape" else value)
            for key, value in descriptor.items()
            if key not in ("data", "expected")
        }
        storage_dtype = np.dtype(record["dtype"])
        record["dtype"] = storage_dtype.str
        if descriptor["direction"] == "input":
            field, value = "input", descriptor["data"]
        else:
            field, value = "expected", descriptor["expected"]
        data_path = case_dir / f"{descriptor['name']}.bin"
        np.asarray(value, dtype=storage_dtype).tofile(data_path)
        record[field] = data_path.relative_to(stage).as_posix()
        buffers.append(record)
    return {
        "name": generated["name"],
        "xadx": xadx_path.relative_to(stage).as_posix(),
        "export": "MLIR_AIE",
        "source": source,
        "buffers": buffers,
    }


def write_manifest(stage, cases, source_revision, system, machine):
    """Inventories a staged package and writes its identity manifest."""
    stage = Path(stage)
    files = []
    for path in sorted(candidate for candidate in stage.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": path.relative_to(stage).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "product": "rocm-npu-test-package",
        "source_revision": source_revision,
        "platform": {"system": system, "machine": machine},
        "cases": cases,
        "files": files,
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def create_archive(stage, output_directory, codec):
    """Creates the platform-selected compressed tar archive."""
    stage = Path(stage)
    output_directory = Path(output_directory)
    archive = output_directory / f"{ARCHIVE_STEM}.{codec}"
    output_directory.mkdir(parents=True, exist_ok=True)
    for other in output_directory.glob(f"{ARCHIVE_STEM}.*"):
        if other != archive:
            other.unlink()
    if codec == "zst":
        process = subprocess.Popen(
            ["zstd", "-6", "-T0", "-q", "-f", "-o", str(archive)],
            stdin=subprocess.PIPE,
        )
        with tarfile.open(fileobj=process.stdin, mode="w|") as output:
            output.add(stage, arcname=".")
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"zstd failed (exit {process.returncode})")
    elif codec == "gz":
        with tarfile.open(archive, "w:gz", compresslevel=6) as output:
            output.add(stage, arcname=".")
    else:
        raise ValueError(f"unsupported NPU test package codec: {codec}")
    return archive


@contextlib.contextmanager
def _open_archive(archive):
    archive = Path(archive)
    if archive.suffix != ".zst":
        with tarfile.open(archive, "r:gz") as source:
            yield source
        return
    process = subprocess.Popen(
        ["zstd", "-q", "-d", "-c", str(archive)], stdout=subprocess.PIPE
    )
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as source:
            yield source
    finally:
        process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError(f"zstd failed while reading {archive}")


def extract_archive(archive, destination):
    """Safely extracts a link-free test package into an empty destination."""
    archive = Path(archive)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with _open_archive(archive) as source:
        links = [
            member.name
            for member in source.getmembers()
            if member.issym() or member.islnk()
        ]
        if links:
            raise RuntimeError("NPU test package contains links")
    with _open_archive(archive) as source:
        source.extractall(destination, filter="data")


def verify_tree(package, expected_cases, source_revision, system):
    """Verifies identity, complete inventory, case data, and XADX containers."""
    package = Path(package)
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing NPU test package manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported NPU test package schema")
    if manifest.get("product") != "rocm-npu-test-package":
        raise RuntimeError("not a rocm-npu-test-package manifest")
    if manifest.get("source_revision") != source_revision:
        raise RuntimeError("NPU test package source revision does not match this checkout")
    if manifest.get("platform", {}).get("system") != system:
        raise RuntimeError("NPU test package platform does not match this runner")

    expected_cases = set(expected_cases)
    cases = manifest.get("cases", [])
    case_names = {case.get("name") for case in cases}
    if case_names != expected_cases or len(cases) != len(expected_cases):
        raise RuntimeError(
            "NPU test package cases do not match the declared suite: "
            f"expected {sorted(expected_cases)}, got {sorted(case_names)}"
        )

    expected_files = {record["path"]: record for record in manifest.get("files", [])}
    actual_files = {
        path.relative_to(package).as_posix(): path
        for path in package.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(actual_files) != set(expected_files):
        raise RuntimeError("NPU test package inventory does not match files on disk")
    for relative, path in actual_files.items():
        record = expected_files[relative]
        if path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"NPU test package file identity mismatch: {relative}")

    for case in cases:
        data = (package / case["xadx"]).read_bytes()
        if len(data) < 8 or data[4:8] != b"XADX":
            raise RuntimeError(f"{case['name']}: malformed XADX container")
        for buffer in case["buffers"]:
            field = "input" if buffer["direction"] == "input" else "expected"
            if not (package / buffer[field]).is_file():
                raise RuntimeError(
                    f"{case['name']}.{buffer['name']}: missing {field} data"
                )
    return manifest
