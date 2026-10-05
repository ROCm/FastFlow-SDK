# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib

import pytest

from build_tools.artifact import (
    directory_state,
    file_inventory,
    reset_directory,
    sha256_file,
    verify_file_inventory,
    verify_sha256_sidecar,
    write_sha256_sidecar,
)

pytestmark = pytest.mark.presubmit


def test_file_inventory_is_portable_deterministic_and_verifiable(tmp_path):
    root = tmp_path / "tree"
    (root / "nested").mkdir(parents=True)
    (root / "z.bin").write_bytes(b"z")
    payload = root / "nested" / "a.bin"
    payload.write_bytes(b"artifact")

    records = file_inventory(root)

    assert [record["path"] for record in records] == ["nested/a.bin", "z.bin"]
    assert records[0] == {
        "path": "nested/a.bin",
        "size": 8,
        "sha256": hashlib.sha256(b"artifact").hexdigest(),
    }
    assert sha256_file(payload) == records[0]["sha256"]
    verify_file_inventory(root, records)

    payload.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="verification failed"):
        verify_file_inventory(root, records)


def test_file_inventory_orders_mixed_case_paths_like_linux(tmp_path):
    licenses = tmp_path / "licenses"
    licenses.mkdir()
    (licenses / "fastflow-sdk-LICENSE").write_bytes(b"sdk")
    (licenses / "FastFlowLM-LICENSE_RUNTIME.txt").write_bytes(b"flm")

    assert [record["path"] for record in file_inventory(tmp_path)] == [
        "licenses/FastFlowLM-LICENSE_RUNTIME.txt",
        "licenses/fastflow-sdk-LICENSE",
    ]


def test_directory_state_honors_excluded_names(tmp_path):
    (tmp_path / "payload").write_text("content", encoding="utf-8")
    marker = tmp_path / ".marker"
    marker.write_text("first", encoding="utf-8")
    before = directory_state(tmp_path, exclude_names=(".marker",))
    marker.write_text("second", encoding="utf-8")
    assert directory_state(tmp_path, exclude_names=(".marker",)) == before


def test_sha256_sidecar_round_trip_and_validation(tmp_path):
    artifact = tmp_path / "payload.bin"
    artifact.write_bytes(b"payload")
    sidecar = write_sha256_sidecar(artifact)
    assert verify_sha256_sidecar(artifact) == sha256_file(artifact)

    sidecar.write_text(f"{'0' * 64}  {artifact.name}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_sha256_sidecar(artifact)
    sidecar.write_text(f"{'0' * 64}  other.bin\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="malformed artifact checksum"):
        verify_sha256_sidecar(artifact)


def test_reset_directory_rejects_root_and_outside_targets(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "target"
    target.mkdir()
    (target / "stale").write_text("stale", encoding="utf-8")

    reset_directory(target, allowed)
    assert list(target.iterdir()) == []
    with pytest.raises(RuntimeError, match="refusing to remove"):
        reset_directory(allowed, allowed)
    with pytest.raises(RuntimeError, match="refusing to remove"):
        reset_directory(tmp_path / "outside", allowed)
