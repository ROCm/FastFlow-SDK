# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage a flat GitHub-upload directory from payload + intake outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fastflow_sdk_qualification import cli
from fastflow_sdk_qualification.candidate import canonical_candidate_bytes
from fastflow_sdk_qualification.hashing import sha256_file
from fastflow_sdk_qualification.offline import (
    EvidenceError,
    intake_evidence,
    pack_evidence,
    stage_github_release,
)
from tests.support.qualification import (
    make_candidate as _candidate,
)
from tests.support.qualification import (
    make_evidence as _evidence,
)
from tests.support.qualification import (
    make_policy as _policy,
)

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def _write_payload(root: Path, files: dict[str, bytes]) -> Path:
    """Write a release-like payload.

    Product files go into ``SHA256SUMS``. ``candidate-manifest.json`` and
    ``candidate-SHA256SUMS`` are required siblings outside that file, matching
    the collected CI release layout.
    """
    root.mkdir(parents=True, exist_ok=True)
    if "candidate-manifest.json" not in files:
        raise AssertionError("test payload requires candidate-manifest.json")
    for name, data in files.items():
        (root / name).write_bytes(data)
    # Nested dirs such as simple/ must be ignored by stage-release.
    simple = root / "simple" / "fastflow-sdk-ops"
    simple.mkdir(parents=True)
    (simple / "index.html").write_text("<html></html>\n", encoding="utf-8")

    product_files = {
        name: data
        for name, data in files.items()
        if name not in {"candidate-manifest.json", "candidate-SHA256SUMS"}
    }
    product_lines = [
        f"{hashlib.sha256(data).hexdigest()}  {name}"
        for name, data in sorted(product_files.items())
    ]
    (root / "SHA256SUMS").write_text(
        "\n".join(product_lines) + ("\n" if product_lines else ""),
        encoding="ascii",
    )
    # Prefer an explicit candidate-SHA256SUMS when the caller supplies one.
    if "candidate-SHA256SUMS" not in files:
        candidate_targets = {
            "candidate-manifest.json": files["candidate-manifest.json"]
        }
        for name, data in files.items():
            if name.startswith("fastflow_sdk_") and name.endswith(".whl"):
                candidate_targets[name] = data
        cand_lines = [
            f"{hashlib.sha256(data).hexdigest()}  {name}"
            for name, data in sorted(candidate_targets.items())
        ]
        (root / "candidate-SHA256SUMS").write_text(
            "\n".join(cand_lines) + "\n", encoding="ascii"
        )
    return root


def _make_intake(tmp_path: Path) -> tuple[Path, bytes, Path, Path]:
    candidate = _candidate()
    for distribution in ("fastflow-sdk-runtime", "fastflow-sdk-fastflowlm"):
        identity = dict(candidate["packages"][distribution][0])
        identity["filename"] = identity["filename"].replace(".whl", "-manylinux.whl")
        identity["sha256"] = "1" * 64
        candidate["packages"][distribution].append(identity)
    candidate_bytes = canonical_candidate_bytes(candidate)
    candidate_path = tmp_path / "candidate-manifest.json"
    candidate_path.write_bytes(candidate_bytes)

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    for name, platform_os in (
        ("windows", "Windows 11"),
        ("linux", "Linux 6.8"),
    ):
        source = _evidence(
            tmp_path / f"evidence-{name}",
            candidate=candidate,
            marker=name,
            platform_os=platform_os,
        )
        pack_evidence(source, incoming / f"{name}.evidence.zip")

    assembled = tmp_path / "assembled"
    aggregate = tmp_path / f"qualification-evidence-{candidate['version']}.zip"
    summary = intake_evidence(
        incoming,
        candidate_path,
        assembled,
        aggregate,
        require_status="pass",
        policy=_policy("windows", "linux"),
    )
    assert summary["status"] == "pass"
    index = assembled / "qualification-index.json"
    assert index.is_file()
    assert aggregate.with_name(aggregate.name + ".sha256").is_file()
    return candidate_path, candidate_bytes, aggregate, index


def test_stage_github_release_builds_exact_upload_set(tmp_path):
    candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {
            "candidate-manifest.json": candidate_bytes,
            "index.html": b"<html>release</html>\n",
            "fastflow-sdk-release-manifest.json": b'{"product":"fastflow-sdk"}\n',
            "fastflow_sdk_ops-0.1.0.devtest-py3-none-any.whl": b"ops-wheel",
            "fastflow_sdk_ops-0.1.0.devtest-py3-none-any.whl.sha256": (
                hashlib.sha256(b"ops-wheel").hexdigest()
                + "  fastflow_sdk_ops-0.1.0.devtest-py3-none-any.whl\n"
            ).encode("ascii"),
            "fastflow-sdk-linux-x86_64.tar.zst": b"linux-archive",
            "fastflow-sdk-windows-x86_64.zip": b"windows-archive",
        },
    )
    upload = tmp_path / "github-upload"
    summary = stage_github_release(
        payload,
        aggregate,
        upload,
        index=index,
        candidate_manifest=candidate_path,
    )
    assert summary["status"] == "pass"
    assert summary["runs"] == 2
    assert summary["candidate_version"] == "1.2.3"
    assert summary["aggregate"] == "qualification-evidence-1.2.3.zip"
    staged = {path.name for path in upload.iterdir() if path.is_file()}
    assert not any(path.is_dir() for path in upload.iterdir())
    expected = {
        "SHA256SUMS",
        "candidate-manifest.json",
        "candidate-SHA256SUMS",
        "index.html",
        "fastflow-sdk-release-manifest.json",
        "fastflow_sdk_ops-0.1.0.devtest-py3-none-any.whl",
        "fastflow_sdk_ops-0.1.0.devtest-py3-none-any.whl.sha256",
        "fastflow-sdk-linux-x86_64.tar.zst",
        "fastflow-sdk-windows-x86_64.zip",
        aggregate.name,
        aggregate.name + ".sha256",
        "qualification-index.json",
    }
    assert staged == expected
    assert summary["file_count"] == len(expected)
    assert set(summary["files"]) == expected
    assert sha256_file(upload / aggregate.name) == summary["aggregate_sha256"]
    assert (upload / "qualification-index.json").read_bytes() == index.read_bytes()
    assert not (upload / "simple").exists()
    # Candidate pair is outside product SHA256SUMS.
    product_listed = {
        line.split("  ", 1)[1]
        for line in (payload / "SHA256SUMS").read_text().splitlines()
        if line
    }
    assert "candidate-manifest.json" not in product_listed
    assert "candidate-SHA256SUMS" not in product_listed


def test_stage_github_release_rejects_payload_checksum_mismatch(tmp_path):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {"candidate-manifest.json": candidate_bytes, "wheel.whl": b"good"},
    )
    (payload / "wheel.whl").write_bytes(b"tampered")
    with pytest.raises(EvidenceError, match="checksum mismatch"):
        stage_github_release(payload, aggregate, tmp_path / "out", index=index)


def test_stage_github_release_rejects_extra_payload_file(tmp_path):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {"candidate-manifest.json": candidate_bytes, "wheel.whl": b"good"},
    )
    (payload / "NOTES.md").write_text("do not upload\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="extra=\\['NOTES.md'\\]"):
        stage_github_release(payload, aggregate, tmp_path / "out", index=index)


def test_stage_github_release_rejects_missing_payload_file(tmp_path):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {"candidate-manifest.json": candidate_bytes, "wheel.whl": b"good"},
    )
    (payload / "wheel.whl").unlink()
    with pytest.raises(
        EvidenceError, match="missing listed file|missing=\\['wheel.whl'\\]"
    ):
        stage_github_release(payload, aggregate, tmp_path / "out", index=index)


def test_stage_github_release_rejects_missing_candidate_checksums(tmp_path):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {"candidate-manifest.json": candidate_bytes, "wheel.whl": b"good"},
    )
    (payload / "candidate-SHA256SUMS").unlink()
    with pytest.raises(EvidenceError, match="candidate-SHA256SUMS is missing"):
        stage_github_release(payload, aggregate, tmp_path / "out", index=index)


def test_stage_github_release_rejects_mismatched_index(tmp_path):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {
            "candidate-manifest.json": candidate_bytes,
            "index.html": b"<html></html>\n",
        },
    )
    bad_index = tmp_path / "bad-index.json"
    bad_index.write_text('{"nope": true}\n', encoding="utf-8")
    with pytest.raises(EvidenceError, match="does not match the aggregate"):
        stage_github_release(payload, aggregate, tmp_path / "out", index=bad_index)


def test_stage_github_release_rejects_missing_sidecar(tmp_path):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {
            "candidate-manifest.json": candidate_bytes,
            "index.html": b"<html></html>\n",
        },
    )
    aggregate.with_name(aggregate.name + ".sha256").unlink()
    with pytest.raises(EvidenceError, match="checksum sidecar is missing"):
        stage_github_release(payload, aggregate, tmp_path / "out", index=index)


def test_stage_github_release_rejects_nested_sha256sums_paths(tmp_path):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "candidate-manifest.json").write_bytes(candidate_bytes)
    (payload / "candidate-SHA256SUMS").write_text(
        f"{hashlib.sha256(candidate_bytes).hexdigest()}  candidate-manifest.json\n",
        encoding="ascii",
    )
    nested = payload / "subdir"
    nested.mkdir()
    (nested / "wheel.whl").write_bytes(b"nested")
    (payload / "SHA256SUMS").write_text(
        f"{hashlib.sha256(b'nested').hexdigest()}  subdir/wheel.whl\n",
        encoding="ascii",
    )
    with pytest.raises(EvidenceError, match="top-level file"):
        stage_github_release(payload, aggregate, tmp_path / "out", index=index)


def test_stage_github_release_rejects_bad_candidate_checksum(tmp_path):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {
            "candidate-manifest.json": candidate_bytes,
            "index.html": b"<html></html>\n",
            "fastflow_sdk_runtime-1.2.3.whl": b"runtime",
        },
    )
    (payload / "candidate-SHA256SUMS").write_text(
        f"{'0' * 64}  candidate-manifest.json\n"
        f"{hashlib.sha256(b'runtime').hexdigest()}  fastflow_sdk_runtime-1.2.3.whl\n",
        encoding="ascii",
    )
    with pytest.raises(EvidenceError, match="candidate checksum mismatch"):
        stage_github_release(payload, aggregate, tmp_path / "out", index=index)


def test_stage_github_release_rejects_candidate_manifest_mismatch(tmp_path):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {
            "candidate-manifest.json": candidate_bytes,
            "index.html": b"<html></html>\n",
        },
    )
    other = tmp_path / "other-candidate.json"
    other.write_bytes(canonical_candidate_bytes(_candidate("f" * 40)))
    with pytest.raises(
        EvidenceError, match="does not match payload candidate-manifest.json"
    ):
        stage_github_release(
            payload,
            aggregate,
            tmp_path / "out",
            index=index,
            candidate_manifest=other,
        )


def test_stage_github_release_rejects_wrong_aggregate_basename(tmp_path):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {
            "candidate-manifest.json": candidate_bytes,
            "index.html": b"<html></html>\n",
        },
    )
    wrong = tmp_path / "aggregate.zip"
    wrong.write_bytes(aggregate.read_bytes())
    wrong.with_name(wrong.name + ".sha256").write_text(
        f"{sha256_file(wrong)}  {wrong.name}\n", encoding="ascii"
    )
    with pytest.raises(EvidenceError, match="qualification aggregate basename must be"):
        stage_github_release(payload, wrong, tmp_path / "out", index=index)


def test_stage_github_release_rejects_nonempty_output_dir(tmp_path):
    candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {
            "candidate-manifest.json": candidate_bytes,
            "index.html": b"<html></html>\n",
        },
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("nope\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="upload output directory is not empty"):
        stage_github_release(
            payload,
            aggregate,
            out,
            index=index,
            candidate_manifest=candidate_path,
        )


def test_stage_github_release_layout_contract_matches_ci_split(tmp_path):
    """Lock the real CI checksum split: product SHA256SUMS vs candidate pair."""
    candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    runtime = b"runtime-wheel-bytes"
    qual = b"qualification-wheel-bytes"
    ops = b"ops-wheel-bytes"
    product = {
        "index.html": b"<html>release</html>\n",
        "fastflow-sdk-release-manifest.json": b'{"product":"fastflow-sdk"}\n',
        "fastflow-sdk-linux-x86_64.tar.zst": b"linux-archive",
        "fastflow-sdk-linux-x86_64.tar.zst.sha256": b"unused-sidecar-text\n",
        "fastflow-sdk-windows-x86_64.zip": b"windows-archive",
        "fastflow-sdk-windows-x86_64.zip.sha256": b"unused-sidecar-text\n",
        "fastflow_sdk_runtime-1.2.3-py3-none-any.whl": runtime,
        "fastflow_sdk_runtime-1.2.3-py3-none-any.whl.sha256": (
            f"{hashlib.sha256(runtime).hexdigest()}  "
            "fastflow_sdk_runtime-1.2.3-py3-none-any.whl\n"
        ).encode("ascii"),
        "fastflow_sdk_qualification-1.2.3-py3-none-any.whl": qual,
        "fastflow_sdk_qualification-1.2.3-py3-none-any.whl.sha256": (
            f"{hashlib.sha256(qual).hexdigest()}  "
            "fastflow_sdk_qualification-1.2.3-py3-none-any.whl\n"
        ).encode("ascii"),
        "fastflow_sdk_ops-1.2.3-py3-none-any.whl": ops,
        "fastflow_sdk_ops-1.2.3-py3-none-any.whl.sha256": (
            f"{hashlib.sha256(ops).hexdigest()}  fastflow_sdk_ops-1.2.3-py3-none-any.whl\n"
        ).encode("ascii"),
    }
    payload = _write_payload(
        tmp_path / "payload",
        {"candidate-manifest.json": candidate_bytes, **product},
    )
    # Explicit CI-shaped candidate-SHA256SUMS: manifest + candidate wheels only.
    cand_lines = [
        f"{hashlib.sha256(candidate_bytes).hexdigest()}  candidate-manifest.json",
        f"{hashlib.sha256(runtime).hexdigest()}  "
        "fastflow_sdk_runtime-1.2.3-py3-none-any.whl",
        f"{hashlib.sha256(qual).hexdigest()}  "
        "fastflow_sdk_qualification-1.2.3-py3-none-any.whl",
    ]
    (payload / "candidate-SHA256SUMS").write_text(
        "\n".join(cand_lines) + "\n", encoding="ascii"
    )

    product_listed = {
        line.split("  ", 1)[1]
        for line in (payload / "SHA256SUMS").read_text().splitlines()
        if line
    }
    candidate_listed = {
        line.split("  ", 1)[1]
        for line in (payload / "candidate-SHA256SUMS").read_text().splitlines()
        if line
    }
    assert "candidate-manifest.json" not in product_listed
    assert "candidate-SHA256SUMS" not in product_listed
    assert "candidate-manifest.json" in candidate_listed
    assert "fastflow_sdk_ops-1.2.3-py3-none-any.whl" in product_listed
    assert "fastflow_sdk_ops-1.2.3-py3-none-any.whl" not in candidate_listed

    upload = tmp_path / "github-upload"
    summary = stage_github_release(
        payload,
        aggregate,
        upload,
        index=index,
        candidate_manifest=candidate_path,
    )
    assert summary["status"] == "pass"
    assert summary["file_count"] == len(list(upload.iterdir()))
    assert (upload / "candidate-manifest.json").read_bytes() == candidate_bytes
    assert (upload / "SHA256SUMS").is_file()
    assert (upload / "candidate-SHA256SUMS").is_file()
    assert not (upload / "simple").exists()


def test_stage_github_release_cli_round_trip(tmp_path, capsys):
    candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {
            "candidate-manifest.json": candidate_bytes,
            "fastflow_sdk_ops-0.1.0.devtest-py3-none-any.whl": b"ops",
        },
    )
    upload = tmp_path / "upload"
    code = cli.main(
        [
            "stage-release",
            "--payload-dir",
            str(payload),
            "--aggregate",
            str(aggregate),
            "--index",
            str(index),
            "--output-dir",
            str(upload),
            "--candidate-manifest",
            str(candidate_path),
            "--json",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "pass"
    assert report["file_count"] == len(list(upload.iterdir()))
    assert (
        upload / "fastflow_sdk_ops-0.1.0.devtest-py3-none-any.whl"
    ).read_bytes() == b"ops"
    assert (upload / aggregate.name).is_file()


def test_stage_github_release_cli_json_error_on_extra_file(tmp_path, capsys):
    _candidate_path, candidate_bytes, aggregate, index = _make_intake(tmp_path)
    payload = _write_payload(
        tmp_path / "payload",
        {
            "candidate-manifest.json": candidate_bytes,
            "index.html": b"<html></html>\n",
        },
    )
    (payload / "EXTRA.txt").write_text("nope\n", encoding="utf-8")
    code = cli.main(
        [
            "stage-release",
            "--payload-dir",
            str(payload),
            "--aggregate",
            str(aggregate),
            "--index",
            str(index),
            "--output-dir",
            str(tmp_path / "upload"),
            "--json",
        ]
    )
    assert code == 3
    payload_out = json.loads(capsys.readouterr().out)
    assert payload_out["status"] == "error"
    assert "EXTRA.txt" in payload_out["error"]
