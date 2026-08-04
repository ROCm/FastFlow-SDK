# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise deterministic offline evidence verification and intake assembly."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]
QUALIFICATION_PYTHON = ROOT / "tools" / "rocm-npu-qualification" / "python"
sys.path.insert(0, str(QUALIFICATION_PYTHON))

from rocm_npu_qualification import cli, installation  # noqa: E402
from rocm_npu_qualification.candidate import (  # noqa: E402
    canonical_candidate_bytes,
)
from rocm_npu_qualification.evidence import EvidenceWriter  # noqa: E402
from rocm_npu_qualification.offline import (  # noqa: E402
    CandidateMismatchError,
    EvidenceError,
    intake_evidence,
    pack_evidence,
    verify_archive,
    verify_evidence,
)


pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def _profile():
    return {
        "schema_version": 1,
        "name": "fast",
        "description": "test",
        "models": [
            {
                "tag": "qwen3:0.6b",
                "source": "https://test",
                "revision": "immutable",
                "identity_policy": "revision-sha256",
                "registry_sha256": "d" * 64,
                "files": [{"path": "model", "sha256": "e" * 64}],
                "xclbins": [{"path": "kernel", "sha256": "f" * 64}],
                "tests": [
                    {
                        "name": "qualification",
                        "kind": "run",
                        "turns": [{"prompt": "test", "expected": "test"}],
                        "setup_commands": [],
                        "process_iterations": 1,
                        "timeout_seconds": 30,
                        "ctx_len": 128,
                    }
                ],
            }
        ],
    }


def _policy(*platforms: str):
    return {
        "schema_version": 1,
        "name": "test-" + "-".join(platforms),
        "requirements": [
            {
                "platform_os": platform,
                "profile": "fast",
                "target_family": "npu2",
                "minimum_runs": 1,
            }
            for platform in platforms
        ],
    }


def _candidate(revision: str = "a" * 40):
    digest = "b" * 64
    return {
        "schema_version": 1,
        "product": "rocm-npu-release-candidate",
        "version": "1.2.3",
        "source_revision": revision,
        "packages": {
            distribution: [
                {
                    "filename": f"{distribution.replace('-', '_')}-1.2.3.whl",
                    "version": "1.2.3",
                    "sha256": digest,
                    "package_manifest_sha256": "c" * 64,
                }
            ]
            for distribution in installation.PACKAGE_DISTRIBUTIONS
        },
        "qualification_contract": {
            "schema_version": 1,
            "native_checks": ["version", "probe", "doctor", "self-test"],
            "xadx_cases": ["xadx-case"],
            "xadx_execution": {
                "lifecycle_rounds": 5,
                "resident_rounds": 10,
                "readback_rounds": 32,
            },
            "xadx_provenance": {
                "case_contract_sha256": "d" * 64,
                "manifest_sha256": "e" * 64,
                "serializer": {
                    "api": "hrx_amdxdna_xadx_serialize",
                    "revision": "hrx-amdxdna-v0",
                },
                "executables": [
                    {
                        "id": "test",
                        "source_xclbin_sha256": "1" * 64,
                        "transaction_sha256": "2" * 64,
                        "xadx_sha256": "3" * 64,
                    }
                ],
            },
            "profiles": {"fast": _profile()},
            "policies": {
                policy["name"]: policy
                for policy in (
                    _policy("windows"),
                    _policy("windows", "linux"),
                )
            },
        },
    }


def _evidence(
    root: Path,
    *,
    candidate=None,
    marker="one",
    status="pass",
    platform_os="Windows 11",
) -> Path:
    writer = EvidenceWriter(root)
    candidate = _candidate() if candidate is None else candidate
    writer.write_candidate_manifest(candidate)
    profile = _profile()
    model = {
        "tag": "qwen3:0.6b",
        "identity_policy": "revision-sha256",
        "source": "https://test",
        "revision": "immutable",
        "status": "pass",
        "cache_directory": "model",
        "registry_sha256": "d" * 64,
        "pulled": False,
        "files": [{"path": "model", "size": 1, "sha256": "e" * 64}],
    }
    log = writer.path(f"logs/tests/{marker}.log")
    log.write_text("passed\n")
    test = {
        "model": model["tag"],
        "name": "qualification",
        "kind": "run",
        "status": status,
        "returncode": 0,
        "timed_out": False,
        "duration_seconds": 1,
        "log": writer.relative(log),
        "turn_count": 1,
        "process_iteration": 1,
        "soak_iterations": 1,
        "metrics": {"responses": 1},
        "artifacts": [],
        **({"error": "expected test failure"} if status != "pass" else {}),
    }
    embedded = canonical_candidate_bytes(candidate)
    evidence = {
        "schema_version": 1,
        "status": status,
        "subject": {
            "version": "1.2.3",
            "source_revision": candidate["source_revision"],
            "candidate_binding": {
                "present": True,
                "manifest_sha256": hashlib.sha256(embedded).hexdigest(),
            },
            "packages": {
                distribution: "1.2.3"
                for distribution in installation.PACKAGE_DISTRIBUTIONS
            },
            "flm": {
                "package_version": "1.2.3",
                "command": "flm",
                "version_output": "FLM v1.2.3",
            },
        },
        "observed_platform": {
            "architecture": "x86_64",
            "os": platform_os,
            "target_family": "npu2",
        },
        "profile": {
            "name": "fast",
            "description": "test",
            "candidate_bound": True,
        },
        "models": [model],
        "tests": [
            test,
            {
                "model": model["tag"],
                "name": "production-xclbin-inventory",
                "kind": "xclbin",
                "status": "pass",
                "duration_seconds": 0,
                "log": "",
                "metrics": {"files_verified": 1},
            },
            *[
                {
                    "model": "native-core",
                    "name": name,
                    "kind": "native",
                    "status": "pass",
                    "returncode": 0,
                    "duration_seconds": 0,
                    "log": writer.relative(log),
                    "metrics": {"report_status": "pass"},
                }
                for name in ("version", "probe", "doctor", "self-test")
            ],
            {
                "model": "fastflowlm-production-kernels",
                "name": "xadx-case",
                "kind": "xadx",
                "status": "pass",
                "duration_seconds": 1,
                "log": "xadx-report.json",
                "rounds": 15,
                "metrics": {
                    "architecture": "AMD_AIE2_NPU2",
                    "lifecycle_rounds": 5,
                    "resident_rounds": 10,
                    "readback_rounds": 0,
                },
                "artifacts": ["xadx-report.json"],
            },
        ],
        "started_at": "2026-08-04T00:00:00Z",
        "finished_at": "2026-08-04T00:01:00Z",
    }
    writer.write_json("profile.json", profile)
    writer.write_json("models.json", evidence["models"])
    writer.write_json("tests.json", evidence["tests"])
    for command in ("version", "probe", "doctor", "self-test"):
        report = {
            "schema_version": 1,
            "status": "pass",
            "command": command,
            "product": {
                "version": "1.2.3",
                "source_revision": candidate["source_revision"],
            },
            "checks": [],
        }
        if command == "probe":
            report["data"] = evidence["observed_platform"]
        writer.write_json(
            "core-version.json" if command == "version" else f"{command}.json",
            report,
        )
    writer.write_json(
        "xadx-report.json",
        {
            "status": "pass",
            "architecture": "AMD_AIE2_NPU2",
            "lifecycle_rounds": 5,
            "resident_rounds": 10,
            "readback_rounds": 32,
            "case_names": ["xadx-case"],
            "failures": [],
        },
    )
    writer.write_junit(evidence["tests"])
    writer.write_json("evidence.json", evidence)
    writer.finalize()
    return root


def _rewrite_evidence(root: Path, mutate) -> None:
    evidence = json.loads((root / "evidence.json").read_text())
    mutate(evidence)
    writer = EvidenceWriter(root, allow_existing=True)
    writer.write_json("models.json", evidence["models"])
    writer.write_json("tests.json", evidence["tests"])
    writer.write_junit(evidence["tests"])
    writer.write_json("evidence.json", evidence)
    writer.finalize()


def test_deterministic_pack_and_offline_verify(tmp_path):
    source = _evidence(tmp_path / "evidence")
    first = tmp_path / "first.evidence.zip"
    second = tmp_path / "second.evidence.zip"
    pack_evidence(source, first)
    pack_evidence(source, second)
    assert first.read_bytes() == second.read_bytes()
    summary = verify_evidence(first)
    assert summary["status"] == "pass"
    assert summary["evidence_status"] == "pass"


def test_verify_rejects_checksum_tampering_and_unlisted_files(tmp_path):
    source = _evidence(tmp_path / "evidence")
    (source / "logs/tests/one.log").write_text("tampered\n")
    with pytest.raises(EvidenceError, match="checksum mismatch"):
        verify_evidence(source)
    source = _evidence(tmp_path / "other")
    (source / "unlisted.txt").write_text("hidden\n")
    with pytest.raises(
        EvidenceError, match="unexpected nesting|complete and exclusive"
    ):
        verify_evidence(source)


def test_verify_binds_subject_identity_and_full_evidence_schema(tmp_path):
    source = _evidence(tmp_path / "identity")
    _rewrite_evidence(
        source,
        lambda evidence: evidence["subject"].update({"source_revision": "f" * 40}),
    )
    with pytest.raises(CandidateMismatchError, match="subject"):
        verify_evidence(source)

    source = _evidence(tmp_path / "schema")
    _rewrite_evidence(
        source,
        lambda evidence: evidence["models"].__setitem__(0, {"status": "pass"}),
    )
    with pytest.raises(EvidenceError, match="v1 schema"):
        verify_evidence(source)


@pytest.mark.parametrize("name", ("../escape", "LOGS/tests/one.log"))
def test_verify_rejects_unsafe_or_case_colliding_zip_members(tmp_path, name):
    source = _evidence(tmp_path / "evidence")
    bundle = tmp_path / "bad.evidence.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            archive.writestr(path.relative_to(source).as_posix(), path.read_bytes())
        archive.writestr(name, b"bad")
    with pytest.raises(EvidenceError):
        verify_evidence(bundle)


def test_verify_rejects_zip_symlink(tmp_path):
    source = _evidence(tmp_path / "evidence")
    bundle = tmp_path / "symlink.evidence.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            archive.writestr(path.relative_to(source).as_posix(), path.read_bytes())
        info = zipfile.ZipInfo("logs/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b"target")
    with pytest.raises(EvidenceError, match="special member"):
        verify_evidence(bundle)


def test_candidate_mismatch_and_intake_round_trip(tmp_path):
    candidate = _candidate()
    candidate_path = tmp_path / "candidate-manifest.json"
    candidate_path.write_bytes(canonical_candidate_bytes(candidate))
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    for marker in ("one", "two"):
        source = _evidence(tmp_path / f"evidence-{marker}", marker=marker)
        pack_evidence(source, incoming / f"{marker}.evidence.zip")
    with pytest.raises(CandidateMismatchError):
        verify_evidence(
            incoming / "one.evidence.zip",
            candidate_manifest=(tmp_path / "wrong.json"),
        )

    # Write the mismatching candidate only after exercising the missing-file path.
    wrong = _candidate("f" * 40)
    wrong_path = tmp_path / "wrong-candidate.json"
    wrong_path.write_bytes(canonical_candidate_bytes(wrong))
    with pytest.raises(CandidateMismatchError):
        verify_evidence(incoming / "one.evidence.zip", candidate_manifest=wrong_path)
    (tmp_path / "unrelated.whl").write_bytes(b"not a candidate wheel")
    assert (
        verify_evidence(
            incoming / "one.evidence.zip",
            candidate_manifest=candidate_path,
        )["status"]
        == "pass"
    )

    output = tmp_path / "assembled"
    aggregate = tmp_path / "aggregate.zip"
    summary = intake_evidence(
        incoming,
        candidate_path,
        output,
        aggregate,
        require_status="pass",
        policy=_policy("windows"),
    )
    assert summary["runs"] == 2
    index = json.loads((output / "qualification-index.json").read_text())
    assert [run["bundle_sha256"] for run in index["runs"]] == sorted(
        run["bundle_sha256"] for run in index["runs"]
    )
    schema = json.loads(
        (
            ROOT
            / "tools/rocm-npu-qualification/python/rocm_npu_qualification"
            / "schemas/qualification-index-v1.schema.json"
        ).read_text()
    )
    Draft202012Validator(schema).validate(index)
    with zipfile.ZipFile(aggregate) as archive:
        assert "qualification-index.json" in archive.namelist()
    assert aggregate.with_name("aggregate.zip.sha256").is_file()
    assert verify_archive(aggregate, candidate_manifest=candidate_path)["runs"] == 2
    with zipfile.ZipFile(aggregate) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    index = json.loads(members["qualification-index.json"])
    index["policy"]["requirements"][0]["matched_runs"] = ["0" * 64]
    members["qualification-index.json"] = (
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    ).encode()
    members["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(members.items())
        if name != "SHA256SUMS"
    ).encode("ascii")
    with zipfile.ZipFile(aggregate, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data)
    aggregate.with_name(aggregate.name + ".sha256").write_text(
        f"{hashlib.sha256(aggregate.read_bytes()).hexdigest()}  {aggregate.name}\n",
        encoding="ascii",
    )
    with pytest.raises(EvidenceError, match="policy attestation"):
        verify_archive(aggregate, candidate_manifest=candidate_path)


def test_release_wide_candidate_combines_windows_and_linux_evidence(tmp_path):
    candidate = _candidate()
    for distribution in ("rocm-npu-runtime", "rocm-npu-fastflowlm"):
        identity = dict(candidate["packages"][distribution][0])
        identity["filename"] = identity["filename"].replace(".whl", "-manylinux.whl")
        identity["sha256"] = "1" * 64
        candidate["packages"][distribution].append(identity)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_bytes(canonical_candidate_bytes(candidate))
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    for name, platform_os in (
        ("windows", "Windows 11"),
        ("linux", "Linux 6.8"),
    ):
        pack_evidence(
            _evidence(
                tmp_path / name,
                candidate=candidate,
                marker=name,
                platform_os=platform_os,
            ),
            incoming / f"{name}.evidence.zip",
        )
    summary = intake_evidence(
        incoming,
        candidate_path,
        tmp_path / "assembled",
        tmp_path / "aggregate.zip",
        require_status="pass",
        policy=_policy("windows", "linux"),
    )
    assert summary["policy"]["satisfied"] is True
    assert len(summary["policy"]["requirements"]) == 2


def test_verify_rejects_incomplete_passing_profile_execution(tmp_path):
    source = _evidence(tmp_path / "evidence")
    _rewrite_evidence(
        source,
        lambda evidence: evidence["tests"].pop(),
    )
    with pytest.raises(EvidenceError, match="execution inventory"):
        verify_evidence(source)


def test_verify_rejects_profile_not_bound_to_qualification_wheel(tmp_path):
    source = _evidence(tmp_path / "evidence")
    profile = json.loads((source / "profile.json").read_text())
    evidence = json.loads((source / "evidence.json").read_text())
    profile["description"] = "forged profile"
    evidence["profile"]["description"] = "forged profile"
    writer = EvidenceWriter(source, allow_existing=True)
    writer.write_json("profile.json", profile)
    writer.write_json("evidence.json", evidence)
    writer.finalize()
    with pytest.raises(EvidenceError, match="qualification wheel"):
        verify_evidence(source)


def test_verify_rejects_platform_not_bound_to_probe(tmp_path):
    source = _evidence(tmp_path / "evidence")
    _rewrite_evidence(
        source,
        lambda evidence: evidence["observed_platform"].__setitem__("os", "Linux 6.8"),
    )
    with pytest.raises(EvidenceError, match="does not match probe"):
        verify_evidence(source)


def test_verify_rejects_missing_or_forged_native_reports(tmp_path):
    source = _evidence(tmp_path / "missing")
    (source / "doctor.json").unlink()
    EvidenceWriter(source, allow_existing=True).finalize()
    with pytest.raises(EvidenceError, match="missing doctor.json"):
        verify_evidence(source)

    source = _evidence(tmp_path / "forged")
    report = json.loads((source / "doctor.json").read_text())
    report["status"] = "fail"
    EvidenceWriter(source, allow_existing=True).write_json("doctor.json", report)
    EvidenceWriter(source, allow_existing=True).finalize()
    with pytest.raises(EvidenceError, match="native test record"):
        verify_evidence(source)


def test_verify_rejects_forged_xadx_report_and_rounds(tmp_path):
    source = _evidence(tmp_path / "report")
    report = json.loads((source / "xadx-report.json").read_text())
    report["resident_rounds"] = 1
    writer = EvidenceWriter(source, allow_existing=True)
    writer.write_json("xadx-report.json", report)
    writer.finalize()
    with pytest.raises(EvidenceError, match="round counts"):
        verify_evidence(source)

    source = _evidence(tmp_path / "record")
    _rewrite_evidence(
        source,
        lambda evidence: evidence["tests"][-1].__setitem__("rounds", 1),
    )
    with pytest.raises(EvidenceError, match="invalid rounds"):
        verify_evidence(source)


def test_verify_rejects_junit_not_derived_from_tests(tmp_path):
    source = _evidence(tmp_path / "evidence")
    junit = (source / "junit.xml").read_text(encoding="utf-8")
    (source / "junit.xml").write_text(
        junit.replace('tests="7"', 'tests="999"'), encoding="utf-8"
    )
    EvidenceWriter(source, allow_existing=True).finalize()
    with pytest.raises(EvidenceError, match="junit.xml does not match"):
        verify_evidence(source)


def test_verify_rejects_model_identity_not_bound_to_profile(tmp_path):
    source = _evidence(tmp_path / "evidence")
    _rewrite_evidence(
        source,
        lambda evidence: evidence["models"][0].__setitem__(
            "source", "https://forged.invalid/model"
        ),
    )
    with pytest.raises(EvidenceError, match="model identity"):
        verify_evidence(source)


def test_intake_rejects_missing_required_platform(tmp_path):
    candidate = _candidate()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_bytes(canonical_candidate_bytes(candidate))
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pack_evidence(
        _evidence(tmp_path / "windows", candidate=candidate),
        incoming / "windows.evidence.zip",
    )
    with pytest.raises(EvidenceError, match="does not satisfy policy"):
        intake_evidence(
            incoming,
            candidate_path,
            tmp_path / "assembled",
            tmp_path / "aggregate.zip",
            require_status="pass",
            policy=_policy("windows", "linux"),
        )


def test_intake_rejects_policy_not_bound_to_candidate(tmp_path):
    candidate = _candidate()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_bytes(canonical_candidate_bytes(candidate))
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pack_evidence(
        _evidence(tmp_path / "windows", candidate=candidate),
        incoming / "windows.evidence.zip",
    )
    policy = _policy("windows")
    policy["requirements"][0]["minimum_runs"] = 2
    with pytest.raises(EvidenceError, match="qualification wheel"):
        intake_evidence(
            incoming,
            candidate_path,
            tmp_path / "assembled",
            tmp_path / "aggregate.zip",
            require_status="pass",
            policy=policy,
        )


def test_failed_intake_leaves_no_partial_output(tmp_path):
    candidate = _candidate()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_bytes(canonical_candidate_bytes(candidate))
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pack_evidence(
        _evidence(tmp_path / "good", candidate=candidate),
        incoming / "a-good.evidence.zip",
    )
    wrong = _candidate("f" * 40)
    pack_evidence(
        _evidence(tmp_path / "wrong", candidate=wrong),
        incoming / "z-wrong.evidence.zip",
    )
    output = tmp_path / "assembled"
    with pytest.raises(CandidateMismatchError):
        intake_evidence(
            incoming,
            candidate_path,
            output,
            tmp_path / "aggregate.zip",
            require_status="pass",
            policy=_policy("windows"),
        )
    assert not output.exists()


def test_intake_rolls_back_directory_if_bundle_publish_fails(monkeypatch, tmp_path):
    candidate = _candidate()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_bytes(canonical_candidate_bytes(candidate))
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    pack_evidence(
        _evidence(tmp_path / "good", candidate=candidate),
        incoming / "good.evidence.zip",
    )
    output = tmp_path / "assembled"
    aggregate = tmp_path / "aggregate.zip"
    original_replace = Path.replace

    def fail_bundle_publish(path, target):
        if Path(target) == aggregate:
            raise OSError("simulated aggregate publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_bundle_publish)
    with pytest.raises(OSError, match="simulated"):
        intake_evidence(
            incoming,
            candidate_path,
            output,
            aggregate,
            require_status="pass",
            policy=_policy("windows"),
        )
    assert not output.exists()


def test_failed_run_is_sealed_when_bundle_output_is_requested(
    monkeypatch, tmp_path, capsys
):
    def fail(args):
        _evidence(args.output, status="fail")
        return 1, {
            "schema_version": 1,
            "status": "fail",
            "output": str(args.output),
        }

    monkeypatch.setattr(cli, "run_qualification", fail)
    bundle = tmp_path / "failed.evidence.zip"
    assert (
        cli.main(
            [
                "run",
                "--output-dir",
                str(tmp_path / "failed"),
                "--bundle-output",
                str(bundle),
                "--json",
            ]
        )
        == 1
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["bundle_output"] == str(bundle.resolve())
    assert verify_evidence(bundle)["evidence_status"] == "fail"


def test_pack_and_verify_cli_do_not_require_plugin_protocol(tmp_path, capsys):
    source = _evidence(tmp_path / "evidence")
    bundle = tmp_path / "run.evidence.zip"
    assert (
        cli.main(
            [
                "pack",
                "--input",
                str(source),
                "--output",
                str(bundle),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["verify", "--input", str(bundle), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
