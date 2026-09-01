# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise deterministic offline evidence verification and intake assembly."""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]

from rocm_npu_qualification import cli, installation
from rocm_npu_qualification.candidate import (
    canonical_candidate_bytes,
)
from rocm_npu_qualification.evidence import EvidenceWriter
from rocm_npu_qualification.offline import (
    CandidateMismatchError,
    EvidenceError,
    _reconcile_xadx_report,
    intake_evidence,
    pack_evidence,
    verify_archive,
    verify_evidence,
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


def test_canonical_candidate_accepts_logical_xadx_case_order():
    manifest = _candidate()
    contract = manifest["qualification_contract"]
    case_contract = contract["xadx_case_contracts"]["xadx-case"]
    contract["xadx_cases"] = ["z-case", "a-case"]
    contract["xadx_case_contracts"] = {
        "z-case": dict(case_contract),
        "a-case": dict(case_contract),
    }

    serialized = json.loads(canonical_candidate_bytes(manifest))
    normalized = installation.normalize_candidate(serialized)

    normalized_contract = normalized["qualification_contract"]
    assert normalized_contract["xadx_cases"] == ["z-case", "a-case"]
    assert set(normalized_contract["xadx_case_contracts"]) == {"z-case", "a-case"}


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


def test_failed_resident_preparation_remains_ingestible_evidence(tmp_path):
    source = _evidence(tmp_path / "resident-preparation")
    failure = {
        "case": "xadx-case",
        "phase": "resident-preparation",
        "round": None,
        "message": "preparation failed",
    }
    report_path = source / "xadx-report.json"
    report = json.loads(report_path.read_text())
    report["status"] = "fail"
    report["readback_by_case"] = {}
    report["failures"] = [failure]
    report["case_results"][0].update(
        {
            "status": "fail",
            "dispatches": 5,
            "readbacks": 0,
            "validated_bytes": 40,
            "duration_seconds": 0.5,
            "failures": [
                {key: value for key, value in failure.items() if key != "case"}
            ],
        }
    )
    writer = EvidenceWriter(source, allow_existing=True)
    writer.write_json("xadx-report.json", report)

    def mark_failed(evidence):
        evidence["status"] = "fail"
        test = next(row for row in evidence["tests"] if row["kind"] == "xadx")
        test["status"] = "fail"
        test["error"] = "resident preparation failed"
        test["rounds"] = 5
        test["duration_seconds"] = 0.5
        test["metrics"]["readback_rounds"] = 0
        test["metrics"]["validated_bytes"] = 40

    _rewrite_evidence(source, mark_failed)
    summary = verify_evidence(source)
    assert summary["evidence_status"] == "fail"


def test_report_without_v1_runtime_fields_is_rejected(tmp_path):
    source = _evidence(tmp_path / "evidence")
    report_path = source / "xadx-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("numeric_runtime")
    report.pop("readback_by_case")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    def remove_new_metric(evidence):
        test = next(record for record in evidence["tests"] if record["kind"] == "xadx")
        test["metrics"].pop("validated_bytes")

    _rewrite_evidence(source, remove_new_metric)
    with pytest.raises(EvidenceError, match="v1 schema"):
        verify_evidence(source)


def test_verify_accepts_model_files_in_different_order(tmp_path):
    source = _evidence(
        tmp_path / "evidence",
        profile_files=[
            {"path": "z-model", "sha256": "e" * 64},
            {"path": "a-template", "sha256": "f" * 64},
        ],
    )
    assert verify_evidence(source)["status"] == "pass"


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

    source = _evidence(tmp_path / "numeric-version")
    _rewrite_evidence(
        source,
        lambda evidence: evidence["subject"]["numeric_runtime"]["numpy"].update(
            {"version": "0"}
        ),
    )
    with pytest.raises(EvidenceError, match="numeric runtime"):
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
    report["resident_rounds"] = 7
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
    with pytest.raises(EvidenceError, match="execution budget"):
        verify_evidence(source)

    # Readback rounds must be accounted for per case, and only for cases that
    # the run actually reported.
    for mutation, message in (
        ({"xadx-case": 5}, "candidate case contracts"),
        ({"xadx-case": 3, "invented-case": 3}, "unknown cases"),
        ("not-a-mapping", "v1 schema"),
    ):
        source = _evidence(tmp_path / f"attribution-{message.split()[0]}")
        report = json.loads((source / "xadx-report.json").read_text())
        report["readback_by_case"] = mutation
        writer = EvidenceWriter(source, allow_existing=True)
        writer.write_json("xadx-report.json", report)
        writer.finalize()
        with pytest.raises(EvidenceError, match=message):
            verify_evidence(source)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("validated_bytes", "invalid per-case metrics"),
        ("duration_seconds", "invalid per-case metrics"),
    ],
)
def test_verify_rejects_empty_passing_case_measurements(tmp_path, field, message):
    source = _evidence(tmp_path / field)
    report_path = source / "xadx-report.json"
    report = json.loads(report_path.read_text())
    report["case_results"][0][field] = 0
    evidence = json.loads((source / "evidence.json").read_text())
    xadx_test = next(test for test in evidence["tests"] if test["kind"] == "xadx")
    if field == "duration_seconds":
        xadx_test[field] = 0
    else:
        xadx_test["metrics"][field] = 0
    writer = EvidenceWriter(source, allow_existing=True)
    writer.write_json("xadx-report.json", report)
    writer.write_json("tests.json", evidence["tests"])
    writer.write_junit(evidence["tests"])
    writer.write_json("evidence.json", evidence)
    writer.finalize()
    with pytest.raises(EvidenceError, match=message):
        verify_evidence(source)


def test_report_cannot_redistribute_or_omit_case_execution(tmp_path):
    source = _evidence(tmp_path / "history")
    report = json.loads((source / "xadx-report.json").read_text())
    evidence = json.loads((source / "evidence.json").read_text())
    report["case_results"][0]["dispatches"] = 56
    report["case_results"][0]["validated_bytes"] = 448
    xadx_test = next(test for test in evidence["tests"] if test["kind"] == "xadx")
    xadx_test["rounds"] = 56
    xadx_test["metrics"]["validated_bytes"] = 448
    with pytest.raises(EvidenceError, match="invalid per-case metrics"):
        _reconcile_xadx_report(report, evidence, _candidate()["qualification_contract"])


def test_passing_case_budget_is_exact_even_when_report_fails(tmp_path):
    source = _evidence(tmp_path / "failed-report")
    report = json.loads((source / "xadx-report.json").read_text())
    evidence = json.loads((source / "evidence.json").read_text())
    contract = _candidate()["qualification_contract"]
    contract["xadx_cases"].append("failed-case")
    contract["xadx_case_contracts"]["failed-case"] = {
        "readback_capable": False,
        "validated_bytes_per_dispatch": 8,
    }
    report["status"] = "fail"
    report["cases"] = 2
    report["case_names"].append("failed-case")
    failure = {
        "case": "failed-case",
        "phase": "resident-preparation",
        "round": None,
        "message": "preparation failed",
    }
    report["failures"] = [failure]
    report["case_results"].append(
        {
            "name": "failed-case",
            "status": "fail",
            "dispatches": 0,
            "readbacks": 0,
            "validated_bytes": 0,
            "duration_seconds": 0,
            "mismatch_count": 0,
            "first_mismatch": None,
            "failures": [
                {key: value for key, value in failure.items() if key != "case"}
            ],
        }
    )
    failed_test = dict(
        next(test for test in evidence["tests"] if test["kind"] == "xadx")
    )
    failed_test["name"] = "failed-case"
    failed_test["status"] = "fail"
    evidence["tests"].append(failed_test)
    report["case_results"][0]["dispatches"] = 0
    report["case_results"][0]["validated_bytes"] = 0
    with pytest.raises(EvidenceError, match="invalid per-case metrics"):
        _reconcile_xadx_report(report, evidence, contract)


def test_report_materializations_are_candidate_bound(tmp_path):
    source = _evidence(tmp_path / "materializations")
    report = json.loads((source / "xadx-report.json").read_text())
    evidence = json.loads((source / "evidence.json").read_text())
    report["materialized_buffers"] = []
    with pytest.raises(EvidenceError, match="materialization inventory"):
        _reconcile_xadx_report(report, evidence, _candidate()["qualification_contract"])


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
