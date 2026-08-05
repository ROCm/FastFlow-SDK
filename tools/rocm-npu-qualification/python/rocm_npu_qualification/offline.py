# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic evidence sealing, offline verification, and intake assembly."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile
from datetime import datetime
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

from .candidate import (
    CandidateError,
    canonical_candidate_bytes,
    normalize_candidate_manifest,
)
from .contracts import (
    COMMAND_REPORT_SCHEMA_VERSION,
    INDEX_SCHEMA_VERSION,
    POLICY_SCHEMA_VERSION,
)
from .evidence import junit_bytes
from .hashing import sha256_file, write_sha256sums
from .policy import PolicyValidationError, normalize_policy
from .schema import (
    SchemaValidationError,
    load_packaged_schema,
    validate_schema,
)


MAX_ARCHIVE_FILES = 4096
MAX_ARCHIVE_MEMBER_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1000
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
ROOT_FILES = {
    "SHA256SUMS",
    "candidate-manifest.json",
    "core-version.json",
    "doctor.json",
    "evidence.json",
    "junit.xml",
    "models.json",
    "probe.json",
    "profile.json",
    "self-test.json",
    "tests.json",
    "xadx-report.json",
}


class EvidenceError(RuntimeError):
    pass


class CandidateMismatchError(EvidenceError):
    pass


class IntakePolicyError(EvidenceError):
    pass


def _safe_name(name: str) -> str:
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or name.startswith("/")
        or unicodedata.normalize("NFC", name) != name
    ):
        raise EvidenceError(f"unsafe archive path: {name!r}")
    parsed = PurePosixPath(name)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise EvidenceError(f"unsafe archive path: {name!r}")
    return parsed.as_posix()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _json_value(data: bytes, name: str) -> Any:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError(f"{name} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            data,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceError(f"{name} contains non-finite number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{name} is invalid JSON") from error


def _json_object(data: bytes, name: str) -> dict[str, Any]:
    value = _json_value(data, name)
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must contain a JSON object")
    return value


def _directory_members(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        raise EvidenceError(f"evidence directory does not exist: {root}")
    members: dict[str, bytes] = {}
    seen: set[str] = set()
    total = 0
    for path in sorted(root.rglob("*")):
        relative = _safe_name(path.relative_to(root).as_posix())
        folded = relative.casefold()
        if folded in seen:
            raise EvidenceError(f"case-colliding evidence path: {relative}")
        seen.add(folded)
        if path.is_symlink():
            raise EvidenceError(f"evidence contains symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise EvidenceError(f"evidence contains special file: {relative}")
        size = path.stat().st_size
        if size > MAX_ARCHIVE_MEMBER_BYTES:
            raise EvidenceError(f"evidence member is too large: {relative}")
        total += size
        if len(members) >= MAX_ARCHIVE_FILES:
            raise EvidenceError("evidence directory has too many members")
        if total > MAX_ARCHIVE_TOTAL_BYTES:
            raise EvidenceError("evidence directory exceeds the size limit")
        members[relative] = path.read_bytes()
    return members


def _archive_members(path: Path | bytes) -> dict[str, bytes]:
    source = io.BytesIO(path) if isinstance(path, bytes) else path
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as error:
        raise EvidenceError(f"evidence archive is invalid: {path}") from error
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_FILES:
            raise EvidenceError("evidence archive has too many members")
        members: dict[str, bytes] = {}
        seen: set[str] = set()
        total = 0
        for info in infos:
            name = _safe_name(info.filename)
            folded = name.casefold()
            if folded in seen:
                raise EvidenceError(f"duplicate or case-colliding archive path: {name}")
            seen.add(folded)
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode) or (mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                raise EvidenceError(f"archive contains special member: {name}")
            if info.is_dir():
                raise EvidenceError(f"archive contains unexpected directory entry: {name}")
            if (
                info.file_size > MAX_ARCHIVE_MEMBER_BYTES
                or info.file_size < 0
                or info.compress_size < 0
            ):
                raise EvidenceError(f"archive member is too large: {name}")
            if info.file_size and (
                info.compress_size == 0
                or info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
            ):
                raise EvidenceError(
                    f"archive member compression ratio is unsafe: {name}"
                )
            total += info.file_size
            if total > MAX_ARCHIVE_TOTAL_BYTES:
                raise EvidenceError("evidence archive expands beyond the size limit")
            try:
                members[name] = archive.read(info)
            except (RuntimeError, zipfile.BadZipFile) as error:
                raise EvidenceError(f"cannot read archive member: {name}") from error
        return members


def _write_deterministic_zip(members: dict[str, bytes], output: Path) -> str:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for name, data in sorted(members.items()):
                safe = _safe_name(name)
                info = zipfile.ZipInfo(safe, FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(info, data)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(output)


def _parse_checksums(members: dict[str, bytes]) -> None:
    checksum_bytes = members.get("SHA256SUMS")
    if checksum_bytes is None:
        raise EvidenceError("evidence is missing SHA256SUMS")
    try:
        lines = checksum_bytes.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise EvidenceError("SHA256SUMS is not ASCII") from error
    expected: dict[str, str] = {}
    folded: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise EvidenceError("SHA256SUMS has a malformed row")
        digest, name = match.groups()
        name = _safe_name(name)
        if name == "SHA256SUMS" or name in expected or name.casefold() in folded:
            raise EvidenceError(f"SHA256SUMS has a duplicate path: {name}")
        expected[name] = digest
        folded.add(name.casefold())
    actual_names = set(members) - {"SHA256SUMS"}
    if set(expected) != actual_names:
        raise EvidenceError(
            "SHA256SUMS is not complete and exclusive; "
            f"missing={sorted(actual_names - set(expected))}, "
            f"unlisted={sorted(set(expected) - actual_names)}"
        )
    for name, digest in expected.items():
        if hashlib.sha256(members[name]).hexdigest() != digest:
            raise EvidenceError(f"evidence checksum mismatch: {name}")


def _validate_layout(members: dict[str, bytes]) -> None:
    if not members or len(members) > MAX_ARCHIVE_FILES:
        raise EvidenceError("evidence member count is invalid")
    required = {
        "SHA256SUMS",
        "candidate-manifest.json",
        "evidence.json",
        "junit.xml",
        "models.json",
        "profile.json",
        "tests.json",
    }
    missing = sorted(required - set(members))
    if missing:
        raise EvidenceError(f"evidence is missing required files: {missing}")
    for name in members:
        top = name.split("/", 1)[0]
        if name not in ROOT_FILES and top not in {"bench", "logs"}:
            raise EvidenceError(f"evidence has unexpected nesting: {name}")


def _iso_timestamp(value: Any, label: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str):
        raise EvidenceError(f"evidence {label} is invalid")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError(f"evidence {label} is invalid") from error


def _normalized_platform(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise EvidenceError("observed platform is invalid")
    os_value = value.get("os")
    if not isinstance(os_value, str):
        raise EvidenceError("observed platform OS is invalid")
    if re.fullmatch(r"Windows(?:\s.*)?", os_value, re.IGNORECASE):
        os_name = "windows"
    elif re.fullmatch(r"Linux(?:\s.*)?", os_value, re.IGNORECASE):
        os_name = "linux"
    else:
        raise EvidenceError(f"observed platform OS is unsupported: {os_value!r}")
    architecture = value.get("architecture")
    target_family = value.get("target_family")
    if (
        not isinstance(architecture, str)
        or not architecture
        or not isinstance(target_family, str)
        or not target_family
    ):
        raise EvidenceError("observed platform identity is incomplete")
    return {
        "architecture": architecture.casefold(),
        "os": os_name,
        "target_family": target_family.casefold(),
    }


def _validate_complete_execution(
    evidence: dict[str, Any],
    profile: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    if evidence["status"] != "pass":
        return
    model_tags = [model["tag"] for model in profile["models"]]
    observed_models = [
        model.get("tag") for model in evidence["models"] if isinstance(model, dict)
    ]
    if observed_models != model_tags:
        raise EvidenceError("passing evidence does not contain every profile model")
    for expected_model, observed_model in zip(profile["models"], evidence["models"]):
        expected_identity = {
            key: expected_model[key]
            for key in (
                "tag",
                "source",
                "revision",
                "identity_policy",
                "registry_sha256",
            )
        }
        observed_identity = {
            key: observed_model.get(key) for key in expected_identity
        }
        expected_files = [
            {"path": record["path"], "sha256": record["sha256"]}
            for record in expected_model["files"]
        ]
        observed_files = [
            {"path": record.get("path"), "sha256": record.get("sha256")}
            for record in observed_model.get("files", [])
            if isinstance(record, dict)
        ]
        expected_files.sort(key=lambda record: (record["path"], record["sha256"]))
        observed_files.sort(key=lambda record: (record["path"], record["sha256"]))
        if (
            observed_identity != expected_identity
            or observed_files != expected_files
        ):
            raise EvidenceError(
                f"passing evidence model identity does not match profile: "
                f"{expected_model['tag']}"
            )

    expected: dict[tuple[str, str, str], dict[str, int]] = {}
    for name in contract["native_checks"]:
        expected[("native", "native-core", name)] = {}
    for name in contract["xadx_cases"]:
        execution = contract["xadx_execution"]
        expected[("xadx", "fastflowlm-production-kernels", name)] = {
            "rounds": (
                execution["lifecycle_rounds"]
                + execution["resident_rounds"]
                + (
                    execution["readback_rounds"]
                    if name == "host-readback/coherence"
                    else 0
                )
            )
        }
    for model in profile["models"]:
        if model["xclbins"]:
            expected[
                ("xclbin", model["tag"], "production-xclbin-inventory")
            ] = {"files_verified": len(model["xclbins"])}
        for test in model["tests"]:
            if test["kind"] == "bench":
                expected[("bench", model["tag"], test["name"])] = {
                    "turn_count": 1,
                    "process_iteration": 1,
                }
                continue
            for iteration in range(1, test["process_iterations"] + 1):
                name = (
                    test["name"]
                    if test["process_iterations"] == 1
                    else f"{test['name']}-{iteration:02d}"
                )
                expected[("run", model["tag"], name)] = {
                    "turn_count": len(test["turns"]),
                    "process_iteration": iteration,
                    "soak_iterations": test["process_iterations"],
                }

    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for test in evidence["tests"]:
        key = (test.get("kind"), test.get("model"), test.get("name"))
        if key in observed:
            raise EvidenceError("passing evidence contains a duplicate test result")
        observed[key] = test
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise EvidenceError(
            "passing evidence execution inventory is incomplete: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, fields in expected.items():
        record = observed[key]
        for field, value in fields.items():
            observed_value = (
                record.get("metrics", {}).get(field)
                if field == "files_verified"
                else record.get(field)
            )
            if observed_value != value:
                raise EvidenceError(
                    f"passing evidence has invalid {field} for {key}"
                )


def _reconcile_native_reports(
    reports: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
    native_checks: list[str],
) -> None:
    tests = {
        test.get("name"): test
        for test in evidence["tests"]
        if test.get("kind") == "native" and test.get("model") == "native-core"
    }
    if set(tests) != set(native_checks):
        raise EvidenceError("native reports do not match the native test inventory")
    for command in native_checks:
        name = f"core-{command}.json" if command == "version" else f"{command}.json"
        # The version report keeps its historical root filename.
        if command == "version":
            name = "core-version.json"
        report = reports.get(name)
        if report is None:
            raise EvidenceError(f"passing evidence is missing {name}")
        if (
            report.get("schema_version") != COMMAND_REPORT_SCHEMA_VERSION
            or report.get("command") != command
            or report.get("status") not in {"pass", "fail", "error"}
            or not isinstance(report.get("product"), dict)
            or report["product"].get("version") != evidence["subject"]["version"]
        ):
            raise EvidenceError(f"{name} does not match the evidence subject")
        source_revision = evidence["subject"].get("source_revision")
        if source_revision and report["product"].get("source_revision") != source_revision:
            raise EvidenceError(f"{name} source revision does not match the candidate")
        test = tests[command]
        report_passed = report["status"] == "pass"
        if (
            test.get("metrics") != {"report_status": report["status"]}
            or (test.get("status") == "pass")
            != (report_passed and test.get("returncode") == 0)
        ):
            raise EvidenceError(f"{name} does not match its native test record")


def _reconcile_xadx_report(
    report: dict[str, Any],
    evidence: dict[str, Any],
    contract: dict[str, Any] | None,
) -> None:
    tests = [
        test
        for test in evidence["tests"]
        if test.get("kind") == "xadx"
        and test.get("model") == "fastflowlm-production-kernels"
    ]
    expected_names = (
        contract["xadx_cases"] if contract is not None else [test.get("name") for test in tests]
    )
    if report.get("case_names") != expected_names:
        raise EvidenceError("xadx-report.json case inventory does not match tests.json")
    status = report.get("status")
    if status not in {"pass", "fail", "unsupported"}:
        raise EvidenceError("xadx-report.json status is invalid")
    rounds = {
        key: report.get(key)
        for key in ("lifecycle_rounds", "resident_rounds", "readback_rounds")
    }
    if any(not isinstance(value, int) or value < 0 for value in rounds.values()):
        raise EvidenceError("xadx-report.json round counts are invalid")
    if contract is not None and rounds != contract["xadx_execution"]:
        raise EvidenceError(
            "xadx-report.json round counts do not match the qualification contract"
        )
    if len(tests) != len(expected_names):
        raise EvidenceError("xadx-report.json does not match the XADX test inventory")
    for test, name in zip(tests, expected_names):
        readback = (
            rounds["readback_rounds"] if name == "host-readback/coherence" else 0
        )
        expected_metrics = {
            "architecture": report.get("architecture", "unknown"),
            "lifecycle_rounds": rounds["lifecycle_rounds"],
            "resident_rounds": rounds["resident_rounds"],
            "readback_rounds": readback,
        }
        if (
            test.get("name") != name
            or test.get("status") != status
            or test.get("rounds")
            != rounds["lifecycle_rounds"] + rounds["resident_rounds"] + readback
            or test.get("metrics") != expected_metrics
            or test.get("log") != "xadx-report.json"
            or test.get("artifacts") != ["xadx-report.json"]
        ):
            raise EvidenceError(
                f"xadx-report.json does not match its test record: {name}"
            )


def _reconcile_reports(
    members: dict[str, bytes],
    evidence: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    contract: dict[str, Any] | None,
) -> None:
    if members["junit.xml"] != junit_bytes(evidence["tests"]):
        raise EvidenceError("junit.xml does not match tests.json")
    if evidence["status"] != "pass":
        return
    native_checks = (
        contract["native_checks"]
        if contract is not None
        else ["version", "probe", "doctor", "self-test"]
    )
    _reconcile_native_reports(reports, evidence, native_checks)
    report = reports.get("xadx-report.json")
    if report is None:
        raise EvidenceError("passing evidence is missing xadx-report.json")
    _reconcile_xadx_report(report, evidence, contract)


def _validate_semantics(
    members: dict[str, bytes],
    canonical_candidate: dict[str, Any] | None,
    canonical_path: Path | None,
) -> dict[str, Any]:
    evidence = _json_object(members["evidence.json"], "evidence.json")
    try:
        validate_schema(
            evidence, load_packaged_schema("evidence-v1.schema.json")
        )
    except (SchemaValidationError, OSError, json.JSONDecodeError) as error:
        raise EvidenceError(
            f"evidence.json does not satisfy the v1 schema: {error}"
        ) from error
    _iso_timestamp(evidence.get("started_at"), "started_at")
    _iso_timestamp(evidence.get("finished_at"), "finished_at", nullable=True)
    if evidence["status"] == "running" or evidence["finished_at"] is None:
        raise EvidenceError("evidence is not finalized")
    models = _json_value(members["models.json"], "models.json")
    tests = _json_value(members["tests.json"], "tests.json")
    profile = _json_object(members["profile.json"], "profile.json")
    if models != evidence["models"] or tests != evidence["tests"]:
        raise EvidenceError("evidence sidecars do not match evidence.json")
    if (
        profile.get("name") != evidence["profile"].get("name")
        or profile.get("description") != evidence["profile"].get("description")
    ):
        raise EvidenceError("profile.json does not match evidence.json")
    if evidence["status"] == "pass":
        if (
            evidence["finished_at"] is None
            or not evidence["models"]
            or not evidence["tests"]
            or any(model.get("status") != "pass" for model in evidence["models"])
            or any(test.get("status") != "pass" for test in evidence["tests"])
        ):
            raise EvidenceError("passing evidence has incomplete or failed results")
    for test in evidence["tests"]:
        if (
            not isinstance(test, dict)
            or test.get("status") not in {"pass", "fail", "unsupported"}
            or not isinstance(test.get("name"), str)
            or not isinstance(test.get("model"), str)
        ):
            raise EvidenceError("evidence contains an invalid test record")
        if test.get("status") != "pass" and not test.get("error"):
            raise EvidenceError("failed or unsupported test has no error")
        references = [test.get("log", ""), *test.get("artifacts", [])]
        for reference in references:
            if not reference:
                continue
            if not isinstance(reference, str):
                raise EvidenceError("test artifact reference is invalid")
            reference = _safe_name(reference)
            if reference not in members:
                raise EvidenceError(f"test artifact is missing: {reference}")
    reports = {}
    for name in (
        "core-version.json",
        "doctor.json",
        "probe.json",
        "self-test.json",
        "xadx-report.json",
    ):
        if name in members:
            reports[name] = _json_object(members[name], name)
    if "probe.json" in reports:
        probe_data = reports["probe.json"].get("data")
        if probe_data != evidence["observed_platform"]:
            raise EvidenceError(
                "evidence observed platform does not match probe.json"
            )
        _normalized_platform(probe_data)
    elif evidence["status"] == "pass":
        raise EvidenceError("passing evidence is missing probe.json")

    embedded = _json_object(
        members["candidate-manifest.json"], "candidate-manifest.json"
    )
    candidate_bound = bool(evidence["profile"].get("candidate_bound"))
    binding = evidence["subject"].get("candidate_binding", {})
    if bool(binding.get("present")) != candidate_bound:
        raise EvidenceError("candidate binding fields disagree")
    if candidate_bound:
        try:
            normalized_embedded = normalize_candidate_manifest(embedded)
        except CandidateError as error:
            raise EvidenceError(str(error)) from error
        embedded_bytes = canonical_candidate_bytes(normalized_embedded)
        if embedded_bytes != members["candidate-manifest.json"]:
            raise EvidenceError("embedded candidate manifest is not canonical")
        digest = hashlib.sha256(embedded_bytes).hexdigest()
        if binding.get("manifest_sha256") != digest:
            raise EvidenceError("candidate manifest digest does not match evidence")
        if canonical_candidate is not None:
            try:
                normalized_canonical = normalize_candidate_manifest(
                    canonical_candidate
                )
            except CandidateError as error:
                raise CandidateMismatchError(str(error)) from error
            if normalized_embedded != normalized_canonical:
                raise CandidateMismatchError(
                    "evidence candidate does not match canonical candidate"
                )
            if canonical_path is not None:
                _verify_available_candidate_wheels(
                    normalized_canonical, canonical_path.parent
                )
        contract = normalized_embedded.get("qualification_contract")
        if contract is None:
            raise EvidenceError(
                "candidate does not contain a qualification execution contract"
            )
        expected_profile = contract["profiles"].get(profile.get("name"))
        if profile != expected_profile:
            raise EvidenceError(
                "profile.json does not match the candidate qualification wheel"
            )
        _validate_complete_execution(evidence, profile, contract)
        versions = {
            name: values[0]["version"]
            for name, values in normalized_embedded["packages"].items()
        }
        subject = evidence["subject"]
        if (
            subject.get("version") != normalized_embedded["version"]
            or subject.get("source_revision")
            != normalized_embedded["source_revision"]
            or subject.get("packages") != versions
        ):
            raise CandidateMismatchError(
                "evidence subject does not match its candidate identity"
            )
    elif canonical_candidate is not None:
        raise CandidateMismatchError("candidate-bound intake rejects ad-hoc evidence")
    _reconcile_reports(members, evidence, reports, contract if candidate_bound else None)
    return evidence


def _verify_available_candidate_wheels(
    candidate: dict[str, Any], root: Path
) -> None:
    identities = [
        identity
        for values in candidate["packages"].values()
        for identity in values
    ]
    expected_names = {identity["filename"] for identity in identities}
    available_names = {
        path.name
        for name in expected_names
        for path in root.rglob(name)
        if path.is_file()
    }
    if not available_names:
        return
    for identity in identities:
        matches = [
            path
            for path in root.rglob(identity["filename"])
            if path.is_file() and not path.is_symlink()
        ]
        if len(matches) != 1 or sha256_file(matches[0]) != identity["sha256"]:
            raise CandidateMismatchError(
                f"canonical candidate wheel does not match: {identity['filename']}"
            )


def verify_evidence(
    source: Path, *, candidate_manifest: Path | None = None
) -> dict[str, Any]:
    source = Path(source)
    members = (
        _directory_members(source)
        if source.is_dir()
        else _archive_members(source)
    )
    _validate_layout(members)
    _parse_checksums(members)
    canonical = None
    if candidate_manifest is not None:
        try:
            canonical = _json_object(
                Path(candidate_manifest).read_bytes(), "canonical candidate manifest"
            )
        except OSError as error:
            raise CandidateMismatchError(
                "canonical candidate manifest cannot be read"
            ) from error
    evidence = _validate_semantics(
        members,
        canonical,
        None if candidate_manifest is None else Path(candidate_manifest).resolve(),
    )
    return {
        "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
        "status": "pass",
        "input": str(source.resolve()),
        "evidence_status": evidence["status"],
        "profile": evidence["profile"]["name"],
        "source_revision": evidence["subject"].get("source_revision", ""),
        "tests": len(evidence["tests"]),
        "models": len(evidence["models"]),
        "archive_sha256": sha256_file(source) if source.is_file() else None,
    }


def _canonical_candidate_from_path(
    candidate_manifest: Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if candidate_manifest is None:
        return None, None
    path = Path(candidate_manifest).resolve()
    try:
        value = _json_object(path.read_bytes(), "canonical candidate manifest")
        return normalize_candidate_manifest(value), path
    except (OSError, CandidateError) as error:
        raise CandidateMismatchError(
            "canonical candidate manifest cannot be read or normalized"
        ) from error


def _verify_external_checksum(source: Path) -> str:
    digest = sha256_file(source)
    sidecar = source.with_name(source.name + ".sha256")
    try:
        value = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(f"aggregate checksum sidecar is missing: {sidecar}") from error
    if value != f"{digest}  {source.name}\n":
        raise EvidenceError("aggregate checksum sidecar does not match the archive")
    return digest


def verify_aggregate(
    source: Path, *, candidate_manifest: Path | None = None
) -> dict[str, Any]:
    source = Path(source).resolve()
    if not source.is_file():
        raise EvidenceError("aggregate verification requires a ZIP archive")
    digest = _verify_external_checksum(source)
    members = _archive_members(source)
    if set(members) - {"SHA256SUMS", "qualification-index.json"} != {
        name for name in members if re.fullmatch(r"runs/[0-9a-f]{64}\.evidence\.zip", name)
    }:
        raise EvidenceError("qualification aggregate layout is invalid")
    _parse_checksums(members)
    index = _json_object(
        members.get("qualification-index.json", b""),
        "qualification-index.json",
    )
    try:
        validate_schema(
            index, load_packaged_schema("qualification-index-v1.schema.json")
        )
    except (SchemaValidationError, OSError, json.JSONDecodeError) as error:
        raise EvidenceError(
            f"qualification-index.json does not satisfy the v1 schema: {error}"
        ) from error
    canonical, canonical_path = _canonical_candidate_from_path(candidate_manifest)
    embedded_candidate = None
    run_paths = {run["bundle_path"] for run in index["runs"]}
    if (
        len(run_paths) != len(index["runs"])
        or run_paths != set(members) - {"SHA256SUMS", "qualification-index.json"}
    ):
        raise EvidenceError("aggregate index does not exclusively list its runs")
    for run in index["runs"]:
        bundle = members[run["bundle_path"]]
        if hashlib.sha256(bundle).hexdigest() != run["bundle_sha256"]:
            raise EvidenceError("aggregate run digest does not match its index")
        run_members = _archive_members(bundle)
        _validate_layout(run_members)
        _parse_checksums(run_members)
        candidate = normalize_candidate_manifest(
            _json_object(
                run_members["candidate-manifest.json"],
                "candidate-manifest.json",
            )
        )
        if embedded_candidate is None:
            embedded_candidate = candidate
        elif candidate != embedded_candidate:
            raise CandidateMismatchError(
                "aggregate runs do not share one release candidate"
            )
        evidence = _validate_semantics(
            run_members,
            canonical if canonical is not None else candidate,
            canonical_path,
        )
        if (
            evidence["status"] != run["status"]
            or evidence["profile"]["name"] != run["profile"]
            or evidence["started_at"] != run["started_at"]
            or evidence["finished_at"] != run["finished_at"]
            or evidence["subject"].get("source_revision") != run["source_revision"]
            or _normalized_platform(evidence["observed_platform"])
            != run["platform"]
        ):
            raise EvidenceError("aggregate run metadata does not match its evidence")
    if embedded_candidate is None:
        raise EvidenceError("aggregate contains no runs")
    candidate_digest = hashlib.sha256(
        canonical_candidate_bytes(embedded_candidate)
    ).hexdigest()
    if (
        index["candidate"]["manifest_sha256"] != candidate_digest
        or index["candidate"]["version"] != embedded_candidate["version"]
        or index["candidate"]["source_revision"]
        != embedded_candidate["source_revision"]
        or any(
            run["candidate_manifest_sha256"] != candidate_digest
            for run in index["runs"]
        )
    ):
        raise CandidateMismatchError("aggregate candidate identity is inconsistent")
    policy = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "name": index["policy"]["name"],
        "requirements": [
            {
                key: requirement[key]
                for key in (
                    "platform_os",
                    "profile",
                    "target_family",
                    "minimum_runs",
                )
            }
            for requirement in index["policy"]["requirements"]
        ],
    }
    _require_candidate_policy(embedded_candidate, policy, context="aggregate")
    if _evaluate_policy(index["runs"], policy) != index["policy"]:
        raise IntakePolicyError("aggregate policy attestation is inconsistent")
    return {
        "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
        "status": "pass",
        "input": str(source),
        "kind": "aggregate",
        "archive_sha256": digest,
        "runs": len(index["runs"]),
        "candidate_manifest_sha256": candidate_digest,
        "policy": index["policy"],
    }


def verify_archive(
    source: Path, *, candidate_manifest: Path | None = None
) -> dict[str, Any]:
    source = Path(source)
    if source.is_file():
        members = _archive_members(source)
        if "qualification-index.json" in members:
            return verify_aggregate(
                source, candidate_manifest=candidate_manifest
            )
    return verify_evidence(source, candidate_manifest=candidate_manifest)


def pack_evidence(source: Path, output: Path) -> dict[str, Any]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output or source in output.parents:
        raise EvidenceError("bundle output must be outside the evidence directory")
    verify_evidence(source)
    members = _directory_members(source)
    digest = _write_deterministic_zip(members, output)
    verify_evidence(output)
    return {
        "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
        "status": "pass",
        "input": str(source),
        "output": str(output),
        "sha256": digest,
        "files": len(members),
    }


def load_qualification_policy(value: str | Path) -> dict[str, Any]:
    path = Path(value)
    try:
        if path.is_file():
            policy = _json_object(path.read_bytes(), "qualification policy")
        else:
            name = str(value)
            if re.fullmatch(r"[a-z0-9-]+", name) is None:
                raise IntakePolicyError("qualification policy name is invalid")
            resource = resources.files("rocm_npu_qualification").joinpath(
                "policies", f"{name}.json"
            )
            policy = _json_object(
                resource.read_bytes(), f"qualification policy {name}"
            )
    except OSError as error:
        raise IntakePolicyError(f"qualification policy cannot be read: {value}") from error
    try:
        return normalize_policy(policy)
    except PolicyValidationError as error:
        raise IntakePolicyError(str(error)) from error


def _require_candidate_policy(
    candidate: dict[str, Any],
    policy: dict[str, Any],
    *,
    context: str,
) -> None:
    packaged_policy = candidate["qualification_contract"]["policies"].get(
        policy["name"]
    )
    if policy != packaged_policy:
        raise IntakePolicyError(
            f"{context} policy does not match the candidate qualification wheel"
        )


def _evaluate_policy(
    entries: list[dict[str, Any]], policy: dict[str, Any]
) -> dict[str, Any]:
    results = []
    for requirement in policy["requirements"]:
        matches = [
            entry["bundle_sha256"]
            for entry in entries
            if entry["platform"].get("os") == requirement["platform_os"]
            and entry["profile"] == requirement["profile"]
            and entry["platform"].get("target_family")
            == requirement["target_family"]
            and entry["status"] == "pass"
        ]
        satisfied = len(matches) >= requirement["minimum_runs"]
        results.append(
            {
                **requirement,
                "matched_runs": sorted(matches),
                "satisfied": satisfied,
            }
        )
    if not all(result["satisfied"] for result in results):
        missing = [
            {
                key: result[key]
                for key in (
                    "platform_os",
                    "profile",
                    "target_family",
                    "minimum_runs",
                )
            }
            for result in results
            if not result["satisfied"]
        ]
        raise IntakePolicyError(
            f"qualification evidence does not satisfy policy {policy['name']}: {missing}"
        )
    return {
        "name": policy["name"],
        "satisfied": True,
        "requirements": results,
    }


def intake_evidence(
    input_directory: Path,
    candidate_manifest: Path,
    output_directory: Path,
    bundle_output: Path,
    *,
    require_status: str | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    input_directory = Path(input_directory).resolve()
    output_directory = Path(output_directory).resolve()
    bundle_output = Path(bundle_output).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise EvidenceError("intake output directory is not empty")
    if output_directory == bundle_output or output_directory in bundle_output.parents:
        raise EvidenceError("aggregate bundle must be outside the intake directory")
    bundles = sorted(input_directory.glob("*.evidence.zip"))
    if not bundles:
        raise EvidenceError("intake input directory has no evidence bundles")
    candidate = _json_object(
        Path(candidate_manifest).read_bytes(), "canonical candidate manifest"
    )
    normalized_candidate = normalize_candidate_manifest(candidate)
    _require_candidate_policy(normalized_candidate, policy, context="intake")
    candidate_digest = hashlib.sha256(
        canonical_candidate_bytes(normalized_candidate)
    ).hexdigest()
    entries = []
    digests: set[str] = set()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=output_directory.name + ".",
            dir=output_directory.parent,
        )
    )
    bundle_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=bundle_output.name + ".",
        suffix=".tmp",
        dir=bundle_output.parent,
        delete=False,
    ) as stream:
        staged_bundle = Path(stream.name)
    staged_bundle.unlink()
    try:
        runs = stage / "runs"
        runs.mkdir()
        for bundle in bundles:
            summary = verify_evidence(
                bundle, candidate_manifest=Path(candidate_manifest)
            )
            if (
                require_status is not None
                and summary["evidence_status"] != require_status
            ):
                raise IntakePolicyError(
                    f"{bundle.name} status is {summary['evidence_status']}; "
                    f"required {require_status}"
                )
            digest = sha256_file(bundle)
            if digest in digests:
                raise EvidenceError(f"duplicate evidence bundle: {bundle.name}")
            digests.add(digest)
            members = _archive_members(bundle)
            evidence = _json_object(members["evidence.json"], "evidence.json")
            target_name = f"{digest}.evidence.zip"
            shutil.copyfile(bundle, runs / target_name)
            platform = _normalized_platform(evidence.get("observed_platform"))
            entries.append(
                {
                    "bundle_sha256": digest,
                    "bundle_path": f"runs/{target_name}",
                    "source_filename": bundle.name,
                    "status": evidence["status"],
                    "profile": evidence["profile"]["name"],
                    "started_at": evidence["started_at"],
                    "finished_at": evidence["finished_at"],
                    "source_revision": evidence["subject"].get(
                        "source_revision", ""
                    ),
                    "candidate_manifest_sha256": candidate_digest,
                    "platform": platform,
                }
            )
        entries.sort(key=lambda entry: entry["bundle_sha256"])
        policy_result = _evaluate_policy(entries, policy)
        index = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "product": "rocm-npu-qualification-index",
            "candidate": {
                "version": normalized_candidate["version"],
                "source_revision": normalized_candidate["source_revision"],
                "manifest_sha256": candidate_digest,
            },
            "policy": policy_result,
            "runs": entries,
        }
        try:
            validate_schema(
                index,
                load_packaged_schema("qualification-index-v1.schema.json"),
            )
        except (SchemaValidationError, OSError, json.JSONDecodeError) as error:
            raise EvidenceError(f"generated intake index is invalid: {error}") from error
        (stage / "qualification-index.json").write_bytes(_json_bytes(index))
        write_sha256sums(stage)
        aggregate_members = _directory_members(stage)
        _parse_checksums(aggregate_members)
        digest = _write_deterministic_zip(aggregate_members, staged_bundle)
        sidecar = bundle_output.with_name(bundle_output.name + ".sha256")
        with tempfile.NamedTemporaryFile(
            prefix=sidecar.name + ".",
            suffix=".tmp",
            dir=sidecar.parent,
            delete=False,
        ) as stream:
            staged_sidecar = Path(stream.name)
            stream.write(f"{digest}  {bundle_output.name}\n".encode("ascii"))
        if output_directory.exists():
            output_directory.rmdir()
        stage.replace(output_directory)
        try:
            staged_bundle.replace(bundle_output)
            staged_sidecar.replace(sidecar)
            verify_aggregate(
                bundle_output, candidate_manifest=Path(candidate_manifest)
            )
        except Exception:
            shutil.rmtree(output_directory, ignore_errors=True)
            bundle_output.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        staged_bundle.unlink(missing_ok=True)
        if "staged_sidecar" in locals():
            staged_sidecar.unlink(missing_ok=True)
    return {
        "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
        "status": "pass",
        "output": str(output_directory),
        "bundle_output": str(bundle_output),
        "sha256": digest,
        "checksum": str(bundle_output.with_name(bundle_output.name + ".sha256")),
        "runs": len(entries),
        "candidate_manifest_sha256": candidate_digest,
        "policy": policy_result,
    }
