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
    NATIVE_CHECKS,
    POLICY_SCHEMA_VERSION,
    POWER_MODE_ROUNDTRIP,
    POWER_MODE_ROUNDTRIP_ARTIFACTS,
    POWER_MODE_ROUNDTRIP_MODE_ARTIFACTS,
    SHARED_CODE_MEMORY_RECLAIM,
    XADX_REPORT_SCHEMA_VERSION,
    XADX_SUBBUFFER_BINDING_TEST,
    XADX_UNIQUE_XCLBIN_CONTEXT_SOAK,
    XADX_UNIQUE_XCLBIN_CONTEXT_SOAK_TEST,
)
from .evidence import junit_bytes
from .hashing import sha256_file, write_sha256sums
from .numeric_runtime import NUMERIC_RUNTIME
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
    # Optional human-readable HRX overlay (diagnostic sweeps write it). Hashed by
    # SHA256SUMS like every other member, but not schema-validated and carries no
    # intake semantics.
    "report.md",
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
            if stat.S_ISLNK(mode) or (
                mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))
            ):
                raise EvidenceError(f"archive contains special member: {name}")
            if info.is_dir():
                raise EvidenceError(
                    f"archive contains unexpected directory entry: {name}"
                )
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
        if name not in ROOT_FILES and top not in {"artifacts", "bench", "logs"}:
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


def _is_allowed_unsupported_test(
    evidence: dict[str, Any], test: dict[str, Any]
) -> bool:
    try:
        platform = _normalized_platform(evidence.get("observed_platform"))
    except EvidenceError:
        return False
    return (
        test.get("status") == "unsupported"
        and test.get("kind") == "native"
        and test.get("model") == "native-core"
        and test.get("name") == POWER_MODE_ROUNDTRIP
        and platform.get("os") == "linux"
    )


def _test_succeeded_or_was_skipped(
    evidence: dict[str, Any], test: dict[str, Any]
) -> bool:
    return test.get("status") == "pass" or _is_allowed_unsupported_test(evidence, test)


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
    for expected_model, observed_model in zip(
        profile["models"], evidence["models"], strict=True
    ):
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
        observed_identity = {key: observed_model.get(key) for key in expected_identity}
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
        if observed_identity != expected_identity or observed_files != expected_files:
            raise EvidenceError(
                f"passing evidence model identity does not match profile: "
                f"{expected_model['tag']}"
            )

    expected: dict[tuple[str, str, str], dict[str, int]] = {}
    for name in contract["native_checks"]:
        expected[("native", "native-core", name)] = {}
    for name in contract["xadx_cases"]:
        expected[("xadx", "fastflowlm-production-kernels", name)] = {}
    expected[
        (
            "xadx",
            "fastflowlm-production-kernels",
            XADX_UNIQUE_XCLBIN_CONTEXT_SOAK_TEST,
        )
    ] = {}
    expected[
        (
            "xadx",
            "fastflowlm-production-kernels",
            XADX_SUBBUFFER_BINDING_TEST,
        )
    ] = {}
    for model in profile["models"]:
        if model["xclbins"]:
            expected[("xclbin", model["tag"], "production-xclbin-inventory")] = {
                "files_verified": len(model["xclbins"])
            }
        for test in model["tests"]:
            if test["kind"] == "serve":
                expected[("serve", model["tag"], test["name"])] = {}
                continue
            if test["kind"] == "bench":
                expected[("bench", model["tag"], test["name"])] = {
                    "turn_count": test.get("iterations", 1),
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
                raise EvidenceError(f"passing evidence has invalid {field} for {key}")

    # Every coherence-capable case receives the declared readback depth. The
    # projected rows carry that attribution, so the total cannot be inferred
    # from one global readback count.
    execution = _xadx_execution(contract, profile["name"])
    resident = execution["lifecycle_rounds"] + execution["resident_rounds"]
    budget = len(contract["xadx_cases"]) * resident + sum(
        observed[key].get("metrics", {}).get("readback_rounds", 0)
        for key in observed
        if key[0] == "xadx"
        and key[2]
        not in {
            XADX_UNIQUE_XCLBIN_CONTEXT_SOAK_TEST,
            XADX_SUBBUFFER_BINDING_TEST,
        }
    )
    performed = sum(
        observed[key].get("rounds", 0)
        for key in observed
        if key[0] == "xadx"
        and key[2]
        not in {
            XADX_UNIQUE_XCLBIN_CONTEXT_SOAK_TEST,
            XADX_SUBBUFFER_BINDING_TEST,
        }
    )
    if performed != budget:
        raise EvidenceError(
            "passing evidence does not account for the XADX execution budget: "
            f"performed={performed}, required={budget}"
        )


def _native_report_name(command: str) -> str:
    # The version report keeps its historical root filename.
    return "core-version.json" if command == "version" else f"{command}.json"


def _validate_core_report(
    report: dict[str, Any],
    evidence: dict[str, Any],
    *,
    command: str,
    name: str,
) -> None:
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


def _powermode_report(
    members: dict[str, bytes], evidence: dict[str, Any], name: str
) -> dict[str, Any]:
    report = _json_object(members[name], name)
    _validate_core_report(report, evidence, command="powermode", name=name)
    return report


def _reported_power_modes(report: dict[str, Any], name: str) -> list[str]:
    data = report.get("data")
    devices = data.get("devices") if isinstance(data, dict) else None
    if not isinstance(data, dict) or not isinstance(devices, list):
        raise EvidenceError(f"{name} has no power-mode device data")
    modes = []
    for device in devices:
        if not isinstance(device, dict) or not isinstance(
            device.get("power_mode"), str
        ):
            raise EvidenceError(f"{name} contains a device without power_mode")
        modes.append(device["power_mode"])
    if not modes:
        raise EvidenceError(f"{name} contains no reported power modes")
    if data.get("reported_device_count") != len(modes):
        raise EvidenceError(f"{name} reported device count does not match devices")
    return modes


def _require_uniform_power_mode(
    report: dict[str, Any],
    name: str,
    expected_mode: str,
    *,
    expected_count: int | None = None,
) -> None:
    if report.get("status") != "pass":
        raise EvidenceError(f"{name} did not pass")
    modes = _reported_power_modes(report, name)
    if expected_count is not None and len(modes) != expected_count:
        raise EvidenceError(f"{name} did not report every baseline device")
    if any(mode != expected_mode for mode in modes):
        raise EvidenceError(f"{name} reported an unexpected power mode")


def _reconcile_powermode_roundtrip(
    members: dict[str, bytes],
    evidence: dict[str, Any],
    test: dict[str, Any],
) -> None:
    if _is_allowed_unsupported_test(evidence, test):
        before_name = POWER_MODE_ROUNDTRIP_ARTIFACTS[0]
        if (
            test.get("returncode") != 0
            or test.get("artifacts") != [before_name]
            or not str(test.get("error", "")).startswith(
                "powermode roundtrip is skipped on Linux"
            )
        ):
            raise EvidenceError("powermode roundtrip skip is not a valid Linux skip")
        before = _powermode_report(members, evidence, before_name)
        baseline_modes = _reported_power_modes(before, before_name)
        if len(set(baseline_modes)) != 1:
            raise EvidenceError("powermode roundtrip baseline was not uniform")
        baseline = baseline_modes[0]
        before_data = before.get("data")
        detected = (
            before_data.get("detected_device_count")
            if isinstance(before_data, dict)
            else None
        )
        reported = (
            before_data.get("reported_device_count")
            if isinstance(before_data, dict)
            else None
        )
        if not isinstance(detected, int) or detected < 1 or detected != reported:
            raise EvidenceError("powermode roundtrip baseline device count is invalid")
        _require_uniform_power_mode(
            before, before_name, baseline, expected_count=detected
        )
        expected_metrics = {
            "baseline_power_mode": baseline,
            "requested_power_modes": [
                mode for mode, _, _ in POWER_MODE_ROUNDTRIP_MODE_ARTIFACTS
            ],
            "confirmed_power_modes": [],
            "detected_device_count": detected,
            "reported_device_count": reported,
        }
        if test.get("metrics") != expected_metrics:
            raise EvidenceError(
                "powermode roundtrip skip metrics do not match artifacts"
            )
        return

    if (
        test.get("status") != "pass"
        or test.get("returncode") != 0
        or test.get("artifacts") != list(POWER_MODE_ROUNDTRIP_ARTIFACTS)
    ):
        raise EvidenceError("powermode roundtrip does not match its native test record")

    reports = {
        name: _powermode_report(members, evidence, name)
        for name in POWER_MODE_ROUNDTRIP_ARTIFACTS
    }
    before = reports[POWER_MODE_ROUNDTRIP_ARTIFACTS[0]]
    baseline_modes = _reported_power_modes(before, POWER_MODE_ROUNDTRIP_ARTIFACTS[0])
    if len(set(baseline_modes)) != 1:
        raise EvidenceError("powermode roundtrip baseline was not uniform")
    baseline = baseline_modes[0]

    before_data = before.get("data")
    detected = (
        before_data.get("detected_device_count")
        if isinstance(before_data, dict)
        else None
    )
    reported = (
        before_data.get("reported_device_count")
        if isinstance(before_data, dict)
        else None
    )
    if not isinstance(detected, int) or detected < 1 or detected != reported:
        raise EvidenceError("powermode roundtrip baseline device count is invalid")

    _require_uniform_power_mode(
        before, POWER_MODE_ROUNDTRIP_ARTIFACTS[0], baseline, expected_count=detected
    )

    requested_modes = []
    confirmed_modes = []
    for expected_mode, set_name, after_name in POWER_MODE_ROUNDTRIP_MODE_ARTIFACTS:
        set_report = reports[set_name]
        data = set_report.get("data")
        if not isinstance(data, dict):
            raise EvidenceError(f"{set_name} has no power-mode set data")
        requested = data.get("requested_power_mode")
        if requested != expected_mode:
            raise EvidenceError("powermode roundtrip requested an unexpected mode")
        requested_modes.append(requested)
        _require_uniform_power_mode(
            set_report,
            set_name,
            expected_mode,
            expected_count=detected,
        )
        if data.get("updated_device_count") != detected:
            raise EvidenceError(
                f"powermode set for {expected_mode} did not update every detected device"
            )
        _require_uniform_power_mode(
            reports[after_name],
            after_name,
            expected_mode,
            expected_count=detected,
        )
        confirmed_modes.append(expected_mode)

    if not {"performance", "turbo"}.issubset(requested_modes):
        raise EvidenceError(
            "powermode roundtrip did not exercise performance and turbo"
        )

    restore = reports[POWER_MODE_ROUNDTRIP_ARTIFACTS[-2]]
    final = reports[POWER_MODE_ROUNDTRIP_ARTIFACTS[-1]]
    restore_data = restore.get("data")
    if (
        not isinstance(restore_data, dict)
        or restore_data.get("requested_power_mode") != baseline
    ):
        raise EvidenceError("powermode roundtrip restore requested the wrong mode")
    _require_uniform_power_mode(
        restore,
        POWER_MODE_ROUNDTRIP_ARTIFACTS[-2],
        baseline,
        expected_count=detected,
    )
    _require_uniform_power_mode(
        final,
        POWER_MODE_ROUNDTRIP_ARTIFACTS[-1],
        baseline,
        expected_count=detected,
    )
    if restore_data.get("updated_device_count") != detected:
        raise EvidenceError("powermode restore did not update every detected device")

    expected_metrics = {
        "baseline_power_mode": baseline,
        "requested_power_modes": requested_modes,
        "confirmed_power_modes": confirmed_modes,
        "restored_power_mode": baseline,
        "detected_device_count": detected,
        "reported_device_count": reported,
    }
    if test.get("metrics") != expected_metrics:
        raise EvidenceError("powermode roundtrip metrics do not match artifacts")


def _reconcile_native_reports(
    members: dict[str, bytes],
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
        if command == POWER_MODE_ROUNDTRIP:
            _reconcile_powermode_roundtrip(members, evidence, tests[command])
            continue
        if command == SHARED_CODE_MEMORY_RECLAIM:
            artifact = "artifacts/native/shared-code-memory-reclaim.json"
            report = _json_object(
                members.get(artifact, b""), "shared-code-memory-reclaim report"
            )
            bounded = report.get("bounded_shared_code_memory")
            allocated = report.get("allocated_bytes")
            budget = report.get("budget_bytes", 0)
            exhaustion = report.get("exhaustion_observed")
            reclaimed = report.get("reclaimed_bytes")
            post_reclaim = report.get("post_reclaim_allocation_bytes")
            valid = (
                report.get("schema_version") == 1
                and report.get("status") == "pass"
                and isinstance(bounded, bool)
                and type(allocated) is int
                and allocated >= 0
                and type(budget) is int
                and budget >= 0
                and isinstance(exhaustion, bool)
                and type(reclaimed) is int
                and reclaimed >= 0
                and type(post_reclaim) is int
                and post_reclaim >= 0
                and (
                    (
                        bounded
                        and budget > 0
                        and allocated >= (budget // 4) * 3
                        and exhaustion
                        and reclaimed >= post_reclaim > 0
                    )
                    or (
                        not bounded
                        and budget == 0
                        and allocated > 0
                        and not exhaustion
                        and reclaimed == 0
                        and post_reclaim == 0
                    )
                )
            )
            test = tests[command]
            expected_metrics = {
                "bounded_shared_code_memory": report.get(
                    "bounded_shared_code_memory", False
                ),
                "allocated_bytes": report.get("allocated_bytes", 0),
                "budget_bytes": report.get("budget_bytes", 0),
                "exhaustion_observed": report.get("exhaustion_observed", False),
                "reclaimed_bytes": report.get("reclaimed_bytes", 0),
                "post_reclaim_allocation_bytes": report.get(
                    "post_reclaim_allocation_bytes", 0
                ),
            }
            if (
                not valid
                or test.get("status") != "pass"
                or test.get("returncode") != 0
                or test.get("artifacts") != [artifact]
                or test.get("metrics") != expected_metrics
                or test.get("log") != "logs/tests/shared-code-memory-reclaim.log"
            ):
                raise EvidenceError(
                    "shared-code-memory-reclaim report does not match its "
                    "native test record"
                )
            continue
        name = _native_report_name(command)
        report = reports.get(name)
        if report is None:
            raise EvidenceError(f"passing evidence is missing {name}")
        _validate_core_report(report, evidence, command=command, name=name)
        test = tests[command]
        report_passed = report["status"] == "pass"
        if test.get("metrics") != {"report_status": report["status"]} or (
            test.get("status") == "pass"
        ) != (report_passed and test.get("returncode") == 0):
            raise EvidenceError(f"{name} does not match its native test record")


def _xadx_execution(contract: dict[str, Any], profile_name: str) -> dict[str, int]:
    """The round policy the contract holds for the profile that ran."""
    execution = contract["xadx_execution"].get(profile_name)
    if execution is None:
        raise EvidenceError(
            "the qualification contract states no XADX round policy for profile "
            f"{profile_name!r}"
        )
    return execution


def _reconcile_xadx_report(
    report: dict[str, Any],
    evidence: dict[str, Any],
    contract: dict[str, Any] | None,
) -> None:
    try:
        validate_schema(report, load_packaged_schema("xadx-report-v1.schema.json"))
    except (SchemaValidationError, OSError, json.JSONDecodeError) as error:
        raise EvidenceError(
            f"xadx-report.json does not satisfy the v1 schema: {error}"
        ) from error
    if report.get("schema_version") != XADX_REPORT_SCHEMA_VERSION:
        raise EvidenceError("xadx-report.json schema version is invalid")
    tests = [
        test
        for test in evidence["tests"]
        if test.get("kind") == "xadx"
        and test.get("model") == "fastflowlm-production-kernels"
    ]
    if report["report_kind"] == "package-validation":
        row = report["case_results"][0]
        expected_metrics = {
            "architecture": report["architecture"],
            "lifecycle_rounds": 0,
            "resident_rounds": 0,
            "readback_rounds": 0,
            "validated_bytes": row["validated_bytes"],
            "mismatch_count": row["mismatch_count"],
            "first_mismatch": row["first_mismatch"],
        }
        expected_failure = [
            {"case": row["name"], **failure} for failure in row["failures"]
        ]
        if (
            len(tests) != 1
            or row["name"] != "package-validation"
            or report["failures"] != expected_failure
            or tests[0].get("name") != row["name"]
            or tests[0].get("status") != "fail"
            or tests[0].get("rounds") != 0
            or tests[0].get("duration_seconds") != row["duration_seconds"]
            or tests[0].get("metrics") != expected_metrics
            or tests[0].get("log") != "xadx-report.json"
            or tests[0].get("artifacts") != ["xadx-report.json"]
        ):
            raise EvidenceError(
                "xadx-report.json package-validation result does not match tests.json"
            )
        return
    expected_names = (
        contract["xadx_cases"]
        if contract is not None
        else [
            test.get("name")
            for test in tests
            if test.get("name")
            not in {
                XADX_UNIQUE_XCLBIN_CONTEXT_SOAK_TEST,
                XADX_SUBBUFFER_BINDING_TEST,
            }
        ]
    )
    soak_tests = [
        test
        for test in tests
        if test.get("name") == XADX_UNIQUE_XCLBIN_CONTEXT_SOAK_TEST
    ]
    recipe_tests = [
        test
        for test in tests
        if test.get("name")
        not in {
            XADX_UNIQUE_XCLBIN_CONTEXT_SOAK_TEST,
            XADX_SUBBUFFER_BINDING_TEST,
        }
    ]
    if report.get("case_names") != expected_names:
        raise EvidenceError("xadx-report.json case inventory does not match tests.json")
    status = report.get("status")
    if status not in {"pass", "fail", "unsupported"}:
        raise EvidenceError("xadx-report.json status is invalid")
    if status != "unsupported" and report.get("numeric_runtime") != NUMERIC_RUNTIME:
        raise EvidenceError(
            "xadx-report.json numeric runtime does not match the qualification contract"
        )
    rounds = {
        key: report.get(key)
        for key in ("lifecycle_rounds", "resident_rounds", "readback_rounds")
    }
    if status == "unsupported":
        rounds = {key: 0 for key in rounds}
    if any(not isinstance(value, int) or value < 0 for value in rounds.values()):
        raise EvidenceError("xadx-report.json round counts are invalid")
    if (
        status != "unsupported"
        and contract is not None
        and rounds != _xadx_execution(contract, evidence["profile"]["name"])
    ):
        raise EvidenceError(
            "xadx-report.json round counts do not match the qualification contract"
        )
    if len(soak_tests) != 1:
        raise EvidenceError("tests.json is missing unique-xclbin-context-soak")
    soak = report.get("unique_xclbin_context_soak")
    soak_test = soak_tests[0]
    if not isinstance(soak, dict):
        raise EvidenceError("xadx-report.json unique-xclbin-context-soak is missing")
    if (
        soak_test.get("status") != soak.get("status")
        or soak_test.get("rounds") != soak.get("images")
        or soak_test.get("duration_seconds") != soak.get("duration_seconds")
        or soak_test.get("metrics")
        != {
            "architecture": report.get("architecture", "unknown"),
            "unique_xclbin_images": soak.get("images"),
        }
        or soak_test.get("log") != "xadx-report.json"
        or soak_test.get("artifacts") != ["xadx-report.json"]
    ):
        raise EvidenceError(
            "xadx-report.json unique-xclbin-context-soak does not match tests.json"
        )
    binding_tests = [
        test for test in tests if test.get("name") == XADX_SUBBUFFER_BINDING_TEST
    ]
    if len(binding_tests) != 1:
        raise EvidenceError("tests.json is missing subbuffer-binding")
    binding = report.get("subbuffer_binding")
    binding_test = binding_tests[0]
    if not isinstance(binding, dict):
        raise EvidenceError("xadx-report.json subbuffer-binding is missing")
    if (
        binding_test.get("status") != binding.get("status")
        or binding_test.get("duration_seconds") != binding.get("duration_seconds")
        or binding_test.get("rounds")
        != (0 if binding.get("status") == "unsupported" else 1)
        or binding_test.get("metrics")
        != {
            "architecture": report.get("architecture", "unknown"),
            "returncode": binding.get("returncode"),
        }
        or binding_test.get("log") != "xadx-report.json"
        or binding_test.get("artifacts") != ["xadx-report.json"]
    ):
        raise EvidenceError(
            "xadx-report.json subbuffer-binding does not match tests.json"
        )
    if status == "pass" and (
        soak.get("status") != "pass"
        or soak.get("images") != XADX_UNIQUE_XCLBIN_CONTEXT_SOAK
    ):
        raise EvidenceError(
            "unique-xclbin-context-soak did not load 33 distinct images"
        )
    if status == "unsupported" and soak.get("status") != "unsupported":
        raise EvidenceError(
            "unsupported XADX report must skip unique-xclbin-context-soak"
        )
    if len(recipe_tests) != len(expected_names):
        raise EvidenceError("xadx-report.json does not match the XADX test inventory")
    case_results = report.get("case_results")
    if (
        not isinstance(case_results, list)
        or [row.get("name") for row in case_results] != expected_names
    ):
        raise EvidenceError("xadx-report.json per-case results are invalid")
    result_by_name = {row["name"]: row for row in case_results}
    attribution = (
        {row["name"]: row["readbacks"] for row in report["case_results"]}
        if status == "unsupported"
        else report.get("readback_by_case")
    )
    if not isinstance(attribution, dict) or any(
        not isinstance(value, int) or value < 0 for value in attribution.values()
    ):
        raise EvidenceError("xadx-report.json readback attribution is invalid")
    if any(value > rounds["readback_rounds"] for value in attribution.values()):
        raise EvidenceError("xadx-report.json overstates per-case readback rounds")
    if set(attribution) - set(expected_names):
        raise EvidenceError(
            "xadx-report.json attributes readback rounds to unknown cases"
        )
    case_contracts = (
        contract.get("xadx_case_contracts") if contract is not None else None
    )
    if status != "unsupported" and case_contracts is not None:
        expected_attribution = {
            name: rounds["readback_rounds"]
            for name in expected_names
            if case_contracts[name]["readback_capable"]
        }
        if (status == "pass" and attribution != expected_attribution) or (
            status == "fail"
            and (
                set(attribution) - set(expected_attribution)
                or any(
                    value > expected_attribution[name]
                    for name, value in attribution.items()
                )
                or any(
                    result_by_name[name].get("status") == "pass"
                    and attribution.get(name) != expected
                    for name, expected in expected_attribution.items()
                )
            )
        ):
            raise EvidenceError(
                "xadx-report.json readback attribution does not match the "
                "candidate case contracts"
            )
        if report.get("materialized_buffers") != contract["xadx_materializations"]:
            raise EvidenceError(
                "xadx-report.json materialization inventory does not match the "
                "candidate"
            )
    row_statuses = [row.get("status") for row in case_results]
    if (status == "pass" and any(value != "pass" for value in row_statuses)) or (
        status == "fail" and "fail" not in row_statuses
    ):
        raise EvidenceError("xadx-report.json aggregate and case statuses disagree")
    if contract is not None:
        provenance = contract["xadx_provenance"]
        if (
            report.get("recipe_sha256") != provenance["recipe_sha256"]
            or report.get("generator_version") != provenance["generator_version"]
            or report.get("executables") != provenance["executables"]
        ):
            raise EvidenceError(
                "xadx-report.json execution provenance does not match the candidate"
            )
    if status == "unsupported":
        if report["failures"] or any(
            row["status"] != "unsupported"
            or row["dispatches"] != 0
            or row["readbacks"] != 0
            or row["validated_bytes"] != 0
            or row["duration_seconds"] != 0
            or row["mismatch_count"] != 0
            or row["first_mismatch"] is not None
            or row["failures"]
            for row in case_results
        ):
            raise EvidenceError("xadx-report.json unsupported results are invalid")
    for test, name, measured in zip(
        recipe_tests, expected_names, case_results, strict=True
    ):
        readback = attribution.get(name, 0)
        expected_dispatches = None
        expected_validated_bytes = None
        if (
            case_contracts is not None
            and isinstance(measured, dict)
            and measured.get("status") == "pass"
        ):
            expected_dispatches = (
                rounds["lifecycle_rounds"] + rounds["resident_rounds"] + readback
            )
            expected_validated_bytes = (
                expected_dispatches
                * case_contracts[name]["validated_bytes_per_dispatch"]
            )
        if (
            not isinstance(measured, dict)
            or set(measured)
            != {
                "name",
                "status",
                "dispatches",
                "readbacks",
                "validated_bytes",
                "duration_seconds",
                "mismatch_count",
                "first_mismatch",
                "failures",
            }
            or measured["status"] not in {"pass", "fail", "unsupported"}
            or not isinstance(measured.get("dispatches"), int)
            or measured["dispatches"] < 0
            or measured.get("readbacks") != readback
            or (
                expected_dispatches is not None
                and measured["dispatches"] != expected_dispatches
            )
            or not isinstance(measured.get("validated_bytes"), int)
            or measured["validated_bytes"] < 0
            or (
                expected_validated_bytes is not None
                and measured["validated_bytes"] != expected_validated_bytes
            )
            or not isinstance(measured.get("duration_seconds"), (int, float))
            or measured["duration_seconds"] < 0
            or not isinstance(measured.get("mismatch_count"), int)
            or measured["mismatch_count"] < 0
            or (measured["status"] == "pass" and measured["failures"])
            or (measured["status"] == "fail" and not measured["failures"])
            or (
                measured["status"] == "pass"
                and measured["dispatches"] > 0
                and (
                    measured["validated_bytes"] <= 0
                    or measured["duration_seconds"] <= 0
                )
            )
            or (measured["status"] == "unsupported" and measured["failures"])
        ):
            raise EvidenceError(
                f"xadx-report.json has invalid per-case metrics: {name}"
            )
        expected_metrics = {
            "architecture": report.get("architecture", "unknown"),
            "lifecycle_rounds": rounds["lifecycle_rounds"],
            "resident_rounds": rounds["resident_rounds"],
            "readback_rounds": readback,
            "validated_bytes": measured["validated_bytes"],
            "mismatch_count": measured["mismatch_count"],
            "first_mismatch": measured["first_mismatch"],
        }
        if (
            test.get("name") != name
            or test.get("status") != measured["status"]
            or test.get("rounds") != measured["dispatches"]
            or test.get("duration_seconds") != measured["duration_seconds"]
            or test.get("metrics") != expected_metrics
            or test.get("log") != "xadx-report.json"
            or test.get("artifacts") != ["xadx-report.json"]
        ):
            raise EvidenceError(
                f"xadx-report.json does not match its test record: {name}"
            )
    flattened_failures = [
        {"case": row["name"], **failure}
        for row in case_results
        for failure in row["failures"]
    ]

    def canonical_failure(failure):
        return (
            failure["case"],
            failure["phase"],
            -1 if failure["round"] is None else failure["round"],
            failure["message"],
        )

    if sorted(report["failures"], key=canonical_failure) != sorted(
        flattened_failures, key=canonical_failure
    ):
        raise EvidenceError("xadx-report.json top-level and per-case failures disagree")


def _reconcile_reports(
    members: dict[str, bytes],
    evidence: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    contract: dict[str, Any] | None,
) -> None:
    if members["junit.xml"] != junit_bytes(evidence["tests"]):
        raise EvidenceError("junit.xml does not match tests.json")
    report = reports.get("xadx-report.json")
    if report is not None:
        _reconcile_xadx_report(report, evidence, contract)
    elif evidence["status"] == "pass":
        raise EvidenceError("passing evidence is missing xadx-report.json")
    if evidence["status"] != "pass":
        return
    native_checks = (
        contract["native_checks"] if contract is not None else list(NATIVE_CHECKS)
    )
    _reconcile_native_reports(members, reports, evidence, native_checks)


def _validate_semantics(
    members: dict[str, bytes],
    canonical_candidate: dict[str, Any] | None,
    canonical_path: Path | None,
) -> dict[str, Any]:
    evidence = _json_object(members["evidence.json"], "evidence.json")
    try:
        validate_schema(evidence, load_packaged_schema("evidence-v1.schema.json"))
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
    if profile.get("name") != evidence["profile"].get("name") or profile.get(
        "description"
    ) != evidence["profile"].get("description"):
        raise EvidenceError("profile.json does not match evidence.json")
    if evidence["status"] == "pass":
        if (
            evidence["finished_at"] is None
            or not evidence["models"]
            or not evidence["tests"]
            or any(model.get("status") != "pass" for model in evidence["models"])
            or any(
                not _test_succeeded_or_was_skipped(evidence, test)
                for test in evidence["tests"]
            )
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
            raise EvidenceError("evidence observed platform does not match probe.json")
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
    observed_numeric = evidence["subject"].get("numeric_runtime", {})
    if {
        distribution: identity.get("version")
        for distribution, identity in observed_numeric.items()
        if isinstance(identity, dict)
    } != NUMERIC_RUNTIME:
        raise EvidenceError(
            "evidence numeric runtime does not match the qualification contract"
        )
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
                normalized_canonical = normalize_candidate_manifest(canonical_candidate)
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
            or subject.get("source_revision") != normalized_embedded["source_revision"]
            or subject.get("packages") != versions
            or {
                distribution: identity["version"]
                for distribution, identity in subject["numeric_runtime"].items()
            }
            != normalized_embedded["numeric_runtime"]
        ):
            raise CandidateMismatchError(
                "evidence subject does not match its candidate identity"
            )
    elif canonical_candidate is not None:
        raise CandidateMismatchError("candidate-bound intake rejects ad-hoc evidence")
    _reconcile_reports(
        members, evidence, reports, contract if candidate_bound else None
    )
    return evidence


def _verify_available_candidate_wheels(candidate: dict[str, Any], root: Path) -> None:
    identities = [
        identity for values in candidate["packages"].values() for identity in values
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
        _directory_members(source) if source.is_dir() else _archive_members(source)
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
        raise EvidenceError(
            f"aggregate checksum sidecar is missing: {sidecar}"
        ) from error
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
        name
        for name in members
        if re.fullmatch(r"runs/[0-9a-f]{64}\.evidence\.zip", name)
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
    if len(run_paths) != len(index["runs"]) or run_paths != set(members) - {
        "SHA256SUMS",
        "qualification-index.json",
    }:
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
            or _normalized_platform(evidence["observed_platform"]) != run["platform"]
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
            return verify_aggregate(source, candidate_manifest=candidate_manifest)
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
            resource = resources.files("fastflow_sdk_qualification").joinpath(
                "policies", f"{name}.json"
            )
            policy = _json_object(resource.read_bytes(), f"qualification policy {name}")
    except OSError as error:
        raise IntakePolicyError(
            f"qualification policy cannot be read: {value}"
        ) from error
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
            and entry["platform"].get("target_family") == requirement["target_family"]
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
                    "source_revision": evidence["subject"].get("source_revision", ""),
                    "candidate_manifest_sha256": candidate_digest,
                    "platform": platform,
                }
            )
        entries.sort(key=lambda entry: entry["bundle_sha256"])
        policy_result = _evaluate_policy(entries, policy)
        index = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "product": "fastflow-sdk-qualification-index",
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
            raise EvidenceError(
                f"generated intake index is invalid: {error}"
            ) from error
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
            verify_aggregate(bundle_output, candidate_manifest=Path(candidate_manifest))
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


def _parse_checksum_records(text: str, *, label: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise EvidenceError(f"{label} has a malformed row")
        digest, name = match.group(1), match.group(2)
        if (
            name != Path(name).name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
        ):
            raise EvidenceError(f"{label} path must be a top-level file: {name}")
        if name in records:
            raise EvidenceError(f"{label} has a duplicate path: {name}")
        records[name] = digest
    if not records:
        raise EvidenceError(f"{label} is empty")
    return records


def _parse_payload_sha256sums(payload_directory: Path) -> dict[str, str]:
    checksums_path = payload_directory / "SHA256SUMS"
    try:
        text = checksums_path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(
            f"release payload SHA256SUMS is missing or unreadable: {checksums_path}"
        ) from error
    records = _parse_checksum_records(text, label="release payload SHA256SUMS")
    if "SHA256SUMS" in records:
        raise EvidenceError("release payload SHA256SUMS must not list itself")
    return records


def _verify_candidate_checksums(payload_directory: Path) -> None:
    """Validate the separate candidate checksum file used by release payloads.

    ``SHA256SUMS`` covers product assets. ``candidate-SHA256SUMS`` separately
    covers ``candidate-manifest.json`` and the qualification candidate wheels;
    those two files are intentionally not listed in ``SHA256SUMS``.
    """
    checksums_path = payload_directory / "candidate-SHA256SUMS"
    manifest_path = payload_directory / "candidate-manifest.json"
    if not checksums_path.is_file():
        raise EvidenceError(
            f"release payload candidate-SHA256SUMS is missing: {checksums_path}"
        )
    if not manifest_path.is_file():
        raise EvidenceError(f"candidate manifest is missing: {manifest_path}")
    try:
        text = checksums_path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(
            "release payload candidate-SHA256SUMS is unreadable"
        ) from error
    records = _parse_checksum_records(text, label="candidate-SHA256SUMS")
    if "candidate-manifest.json" not in records:
        raise EvidenceError("candidate-SHA256SUMS must list candidate-manifest.json")
    for name, digest in sorted(records.items()):
        path = payload_directory / name
        if not path.is_file():
            raise EvidenceError(f"candidate checksum target is missing: {name}")
        if sha256_file(path) != digest:
            raise EvidenceError(f"candidate checksum mismatch: {name}")


def stage_github_release(
    payload_directory: Path,
    aggregate: Path,
    output_directory: Path,
    *,
    index: Path | None = None,
    candidate_manifest: Path | None = None,
) -> dict[str, Any]:
    """Stage a flat GitHub-upload directory from a release payload plus intake.

    The upload set is exactly:

    - every file listed in the payload ``SHA256SUMS``, plus ``SHA256SUMS``
    - ``candidate-manifest.json`` and ``candidate-SHA256SUMS`` (separate from
      ``SHA256SUMS``, matching the collected release payload layout)
    - the aggregate ZIP, its external ``.sha256`` sidecar, and
      ``qualification-index.json``

    The aggregate basename must be ``qualification-evidence-<version>.zip``
    for the candidate version. ``output_directory`` must not exist or must be
    empty (same rule as intake). Nested payload directories such as ``simple/``
    are ignored. Any other top-level payload file fails the command.
    """
    payload_directory = Path(payload_directory).resolve()
    aggregate = Path(aggregate).resolve()
    output_directory = Path(output_directory).resolve()
    if not payload_directory.is_dir():
        raise EvidenceError(
            f"release payload directory does not exist: {payload_directory}"
        )
    if not aggregate.is_file():
        raise EvidenceError(f"qualification aggregate is missing: {aggregate}")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise EvidenceError("upload output directory is not empty")
    if (
        output_directory == payload_directory
        or output_directory in payload_directory.parents
    ):
        raise EvidenceError(
            "upload output directory must be outside the payload directory"
        )
    if aggregate.parent == output_directory or output_directory in aggregate.parents:
        raise EvidenceError(
            "upload output directory must not contain the aggregate input path"
        )

    payload_records = _parse_payload_sha256sums(payload_directory)
    for name, digest in sorted(payload_records.items()):
        path = payload_directory / name
        if not path.is_file():
            raise EvidenceError(f"release payload is missing listed file: {name}")
        if sha256_file(path) != digest:
            raise EvidenceError(f"release payload checksum mismatch: {name}")
    _verify_candidate_checksums(payload_directory)

    # Product SHA256SUMS plus the separate candidate checksum pair.
    candidate_names = {"candidate-manifest.json", "candidate-SHA256SUMS"}
    payload_names = set(payload_records) | {"SHA256SUMS"} | candidate_names
    top_level_files = {
        path.name for path in payload_directory.iterdir() if path.is_file()
    }
    missing_payload = sorted(payload_names - top_level_files)
    extra_payload = sorted(top_level_files - payload_names)
    if missing_payload or extra_payload:
        raise EvidenceError(
            "release payload top-level files do not match the upload contract: "
            f"missing={missing_payload} extra={extra_payload}"
        )

    candidate_path = (
        Path(candidate_manifest).resolve()
        if candidate_manifest is not None
        else payload_directory / "candidate-manifest.json"
    )
    if not candidate_path.is_file():
        raise EvidenceError(f"candidate manifest is missing: {candidate_path}")
    payload_candidate = payload_directory / "candidate-manifest.json"
    if candidate_path.read_bytes() != payload_candidate.read_bytes():
        raise EvidenceError(
            "candidate manifest does not match payload candidate-manifest.json"
        )
    candidate = normalize_candidate_manifest(
        _json_object(payload_candidate.read_bytes(), "candidate-manifest.json")
    )
    expected_aggregate = f"qualification-evidence-{candidate['version']}.zip"
    if aggregate.name != expected_aggregate:
        raise EvidenceError(
            "qualification aggregate basename must be "
            f"{expected_aggregate}: {aggregate.name}"
        )
    checksums_path = payload_directory / "SHA256SUMS"

    index_path = (
        Path(index).resolve()
        if index is not None
        else aggregate.with_name("qualification-index.json")
    )
    if not index_path.is_file():
        raise EvidenceError(f"qualification-index.json is missing: {index_path}")
    with zipfile.ZipFile(aggregate) as archive:
        try:
            embedded_index = archive.read("qualification-index.json")
        except KeyError as error:
            raise EvidenceError(
                "qualification aggregate is missing qualification-index.json"
            ) from error
    if index_path.read_bytes() != embedded_index:
        raise EvidenceError(
            "qualification-index.json does not match the aggregate contents"
        )

    aggregate_digest = _verify_external_checksum(aggregate)
    verify_summary = verify_aggregate(aggregate, candidate_manifest=candidate_path)

    sidecar = aggregate.with_name(aggregate.name + ".sha256")
    evidence_names = {
        aggregate.name,
        sidecar.name,
        "qualification-index.json",
    }
    overlap = evidence_names & payload_names
    if overlap:
        raise EvidenceError(
            "release payload already contains evidence upload names: "
            + ", ".join(sorted(overlap))
        )
    expected = payload_names | evidence_names

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=output_directory.name + ".",
            dir=output_directory.parent,
        )
    )
    try:
        for name in sorted(payload_names):
            shutil.copyfile(payload_directory / name, stage / name)
        shutil.copyfile(aggregate, stage / aggregate.name)
        shutil.copyfile(sidecar, stage / sidecar.name)
        shutil.copyfile(index_path, stage / "qualification-index.json")
        staged_names = {path.name for path in stage.iterdir() if path.is_file()}
        if any(path.is_dir() for path in stage.iterdir()):
            raise EvidenceError("staged upload directory contains nested directories")
        missing = sorted(expected - staged_names)
        extra = sorted(staged_names - expected)
        if missing or extra:
            raise EvidenceError(
                "staged GitHub upload set is incorrect: "
                f"missing={missing} extra={extra}"
            )
        for name, digest in payload_records.items():
            if sha256_file(stage / name) != digest:
                raise EvidenceError(f"staged payload checksum mismatch: {name}")
        if (stage / "SHA256SUMS").read_bytes() != checksums_path.read_bytes():
            raise EvidenceError("staged SHA256SUMS mismatch")
        if sha256_file(stage / aggregate.name) != aggregate_digest:
            raise EvidenceError("staged aggregate checksum mismatch")
        if (stage / sidecar.name).read_text(encoding="ascii") != (
            f"{aggregate_digest}  {aggregate.name}\n"
        ):
            raise EvidenceError("staged aggregate checksum sidecar mismatch")
        if (stage / "qualification-index.json").read_bytes() != embedded_index:
            raise EvidenceError("staged qualification-index.json mismatch")
        if output_directory.exists():
            output_directory.rmdir()
        stage.replace(output_directory)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
        "status": "pass",
        "output": str(output_directory),
        "files": sorted(expected),
        "file_count": len(expected),
        "aggregate": aggregate.name,
        "aggregate_sha256": aggregate_digest,
        "candidate_manifest_sha256": verify_summary["candidate_manifest_sha256"],
        "candidate_version": candidate["version"],
        "runs": verify_summary["runs"],
        "policy": verify_summary["policy"],
    }
