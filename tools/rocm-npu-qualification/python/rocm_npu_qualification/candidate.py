# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Release-candidate identity generation and normalization."""

from __future__ import annotations

import email.parser
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from .contracts import (
    CANDIDATE_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    QUALIFICATION_CONTRACT_SCHEMA_VERSION,
    XADX_LIFECYCLE_ROUNDS,
    XADX_READBACK_ROUNDS,
    XADX_RESIDENT_ROUNDS,
)
from .hashing import sha256_file
from .policy import PolicyValidationError, normalize_policy
from .profiles import ProfileError, validate_profile


QUALIFICATION_DISTRIBUTIONS = (
    "rocm-npu-runtime",
    "rocm-npu-fastflowlm",
    "rocm-npu-qualification",
)


class CandidateError(RuntimeError):
    pass


def _normalize_policy(name: str, value: Any) -> dict[str, Any]:
    try:
        return normalize_policy(value, expected_name=name)
    except PolicyValidationError as error:
        raise CandidateError(f"candidate {error}: {name}") from error


def _normalize_qualification_contract(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "native_checks",
            "xadx_cases",
            "xadx_execution",
            "xadx_provenance",
            "profiles",
            "policies",
        }
        or value.get("schema_version") != QUALIFICATION_CONTRACT_SCHEMA_VERSION
        or value.get("native_checks") != ["version", "probe", "doctor", "self-test"]
        or not isinstance(value.get("xadx_cases"), list)
        or not value["xadx_cases"]
        or len(set(value["xadx_cases"])) != len(value["xadx_cases"])
        or not all(isinstance(name, str) and name for name in value["xadx_cases"])
        or value.get("xadx_execution")
        != {
            "lifecycle_rounds": XADX_LIFECYCLE_ROUNDS,
            "resident_rounds": XADX_RESIDENT_ROUNDS,
            "readback_rounds": XADX_READBACK_ROUNDS,
        }
        or not isinstance(value.get("xadx_provenance"), dict)
        or not isinstance(value.get("profiles"), dict)
        or not value["profiles"]
        or not isinstance(value.get("policies"), dict)
        or not value["policies"]
    ):
        raise CandidateError("candidate qualification contract is invalid")
    profiles = {}
    for name, profile in sorted(value["profiles"].items()):
        try:
            profile = validate_profile(profile)
        except ProfileError as error:
            raise CandidateError(
                f"candidate profile {name!r} is invalid: {error}"
            ) from error
        if profile["name"] != name:
            raise CandidateError("candidate profile name does not match its key")
        profiles[name] = profile
    policies = {
        name: _normalize_policy(name, policy)
        for name, policy in sorted(value["policies"].items())
    }
    if any(
        requirement["profile"] not in profiles
        for policy in policies.values()
        for requirement in policy["requirements"]
    ):
        raise CandidateError("candidate policy references an unknown profile")
    provenance = value["xadx_provenance"]
    serializer = provenance.get("serializer")
    executables = provenance.get("executables")
    if (
        set(provenance)
        != {
            "case_contract_sha256",
            "manifest_sha256",
            "serializer",
            "executables",
        }
        or re.fullmatch(r"[0-9a-f]{64}", provenance.get("case_contract_sha256", ""))
        is None
        or re.fullmatch(r"[0-9a-f]{64}", provenance.get("manifest_sha256", ""))
        is None
        or serializer
        != {
            "api": "hrx_amdxdna_xadx_serialize",
            "revision": "hrx-amdxdna-v0",
        }
        or not isinstance(executables, list)
        or not executables
    ):
        raise CandidateError("candidate XADX provenance is invalid")
    normalized_executables = []
    for executable in executables:
        if (
            not isinstance(executable, dict)
            or set(executable)
            != {
                "id",
                "source_xclbin_sha256",
                "transaction_sha256",
                "xadx_sha256",
            }
            or not isinstance(executable.get("id"), str)
            or not executable["id"]
            or any(
                re.fullmatch(r"[0-9a-f]{64}", executable.get(key, "")) is None
                for key in (
                    "source_xclbin_sha256",
                    "transaction_sha256",
                    "xadx_sha256",
                )
            )
        ):
            raise CandidateError("candidate XADX executable provenance is invalid")
        normalized_executables.append(dict(executable))
    if len({item["id"] for item in normalized_executables}) != len(
        normalized_executables
    ):
        raise CandidateError("candidate XADX executable IDs are not unique")
    return {
        "schema_version": QUALIFICATION_CONTRACT_SCHEMA_VERSION,
        "native_checks": ["version", "probe", "doctor", "self-test"],
        "xadx_cases": list(value["xadx_cases"]),
        "xadx_execution": dict(value["xadx_execution"]),
        "xadx_provenance": {
            "case_contract_sha256": provenance["case_contract_sha256"],
            "manifest_sha256": provenance["manifest_sha256"],
            "serializer": dict(serializer),
            "executables": sorted(normalized_executables, key=lambda item: item["id"]),
        },
        "profiles": profiles,
        "policies": policies,
    }


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CandidateError(f"candidate manifest has invalid {label}")
    return value


def normalize_candidate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise CandidateError("candidate manifest schema version is invalid")
    version = manifest.get("version")
    source_revision = manifest.get("source_revision", manifest.get("source_sha"))
    if (
        not isinstance(version, str)
        or re.fullmatch(r"[A-Za-z0-9.!+_-]+", version) is None
        or not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", source_revision) is None
    ):
        raise CandidateError("candidate manifest identity is invalid")
    identities: dict[str, list[dict[str, str]]] = {
        distribution: [] for distribution in QUALIFICATION_DISTRIBUTIONS
    }
    packages = manifest.get("packages")
    if set(manifest) != {
        "schema_version",
        "product",
        "version",
        "source_revision",
        "packages",
        "qualification_contract",
    }:
        raise CandidateError("candidate manifest properties are invalid")
    if manifest.get("product") != "rocm-npu-release-candidate":
        raise CandidateError("candidate manifest product is invalid")
    if not isinstance(packages, dict) or set(packages) != set(
        QUALIFICATION_DISTRIBUTIONS
    ):
        raise CandidateError(
            "candidate manifest package set must contain exactly the "
            "three qualification distributions"
        )
    qualification_contract = _normalize_qualification_contract(
        manifest.get("qualification_contract")
    )
    if not packages:
        raise CandidateError("candidate manifest package identities are required")
    for distribution, value in packages.items():
        values = value if isinstance(value, list) else [value]
        for identity in values:
            if not isinstance(identity, dict):
                raise CandidateError(
                    f"candidate identity for {distribution} is invalid"
                )
            filename = identity.get("filename")
            identity_version = identity.get("version", version)
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename) is None
                or not isinstance(identity_version, str)
                or identity_version != version
            ):
                raise CandidateError(
                    f"candidate identity for {distribution} is incomplete"
                )
            identities[distribution].append(
                {
                    "filename": filename,
                    "version": identity_version,
                    "sha256": _digest(
                        identity.get("sha256"), f"{distribution} wheel SHA256"
                    ),
                    "package_manifest_sha256": _digest(
                        identity.get("package_manifest_sha256"),
                        f"{distribution} package manifest SHA256",
                    ),
                }
            )
    missing = sorted(name for name, values in identities.items() if not values)
    if missing:
        raise CandidateError(
            f"candidate manifest is missing package identities: {missing}"
        )
    for distribution, values in identities.items():
        keys = {(identity["filename"], identity["sha256"]) for identity in values}
        if not values or len(keys) != len(values):
            raise CandidateError(
                f"candidate identities for {distribution} are duplicated"
            )
        values.sort(key=lambda identity: (identity["filename"], identity["sha256"]))
    normalized = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "product": "rocm-npu-release-candidate",
        "version": version,
        "source_revision": source_revision,
        "packages": identities,
    }
    normalized["qualification_contract"] = qualification_contract
    return normalized


def canonical_candidate_bytes(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(normalize_candidate_manifest(manifest), indent=2, sort_keys=True)
        + "\n"
    ).encode()


def _qualification_contract(archive: zipfile.ZipFile) -> dict[str, Any]:
    prefix = "rocm_npu_qualification/profiles/"
    raw_profiles = {
        Path(name).stem: json.loads(archive.read(name))
        for name in archive.namelist()
        if name.startswith(prefix) and name.endswith(".json")
    }
    loading: set[str] = set()

    def resolve(name: str) -> dict[str, Any]:
        if name in loading or name not in raw_profiles:
            raise CandidateError(
                f"qualification profile inheritance is invalid: {name}"
            )
        value = raw_profiles[name]
        if isinstance(value, dict) and "extends" in value:
            loading.add(name)
            parent = resolve(value["extends"])
            loading.remove(name)
            value = {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "name": value.get("name"),
                "description": value.get("description"),
                "models": [*parent["models"], *value.get("models", [])],
            }
        try:
            return validate_profile(value)
        except ProfileError as error:
            raise CandidateError(
                f"qualification wheel profile is invalid: {name}: {error}"
            ) from error

    profiles = {name: resolve(name) for name in sorted(raw_profiles)}
    policy_prefix = "rocm_npu_qualification/policies/"
    policies = {
        Path(name).stem: json.loads(archive.read(name))
        for name in archive.namelist()
        if name.startswith(policy_prefix) and name.endswith(".json")
    }
    xadx_names = [
        name
        for name in archive.namelist()
        if name.endswith("/xadx_assets/manifest.json")
    ]
    if len(xadx_names) != 1:
        raise CandidateError("qualification wheel XADX manifest is missing")
    xadx = json.loads(archive.read(xadx_names[0]))
    cases = xadx.get("cases") if isinstance(xadx, dict) else None
    if not isinstance(cases, list):
        raise CandidateError("qualification wheel XADX cases are invalid")
    if not all(isinstance(case, dict) for case in cases):
        raise CandidateError("qualification wheel XADX cases are invalid")
    case_contract_name = "rocm_npu_qualification/xadx_cases.json"
    try:
        case_contract_bytes = archive.read(case_contract_name)
        case_contract = json.loads(case_contract_bytes)
    except (KeyError, json.JSONDecodeError) as error:
        raise CandidateError("qualification XADX case contract is missing") from error
    case_contract_sha256 = hashlib.sha256(case_contract_bytes).hexdigest()
    expected_case_names = [
        f"{family}/{suffix}"
        for family in case_contract.get("families", [])
        for suffix in case_contract.get("case_suffixes", [])
    ] + case_contract.get("global_cases", [])
    if (
        case_contract.get("schema_version") != 1
        or xadx.get("schema_version") != 1
        or xadx.get("case_contract_sha256") != case_contract_sha256
        or [case.get("name") for case in cases] != expected_case_names
    ):
        raise CandidateError("qualification XADX case contract does not match assets")
    executable_provenance = []
    for executable in xadx.get("executables", []):
        executable_provenance.append(
            {
                key: executable.get(key)
                for key in (
                    "id",
                    "source_xclbin_sha256",
                    "transaction_sha256",
                    "xadx_sha256",
                )
            }
        )
    return _normalize_qualification_contract(
        {
            "schema_version": QUALIFICATION_CONTRACT_SCHEMA_VERSION,
            "native_checks": ["version", "probe", "doctor", "self-test"],
            "xadx_cases": [case.get("name") for case in cases],
            "xadx_execution": {
                "lifecycle_rounds": XADX_LIFECYCLE_ROUNDS,
                "resident_rounds": XADX_RESIDENT_ROUNDS,
                "readback_rounds": XADX_READBACK_ROUNDS,
            },
            "xadx_provenance": {
                "case_contract_sha256": case_contract_sha256,
                "manifest_sha256": hashlib.sha256(
                    json.dumps(
                        xadx, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "serializer": xadx.get("serializer"),
                "executables": executable_provenance,
            },
            "profiles": profiles,
            "policies": policies,
        }
    )


def _wheel_identity(
    wheel: Path,
) -> tuple[str, str, dict[str, str], dict[str, Any] | None] | None:
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            manifest_names = [
                name
                for name in archive.namelist()
                if name.count("/") == 1 and name.endswith("/manifest.json")
            ]
            if len(metadata_names) != 1 or len(manifest_names) != 1:
                raise CandidateError(
                    f"wheel must contain one METADATA and package manifest: {wheel}"
                )
            metadata = email.parser.BytesParser().parsebytes(
                archive.read(metadata_names[0])
            )
            manifest_bytes = archive.read(manifest_names[0])
            contract = (
                _qualification_contract(archive)
                if metadata["Name"] == "rocm-npu-qualification"
                else None
            )
    except (OSError, zipfile.BadZipFile) as error:
        raise CandidateError(f"candidate wheel is invalid: {wheel}") from error
    if metadata["Name"] not in QUALIFICATION_DISTRIBUTIONS:
        return None
    try:
        package_manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise CandidateError(f"invalid wheel package manifest: {wheel}") from error
    distribution = metadata["Name"]
    version = metadata["Version"]
    revision = (
        package_manifest.get("sources", {}).get("rocm-npu-staging", {}).get("revision")
    )
    if (
        distribution not in QUALIFICATION_DISTRIBUTIONS
        or package_manifest.get("product") != distribution
        or package_manifest.get("version") != version
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40,64}", revision) is None
    ):
        raise CandidateError(f"wheel package identity is invalid: {wheel}")
    return (
        distribution,
        revision,
        {
            "filename": wheel.name,
            "version": version,
            "sha256": sha256_file(wheel),
            "package_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        contract,
    )


def qualification_wheel_contract(wheel: Path) -> dict[str, Any]:
    """Returns the normalized qualification/XADX contract for parity checks."""
    identity = _wheel_identity(Path(wheel))
    if identity is None or identity[0] != "rocm-npu-qualification":
        raise CandidateError(f"not a qualification wheel: {wheel}")
    contract = identity[3]
    if contract is None:
        raise CandidateError(f"qualification contract is missing: {wheel}")
    return contract


def _wheel_members(wheel: Path) -> dict[str, tuple[bytes, int]]:
    try:
        with zipfile.ZipFile(wheel) as archive:
            members: dict[str, tuple[bytes, int]] = {}
            for info in archive.infolist():
                if info.filename in members:
                    raise CandidateError(
                        f"qualification wheel has duplicate member: {info.filename}"
                    )
                members[info.filename] = (
                    archive.read(info),
                    (info.external_attr >> 16) & 0xFFFF,
                )
            return members
    except zipfile.BadZipFile as error:
        raise CandidateError(f"qualification wheel is invalid: {wheel}") from error


def verify_qualification_wheel_parity(
    canonical_wheel: Path, candidate_wheel: Path
) -> None:
    canonical = qualification_wheel_contract(canonical_wheel)
    candidate = qualification_wheel_contract(candidate_wheel)
    if canonical != candidate:
        raise CandidateError("qualification wheel contracts or XADX provenance differ")
    canonical_members = _wheel_members(canonical_wheel)
    candidate_members = _wheel_members(candidate_wheel)
    if set(canonical_members) != set(candidate_members):
        raise CandidateError("qualification wheel member sets differ")
    for name in sorted(canonical_members):
        canonical_bytes, canonical_mode = canonical_members[name]
        candidate_bytes, candidate_mode = candidate_members[name]
        if canonical_bytes != candidate_bytes:
            raise CandidateError(
                f"qualification wheel member content differs: {name}"
            )
        if canonical_mode != candidate_mode:
            raise CandidateError(f"qualification wheel member mode differs: {name}")


def write_candidate_manifest(
    wheel_directories: list[Path],
    output: Path,
    *,
    expected_source_revision: str | None = None,
) -> Path:
    if (
        expected_source_revision is not None
        and re.fullmatch(r"[0-9a-f]{40,64}", expected_source_revision) is None
    ):
        raise CandidateError("expected source revision must be a full object ID")
    packages: dict[str, list[dict[str, str]]] = {
        distribution: [] for distribution in QUALIFICATION_DISTRIBUTIONS
    }
    revisions: set[str] = set()
    wheels: list[Path] = []
    discovered_paths: set[Path] = set()
    qualification_contract = None
    for directory in wheel_directories:
        for wheel in sorted(Path(directory).rglob("*.whl")):
            resolved_wheel = wheel.resolve()
            if resolved_wheel in discovered_paths:
                raise CandidateError(f"candidate wheel path is duplicated: {wheel}")
            discovered_paths.add(resolved_wheel)
            wheel_identity = _wheel_identity(wheel)
            if wheel_identity is None:
                continue
            distribution, revision, identity, contract = wheel_identity
            if identity in packages[distribution]:
                raise CandidateError(
                    f"candidate wheel identity is duplicated: {identity['filename']}"
                )
            packages[distribution].append(identity)
            wheels.append(resolved_wheel)
            revisions.add(revision)
            if contract is not None:
                if (
                    qualification_contract is not None
                    and qualification_contract != contract
                ):
                    raise CandidateError(
                        "qualification wheel contracts differ across candidates"
                    )
                qualification_contract = contract
    missing = sorted(
        distribution for distribution, identities in packages.items() if not identities
    )
    if missing:
        raise CandidateError(f"candidate wheel set is incomplete: {missing}")
    if qualification_contract is None:
        raise CandidateError("candidate qualification contract is missing")
    versions = {
        identity["version"]
        for identities in packages.values()
        for identity in identities
    }
    if len(versions) != 1:
        raise CandidateError(f"candidate wheel versions differ: {sorted(versions)}")
    if len(revisions) != 1:
        raise CandidateError(
            f"candidate wheel source revisions differ: {sorted(revisions)}"
        )
    source_revision = revisions.pop()
    if (
        expected_source_revision is not None
        and source_revision != expected_source_revision
    ):
        raise CandidateError(
            "candidate wheel source revision does not match expected revision"
        )
    manifest = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "product": "rocm-npu-release-candidate",
        "version": versions.pop(),
        "source_revision": source_revision,
        "packages": packages,
        "qualification_contract": qualification_contract,
    }
    output = Path(output)
    root = output.parent.resolve()
    output_path = output.resolve()
    try:
        output_relative = output_path.relative_to(root).as_posix()
        wheel_members = [
            (wheel.relative_to(root).as_posix(), wheel) for wheel in wheels
        ]
    except ValueError as error:
        raise CandidateError(
            "candidate wheels and manifest must share one artifact root"
        ) from error

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = canonical_candidate_bytes(manifest)
    temporary = output.with_name(output.name + ".tmp")
    checksum = root / "candidate-SHA256SUMS"
    checksum_temporary = checksum.with_name(checksum.name + ".tmp")
    checksums = [
        (output_relative, hashlib.sha256(manifest_bytes).hexdigest()),
        *((relative, sha256_file(wheel)) for relative, wheel in wheel_members),
    ]
    try:
        temporary.write_bytes(manifest_bytes)
        checksum_temporary.write_text(
            "".join(
                f"{digest}  {relative}\n"
                for relative, digest in sorted(checksums)
            ),
            encoding="ascii",
        )
        temporary.replace(output)
        # The checksum is the commit marker: readers require it before consuming
        # the manifest and verify every listed artifact.
        checksum_temporary.replace(checksum)
    finally:
        temporary.unlink(missing_ok=True)
        checksum_temporary.unlink(missing_ok=True)
    return output
