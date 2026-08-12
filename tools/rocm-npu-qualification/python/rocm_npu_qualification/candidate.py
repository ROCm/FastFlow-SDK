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
    KERNEL_RECIPE_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    QUALIFICATION_CONTRACT_SCHEMA_VERSION,
    xadx_rounds,
)
from .hashing import sha256_file
from .numeric_runtime import NUMERIC_RUNTIME
from .policy import PolicyValidationError, normalize_policy
from .profiles import ProfileError, validate_profile
from .schema import SchemaValidationError, load_packaged_schema, validate_schema


QUALIFICATION_DISTRIBUTIONS = (
    "rocm-npu-runtime",
    "rocm-npu-fastflowlm",
    "rocm-npu-qualification",
)
FASTFLOWLM_DISTRIBUTION = "rocm-npu-fastflowlm"


class CandidateError(RuntimeError):
    pass


def _normalize_policy(name: str, value: Any) -> dict[str, Any]:
    try:
        return normalize_policy(value, expected_name=name)
    except PolicyValidationError as error:
        raise CandidateError(f"candidate {error}: {name}") from error


def _normalize_qualification_contract(value: Any) -> dict[str, Any]:
    try:
        validate_schema(
            value, load_packaged_schema("qualification-contract-v1.schema.json")
        )
    except (SchemaValidationError, OSError, json.JSONDecodeError) as error:
        raise CandidateError(
            f"candidate qualification contract does not satisfy the v1 schema: {error}"
        ) from error
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema_version",
            "native_checks",
            "xadx_cases",
            "xadx_case_contracts",
            "xadx_materializations",
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
        or not isinstance(value.get("xadx_execution"), dict)
        or not isinstance(value.get("xadx_provenance"), dict)
        or not isinstance(value.get("profiles"), dict)
        or not value["profiles"]
        or not isinstance(value.get("policies"), dict)
        or not value["policies"]
    ):
        raise CandidateError("candidate qualification contract is invalid")
    case_contracts = value["xadx_case_contracts"]
    if (
        not isinstance(case_contracts, dict)
        or set(case_contracts) != set(value["xadx_cases"])
        or any(
            not isinstance(contract, dict)
            or set(contract)
            != {"readback_capable", "validated_bytes_per_dispatch"}
            or not isinstance(contract["readback_capable"], bool)
            or not isinstance(contract["validated_bytes_per_dispatch"], int)
            or contract["validated_bytes_per_dispatch"] < 1
            for contract in case_contracts.values()
        )
    ):
        raise CandidateError("candidate XADX case contracts are invalid")
    materializations = value["xadx_materializations"]
    if (
        not isinstance(materializations, list)
        or not materializations
        or any(
            not isinstance(item, dict)
            or set(item) != {"source", "size", "sha256", "contract_sha256"}
            or not isinstance(item["source"], dict)
            or not isinstance(item["size"], int)
            or item["size"] < 1
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            or re.fullmatch(r"[0-9a-f]{64}", item["contract_sha256"]) is None
            or item["sha256"] != item["source"].get("sha256")
            or item["contract_sha256"]
            != hashlib.sha256(
                json.dumps(
                    item["source"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            for item in materializations
        )
        or [item["contract_sha256"] for item in materializations]
        != sorted({item["contract_sha256"] for item in materializations})
    ):
        raise CandidateError("candidate XADX materialization contracts are invalid")
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
    execution = value["xadx_execution"]
    if execution != {name: xadx_rounds(name) for name in profiles}:
        raise CandidateError(
            "candidate XADX execution policy does not match its profiles"
        )
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
    expected_provenance_keys = {
        "recipe_sha256",
        "manifest_sha256",
        "serializer",
        "executables",
        "generator_version",
    }
    if (
        set(provenance)
        != expected_provenance_keys
        or re.fullmatch(r"[0-9a-f]{64}", provenance.get("recipe_sha256", "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", provenance.get("manifest_sha256", ""))
        is None
        or serializer
        != {
            "api": "hrx_amdxdna_xadx_serialize",
            "revision": "hrx-amdxdna-v0",
        }
        or not isinstance(executables, list)
        or not executables
        or provenance.get("generator_version") != "kernel-recipe-generator-v1"
    ):
        raise CandidateError("candidate XADX provenance is invalid")
    normalized_executables = []
    for executable in executables:
        modern_keys = {
            "id",
            "xclbin_path",
            "xclbin_sha256",
            "transaction",
            "transaction_sha256",
            "xadx",
            "xadx_sha256",
        }
        if (
            not isinstance(executable, dict)
            or set(executable)
            != modern_keys | ({"execution"} if "execution" in executable else set())
            or not isinstance(executable.get("id"), str)
            or not executable["id"]
            or any(
                re.fullmatch(r"[0-9a-f]{64}", executable.get(key, "")) is None
                for key in (
                    "xclbin_sha256",
                    "transaction_sha256",
                    "xadx_sha256",
                )
            )
            or (
                "execution" in executable
                and (
                    not isinstance(executable["execution"], str)
                    or not executable["execution"]
                )
            )
            or not isinstance(executable.get("xclbin_path"), str)
            or Path(executable["xclbin_path"]).is_absolute()
            or ".." in Path(executable["xclbin_path"]).parts
            or not isinstance(executable.get("transaction"), str)
            or Path(executable["transaction"]).is_absolute()
            or ".." in Path(executable["transaction"]).parts
            or not isinstance(executable.get("xadx"), str)
            or Path(executable["xadx"]).is_absolute()
            or ".." in Path(executable["xadx"]).parts
        ):
            raise CandidateError("candidate XADX executable provenance is invalid")
        normalized_executables.append(dict(executable))
    if len(
        {
            (item["id"], item.get("execution"))
            for item in normalized_executables
        }
    ) != len(
        normalized_executables
    ):
        raise CandidateError("candidate XADX execution identities are not unique")
    return {
        "schema_version": QUALIFICATION_CONTRACT_SCHEMA_VERSION,
        "native_checks": ["version", "probe", "doctor", "self-test"],
        "xadx_cases": list(value["xadx_cases"]),
        "xadx_case_contracts": {
            name: dict(contract) for name, contract in case_contracts.items()
        },
        "xadx_materializations": [
            {
                "source": item["source"],
                "size": item["size"],
                "sha256": item["sha256"],
                "contract_sha256": item["contract_sha256"],
            }
            for item in materializations
        ],
        "xadx_execution": {
            name: dict(rounds) for name, rounds in sorted(execution.items())
        },
        "xadx_provenance": {
            "recipe_sha256": provenance["recipe_sha256"],
            "manifest_sha256": provenance["manifest_sha256"],
            "serializer": dict(serializer),
            "executables": sorted(
                normalized_executables,
                key=lambda item: (item["id"], item.get("execution", "")),
            ),
            "generator_version": provenance["generator_version"],
        },
        "profiles": profiles,
        "policies": policies,
    }


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CandidateError(f"candidate manifest has invalid {label}")
    return value


def normalize_candidate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        validate_schema(manifest, load_packaged_schema("candidate-v1.schema.json"))
    except (SchemaValidationError, OSError, json.JSONDecodeError) as error:
        raise CandidateError(
            f"candidate manifest does not satisfy the v1 schema: {error}"
        ) from error
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
        "numeric_runtime",
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
    numeric_runtime = manifest.get("numeric_runtime")
    if numeric_runtime != NUMERIC_RUNTIME:
        raise CandidateError(
            "candidate numeric runtime versions do not match the qualification contract"
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
        "numeric_runtime": dict(NUMERIC_RUNTIME),
    }
    normalized["qualification_contract"] = qualification_contract
    return normalized


def canonical_candidate_bytes(manifest: dict[str, Any]) -> bytes:
    return (
        json.dumps(normalize_candidate_manifest(manifest), indent=2, sort_keys=True)
        + "\n"
    ).encode()


def _recipe_case_contracts(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for case in cases:
        expected = [
            buffer
            for stage in case.get("stages", ())
            for buffer in stage.get("buffers", ())
            if "expected" in buffer
        ]
        if len(expected) != 1:
            raise CandidateError(
                f"qualification recipe case has invalid expectations: {case.get('name')}"
            )
        result[case["name"]] = {
            "readback_capable": bool(case.get("readback_variants")),
            "validated_bytes_per_dispatch": expected[0]["size"],
        }
    return result


def _recipe_materializations(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    materializations: dict[str, dict[str, Any]] = {}

    def record(specification: Any, size: int) -> None:
        if not isinstance(specification, dict) or "stage_output" in specification:
            return
        canonical = json.dumps(
            specification, sort_keys=True, separators=(",", ":")
        ).encode()
        contract_sha256 = hashlib.sha256(canonical).hexdigest()
        identity = {
            "source": specification,
            "size": size,
            "sha256": specification["sha256"],
            "contract_sha256": contract_sha256,
        }
        previous = materializations.setdefault(contract_sha256, identity)
        if previous != identity:
            raise CandidateError(
                "qualification recipe reuses a generator with conflicting sizes"
            )

    for case in cases:
        expected_size = next(
            buffer["size"]
            for stage in case["stages"]
            for buffer in stage["buffers"]
            if "expected" in buffer
        )
        buffers = {
            (stage["name"], buffer["name"]): buffer
            for stage in case["stages"]
            for buffer in stage["buffers"]
        }
        for buffer in buffers.values():
            record(buffer.get("source"), buffer["size"])
            record(buffer.get("expected"), buffer["size"])
        for variant in case.get("readback_variants", ()):
            record(
                variant.get("source"),
                buffers[(variant["stage"], variant["buffer"])]["size"],
            )
            record(variant.get("expected"), expected_size)
    return [materializations[key] for key in sorted(materializations)]


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
    recipe_name = "rocm_npu_qualification/xadx_assets/recipe.json"
    if recipe_name not in archive.namelist():
        raise CandidateError("qualification kernel recipe is missing")
    try:
        recipe_bytes = archive.read(recipe_name)
        recipe = json.loads(recipe_bytes)
    except (KeyError, json.JSONDecodeError) as error:
        raise CandidateError("qualification kernel recipe is missing") from error
    cases = recipe.get("cases") if isinstance(recipe, dict) else None
    implementations = recipe.get("implementations") if isinstance(recipe, dict) else None
    if (
        xadx.get("schema_version") != 1
        or recipe.get("schema_version") != KERNEL_RECIPE_SCHEMA_VERSION
        or recipe.get("product") != "rocm-npu-fastflowlm-kernel-recipe"
        or recipe.get("target") != "npu2"
        or recipe.get("generator_version") != "kernel-recipe-generator-v1"
        or not isinstance(cases, list)
        or not cases
        or not isinstance(implementations, list)
        or not implementations
        or xadx.get("recipe", {}).get("sha256")
        != hashlib.sha256(recipe_bytes).hexdigest()
    ):
        raise CandidateError("qualification kernel recipe does not match assets")
    executable_provenance = []
    for implementation in implementations:
        executions = [
            (None, implementation),
            *[
                (variant.get("id"), variant)
                for variant in implementation.get("execution_variants", ())
            ],
        ]
        for execution, executable in executions:
            label = implementation.get("id")
            if execution is not None:
                label += f"/{execution}"
            xadx_path = (
                "rocm_npu_qualification/xadx_assets/"
                + executable.get("xadx", "")
            )
            try:
                xadx_bytes = archive.read(xadx_path)
            except KeyError as error:
                raise CandidateError(
                    f"qualification recipe XADX is missing: {label}"
                ) from error
            if hashlib.sha256(xadx_bytes).hexdigest() != executable.get(
                "xadx_sha256"
            ):
                raise CandidateError(
                    f"qualification recipe XADX is corrupt: {label}"
                )
            transaction_path = (
                "rocm_npu_qualification/xadx_assets/"
                + executable.get("transaction", "")
            )
            try:
                transaction_bytes = archive.read(transaction_path)
            except KeyError as error:
                raise CandidateError(
                    f"qualification recipe transaction is missing: {label}"
                ) from error
            if hashlib.sha256(transaction_bytes).hexdigest() != executable.get(
                "transaction_sha256"
            ):
                raise CandidateError(
                    f"qualification recipe transaction is corrupt: {label}"
                )
            executable_provenance.append(
                {
                    "id": implementation.get("id"),
                    "xclbin_path": implementation.get("xclbin_path"),
                    "xclbin_sha256": implementation.get("xclbin_sha256"),
                    "transaction": executable.get("transaction"),
                    "transaction_sha256": executable.get("transaction_sha256"),
                    "xadx": executable.get("xadx"),
                    "xadx_sha256": executable.get("xadx_sha256"),
                    **(
                        {}
                        if execution is None
                        else {"execution": execution}
                    ),
                }
            )
    return _normalize_qualification_contract(
        {
            "schema_version": QUALIFICATION_CONTRACT_SCHEMA_VERSION,
            "native_checks": ["version", "probe", "doctor", "self-test"],
            "xadx_cases": [case.get("name") for case in cases],
            "xadx_case_contracts": _recipe_case_contracts(cases),
            "xadx_materializations": _recipe_materializations(cases),
            "xadx_execution": {
                name: xadx_rounds(name) for name in sorted(profiles)
            },
            "xadx_provenance": {
                "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
                "manifest_sha256": hashlib.sha256(
                    json.dumps(
                        xadx, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "serializer": xadx.get("serializer"),
                "generator_version": recipe["generator_version"],
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
            if len(metadata_names) != 1:
                raise CandidateError(f"wheel must contain one METADATA: {wheel}")
            metadata = email.parser.BytesParser().parsebytes(
                archive.read(metadata_names[0])
            )
            normalized_distribution = re.sub(
                r"[-_.]+", "-", metadata["Name"]
            ).lower()
            if normalized_distribution in NUMERIC_RUNTIME:
                return None
            manifest_names = [
                name
                for name in archive.namelist()
                if name.count("/") == 1 and name.endswith("/manifest.json")
            ]
            if len(manifest_names) != 1:
                raise CandidateError(
                    f"wheel must contain one METADATA and package manifest: {wheel}"
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


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return (
            f"{path} has different types: "
            f"canonical={type(left).__name__}, candidate={type(right).__name__}"
        )
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return (
                f"{path} has different keys: "
                f"missing={sorted(left_keys - right_keys)}, "
                f"unexpected={sorted(right_keys - left_keys)}"
            )
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return (
                f"{path} has different lengths: "
                f"canonical={len(left)}, candidate={len(right)}"
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_item, right_item, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if left != right:
        return f"{path} differs: canonical={left!r}, candidate={right!r}"
    return None


def verify_qualification_wheel_parity(
    canonical_wheel: Path, candidate_wheel: Path
) -> None:
    canonical = qualification_wheel_contract(canonical_wheel)
    candidate = qualification_wheel_contract(candidate_wheel)
    if canonical != candidate:
        difference = _first_difference(canonical, candidate)
        raise CandidateError(
            "qualification wheel contracts or XADX provenance differ: "
            f"{difference or 'unknown difference'}"
        )
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


def _verify_recipe_source_xclbins(contract, fastflowlm_wheels):
    provenance = contract["xadx_provenance"]
    if "recipe_sha256" not in provenance:
        return
    for wheel in fastflowlm_wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            for executable in provenance["executables"]:
                suffix = (
                    "/share/flm/xclbins/"
                    + executable["xclbin_path"].replace("\\", "/")
                )
                matches = [name for name in names if name.endswith(suffix)]
                if len(matches) != 1:
                    raise CandidateError(
                        f"{wheel.name} does not contain exactly one source xclbin "
                        f"for {executable['id']}: {executable['xclbin_path']}"
                    )
                digest = hashlib.sha256(archive.read(matches[0])).hexdigest()
                if digest != executable["xclbin_sha256"]:
                    raise CandidateError(
                        f"{wheel.name} source xclbin differs from recipe: "
                        f"{executable['id']}"
                    )


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
    fastflowlm_wheels: list[Path] = []
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
            if distribution == FASTFLOWLM_DISTRIBUTION:
                fastflowlm_wheels.append(resolved_wheel)
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
    _verify_recipe_source_xclbins(qualification_contract, fastflowlm_wheels)
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
        "numeric_runtime": dict(NUMERIC_RUNTIME),
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
