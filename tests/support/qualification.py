# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qualification wheel and offline-evidence fixture factories."""

from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path

from build_tools.rocm_npu_wheel_common import (
    FASTFLOWLM_DISTRIBUTION as _FASTFLOWLM_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION as _QUALIFICATION_DISTRIBUTION,
)
from rocm_npu_qualification import installation
from rocm_npu_qualification.candidate import (
    canonical_candidate_bytes as _canonical_candidate_bytes,
)
from rocm_npu_qualification.evidence import EvidenceWriter
from rocm_npu_qualification.numeric_runtime import (
    NUMERIC_RUNTIME as _NUMERIC_RUNTIME,
)

__all__ = (
    "make_candidate",
    "make_evidence",
    "make_policy",
    "make_profile",
    "write_candidate_wheel",
)


def _record(path: str, content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    return f"{path},sha256={digest.decode()},{len(content)}"


def write_candidate_wheel(
    root: Path,
    distribution: str,
    package: str,
    *,
    xadx_data: bytes = b"xadx",
    oracle_sha: str = "5" * 64,
    qualification_code: bytes = b"QUALIFICATION = True\n",
    qualification_mode: int = 0o100644,
    extra_member: str | None = None,
) -> Path:
    """Write a minimal wheel carrying candidate-identity fixture metadata."""
    root.mkdir(parents=True, exist_ok=True)
    normalized = distribution.replace("-", "_")
    wheel = root / f"{normalized}-1.2.3-py3-none-any.whl"
    dist_info = f"{normalized}-1.2.3.dist-info"
    metadata_name = f"{dist_info}/METADATA"
    manifest_name = f"{package}/manifest.json"
    wheel_name = f"{dist_info}/WHEEL"
    metadata_bytes = (
        f"Metadata-Version: 2.4\nName: {distribution}\nVersion: 1.2.3\n"
    ).encode()
    manifest = json.dumps(
        {
            "schema_version": 1,
            "product": distribution,
            "version": "1.2.3",
            "sources": {"rocm-npu-staging": {"revision": "a" * 40}},
            "files": [],
        },
        sort_keys=True,
    ).encode()
    record_name = f"{dist_info}/RECORD"

    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(metadata_name, metadata_bytes)
        archive.writestr(wheel_name, b"Wheel-Version: 1.0\nTag: py3-none-any\n")
        archive.writestr(manifest_name, manifest)
        if distribution != _QUALIFICATION_DISTRIBUTION:
            archive.writestr(f"{package}/__init__.py", b"")
        if distribution == _FASTFLOWLM_DISTRIBUTION:
            archive.writestr(
                "rocm_npu_fastflowlm/share/flm/xclbins/xclbins/test.xclbin",
                b"xclbin",
            )
        if distribution == _QUALIFICATION_DISTRIBUTION:
            code = zipfile.ZipInfo("rocm_npu_qualification/__init__.py")
            code.external_attr = qualification_mode << 16
            archive.writestr(code, qualification_code)
            transaction_data = b"transaction"
            xclbin_sha = hashlib.sha256(b"xclbin").hexdigest()
            recipe = {
                "schema_version": 1,
                "product": "rocm-npu-fastflowlm-kernel-recipe",
                "target": "npu2",
                "generator_version": "kernel-recipe-generator-v1",
                "implementations": [
                    {
                        "id": f"dequant-{xclbin_sha[:16]}",
                        "xclbin_path": "xclbins/test.xclbin",
                        "xclbin_sha256": xclbin_sha,
                        "transaction": "transactions/test.bin",
                        "transaction_sha256": hashlib.sha256(
                            transaction_data
                        ).hexdigest(),
                        "xadx": "xadx/test.xadx",
                        "xadx_sha256": hashlib.sha256(xadx_data).hexdigest(),
                    }
                ],
                "coverage_limits": [],
                "cases": [
                    {
                        "name": "case",
                        "stages": [
                            {
                                "name": "dequant",
                                "implementation": f"dequant-{xclbin_sha[:16]}",
                                "buffers": [
                                    {
                                        "name": "output",
                                        "direction": "output",
                                        "size": 8,
                                        "expected": {
                                            "id": "zero",
                                            "parameters": {"size": 8},
                                            "sha256": hashlib.sha256(
                                                bytes(8)
                                            ).hexdigest(),
                                        },
                                        "comparison": {"kind": "exact"},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
            recipe_bytes = json.dumps(recipe).encode()
            profile = {
                "schema_version": 1,
                "name": "fast",
                "description": "test",
                "models": [
                    {
                        "tag": "model",
                        "source": "https://test",
                        "revision": "revision",
                        "identity_policy": "revision-sha256",
                        "registry_sha256": "a" * 64,
                        "files": [{"path": "model", "sha256": "b" * 64}],
                        "xclbins": [],
                        "tests": [
                            {
                                "name": "run",
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
            archive.writestr(
                "rocm_npu_qualification/profiles/fast.json",
                json.dumps(profile),
            )
            archive.writestr(
                "rocm_npu_qualification/xadx_assets/manifest.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "recipe": {
                            "path": "recipe.json",
                            "sha256": hashlib.sha256(recipe_bytes).hexdigest(),
                        },
                        "serializer": {
                            "api": "hrx_amdxdna_xadx_serialize",
                            "revision": "hrx-amdxdna-v0",
                        },
                        "files": [
                            {
                                "path": "data/oracle.bin.gz",
                                "size": 123,
                                "sha256": oracle_sha,
                            }
                        ],
                    }
                ),
            )
            archive.writestr(
                "rocm_npu_qualification/xadx_assets/recipe.json", recipe_bytes
            )
            archive.writestr(
                "rocm_npu_qualification/xadx_assets/xadx/test.xadx", xadx_data
            )
            archive.writestr(
                "rocm_npu_qualification/xadx_assets/transactions/test.bin",
                transaction_data,
            )
            archive.writestr(
                "rocm_npu_qualification/policies/release.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": "release",
                        "requirements": [
                            {
                                "platform_os": "windows",
                                "profile": "fast",
                                "target_family": "npu2",
                                "minimum_runs": 1,
                            }
                        ],
                    }
                ),
            )
            if extra_member is not None:
                archive.writestr(extra_member, b"extra")
        archive.writestr(
            record_name,
            "\n".join(
                [
                    *(_record(name, archive.read(name)) for name in archive.namelist()),
                    f"{record_name},,",
                ]
            )
            + "\n",
        )
    return wheel


def make_profile() -> dict[str, object]:
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


def make_policy(*platforms: str) -> dict[str, object]:
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


def make_candidate(revision: str = "a" * 40) -> dict[str, object]:
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
        "numeric_runtime": dict(_NUMERIC_RUNTIME),
        "qualification_contract": {
            "schema_version": 1,
            "native_checks": ["version", "probe", "doctor", "self-test"],
            "xadx_cases": ["xadx-case"],
            "xadx_case_contracts": {
                "xadx-case": {
                    "readback_capable": True,
                    "validated_bytes_per_dispatch": 8,
                }
            },
            "xadx_materializations": [
                {
                    "source": {
                        "id": "zero",
                        "parameters": {"size": 8},
                        "sha256": "8" * 64,
                    },
                    "size": 8,
                    "sha256": "8" * 64,
                    "contract_sha256": hashlib.sha256(
                        json.dumps(
                            {
                                "id": "zero",
                                "parameters": {"size": 8},
                                "sha256": "8" * 64,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                }
            ],
            "xadx_execution": {
                "fast": {
                    "lifecycle_rounds": 5,
                    "resident_rounds": 20,
                    "readback_rounds": 32,
                }
            },
            "xadx_provenance": {
                "recipe_sha256": "d" * 64,
                "manifest_sha256": "e" * 64,
                "serializer": {
                    "api": "hrx_amdxdna_xadx_serialize",
                    "revision": "hrx-amdxdna-v0",
                },
                "generator_version": "kernel-recipe-generator-v1",
                "executables": [
                    {
                        "id": "test",
                        "xclbin_path": "xclbins/test.xclbin",
                        "xclbin_sha256": "1" * 64,
                        "transaction": "transactions/test.bin",
                        "transaction_sha256": "2" * 64,
                        "xadx": "xadx/test.xadx",
                        "xadx_sha256": "3" * 64,
                    }
                ],
            },
            "profiles": {"fast": make_profile()},
            "policies": {
                policy["name"]: policy
                for policy in (
                    make_policy("windows"),
                    make_policy("windows", "linux"),
                )
            },
        },
    }


def make_evidence(
    root: Path,
    *,
    candidate=None,
    marker="one",
    status="pass",
    platform_os="Windows 11",
    profile_files=None,
) -> Path:
    """Create a sealed qualification evidence tree for offline-intake tests."""
    writer = EvidenceWriter(root)
    candidate = make_candidate() if candidate is None else candidate
    profile = make_profile()
    if profile_files is not None:
        profile["models"][0]["files"] = profile_files
        candidate["qualification_contract"]["profiles"]["fast"] = profile
    writer.write_candidate_manifest(candidate)
    model = {
        "tag": "qwen3:0.6b",
        "identity_policy": "revision-sha256",
        "source": "https://test",
        "revision": "immutable",
        "status": "pass",
        "cache_directory": "model",
        "registry_sha256": "d" * 64,
        "pulled": False,
        "files": [
            {**record, "size": 1}
            for record in sorted(
                profile["models"][0]["files"], key=lambda record: record["path"]
            )
        ],
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
    embedded = _canonical_candidate_bytes(candidate)
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
            "numeric_runtime": {
                distribution: {
                    "version": version,
                    "compatibility_tags": ["cp312-cp312-win_amd64"],
                    "record_sha256": "6" * 64,
                    "payload_sha256": "7" * 64,
                }
                for distribution, version in _NUMERIC_RUNTIME.items()
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
                "rounds": 57,
                "metrics": {
                    "architecture": "AMD_AIE2_NPU2",
                    "lifecycle_rounds": 5,
                    "resident_rounds": 20,
                    "readback_rounds": 32,
                    "validated_bytes": 456,
                    "mismatch_count": 0,
                    "first_mismatch": None,
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
            "schema_version": 1,
            "report_kind": "execution",
            "status": "pass",
            "recipe_sha256": "d" * 64,
            "generator_version": "kernel-recipe-generator-v1",
            "executables": candidate["qualification_contract"]["xadx_provenance"][
                "executables"
            ],
            "architecture": "AMD_AIE2_NPU2",
            "lifecycle_rounds": 5,
            "resident_rounds": 20,
            "readback_rounds": 32,
            "cases": 1,
            "readback_by_case": {"xadx-case": 32},
            "numeric_runtime": dict(_NUMERIC_RUNTIME),
            "case_results": [
                {
                    "name": "xadx-case",
                    "status": "pass",
                    "dispatches": 57,
                    "readbacks": 32,
                    "validated_bytes": 456,
                    "duration_seconds": 1,
                    "mismatch_count": 0,
                    "first_mismatch": None,
                    "failures": [],
                }
            ],
            "case_names": ["xadx-case"],
            "materialized_buffers": candidate["qualification_contract"][
                "xadx_materializations"
            ],
            "failures": [],
        },
    )
    writer.write_junit(evidence["tests"])
    writer.write_json("evidence.json", evidence)
    writer.finalize()
    return root
