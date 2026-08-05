# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate qualification candidate identity and contract normalization."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "rocm-npu-qualification" / "python"))

from build_tools.rocm_npu_wheel_common import (  # noqa: E402
    FASTFLOWLM_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
)
from rocm_npu_qualification import candidate, cli, installation  # noqa: E402


pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def _write_wheel(
    root: Path,
    distribution: str,
    package: str,
    *,
    xadx_sha: str = "3" * 64,
    oracle_sha: str = "5" * 64,
    qualification_code: bytes = b"QUALIFICATION = True\n",
    qualification_mode: int = 0o100644,
    extra_member: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    wheel = root / f"{distribution.replace('-', '_')}-1.2.3-py3-none-any.whl"
    dist_info = f"{distribution.replace('-', '_')}-1.2.3.dist-info"
    metadata_name = f"{dist_info}/METADATA"
    manifest_name = f"{package}/manifest.json"
    metadata_bytes = (
        f"Metadata-Version: 2.4\nName: {distribution}\nVersion: 1.2.3\n"
    ).encode()
    manifest = json.dumps(
        {
            "schema_version": 1,
            "product": distribution,
            "version": "1.2.3",
            "sources": {
                "rocm-npu-staging": {"revision": "a" * 40},
            },
            "files": [],
        },
        sort_keys=True,
    ).encode()
    record_name = f"{dist_info}/RECORD"

    def record(path: str, content: bytes) -> str:
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
        return f"{path},sha256={digest.decode()},{len(content)}"

    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(metadata_name, metadata_bytes)
        archive.writestr(manifest_name, manifest)
        if distribution == QUALIFICATION_DISTRIBUTION:
            code = zipfile.ZipInfo("rocm_npu_qualification/__init__.py")
            code.external_attr = qualification_mode << 16
            archive.writestr(code, qualification_code)
            case_contract = json.dumps(
                {
                    "schema_version": 1,
                    "families": [],
                    "case_suffixes": [],
                    "global_cases": ["case"],
                }
            ).encode()
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
                "rocm_npu_qualification/xadx_cases.json",
                case_contract,
            )
            archive.writestr(
                "rocm_npu_qualification/xadx_assets/manifest.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "case_contract_sha256": hashlib.sha256(
                            case_contract
                        ).hexdigest(),
                        "serializer": {
                            "api": "hrx_amdxdna_xadx_serialize",
                            "revision": "hrx-amdxdna-v0",
                        },
                        "executables": [
                            {
                                "id": "test",
                                "source_xclbin_sha256": "1" * 64,
                                "transaction_sha256": "2" * 64,
                                "xadx_sha256": xadx_sha,
                            }
                        ],
                        "cases": [{"name": "case"}],
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
                (
                    record(metadata_name, metadata_bytes),
                    record(manifest_name, manifest),
                    f"{record_name},,",
                )
            )
            + "\n",
        )
    return wheel


def test_qualification_wheel_parity_binds_xadx_hashes(tmp_path):
    canonical = _write_wheel(
        tmp_path / "canonical",
        QUALIFICATION_DISTRIBUTION,
        "rocm_npu_qualification",
    )
    matching = _write_wheel(
        tmp_path / "matching",
        QUALIFICATION_DISTRIBUTION,
        "rocm_npu_qualification",
    )
    candidate.verify_qualification_wheel_parity(canonical, matching)
    different = _write_wheel(
        tmp_path / "different",
        QUALIFICATION_DISTRIBUTION,
        "rocm_npu_qualification",
        xadx_sha="4" * 64,
    )
    with pytest.raises(candidate.CandidateError, match="XADX provenance"):
        candidate.verify_qualification_wheel_parity(canonical, different)
    different_oracle = _write_wheel(
        tmp_path / "different-oracle",
        QUALIFICATION_DISTRIBUTION,
        "rocm_npu_qualification",
        oracle_sha="6" * 64,
    )
    with pytest.raises(candidate.CandidateError, match="XADX provenance"):
        candidate.verify_qualification_wheel_parity(canonical, different_oracle)


def test_qualification_wheel_parity_binds_every_member_and_mode(tmp_path):
    canonical = _write_wheel(
        tmp_path / "canonical",
        QUALIFICATION_DISTRIBUTION,
        "rocm_npu_qualification",
    )
    different_content = _write_wheel(
        tmp_path / "different-content",
        QUALIFICATION_DISTRIBUTION,
        "rocm_npu_qualification",
        qualification_code=b"QUALIFICATION = False\n",
    )
    with pytest.raises(candidate.CandidateError, match="member content differs"):
        candidate.verify_qualification_wheel_parity(canonical, different_content)

    different_mode = _write_wheel(
        tmp_path / "different-mode",
        QUALIFICATION_DISTRIBUTION,
        "rocm_npu_qualification",
        qualification_mode=0o100755,
    )
    with pytest.raises(candidate.CandidateError, match="member mode differs"):
        candidate.verify_qualification_wheel_parity(canonical, different_mode)

    extra_member = _write_wheel(
        tmp_path / "extra-member",
        QUALIFICATION_DISTRIBUTION,
        "rocm_npu_qualification",
        extra_member="rocm_npu_qualification/unexpected.py",
    )
    with pytest.raises(candidate.CandidateError, match="member sets differ"):
        candidate.verify_qualification_wheel_parity(canonical, extra_member)


def test_candidate_manifest_binds_exact_wheels(tmp_path):
    _write_wheel(tmp_path, RUNTIME_DISTRIBUTION, "rocm_npu_runtime")
    _write_wheel(tmp_path, FASTFLOWLM_DISTRIBUTION, "rocm_npu_fastflowlm")
    _write_wheel(tmp_path, QUALIFICATION_DISTRIBUTION, "rocm_npu_qualification")
    output = candidate.write_candidate_manifest(
        [tmp_path],
        tmp_path / "candidate.json",
        expected_source_revision="a" * 40,
    )
    candidate_manifest = json.loads(output.read_text(encoding="utf-8"))
    assert candidate_manifest["source_revision"] == "a" * 40
    assert set(candidate_manifest["packages"]) == set(
        installation.PACKAGE_DISTRIBUTIONS
    )
    for identities in candidate_manifest["packages"].values():
        assert len(identities) == 1
        identity = identities[0]
        assert len(identity["sha256"]) == 64
        assert len(identity["package_manifest_sha256"]) == 64
    normalized = installation.normalize_candidate(candidate_manifest)
    selected = {
        distribution: identities[0]
        for distribution, identities in normalized["packages"].items()
    }
    installation.verify_candidate_wheels(selected, output)
    with pytest.raises(RuntimeError, match="does not match expected revision"):
        candidate.write_candidate_manifest(
            [tmp_path],
            tmp_path / "wrong-source.json",
            expected_source_revision="b" * 40,
        )
    next(tmp_path.glob("rocm_npu_runtime-*.whl")).write_bytes(b"substituted")
    with pytest.raises(cli.QualificationError, match="wheel SHA256"):
        installation.verify_candidate_wheels(selected, output)


def test_candidate_rejects_external_wheels_before_publishing(tmp_path):
    wheels = tmp_path / "external-wheels"
    _write_wheel(wheels, RUNTIME_DISTRIBUTION, "rocm_npu_runtime")
    _write_wheel(wheels, FASTFLOWLM_DISTRIBUTION, "rocm_npu_fastflowlm")
    _write_wheel(wheels, QUALIFICATION_DISTRIBUTION, "rocm_npu_qualification")
    artifact_root = tmp_path / "candidate"
    output = artifact_root / "candidate-manifest.json"

    with pytest.raises(
        candidate.CandidateError,
        match="wheels and manifest must share one artifact root",
    ):
        candidate.write_candidate_manifest([wheels], output)

    assert not output.exists()
    assert not (artifact_root / "candidate-SHA256SUMS").exists()


def test_candidate_rejects_duplicate_physical_wheel_identity(tmp_path):
    wheels = tmp_path / "wheels"
    duplicate = wheels / "duplicate"
    runtime = _write_wheel(wheels, RUNTIME_DISTRIBUTION, "rocm_npu_runtime")
    _write_wheel(wheels, FASTFLOWLM_DISTRIBUTION, "rocm_npu_fastflowlm")
    _write_wheel(wheels, QUALIFICATION_DISTRIBUTION, "rocm_npu_qualification")
    duplicate.mkdir()
    (duplicate / runtime.name).write_bytes(runtime.read_bytes())
    with pytest.raises(candidate.CandidateError, match="identity is duplicated"):
        candidate.write_candidate_manifest(
            [wheels], tmp_path / "candidate-manifest.json"
        )


def test_candidate_checksum_is_the_publication_marker(monkeypatch, tmp_path):
    wheels = tmp_path / "wheels"
    _write_wheel(wheels, RUNTIME_DISTRIBUTION, "rocm_npu_runtime")
    _write_wheel(wheels, FASTFLOWLM_DISTRIBUTION, "rocm_npu_fastflowlm")
    _write_wheel(wheels, QUALIFICATION_DISTRIBUTION, "rocm_npu_qualification")
    output = tmp_path / "candidate-manifest.json"
    checksum = tmp_path / "candidate-SHA256SUMS"
    original_replace = Path.replace

    def fail_marker(path, target):
        if Path(target) == checksum:
            raise OSError("simulated marker publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_marker)
    with pytest.raises(OSError, match="marker publication"):
        candidate.write_candidate_manifest([wheels], output)
    assert output.is_file()
    assert not checksum.exists()


def test_installed_candidate_cli_infers_revision_without_plugin_protocol(
    tmp_path, capsys
):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _write_wheel(wheels, RUNTIME_DISTRIBUTION, "rocm_npu_runtime")
    _write_wheel(wheels, FASTFLOWLM_DISTRIBUTION, "rocm_npu_fastflowlm")
    _write_wheel(wheels, QUALIFICATION_DISTRIBUTION, "rocm_npu_qualification")
    output = tmp_path / "candidate-manifest.json"
    release_checksums = tmp_path / "SHA256SUMS"
    release_checksums.write_text("release payload checksums\n", encoding="ascii")
    assert (
        cli.main(
            [
                "candidate",
                "--wheel-dir",
                str(wheels),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["source_revision"] == "a" * 40
    assert (tmp_path / "candidate-SHA256SUMS").is_file()
    assert release_checksums.read_text(encoding="ascii") == (
        "release payload checksums\n"
    )


def test_candidate_rejects_untrusted_fields():
    digest = "a" * 64
    manifest = {
        "schema_version": 1,
        "product": "rocm-npu-release-candidate",
        "version": "1.2.3",
        "source_revision": "a" * 40,
        "authorization": "Bearer secret",
        "packages": {
            distribution: {
                "filename": f"{distribution}.whl",
                "version": "1.2.3",
                "sha256": digest,
                "package_manifest_sha256": digest,
            }
            for distribution in installation.PACKAGE_DISTRIBUTIONS
        },
    }
    with pytest.raises(cli.QualificationError, match="properties"):
        installation.normalize_candidate(manifest)


def test_legacy_candidate_shape_is_rejected():
    with pytest.raises(cli.QualificationError, match="properties"):
        installation.normalize_candidate(
            {
                "schema_version": 1,
                "product": "rocm-npu",
                "version": "1.2.3",
                "source_sha": "a" * 40,
                "artifacts": [
                    {
                        "distribution": [],
                        "filename": "invalid.whl",
                        "version": "1.2.3",
                        "sha256": "b" * 64,
                        "package_manifest_sha256": "c" * 64,
                    }
                ],
            }
        )


def test_installed_package_files_and_manifests_are_verified(monkeypatch, tmp_path):
    modules = {}
    identities = {}
    for distribution in installation.PACKAGE_DISTRIBUTIONS:
        package_root = tmp_path / distribution
        package_root.mkdir()
        module = package_root / "__init__.py"
        payload = package_root / "payload.txt"
        module.write_text("", encoding="utf-8")
        payload.write_text(distribution, encoding="utf-8")
        manifest = {
            "files": [
                {
                    "path": "payload.txt",
                    "size": payload.stat().st_size,
                    "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                }
            ]
        }
        manifest_path = package_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        modules[distribution.replace("-", "_")] = SimpleNamespace(__file__=str(module))
        identities[distribution] = [
            {
                "filename": f"{distribution}.whl",
                "version": "1.2.3",
                "sha256": "b" * 64,
                "package_manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        ]

    monkeypatch.setattr(
        installation.importlib,
        "import_module",
        lambda name: modules[name],
    )
    installed = {
        distribution: "1.2.3" for distribution in installation.PACKAGE_DISTRIBUTIONS
    }
    installation.verify_installed_packages({"packages": identities}, installed)
    (tmp_path / RUNTIME_DISTRIBUTION / "payload.txt").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(cli.QualificationError, match="file verification failed"):
        installation.verify_installed_packages({"packages": identities}, installed)


def test_installed_dist_info_is_bound_to_candidate_wheel(monkeypatch, tmp_path):
    wheel = _write_wheel(tmp_path, RUNTIME_DISTRIBUTION, "rocm_npu_runtime")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        record_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        )
        metadata_path = installed / metadata_name
    monkeypatch.setattr(
        installation.metadata,
        "distribution",
        lambda _name: SimpleNamespace(
            locate_file=lambda member: installed / str(member)
        ),
    )
    installation.verify_installed_wheel_metadata({RUNTIME_DISTRIBUTION: wheel})
    record_path = installed / record_name
    original_record = record_path.read_text(encoding="utf-8")
    record_path.write_text(original_record + "evil.pth,,\n", encoding="utf-8")
    with pytest.raises(cli.QualificationError, match="unexpected file"):
        installation.verify_installed_wheel_metadata({RUNTIME_DISTRIBUTION: wheel})
    record_path.write_text(original_record, encoding="utf-8")
    metadata_path.write_text("substituted", encoding="utf-8")
    with pytest.raises(cli.QualificationError, match="wheel metadata"):
        installation.verify_installed_wheel_metadata({RUNTIME_DISTRIBUTION: wheel})
