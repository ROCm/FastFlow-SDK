# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate qualification candidate identity and contract normalization."""

from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from build_tools.rocm_npu_wheel_common import (
    FASTFLOWLM_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
)
from rocm_npu_qualification import candidate, cli, installation
from tests.support.qualification import write_candidate_wheel as _write_wheel

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def test_recipe_materializations_bind_preserved_chain_buffers():
    def generator(name: str, digest: str) -> dict:
        return {
            "id": name,
            "parameters": {"size": 8},
            "sha256": digest * 64,
        }

    cases = [
        {
            "name": "chained",
            "stages": [
                {
                    "name": "dequant",
                    "buffers": [
                        {
                            "name": "input",
                            "direction": "input",
                            "size": 8,
                            "source": generator("source", "1"),
                        },
                        {
                            "name": "weight",
                            "direction": "output",
                            "size": 8,
                            "preserved": generator("preserved", "2"),
                        },
                    ],
                },
                {
                    "name": "mm",
                    "buffers": [
                        {
                            "name": "weight",
                            "direction": "input",
                            "size": 8,
                            "source": {"stage_output": "dequant.weight"},
                        },
                        {
                            "name": "output",
                            "direction": "output",
                            "size": 8,
                            "expected": generator("expected", "3"),
                        },
                    ],
                },
            ],
        }
    ]

    materializations = candidate._recipe_materializations(cases)

    assert {item["source"]["id"] for item in materializations} == {
        "source",
        "preserved",
        "expected",
    }
    assert [item["contract_sha256"] for item in materializations] == sorted(
        item["contract_sha256"] for item in materializations
    )


def _write_numeric_wheel(
    root: Path,
    distribution: str,
    version: str,
    tag: str = "py3-none-any",
) -> Path:
    normalized = distribution.replace("-", "_")
    wheel = root / f"{normalized}-{version}-{tag}.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    members = {
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.4\nName: {normalized}\nVersion: {version}\n"
        ).encode(),
        f"{dist_info}/WHEEL": f"Wheel-Version: 1.0\nTag: {tag}\n".encode(),
        f"{normalized}/__init__.py": f"__version__ = {version!r}\n".encode(),
        f"{normalized}.libs/runtime.bin": b"native-runtime",
    }

    def record(path: str, content: bytes) -> str:
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
        return f"{path},sha256={digest.decode()},{len(content)}"

    record_name = f"{dist_info}/RECORD"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, content in members.items():
            archive.writestr(path, content)
        archive.writestr(
            record_name,
            "\n".join(
                [
                    *(record(path, content) for path, content in members.items()),
                    f"{record_name},,",
                ]
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
        xadx_data=b"different-xadx",
    )
    with pytest.raises(
        candidate.CandidateError,
        match=r"\$\.xadx_provenance\.executables\[0\]\.xadx_sha256",
    ):
        candidate.verify_qualification_wheel_parity(canonical, different)
    different_oracle = _write_wheel(
        tmp_path / "different-oracle",
        QUALIFICATION_DISTRIBUTION,
        "rocm_npu_qualification",
        oracle_sha="6" * 64,
    )
    with pytest.raises(
        candidate.CandidateError,
        match=r"\$\.xadx_provenance\.manifest_sha256",
    ):
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


def test_recipe_provenance_binds_source_xclbin_bytes(tmp_path):
    payload = b"source xclbin"
    digest = hashlib.sha256(payload).hexdigest()
    wheel = tmp_path / "fastflowlm.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "rocm_npu_fastflowlm/share/flm/xclbins/Family/mm.xclbin", payload
        )
    contract = {
        "xadx_provenance": {
            "recipe_sha256": "0" * 64,
            "executables": [
                {
                    "id": f"mm-{digest[:16]}",
                    "xclbin_path": "Family/mm.xclbin",
                    "xclbin_sha256": digest,
                    "transaction_sha256": "1" * 64,
                    "xadx_sha256": "2" * 64,
                }
            ],
        }
    }
    candidate._verify_recipe_source_xclbins(contract, [wheel])
    contract["xadx_provenance"]["executables"][0]["xclbin_sha256"] = "f" * 64
    with pytest.raises(candidate.CandidateError, match="differs from recipe"):
        candidate._verify_recipe_source_xclbins(contract, [wheel])


def test_recipe_provenance_refuses_missing_source_xclbin(tmp_path):
    wheel = tmp_path / "fastflowlm.whl"
    with zipfile.ZipFile(wheel, "w"):
        pass
    contract = {
        "xadx_provenance": {
            "recipe_sha256": "0" * 64,
            "executables": [
                {
                    "id": "mm-0000000000000000",
                    "xclbin_path": "Family/mm.xclbin",
                    "xclbin_sha256": "0" * 64,
                    "transaction_sha256": "1" * 64,
                    "xadx_sha256": "2" * 64,
                }
            ],
        }
    }
    with pytest.raises(candidate.CandidateError, match="exactly one source xclbin"):
        candidate._verify_recipe_source_xclbins(contract, [wheel])


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
    assert candidate_manifest["numeric_runtime"] == {
        "numpy": "2.4.6",
        "ml-dtypes": "0.6.0",
    }
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


def test_candidate_does_not_require_numeric_wheels_in_release_payload(tmp_path):
    _write_wheel(tmp_path, RUNTIME_DISTRIBUTION, "rocm_npu_runtime")
    _write_wheel(tmp_path, FASTFLOWLM_DISTRIBUTION, "rocm_npu_fastflowlm")
    _write_wheel(tmp_path, QUALIFICATION_DISTRIBUTION, "rocm_npu_qualification")
    output = candidate.write_candidate_manifest(
        [tmp_path], tmp_path / "candidate-manifest.json"
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["numeric_runtime"] == {
        "numpy": "2.4.6",
        "ml-dtypes": "0.6.0",
    }


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
    with pytest.raises(cli.QualificationError, match="v1 schema"):
        installation.normalize_candidate(manifest)


def test_legacy_candidate_shape_is_rejected():
    with pytest.raises(cli.QualificationError, match="v1 schema"):
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


def test_installed_numeric_record_trees_are_evidence_bound(monkeypatch, tmp_path):
    installed_root = tmp_path / "installed"
    distributions = {}

    class InstalledDistribution:
        def __init__(self, root, distribution, version):
            self.root = root
            self.version = version
            normalized = distribution.replace("-", "_")
            self.dist_info = root / f"{normalized}-{version}.dist-info"

        def read_text(self, name):
            path = self.dist_info / name
            return path.read_text(encoding="utf-8") if path.is_file() else None

        def locate_file(self, member):
            return self.root / str(member)

        @property
        def files(self):
            rows = [
                row
                for row in installation.csv.reader(
                    self.read_text("RECORD").splitlines()
                )
                if row
            ]
            result = []
            for row in rows:
                item = installation.metadata.PackagePath(row[0])
                item.dist = self
                item.hash = installation.metadata.FileHash(row[1]) if row[1] else None
                item.size = int(row[2]) if row[2] else None
                result.append(item)
            return result

    for distribution, version in (("numpy", "2.4.6"), ("ml-dtypes", "0.6.0")):
        wheel = _write_numeric_wheel(tmp_path, distribution, version)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(installed_root)
        distributions[distribution] = InstalledDistribution(
            installed_root, distribution, version
        )
    monkeypatch.setattr(
        installation.metadata,
        "distribution",
        lambda name: distributions[name],
    )
    identities = installation.numeric_runtime_identities()
    assert {
        distribution: identity["version"]
        for distribution, identity in identities.items()
    } == {"numpy": "2.4.6", "ml-dtypes": "0.6.0"}
    manifest = {"numeric_runtime": {"numpy": "2.4.6", "ml-dtypes": "0.6.0"}}
    installation.verify_numeric_runtime(manifest)

    numpy_payload = installed_root / "numpy/__init__.py"
    original_payload = numpy_payload.read_bytes()
    numpy_payload.write_bytes(b"tampered")
    with pytest.raises(cli.QualificationError, match="payload hash"):
        installation.verify_numeric_runtime(manifest)
    numpy_payload.write_bytes(original_payload)

    sibling_extra = installed_root / "numpy.libs/injected.dll"
    sibling_extra.write_bytes(b"injected")
    with pytest.raises(cli.QualificationError, match="unrecorded file"):
        installation.verify_numeric_runtime(manifest)
    sibling_extra.unlink()

    numpy_record = installed_root / "numpy-2.4.6.dist-info/RECORD"
    numpy_record.write_text(
        numpy_record.read_text(encoding="utf-8") + "injected.pth,,\n",
        encoding="utf-8",
    )
    with pytest.raises(cli.QualificationError, match="unbound file"):
        installation.verify_numeric_runtime(manifest)
