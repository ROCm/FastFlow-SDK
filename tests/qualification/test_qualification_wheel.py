# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate qualification wheel contents, metadata, and bundled licenses."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

from build_tools.rocm_npu_qualification_wheel import build_qualification_wheel


pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]

ROOT = Path(__file__).resolve().parents[2]


def _load_test_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_xadx_assets(root: Path) -> Path:
    root.mkdir()
    (root / "executables").mkdir()
    records = []
    executables = []
    for family in ("Qwen3-0.6B-NPU2", "LFM2-1.2B-NPU2"):
        for kernel in ("dequant", "mm"):
            relative = f"executables/{family}-{kernel}.xadx"
            path = root / relative
            path.write_bytes(b"\0\0\0\0XADX" + relative.encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(
                {"path": relative, "size": path.stat().st_size, "sha256": digest}
            )
            executables.append(
                {
                    "id": f"{family}-{kernel}",
                    "family": family,
                    "kernel": kernel,
                    "source_xclbin": f"{family}/{kernel}.xclbin",
                    "xadx": relative,
                    "source_xclbin_size": 1,
                    "source_xclbin_sha256": "1" * 64,
                    "transaction_size": 1,
                    "transaction_sha256": "2" * 64,
                    "xadx_size": path.stat().st_size,
                    "xadx_sha256": digest,
                }
            )
    manifest = {
        "schema_version": 1,
        "product": "rocm-npu-fastflowlm-xadx",
        "target": "npu2",
        "case_contract_sha256": "0" * 64,
        "serializer": {
            "api": "hrx_amdxdna_xadx_serialize",
            "revision": "hrx-amdxdna-v0",
        },
        "executables": executables,
        "cases": [
            {
                "name": name,
                "xadx": executables[index % len(executables)]["xadx"],
                "export": "MLIR_AIE",
                "buffers": [
                    {
                        "name": "output",
                        "direction": "output",
                        "size": 2,
                        "expected": {
                            "kind": "generator",
                            "id": "zero",
                            "parameters": {"size": 2},
                        },
                        "comparison": "exact",
                    },
                    {
                        "name": "input",
                        "direction": "input",
                        "size": 2,
                        "input": {
                            "kind": "generator",
                            "id": "zero",
                            "parameters": {"size": 2},
                        },
                    },
                ],
                "readback_variants": (
                    [
                        {
                            "buffer": "input",
                            "input": {
                                "kind": "generator",
                                "id": "zero",
                                "parameters": {"size": 2},
                            },
                            "expected": {
                                "kind": "generator",
                                "id": "zero",
                                "parameters": {"size": 2},
                            },
                        }
                    ]
                    if name == "host-readback/coherence"
                    else []
                ),
            }
            for index, name in enumerate(
                (
                    "Qwen3-0.6B-NPU2/dequant/zero-baseline",
                    "Qwen3-0.6B-NPU2/mm/structured",
                    "LFM2-1.2B-NPU2/dequant/zero-baseline",
                    "LFM2-1.2B-NPU2/mm/structured",
                    "host-readback/coherence",
                )
            )
        ],
        "files": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_qualification_wheel_is_pure_and_owns_only_plugin_entrypoint(
    monkeypatch, tmp_path
):
    assets = _fake_xadx_assets(tmp_path / "xadx")
    upstream_licenses = {
        ROOT / "projects/FastFlowLM/LICENSE_RUNTIME.txt",
        ROOT / "projects/FastFlowLM/TERMS.md",
    }
    path_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: False if path in upstream_licenses else path_is_file(path),
    )
    wheel = build_qualification_wheel(
        ROOT,
        output_directory=tmp_path,
        xadx_assets=assets,
        version="1.2.3",
    )
    assert wheel.name == "rocm_npu_qualification-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        entry_points = archive.read(
            "rocm_npu_qualification-1.2.3.dist-info/entry_points.txt"
        ).decode()
        metadata_text = archive.read(
            "rocm_npu_qualification-1.2.3.dist-info/METADATA"
        ).decode()
        package_manifest = json.loads(
            archive.read("rocm_npu_qualification/manifest.json")
        )
        sbom = json.loads(
            archive.read("rocm_npu_qualification-1.2.3.dist-info/sboms/cyclonedx.json")
        )
        names = set(archive.namelist())
        installed = tmp_path / "installed"
        archive.extractall(installed)
    assert "rocm-npu-qualify = rocm_npu_qualification.cli:main" in entry_points
    assert "\nrocm-npu =" not in entry_points
    assert "Requires-Dist: rocm-npu-fastflowlm ==1.2.3" in metadata_text
    assert (
        "License-Expression: MIT AND LicenseRef-FastFlowLM-Proprietary-Kernel-Terms"
    ) in metadata_text
    assert "rocm_npu_qualification/xadx_assets/manifest.json" in names
    assert "rocm_npu_qualification/xadx_cases.json" in names
    assert "rocm_npu_qualification/xadx_generators.py" in names
    assert not any(name.endswith(".xclbin") for name in names)
    assert not any(
        "/xadx_assets/data/" in name and name.endswith(".bin") for name in names
    )
    assert "rocm_npu_qualification/candidate.py" in names
    assert "rocm_npu_qualification/offline.py" in names
    assert "rocm_npu_qualification/schemas/qualification-index-v1.schema.json" in names
    assert "rocm_npu_qualification/bench/bench_1k_smoke.json" in names
    assert "rocm_npu_qualification/bench/bench_4k_smoke.json" in names
    assert "rocm_npu_qualification/licenses/FastFlowLM-LICENSE_RUNTIME.txt" in names
    assert "rocm_npu_qualification/licenses/FastFlowLM-TERMS.md" in names
    provenance = package_manifest["embedded_binaries"]["fastflowlm_xadx"]
    assert provenance["target"] == "npu2"
    assert provenance["case_contract_sha256"] == "0" * 64
    assert len(provenance["executables"]) == 4
    assert sbom["metadata"]["component"]["licenses"] == [
        {"expression": ("MIT AND LicenseRef-FastFlowLM-Proprietary-Kernel-Terms")}
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    for command in ("candidate", "pack", "verify", "intake"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rocm_npu_qualification.cli",
                command,
                "--help",
                "--json",
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == "pass"


def test_installed_wheel_executes_offline_commands_and_packaged_resources(
    monkeypatch, tmp_path
):
    assets = _fake_xadx_assets(tmp_path / "xadx")
    upstream_licenses = {
        ROOT / "projects/FastFlowLM/LICENSE_RUNTIME.txt",
        ROOT / "projects/FastFlowLM/TERMS.md",
    }
    path_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: False if path in upstream_licenses else path_is_file(path),
    )
    wheel = build_qualification_wheel(
        ROOT,
        output_directory=tmp_path,
        xadx_assets=assets,
        version="1.2.3",
    )
    environment_root = tmp_path / "venv"
    venv.EnvBuilder(
        with_pip=True, clear=True, system_site_packages=True
    ).create(environment_root)
    python = environment_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [python, "-m", "pip", "install", "--no-deps", wheel],
        check=True,
        capture_output=True,
        text=True,
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    resource_script = r"""
from importlib import resources
from rocm_npu_qualification import xadx

class Runtime:
    def __init__(self):
        self.resources = []
        self._flush = lambda *args: None
    def _check(self, status, operation):
        assert status is None
    def architecture(self):
        return "AMD XDNA2 NPU"
    def load(self, data, export):
        self.resources.append(("executable", object(), data))
        return object(), 0
    def allocate(self, data):
        result = (object(), bytearray(data))
        self.resources.append(("buffer", *result))
        return result
    def run(self, executable, ordinal, buffers):
        buffers[0][1][:] = buffers[1][1]
    def read(self, buffer):
        return bytes(buffer[1])
    def close(self):
        self.resources.clear()

package = resources.files("rocm_npu_qualification")
assert package.joinpath("bench", "bench_1k_smoke.json").is_file()
assert package.joinpath("bench", "bench_4k_smoke.json").is_file()
with resources.as_file(package.joinpath("xadx_assets")) as root:
    report = xadx.run_package(
        root, runtime_factory=Runtime,
        lifecycle_rounds=1, resident_rounds=1, readback_rounds=1,
    )
assert report["status"] == "pass", report
"""
    subprocess.run(
        [python, "-c", resource_script],
        check=True,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    candidate_tests = _load_test_module(
        "installed_candidate_fixtures",
        ROOT
        / "tools"
        / "rocm-npu-qualification"
        / "tests"
        / "test_candidate_identity.py",
    )
    wheels = tmp_path / "candidate-wheels"
    candidate_tests._write_wheel(
        wheels, candidate_tests.RUNTIME_DISTRIBUTION, "rocm_npu_runtime"
    )
    candidate_tests._write_wheel(
        wheels, candidate_tests.FASTFLOWLM_DISTRIBUTION, "rocm_npu_fastflowlm"
    )
    candidate_tests._write_wheel(
        wheels,
        candidate_tests.QUALIFICATION_DISTRIBUTION,
        "rocm_npu_qualification",
    )

    def run_cli(*arguments: object) -> dict[str, object]:
        result = subprocess.run(
            [
                python,
                "-m",
                "rocm_npu_qualification.cli",
                *(str(argument) for argument in arguments),
                "--json",
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        report = json.loads(result.stdout)
        assert report["status"] == "pass"
        return report

    generated_candidate = tmp_path / "generated-candidate.json"
    run_cli(
        "candidate",
        "--wheel-dir",
        wheels,
        "--output",
        generated_candidate,
    )
    assert generated_candidate.is_file()

    offline_tests = _load_test_module(
        "installed_offline_fixtures",
        ROOT
        / "tools"
        / "rocm-npu-qualification"
        / "tests"
        / "test_offline_intake.py",
    )
    evidence = offline_tests._evidence(tmp_path / "evidence")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    bundle = incoming / "run.evidence.zip"
    run_cli("pack", "--input", evidence, "--output", bundle)
    run_cli("verify", "--input", bundle)
    canonical_candidate = tmp_path / "candidate.json"
    canonical_candidate.write_bytes(
        offline_tests.canonical_candidate_bytes(offline_tests._candidate())
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(offline_tests._policy("windows")), encoding="utf-8"
    )
    run_cli(
        "intake",
        "--input-dir",
        incoming,
        "--candidate-manifest",
        canonical_candidate,
        "--output-dir",
        tmp_path / "intake",
        "--bundle-output",
        tmp_path / "aggregate.zip",
        "--policy",
        policy,
        "--require-status",
        "pass",
    )


@pytest.mark.skipif(
    not (ROOT / "projects/FastFlowLM/LICENSE_RUNTIME.txt").is_file(),
    reason="FastFlowLM submodule is not checked out",
)
def test_qualification_license_snapshots_match_fastflowlm():
    snapshots = ROOT / "tools/rocm-npu-qualification/licenses"
    upstream = ROOT / "projects/FastFlowLM"

    def normalized(path):
        return [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]

    assert normalized(snapshots / "FastFlowLM-LICENSE_RUNTIME.txt") == normalized(
        upstream / "LICENSE_RUNTIME.txt"
    )
    assert normalized(snapshots / "FastFlowLM-TERMS.md") == normalized(
        upstream / "TERMS.md"
    )
