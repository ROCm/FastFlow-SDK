# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate qualification wheel contents, metadata, and bundled licenses."""

from __future__ import annotations

import email.parser
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

from build_tools.fastflow_sdk_qualification_wheel import build_qualification_wheel
from build_tools.fastflow_sdk_wheel_common import (
    FASTFLOWLM_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
)
from build_tools.fastflowlm_xadx_assets import verify_kernel_recipe_assets
from fastflow_sdk_qualification.candidate import canonical_candidate_bytes
from fastflow_sdk_qualification.numeric_runtime import NUMERIC_RUNTIME
from tests.support.qualification import (
    make_candidate,
    make_evidence,
    make_policy,
    write_candidate_wheel,
)

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]

ROOT = Path(__file__).resolve().parents[2]


def _repack_installed_wheel(distribution: str, output: Path) -> Path:
    """Create an offline wheel from the test environment's recorded install."""
    installed = importlib.metadata.distribution(distribution)
    wheel_metadata = installed.read_text("WHEEL")
    assert wheel_metadata is not None
    tag = email.parser.Parser().parsestr(wheel_metadata)["Tag"]
    normalized = distribution.replace("-", "_")
    wheel = output / f"{normalized}-{installed.version}-{tag}.whl"
    with zipfile.ZipFile(
        wheel, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for relative in installed.files or ():
            if ".." in Path(relative).parts:
                continue
            source = Path(installed.locate_file(relative))
            if source.is_file():
                archive.write(source, Path(relative).as_posix())
    return wheel


def _fake_xadx_assets(root: Path) -> Path:
    root.mkdir()
    (root / "xadx").mkdir()
    (root / "transactions").mkdir()
    transaction = root / "transactions/kernel.bin"
    transaction.write_bytes(b"transaction")
    transaction_sha256 = hashlib.sha256(transaction.read_bytes()).hexdigest()
    executables = []
    for index in range(8):
        kernel = "dequant" if index < 5 else "mm"
        source_digest = f"{index + 1:016x}" + "0" * 48
        id = f"{kernel}-{source_digest[:16]}"
        relative = f"xadx/{id}.xadx"
        path = root / relative
        path.write_bytes(b"\0\0\0\0XADX" + relative.encode())
        executables.append(
            {
                "id": id,
                "kernel": kernel,
                "builder": "q4_1" if kernel == "dequant" else "gemm",
                "shape": {"m": 1, "k": 1, "n": 1},
                "xclbin_sha256": source_digest,
                "xclbin_path": f"Family-{index}/{kernel}.xclbin",
                "transaction": "transactions/kernel.bin",
                "transaction_sha256": transaction_sha256,
                "xadx": relative,
                "xadx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "export": "MLIR_AIE",
                "families": [f"Family-{index}"],
            }
        )
    zero = {
        "id": "zero",
        "parameters": {"size": 2},
        "sha256": hashlib.sha256(bytes(2)).hexdigest(),
    }
    cases = [
        {
            "name": f"case-{index:02d}",
            "stages": [
                {
                    "name": "kernel",
                    "implementation": executables[index % len(executables)]["id"],
                    "buffers": [
                        {
                            "name": "output",
                            "direction": "output",
                            "size": 2,
                            "expected": dict(zero),
                            "comparison": {"kind": "exact"},
                        },
                        {
                            "name": "input",
                            "direction": "input",
                            "size": 2,
                            "source": dict(zero),
                        },
                    ],
                }
            ],
        }
        for index in range(len(executables))
    ]
    recipe = {
        "schema_version": 1,
        "product": "fastflow-sdk-fastflowlm-kernel-recipe",
        "target": "npu2",
        "generator_version": "kernel-recipe-generator-v1",
        "implementations": executables,
        "coverage_limits": [
            {
                "id": f"mm-{index + 20:016x}",
                "scope": "dispatch",
                "kernel": "mm",
                "representative": f"Limited-{index}",
                "xclbin_sha256": f"{index + 20:016x}" + "0" * 48,
                "families": [f"Limited-{index}"],
                "reason": "no supported builder",
            }
            for index in range(4)
        ],
        "cases": cases,
    }
    recipe_path = root / "recipe.json"
    recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
    capability = root / "capability-manifest.json"
    capability.write_text(
        json.dumps(
            {
                "implementations": {
                    "entries": {entry["id"]: {} for entry in executables},
                    "unreached": {
                        entry["id"]: {} for entry in recipe["coverage_limits"]
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    def record(path):
        return {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    records = [record(path) for path in sorted(root.rglob("*")) if path.is_file()]
    manifest = {
        "schema_version": 1,
        "product": "fastflow-sdk-fastflowlm-xadx",
        "target": "npu2",
        "serializer": {
            "api": "hrx_amdxdna_xadx_serialize",
            "revision": "hrx-amdxdna-v0",
        },
        "recipe": record(recipe_path),
        "capabilities": record(capability),
        "files": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_asset_verification_cross_checks_recipe_transaction_identity(tmp_path):
    assets = _fake_xadx_assets(tmp_path / "xadx")
    transaction = assets / "transactions/kernel.bin"
    transaction.write_bytes(b"different transaction")
    manifest_path = assets / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        record
        for record in manifest["files"]
        if record["path"] == "transactions/kernel.bin"
    )
    record["size"] = transaction.stat().st_size
    record["sha256"] = hashlib.sha256(transaction.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="recipe transaction digest differs"):
        verify_kernel_recipe_assets(assets)


def test_qualification_wheel_is_pure_and_owns_only_plugin_entrypoint(
    monkeypatch, tmp_path
):
    assets = _fake_xadx_assets(tmp_path / "xadx")
    upstream_licenses = {
        ROOT / "projects/FastFlowLM/LICENSE_RUNTIME.txt",
        ROOT / "projects/FastFlowLM/LICENSE_BINARY.txt",
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
    assert wheel.name == "fastflow_sdk_qualification-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        entry_points = archive.read(
            "fastflow_sdk_qualification-1.2.3.dist-info/entry_points.txt"
        ).decode()
        metadata_text = archive.read(
            "fastflow_sdk_qualification-1.2.3.dist-info/METADATA"
        ).decode()
        package_manifest = json.loads(
            archive.read("fastflow_sdk_qualification/manifest.json")
        )
        sbom = json.loads(
            archive.read(
                "fastflow_sdk_qualification-1.2.3.dist-info/sboms/cyclonedx.json"
            )
        )
        names = set(archive.namelist())
        installed = tmp_path / "installed"
        archive.extractall(installed)
    assert "fastflow-qualify = fastflow_sdk_qualification.cli:main" in entry_points
    assert "\nfastflow =" not in entry_points
    assert "Requires-Dist: fastflow-sdk-fastflowlm ==1.2.3" in metadata_text
    assert "Requires-Dist: numpy ==2.4.6" in metadata_text
    assert "Requires-Dist: ml-dtypes ==0.6.0" in metadata_text
    assert "Requires-Python: >=3.11,<3.15" in metadata_text
    assert (
        "License-Expression: MIT AND LicenseRef-FastFlowLM-Proprietary-Kernel-Terms"
    ) in metadata_text
    assert "fastflow_sdk_qualification/xadx_assets/manifest.json" in names
    assert "fastflow_sdk_qualification/xadx_assets/recipe.json" in names
    assert "fastflow_sdk_qualification/recipe_generators.py" in names
    assert not any(name.endswith(".xclbin") for name in names)
    assert not any(
        "/xadx_assets/data/" in name and name.endswith(".bin") for name in names
    )
    assert "fastflow_sdk_qualification/candidate.py" in names
    assert "fastflow_sdk_qualification/offline.py" in names
    assert (
        "fastflow_sdk_qualification/schemas/qualification-index-v1.schema.json" in names
    )
    assert "fastflow_sdk_qualification/schemas/candidate-v1.schema.json" in names
    assert (
        "fastflow_sdk_qualification/schemas/qualification-contract-v1.schema.json"
        in names
    )
    assert "fastflow_sdk_qualification/schemas/kernel-recipe-v1.schema.json" in names
    assert "fastflow_sdk_qualification/schemas/xadx-report-v1.schema.json" in names
    assert "fastflow_sdk_qualification/bench/bench_1k_smoke.json" in names
    assert "fastflow_sdk_qualification/bench/bench_4k_smoke.json" in names
    assert "fastflow_sdk_qualification/bench/bench_8k_smoke.json" in names
    assert "fastflow_sdk_qualification/licenses/FastFlowLM-LICENSE_RUNTIME.txt" in names
    assert "fastflow_sdk_qualification/licenses/FastFlowLM-LICENSE_BINARY.txt" in names
    # The reference transcript is vendored in git, so every wheel must carry it;
    # the paired audio clip is copied from the FastFlowLM checkout, so it must be
    # present whenever the submodule is (a pin bump that drops it fails the build
    # via _stage_asr_sample).
    assert "fastflow_sdk_qualification/assets/asr_sample.reference.txt" in names
    if (ROOT / "projects/FastFlowLM/src").is_dir():
        assert "fastflow_sdk_qualification/assets/asr_sample.wav" in names
    provenance = package_manifest["embedded_binaries"]["fastflowlm_xadx"]
    assert provenance["target"] == "npu2"
    assert (
        provenance["recipe_sha256"]
        == hashlib.sha256((assets / "recipe.json").read_bytes()).hexdigest()
    )
    assert len(provenance["executables"]) == len(
        json.loads((assets / "recipe.json").read_text(encoding="utf-8"))[
            "implementations"
        ]
    )
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
                "fastflow_sdk_qualification.cli",
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
        ROOT / "projects/FastFlowLM/LICENSE_BINARY.txt",
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
    staged_products = tmp_path / "staged-products"
    product_wheels = [
        write_candidate_wheel(
            staged_products,
            RUNTIME_DISTRIBUTION,
            "fastflow_sdk_runtime",
        ),
        write_candidate_wheel(
            staged_products,
            FASTFLOWLM_DISTRIBUTION,
            "fastflow_sdk_fastflowlm",
        ),
    ]
    environment_root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, clear=True, system_site_packages=False).create(
        environment_root
    )
    python = environment_root / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    numeric_wheels = [
        _repack_installed_wheel(distribution, tmp_path)
        for distribution in NUMERIC_RUNTIME
    ]
    installation = subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            tmp_path,
            "--no-deps",
            wheel,
            *product_wheels,
            *numeric_wheels,
        ],
        capture_output=True,
        text=True,
    )
    assert installation.returncode == 0, installation.stderr
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    resource_script = r"""
import json
from importlib import resources
from fastflow_sdk_qualification import xadx

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

package = resources.files("fastflow_sdk_qualification")
assert package.joinpath("bench", "bench_1k_smoke.json").is_file()
assert package.joinpath("bench", "bench_4k_smoke.json").is_file()
assert package.joinpath("bench", "bench_8k_smoke.json").is_file()
with resources.as_file(package.joinpath("xadx_assets")) as root:
    recipe = json.loads((root / "recipe.json").read_text())
    report = xadx.run_recipe(
        recipe, root, runtime_factory=Runtime,
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

    wheels = tmp_path / "candidate-wheels"
    write_candidate_wheel(wheels, RUNTIME_DISTRIBUTION, "fastflow_sdk_runtime")
    write_candidate_wheel(wheels, FASTFLOWLM_DISTRIBUTION, "fastflow_sdk_fastflowlm")
    write_candidate_wheel(
        wheels,
        QUALIFICATION_DISTRIBUTION,
        "fastflow_sdk_qualification",
    )

    def run_cli(*arguments: object) -> dict[str, object]:
        result = subprocess.run(
            [
                python,
                "-m",
                "fastflow_sdk_qualification.cli",
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

    evidence = make_evidence(tmp_path / "evidence")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    bundle = incoming / "run.evidence.zip"
    run_cli("pack", "--input", evidence, "--output", bundle)
    run_cli("verify", "--input", bundle)
    canonical_root = tmp_path / "canonical"
    canonical_root.mkdir()
    canonical_candidate = canonical_root / "candidate.json"
    canonical_candidate.write_bytes(canonical_candidate_bytes(make_candidate()))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(make_policy("windows")), encoding="utf-8")
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
    snapshots = ROOT / "tools/fastflow-sdk-qualification/licenses"
    upstream = ROOT / "projects/FastFlowLM"

    def normalized(path):
        return [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]

    assert normalized(snapshots / "FastFlowLM-LICENSE_RUNTIME.txt") == normalized(
        upstream / "LICENSE_RUNTIME.txt"
    )
    assert normalized(snapshots / "FastFlowLM-LICENSE_BINARY.txt") == normalized(
        upstream / "LICENSE_BINARY.txt"
    )


def test_stage_asr_sample_fails_when_checkout_present_but_clip_missing(tmp_path):
    from build_tools.fastflow_sdk_qualification_wheel import _stage_asr_sample

    root = tmp_path / "root"
    (root / "projects/FastFlowLM/src").mkdir(parents=True)
    package_root = root / "package"
    (package_root / "assets").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="pinned ASR fixture"):
        _stage_asr_sample(root, package_root)


def test_stage_asr_sample_omits_clip_when_checkout_absent(tmp_path):
    from build_tools.fastflow_sdk_qualification_wheel import _stage_asr_sample

    root = tmp_path / "root"
    package_root = root / "package"
    (package_root / "assets").mkdir(parents=True)
    _stage_asr_sample(root, package_root)
    assert not (package_root / "assets" / "asr_sample.wav").exists()


@pytest.mark.skipif(
    not (ROOT / "projects/FastFlowLM/src/tb_files/Recording.wav").is_file(),
    reason="FastFlowLM submodule is not checked out",
)
def test_qualification_wheel_vendors_pinned_asr_sample(tmp_path):
    from build_tools.fastflow_sdk_qualification_wheel import ASR_SAMPLE_SHA256

    assets = _fake_xadx_assets(tmp_path / "xadx")
    wheel = build_qualification_wheel(
        ROOT,
        output_directory=tmp_path,
        xadx_assets=assets,
        version="1.2.3",
    )
    member = "fastflow_sdk_qualification/assets/asr_sample.wav"
    with zipfile.ZipFile(wheel) as archive:
        assert member in set(archive.namelist())
        digest = hashlib.sha256(archive.read(member)).hexdigest()
    # The clip is copied from the pinned FastFlowLM submodule, hash-verified.
    assert digest == ASR_SAMPLE_SHA256
