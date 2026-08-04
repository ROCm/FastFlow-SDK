# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate packaged XADX assets and their compiler-free execution helpers."""

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

from build_tools import fastflowlm_kernel_tests as kernels

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tools/rocm-npu-qualification/python"))

from rocm_npu_qualification import xadx  # noqa: E402


def test_authoritative_core_runtime_is_used_and_fails_closed(monkeypatch):
    attempts = []

    def fail_load(path, **kwargs):
        attempts.append(path)
        raise OSError("not loadable")

    monkeypatch.setattr(xadx.ctypes, "CDLL", fail_load)
    selected = str(Path("candidate") / "hrx.dll")
    with pytest.raises(xadx.XadxError, match="ROCM_NPU_RUNTIME_PATH"):
        xadx._load_library(
            {
                "ROCM_NPU_RUNTIME_PATH": selected,
                "HRX_LIBHRX": "unrelated-hrx.dll",
            }
        )
    assert attempts == [selected]


def _record(root, relative):
    path = root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _synthetic_package(root, architecture="npu2"):
    (root / "executables").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "executables/test.xadx").write_bytes(b"\0\0\0\0XADXsynthetic")
    (root / "data/input.bin").write_bytes(b"\x12\x34")
    (root / "data/expected.bin").write_bytes(b"\x12\x34")
    (root / "data/input-b.bin").write_bytes(b"\x56\x78")
    (root / "data/expected-b.bin").write_bytes(b"\x56\x78")
    files = [
        _record(root, relative)
        for relative in (
            "data/expected.bin",
            "data/expected-b.bin",
            "data/input.bin",
            "data/input-b.bin",
            "executables/test.xadx",
        )
    ]
    manifest = {
        "schema_version": 1,
        "product": "rocm-npu-fastflowlm-xadx",
        "target": architecture,
        "serializer": {
            "api": "hrx_amdxdna_xadx_serialize",
            "revision": "hrx-amdxdna-v0",
        },
        "case_contract_sha256": "0" * 64,
        "files": files,
        "cases": [
            {
                "name": "synthetic/exact",
                "xadx": "executables/test.xadx",
                "export": "MLIR_AIE",
                "buffers": [
                    {
                        "name": "output",
                        "direction": "output",
                        "size": 2,
                        "expected": "data/expected.bin",
                        "comparison": "exact",
                    },
                    {
                        "name": "input",
                        "direction": "input",
                        "size": 2,
                        "input": "data/input.bin",
                    },
                ],
                "readback_variants": [
                    {
                        "buffer": "input",
                        "input": "data/input.bin",
                        "expected": "data/expected.bin",
                    },
                    {
                        "buffer": "input",
                        "input": "data/input-b.bin",
                        "expected": "data/expected-b.bin",
                    },
                ],
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class _FakeRuntime:
    architecture_name = "AMD_AIE2_NPU2"

    def __init__(self):
        self.resources = []
        self.closed = False
        self._flush = lambda *args: None

    def _check(self, status, operation):
        assert status is None

    def architecture(self):
        return self.architecture_name

    def load(self, data, export):
        assert data[4:8] == b"XADX"
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
        self.closed = True
        self.resources.clear()


def test_runner_is_no_numpy_and_runs_bounded_exact_case(tmp_path):
    _synthetic_package(tmp_path)
    source = Path(xadx.__file__).read_text(encoding="utf-8")
    assert "import numpy" not in source.lower()
    report = xadx.run_package(
        tmp_path,
        runtime_factory=_FakeRuntime,
        lifecycle_rounds=1,
        resident_rounds=2,
        readback_rounds=0,
    )
    assert report["status"] == "pass"
    assert report["materialized_buffers"]
    with pytest.raises(ValueError, match="safety bound"):
        xadx.run_package(tmp_path, runtime_factory=_FakeRuntime, resident_rounds=1001)


@pytest.mark.parametrize(
    "architecture", ("", "unknown-future-device", "AMD_AIE2_NPU1")
)
def test_runner_reports_unrecognized_architecture_as_unsupported(
    tmp_path, architecture
):
    _synthetic_package(tmp_path)

    class Unsupported(_FakeRuntime):
        architecture_name = architecture

    report = xadx.run_package(tmp_path, runtime_factory=Unsupported)
    assert report["status"] == "unsupported"
    assert architecture in report["reason"]


def test_runner_accepts_xdna2_architecture(tmp_path):
    _synthetic_package(tmp_path)

    class Xdna2(_FakeRuntime):
        architecture_name = "AMD XDNA2 accelerator"

    assert xadx.run_package(
        tmp_path,
        runtime_factory=Xdna2,
        lifecycle_rounds=1,
        resident_rounds=0,
        readback_rounds=0,
    )["status"] == "pass"


def test_runner_binds_generic_amdxdna_to_probed_npu2_family(tmp_path):
    _synthetic_package(tmp_path)

    class Amdxdna(_FakeRuntime):
        architecture_name = "amdxdna"

    assert (
        xadx.run_package(
            tmp_path,
            runtime_factory=Amdxdna,
            lifecycle_rounds=1,
            resident_rounds=0,
            readback_rounds=0,
        )["status"]
        == "unsupported"
    )
    assert (
        xadx.run_package(
            tmp_path,
            runtime_factory=Amdxdna,
            lifecycle_rounds=1,
            resident_rounds=0,
            readback_rounds=0,
            target_family="npu2",
        )["status"]
        == "pass"
    )
    assert (
        xadx.run_package(
            tmp_path,
            runtime_factory=Amdxdna,
            lifecycle_rounds=1,
            resident_rounds=0,
            readback_rounds=0,
            target_family="npu1",
        )["status"]
        == "unsupported"
    )


def test_runner_closes_runtime_when_generator_setup_fails(tmp_path, monkeypatch):
    _synthetic_package(tmp_path)
    runtime = _FakeRuntime()

    def fail_generator():
        raise xadx.XadxError("generator setup failed")

    monkeypatch.setattr(xadx, "_generator_module", fail_generator)
    with pytest.raises(xadx.XadxError, match="generator setup failed"):
        xadx.run_package(tmp_path, runtime_factory=lambda: runtime)
    assert runtime.closed


def test_runner_detects_inventory_corruption_before_hrx(tmp_path):
    _synthetic_package(tmp_path)
    (tmp_path / "data/input.bin").write_bytes(b"tampered")
    with pytest.raises(xadx.XadxError, match="corruption"):
        xadx.run_package(tmp_path, runtime_factory=_FakeRuntime)


def test_exact_and_bf16_contracts_reject_adversarial_values():
    assert xadx._compare(b"abc", b"abc", "exact", 0, 0) is None
    assert "exact mismatch" in xadx._compare(b"abc", b"axc", "exact", 0, 0)
    one = struct.pack("<H", 0x3F80)
    near = struct.pack("<H", 0x3F81)
    nan = struct.pack("<H", 0x7FC0)
    assert xadx._compare(near, one, "bf16-tolerance", 0.01, 0) is None
    assert "BF16 mismatches" in xadx._compare(nan, one, "bf16-tolerance", 1e9, 1e9)


def test_asset_verifier_rejects_extra_and_corrupt_files(tmp_path):
    _synthetic_package(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["executables"] = [
        {
            "id": "synthetic",
            "xadx": "executables/test.xadx",
            "xadx_size": (tmp_path / "executables/test.xadx").stat().st_size,
            "xadx_sha256": hashlib.sha256(
                (tmp_path / "executables/test.xadx").read_bytes()
            ).hexdigest(),
            "source_xclbin_sha256": "1" * 64,
            "transaction_sha256": "2" * 64,
        }
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert kernels.verify_xadx_assets(tmp_path)["target"] == "npu2"
    (tmp_path / "extra").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="inventory"):
        kernels.verify_xadx_assets(tmp_path)


def test_runtime_generator_parity_rejects_cpp_drift(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    specifications = {}
    for index, name in enumerate(
        ("mm-weight.bin", "mm-activation-a.bin", "mm-activation-b.bin"), start=1
    ):
        payload = bytes(index)
        (data / name).write_bytes(payload)
        specifications[f"data/{name}"] = {
            "kind": "generator",
            "id": "zero",
            "parameters": {"size": index},
        }

    kernels._assert_runtime_generator_parity(tmp_path, specifications)
    (data / "mm-activation-b.bin").write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="mm-activation-b.bin"):
        kernels._assert_runtime_generator_parity(tmp_path, specifications)


def test_generation_records_source_transaction_and_serializer_identity(
    tmp_path, monkeypatch
):
    build = tmp_path / "build"
    build.mkdir()
    generator = build / kernels.paths.executable_name(kernels.XADX_GENERATOR_TARGET)
    generator.write_bytes(b"mock generator")
    xclbins = tmp_path / "xclbins"
    records = []
    for family in ("Qwen3", "LFM2"):
        directory = xclbins / family
        directory.mkdir(parents=True)
        for kernel in ("dequant", "mm"):
            source = directory / f"{kernel}.xclbin"
            source.write_bytes(f"{family}-{kernel}".encode())
            records.append(
                {
                    "id": f"{family}-{kernel}",
                    "family": family,
                    "kernel": kernel,
                    "source_xclbin": f"{family}/{kernel}.xclbin",
                    "xadx": f"executables/{family}-{kernel}.xadx",
                }
            )

    def fake_generator(command, env):
        destination = Path(command[-1])
        (destination / "executables").mkdir(parents=True)
        (destination / "data").mkdir()
        (destination / ".transactions").mkdir()
        (destination / ".transactions/dequant.bin").write_bytes(b"dequant txn")
        (destination / ".transactions/mm.bin").write_bytes(b"mm txn")
        for record in records:
            (destination / record["xadx"]).write_bytes(
                b"\0\0\0\0XADX" + record["id"].encode()
            )
        for name, data in {
            "dequant-weight-zero.bin": b"\0",
            "dequant-parameter-zero.bin": bytes(4096),
            "dequant-expected-zero.bin": b"\0\0",
            "mm-weight.bin": bytes(4098),
            "mm-activation-a.bin": b"\0\0",
            "mm-activation-b.bin": b"\0\0",
            "mm-expected-a.bin": b"\0\0",
            "mm-expected-b.bin": b"\0\0",
        }.items():
            (destination / "data" / name).write_bytes(data)
        (destination / ".generation.json").write_text(
            json.dumps(
                {
                    "serializer_api": "hrx_amdxdna_xadx_serialize",
                    "serializer_revision": "hrx-amdxdna-v0",
                    "transactions": {
                        "dequant": ".transactions/dequant.bin",
                        "mm": ".transactions/mm.bin",
                    },
                    "executables": records,
                    "shape": {"m": 1, "k": 1, "n": 1},
                    "tolerances": {"atol": 0.0625, "rtol": 0.01},
                }
            )
        )

    monkeypatch.setattr(kernels, "BUILD", build)
    case_contract = tmp_path / "xadx_cases.json"
    case_contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shape": {"m": 1, "k": 1, "n": 1},
                "families": ["Qwen3", "LFM2"],
                "case_suffixes": [
                    "dequant/zero-baseline",
                    "mm/structured",
                ],
                "global_cases": ["host-readback/coherence"],
            }
        )
    )
    monkeypatch.setattr(kernels, "XADX_CASES_SOURCE", case_contract)
    monkeypatch.setattr(kernels.environment, "sh", fake_generator)
    monkeypatch.setattr(kernels, "_assert_runtime_generator_parity", lambda *args: None)
    destination = kernels.generate_xadx_assets(xclbins, tmp_path / "package")
    manifest = kernels.verify_xadx_assets(destination)
    assert len(manifest["executables"]) == 4
    assert len(manifest["cases"]) == 5
    assert manifest["cases"][-1]["name"] == "host-readback/coherence"
    assert all(record["source_xclbin_sha256"] for record in manifest["executables"])
    assert all(record["transaction_sha256"] for record in manifest["executables"])
    assert {path.suffix for path in (destination / "data").iterdir()} == {".gz"}
    assert all(
        isinstance(buffer.get("input", buffer.get("expected")), dict)
        for case in manifest["cases"]
        for buffer in case["buffers"]
    )
    assert not (destination / ".transactions").exists()
