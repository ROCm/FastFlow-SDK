# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Policy tests for the native runtime CLI and its product integration."""

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from build_tools import validate_rocm_npu_cli
from build_tools.validate_rocm_npu_cli import run_reports, validate_report


pytestmark = pytest.mark.presubmit

ROOT = Path(__file__).resolve().parents[2]
REPORT_PRODUCT = {
    "version": "1.0",
    "source_revision": "0123456789abcdef",
    "build_platform": "Linux-x86_64",
}


@pytest.mark.parametrize(
    ("relative", "expected"),
    (
        (
            "third_party/nlohmann_json/include/nlohmann/json.hpp",
            "aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63",
        ),
        (
            "third_party/nlohmann_json/LICENSE.MIT",
            "c0d068392ea65358b798b8c165103560f06e9e3b38c4ab4e2d8810a7b931af86",
        ),
    ),
)
def test_nlohmann_json_vendored_files_match_approved_contents(relative, expected):
    content = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(content).hexdigest() == expected


def test_native_cli_does_not_link_forbidden_product_dependencies():
    cmake = (ROOT / "tools/rocm-npu/CMakeLists.txt").read_text(encoding="utf-8")
    assert "rocm_npu_nlohmann_json" in cmake
    assert "JSON_USE_IMPLICIT_CONVERSIONS=0" in cmake
    assert 'PATTERN "README.md" EXCLUDE' in cmake
    assert 'PATTERN "regenerate-*.sh" EXCLUDE' in cmake
    for forbidden in ("LLVM", "Boost", "FastFlowLM", "Python"):
        assert forbidden not in cmake


def test_native_cli_json_contract_documents_stable_v1_envelope():
    contract = (
        ROOT / "docs/contracts/runtime-cli-json.md"
    ).read_text(encoding="utf-8")
    assert "schema_version" in contract
    assert "command" in contract
    assert "status" in contract
    assert "product" in contract
    assert "checks" in contract
    assert "`data`: optional command-specific object" in contract
    assert "Statuses are `pass`, `fail`, or `error`" in contract
    assert "runtime.library.load" in contract
    assert "runtime.abi" in contract
    assert "self_test.cleanup" in contract
    assert "self_test.output" in contract
    assert "rocm-npu-report-v1.schema.json" not in contract


def test_self_test_fixture_hashes_match_provenance():
    fixture = (
        ROOT
        / "tools"
        / "rocm-npu"
        / "testdata"
        / "self-test"
        / "npu2-vector-scalar-add"
    )
    manifest = json.loads(
        (fixture / "manifest.json").read_text(encoding="utf-8")
    )
    hashes = manifest["producer"]["artifacts"]
    input_bytes = b"".join(struct.pack("<i", value) for value in range(1, 1025))
    expected_bytes = b"".join(
        struct.pack("<i", value) for value in range(2, 1026)
    )
    assert manifest["dispatch"] == {
        "workgroup_count": [1, 1, 1],
        "workgroup_size": [1, 1, 1],
        "subgroup_size": 0,
        "constant_byte_length": 0,
        "export_binding_count": None,
    }
    assert manifest["buffers"] == [
        {
            "name": "input",
            "direction": "input",
            "dtype": "int32",
            "shape": [1024],
            "byte_length": 4096,
            "generator": {"kind": "iota_i32", "start": 1},
        },
        {
            "name": "output",
            "direction": "output",
            "dtype": "int32",
            "shape": [1024],
            "byte_length": 4096,
            "generator": {
                "kind": "add_i32_scalar",
                "source": "input",
                "scalar": 1,
            },
        },
    ]
    assert not (fixture / "input.bin").exists()
    assert not (fixture / "expected.bin").exists()
    assert hashlib.sha256((fixture / "kernel.xadx").read_bytes()).hexdigest() == (
        hashes["xadx_sha256"]
    )
    assert hashlib.sha256(input_bytes).hexdigest() == hashes["input_sha256"]
    assert hashlib.sha256(expected_bytes).hexdigest() == hashes["expected_sha256"]


def test_self_test_is_explicitly_scoped_to_npu2():
    source = (
        ROOT / "tools/rocm-npu/src/commands/self_test.cc"
    ).read_text(encoding="utf-8")
    docs = (ROOT / "docs/runtime-cli.md").read_text(encoding="utf-8")
    assert 'platform.target_family != "npu2"' in source
    assert "supported only on NPU2" in source
    assert "NPU1 devices remain supported by probe and doctor" in docs


def test_hardware_report_runner_skips_npu1_self_test(monkeypatch, tmp_path):
    commands = []

    def fake_run(arguments, **_kwargs):
        command = arguments[1]
        commands.append(command)
        report = {
            "schema_version": 1,
            "command": "plugin list" if command == "plugin" else command,
            "status": "pass",
            "product": REPORT_PRODUCT,
            "checks": [],
            "data": {"target_family": "npu"} if command == "probe" else {},
        }
        return SimpleNamespace(
            stdout=json.dumps(report), stderr="", returncode=0
        )

    monkeypatch.setattr(validate_rocm_npu_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(validate_rocm_npu_cli, "validate_report", lambda *a, **k: None)
    run_reports(
        Path("rocm-npu"),
        tmp_path,
        require_hardware=True,
        prefix="",
    )
    assert "self-test" not in commands


@pytest.mark.parametrize(
    "workflow",
    ("fastflowlm_hrx_linux.yml", "fastflowlm_hrx_windows.yml"),
)
def test_runtime_workflows_build_and_execute_native_cli(workflow):
    source = (ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
    assert "dev.py build rocm-npu-cli --fresh" in source
    assert "ctest" in source
    assert "validate_rocm_npu_cli.py" in source
    assert source.count("--require-hardware") == 1
    assert "Upload native CLI hardware reports" in source
    assert "Download standalone runtime archive" in source
    assert "Test standalone runtime archive without Python" in source
    assert ".native-artifact" in source
    assert "archive-doctor.json" in source
    archive_step = source.split(
        "- name: Test standalone runtime archive without Python", 1
    )[1].split("- name: Upload native CLI hardware reports", 1)[0]
    assert "validate_rocm_npu_cli.py" not in archive_step
    if workflow.endswith("linux.yml"):
        assert "cmake -E tar xvf" in archive_step
        assert "dnf install" not in archive_step
    else:
        assert "shell: powershell" in archive_step
        assert "shell: pwsh" not in archive_step
    assert "rocm-npu" in archive_step
    assert "archive-probe.json" in archive_step


def test_hardware_report_validator_rejects_duplicate_check_ids():
    report = {
        "schema_version": 1,
        "command": "doctor",
        "status": "pass",
        "product": REPORT_PRODUCT,
        "checks": [
            {"id": "device.present", "status": "pass", "message": "present"},
            {"id": "device.present", "status": "pass", "message": "present"},
        ],
        "data": {},
    }
    with pytest.raises(RuntimeError, match="duplicate check id"):
        validate_report(report, "doctor", require_hardware=True)


@pytest.mark.parametrize(
    ("report_update", "message"),
    (
        ({"product": {"version": "1.0"}}, "source_revision"),
        (
            {
                "checks": [
                    {
                        "id": "device.present",
                        "status": "invalid",
                        "message": "present",
                    }
                ]
            },
            "invalid check status",
        ),
        (
            {
                "checks": [
                    {
                        "id": "DevicePresent",
                        "status": "pass",
                        "message": "present",
                    }
                ]
            },
            "malformed check id",
        ),
        (
            {"checks": [{"id": "device.present", "status": "pass"}]},
            "check message is missing",
        ),
        ({"data": []}, "data must be an object"),
    ),
)
def test_hardware_report_validator_enforces_documented_envelope(
    report_update, message
):
    report = {
        "schema_version": 1,
        "command": "plugin list",
        "status": "pass",
        "product": REPORT_PRODUCT,
        "checks": [],
    }
    report.update(report_update)
    with pytest.raises(RuntimeError, match=message):
        validate_report(report, "plugin-list", require_hardware=False)


def test_hardware_report_validator_allows_reports_without_data():
    report = {
        "schema_version": 1,
        "command": "plugin list",
        "status": "pass",
        "product": REPORT_PRODUCT,
        "checks": [],
    }
    validate_report(report, "plugin-list", require_hardware=False)


def test_hardware_report_validator_allows_optional_ioctl_errors():
    report = {
        "schema_version": 1,
        "command": "probe",
        "status": "pass",
        "product": REPORT_PRODUCT,
        "checks": [],
        "data": {
            "os": "Linux",
            "detected_device_count": 1,
            "usable_device_count": 1,
            "target_family": "npu2",
            "runtime": {"loadable": True},
            "devices": [
                {
                    "name": "accel0",
                    "usable": True,
                    "ioctl": {"firmware_version_error": "not supported"},
                }
            ],
        },
    }
    validate_report(report, "probe", require_hardware=True)

    report["data"] = {
        "os": "Windows",
        "detected_device_count": 1,
        "usable_device_count": 1,
        "target_family": "npu2",
        "runtime": {"loadable": True},
        "devices": [{"name": "NPU", "healthy": True}],
    }
    validate_report(report, "probe", require_hardware=True)


@pytest.mark.parametrize(
    "module",
    ("rocm_npu_linux_wheels.py", "rocm_npu_windows_wheels.py"),
)
def test_runtime_wheels_ship_native_cli_fixture_and_license(module):
    source = (ROOT / "build_tools" / module).read_text(encoding="utf-8")
    assert "rocm-npu" in source
    assert "share/rocm-npu/self-test" in source
    assert "nlohmann-json-LICENSE.MIT" in source
    assert 'console_scripts={"rocm-npu": f"{RUNTIME_PACKAGE}:main"}' in source
