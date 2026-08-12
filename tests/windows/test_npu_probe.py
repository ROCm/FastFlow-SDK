# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import hashlib
import json
from unittest import mock

import pytest

from build_tools.hrx_runtime import bind_libhrx as shared_bind_libhrx
from build_tools.windows_npu import (
    device_is_healthy,
    driver_firmware_packages,
    parse_device_query,
    probe_runtime,
)

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.windows,
]


class TestWindowsFirmwarePackage:
    def test_reads_release_and_hash_without_xrt(self, tmp_path):
        # Real AMD firmware headers place Release immediately after a hex hash.
        content = b"deadbeefRelease 1.1.2.64\0payload"
        driver = tmp_path / "ipustack.sys"
        driver.write_bytes(b"driver")
        firmware = tmp_path / "ipu_3_pasid.sbin"
        firmware.write_bytes(content)

        packages = driver_firmware_packages(driver)

        assert len(packages) == 1
        assert packages[0]["name"] == "ipu_3_pasid.sbin"
        assert packages[0]["release"] == "1.1.2.64"
        assert packages[0]["sha256"] == hashlib.sha256(content).hexdigest()

    def test_reports_unknown_for_unversioned_firmware(self, tmp_path):
        driver = tmp_path / "ipustack.sys"
        driver.write_bytes(b"driver")
        (tmp_path / "ipu_pasid.sbin").write_bytes(b"no release metadata")

        packages = driver_firmware_packages(driver)

        assert packages[0]["release"] == "unknown"


class TestWindowsDeviceQuery:
    def test_preserves_multiple_mcdm_devices(self):
        devices = [
            {"device_id": "PCI\\VEN_1022&DEV_17F0", "status": "OK", "problem": "0"},
            {"device_id": "PCI\\VEN_1022&DEV_1502", "status": "OK", "problem": "CM_PROB_NONE"},
        ]
        assert parse_device_query(json.dumps(devices)) == devices
        assert all(device_is_healthy(device) for device in devices)

    @pytest.mark.parametrize("output", ("[]", "null", "[1]"))
    def test_rejects_empty_or_non_object_results(self, output):
        with pytest.raises(ValueError):
            parse_device_query(output)

    def test_unhealthy_pnp_state_is_reported(self):
        assert not device_is_healthy({"status": "Error", "problem": "22"})


class TestWindowsRuntimeProbe:
    @pytest.mark.parametrize(
        ("device_count", "expected_result", "state"),
        ((1, 0, "ok "), (0, 1, "MISS")),
    )
    def test_probes_explicit_packaged_library(
        self, tmp_path, capsys, device_count, expected_result, state
    ):
        class Function:
            def __init__(self, implementation=lambda *args: None):
                self.implementation = implementation

            def __call__(self, *args):
                return self.implementation(*args)

        def set_device_count(pointer):
            pointer._obj.value = device_count

        library = type("Library", (), {})()
        for name in (
            "hrx_status_code",
            "hrx_status_ignore",
            "hrx_status_free_message",
            "hrx_status_to_string",
            "hrx_gpu_initialize",
            "hrx_gpu_shutdown",
        ):
            setattr(library, name, Function())
        library.hrx_gpu_device_count = Function(set_device_count)

        path = tmp_path / "hrx.dll"
        path.write_bytes(b"MZ")
        with mock.patch(
            "build_tools.windows_npu.print_driver_info", return_value=0
        ), mock.patch(
            "build_tools.windows_npu.os.add_dll_directory",
            return_value=contextlib.nullcontext(),
        ), mock.patch(
            "build_tools.windows_npu.ctypes.WinDLL", return_value=library
        ) as load, mock.patch(
            "build_tools.windows_npu.bind_libhrx",
            wraps=shared_bind_libhrx,
        ) as bind:
            assert probe_runtime(path) == expected_result
        load.assert_called_once_with(str(path.resolve()))
        bind.assert_called_once_with(library)
        output = capsys.readouterr().out
        assert f"  [{state}] hrx gpu devices: {device_count}" in output
        assert output.endswith(
            f"npu-probe: {'ok' if expected_result == 0 else 'FAIL'}\n"
        )
