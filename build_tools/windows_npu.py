# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Windows NPU discovery, firmware metadata, and HRX runtime probes."""

import contextlib
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def parse_device_query(output: str) -> list[dict]:
    """Parses the PowerShell query and preserves every amdxdna MCDM device."""
    value = json.loads(output)
    devices = [value] if isinstance(value, dict) else value
    if not isinstance(devices, list) or not devices:
        raise ValueError("device query did not return a non-empty object list")
    if not all(isinstance(device, dict) for device in devices):
        raise ValueError("device query returned a non-object entry")
    return devices


def device_is_healthy(device: dict) -> bool:
    return device.get("status") == "OK" and device.get("problem") in (
        "CM_PROB_NONE",
        "0",
        0,
    )


def driver_firmware_packages(driver_path) -> list[dict]:
    """Returns release metadata for firmware shipped beside an active KMD."""
    if not driver_path:
        return []
    driver_dir = Path(driver_path).parent
    packages = []
    for path in sorted(driver_dir.glob("ipu*_pasid.sbin"), key=lambda p: p.name.lower()):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                header = file.read(4096)
                digest.update(header)
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as error:
            packages.append({"name": path.name, "error": str(error)})
            continue
        match = re.search(rb"Release\s+([0-9]+(?:\.[0-9]+){3})\b", header)
        packages.append(
            {
                "name": path.name,
                "release": match.group(1).decode("ascii") if match else "unknown",
                "sha256": digest.hexdigest(),
            }
        )
    return packages


def query_devices() -> tuple[list[dict] | None, str | None]:
    """Queries every present amdxdna MCDM device without opening it."""
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not powershell:
        return None, "PowerShell was not found"
    script = r"""
$ErrorActionPreference = 'Stop'
$candidates = @(Get-PnpDevice -PresentOnly |
  Where-Object {
    $_.Class -eq 'ComputeAccelerator' -and
    $_.InstanceId -like 'PCI\VEN_1022*'
  })
$devices = @($candidates | ForEach-Object {
$pnp = $_
$deviceId = $pnp.InstanceId
$service = (Get-PnpDeviceProperty -InstanceId $deviceId `
  -KeyName 'DEVPKEY_Device_Service').Data
if ($service -ne 'IpuMcdmDriver') { return }
$driverVersion = (Get-PnpDeviceProperty -InstanceId $deviceId `
  -KeyName 'DEVPKEY_Device_DriverVersion').Data
$driverDateValue = (Get-PnpDeviceProperty -InstanceId $deviceId `
  -KeyName 'DEVPKEY_Device_DriverDate').Data
$driverDate = if ($driverDateValue -is [DateTime]) {
  $driverDateValue.ToString('yyyy-MM-dd')
} else {
  [string]$driverDateValue
}
$driverProvider = (Get-PnpDeviceProperty -InstanceId $deviceId `
  -KeyName 'DEVPKEY_Device_DriverProvider').Data
$infName = (Get-PnpDeviceProperty -InstanceId $deviceId `
  -KeyName 'DEVPKEY_Device_DriverInfPath').Data
$systemDriver = Get-CimInstance Win32_SystemDriver -Filter "Name='$service'"
$driverPath = if ($systemDriver) { $systemDriver.PathName.Trim('"') } else { $null }
if ($driverPath -and $driverPath.StartsWith('\SystemRoot\')) {
  $driverPath = Join-Path $env:SystemRoot $driverPath.Substring(12)
}
[PSCustomObject]@{
  device_name = $pnp.FriendlyName
  device_id = $deviceId
  driver_version = [string]$driverVersion
  driver_date = $driverDate
  driver_provider = [string]$driverProvider
  inf_name = [string]$infName
  status = [string]$pnp.Status
  problem = [string]$pnp.Problem
  service = [string]$service
  driver_path = [string]$driverPath
}
})
if ($devices.Count -eq 0) { throw 'amdxdna MCDM device was not found' }
ConvertTo-Json -InputObject $devices -Compress
"""
    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit code {result.returncode}"
        )
        return None, detail
    try:
        return parse_device_query(result.stdout), None
    except (json.JSONDecodeError, ValueError) as error:
        return None, f"could not parse PowerShell device query: {error}"


def print_driver_info() -> int:
    """Prints Windows NPU PnP, driver, and firmware information."""
    print("== amdxdna driver / firmware ==")
    devices, error = query_devices()
    if error:
        print(f"  [MISS] Windows NPU device query failed: {error}")
        return 1

    assert devices is not None
    all_healthy = True
    for index, info in enumerate(devices):
        prefix = f"device {index}: " if len(devices) > 1 else "device: "
        print(f"  [ok ] {prefix}{info['device_name']}")
        print(f"  [info] device id: {info['device_id']}")
        print(
            f"  [info] driver: {info['driver_version']} "
            f"({info['driver_provider']}, {info['driver_date']}, {info['inf_name']})"
        )
        print(f"  [info] service: {info['service']} ({info['driver_path']})")
        healthy = device_is_healthy(info)
        all_healthy &= healthy
        state = "ok " if healthy else "MISS"
        print(f"  [{state}] PnP status: {info['status']} / {info['problem']}")

        packages = driver_firmware_packages(info.get("driver_path"))
        if packages:
            print("  [info] driver-package firmware candidates:")
            for package in packages:
                if package_error := package.get("error"):
                    print(f"    {package['name']}: unreadable ({package_error})")
                else:
                    print(
                        f"    {package['name']}: release {package['release']}, "
                        f"sha256 {package['sha256']}"
                    )
            print(
                "  [note] Windows PnP does not expose which firmware candidate "
                "the KMD loaded"
            )
        else:
            print("  [note] no firmware package metadata found beside the active KMD")
    return 0 if all_healthy else 1


def _bind_probe_api(library) -> None:
    void_pointer = ctypes.c_void_p
    pointer = ctypes.POINTER
    signatures = {
        "hrx_status_code": (ctypes.c_int, [void_pointer]),
        "hrx_status_ignore": (None, [void_pointer]),
        "hrx_status_free_message": (None, [ctypes.c_char_p]),
        "hrx_status_to_string": (
            void_pointer,
            [void_pointer, pointer(ctypes.c_char_p), pointer(ctypes.c_size_t)],
        ),
        "hrx_gpu_initialize": (void_pointer, [ctypes.c_uint32]),
        "hrx_gpu_shutdown": (void_pointer, []),
        "hrx_gpu_device_count": (void_pointer, [pointer(ctypes.c_int)]),
    }
    for name, (result_type, argument_types) in signatures.items():
        function = getattr(library, name)
        function.restype = result_type
        function.argtypes = argument_types


def _check_status(library, status, operation: str) -> tuple[bool, str]:
    if status is None:
        return True, "ok"
    message = ctypes.c_char_p()
    length = ctypes.c_size_t()
    library.hrx_status_ignore(
        library.hrx_status_to_string(
            status, ctypes.byref(message), ctypes.byref(length)
        )
    )
    code = library.hrx_status_code(status)
    text = message.value.decode("utf-8", "replace") if message.value else "?"
    if message.value:
        library.hrx_status_free_message(message)
    library.hrx_status_ignore(status)
    return False, f"{operation} failed (code {code}): {text}"


def probe_runtime(library_path: Path) -> int:
    """Probes the Windows NPU and initializes the specified packaged HRX runtime."""
    result = print_driver_info()
    print("== NPU device / libhrx probe ==")
    library_path = Path(library_path).resolve()
    if not library_path.is_file():
        print(f"  [MISS] hrx.dll: {library_path}")
        return 1
    print(f"  [ok ] hrx.dll: {library_path}")

    with contextlib.ExitStack() as stack:
        if hasattr(os, "add_dll_directory"):
            stack.enter_context(os.add_dll_directory(str(library_path.parent)))
        try:
            library = ctypes.WinDLL(str(library_path))
        except OSError as error:
            print(f"  [MISS] loading hrx.dll failed: {error}")
            return 1
        _bind_probe_api(library)
        ok, message = _check_status(
            library, library.hrx_gpu_initialize(0), "hrx_gpu_initialize"
        )
        if not ok:
            print(f"  [MISS] {message}")
            return 1
        print("  [ok ] hrx_gpu_initialize")

        count = ctypes.c_int(0)
        ok, message = _check_status(
            library,
            library.hrx_gpu_device_count(ctypes.byref(count)),
            "hrx_gpu_device_count",
        )
        if not ok:
            print(f"  [MISS] {message}")
            result = 1
        else:
            print(f"  [{'ok ' if count.value else 'MISS'}] hrx gpu devices: {count.value}")
            if count.value == 0:
                result = 1
        library.hrx_status_ignore(library.hrx_gpu_shutdown())
    print("npu-probe: " + ("ok" if result == 0 else "FAIL"))
    return result
