# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure Windows NPU probe parsing helpers."""

import hashlib
import json
import re
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
