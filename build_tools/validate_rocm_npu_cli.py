# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run and validate machine-readable native rocm-npu CLI reports."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


CHECK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
STATUSES = {"pass", "fail", "error"}

COMMANDS = {
    "version": ("version",),
    "probe": ("probe",),
    "plugin-list": ("plugin", "list"),
    "doctor": ("doctor",),
    "self-test": ("self-test",),
}


def validate_report(
    report: object, command: str, *, require_hardware: bool
) -> None:
    if not isinstance(report, dict):
        raise RuntimeError(f"{command}: report is not a JSON object")
    expected_command = "plugin list" if command == "plugin-list" else command
    if report.get("schema_version") != 1 or report.get("command") != expected_command:
        raise RuntimeError(f"{command}: invalid report envelope")
    if report.get("status") not in STATUSES:
        raise RuntimeError(f"{command}: invalid status")
    product = report.get("product")
    if not isinstance(product, dict):
        raise RuntimeError(f"{command}: product metadata is missing")
    for key in ("version", "source_revision", "build_platform"):
        if not isinstance(product.get(key), str) or not product[key]:
            raise RuntimeError(f"{command}: product {key} is missing")
    checks = report.get("checks")
    if not isinstance(checks, list):
        raise RuntimeError(f"{command}: checks must be an array")
    check_by_id = {}
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("id"), str):
            raise RuntimeError(f"{command}: malformed check")
        if not CHECK_ID_PATTERN.fullmatch(check["id"]):
            raise RuntimeError(f"{command}: malformed check id {check['id']!r}")
        if check.get("status") not in STATUSES:
            raise RuntimeError(f"{command}: invalid check status")
        if not isinstance(check.get("message"), str):
            raise RuntimeError(f"{command}: check message is missing")
        if check["id"] in check_by_id:
            raise RuntimeError(f"{command}: duplicate check id {check['id']}")
        check_by_id[check["id"]] = check

    data = report.get("data", {})
    if not isinstance(data, dict):
        raise RuntimeError(f"{command}: data must be an object")
    if command == "version":
        runtime = data.get("runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError("version: runtime facts are missing")
        if require_hardware and not runtime.get("loadable"):
            raise RuntimeError("version: packaged HRX is not loadable")
    elif command == "probe" and require_hardware:
        if data.get("detected_device_count", 0) < 1:
            raise RuntimeError("probe: no NPU device detected")
        if data.get("usable_device_count", 0) < 1:
            raise RuntimeError("probe: no NPU device usable")
        if not data.get("target_family"):
            raise RuntimeError("probe: target family was not inferred")
        runtime = data.get("runtime")
        if not isinstance(runtime, dict) or not runtime.get("loadable"):
            raise RuntimeError("probe: packaged HRX is not loadable")
        if data.get("os") == "Linux":
            usable_devices = [
                device
                for device in data.get("devices", [])
                if device.get("usable")
            ]
            if not any(
                isinstance(device.get("ioctl"), dict)
                for device in usable_devices
            ):
                raise RuntimeError(
                    "probe: no usable device attempted ioctl diagnostics"
                )
    elif command in {"doctor", "self-test"} and require_hardware:
        if report.get("status") != "pass":
            raise RuntimeError(f"{command}: hardware report did not pass")
        required = {
            "package.manifest",
            "device.present",
            "device.access",
            "runtime.library.present",
            "runtime.library.load",
            "runtime.symbols",
            "runtime.abi",
            "runtime.initialize",
            "runtime.device_count",
            "runtime.shutdown",
        }
        if command == "self-test":
            required.update(
                {"self_test.fixture", "self_test.cleanup", "self_test.output"}
            )
        missing = sorted(required - set(check_by_id))
        failing = sorted(
            check_id
            for check_id in required & set(check_by_id)
            if check_by_id[check_id].get("status") != "pass"
        )
        if missing or failing:
            raise RuntimeError(
                f"{command}: required checks missing={missing}, failing={failing}"
            )


def run_reports(
    cli: Path,
    output_directory: Path,
    *,
    require_hardware: bool,
    prefix: str,
    runtime: Path | None = None,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    hardware_target = None
    for name, arguments in COMMANDS.items():
        if (
            name == "self-test"
            and require_hardware
            and hardware_target != "npu2"
        ):
            continue
        result = subprocess.run(
            [
                str(cli),
                *arguments,
                "--json",
                *([] if runtime is None else ["--runtime", str(runtime)]),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        destination = output_directory / f"{prefix}{name}.json"
        destination.write_text(result.stdout, encoding="utf-8")
        try:
            report = json.loads(result.stdout)
            validate_report(report, name, require_hardware=require_hardware)
            if name == "probe":
                hardware_target = report.get("data", {}).get("target_family")
        except (json.JSONDecodeError, RuntimeError) as error:
            raise RuntimeError(
                f"{name} validation failed (exit {result.returncode}): {error}; "
                f"stderr={result.stderr.strip()!r}"
            ) from error
        if require_hardware and result.returncode != 0:
            raise RuntimeError(f"{name} returned {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--require-hardware", action="store_true")
    args = parser.parse_args()
    run_reports(
        args.cli,
        args.output_dir,
        require_hardware=args.require_hardware,
        prefix=args.prefix,
        runtime=args.runtime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
