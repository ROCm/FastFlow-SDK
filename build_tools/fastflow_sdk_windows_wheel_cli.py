# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build, verify, and exercise Windows FastFlow SDK runtime wheels."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_tools.fastflowlm_package import (
    FASTFLOWLM_ARTIFACTS,
    WINDOWS_WHEEL_VENV,
    package_fastflowlm_wheels,
    probe_runtime_package,
    smoke_fastflowlm,
    verify_fastflowlm_wheels,
)
from build_tools.package_version import PACKAGE_VERSION_ENV


def _is_windows_host() -> bool:
    return os.name == "nt"


def main() -> int:
    if not _is_windows_host():
        raise RuntimeError("Windows FastFlow SDK wheel commands require a Windows host")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    package = subparsers.add_parser("package")
    package.add_argument(
        "--version",
        help=f"PEP 440 package version (or ${PACKAGE_VERSION_ENV})",
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--dir", type=Path, default=FASTFLOWLM_ARTIFACTS)
    verify.add_argument("--venv", type=Path, default=WINDOWS_WHEEL_VENV)
    subparsers.add_parser("probe")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--inference-iterations", type=int, default=1)
    args = parser.parse_args()

    if args.command == "package":
        wheels = package_fastflowlm_wheels(args.version)
        for product, wheel in sorted(wheels.items()):
            print(f"{product}: {wheel}")
    elif args.command == "verify":
        verify_fastflowlm_wheels(args.dir, args.venv)
    elif args.command == "probe":
        return probe_runtime_package()
    elif args.command == "smoke":
        return smoke_fastflowlm(args.inference_iterations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
