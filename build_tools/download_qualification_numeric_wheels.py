# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Download a complete numeric wheelhouse for optional offline qualification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "tools/rocm-npu-qualification/python/rocm_npu_qualification"
    / "numeric-runtime.json"
)


def numeric_runtime_contract(path: Path = CONTRACT_PATH) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != 1
        or not document.get("python_abis")
        or not document.get("packages")
    ):
        raise RuntimeError(f"invalid numeric runtime contract: {path}")
    return document


def download_numeric_wheels(destination: Path) -> None:
    contract = numeric_runtime_contract()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    requirements = [
        f"{distribution}=={version}"
        for distribution, version in sorted(contract["packages"].items())
    ]
    for abi in contract["python_abis"]:
        python_version = f"{abi[2]}.{abi[3:]}"
        for platform in ("manylinux_2_28_x86_64", "win_amd64"):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    "--disable-pip-version-check",
                    "--no-deps",
                    "--only-binary=:all:",
                    "--implementation",
                    "cp",
                    "--python-version",
                    python_version,
                    "--abi",
                    abi,
                    "--platform",
                    platform,
                    "--dest",
                    str(destination),
                    *requirements,
                ],
                check=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    download_numeric_wheels(parser.parse_args().destination)


if __name__ == "__main__":
    main()
