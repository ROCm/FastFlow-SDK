# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Verify an installed qualification dependency closure."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "tools/rocm-npu-qualification/python/rocm_npu_qualification"
    / "numeric-runtime.json"
)


def normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def expected_distributions(version: str, contract_path: Path = CONTRACT_PATH) -> dict:
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    packages = contract.get("packages")
    if contract.get("schema_version") != 1 or not isinstance(packages, dict):
        raise RuntimeError(f"invalid numeric runtime contract: {contract_path}")
    return {
        "rocm-npu-qualification": version,
        "rocm-npu-fastflowlm": version,
        "rocm-npu-runtime": version,
        **packages,
    }


def verify_installed_distributions(
    target: Path, version: str, contract_path: Path = CONTRACT_PATH
) -> None:
    installed = {
        normalize_distribution(distribution.metadata["Name"]): distribution.version
        for distribution in metadata.distributions(path=[str(target)])
    }
    expected = expected_distributions(version, contract_path)
    if installed != expected:
        raise RuntimeError(
            f"qualification dependency closure differs from the contract: "
            f"expected {expected}, got {installed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    args = parser.parse_args()
    verify_installed_distributions(args.target, args.version, args.contract)


if __name__ == "__main__":
    main()
