# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Verify an installed qualification dependency closure."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import re
from pathlib import Path

from build_tools import release_policy


def normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def expected_distributions(
    version: str,
    contract_path: Path = release_policy.NUMERIC_RUNTIME_CONTRACT_PATH,
) -> dict:
    packages = release_policy.numeric_package_versions(contract_path)
    return {
        "fastflow-sdk-qualification": version,
        "fastflow-sdk-fastflowlm": version,
        "fastflow-sdk-runtime": version,
        **packages,
    }


def verify_installed_distributions(
    target: Path,
    version: str,
    contract_path: Path = release_policy.NUMERIC_RUNTIME_CONTRACT_PATH,
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
    parser.add_argument(
        "--contract",
        type=Path,
        default=release_policy.NUMERIC_RUNTIME_CONTRACT_PATH,
    )
    args = parser.parse_args()
    verify_installed_distributions(args.target, args.version, args.contract)


if __name__ == "__main__":
    main()
