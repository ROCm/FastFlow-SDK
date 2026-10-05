# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare qualification contracts, member bytes, names, and normalized modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "fastflow-sdk-qualification" / "python"))

from fastflow_sdk_qualification.candidate import (
    verify_qualification_wheel_parity,
)


def _one_wheel(value: Path) -> Path:
    if value.is_file():
        return value
    wheels = sorted(value.glob("fastflow_sdk_qualification-*.whl"))
    if len(wheels) != 1:
        raise ValueError(
            f"expected one qualification wheel in {value}, found {len(wheels)}"
        )
    return wheels[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    verify_qualification_wheel_parity(
        _one_wheel(args.canonical), _one_wheel(args.candidate)
    )
    print("qualification wheel parity: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
