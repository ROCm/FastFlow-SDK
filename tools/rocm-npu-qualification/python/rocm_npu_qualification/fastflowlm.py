# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reusable FastFlowLM installed-product transcript semantics."""

from __future__ import annotations

import re
from pathlib import Path

from .hashing import sha256_file


RUNTIME_ERROR_PATTERN = re.compile(
    r"\[hrx\]\[ERROR\]|INTERNAL;|ERT_CMD_CHAIN did not complete|ert state [0-9]+"
)
MAX_LENGTH_PATTERN = re.compile(r"\[WARNING\].*Max length reached")


def extract_raw_outputs(text: str) -> list[str]:
    outputs = []
    lines = None
    for line in text.splitlines():
        if "Model RAW Output:" in line:
            if lines is not None:
                outputs.append("\n".join(lines).strip())
            lines = []
            continue
        if lines is not None and (
            line.startswith(">>>") or line.startswith("[FLM]")
        ):
            outputs.append("\n".join(lines).strip())
            lines = None
            continue
        if lines is not None:
            lines.append(line)
    if lines is not None:
        outputs.append("\n".join(lines).strip())
    return outputs


def normalize_smoke_phrase(text: str) -> str:
    text = text.strip()
    previous = None
    while text != previous:
        previous = text
        text = re.sub(r"`([^`\n]+)`", r"\1", text)
        text = re.sub(r"(\*\*|__|\*|_)([^*_`\n]+?)\1", r"\2", text)
    return text


def smoke_phrase_matches(raw_text: str, expected: str) -> bool:
    return normalize_smoke_phrase(raw_text) == expected.strip()


def verify_xclbin_inventory(
    xclbin_root: Path, expected_records: list[dict[str, str]]
) -> list[str]:
    """Return identity errors for an expected FastFlowLM xclbin inventory."""
    errors = []
    for expected in expected_records:
        path = xclbin_root / expected["path"]
        if not path.is_file():
            errors.append(f"missing {expected['path']}")
        elif sha256_file(path) != expected["sha256"]:
            errors.append(f"SHA256 mismatch for {expected['path']}")
    return errors


def require_1k_checkpoint(
    text: str,
    *,
    minimum: int = 950,
    maximum: int = 1024,
    expected_count: int = 1,
) -> int:
    values = [
        int(value)
        for value in re.findall(
            r"Creating checkpoint at context length\s+(\d+)", text
        )
    ]
    if not values:
        values = [
            int(value) * 1024
            for value in re.findall(
                r"Starting benchmark for\s+(\d+)k\b", text
            )
        ]
    matching = [value for value in values if minimum <= value <= maximum]
    if len(matching) != expected_count:
        raise ValueError(
            f"benchmark must report exactly {expected_count} checkpoint(s) in "
            f"{minimum}..{maximum}; "
            f"got {values or 'none'}"
        )
    return matching[0]


def require_benchmark_schedule(text: str, *, iterations: int) -> int:
    """Return the scheduled case count after validating FLM's iteration matrix."""
    cases = [
        (int(context), int(iteration))
        for context, iteration in re.findall(
            r"Starting benchmark for\s+(\d+)k and iteration\s+(\d+)", text
        )
    ]
    if not cases:
        raise ValueError("FastFlowLM benchmark did not report a scheduled case")
    contexts = {context for context, _ in cases}
    expected = {
        (context, iteration)
        for context in contexts
        for iteration in range(1, iterations + 1)
    }
    if set(cases) != expected or len(cases) != len(expected):
        raise ValueError(
            "FastFlowLM benchmark schedule is incomplete or duplicated: "
            f"expected {len(expected)} unique case(s), got {len(cases)}"
        )
    return len(expected)


def fresh_benchmark_csvs(
    directory: Path, model: str, *, started_ns: int
) -> list[Path]:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return sorted(
        (
            path
            for path in directory.glob(f"bench_{safe_model}_*.csv")
            if path.stat().st_mtime_ns >= started_ns
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
