# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reusable FastFlowLM installed-product transcript semantics."""

from __future__ import annotations

import re
from pathlib import Path


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


def require_1k_checkpoint(
    text: str, *, minimum: int = 950, maximum: int = 1024
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
    if len(matching) != 1:
        raise ValueError(
            f"benchmark must report exactly one checkpoint in {minimum}..{maximum}; "
            f"got {values or 'none'}"
        )
    return matching[0]


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
