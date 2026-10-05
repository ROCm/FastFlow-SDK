# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reusable FastFlowLM installed-product transcript semantics."""

from __future__ import annotations

import re
from pathlib import Path

from .hashing import sha256_file

RUNTIME_ERROR_PATTERN = re.compile(
    r"\[hrx\]\[ERROR\]|INTERNAL;|ERT_CMD_CHAIN did not complete|ert state [0-9]+|"
    r"D3DKMTCreateContextVirtual|0xC01E0009|0xc01e0009"
)
# Known-benign HRX diagnostics that are emitted on the text path and do not
# indicate a runtime failure. ``run::start with no bindings`` is logged by
# omni/vision models (e.g. gemma4) when the unused image/audio tower bindings
# are absent for a text-only prompt; the text decode still completes normally.
_BENIGN_RUNTIME_ERROR_PATTERN = re.compile(r"run::start with no bindings")
MAX_LENGTH_PATTERN = re.compile(r"\[WARNING\].*Max length reached")
# A benchmark prompt may instruct the model to begin with a terse pass sentinel
# (e.g. ``HRX-NPU-4K-PASS``). Small/omni models sometimes emit only the sentinel
# for a stage; treat that as coherent output rather than failing the coherence
# check, since per-iteration integrity is enforced elsewhere.
BENCHMARK_PASS_SENTINEL_PATTERN = re.compile(
    r"HRX[- ]NPU[- ]\d+K[- ]PASS", re.IGNORECASE
)


def has_runtime_error(text: str) -> bool:
    """Return True only for non-benign FastFlowLM/HRX runtime errors.

    A line is treated as fatal when it matches :data:`RUNTIME_ERROR_PATTERN`
    and is not one of the known-benign diagnostics (see
    :data:`_BENIGN_RUNTIME_ERROR_PATTERN`).
    """
    for line in text.splitlines():
        if RUNTIME_ERROR_PATTERN.search(
            line
        ) and not _BENIGN_RUNTIME_ERROR_PATTERN.search(line):
            return True
    return False


def extract_raw_outputs(text: str) -> list[str]:
    outputs = []
    lines = None
    for line in text.splitlines():
        if "Model RAW Output:" in line:
            if lines is not None:
                outputs.append("\n".join(lines).strip())
            lines = []
            continue
        if lines is not None and (line.startswith(">>>") or line.startswith("[FLM]")):
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
        for value in re.findall(r"Creating checkpoint at context length\s+(\d+)", text)
    ]
    if not values:
        values = [
            int(value) * 1024
            for value in re.findall(r"Starting benchmark for\s+(\d+)k\b", text)
        ]
    matching = [value for value in values if minimum <= value <= maximum]
    # Require AT LEAST one checkpoint per scheduled iteration in the target
    # window. Gemma-family (sliding-window/local attention) models legitimately
    # emit >1 checkpoint per iteration in the window; per-iteration integrity is
    # already enforced by require_benchmark_schedule + raw-output count.
    if len(matching) < expected_count:
        raise ValueError(
            f"benchmark must report at least {expected_count} checkpoint(s) in "
            f"{minimum}..{maximum}; "
            f"got {values or 'none'}"
        )
    return max(matching)


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


def fresh_benchmark_csvs(directory: Path, model: str, *, started_ns: int) -> list[Path]:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return sorted(
        (
            path
            for path in directory.glob(f"bench_{safe_model}_*.csv")
            if path.stat().st_mtime_ns >= started_ns
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
