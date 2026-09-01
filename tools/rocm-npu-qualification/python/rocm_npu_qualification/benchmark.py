# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse and validate FastFlowLM benchmark checkpoint summaries."""

from __future__ import annotations

import math

BENCHMARK_CSV_HEADER = (
    "context_length_k",
    "ttft_avg_s",
    "ttft_std_s",
    "ttft_min_s",
    "ttft_max_s",
    "prefill_avg_toks_per_s",
    "prefill_std_toks_per_s",
    "prefill_min_toks_per_s",
    "prefill_max_toks_per_s",
    "decoding_avg_toks_per_s",
    "decoding_std_toks_per_s",
    "decoding_min_toks_per_s",
    "decoding_max_toks_per_s",
)


def benchmark_csv_checkpoint(
    rows: list[list[str]], checkpoint_k: int
) -> tuple[list[str], dict[str, int | float]]:
    if not rows:
        raise ValueError("FastFlowLM benchmark CSV is empty")
    header = tuple(cell.strip() for cell in rows[0])
    if header and header[0].startswith("\ufeff"):
        header = (header[0].lstrip("\ufeff"), *header[1:])
    if header != BENCHMARK_CSV_HEADER:
        raise ValueError("FastFlowLM benchmark CSV header is invalid")

    parsed_rows: list[tuple[list[str], dict[str, int | float]]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != len(BENCHMARK_CSV_HEADER):
            raise ValueError(
                f"FastFlowLM benchmark CSV row {row_number} has "
                f"{len(row)} columns; expected {len(BENCHMARK_CSV_HEADER)}"
            )
        try:
            context_length = int(row[0].strip())
            numeric = [float(value.strip()) for value in row[1:]]
        except ValueError as error:
            raise ValueError(
                f"FastFlowLM benchmark CSV row {row_number} is not numeric"
            ) from error
        if context_length <= 0 or not all(math.isfinite(value) for value in numeric):
            raise ValueError(
                f"FastFlowLM benchmark CSV row {row_number} has invalid values"
            )
        for index, value in enumerate(numeric, start=1):
            if value < 0 or (index not in {2, 6, 10} and value == 0):
                raise ValueError(
                    f"FastFlowLM benchmark CSV row {row_number} has "
                    "non-positive timing or throughput"
                )
        for average, minimum, maximum in ((1, 3, 4), (5, 7, 8), (9, 11, 12)):
            if not numeric[minimum - 1] <= numeric[average - 1] <= numeric[maximum - 1]:
                raise ValueError(
                    f"FastFlowLM benchmark CSV row {row_number} has "
                    "inconsistent summary statistics"
                )
        parsed = {
            BENCHMARK_CSV_HEADER[0]: context_length,
            **{name: value for name, value in zip(BENCHMARK_CSV_HEADER[1:], numeric)},
        }
        parsed_rows.append((row, parsed))

    matches = [
        (row, parsed)
        for row, parsed in parsed_rows
        if parsed["context_length_k"] == checkpoint_k
    ]
    if len(matches) != 1:
        raise ValueError(
            "FastFlowLM benchmark CSV must contain exactly one target "
            f"checkpoint row; found {len(matches)}"
        )
    return matches[0]
