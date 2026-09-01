# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure contract planning for packaged FastFlowLM kernel recipes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_MATERIALIZED_FIELDS = ("source", "expected", "preserved")


def _expected_size(case: dict[str, Any]) -> int:
    """Return the size of the output used to judge a recipe case."""
    for stage in reversed(case["stages"]):
        for record in reversed(stage["buffers"]):
            if record["direction"] == "output" and "expected" in record:
                return record["size"]
    raise ValueError(f"{case['name']} declares no expectation")


def recipe_materialization_requests(
    cases: Iterable[dict[str, Any]],
) -> list[tuple[dict[str, Any], int]]:
    """Enumerate every generator specification and its required output size."""
    requests = []
    for case in cases:
        records = [record for stage in case["stages"] for record in stage["buffers"]]
        buffers = {
            (stage["name"], record["name"]): record
            for stage in case["stages"]
            for record in stage["buffers"]
        }
        for record in records:
            for field in _MATERIALIZED_FIELDS:
                specification = record.get(field)
                if (
                    isinstance(specification, dict)
                    and "stage_output" not in specification
                ):
                    requests.append((specification, record["size"]))
        for variant in case.get("readback_variants", ()):
            requests.append(
                (
                    variant["source"],
                    buffers[(variant["stage"], variant["buffer"])]["size"],
                )
            )
            requests.append((variant["expected"], _expected_size(case)))
    return requests
