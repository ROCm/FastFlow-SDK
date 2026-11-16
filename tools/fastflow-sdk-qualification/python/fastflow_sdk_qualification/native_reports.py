# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validation for raw reports emitted by packaged HRX qualification tools."""

from __future__ import annotations

from typing import Any

# `hrx-amdxdna-resource-pressure` report version pinned with the HRX runtime.
SHARED_CODE_MEMORY_REPORT_SCHEMA_VERSION = 2

_SHARED_CODE_MEMORY_DEFAULTS: dict[str, Any] = {
    "bounded_shared_code_memory": False,
    "allocated_bytes": 0,
    "budget_bytes": 0,
    "exhaustion_observed": False,
    "reclaimed_bytes": 0,
    "post_reclaim_allocation_bytes": 0,
    "retained_command_code_bytes": 0,
    "churned_command_code_bytes": 0,
    "fragmented_large_allocation_bytes": 0,
}

_SHARED_CODE_MEMORY_BYTE_FIELDS = tuple(
    name
    for name, default in _SHARED_CODE_MEMORY_DEFAULTS.items()
    if type(default) is int
)


def shared_code_memory_report_passed(report: Any) -> bool:
    """Return whether a resource-pressure report is a complete passing report.

    Bounded drivers must exhaust and reclaim their shared code-memory budget,
    advertise a retained command-code ceiling, and still allocate three
    quarters of the budget as one instruction buffer after command churn at
    that ceiling. Unbounded drivers run only the instruction smoke.
    """
    if not isinstance(report, dict):
        return False
    if (
        report.get("schema_version") != SHARED_CODE_MEMORY_REPORT_SCHEMA_VERSION
        or report.get("status") != "pass"
    ):
        return False
    bounded = report.get("bounded_shared_code_memory")
    exhaustion = report.get("exhaustion_observed")
    if not isinstance(bounded, bool) or not isinstance(exhaustion, bool):
        return False
    values = {name: report.get(name) for name in _SHARED_CODE_MEMORY_BYTE_FIELDS}
    if not all(type(value) is int and value >= 0 for value in values.values()):
        return False
    budget = values["budget_bytes"]
    retained = values["retained_command_code_bytes"]
    if bounded:
        return (
            budget > 0
            and values["allocated_bytes"] >= (budget // 4) * 3
            and exhaustion
            and values["reclaimed_bytes"] >= values["post_reclaim_allocation_bytes"] > 0
            and retained > 0
            and values["churned_command_code_bytes"] >= 4 * retained
            and values["fragmented_large_allocation_bytes"] >= (budget // 4) * 3
        )
    return (
        budget == 0
        and values["allocated_bytes"] > 0
        and not exhaustion
        and values["reclaimed_bytes"] == 0
        and values["post_reclaim_allocation_bytes"] == 0
        and retained == 0
        and values["churned_command_code_bytes"] == 0
        and values["fragmented_large_allocation_bytes"] == 0
    )


def shared_code_memory_metrics(report: Any) -> dict[str, Any]:
    """Project a resource-pressure report onto its native test-record metrics."""
    data = report if isinstance(report, dict) else {}
    metrics: dict[str, Any] = {}
    for name, default in _SHARED_CODE_MEMORY_DEFAULTS.items():
        value = data.get(name)
        metrics[name] = value if type(value) is type(default) else default
    return metrics
