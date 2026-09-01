# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Text-level GitHub Actions policy helpers.

These helpers intentionally do not parse YAML. Workflow expressions and the
historical YAML 1.1 treatment of ``on`` make a generic YAML loader a poor fit
for policy tests that need to inspect the checked-in source.
"""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"


def read_workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def workflow_job(workflow: str, name: str) -> str:
    """Return one top-level job block from workflow source."""
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n.*?"
        r"(?=^  [A-Za-z_][A-Za-z0-9_-]*:\n|\Z)",
        workflow,
    )
    assert match is not None, f"workflow job is missing: {name}"
    return match.group(0)
