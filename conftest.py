# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared pytest policy for repository-local tests."""

import sys
from pathlib import Path

import pytest

PROFILE_MARKERS = frozenset({"presubmit", "compiler_stack", "fastflowlm"})
ROOT = Path(__file__).resolve().parent
TEST_IMPORT_ROOTS = (
    ROOT / "tools" / "fastflow-sdk-qualification" / "python",
    ROOT / "tests" / "npu",
)


def _prioritize_test_import_roots():
    """Keep repository test imports ahead of ambient installations."""
    values = [str(import_root) for import_root in TEST_IMPORT_ROOTS]
    sys.path[:] = [*values, *(entry for entry in sys.path if entry not in values)]


_prioritize_test_import_roots()


def pytest_collection_modifyitems(items):
    """Enforce CI ownership and skip incompatible host-specific tests."""
    host = (
        "windows"
        if sys.platform == "win32"
        else "linux"
        if sys.platform.startswith("linux")
        else None
    )
    unowned = []
    for item in items:
        markers = {marker.name for marker in item.iter_markers()}
        if not markers & PROFILE_MARKERS:
            unowned.append(item.nodeid)
        for platform in ("linux", "windows"):
            if platform in markers and platform != host:
                item.add_marker(
                    pytest.mark.skip(reason=f"requires {platform} host semantics")
                )
    if unowned:
        raise pytest.UsageError(
            "tests missing a CI profile marker:\n  " + "\n  ".join(unowned)
        )
