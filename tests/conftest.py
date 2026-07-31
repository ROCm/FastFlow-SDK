# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared pytest policy for repository-local tests."""

import sys

import pytest


PROFILE_MARKERS = frozenset({"presubmit", "compiler_stack", "fastflowlm"})


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
                    pytest.mark.skip(
                        reason=f"requires {platform} host semantics"
                    )
                )
    if unowned:
        raise pytest.UsageError(
            "tests missing a CI profile marker:\n  " + "\n  ".join(unowned)
        )
