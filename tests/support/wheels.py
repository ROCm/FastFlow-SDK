# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Wheel fixture builders shared by packaging tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def make_package_tree(
    root: Path,
    package_name: str,
    *,
    exist_ok: bool = False,
    initialize: Callable[[Path], None] | None = None,
) -> Path:
    """Create the minimal package tree accepted by the wheel builders."""
    tree = root / f"tree-{package_name}"
    package = tree / package_name
    package.mkdir(parents=True, exist_ok=exist_ok)
    (package / "__init__.py").write_text("", encoding="utf-8")
    if initialize is not None:
        initialize(package)
    return tree
