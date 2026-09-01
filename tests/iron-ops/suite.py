# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compatibility import for the production-owned iron-ops suite contract."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_tools.iron import DEFAULT, OPERATORS, Operator, select  # noqa: E402

__all__ = ["DEFAULT", "OPERATORS", "Operator", "select"]
