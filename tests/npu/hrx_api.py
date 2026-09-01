# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compatibility import for the production-owned public HRX ctypes contract."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_tools.hrx_runtime import *  # noqa: E402,F403
