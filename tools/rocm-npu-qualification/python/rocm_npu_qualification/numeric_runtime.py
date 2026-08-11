# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single source of truth for qualification numeric-runtime requirements."""

from __future__ import annotations

import json
from pathlib import Path


_DOCUMENT = json.loads(
    Path(__file__).with_name("numeric-runtime.json").read_text(encoding="utf-8")
)
if (
    _DOCUMENT.get("schema_version") != 1
    or not isinstance(_DOCUMENT.get("packages"), dict)
    or not isinstance(_DOCUMENT.get("python_abis"), list)
):
    raise RuntimeError("invalid qualification numeric-runtime contract")

NUMERIC_RUNTIME = dict(_DOCUMENT["packages"])
NUMERIC_REQUIRES_PYTHON = _DOCUMENT["requires_python"]
NUMERIC_PYTHON_ABIS = tuple(_DOCUMENT["python_abis"])
NUMERIC_DISTRIBUTIONS = tuple(sorted(NUMERIC_RUNTIME))
NUMERIC_REQUIREMENTS = tuple(
    f"{distribution} =={version}"
    for distribution, version in sorted(NUMERIC_RUNTIME.items())
)
