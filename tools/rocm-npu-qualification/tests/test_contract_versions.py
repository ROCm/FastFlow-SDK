# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep packaged schema assets aligned with their owning version constants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rocm_npu_qualification.contracts import (
    EVIDENCE_SCHEMA_VERSION,
    INDEX_SCHEMA_VERSION,
    POLICY_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
)


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "rocm_npu_qualification"
)
pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def _json(relative: str):
    return json.loads((PACKAGE_ROOT / relative).read_text(encoding="utf-8"))


def test_packaged_schema_versions_match_runtime_contracts():
    assert (
        _json("schemas/evidence-v1.schema.json")["properties"]["schema_version"][
            "const"
        ]
        == EVIDENCE_SCHEMA_VERSION
    )
    assert (
        _json("schemas/profile-v1.schema.json")["properties"]["schema_version"][
            "const"
        ]
        == PROFILE_SCHEMA_VERSION
    )
    assert (
        _json("schemas/qualification-index-v1.schema.json")["properties"][
            "schema_version"
        ]["const"]
        == INDEX_SCHEMA_VERSION
    )


def test_packaged_profiles_and_policies_use_owned_versions():
    for path in (PACKAGE_ROOT / "profiles").glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == (
            PROFILE_SCHEMA_VERSION
        )
    assert _json("policies/release.json")["schema_version"] == POLICY_SCHEMA_VERSION
