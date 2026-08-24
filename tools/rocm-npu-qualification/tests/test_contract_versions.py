# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep packaged schema assets aligned with their owning version constants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rocm_npu_qualification.contracts import (
    CANDIDATE_SCHEMA_VERSION,
    KERNEL_RECIPE_SCHEMA_VERSION,
    QUALIFICATION_CONTRACT_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    INDEX_SCHEMA_VERSION,
    POLICY_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    XADX_REPORT_SCHEMA_VERSION,
    XADX_ROUNDS,
    xadx_rounds,
)
from rocm_npu_qualification import recipe_generators
from rocm_npu_qualification.schema import SchemaValidationError, validate_schema


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
        _json("schemas/candidate-v1.schema.json")["properties"]["schema_version"][
            "const"
        ]
        == CANDIDATE_SCHEMA_VERSION
    )
    assert (
        _json("schemas/qualification-contract-v1.schema.json")["properties"][
            "schema_version"
        ]["const"]
        == QUALIFICATION_CONTRACT_SCHEMA_VERSION
    )
    assert (
        _json("schemas/kernel-recipe-v1.schema.json")["properties"][
            "schema_version"
        ]["const"]
        == KERNEL_RECIPE_SCHEMA_VERSION
    )
    assert (
        _json("schemas/xadx-report-v1.schema.json")["properties"][
            "schema_version"
        ]["const"]
        == XADX_REPORT_SCHEMA_VERSION
    )
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
    assert (
        _json("schemas/policy-v1.schema.json")["properties"]["schema_version"]["const"]
        == POLICY_SCHEMA_VERSION
    )


@pytest.mark.parametrize(
    ("value", "schema"),
    [
        (3, {"type": "integer", "maximum": 2}),
        ([1, 2], {"type": "array", "maxItems": 1}),
        ([1, 1], {"type": "array", "uniqueItems": True}),
        (1, {"oneOf": [{"const": 1}, {"type": "integer"}]}),
        ("forbidden", {"not": {"const": "forbidden"}}),
        (float("nan"), {"type": "number"}),
        (float("inf"), {"type": "number"}),
    ],
)
def test_offline_schema_validator_enforces_safety_keywords(value, schema):
    with pytest.raises(SchemaValidationError):
        validate_schema(value, schema)


def test_generator_preflight_rejects_unsafe_shapes_before_allocation():
    with pytest.raises(ValueError, match="supported bounds"):
        recipe_generators.predicted_size("zero", {"size": 16 * 1024 * 1024 + 1})
    with pytest.raises(ValueError, match="block divisibility"):
        recipe_generators.predicted_size(
            "q4nx-packed",
            {
                "m": 32,
                "k": 256,
                "profile": "shared-subset",
                "row_block": 32,
                "col_block": 128,
                "parallel": 7,
            },
        )
    with pytest.raises(ValueError, match="working-memory limit"):
        recipe_generators.predicted_size(
            "gemm-reference",
            {
                "m": 1 << 20,
                "k": 1024,
                "n": 1,
                "profile": "shared-subset",
                "salt": 1,
            },
        )
    with pytest.raises(ValueError, match="supported bounds"):
        recipe_generators.predicted_size(
            "gemm-reference",
            {
                "m": 32,
                "k": 1024,
                "n": 128,
                "profile": "shared-subset",
                "salt": 256,
            },
        )


def test_every_packaged_profile_declares_an_xadx_round_policy():
    packaged = {path.stem for path in (PACKAGE_ROOT / "profiles").glob("*.json")}
    # Every packaged profile needs its own XADX round policy, since a profile
    # that falls back to the default is silently qualified at a depth it did not
    # ask for. The "all-models", "bench-all", and "bench-dev" diagnostic
    # sweeps are not packaged profiles (they ship no profiles/*.json), but they
    # still declare their policy explicitly rather than falling through to the
    # default.
    sweeps = {"all-models", "bench-all", "bench-dev"}
    assert set(XADX_ROUNDS) == packaged | sweeps
    assert not (sweeps & packaged)


def test_every_profile_preserves_the_approved_stress_contract():
    assert all(
        rounds
        == {
            "lifecycle_rounds": 5,
            "resident_rounds": 20,
            "readback_rounds": 32,
        }
        for rounds in XADX_ROUNDS.values()
    )


def test_unknown_profile_is_qualified_at_default_depth():
    assert xadx_rounds("not-a-profile") == XADX_ROUNDS["default"]
    assert xadx_rounds("fast") is not XADX_ROUNDS["fast"]
