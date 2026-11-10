# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds the kernel qualification recipe from the checked-in measured facts.

These run without a build tree or an NPU, which is the point of keeping the
fact-to-case transformation pure: the recipe's shape is reviewable here, and only
the digests that bind it to real bytes need a build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from build_tools.fastflowlm_kernel_recipe import (
    RecipeError,
    attest_recipe_generators,
    build_recipe,
    validate_recipe,
)
from tests.support.modules import load_module

ROOT = Path(__file__).resolve().parents[2]


def _xadx_rounds():
    path = (
        ROOT
        / "tools/fastflow-sdk-qualification/python/fastflow_sdk_qualification/contracts.py"
    )
    return load_module("qualification_contracts", path).XADX_ROUNDS


XADX_ROUNDS = _xadx_rounds()
CAPABILITIES = (
    ROOT
    / "tools/fastflow-sdk-qualification/python/fastflow_sdk_qualification"
    / "fastflowlm-kernel-capabilities-v1.json"
)
SCHEMA = ROOT / (
    "tools/fastflow-sdk-qualification/python/fastflow_sdk_qualification"
    "/schemas/kernel-recipe-v1.schema.json"
)
pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def _capabilities():
    return json.loads(CAPABILITIES.read_text(encoding="utf-8"))


def _catalogue():
    """The implementation inventory the probe reports, as facts describe it.

    Families matter beyond provenance: which MM implementation consumes a given
    dequant output is derived from the families they share, so the pairs here are
    what make the chained cases exist.
    """
    lfm = ["LFM2-1.2B-NPU2", "Qwen2.5-3B-Instruct-NPU2"]
    qwen = ["Qwen3-0.6B-NPU2", "Llama-3.1-8B-NPU2"]
    gpt_oss = ["GPT-OSS-20B-NPU2"]
    gemma = ["Gemma3-1B-NPU2"]
    return [
        {"id": "dequant-f026590bcf8694b5", "kernel": "dequant", "families": lfm},
        {"id": "dequant-e3dda597bab4918c", "kernel": "dequant", "families": gpt_oss},
        {"id": "dequant-1fae7d796ad2d91b", "kernel": "dequant", "families": gemma},
        {"id": "dequant-a95b04372f722ae3", "kernel": "dequant", "families": qwen},
        {
            "id": "dequant-cc585404562243d3",
            "kernel": "dequant",
            "families": ["Qwen3-4B-NPU2"],
        },
        {
            "id": "mm-0640ee959bfdb7bb",
            "kernel": "mm",
            "families": lfm + qwen + ["Qwen3-4B-NPU2"],
        },
        {
            "id": "mm-05fd4cb8d5c19a7d",
            "kernel": "mm",
            "families": ["Embedding-Gemma-300M-NPU2"],
        },
        {
            "id": "mm-2220687b4183b1c1",
            "kernel": "mm",
            "families": ["MiniCPM-OCR-1.0-NPU2"],
        },
        {
            "id": "mm-88e2eaf12f653269",
            "kernel": "mm",
            "families": [
                "Qwen3.5-0.8B-NPU2",
                "Qwen3.5-2B-NPU2",
                "Qwen3.5-4B-NPU2",
                "Qwen3.5-9B-NPU2",
                "Qwen3.6-35B-A3B-NPU2",
            ],
        },
        {
            "id": "mm-c43866e708564c76",
            "kernel": "mm",
            "families": gpt_oss,
        },
        {"id": "mm-49a889a56be431c5", "kernel": "mm", "families": gemma},
    ]


def _records():
    return [
        {
            **record,
            "representative": record["families"][0],
            "xclbin_bytes": 4096,
            "xclbin_sha256": record["id"].split("-", 1)[1] + "0" * 48,
        }
        for record in _catalogue()
    ]


def _digests(ids=None):
    ids = ids if ids is not None else [record["id"] for record in _catalogue()]
    digests = {
        id: {
            "xclbin_sha256": id.split("-", 1)[1] + "0" * 48,
            "xclbin_path": f"xclbins/{id}.xclbin",
            "transaction": f"transactions/{id}.bin",
            "transaction_sha256": f"{index + 100:064x}",
            "xadx": f"xadx/{id}.xadx",
            "xadx_sha256": f"{index + 200:064x}",
        }
        for index, id in enumerate(ids)
    }
    if "mm-0640ee959bfdb7bb" in digests:
        digests["mm-0640ee959bfdb7bb"]["execution_variants"] = {
            "production": {
                "transaction": "transactions/mm-production.bin",
                "transaction_sha256": f"{901:064x}",
                "xadx": "xadx/mm-production.xadx",
                "xadx_sha256": f"{902:064x}",
            }
        }
    return digests


def _recipe():
    return build_recipe(_capabilities(), _records(), _digests())


def test_readback_rounds_evenly_cover_each_cases_actual_variants():
    variant_counts = {
        len(case["readback_variants"])
        for case in _recipe()["cases"]
        if case.get("readback_variants")
    }
    assert variant_counts
    for name, rounds in XADX_ROUNDS.items():
        assert all(
            rounds["readback_rounds"] % count == 0 for count in variant_counts
        ), name


def test_recipe_validates_and_covers_every_reachable_implementation():
    recipe = _recipe()
    attest_recipe_generators(recipe, lambda specification, size: bytes(size))
    summary = validate_recipe(recipe, SCHEMA)
    capabilities = _capabilities()
    reached = capabilities["implementations"]["entries"]
    assert summary["implementations"] == len(reached)
    assert summary["numeric_coverage"] == sum(
        facts.get("numeric_limit") is None for facts in reached.values()
    )
    assert summary["cases"] == len(recipe["cases"])
    assert summary["chained"] == sum(
        len(case["stages"]) > 1 for case in recipe["cases"]
    )
    assert summary["coverage_limits"] == len(
        capabilities["implementations"]["unreached"]
    ) + sum(facts.get("numeric_limit") is not None for facts in reached.values())


def test_unreachable_implementations_are_declared_rather_than_dropped():
    recipe = _recipe()
    dispatched = {entry["id"] for entry in recipe["implementations"]}
    limited = {
        "mm-05fd4cb8d5c19a7d",
        "mm-2220687b4183b1c1",
        "mm-88e2eaf12f653269",
        "mm-c43866e708564c76",
    }
    assert limited.isdisjoint(dispatched)
    limits = {limit["representative"] for limit in recipe["coverage_limits"]}
    assert "Qwen3.5-0.8B-NPU2" in limits
    assert "Embedding-Gemma-300M-NPU2" in limits
    assert "MiniCPM-OCR-1.0-NPU2" in limits
    assert "GPT-OSS-20B-NPU2" in limits
    numeric = {
        stage["implementation"]
        for case in recipe["cases"]
        if not case["name"].endswith("zero-baseline")
        for stage in case["stages"]
    }
    assert limited.isdisjoint(numeric)


def test_every_implementation_gets_a_zero_baseline():
    recipe = _recipe()
    baselines = {
        case["stages"][0]["implementation"]
        for case in recipe["cases"]
        if case["name"].endswith("zero-baseline")
    }
    assert baselines == {entry["id"] for entry in recipe["implementations"]}


def test_chained_cases_consume_the_stage_they_follow():
    chained = [case for case in _recipe()["cases"] if len(case["stages"]) > 1]
    assert chained
    for case in chained:
        dequant, mm = case["stages"]
        assert dequant["name"] == "dequant"
        assert mm["name"] == "mm"
        weight = next(b for b in mm["buffers"] if b["name"] == "weight")
        assert weight["source"] == {"stage_output": "dequant.weight"}
        # The stage that hands its output on carries no expectation of its own:
        # the chain is judged on the product it feeds.
        assert "expected" not in dequant["buffers"][0]
        assert mm["buffers"][0]["expected"]["id"] == "chained-reference"


def test_readback_variants_rewrite_the_activation_of_a_resident_case():
    cases = {case["name"]: case for case in _recipe()["cases"]}
    coherence = [name for name, case in cases.items() if "readback_variants" in case]
    assert sorted(coherence) == [
        "mm-0640ee959bfdb7bb/production/shared-subset",
        "mm-0640ee959bfdb7bb/shared-subset",
        "mm-49a889a56be431c5/shared-subset",
    ]
    for name in coherence:
        variants = cases[name]["readback_variants"]
        assert [variant["buffer"] for variant in variants] == [
            "activation",
            "activation",
        ]
        salts = [variant["source"]["parameters"]["salt"] for variant in variants]
        assert salts == [46, 54]
        # Each variant must expect its own product, or a rewritten input would
        # be compared against the output of the one before it.
        assert (
            len({variant["expected"]["parameters"]["salt"] for variant in variants})
            == 2
        )


def test_recipe_strictly_covers_legacy_populations_at_n128():
    cases = {case["name"]: case for case in _recipe()["cases"]}
    generic = "mm-0640ee959bfdb7bb"
    for profile in ("zero", "shared-subset", "full-nibble", "full-affine"):
        stage = cases[f"{generic}/production/{profile}"]["stages"][0]
        assert stage["execution"] == "production"
        assert stage["buffers"][0]["size"] == 2 * 1024 * 128
        assert stage["buffers"][2]["size"] == 2 * 1024 * 128
        if profile != "zero":
            assert stage["buffers"][2]["source"]["parameters"]["salt"] == 29

    for dequant in (
        "dequant-f026590bcf8694b5",
        "dequant-a95b04372f722ae3",
        "dequant-cc585404562243d3",
    ):
        for profile in ("shared-subset", "full-nibble", "full-affine"):
            name = f"{dequant}+{generic}/production/{profile}"
            assert cases[name]["stages"][1]["execution"] == "production"
            assert (
                cases[name]["stages"][1]["buffers"][0]["expected"]["parameters"]["salt"]
                == 29
            )

    for dequant in (
        "dequant-f026590bcf8694b5",
        "dequant-a95b04372f722ae3",
    ):
        assert f"{dequant}/shared-subset" in cases


def test_dequant_expectations_carry_the_measured_order_and_semantics():
    entries = _capabilities()["implementations"]["entries"]
    for case in _recipe()["cases"]:
        stage = case["stages"][0]
        id = stage["implementation"]
        if not id.startswith("dequant") or case["name"].endswith("zero-baseline"):
            continue
        if len(case["stages"]) > 1:
            continue
        expected = stage["buffers"][0]["expected"]["parameters"]
        assert expected["order"] == entries[id]["dequant_output_order"]
        assert expected["semantics"] == entries[id]["dequant_semantics"]
        packed = stage["buffers"][1]["source"]["parameters"]
        blocks = entries[id]["q4nx_blocks"]
        assert packed["row_block"] == blocks["row_block"]
        assert packed["col_block"] == blocks["col_block"]
        assert packed["parallel"] == blocks["parallel"]


def test_dequant_output_is_compared_exactly():
    # The dequant oracle reproduces the kernel bit for bit, so any tolerance
    # here would be hiding a disagreement rather than accommodating one.
    for case in _recipe()["cases"]:
        for stage in case["stages"]:
            if not stage["implementation"].startswith("dequant"):
                continue
            output = stage["buffers"][0]
            if "expected" in output:
                assert output["comparison"] == {"kind": "exact"}


def test_mm_bounds_are_not_inherited_from_the_per_family_policy():
    capabilities = _capabilities()
    recipe = capabilities["recipe_policy"]["gemm_absolute_tolerance"]
    families = {
        family["gemm_policy"]["full_affine_absolute_tolerance"]
        for family in capabilities["families"]
    }
    # An MM implementation is shared across families, so it is tempting to give it
    # the loosest family bound. That bound describes a weight the recipe never
    # builds: mm_weight dequantizes under logical semantics whatever family ships
    # the implementation, so the population depends on the profile alone.
    assert recipe["full-affine"] < max(families)
    # Logical semantics subtracts the zero point before scaling rather than adding
    # after, which leaves the affine population in the same magnitude class as the
    # plain nibble one, so the two reach the same unit in the last place.
    assert recipe["full-affine"] == recipe["full-nibble"]


def test_chaining_is_allowed_more_error_than_the_plain_product():
    policy = _capabilities()["recipe_policy"]
    plain = policy["gemm_absolute_tolerance"]
    chained = policy["chained_absolute_tolerance"]
    # Chaining permutes the weight, which puts differently scaled rows into one
    # accumulation and so reaches larger sums than the plain product does. A
    # chained bound tighter than the plain one would mean that reasoning is wrong.
    for profile, bound in chained.items():
        if profile in plain:
            assert bound >= plain[profile]


def test_recipe_refuses_an_implementation_without_build_digests():
    with pytest.raises(RecipeError, match="no build digests"):
        build_recipe(
            _capabilities(), _records(), _digests(["dequant-f026590bcf8694b5"])
        )


def test_recipe_refuses_a_dequant_with_an_ambiguous_consumer():
    # The chain a dequant feeds is implied by shared families rather than chosen,
    # so a catalogue that implies two consumers is a fact that changed underneath
    # the recipe, not something to resolve by picking one.
    records = _records()
    for record in records:
        if record["id"] == "mm-49a889a56be431c5":
            record["families"] = record["families"] + ["Qwen3-0.6B-NPU2"]
    with pytest.raises(RecipeError, match="shares families with 2"):
        build_recipe(_capabilities(), records, _digests())
