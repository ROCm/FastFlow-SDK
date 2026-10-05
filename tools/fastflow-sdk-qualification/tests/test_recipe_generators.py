# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard the structural invariants of the recipe's generated buffers.

The default 32x256 layout is byte-exact against FastFlowLM's converter.
Non-default layouts are C++/Python interpreter agreement over measured facts,
with device qualification as the independent authority. These tests cover what
holds without a build: real permutations, complete packing, and refusal to guess.
"""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from fastflow_sdk_qualification import recipe_generators as generators

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "python" / "fastflow_sdk_qualification"
)
pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]

SHAPE = (256, 256)


def test_bf16_rounds_to_nearest_even():
    # BF16 keeps 7 mantissa bits, so 2**-8 is exactly halfway above 1.0.
    # Halfway cases must round to an even low bit, not away from zero.
    halfway = np.array([1.0 + 2**-8, 1.0 + 3 * 2**-8], dtype=np.float32)
    assert generators.bf16(halfway).tolist() == [0x3F80, 0x3F82]
    assert generators.widen(generators.bf16([1.0, -2.5])).tolist() == [1.0, -2.5]


def test_bf16_conversion_preserves_special_values():
    values = np.asarray(
        [
            0.0,
            -0.0,
            np.finfo(np.float32).smallest_subnormal,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    expected = values.astype(ml_dtypes.bfloat16).view(np.uint16)
    assert np.array_equal(generators.bf16(values), expected)


def test_coordinate_probe_pairs_survive_bf16_without_aliasing():
    blocks = 32 * 32768 // generators.GROUP_COLUMNS
    index = np.arange(blocks, dtype=np.int64)
    low = index & 0x7F
    high = index >> 7
    assert len(set(zip(low.tolist(), high.tolist(), strict=True))) == blocks
    assert np.array_equal(generators.widen(generators.bf16(low)).astype(np.int64), low)
    assert np.array_equal(
        generators.widen(generators.bf16(high)).astype(np.int64), high
    )


@pytest.mark.parametrize(
    "bits",
    [
        generators.MM_WEIGHT_BITS,
        *generators.MM_ACTIVATION_LAYOUTS.values(),
        *generators.DEQUANT_OUTPUT_ORDERS.values(),
    ],
)
def test_layouts_are_bit_permutations(bits):
    assert sorted(bits) == list(range(len(bits)))
    count = 1 << len(bits)
    assert np.unique(generators.permute(count, bits)).size == count


# The block shapes the shipped overlays were measured to read.
BLOCK_SHAPES = [(32, 256, 16), (32, 128, 16), (16, 128, 16)]


@pytest.mark.parametrize("blocks", BLOCK_SHAPES)
def test_q4nx_packing_accounts_for_every_weight(blocks):
    rows, columns = SHAPE
    tensors = generators.source_tensors(rows, columns, "full-affine")
    packed = generators.pack_q4nx(*tensors, *blocks)
    # Half a byte per weight, plus a scale and a zero point per 32-column block.
    assert len(packed) == rows * columns // 2 + 4 * rows * columns // 32


@pytest.mark.parametrize("blocks", BLOCK_SHAPES)
def test_q4nx_packing_round_trips_through_the_chunk_shape(blocks):
    """The chunk shape alone must place every nibble, with no measured table."""
    row_block, col_block, parallel = blocks
    rows, columns = SHAPE
    quantized, scale, zero_point = generators.source_tensors(
        rows, columns, "full-affine"
    )
    packed = generators.pack_q4nx(quantized, scale, zero_point, *blocks)

    chunk = row_block * col_block * 5 // 8
    header = chunk - row_block * col_block // 2
    buffer = np.frombuffer(packed, dtype=np.uint8).reshape(-1, chunk)
    paired = buffer[:, header:].reshape(
        -1, row_block // parallel, col_block, parallel // 2, 1
    )
    recovered = np.concatenate([paired & 0xF, paired >> 4], axis=-1)
    recovered = recovered.transpose(0, 1, 3, 4, 2).reshape(
        rows // row_block, columns // col_block, row_block, col_block
    )
    assert np.array_equal(
        recovered.transpose(0, 2, 1, 3).reshape(rows, columns), quantized
    )


def test_q4nx_packing_refuses_a_shape_its_chunks_do_not_tile():
    tensors = generators.source_tensors(*SHAPE, "shared-subset")
    with pytest.raises(ValueError, match="does not divide"):
        generators.pack_q4nx(*tensors, row_block=48, col_block=256, parallel=16)


def test_dequant_refuses_an_unmeasured_output_order():
    tensors = generators.source_tensors(*SHAPE, "shared-subset")
    with pytest.raises(ValueError, match="no measured dequant output order"):
        generators.dequant_expected(*tensors, "logical", "row-major")


def test_dequant_refuses_a_shape_its_order_was_not_measured_at():
    tensors = generators.source_tensors(64, 256, "shared-subset")
    with pytest.raises(ValueError, match="a different weight count"):
        generators.dequant_expected(*tensors, "logical", "mm")


def test_chained_reference_multiplies_the_weight_as_it_was_written():
    """The chain is the product of the permuted weight, not of the logical one."""
    m = k = 1024
    n = 128
    quantized, scale, zero_point = generators.source_tensors(m, k, "full-affine")
    written = generators.widen(
        np.frombuffer(
            generators.dequant_expected(
                quantized, scale, zero_point, "qwen-affine", "mm"
            ),
            dtype=np.uint16,
        )
    ).reshape(m, k)
    right = generators.widen(generators.activation(k, n, 41))
    want = generators.bf16(
        (written.astype(np.float64) @ right.astype(np.float64)).astype(np.float32)
    ).reshape(-1)
    chained = generators.chained_reference(
        m, k, n, "full-affine", 41, "qwen-affine", "mm"
    )
    assert np.array_equal(np.frombuffer(chained, dtype=np.uint16), want)
    assert chained != generators.gemm_reference(m, k, n, "full-affine", 41)


def test_activation_layout_refuses_an_unmeasured_transaction():
    with pytest.raises(ValueError, match="no measured activation layout"):
        generators.mm_activation_layout(1024, 256, 29, "gemm")


def test_every_generator_the_schema_names_is_implemented():
    schema = json.loads(
        (PACKAGE_ROOT / "schemas/kernel-recipe-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    named = schema["$defs"]["generator"]["properties"]["id"]["enum"]
    for generator in named:
        with pytest.raises(ValueError, match="invalid parameter names"):
            # Preflight recognizes the id before rejecting its missing contract.
            generators.materialize(generator, {})


def test_unknown_generator_is_refused():
    with pytest.raises(ValueError, match="unknown recipe generator"):
        generators.materialize("q4nx-unpacked", {})
