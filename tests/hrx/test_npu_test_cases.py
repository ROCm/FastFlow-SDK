# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from build_tools import npu_test_cases as cases

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]


@pytest.fixture(scope="module", autouse=True)
def _load_npu_cases():
    global np
    import numpy as np


def test_round_to_nearest_even_ties():
    values = np.array([0x3F808000, 0x3F818000], dtype=np.uint32).view(np.float32)
    converted = cases.float32_to_bf16(values)
    np.testing.assert_array_equal(
        converted, np.array([0x3F80, 0x3F82], dtype=np.uint16)
    )


def test_round_trip_matches_bf16_precision():
    values = np.array(
        [-3.75, -0.0, 0.0, 0.1, 1.0, 17.25, np.inf, -np.inf],
        dtype=np.float32,
    )
    storage = cases.float32_to_bf16(values)
    round_trip = cases.bf16_to_float32(storage)
    np.testing.assert_array_equal(cases.float32_to_bf16(round_trip), storage)
    assert round_trip.dtype == np.float32


def test_nan_remains_nan():
    result = cases.bf16_to_float32(
        cases.float32_to_bf16(np.array([np.nan], dtype=np.float32))
    )
    assert np.isnan(result[0])


@pytest.mark.parametrize(
    ("name", "buffer_expectations"),
    [
        (
            "vector_scalar_add",
            [
                ("input", (1024,), "int32"),
                ("output", (1024,), "int32"),
            ],
        ),
        (
            "vector_scalar_add_chain4",
            [
                ("input", (1024,), "int32"),
                ("output", (1024,), "int32"),
            ],
        ),
        (
            "elementwise_add",
            [
                ("lhs", (2048,), "uint16"),
                ("rhs", (2048,), "uint16"),
                ("output", (2048,), "uint16"),
            ],
        ),
        (
            "relu",
            [
                ("input", (2048,), "uint16"),
                ("output", (2048,), "uint16"),
            ],
        ),
        (
            "gemv",
            [
                ("matrix", (128, 128), "uint16"),
                ("vector", (128,), "uint16"),
                ("output", (128,), "uint16"),
            ],
        ),
        (
            "elementwise_mul",
            [
                ("lhs", (2048,), "uint16"),
                ("rhs", (2048,), "uint16"),
                ("output", (2048,), "uint16"),
            ],
        ),
        (
            "axpy",
            [
                ("lhs", (2048,), "uint16"),
                ("rhs", (2048,), "uint16"),
                ("output", (2048,), "uint16"),
            ],
        ),
        ("gelu", [("input", (2048,), "uint16"), ("output", (2048,), "uint16")]),
        ("leaky_relu", [("input", (2048,), "uint16"), ("output", (2048,), "uint16")]),
        ("sigmoid", [("input", (2048,), "uint16"), ("output", (2048,), "uint16")]),
        ("tanh", [("input", (2048,), "uint16"), ("output", (2048,), "uint16")]),
        ("silu", [("input", (2048,), "uint16"), ("output", (2048,), "uint16")]),
        ("mem_copy", [("input", (2048,), "uint16"), ("output", (2048,), "uint16")]),
        ("layer_norm", [("input", (2048,), "uint16"), ("output", (2048,), "uint16")]),
        (
            "rms_norm",
            [("input", (1, 2048), "uint16"), ("output", (1, 2048), "uint16")],
        ),
        (
            "softmax",
            [("input", (32, 64), "uint16"), ("output", (32, 64), "uint16")],
        ),
        (
            "gemm",
            [
                ("lhs", (256, 64), "uint16"),
                ("rhs", (64, 64), "uint16"),
                ("output", (256, 64), "uint16"),
            ],
        ),
        (
            "transpose",
            [("input", (64, 64), "uint16"), ("output", (64, 64), "uint16")],
        ),
        (
            "repeat",
            [("input", (32, 1024), "uint16"), ("output", (64, 1024), "uint16")],
        ),
        (
            "add_relu_fused",
            [
                ("inputs", (8192,), "uint16"),
                ("output", (4096,), "uint16"),
                ("scratch", (4096,), "uint16"),
            ],
        ),
        (
            "strided_copy",
            [("input", (2048,), "uint16"), ("output", (1024,), "uint16")],
        ),
    ],
)
def test_exact_suite_shapes_storage_and_binding_order(name, buffer_expectations):
    generated = cases.generate_case(name)
    assert generated["name"] == name
    actual = []
    for descriptor in generated["buffers"]:
        data = descriptor.get("data", descriptor.get("expected"))
        actual.append((descriptor["name"], data.shape, data.dtype.name))
    assert actual == buffer_expectations


def test_vector_scalar_add_reference_and_value_coverage():
    generated = cases.vector_scalar_add()
    source = generated["buffers"][0]["data"]
    expected = generated["buffers"][1]["expected"]
    np.testing.assert_array_equal(expected, source + 1)
    assert np.any(source < 0)
    assert np.any(source == 0)
    assert np.any(source > 0)
    chained = cases.vector_scalar_add_chain4()
    np.testing.assert_array_equal(chained["buffers"][0]["data"], source)
    np.testing.assert_array_equal(chained["buffers"][1]["expected"], expected)


def test_bf16_references_are_computed_from_rounded_inputs():
    add = cases.elementwise_add()["buffers"]
    add_reference = cases.float32_to_bf16(
        cases.bf16_to_float32(add[0]["data"]) + cases.bf16_to_float32(add[1]["data"])
    )
    np.testing.assert_array_equal(add[2]["expected"], add_reference)

    relu = cases.relu()["buffers"]
    relu_input = cases.bf16_to_float32(relu[0]["data"])
    relu_reference = cases.float32_to_bf16(np.maximum(relu_input, np.float32(0)))
    np.testing.assert_array_equal(relu[1]["expected"], relu_reference)
    assert np.any(relu_input < 0)
    assert np.any(relu_input == 0)
    assert np.any(relu_input > 0)


def test_gemv_reference_is_float32_then_bf16_rounded():
    buffers = cases.gemv()["buffers"]
    matrix = cases.bf16_to_float32(buffers[0]["data"])
    vector = cases.bf16_to_float32(buffers[1]["data"])
    reference = cases.float32_to_bf16(matrix @ vector)
    np.testing.assert_array_equal(buffers[2]["expected"], reference)
    output = cases.bf16_to_float32(reference)
    assert np.any(output < 0)
    assert np.any(output == 0)
    assert np.any(output > 0)


def test_binary_and_copy_references_match_rounded_inputs():
    mul = cases.elementwise_mul()["buffers"]
    np.testing.assert_array_equal(
        mul[2]["expected"],
        cases.float32_to_bf16(
            cases.bf16_to_float32(mul[0]["data"])
            * cases.bf16_to_float32(mul[1]["data"])
        ),
    )
    axpy = cases.axpy()["buffers"]
    np.testing.assert_array_equal(
        axpy[2]["expected"],
        cases.float32_to_bf16(
            np.float32(3) * cases.bf16_to_float32(axpy[0]["data"])
            + cases.bf16_to_float32(axpy[1]["data"])
        ),
    )
    copied = cases.mem_copy()["buffers"]
    np.testing.assert_array_equal(copied[1]["expected"], copied[0]["data"])


def test_gemm_and_layout_references():
    gemm = cases.gemm()["buffers"]
    reference = cases.float32_to_bf16(
        cases.bf16_to_float32(gemm[0]["data"]) @ cases.bf16_to_float32(gemm[1]["data"])
    )
    np.testing.assert_array_equal(gemm[2]["expected"], reference)

    transposed = cases.transpose()["buffers"]
    np.testing.assert_array_equal(
        transposed[1]["expected"],
        cases.float32_to_bf16(cases.bf16_to_float32(transposed[0]["data"]).T),
    )
    repeated = cases.repeat()["buffers"]
    np.testing.assert_array_equal(
        repeated[1]["expected"],
        cases.float32_to_bf16(
            np.repeat(cases.bf16_to_float32(repeated[0]["data"]), 2, axis=0)
        ),
    )


def test_strided_copy_alternates_parameterized_input_offsets():
    generated = cases.strided_copy()
    source = generated["buffers"][0]["data"]
    np.testing.assert_array_equal(generated["buffers"][1]["expected"], source[:1024])
    dispatches = generated["dispatches"]
    assert [dispatch["name"] for dispatch in dispatches] == [
        "input_offset_0",
        "input_offset_1024",
    ]
    assert [
        int.from_bytes(dispatch["constants"], "little", signed=True)
        for dispatch in dispatches
    ] == [0, 1024]
    np.testing.assert_array_equal(dispatches[0]["expected"]["output"], source[:1024])
    np.testing.assert_array_equal(dispatches[1]["expected"]["output"], source[1024:])
    assert dispatches[0]["expected"]["output"].dtype == np.uint16


def test_add_relu_fused_uses_consolidated_arenas_and_rounded_intermediate():
    buffers = cases.add_relu_fused()["buffers"]
    inputs = buffers[0]["data"]
    lhs = cases.bf16_to_float32(inputs[:4096])
    rhs = cases.bf16_to_float32(inputs[4096:])
    summed = cases.float32_to_bf16(lhs + rhs)
    expected = cases.float32_to_bf16(
        np.maximum(cases.bf16_to_float32(summed), np.float32(0))
    )
    np.testing.assert_array_equal(buffers[1]["expected"], expected)
    np.testing.assert_array_equal(buffers[2]["data"], np.zeros(4096, dtype=np.uint16))
    assert np.any(lhs + rhs < 0)
    assert np.any(lhs + rhs > 0)


def test_unary_and_norm_references_match_rounded_inputs():
    relu_like = cases.leaky_relu()["buffers"]
    source = cases.bf16_to_float32(relu_like[0]["data"])
    np.testing.assert_array_equal(
        relu_like[1]["expected"],
        cases.float32_to_bf16(np.where(source >= 0, source, np.float32(0.01) * source)),
    )

    silu = cases.silu()["buffers"]
    x = cases.bf16_to_float32(silu[0]["data"])
    np.testing.assert_array_equal(
        silu[1]["expected"],
        cases.float32_to_bf16(x / (np.float32(1) + np.exp(-x))),
    )

    rms = cases.rms_norm()["buffers"]
    rms_in = cases.bf16_to_float32(rms[0]["data"])
    denom = np.sqrt(np.mean(rms_in * rms_in, axis=-1, keepdims=True) + np.float32(1e-5))
    np.testing.assert_array_equal(
        rms[1]["expected"], cases.float32_to_bf16(rms_in / denom)
    )

    softmax = cases.softmax()["buffers"]
    logits = cases.bf16_to_float32(softmax[0]["data"])
    shifted = logits - logits.max(axis=-1, keepdims=True)
    expo = np.exp(shifted)
    np.testing.assert_array_equal(
        softmax[1]["expected"],
        cases.float32_to_bf16(expo / expo.sum(axis=-1, keepdims=True)),
    )
