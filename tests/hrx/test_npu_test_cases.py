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
