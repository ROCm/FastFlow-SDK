# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Deterministic NumPy-only data generators for packaged NPU tests.

Each generator returns a dictionary with a case ``name`` and an ordered
``buffers`` list. Buffer order is kernel binding order. Input descriptors carry
``data`` and output descriptors carry ``expected``; the remaining fields map
directly to the manifest schema consumed by ``run_test_package.py``.

BF16 is represented as raw ``numpy.uint16`` storage. No optional NumPy dtype
extensions are required.
"""

import numpy as np


def float32_to_bf16(values):
    """Converts float32 values to raw BF16 with round-to-nearest-even."""
    floats = np.asarray(values, dtype=np.float32)
    bits = floats.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    result = (rounded >> 16).astype(np.uint16)

    # Keep NaNs as NaNs if rounding their payload would otherwise produce inf.
    is_nan = ((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & (
        (bits & np.uint32(0x007FFFFF)) != 0
    )
    nan_result = ((bits >> 16) | np.uint32(0x0040)).astype(np.uint16)
    return np.where(is_nan, nan_result, result).astype(np.uint16)


def bf16_to_float32(values):
    """Converts raw BF16 storage to exact float32 values."""
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def _bf16_input(name, values, shape):
    return {
        "name": name,
        "direction": "input",
        "dtype": "uint16",
        "logical_dtype": "bfloat16",
        "shape": tuple(shape),
        "data": float32_to_bf16(values).reshape(shape),
    }


def _bf16_output(name, values, shape, rtol=0.01, atol=0.01):
    return {
        "name": name,
        "direction": "output",
        "dtype": "uint16",
        "logical_dtype": "bfloat16",
        "shape": tuple(shape),
        "expected": float32_to_bf16(values).reshape(shape),
        "rtol": rtol,
        "atol": atol,
    }


def vector_scalar_add():
    """Generates the 1024-element int32 ``input + 1`` case."""
    source = np.arange(-512, 512, dtype=np.int32)
    return {
        "name": "vector_scalar_add",
        "buffers": [
            {
                "name": "input",
                "direction": "input",
                "dtype": "int32",
                "shape": (1024,),
                "data": source,
            },
            {
                "name": "output",
                "direction": "output",
                "dtype": "int32",
                "shape": (1024,),
                "expected": source + np.int32(1),
                "rtol": 0.0,
                "atol": 0.0,
            },
        ],
    }


def vector_scalar_add_chain4():
    """Generates vector-add numerics for a four-run XADX entry point."""
    generated = vector_scalar_add()
    generated["name"] = "vector_scalar_add_chain4"
    return generated


def elementwise_add():
    """Generates the 2048-element BF16 elementwise-add case."""
    indices = np.arange(2048, dtype=np.int32)
    lhs_values = ((indices % 33) - 16).astype(np.float32) / np.float32(4)
    rhs_values = ((indices % 17) - 8).astype(np.float32) / np.float32(8)
    lhs = float32_to_bf16(lhs_values)
    rhs = float32_to_bf16(rhs_values)
    expected = bf16_to_float32(lhs) + bf16_to_float32(rhs)
    return {
        "name": "elementwise_add",
        "buffers": [
            _bf16_input("lhs", lhs_values, (2048,)),
            _bf16_input("rhs", rhs_values, (2048,)),
            _bf16_output("output", expected, (2048,)),
        ],
    }


def relu():
    """Generates the 2048-element BF16 ReLU case."""
    indices = np.arange(2048, dtype=np.int32)
    values = ((indices % 65) - 32).astype(np.float32) / np.float32(8)
    values[0] = np.float32(0)
    source = float32_to_bf16(values)
    expected = np.maximum(bf16_to_float32(source), np.float32(0))
    return {
        "name": "relu",
        "buffers": [
            _bf16_input("input", values, (2048,)),
            _bf16_output("output", expected, (2048,)),
        ],
    }


def gemv():
    """Generates the M=128, K=128 BF16 matrix-vector case."""
    rows = np.arange(128, dtype=np.int32)[:, None]
    columns = np.arange(128, dtype=np.int32)[None, :]
    matrix_values = (
        ((rows * 3 + columns * 5) % 17) - 8
    ).astype(np.float32) / np.float32(16)
    vector_values = (
        ((np.arange(128, dtype=np.int32) * 3) % 13) - 6
    ).astype(np.float32) / np.float32(8)
    matrix_values[0, :] = 0
    vector_values[0] = 0

    matrix = float32_to_bf16(matrix_values)
    vector = float32_to_bf16(vector_values)
    expected = bf16_to_float32(matrix) @ bf16_to_float32(vector)
    return {
        "name": "gemv",
        "buffers": [
            _bf16_input("matrix", matrix_values, (128, 128)),
            _bf16_input("vector", vector_values, (128,)),
            _bf16_output(
                "output", expected, (128,), rtol=0.02, atol=0.125
            ),
        ],
    }


CASE_GENERATORS = {
    "vector_scalar_add": vector_scalar_add,
    "vector_scalar_add_chain4": vector_scalar_add_chain4,
    "elementwise_add": elementwise_add,
    "relu": relu,
    "gemv": gemv,
}


def generate_case(name):
    """Generates one named suite case."""
    try:
        return CASE_GENERATORS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown test case: {name}") from exc

