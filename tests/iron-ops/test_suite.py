# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from build_tools.iron import DEFAULT, OPERATORS, Operator, select
from build_tools.npu_test_cases import CASE_GENERATORS, generate_case

pytestmark = pytest.mark.presubmit

_BINARY = {"elementwise_add", "elementwise_mul", "axpy"}
_UNARY = {
    "relu",
    "gelu",
    "leaky_relu",
    "sigmoid",
    "tanh",
    "silu",
    "mem_copy",
    "layer_norm",
}


def _shapes_from_build_kwargs(operator: Operator):
    """Buffer names and shapes implied by OPERATORS.build_kwargs."""
    name = operator.name
    kwargs = operator.build_kwargs
    if name in _BINARY:
        size = (kwargs["size"],)
        return [("lhs", size), ("rhs", size), ("output", size)]
    if name in _UNARY:
        size = (kwargs["size"],)
        return [("input", size), ("output", size)]
    if name == "rms_norm":
        rows = kwargs["size"] // kwargs["tile_size"]
        tile = (rows, kwargs["tile_size"])
        return [("input", tile), ("output", tile)]
    if name == "gemv":
        return [
            ("matrix", (kwargs["M"], kwargs["K"])),
            ("vector", (kwargs["K"],)),
            ("output", (kwargs["M"],)),
        ]
    if name == "gemm":
        return [
            ("lhs", (kwargs["M"], kwargs["K"])),
            ("rhs", (kwargs["K"], kwargs["N"])),
            ("output", (kwargs["M"], kwargs["N"])),
        ]
    if name == "softmax":
        shape = (kwargs["rows"], kwargs["cols"])
        return [("input", shape), ("output", shape)]
    if name == "transpose":
        return [
            ("input", (kwargs["M"], kwargs["N"])),
            ("output", (kwargs["N"], kwargs["M"])),
        ]
    if name == "repeat":
        rows, cols, repeat = kwargs["rows"], kwargs["cols"], kwargs["repeat"]
        return [("input", (rows, cols)), ("output", (rows * repeat, cols))]
    raise AssertionError(f"{name}: no packaged shape contract")


def test_suite_covers_the_operator_catalog():
    names = [operator.name for operator in OPERATORS]
    assert names[0] == DEFAULT == "elementwise_add"
    assert names == sorted(set(names), key=names.index)
    expected = {
        "elementwise_add",
        "elementwise_mul",
        "axpy",
        "relu",
        "gelu",
        "leaky_relu",
        "sigmoid",
        "tanh",
        "layer_norm",
        "silu",
        "rms_norm",
        "dequant",
        "mem_copy",
        "gemv",
        "gemm",
        "softmax",
        "rope",
        "transpose",
        "repeat",
    }
    assert set(names) == expected
    for operator in OPERATORS:
        assert operator.build_kwargs
        assert operator.device == "npu2"


def test_select_filters_by_name():
    chosen = select("relu")
    assert len(chosen) == 1
    assert chosen[0].name == "relu"
    with pytest.raises(KeyError, match="not in the iron-ops suite"):
        select("not-an-operator")


def test_packaged_numeric_cases_cover_single_kernel_ops():
    packaged = {
        name for name in CASE_GENERATORS if name in {op.name for op in OPERATORS}
    }
    assert packaged == {
        "elementwise_add",
        "elementwise_mul",
        "axpy",
        "relu",
        "gelu",
        "leaky_relu",
        "sigmoid",
        "tanh",
        "silu",
        "layer_norm",
        "rms_norm",
        "mem_copy",
        "softmax",
        "gemv",
        "gemm",
        "transpose",
        "repeat",
    }
    assert not packaged & {"dequant", "rope"}


def test_packaged_generator_shapes_match_compile_kwargs():
    for operator in OPERATORS:
        if operator.name not in CASE_GENERATORS:
            continue
        expected = _shapes_from_build_kwargs(operator)
        actual = [
            (buffer["name"], tuple(buffer["shape"]))
            for buffer in generate_case(operator.name)["buffers"]
        ]
        assert actual == expected, operator.name
