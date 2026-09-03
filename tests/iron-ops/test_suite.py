# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from build_tools.iron import (
    DEFAULT,
    OPERATORS,
    Operator,
    host_compile_suite,
    require_runtime_image,
    select,
)
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
        "strided_copy",
        "mha",
        "swiglu_decode",
        "swiglu_prefill",
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
    assert not packaged & {
        "dequant",
        "rope",
        "mha",
        "strided_copy",
        "swiglu_decode",
        "swiglu_prefill",
    }


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


def test_mha_is_host_compiled_with_local_pytest_filter():
    mha = next(operator for operator in OPERATORS if operator.name == "mha")
    assert mha.host_compile is True
    assert mha.hrx_pytest_filter == "test_mha"
    compile_names = [operator.name for operator in host_compile_suite()]
    assert "mha" in compile_names
    assert "swiglu_decode" not in compile_names
    assert "swiglu_prefill" not in compile_names


def test_compile_only_ops_have_no_npu_filter():
    by_name = {operator.name: operator for operator in OPERATORS}
    assert by_name["swiglu_decode"].hrx_pytest_filter is None
    assert by_name["swiglu_prefill"].hrx_pytest_filter is None
    assert by_name["swiglu_decode"].host_compile is False
    assert by_name["swiglu_prefill"].host_compile is False
    # Packaged on NPU CI; IRON pytest has no matching non-extensive compile-shape case.
    for name in ("gemm", "softmax", "transpose", "repeat"):
        assert by_name[name].host_compile is True
        assert by_name[name].hrx_pytest_filter is None
    assert by_name["strided_copy"].hrx_pytest_filter == (
        "test_strided_copy and (contiguous or two_channels)"
    )


def test_compile_job_count_caps_auto_and_honors_jobs():
    from build_tools.iron import compile_job_count

    assert compile_job_count(0) == 1
    assert compile_job_count(1) == 1
    assert compile_job_count(20, jobs=1) == 1
    assert compile_job_count(20, jobs=3) == 3
    assert compile_job_count(2, jobs=8) == 2
    auto = compile_job_count(20)
    assert 1 <= auto <= 8


def test_require_runtime_image_rejects_empty_and_non_elf(tmp_path):
    missing = tmp_path / "missing.elf"
    with pytest.raises(RuntimeError, match="was not produced"):
        require_runtime_image(missing, elf=True)
    empty = tmp_path / "empty.elf"
    empty.write_bytes(b"")
    with pytest.raises(RuntimeError, match="was not produced"):
        require_runtime_image(empty, elf=True)
    junk = tmp_path / "junk.elf"
    junk.write_bytes(b"MZ")
    with pytest.raises(RuntimeError, match="is not an ELF image"):
        require_runtime_image(junk, elf=True)
    elf = tmp_path / "ok.elf"
    elf.write_bytes(b"\x7fELF" + b"\0" * 12)
    require_runtime_image(elf, elf=True)
    xclbin = tmp_path / "ok.xclbin"
    xclbin.write_bytes(b"xclb")
    require_runtime_image(xclbin)
