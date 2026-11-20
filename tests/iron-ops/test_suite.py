# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ast
import os

import pytest

from build_tools import paths
from build_tools.iron import (
    DEFAULT,
    OPERATORS,
    SEQUENCES,
    Operator,
    host_compile_suite,
    require_host_library,
    require_runtime_image,
    select,
)
from build_tools.npu_test_cases import CASE_GENERATORS, generate_case

pytestmark = pytest.mark.presubmit

_BINARY = {"elementwise_add", "elementwise_mul", "axpy"}
_UNARY = {"relu", "gelu", "sigmoid", "tanh", "silu"}
_ROWS = {"layer_norm", "rms_norm"}


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
    if name == "copy":
        size = (kwargs["input_buffer_size"],)
        return [("input", size), ("output", size)]
    if name in _ROWS:
        tile = (kwargs["rows"], kwargs["tile_size"])
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
    if name == "strided_copy":
        return [
            ("input", (kwargs["input_buffer_size"],)),
            ("output", (kwargs["length"],)),
            ("scratch", (1,)),
        ]
    raise AssertionError(f"{name}: no packaged shape contract")


def _declared_names(init):
    """The names an ``iron/operators`` ``__init__.py`` lists in ``__all__``."""
    for node in ast.parse(init.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError(f"{init}: no __all__")


def test_suite_covers_the_operator_catalog():
    names = [operator.name for operator in OPERATORS]
    assert names[0] == DEFAULT == "elementwise_add"
    assert names == sorted(set(names), key=names.index)
    for operator in OPERATORS:
        assert operator.build_kwargs
        assert operator.device == "npu2"
        assert operator.class_name, operator.name


def test_suite_classes_are_iron_s_whole_catalog():
    """Every IRON operator class is compiled here."""
    operators = paths.PROJ / "iron-ops" / "iron" / "operators"
    if not operators.is_dir():
        pytest.skip("iron-ops source is not checked out for this CI profile")
    iron_catalog = _declared_names(operators / "__init__.py") | {
        f"flm.{name}" for name in _declared_names(operators / "flm" / "__init__.py")
    }
    ours = {
        f"flm.{operator.class_name}"
        if (operator.module or operator.name).startswith("flm.")
        else operator.class_name
        for operator in OPERATORS
    }
    assert ours == iron_catalog


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
        "sigmoid",
        "tanh",
        "silu",
        "layer_norm",
        "rms_norm",
        "copy",
        "softmax",
        "gemv",
        "gemm",
        "strided_copy",
    }
    assert not packaged & {
        "rope",
        "mha",
        "flm.gemm",
        "flm.dequant",
        "flm.layer",
        "flm.lm_head",
        "flm.prefill_attn",
        "flm.prefill_sliding_attn",
    }


def test_packaged_fused_sequence_catalog():
    assert [sequence.name for sequence in SEQUENCES] == ["add_relu_fused"]
    sequence = SEQUENCES[0]
    assert (
        sequence.size,
        sequence.tile_size,
        sequence.num_aie_columns,
    ) == (4096, 1024, 4)
    assert sequence.name in CASE_GENERATORS


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


def test_mha_is_host_compiled_with_its_iron_case():
    mha = next(operator for operator in OPERATORS if operator.name == "mha")
    assert mha.host_compile is True
    assert mha.hrx_pytest_case == "npu2-MHA-num_heads_1-seq_len_16384-num_pipelines_8"
    assert "mha" in [operator.name for operator in host_compile_suite()]


def test_every_operator_is_host_compiled():
    assert host_compile_suite() == OPERATORS


def test_shapes_iron_declares_no_case_for_have_no_npu_case():
    by_name = {operator.name: operator for operator in OPERATORS}
    # Packaged on NPU CI; IRON's catalog has no case at this compile shape.
    for name in ("copy", "gemm", "softmax", "strided_copy"):
        assert by_name[name].host_compile is True
        assert by_name[name].hrx_pytest_case is None


def test_full_elf_entries_are_graphs():
    from build_tools.iron import FULL_ELF_GRAPHS

    assert {operator.name for operator in OPERATORS if operator.full_elf} == set(
        FULL_ELF_GRAPHS
    )


def test_flm_ports_use_the_smallest_upstream_shape():
    by_name = {operator.name: operator for operator in OPERATORS}
    compiled = {
        "flm.gemm",
        "flm.dequant",
        "flm.layer",
        "flm.lm_head",
        "flm.prefill_attn",
        "flm.prefill_sliding_attn",
    }
    for name in compiled:
        assert by_name[name].host_compile is True
        assert name in {operator.name for operator in host_compile_suite()}
    assert by_name["flm.prefill_attn"].class_name == "PrefillAttention"
    sliding = by_name["flm.prefill_sliding_attn"]
    assert sliding.module == "flm.prefill_attn"
    assert sliding.class_name == "PrefillSlidingAttention"
    assert by_name["flm.layer"].build_kwargs["geometry"] == "FLM_GEMMA4_E2B_DECODE"
    assert by_name["flm.gemm"].class_name == "ProjectionGEMM"
    assert by_name["flm.gemm"].build_kwargs == {
        "M": 256,
        "K": 512,
        "N": 1024,
        "precision": "high",
    }
    assert by_name["flm.dequant"].build_kwargs == {"K": 512, "N": 128}
    assert by_name["flm.lm_head"].build_kwargs == {
        "dim": 1536,
        "vocab": 4096,
        "softcap": 1000.0,
    }


def test_flm_ports_are_in_the_prebuilt_package():
    from build_tools.flm_npu_cases import CASE_NAMES
    from build_tools.npu_test_package import npu_case_names

    names = set(npu_case_names())
    assert set(CASE_NAMES) <= names
    assert {
        "flm.gemm",
        "flm.dequant",
        "flm.lm_head",
        "flm.prefill_attn",
        "flm.prefill_sliding_attn",
        "flm.layer-5",
        "flm.layer-700",
    } == set(CASE_NAMES)
    assert not set(CASE_NAMES) & set(CASE_GENERATORS)


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


def test_require_host_library_checks_the_host_image(tmp_path):
    missing = tmp_path / "missing.dll"
    with pytest.raises(RuntimeError, match="was not produced"):
        require_host_library(missing)
    empty = tmp_path / "empty.dll"
    empty.write_bytes(b"")
    with pytest.raises(RuntimeError, match="was not produced"):
        require_host_library(empty)
    junk = tmp_path / "junk.dll"
    junk.write_bytes(b"not a library")
    with pytest.raises(RuntimeError, match="is not a"):
        require_host_library(junk)
    library = tmp_path / ("ok.dll" if os.name == "nt" else "ok.so")
    library.write_bytes((b"MZ" if os.name == "nt" else b"\x7fELF") + b"\0" * 12)
    require_host_library(library)


class _Entry:
    def __init__(self, dispatch_library=None, params=None):
        self.dispatch_library = dispatch_library
        self.params = params


class _Artifacts:
    def __init__(self, *, full_elf, image, insts=None, entry=None, buffers=None):
        self.full_elf = full_elf
        self.image = image
        self.insts = insts
        self.entry = entry
        self.buffers = buffers or {}

    @property
    def params(self):
        return None if self.entry is None else self.entry.params


def test_classify_names_each_kind_of_image(tmp_path):
    from build_tools.iron import classify_compiled_images

    xclbin, insts, library, elf = (
        tmp_path / name for name in ("a.xclbin", "insts.bin", "d.dll", "a.elf")
    )
    assert classify_compiled_images(
        _Artifacts(full_elf=True, image=elf, entry=_Entry())
    ) == ("elf", [elf])
    assert classify_compiled_images(
        _Artifacts(full_elf=False, image=xclbin, insts=insts, entry=_Entry())
    ) == ("xclbin", [xclbin, insts])
    assert classify_compiled_images(
        _Artifacts(full_elf=False, image=xclbin, insts=insts, entry=_Entry(library))
    ) == ("dispatch", [xclbin, library])
    with pytest.raises(RuntimeError, match="neither an instruction stream"):
        classify_compiled_images(_Artifacts(full_elf=False, image=xclbin))


def test_stage_copies_a_full_elf_with_its_table_and_layout(tmp_path):
    import json

    from build_tools.iron import Compiled, stage_compiled_images

    cache = tmp_path / "cache"
    cache.mkdir()
    elf = cache / "design.elf"
    elf.write_bytes(b"\x7fELF")
    params = cache / "params.txt"
    params.write_text("1 1\ninput_offset 0 i32 addr - -\n0\n0\n")
    artifacts = _Artifacts(
        full_elf=True,
        image=elf,
        entry=_Entry(params=params),
        buffers={"x": ("input", 0, 4096), "y": ("output", 0, 2048)},
    )
    arguments = {"input": 4096, "output": 2048, "scratch": 0}
    kind, staged = stage_compiled_images(
        Compiled(artifacts, arguments), tmp_path / "out"
    )
    assert kind == "elf"
    assert [path.name for path in staged] == ["final.elf"]
    assert (tmp_path / "out" / "params.txt").read_text() == params.read_text()
    layout = json.loads((tmp_path / "out" / "layout.json").read_text())
    assert layout == {
        "arguments": arguments,
        "buffers": {"x": ["input", 0, 4096], "y": ["output", 0, 2048]},
    }


def test_stage_refuses_a_full_elf_without_a_table_or_arguments(tmp_path):
    from build_tools.iron import Compiled, stage_compiled_images

    elf = tmp_path / "design.elf"
    elf.write_bytes(b"\x7fELF")
    params = tmp_path / "params.txt"
    params.write_text("0 0\n0\n0\n")
    with pytest.raises(RuntimeError, match="without a parameter table"):
        stage_compiled_images(
            Compiled(_Artifacts(full_elf=True, image=elf, entry=_Entry()), {}),
            tmp_path / "out",
        )
    with pytest.raises(RuntimeError, match="without its argument sizes"):
        stage_compiled_images(
            Compiled(_Artifacts(full_elf=True, image=elf, entry=_Entry(params=params))),
            tmp_path / "out",
        )


class Gather:
    pass


class Copy:
    pass


class _Step:
    def __init__(self, op):
        self.op = op


class _Graph:
    """A graph whose compile records ``steps`` and allocates no arguments."""

    def __init__(self, *steps):
        self.steps = steps

    def compile(self, **_):
        class _Sizes:
            @staticmethod
            def arguments():
                return {}

        class _Sequence:
            buffer_sizes = _Sizes

        class _Recording:
            steps = [_Step(op) for op in self.steps]

        class _Compiled:
            artifacts = None
            recording = _Recording
            sequence = _Sequence

        return _Compiled


def test_full_elf_compile_requires_the_step_it_covers():
    from build_tools.iron import compile_full_elf

    compile_full_elf(_Graph(Copy(), Gather()), {}, None, "Gather")
    compile_full_elf(_Graph(Copy()), {}, None)
    with pytest.raises(RuntimeError, match="graph recorded Copy, not Gather"):
        compile_full_elf(_Graph(Copy()), {}, None, "Gather")
