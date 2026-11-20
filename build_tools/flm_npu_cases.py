# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Numeric cases for the flm ports in the prebuilt NPU package.

Inputs and expected outputs come from IRON: each operator's declared test
draw and ``reference()``, or, for the decode layer and prefill attention,
the input generators their own tests use. The module itself does not import
the ``iron`` package: presubmit reads ``CASE_NAMES`` before iron-ops is
installed. The case builders import IRON after the compiler is on
``sys.path``. They stay out of ``npu_test_cases.CASE_GENERATORS``, which
that same pytest job imports. ``package-npu-test-package`` appends every name
here after ``test-iron-ops`` has produced the artifacts.

The decode layer's inputs depend on the token position, so it is two cases
of one compiled xclbin. The other operators are one case each.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

from build_tools import iron, paths
from build_tools.dispatch_transactions import generate_dispatch_transaction

# Manifest names. flm.layer is two cases because each position has its own inputs.
CASE_NAMES = (
    "flm.gemm",
    "flm.dequant",
    "flm.layer-5",
    "flm.layer-700",
    "flm.lm_head",
    "flm.prefill_attn",
    "flm.prefill_sliding_attn",
)

_OPERATORS = (
    "flm.gemm",
    "flm.dequant",
    "flm.layer",
    "flm.lm_head",
    "flm.prefill_attn",
    "flm.prefill_sliding_attn",
)

_LAYER_CONTEXTS = (5, 700)
_LAYER_MAX_L = 1024
_PREFILL_RANGES = ((0, 128, 1024), (128, 256, 1024))
_SLIDING_RANGES = ((0, 128, 2048), (512, 1024, 2048))
# iron-ops' flm.layer and flm.prefill_attn tests bound the same errors.
_PREFILL_ATOL = 0.25
_PREFILL_MEAN = 0.02
_LAYER_RTOL_X = 1e-2
_LAYER_RTOL_KV = 2e-3


def append_cases(stage, cases, serializer, compiler_prefix=None):
    """Append one packaged case per flm artifact under ``out/iron-ops``."""
    _prepare_imports(compiler_prefix)
    by_name = {operator.name: operator for operator in iron.OPERATORS}
    for name in _OPERATORS:
        build_dir = paths.OUT / "iron-ops" / name
        cases.extend(_cases_for(stage, by_name[name], build_dir, serializer))


def _prepare_imports(compiler_prefix):
    """Put the compiler that built the artifacts, and iron-ops, on this process."""
    os.environ.setdefault("IRON_RUNTIME", "cpu")
    env = iron.iron_ops_env(compiler_prefix)
    os.environ.update(env)
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)


def _cases_for(stage, operator, build_dir, serializer):
    import aie.utils as aie_utils
    from aie.iron.device import NPU2

    device = NPU2()
    aie_utils.set_current_device(device)
    cls = iron.operator_class(operator.module or operator.name, operator.class_name)
    built = iron.construct_operator(
        cls, iron.resolve_build_kwargs(operator.build_kwargs)
    ).resolved(device)
    kind = operator.name
    if kind == "flm.gemm":
        generated = _gemm_case(built)
    elif kind == "flm.dequant":
        generated = _dequant_case(built)
    elif kind == "flm.lm_head":
        generated = _lm_head_case(built)
    elif kind == "flm.layer":
        return _layer_cases(stage, operator, built, build_dir)
    elif kind in ("flm.prefill_attn", "flm.prefill_sliding_attn"):
        return [_attention_case(stage, operator, built, build_dir)]
    else:
        raise RuntimeError(f"no NPU package case for {kind}")
    return [_frozen(stage, operator, built, build_dir, serializer, generated)]


def _frozen(stage, operator, built, build_dir, serializer, generated):
    from build_tools.npu_test_package import _single_artifact, write_case

    _require_binding_order(built, generated)
    xclbin = _single_artifact(build_dir, "*.xclbin", "operator xclbin")
    insts = _single_artifact(build_dir, "*.bin", "operator transaction")
    iron.require_txn_devgen(insts, operator.device)
    return write_case(
        stage,
        generated,
        serializer.serialize_xadx(xclbin.read_bytes(), insts.read_bytes(), "MLIR_AIE"),
        _source(operator),
    )


def _require_binding_order(built, generated):
    """The case binds the kernel's arguments in the operator's buffer order."""
    want = [buffer.name for buffer in built.buffers]
    got = [buffer["name"] for buffer in generated["buffers"]]
    if got != want:
        raise RuntimeError(
            f"{generated['name']}: case buffers {got} are not "
            f"{type(built).__name__}'s arguments {want}"
        )


def _source(operator, **extra):
    source = {
        "kind": "iron-ops",
        "name": operator.name,
        "device": operator.device,
        "build_kwargs": {
            key: value
            for key, value in operator.build_kwargs.items()
            if not hasattr(value, "name")
        },
    }
    source.update(extra)
    return source


def _bf16_bits(values):
    array = np.ascontiguousarray(values)
    if array.dtype != np.uint16:
        array = array.astype(bfloat16).view(np.uint16)
    return array


def _bf16_buffer(name, direction, bits, rtol=0.0, atol=0.0):
    bits = np.asarray(bits, dtype=np.uint16)
    record = {
        "name": name,
        "direction": direction,
        "dtype": "uint16",
        "logical_dtype": "bfloat16",
        "shape": tuple(bits.shape),
    }
    if direction in ("input", "inout"):
        record["data"] = bits
    if direction in ("output", "inout"):
        record["expected"] = bits
        record["rtol"] = rtol
        record["atol"] = atol
    return record


def _raw_buffer(name, direction, values, dtype, rtol=0.0, atol=0.0):
    values = np.asarray(values, dtype=dtype)
    record = {
        "name": name,
        "direction": direction,
        "dtype": np.dtype(dtype).name,
        "shape": tuple(values.shape),
    }
    if direction in ("input", "inout"):
        record["data"] = values
    if direction in ("output", "inout"):
        record["expected"] = values
        record["rtol"] = rtol
        record["atol"] = atol
    return record


def _declared_vectors(op):
    """The operator's own test vectors: its declared draw, then ``reference()``."""
    from iron.common.testing import vectors

    test = type(op).test
    if test is None:
        raise RuntimeError(f"{type(op).__name__} declares no test vectors")
    return vectors(op, **(test.draw(op) if test.draw is not None else {}))


def _bound(op, drawn):
    """The largest absolute error ``op.tolerance()`` allows on these inputs."""
    tolerance = op.tolerance()
    if tolerance is None or tolerance.bound is None:
        raise RuntimeError(f"{type(op).__name__} declares no bounded tolerance")
    return float(np.max(tolerance.bound(*(drawn.inputs[b.name] for b in op.inputs))))


def _gemm_case(op):
    """ProjectionGEMM declares no draw: IRON's ``flm_vectors`` draws A normal and
    B row-major ``(K, N)``, which the operator binds packed (``pack_b``)."""
    from iron.common.testing import vectors

    drawn = vectors(op, normal=(op.A,), given={op.B: (op.K, op.N)})
    packed = np.asarray(op.pack_b(drawn["B"]))
    b_buffer = (
        _raw_buffer("B", "input", packed.reshape(-1), "uint8")
        if packed.dtype == np.uint8
        else _bf16_buffer("B", "input", _bf16_bits(packed))
    )
    return {
        "name": "flm.gemm",
        "buffers": [
            _bf16_buffer("A", "input", _bf16_bits(drawn["A"])),
            b_buffer,
            _bf16_buffer("C", "output", _bf16_bits(drawn["C"]), atol=_bound(op, drawn)),
        ],
    }


def _dequant_case(op):
    drawn = _declared_vectors(op)
    return {
        "name": "flm.dequant",
        "buffers": [
            _raw_buffer("qw", "input", drawn["qw"], "uint8"),
            _raw_buffer("out", "output", drawn["out"], "uint8"),
        ],
    }


def _lm_head_case(op):
    drawn = _declared_vectors(op)
    return {
        "name": "flm.lm_head",
        "buffers": [
            _bf16_buffer("y", "output", _bf16_bits(drawn["y"]), atol=_bound(op, drawn)),
            _raw_buffer("w", "input", np.asarray(drawn["w"]).reshape(-1), "uint32"),
            _bf16_buffer("x", "input", _bf16_bits(drawn["x"])),
        ],
    }


def _fit(values, buffer):
    """``values`` at the start of a zeroed bf16 buffer of ``buffer``'s size."""
    out = np.zeros(int(np.prod(buffer.shape)), dtype=np.uint16)
    bits = np.ascontiguousarray(values).reshape(-1).view(np.uint8)
    fitted = min(bits.size, out.nbytes)
    out.view(np.uint8)[:fitted] = bits[:fitted]
    return out


def _layer_cases(stage, operator, op, build_dir):
    from iron.operators.flm.layer.reference import generate_inputs

    from build_tools.npu_test_package import _single_artifact, write_dispatch_case

    library = _library(build_dir)
    params = _dispatch_params(op)
    xclbin = _single_artifact(build_dir, "*.xclbin", "operator xclbin").read_bytes()
    g = op.dims
    model_dim, dk = g.geometry.model_dim, g.dk
    cases = []
    for context_len in _LAYER_CONTEXTS:
        drawn = generate_inputs(g, context_len, _LAYER_MAX_L, seed=0)
        bits = [
            _fit(values, buffer)
            for values, buffer in zip(drawn, op.buffers, strict=True)
        ]
        ref_x, ref_kv = op.reference(
            *(b.view(bfloat16) for b in bits), context_len, _LAYER_MAX_L
        )
        x, proj, rms, rope_rms, kv = bits
        row = g.kv_row(context_len)
        k_start = row * dk
        v_start = g.kv_rows(_LAYER_MAX_L) * dk + row * dk
        generated = {
            "name": f"flm.layer-{context_len}",
            "buffers": [
                _bf16_buffer("x", "inout", x),
                _bf16_buffer("proj", "input", proj),
                _bf16_buffer("rms", "input", rms),
                _bf16_buffer("rope_rms", "input", rope_rms),
                _bf16_buffer("kv", "inout", kv),
            ],
            "dispatches": [
                {
                    "name": f"context_{context_len}",
                    "transaction": generate_dispatch_transaction(
                        library,
                        params,
                        {"context_len": context_len, "max_l": _LAYER_MAX_L},
                    ),
                    "expected": {
                        "x": _bf16_bits(ref_x).reshape(x.shape),
                        "kv": _bf16_bits(ref_kv).reshape(kv.shape),
                    },
                    "compare_slices": {
                        "x": [[0, model_dim]],
                        "kv": [[k_start, k_start + dk], [v_start, v_start + dk]],
                    },
                    "rel_l2": {"x": _LAYER_RTOL_X, "kv": _LAYER_RTOL_KV},
                }
            ],
        }
        _require_binding_order(op, generated)
        # The buffer-level expected is the input. The dispatch expected is the
        # updated activation and cache; untouched rows must match bit for bit.
        generated["buffers"][0]["expected"] = x
        generated["buffers"][4]["expected"] = kv
        cases.append(
            write_dispatch_case(
                stage,
                generated,
                xclbin,
                _source(operator, context_len=context_len, max_l=_LAYER_MAX_L),
            )
        )
    return cases


def _attention_case(stage, operator, op, build_dir):
    from build_tools.npu_test_package import _single_artifact, write_dispatch_case

    ranges = _SLIDING_RANGES if op.window is not None else _PREFILL_RANGES
    dh = op.head_dim
    heads = op.num_heads
    rng = np.random.default_rng(0)
    q = (rng.standard_normal(op.max_context * heads * dh) * 0.2).astype(bfloat16)
    max_l = max(item[2] for item in ranges)
    half = max_l * op.num_kv_heads * dh
    kv = np.zeros(2 * op.max_context * op.num_kv_heads * dh, dtype=bfloat16)
    kv[:half] = (rng.standard_normal(half) * 0.2).astype(bfloat16)
    kv[half : 2 * half] = rng.standard_normal(half).astype(bfloat16)
    out_bits = np.zeros(q.shape, dtype=np.uint16)
    library = _library(build_dir)
    params = _dispatch_params(op)
    dispatches = []
    for begin, end, limit in ranges:
        active = (end - begin) * heads * dh
        expected = np.asarray(op.reference(q, kv, begin, end, limit)).reshape(-1)
        full = np.zeros(out_bits.shape, dtype=np.uint16)
        full[:active] = _bf16_bits(expected.astype(np.float32))[:active]
        dispatches.append(
            {
                "name": f"L{begin}_{end}",
                "transaction": generate_dispatch_transaction(
                    library,
                    params,
                    {"L_begin": begin, "L_end": end, "max_l": limit},
                ),
                "expected": {"o": full},
                "compare_prefix": {"o": active},
                "mean_atol": {"o": _PREFILL_MEAN},
            }
        )
    generated = {
        "name": operator.name,
        "buffers": [
            _bf16_buffer("o", "output", out_bits, atol=_PREFILL_ATOL),
            _bf16_buffer("q", "input", _bf16_bits(q)),
            _bf16_buffer("kv", "input", _bf16_bits(kv)),
        ],
        "dispatches": dispatches,
    }
    _require_binding_order(op, generated)
    xclbin = _single_artifact(build_dir, "*.xclbin", "operator xclbin").read_bytes()
    return write_dispatch_case(stage, generated, xclbin, _source(operator))


def _dispatch_params(op):
    """The dispatch library's parameters, in the order it takes them."""
    from iron.common.design import OperatorDesign

    return list(OperatorDesign(op, full_elf=False).compilable().dispatch_params)


def _library(directory):
    from build_tools.npu_test_package import _single_artifact

    suffix = ".dll" if os.name == "nt" else ".so"
    return _single_artifact(directory, f"*{suffix}", "dispatch library")
