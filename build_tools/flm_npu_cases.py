# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Numeric cases for the flm ports in the prebuilt NPU package.

These oracles use NumPy. The module itself does not import the ``iron``
package: presubmit reads ``CASE_NAMES`` before iron-ops is installed. The case
builders import IRON after the compiler is on ``sys.path``. They stay out of
``npu_test_cases.CASE_GENERATORS``, which that same pytest job imports.
``package-npu-test-package`` appends every name here after ``test-iron-ops``
has produced the artifacts.

The decode layer's inputs depend on the token position, so it is two cases
of one compiled xclbin. The other six operators are one case each.
"""

import os
import sys

import numpy as np
from ml_dtypes import bfloat16

from build_tools import iron, paths
from build_tools.dispatch_transactions import generate_dispatch_transaction
from build_tools.npu_test_cases import float32_to_bf16

# Manifest names. flm.layer is two cases because each position has its own inputs.
CASE_NAMES = (
    "flm.gemm",
    "flm.dequant",
    "flm.layer-5",
    "flm.layer-700",
    "flm.lm_head",
    "flm.prefill_attn",
    "flm.prefill_sliding_attn",
    "flm.mm_prebuilt",
)

_OPERATORS = (
    "flm.gemm",
    "flm.dequant",
    "flm.layer",
    "flm.lm_head",
    "flm.prefill_attn",
    "flm.prefill_sliding_attn",
    "flm.mm_prebuilt",
)

_LAYER_CONTEXTS = (5, 700)
_LAYER_MAX_L = 1024
_PREFILL_RANGES = ((0, 128, 1024), (128, 256, 1024))
_SLIDING_RANGES = ((0, 128, 2048), (512, 1024, 2048))
_NPU2_COLS = 8
_NPU2_ROWS = 4
_GEMM_BUDGET = 0.004
_MM_BUDGET = 0.02
_PREFILL_ATOL = 0.25
_PREFILL_MEAN = 0.02
_LAYER_RTOL_X = 1e-2
_LAYER_RTOL_KV = 2e-3
_PROJECTION_ERROR = 0.025


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

    aie_utils.set_current_device(NPU2())
    kind = operator.name
    if kind == "flm.gemm":
        return [_frozen(stage, operator, build_dir, serializer, _gemm_case(False))]
    if kind == "flm.mm_prebuilt":
        return [_frozen(stage, operator, build_dir, serializer, _gemm_case(True))]
    if kind == "flm.dequant":
        return [_frozen(stage, operator, build_dir, serializer, _dequant_case())]
    if kind == "flm.lm_head":
        return [_frozen(stage, operator, build_dir, serializer, _lm_head_case())]
    if kind == "flm.layer":
        return _layer_cases(stage, operator, build_dir)
    if kind == "flm.prefill_attn":
        return [_attention_case(stage, operator, build_dir, sliding=False)]
    if kind == "flm.prefill_sliding_attn":
        return [_attention_case(stage, operator, build_dir, sliding=True)]
    raise RuntimeError(f"no NPU package case for {kind}")


def _frozen(stage, operator, build_dir, serializer, generated):
    from build_tools.npu_test_package import _single_artifact, write_case

    xclbin = _single_artifact(build_dir, "*.xclbin", "operator xclbin")
    insts = _single_artifact(build_dir, "*.bin", "operator transaction")
    iron.require_txn_devgen(insts, operator.device)
    return write_case(
        stage,
        generated,
        serializer.serialize_xadx(xclbin.read_bytes(), insts.read_bytes(), "MLIR_AIE"),
        _source(operator),
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


def _gemm_case(prebuilt):
    name = "flm.mm_prebuilt" if prebuilt else "flm.gemm"
    rng = np.random.default_rng(42)
    a = (rng.standard_normal((256, 512)) * np.float32(4)).astype(bfloat16)
    b = (rng.random((512, 1024)) * np.float32(4)).astype(bfloat16)
    # fp32 accumulate, then bf16, matching the kernel's output stage with no
    # epilogue. The packaged shapes use Epilogue.NONE.
    c = (a.astype(np.float32) @ b.astype(np.float32)).astype(bfloat16)
    if prebuilt:
        from iron.operators.flm.mm_prebuilt.op import MMPrebuilt

        operator = MMPrebuilt(M=256, K=512, N=1024)
        budget = _MM_BUDGET
    else:
        from iron.operators.flm.gemm.op import GEMM

        operator = GEMM(M=256, K=512, N=1024)
        budget = _GEMM_BUDGET
    packed = operator.pack_B(b)
    if packed.dtype == np.dtype(bfloat16):
        b_buffer = _bf16_buffer("B", "input", _bf16_bits(packed).reshape(512, 1024))
    else:
        b_buffer = _raw_buffer(
            "B", "input", np.asarray(packed).reshape(-1), packed.dtype
        )
    mass = float(
        512 * np.abs(a.astype(np.float32)).mean() * np.abs(b.astype(np.float32)).mean()
    )
    return {
        "name": name,
        "buffers": [
            _bf16_buffer("A", "input", _bf16_bits(a)),
            b_buffer,
            _bf16_buffer(
                "C",
                "output",
                _bf16_bits(c),
                rtol=0.04,
                atol=float(budget * mass),
            ),
        ],
    }


def _dequant_case():
    from iron.operators.flm.dequant.reference import random_q4nx, reference

    weights = random_q4nx(1024, 128, seed=0)
    packed = reference(weights, 1024, 128)
    return {
        "name": "flm.dequant",
        "buffers": [
            _raw_buffer("qw", "input", weights, "uint8"),
            _raw_buffer("packed", "output", packed, "uint8"),
        ],
    }


def _lm_head_case():
    from iron.operators.flm.lm_head.op import LMHead
    from iron.operators.flm.lm_head.reference import dequantize, reference
    from iron.operators.flm.q4nx import GROUP, K_TILE, M_TILE

    dim, vocab, softcap = 1536, 4096, 1000.0
    rng = np.random.default_rng(0)
    groups = K_TILE // GROUP
    n_blocks = vocab * dim // (M_TILE * K_TILE)
    params = np.concatenate(
        [
            rng.uniform(0, 0.02, (n_blocks, groups * M_TILE)),
            rng.uniform(-0.15, 0, (n_blocks, groups * M_TILE)),
        ],
        axis=1,
    )
    params = params.astype(bfloat16).view(np.uint16).astype("<u2").view(np.uint8)
    codes = rng.integers(0, 256, (n_blocks, M_TILE * K_TILE // 2), dtype=np.uint8)
    weights = np.concatenate([params, codes], axis=1).reshape(-1)
    token = np.concatenate(
        [rng.standard_normal(dim), rng.uniform(0.5, 1.5, dim)]
    ).astype(bfloat16)
    words = np.asarray(weights).reshape(-1).view(np.uint32)
    operator = LMHead(dim=dim, vocab=vocab, softcap=softcap)
    logits = np.asarray(operator.reference(words, token), dtype=np.float64)
    uncapped = reference(
        dequantize(weights, dim, vocab, _NPU2_COLS, _NPU2_ROWS),
        np.asarray(token, dtype=np.float64),
        1e30,
    )
    bound = _PROJECTION_ERROR * float(np.max(np.abs(uncapped)))
    token_bits = np.asarray(token, dtype=bfloat16).view(np.uint16)

    return {
        "name": "flm.lm_head",
        "buffers": [
            _bf16_buffer(
                "y",
                "output",
                float32_to_bf16(logits.astype(np.float32)),
                atol=bound,
            ),
            _raw_buffer("w", "input", words, "uint32"),
            _bf16_buffer("x", "input", token_bits),
        ],
    }


def _layer_cases(stage, operator, build_dir):
    from iron.operators.flm.layer.design import arg_sizes
    from iron.operators.flm.layer.op import DecodeLayer
    from iron.operators.flm.layer.reference import (
        generate_inputs,
        kv_row,
        kv_rows,
        layer_dims,
        reference,
    )

    from build_tools.npu_test_package import _single_artifact, write_dispatch_case

    kwargs = iron.resolve_build_kwargs(operator.build_kwargs)
    geometry = kwargs["geometry"]
    layer_type = kwargs["layer_type"]
    compiled = DecodeLayer(geometry=geometry, layer_type=layer_type)
    sizes = arg_sizes(geometry)
    dims = layer_dims(geometry, layer_type)
    library = _library(build_dir)
    xclbin = _single_artifact(build_dir, "*.xclbin", "operator xclbin").read_bytes()
    cases = []
    for context_len in _LAYER_CONTEXTS:
        x, proj, rms, rope_rms, kv = generate_inputs(
            geometry, layer_type, context_len, _LAYER_MAX_L, seed=0
        )
        x = _fit_u16(x, sizes["x"])
        rms = _fit_u16(rms, sizes["rms"])
        rope_rms = _fit_u16(rope_rms, sizes["rope_rms"])
        kv = _fit_u16(kv, sizes["kv"])
        proj_bits = _fit_bytes(proj, sizes["proj"] * np.dtype(np.uint16).itemsize)
        ref_x, ref_kv = reference(
            geometry,
            layer_type,
            x,
            proj_bits,
            rms,
            rope_rms,
            kv,
            context_len,
            _LAYER_MAX_L,
        )
        model_dim = dims["model_dim"]
        dk = dims["dk"]
        row = kv_row(geometry, layer_type, context_len)
        rows = kv_rows(geometry, layer_type, _LAYER_MAX_L)
        k_start = row * dk
        v_start = rows * dk + row * dk
        generated = {
            "name": f"flm.layer-{context_len}",
            "buffers": [
                _bf16_buffer("x", "inout", x),
                _bf16_buffer("proj", "input", proj_bits.view(np.uint16)),
                _bf16_buffer("rms", "input", rms),
                _bf16_buffer("rope_rms", "input", rope_rms),
                _bf16_buffer("kv", "inout", kv),
            ],
            "dispatches": [
                {
                    "name": f"context_{context_len}",
                    "transaction": _transaction(
                        library,
                        compiled,
                        {"context_len": context_len, "max_l": _LAYER_MAX_L},
                    ),
                    "expected": {
                        "x": np.asarray(ref_x, dtype=np.uint16).reshape(x.shape),
                        "kv": np.asarray(ref_kv, dtype=np.uint16).reshape(kv.shape),
                    },
                    "compare_slices": {
                        "x": [[0, model_dim]],
                        "kv": [[k_start, k_start + dk], [v_start, v_start + dk]],
                    },
                    "rel_l2": {"x": _LAYER_RTOL_X, "kv": _LAYER_RTOL_KV},
                }
            ],
        }
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


def _attention_case(stage, operator, build_dir, sliding):
    from iron.operators.flm.prefill_attn.op import (
        PrefillAttention,
        PrefillSlidingAttention,
    )
    from iron.operators.flm.prefill_attn.reference import reference

    from build_tools.npu_test_package import _single_artifact, write_dispatch_case

    kwargs = dict(operator.build_kwargs)
    cls = PrefillSlidingAttention if sliding else PrefillAttention
    compiled = cls(**kwargs)
    ranges = _SLIDING_RANGES if sliding else _PREFILL_RANGES
    dh = compiled.head_dim
    heads = compiled.num_heads
    kv_heads = compiled.num_kv_heads
    rng = np.random.default_rng(0)
    q = (rng.standard_normal(compiled.max_context * heads * dh) * 0.2).astype(bfloat16)
    max_l = max(item[2] for item in ranges)
    half = max_l * kv_heads * dh
    kv = np.zeros(2 * compiled.max_context * kv_heads * dh, dtype=bfloat16)
    kv[:half] = (rng.standard_normal(half) * 0.2).astype(bfloat16)
    kv[half : 2 * half] = rng.standard_normal(half).astype(bfloat16)
    q_bits = np.asarray(q).view(np.uint16)
    kv_bits = np.asarray(kv).view(np.uint16)
    out_bits = np.zeros(q_bits.shape, dtype=np.uint16)
    dispatches = []
    for begin, end, limit in ranges:
        active = (end - begin) * heads * dh
        expected = reference(
            q,
            kv,
            begin,
            end,
            limit,
            heads,
            kv_heads,
            dh,
            compiled.window,
        ).reshape(-1)
        full = np.zeros(out_bits.shape, dtype=np.uint16)
        full[:active] = float32_to_bf16(np.asarray(expected, dtype=np.float32))
        dispatches.append(
            {
                "name": f"L{begin}_{end}",
                "transaction": _transaction(
                    _library(build_dir),
                    compiled,
                    {"L_begin": begin, "L_end": end, "max_l": limit},
                ),
                "expected": {"out": full},
                "compare_prefix": {"out": active},
                "mean_atol": {"out": _PREFILL_MEAN},
            }
        )
    generated = {
        "name": operator.name,
        "buffers": [
            _bf16_buffer("out", "output", out_bits, atol=_PREFILL_ATOL),
            _bf16_buffer("q", "input", q_bits),
            _bf16_buffer("kv", "input", kv_bits),
        ],
        "dispatches": dispatches,
    }
    xclbin = _single_artifact(build_dir, "*.xclbin", "operator xclbin").read_bytes()
    return write_dispatch_case(stage, generated, xclbin, _source(operator))


def _transaction(library, operator, values):
    return generate_dispatch_transaction(
        library, list(operator.get_dispatch_params()), values
    )


def _library(directory):
    from build_tools.npu_test_package import _single_artifact

    suffix = ".dll" if os.name == "nt" else ".so"
    return _single_artifact(directory, f"*{suffix}", "dispatch library")


def _fit_u16(values, length):
    out = np.zeros(int(length), dtype=np.uint16)
    bits = np.asarray(values).reshape(-1).view(np.uint16)
    out[: bits.size] = bits
    return out


def _fit_bytes(values, nbytes):
    out = np.zeros(int(nbytes), dtype=np.uint8)
    blob = np.asarray(values, dtype=np.uint8).reshape(-1)
    out[: blob.size] = blob
    return out
