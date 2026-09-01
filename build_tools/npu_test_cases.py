# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Deterministic NumPy-only data generators for packaged NPU tests.

Each generator returns a dictionary with a case ``name`` and an ordered
``buffers`` list. Buffer order is kernel binding order. Input descriptors carry
``data`` and output descriptors carry ``expected``; the remaining fields map
directly to the manifest schema consumed by ``run_test_package.py``.

BF16 is represented as raw ``numpy.uint16`` storage. No optional NumPy dtype
extensions are required.

``package-npu-test-package`` includes an iron-ops compile artifact for every
operator name that also appears here. dequant and rope stay compile-only until
they get generators. MHA and SwiGLU are not in the CPU compile list (torch).
"""


def _numpy():
    """Import NumPy only when numeric cases are generated."""
    import numpy

    return numpy


def float32_to_bf16(values):
    """Converts float32 values to raw BF16 with round-to-nearest-even."""
    np = _numpy()
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
    np = _numpy()
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
    np = _numpy()
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
    np = _numpy()
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
    np = _numpy()
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
    np = _numpy()
    rows = np.arange(128, dtype=np.int32)[:, None]
    columns = np.arange(128, dtype=np.int32)[None, :]
    matrix_values = (((rows * 3 + columns * 5) % 17) - 8).astype(
        np.float32
    ) / np.float32(16)
    vector_values = (((np.arange(128, dtype=np.int32) * 3) % 13) - 6).astype(
        np.float32
    ) / np.float32(8)
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
            _bf16_output("output", expected, (128,), rtol=0.02, atol=0.125),
        ],
    }


def _signed_ramp(size):
    """Deterministic signed BF16-friendly ramp used by the unary cases."""
    np = _numpy()
    indices = np.arange(size, dtype=np.int32)
    values = ((indices % 65) - 32).astype(np.float32) / np.float32(8)
    values[0] = np.float32(0)
    return values


def _unary_case(name, expected_from_rounded, *, size=2048, rtol=0.02, atol=0.02):
    values = _signed_ramp(size)
    source = float32_to_bf16(values)
    expected = expected_from_rounded(bf16_to_float32(source))
    return {
        "name": name,
        "buffers": [
            _bf16_input("input", values, (size,)),
            _bf16_output("output", expected, (size,), rtol=rtol, atol=atol),
        ],
    }


def elementwise_mul():
    """Generates the 2048-element BF16 elementwise-mul case."""
    generated = elementwise_add()
    lhs = generated["buffers"][0]["data"]
    rhs = generated["buffers"][1]["data"]
    expected = bf16_to_float32(lhs) * bf16_to_float32(rhs)
    generated["name"] = "elementwise_mul"
    generated["buffers"][2] = _bf16_output("output", expected, (2048,))
    return generated


def axpy():
    """Generates the 2048-element BF16 AXPY case ``3 * A + B``."""
    np = _numpy()
    generated = elementwise_add()
    lhs = generated["buffers"][0]["data"]
    rhs = generated["buffers"][1]["data"]
    expected = np.float32(3) * bf16_to_float32(lhs) + bf16_to_float32(rhs)
    generated["name"] = "axpy"
    generated["buffers"][2] = _bf16_output("output", expected, (2048,))
    return generated


def gelu():
    """Generates the 2048-element BF16 GELU (tanh approximation) case."""
    np = _numpy()

    def expected(x):
        inner = np.sqrt(np.float32(2) / np.float32(np.pi)) * (
            x + np.float32(0.044715) * x * x * x
        )
        return np.float32(0.5) * x * (np.float32(1) + np.tanh(inner))

    return _unary_case("gelu", expected)


def leaky_relu():
    """Generates the 2048-element BF16 leaky-ReLU case (alpha=0.01)."""
    np = _numpy()

    def expected(x):
        return np.where(x >= 0, x, np.float32(0.01) * x)

    return _unary_case("leaky_relu", expected)


def sigmoid():
    """Generates the 2048-element BF16 sigmoid case."""
    np = _numpy()

    def expected(x):
        return np.float32(1) / (np.float32(1) + np.exp(-x))

    return _unary_case("sigmoid", expected)


def tanh():
    """Generates the 2048-element BF16 tanh case."""
    np = _numpy()
    return _unary_case("tanh", np.tanh, rtol=0.05, atol=0.05)


def silu():
    """Generates the 2048-element BF16 SiLU case."""
    np = _numpy()

    def expected(x):
        return x / (np.float32(1) + np.exp(-x))

    return _unary_case("silu", expected)


def mem_copy():
    """Generates the 2048-element BF16 identity copy case."""
    return _unary_case("mem_copy", lambda x: x, rtol=0.0, atol=0.0)


def layer_norm():
    """Generates one-row BF16 layer-norm over 2048 columns, no affine params.

    rtol/atol 0.1 is the observed NPU slack at this shape; the other unaries
    sit at 0.02. Do not tighten without a passing on-device re-run.
    """
    np = _numpy()
    values = _signed_ramp(2048)
    source = bf16_to_float32(float32_to_bf16(values))
    mean = source.mean()
    var = source.var()
    expected = (source - mean) / np.sqrt(var + np.float32(1e-5))
    return {
        "name": "layer_norm",
        "buffers": [
            _bf16_input("input", values, (2048,)),
            _bf16_output("output", expected, (2048,), rtol=0.1, atol=0.1),
        ],
    }


def rms_norm():
    """Generates unweighted BF16 RMSNorm over one 2048-wide tile."""
    np = _numpy()
    values = _signed_ramp(2048).reshape(1, 2048)
    source = bf16_to_float32(float32_to_bf16(values))
    rms = np.sqrt(np.mean(source * source, axis=-1, keepdims=True) + np.float32(1e-5))
    expected = source / rms
    return {
        "name": "rms_norm",
        "buffers": [
            _bf16_input("input", values, (1, 2048)),
            _bf16_output("output", expected, (1, 2048), rtol=0.02, atol=0.02),
        ],
    }


def softmax():
    """Generates the 32x64 BF16 row-wise softmax case."""
    np = _numpy()
    rows = np.arange(32, dtype=np.int32)[:, None]
    columns = np.arange(64, dtype=np.int32)[None, :]
    values = (((rows * 3 + columns) % 17) - 8).astype(np.float32) / np.float32(4)
    source = bf16_to_float32(float32_to_bf16(values))
    shifted = source - source.max(axis=-1, keepdims=True)
    expo = np.exp(shifted)
    expected = expo / expo.sum(axis=-1, keepdims=True)
    return {
        "name": "softmax",
        "buffers": [
            _bf16_input("input", values, (32, 64)),
            _bf16_output("output", expected, (32, 64), rtol=0.02, atol=0.02),
        ],
    }


def gemm():
    """Generates the M=256, K=64, N=64 BF16 matmul case."""
    np = _numpy()
    m = np.arange(256, dtype=np.int32)[:, None]
    k = np.arange(64, dtype=np.int32)[None, :]
    n = np.arange(64, dtype=np.int32)[None, :]
    a_values = (((m * 3 + k) % 17) - 8).astype(np.float32) / np.float32(16)
    b_values = (((k.T * 5 + n) % 13) - 6).astype(np.float32) / np.float32(16)
    a_values[0, :] = 0
    b_values[:, 0] = 0
    a = float32_to_bf16(a_values)
    b = float32_to_bf16(b_values)
    expected = bf16_to_float32(a) @ bf16_to_float32(b)
    return {
        "name": "gemm",
        "buffers": [
            _bf16_input("lhs", a_values, (256, 64)),
            _bf16_input("rhs", b_values, (64, 64)),
            _bf16_output("output", expected, (256, 64), rtol=0.02, atol=0.125),
        ],
    }


def transpose():
    """Generates the 64x64 BF16 transpose case."""
    np = _numpy()
    rows = np.arange(64, dtype=np.int32)[:, None]
    columns = np.arange(64, dtype=np.int32)[None, :]
    values = (((rows * 3 + columns) % 17) - 8).astype(np.float32) / np.float32(8)
    source = bf16_to_float32(float32_to_bf16(values))
    expected = source.T.copy()
    return {
        "name": "transpose",
        "buffers": [
            _bf16_input("input", values, (64, 64)),
            _bf16_output("output", expected, (64, 64), rtol=0.0, atol=0.0),
        ],
    }


def repeat():
    """Generates the 32x1024 BF16 repeat-interleave-by-2 case."""
    np = _numpy()
    rows = np.arange(32, dtype=np.int32)[:, None]
    columns = np.arange(1024, dtype=np.int32)[None, :]
    values = (((rows * 3 + columns) % 17) - 8).astype(np.float32) / np.float32(8)
    source = bf16_to_float32(float32_to_bf16(values))
    expected = np.repeat(source, 2, axis=0)
    return {
        "name": "repeat",
        "buffers": [
            _bf16_input("input", values, (32, 1024)),
            _bf16_output("output", expected, (64, 1024), rtol=0.0, atol=0.0),
        ],
    }


CASE_GENERATORS = {
    "vector_scalar_add": vector_scalar_add,
    "vector_scalar_add_chain4": vector_scalar_add_chain4,
    "elementwise_add": elementwise_add,
    "elementwise_mul": elementwise_mul,
    "axpy": axpy,
    "relu": relu,
    "gelu": gelu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "silu": silu,
    "layer_norm": layer_norm,
    "rms_norm": rms_norm,
    "mem_copy": mem_copy,
    "softmax": softmax,
    "gemv": gemv,
    "gemm": gemm,
    "transpose": transpose,
    "repeat": repeat,
}


def generate_case(name):
    """Generates one named suite case."""
    try:
        return CASE_GENERATORS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown test case: {name}") from exc
