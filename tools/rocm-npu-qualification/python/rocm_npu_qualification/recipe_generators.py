# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Buffers and oracles named by the kernel recipe.

Every buffer a recipe case needs is computed here rather than shipped, so the
qualification asset stays small and a generator change fails loudly against the
C++ contract instead of silently re-baselining what was qualified.

BF16 is carried as raw uint16 bits throughout, because those bits are the wire
format the device reads and writes. The device contract is bit-exact for
dequantization, so widening to float and back would invite a rounding difference
that only shows up as a comparison failure much later. Conversion goes through
ml_dtypes rather than hand-rolled bit arithmetic, so the rounding mode is the
library's problem and not a trick to re-derive when reading this.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

# Quantized value population -> (salt, nibble levels, populated zero point).
# The salt is part of the profile because it selects the value stream the C++
# contract generates for that population; it is not free to vary per case.
PROFILES = {
    "shared-subset": (29, 8, False),
    "full-nibble": (36, 16, False),
    "full-affine": (40, 16, True),
}

# Q4NX geometry, following the packer ROCm publishes in FLM_Q4NX_Converter. A
# chunk is row_block x col_block, row-chunk major, and holds block-major scales,
# then block-major zero points, then nibbles: the rows of a chunk split into
# groups of `parallel`, each group written column-major with two adjacent rows
# sharing a byte, the even row in the low nibble.
#
# The per-model configs in that repository differ only in these three numbers,
# which is what makes the overlays that reject the usual weights readable: they
# are the same format at a narrower column block, not a second format.
GROUP_COLUMNS = 32
Q4NX_BLOCKS = {"row_block": 32, "col_block": 256, "parallel": 16}

# A dequant kernel writes its result permuted rather than row-major, so an
# expectation has to be permuted the same way. The order belongs to the
# transaction and is measured, not derived, so each one is named.
MM_WEIGHT_BITS = (10, 11, 12, 3, 4, 5, 6, 7, 9, 0, 1, 2, 13, 14, 8, 16, 15, 17, 18, 19)
DEQUANT_OUTPUT_ORDERS = {
    "mm": MM_WEIGHT_BITS,
    "gpt-oss": (0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 7, 9, 8, 15, 16, 17, 18, 19),
    "gemma-text": (0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 7, 8, 9, 15, 16, 17, 18, 19),
}

# The physical activation layout belongs to the transaction, not the shape, so
# each builder gets its own entry. Every one is a bit permutation mapping a
# physical buffer index to the logical element it holds.
MM_ACTIVATION_LAYOUTS = {
    ("gemm", 128): (0, 1, 2, 10, 11, 12, 13, 14, 15, 7, 8, 9, 3, 4, 5, 6, 16),
    ("gemm", 512): (0, 1, 2, 12, 13, 14, 15, 16, 17, 9, 10, 11, 3, 4, 5, 6, 18, 7, 8),
    ("gpt-oss-gemm", 512): (
        9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 16, 17, 18, 7, 8,
    ),
    ("gemma-gemm", 512): (
        9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 16, 17, 18, 5, 6, 7, 8,
    ),
}

# The MM weight buffer is plain row-major with a trailing parameter block.
WEIGHT_PADDING = 4096
MAX_BUFFER_BYTES = 16 * 1024 * 1024
MAX_GENERATOR_WORKING_BYTES = 256 * 1024 * 1024


def bf16(values) -> np.ndarray:
    """Round-to-nearest-even float -> BF16 bits, matching FastFlowLM's bf16."""
    narrowed = np.asarray(values, dtype=np.float32).astype(ml_dtypes.bfloat16)
    return narrowed.view(np.uint16)


def widen(bits) -> np.ndarray:
    """BF16 bits -> float32."""
    return np.asarray(bits, dtype=np.uint16).view(ml_dtypes.bfloat16).astype(np.float32)


def permute(count: int, bits) -> np.ndarray:
    """Apply a bit permutation to every index below `count`."""
    source = np.arange(count, dtype=np.int64)
    result = np.zeros(count, dtype=np.int64)
    for position, destination in enumerate(bits):
        result |= ((source >> position) & 1) << destination
    return result


def source_tensors(rows: int, columns: int, profile: str):
    """The pre-conversion quantized tensors the C++ contract generates."""
    salt, levels, affine = PROFILES[profile]
    row = np.arange(rows, dtype=np.int64)[:, None]
    column = np.arange(columns, dtype=np.int64)[None, :]
    quantized = (
        row * 3 + ((column // 8) % (columns // 8)) * 11 + (column % 8) * 5 + salt * 7
    ) % levels

    block = np.arange(columns // GROUP_COLUMNS, dtype=np.int64)[None, :]
    scale = widen(bf16((1 + (row * 5 + block + salt) % 7) / 64.0))
    zero_point = (
        (row * 3 + block + salt) % 5 - 2 if affine else np.zeros_like(row + block)
    )
    return quantized, scale, zero_point.astype(np.int64)


def dequantize(quantized, scale, zero_point, semantics: str) -> np.ndarray:
    """The dequantization contracts the families disagree on, as BF16 bits."""
    wide_scale = np.repeat(scale, GROUP_COLUMNS, axis=1)
    wide_zero = np.repeat(zero_point, GROUP_COLUMNS, axis=1).astype(np.float32)
    unsigned = quantized.astype(np.float32)
    signed = np.where(quantized >= 8, quantized - 16, quantized).astype(np.float32)
    if semantics == "logical":
        return bf16((unsigned - wide_zero) * wide_scale)
    if semantics == "signed-nibble":
        return bf16((signed - wide_zero) * wide_scale)
    if semantics == "qwen-affine":
        return bf16(unsigned * wide_scale + wide_zero)
    if semantics == "lfm-affine":
        return bf16(signed * wide_scale)
    raise ValueError(f"unknown dequantization semantics: {semantics}")


def pack_q4nx(quantized, scale, zero_point, row_block=None, col_block=None,
              parallel=None) -> bytes:
    """Pack row-major quantization inputs into the Q4NX byte layout."""
    row_block = row_block or Q4NX_BLOCKS['row_block']
    col_block = col_block or Q4NX_BLOCKS['col_block']
    parallel = parallel or Q4NX_BLOCKS['parallel']
    rows, columns = quantized.shape
    if rows % row_block or columns % col_block or row_block % parallel:
        raise ValueError(
            f'a {rows}x{columns} tensor does not divide into {row_block}x{col_block} '
            f'chunks of {parallel} parallel rows'
        )
    if col_block % GROUP_COLUMNS or parallel % 2:
        raise ValueError(f'invalid Q4NX block shape {row_block}x{col_block}/{parallel}')
    row_chunks, column_chunks = rows // row_block, columns // col_block
    chunks = row_chunks * column_chunks

    def fields(values):
        """Scales and zero points are block-major within a chunk."""
        return (
            values.reshape(row_chunks, row_block, column_chunks,
                           col_block // GROUP_COLUMNS)
            .transpose(0, 2, 3, 1)
            .reshape(chunks, -1)
        )

    # Rows split into groups of parallel; each group is written column-major
    # with two adjacent rows sharing a byte, the even row in the low nibble.
    paired = (
        quantized.reshape(row_chunks, row_block, column_chunks, col_block)
        .transpose(0, 2, 1, 3)
        .reshape(chunks, row_block // parallel, parallel // 2, 2, col_block)
        .transpose(0, 1, 4, 2, 3)
        .astype(np.uint8)
        & 0xF
    )
    nibbles = (paired[..., 0] | (paired[..., 1] << 4)).reshape(chunks, -1)

    return np.concatenate(
        [
            fields(bf16(scale)).view(np.uint8),
            fields(bf16(zero_point)).view(np.uint8),
            nibbles,
        ],
        axis=1,
    ).reshape(-1).tobytes()


def dequant_expected(quantized, scale, zero_point, semantics, order) -> bytes:
    """The dequantized result in the order the kernel writes it."""
    bits = DEQUANT_OUTPUT_ORDERS.get(order)
    if bits is None:
        raise ValueError(f"no measured dequant output order named {order!r}")
    values = dequantize(quantized, scale, zero_point, semantics).reshape(-1)
    if values.size != 1 << len(bits):
        raise ValueError(
            f"the {order} output order was measured over {1 << len(bits)} weights, "
            f"not {values.size}; a different weight count is a different permutation"
        )
    return values[permute(values.size, bits)].tobytes()


def mm_weight(rows: int, columns: int, profile: str) -> bytes:
    """The dequantized weight the MM kernel reads, row-major plus padding."""
    quantized, scale, zero_point = source_tensors(rows, columns, profile)
    logical = dequantize(quantized, scale, zero_point, "logical")
    return logical.tobytes() + bytes(WEIGHT_PADDING)


def activation(k: int, n: int, salt: int) -> np.ndarray:
    """Logical KxN activation, as BF16 bits."""
    row = np.arange(k, dtype=np.int64)[:, None]
    column = np.arange(n, dtype=np.int64)[None, :]
    centered = (row * 13 + column * 7 + (row ^ column) * 3 + salt * 11) % 41 - 20
    return bf16(np.where((row * n + column) % 53 == 0, 0.0, centered / 64.0))


def mm_activation_layout(k: int, n: int, salt: int, builder: str) -> bytes:
    """The activation in the physical order this builder's transaction reads."""
    bits = MM_ACTIVATION_LAYOUTS.get((builder, n))
    if bits is None:
        raise ValueError(f"no measured activation layout for {builder} at n={n}")
    logical = activation(k, n, salt).reshape(-1)
    return logical[permute(k * n, bits)].tobytes()


def _product(weight: np.ndarray, k: int, n: int, salt: int) -> bytes:
    """A GEMM against the standard activation, as BF16 bits.

    The device accumulates in a different order than any single CPU order, so
    this pins the value rather than the summation: accumulate wide, then round
    once, and let the case tolerance absorb the device ordering difference.
    These generated BF16 values are bounded dyadic fractions with a common
    denominator of 4096; the sum of 1024 products fits well inside float64's
    53-bit significand. CPU/BLAS reassociation therefore cannot change this
    reference value, and Windows/Linux wheel byte parity checks that premise.
    """
    right = widen(activation(k, n, salt)).astype(np.float64)
    return bf16((weight.astype(np.float64) @ right).astype(np.float32)).tobytes()


def gemm_reference(m: int, k: int, n: int, profile: str, salt: int) -> bytes:
    """The CPU GEMM the MM cases are compared against, within tolerance."""
    quantized, scale, zero_point = source_tensors(m, k, profile)
    return _product(widen(dequantize(quantized, scale, zero_point, "logical")), k, n, salt)


def chained_reference(
    m: int, k: int, n: int, profile: str, salt: int, semantics: str, order: str
) -> bytes:
    """The product an MM stage forms from the weight a dequant stage wrote.

    A dequant kernel writes its result permuted while the MM kernel reads its
    weight row-major, so the chain is not the plain product. It is not opaque
    either: the MM kernel reads the bytes where they land, which makes the
    chained result the product of the weight exactly as it was written.
    """
    quantized, scale, zero_point = source_tensors(m, k, profile)
    written = np.frombuffer(
        dequant_expected(quantized, scale, zero_point, semantics, order),
        dtype=np.uint16,
    )
    return _product(widen(written).reshape(m, k), k, n, salt)


def predicted_size(generator: str, parameters: dict) -> int:
    """Validate a generator contract and predict bytes before allocating NumPy."""
    if not isinstance(parameters, dict):
        raise ValueError("recipe generator parameters must be an object")
    specifications = {
        "zero": ({"size"}, set()),
        "q4nx-packed": (
            {"m", "k", "profile", "row_block", "col_block", "parallel"},
            set(),
        ),
        "q4nx-dequant-expected": (
            {"m", "k", "profile", "semantics", "order"},
            set(),
        ),
        "chained-weight": (
            {"m", "k", "profile", "semantics", "order"},
            set(),
        ),
        "mm-weight": ({"m", "k", "profile"}, set()),
        "mm-activation-layout": ({"k", "n", "salt", "builder"}, set()),
        "gemm-reference": ({"m", "k", "n", "profile", "salt"}, set()),
        "chained-reference": (
            {"m", "k", "n", "profile", "salt", "semantics", "order"},
            set(),
        ),
    }
    if generator not in specifications:
        raise ValueError(f"unknown recipe generator: {generator}")
    required, optional = specifications[generator]
    if set(parameters) - required - optional or not required <= set(parameters):
        raise ValueError(f"{generator} has invalid parameter names")

    def integer(name: str, *, minimum: int = 1, maximum: int = 1 << 20) -> int:
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{generator}.{name} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"{generator}.{name} is outside its supported bounds")
        return value

    if generator == "zero":
        size = integer("size", maximum=MAX_BUFFER_BYTES)
    else:
        profile = parameters.get("profile")
        if profile is not None and profile not in PROFILES:
            raise ValueError(f"{generator}.profile is not supported")
        m = integer("m") if "m" in parameters else None
        k = integer("k") if "k" in parameters else None
        n = integer("n") if "n" in parameters else None
        if "salt" in parameters:
            integer("salt", minimum=0, maximum=255)
        semantics = parameters.get("semantics")
        if semantics is not None and semantics not in {
            "logical",
            "signed-nibble",
            "qwen-affine",
            "lfm-affine",
        }:
            raise ValueError(f"{generator}.semantics is not supported")
        order = parameters.get("order")
        if order is not None and order not in DEQUANT_OUTPUT_ORDERS:
            raise ValueError(f"{generator}.order is not supported")
        builder = parameters.get("builder")
        if builder is not None and (builder, n) not in MM_ACTIVATION_LAYOUTS:
            raise ValueError(f"{generator}.builder/n is not a measured layout")
        if generator == "q4nx-packed":
            row_block = integer("row_block")
            col_block = integer("col_block")
            parallel = integer("parallel")
            if (
                m % row_block
                or k % col_block
                or row_block % parallel
                or col_block % GROUP_COLUMNS
                or parallel % 2
            ):
                raise ValueError("q4nx-packed has invalid block divisibility")
            size = m * k * 5 // 8
        elif generator == "q4nx-dequant-expected":
            if m * k != 1 << len(DEQUANT_OUTPUT_ORDERS[order]):
                raise ValueError("dequant output dimensions do not match measured order")
            size = 2 * m * k
        elif generator == "chained-weight":
            if m * k != 1 << len(DEQUANT_OUTPUT_ORDERS[order]):
                raise ValueError("chained weight dimensions do not match measured order")
            size = 2 * m * k + WEIGHT_PADDING
        elif generator == "mm-weight":
            size = 2 * m * k + WEIGHT_PADDING
        elif generator == "mm-activation-layout":
            if k * n != 1 << len(MM_ACTIVATION_LAYOUTS[(builder, n)]):
                raise ValueError("activation dimensions do not match measured layout")
            size = 2 * k * n
        else:
            size = 2 * m * n
    if size > MAX_BUFFER_BYTES:
        raise ValueError(
            f"{generator} predicts {size} bytes, exceeding the 16 MiB limit"
        )
    if generator == "zero":
        working_elements = size
    elif generator in {
        "q4nx-packed",
        "q4nx-dequant-expected",
        "chained-weight",
        "mm-weight",
    }:
        working_elements = m * k
    elif generator == "mm-activation-layout":
        working_elements = k * n
    else:
        working_elements = m * k + k * n + m * n
    # NumPy expressions keep several int64/float32 temporaries alive at once.
    # A deliberately conservative multiplier makes this a safety boundary, not
    # an estimate of the final output allocation.
    bytes_per_element = 1 if generator == "zero" else 128
    if working_elements * bytes_per_element > MAX_GENERATOR_WORKING_BYTES:
        raise ValueError(
            f"{generator} exceeds the 256 MiB generator working-memory limit"
        )
    return size


def materialize(generator: str, parameters: dict) -> bytes:
    """Produce the bytes a recipe generator specification names."""
    size = predicted_size(generator, parameters)
    if generator == "zero":
        return bytes(size)

    profile = parameters.get("profile")
    if generator == "q4nx-packed":
        tensors = source_tensors(int(parameters["m"]), int(parameters["k"]), profile)
        return pack_q4nx(
            *tensors,
            row_block=parameters.get("row_block"),
            col_block=parameters.get("col_block"),
            parallel=parameters.get("parallel"),
        )

    if generator == "q4nx-dequant-expected":
        tensors = source_tensors(int(parameters["m"]), int(parameters["k"]), profile)
        return dequant_expected(
            *tensors, parameters["semantics"], parameters["order"]
        )

    if generator == "chained-weight":
        tensors = source_tensors(int(parameters["m"]), int(parameters["k"]), profile)
        return (
            dequant_expected(
                *tensors, parameters["semantics"], parameters["order"]
            )
            + bytes(WEIGHT_PADDING)
        )

    if generator == "mm-weight":
        return mm_weight(int(parameters["m"]), int(parameters["k"]), profile)

    if generator == "mm-activation-layout":
        return mm_activation_layout(
            int(parameters["k"]),
            int(parameters["n"]),
            int(parameters["salt"]),
            parameters["builder"],
        )

    if generator == "gemm-reference":
        return gemm_reference(
            int(parameters["m"]),
            int(parameters["k"]),
            int(parameters["n"]),
            profile,
            int(parameters["salt"]),
        )

    if generator == "chained-reference":
        return chained_reference(
            int(parameters["m"]),
            int(parameters["k"]),
            int(parameters["n"]),
            profile,
            int(parameters["salt"]),
            parameters["semantics"],
            parameters["order"],
        )

    raise ValueError(f"unknown recipe generator: {generator}")
