# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic, standard-library-only buffers for XADX qualification."""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
from pathlib import Path


def _bf16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (bits >> 16) & 0xFFFF)


def _mm_weight(parameters: dict) -> bytes:
    m = int(parameters["m"])
    k = int(parameters["k"])
    salt = int(parameters["salt"])
    result = bytearray()
    for row in range(m):
        for column in range(k):
            column_word = (column // 8) % (k // 8)
            nibble = column % 8
            quantized = (row * 3 + column_word * 11 + nibble * 5 + salt * 7) % 8
            column_block = (column // 32) % (k // 32)
            scale = (1 + (row * 5 + column_block + salt) % 7) / 64.0
            result.extend(_bf16(quantized * scale))
    result.extend(bytes(int(parameters.get("padding", 4096))))
    return bytes(result)


def _activation_value(row: int, column: int, n: int, salt: int) -> bytes:
    index = row * n + column
    centered = (row * 13 + column * 7 + (row ^ column) * 3 + salt * 11) % 41 - 20
    return _bf16(0.0 if index % 53 == 0 else centered / 64.0)


def _activation_logical_index(physical_index: int) -> int:
    physical_to_logical = (0, 1, 2, 10, 11, 12, 13, 14, 15, 7, 8, 9, 3, 4, 5, 6, 16)
    logical = 0
    for physical_bit, logical_bit in enumerate(physical_to_logical):
        logical |= ((physical_index >> physical_bit) & 1) << logical_bit
    return logical


def _mm_activation(parameters: dict) -> bytes:
    k = int(parameters["k"])
    n = int(parameters["n"])
    salt = int(parameters["salt"])
    if k != 1024 or n != 128:
        raise ValueError("GEMM activation layout is defined for 1024x128")
    result = bytearray()
    for physical in range(k * n):
        logical = _activation_logical_index(physical)
        row, column = divmod(logical, n)
        result.extend(_activation_value(row, column, n, salt))
    return bytes(result)


def materialize(specification, root: Path) -> tuple[bytes, dict]:
    """Materializes a file/generator contract and returns audit metadata."""
    if isinstance(specification, str):
        specification = {"kind": "file", "path": specification}
    if not isinstance(specification, dict):
        raise ValueError("XADX buffer source must be a path or object")
    kind = specification.get("kind")
    if kind == "file":
        data = (root / specification["path"]).read_bytes()
    elif kind == "gzip":
        data = gzip.decompress((root / specification["path"]).read_bytes())
    elif kind == "generator":
        generator = specification.get("id")
        parameters = specification.get("parameters", {})
        if generator == "zero":
            data = bytes(int(parameters["size"]))
        elif generator == "mm-weight-bf16":
            data = _mm_weight(parameters)
        elif generator == "mm-activation-bf16":
            data = _mm_activation(parameters)
        else:
            raise ValueError(f"unknown XADX buffer generator: {generator}")
    else:
        raise ValueError(f"unknown XADX buffer source kind: {kind}")
    metadata = {
        "source": specification,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    metadata["contract_sha256"] = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return data, metadata
