#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Runs prebuilt XADX test packages directly through the public HRX ABI.

``manifest.json`` has this intentionally small schema::

    {
      "cases": [{
        "name": "vector_scalar_add",
        "xadx": "vector_scalar_add/kernel.xadx",
        "export": "MLIR_AIE",
        "buffers": [{
          "name": "input",
          "direction": "input",
          "dtype": "int32",
          "shape": [1024],
          "input": "vector_scalar_add/input.bin"
        }, {
          "name": "output",
          "direction": "output",
          "dtype": "int32",
          "shape": [1024],
          "expected": "vector_scalar_add/output.bin",
          "rtol": 0.0,
          "atol": 0.0
        }]
      }]
    }

Buffers are listed in kernel binding order. Data files are raw, contiguous
row-major arrays in the declared storage ``dtype``. BF16 buffers use
``dtype: "uint16"`` and ``logical_dtype: "bfloat16"``; comparisons decode
their raw storage to float32. Paths are relative to the package root.

This module depends only on the Python standard library, NumPy, and the sibling
``hrx_api`` helper.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from hrx_api import HrxRuntime


def _read_array(package_root, relative_path, dtype, shape):
    path = package_root / relative_path
    count = int(np.prod(shape, dtype=np.int64))
    array = np.fromfile(path, dtype=dtype)
    if array.size != count:
        raise ValueError(
            f"{path} has {array.size} elements; expected {count} for shape {shape}"
        )
    return array.reshape(shape)


def _bf16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def _comparison_arrays(actual, expected, logical_dtype):
    if logical_dtype == "bfloat16":
        return _bf16_to_float32(actual), _bf16_to_float32(expected)
    if logical_dtype not in (None, ""):
        raise ValueError(f"unsupported logical_dtype: {logical_dtype}")
    return actual, expected


def _compare(actual, expected, logical_dtype, rtol, atol):
    actual, expected = _comparison_arrays(actual, expected, logical_dtype)
    if np.issubdtype(actual.dtype, np.inexact):
        matches = np.isclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=False
        )
    else:
        matches = actual == expected
    if bool(np.all(matches)):
        return None

    flat_matches = matches.reshape(-1)
    first = int(np.flatnonzero(~flat_matches)[0])
    actual_value = actual.reshape(-1)[first]
    expected_value = expected.reshape(-1)[first]
    mismatch_count = int(np.count_nonzero(~matches))
    return (
        f"{mismatch_count}/{matches.size} mismatches; first at {first}: "
        f"got {actual_value}, expected {expected_value}"
    )


def _run_case(runtime, package_root, case):
    name = case["name"]
    executable = None
    buffers = []
    output_records = []
    try:
        executable = runtime.load_xadx((package_root / case["xadx"]).read_bytes())
        export_ordinal = executable.lookup_export(case["export"])

        for descriptor in case["buffers"]:
            buffer_name = descriptor["name"]
            direction = descriptor["direction"]
            if direction not in ("input", "output"):
                raise ValueError(
                    f"{name}.{buffer_name}: direction must be input or output"
                )
            dtype = np.dtype(descriptor["dtype"])
            shape = tuple(int(dimension) for dimension in descriptor["shape"])
            if descriptor.get("logical_dtype") == "bfloat16" and dtype != np.uint16:
                raise ValueError(
                    f"{name}.{buffer_name}: bfloat16 storage dtype must be uint16"
                )

            buffer = runtime.allocate_buffer(shape, dtype)
            buffers.append(buffer)
            if direction == "input":
                source = _read_array(
                    package_root, descriptor["input"], dtype, shape
                )
                np.copyto(buffer.array, source, casting="no")
            else:
                buffer.array.fill(0)
                expected = _read_array(
                    package_root, descriptor["expected"], dtype, shape
                )
                output_records.append((descriptor, buffer, expected))
            buffer.flush()

        runtime.dispatch(executable, export_ordinal, buffers)
        runtime.synchronize()

        failures = []
        for descriptor, buffer, expected in output_records:
            buffer.invalidate()
            failure = _compare(
                buffer.array,
                expected,
                descriptor.get("logical_dtype"),
                float(descriptor.get("rtol", 0.0)),
                float(descriptor.get("atol", 0.0)),
            )
            if failure:
                failures.append(f"{descriptor['name']}: {failure}")
        return failures
    finally:
        for buffer in reversed(buffers):
            buffer.close()
        if executable is not None:
            executable.close()


def run_package(package_root, selected_cases=None, runtime_factory=HrxRuntime, stdout=None):
    """Runs selected manifest cases and returns the number that failed."""
    package_root = Path(package_root)
    stdout = sys.stdout if stdout is None else stdout
    manifest = json.loads((package_root / "manifest.json").read_text("utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.json must contain a cases array")

    names = [case.get("name") for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("manifest case names must be unique")
    if selected_cases:
        requested = set(selected_cases)
        missing = requested.difference(names)
        if missing:
            raise ValueError(f"unknown selected case(s): {', '.join(sorted(missing))}")
        cases = [case for case in cases if case["name"] in requested]

    runtime = runtime_factory()
    failed = 0
    try:
        for case in cases:
            name = case.get("name", "<unnamed>")
            try:
                failures = _run_case(runtime, package_root, case)
            except Exception as exc:
                failures = [str(exc)]
            if failures:
                failed += 1
                print(f"{name}: FAIL", file=stdout)
                for failure in failures:
                    print(f"  {failure}", file=stdout)
            else:
                print(f"{name}: PASS", file=stdout)
    finally:
        runtime.close()
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="run only this case; may be repeated",
    )
    arguments = parser.parse_args(argv)
    try:
        return 1 if run_package(arguments.package_root, arguments.cases) else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

