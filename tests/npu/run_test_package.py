#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Runs prebuilt XADX test packages directly through the public HRX ABI.

``manifest.json`` has this intentionally small schema::

    {
        "cases": [
            {
                "name": "vector_scalar_add",
                "xadx": "vector_scalar_add/kernel.xadx",
                "export": "MLIR_AIE",
                "buffers": [
                    {
                        "name": "input",
                        "direction": "input",
                        "dtype": "int32",
                        "shape": [1024],
                        "input": "vector_scalar_add/input.bin",
                    },
                    {
                        "name": "output",
                        "direction": "output",
                        "dtype": "int32",
                        "shape": [1024],
                        "expected": "vector_scalar_add/output.bin",
                        "rtol": 0.0,
                        "atol": 0.0,
                    },
                ],
            }
        ]
    }

Buffers are listed in kernel binding order. Data files are raw, contiguous
row-major arrays in the declared storage ``dtype``. BF16 buffers use
``dtype: "uint16"`` and ``logical_dtype: "bfloat16"``; comparisons decode
their raw storage to float32. Paths are relative to the package root.

This module depends only on the Python standard library, NumPy, and the
production-owned ``build_tools.hrx_runtime`` contract.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_tools.hrx_runtime import HrxRuntime


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
        matches = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
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


class PreparedCase:
    def __init__(self, case):
        self.case = case
        self.executable = None
        self.export_ordinal = None
        self.buffers = []
        self.output_records = []

    def close(self):
        for buffer in reversed(self.buffers):
            buffer.close()
        self.buffers.clear()
        if self.executable is not None:
            self.executable.close()
            self.executable = None


def _prepare_case(runtime, package_root, case):
    name = case["name"]
    prepared = PreparedCase(case)
    try:
        prepared.executable = runtime.load_xadx(
            (package_root / case["xadx"]).read_bytes()
        )
        prepared.export_ordinal = prepared.executable.lookup_export(case["export"])

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
            prepared.buffers.append(buffer)
            if direction == "input":
                source = _read_array(package_root, descriptor["input"], dtype, shape)
                np.copyto(buffer.array, source, casting="no")
            else:
                buffer.array.fill(0)
                expected = _read_array(
                    package_root, descriptor["expected"], dtype, shape
                )
                prepared.output_records.append((descriptor, buffer, expected))
            buffer.flush()
        return prepared
    except Exception:
        prepared.close()
        raise


def _dispatch_and_validate(runtime, prepared):
    for _, buffer, _ in prepared.output_records:
        buffer.array.fill(0)
        buffer.flush()
    runtime.dispatch(prepared.executable, prepared.export_ordinal, prepared.buffers)
    runtime.synchronize()

    failures = []
    for descriptor, buffer, expected in prepared.output_records:
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


def run_package(
    package_root,
    selected_cases=None,
    runtime_factory=HrxRuntime,
    stdout=None,
    lifecycle_rounds=5,
    resident_rounds=10,
):
    """Runs deterministic lifecycle and resident round-robin stress modes."""
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
    failures_by_name = {case["name"]: [] for case in cases}
    try:
        for round_index in range(lifecycle_rounds):
            for case in cases:
                prepared = None
                try:
                    prepared = _prepare_case(runtime, package_root, case)
                    failures_by_name[case["name"]].extend(
                        f"lifecycle round {round_index + 1}: {failure}"
                        for failure in _dispatch_and_validate(runtime, prepared)
                    )
                except Exception as exc:
                    failures_by_name[case["name"]].append(
                        f"lifecycle round {round_index + 1}: {exc}"
                    )
                finally:
                    if prepared is not None:
                        prepared.close()

        resident = []
        for case in cases:
            try:
                resident.append(_prepare_case(runtime, package_root, case))
            except Exception as exc:
                failures_by_name[case["name"]].append(f"resident setup: {exc}")
        try:
            for round_index in range(resident_rounds):
                for prepared in resident:
                    name = prepared.case["name"]
                    try:
                        failures_by_name[name].extend(
                            f"resident round {round_index + 1}: {failure}"
                            for failure in _dispatch_and_validate(runtime, prepared)
                        )
                    except Exception as exc:
                        failures_by_name[name].append(
                            f"resident round {round_index + 1}: {exc}"
                        )
        finally:
            for prepared in reversed(resident):
                prepared.close()
    finally:
        runtime.close()
    for case in cases:
        name = case["name"]
        failures = failures_by_name[name]
        print(f"{name}: {'FAIL' if failures else 'PASS'}", file=stdout)
        for failure in failures:
            print(f"  {failure}", file=stdout)
    return sum(bool(failures_by_name[case["name"]]) for case in cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="run only this case; may be repeated",
    )
    parser.add_argument("--lifecycle-rounds", type=int, default=5)
    parser.add_argument("--resident-rounds", type=int, default=10)
    arguments = parser.parse_args(argv)
    try:
        if arguments.lifecycle_rounds < 0 or arguments.resident_rounds < 0:
            raise ValueError("stress round counts must be nonnegative")
        return (
            1
            if run_package(
                arguments.package_root,
                arguments.cases,
                lifecycle_rounds=arguments.lifecycle_rounds,
                resident_rounds=arguments.resident_rounds,
            )
            else 0
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
