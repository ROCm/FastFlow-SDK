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

from build_tools import npu_test_package
from build_tools.hrx_runtime import (
    HRX_AMDXDNA_DEVICE_CAPABILITY_FULL_ELF_LOAD_PDI,
    HrxRuntime,
)


def _runtime_supported_features(runtime):
    capabilities = runtime.amdxdna_capabilities()
    if capabilities & HRX_AMDXDNA_DEVICE_CAPABILITY_FULL_ELF_LOAD_PDI:
        return {npu_test_package.FULL_ELF_LOAD_PDI_FEATURE}
    return set()


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


def _rel_l2(actual, expected):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    return float(
        np.linalg.norm(actual - expected) / max(float(np.linalg.norm(expected)), 1e-30)
    )


def _check_region(descriptor, actual, expected, limits, failures, label):
    """Elementwise, mean, or relative-L2 check of one active region."""
    name = descriptor["name"]
    rel_limit = (limits.get("rel_l2") or {}).get(name)
    if rel_limit is not None:
        compared_actual, compared_expected = _comparison_arrays(
            actual, expected, descriptor.get("logical_dtype")
        )
        error = _rel_l2(compared_actual, compared_expected)
        if error > float(rel_limit):
            failures.append(f"{name}{label}: relative L2 {error:.4f} > {rel_limit}")
        return
    failure = _compare(
        actual,
        expected,
        descriptor.get("logical_dtype"),
        float(descriptor.get("rtol", 0.0)),
        float(descriptor.get("atol", 0.0)),
    )
    if failure:
        failures.append(f"{name}{label}: {failure}")
        return
    mean_limit = (limits.get("mean_atol") or {}).get(name)
    if mean_limit is None:
        return
    compared_actual, compared_expected = _comparison_arrays(
        actual, expected, descriptor.get("logical_dtype")
    )
    mean_error = float(
        np.mean(
            np.abs(
                compared_actual.astype(np.float64)
                - compared_expected.astype(np.float64)
            )
        )
    )
    if mean_error > float(mean_limit):
        failures.append(
            f"{name}{label}: mean abs error {mean_error:.4f} > {mean_limit}"
        )


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
        self.dispatches = []
        self.xclbin = b""

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
        prepared.xclbin = b""
        if case.get("xclbin"):
            prepared.xclbin = (package_root / case["xclbin"]).read_bytes()
        if case.get("xadx"):
            prepared.executable = runtime.load_xadx(
                (package_root / case["xadx"]).read_bytes()
            )
            prepared.export_ordinal = prepared.executable.lookup_export(case["export"])

        for descriptor in case["buffers"]:
            buffer_name = descriptor["name"]
            direction = descriptor["direction"]
            if direction not in ("input", "output", "inout"):
                raise ValueError(
                    f"{name}.{buffer_name}: direction must be input, output, or inout"
                )
            dtype = np.dtype(descriptor["dtype"])
            shape = tuple(int(dimension) for dimension in descriptor["shape"])
            if descriptor.get("logical_dtype") == "bfloat16" and dtype != np.uint16:
                raise ValueError(
                    f"{name}.{buffer_name}: bfloat16 storage dtype must be uint16"
                )

            buffer = runtime.allocate_buffer(shape, dtype)
            prepared.buffers.append(buffer)
            initial = None
            expected = None
            if direction in ("input", "inout"):
                initial = _read_array(package_root, descriptor["input"], dtype, shape)
                np.copyto(buffer.array, initial, casting="no")
            else:
                buffer.array.fill(0)
            if direction in ("output", "inout"):
                expected = _read_array(
                    package_root, descriptor["expected"], dtype, shape
                )
                prepared.output_records.append((descriptor, buffer, expected, initial))
            buffer.flush()
        for dispatch in case.get("dispatches", ()):
            expected = {}
            for descriptor, _, default_expected, _initial in prepared.output_records:
                relative_path = dispatch.get("expected", {}).get(descriptor["name"])
                expected[descriptor["name"]] = (
                    _read_array(
                        package_root,
                        relative_path,
                        np.dtype(descriptor["dtype"]),
                        tuple(int(dimension) for dimension in descriptor["shape"]),
                    )
                    if relative_path
                    else default_expected
                )
            if "transaction" in dispatch:
                payload = (package_root / dispatch["transaction"]).read_bytes()
                constants = b""
                transaction = payload
            else:
                constants = (package_root / dispatch["constants"]).read_bytes()
                transaction = b""
            prepared.dispatches.append(
                (
                    dispatch["name"],
                    constants,
                    transaction,
                    expected,
                    {
                        "compare_prefix": dispatch.get("compare_prefix") or {},
                        "mean_atol": dispatch.get("mean_atol") or {},
                        "compare_slices": dispatch.get("compare_slices") or {},
                        "rel_l2": dispatch.get("rel_l2") or {},
                    },
                )
            )
        return prepared
    except Exception:
        prepared.close()
        raise


def _dispatch_and_validate(runtime, prepared, dispatch_index=0):
    for descriptor, buffer, _expected, initial in prepared.output_records:
        if descriptor["direction"] == "inout":
            np.copyto(buffer.array, initial, casting="no")
        else:
            buffer.array.fill(0)
        buffer.flush()
    dispatch = (
        prepared.dispatches[dispatch_index % len(prepared.dispatches)]
        if prepared.dispatches
        else (None, b"", b"", {}, {})
    )
    _, constants, transaction, expected_by_name, limits = dispatch
    executable = prepared.executable
    export_ordinal = prepared.export_ordinal
    owned_executable = None
    if transaction:
        owned_executable = runtime.load_xclbin_transaction(
            prepared.xclbin, transaction, prepared.case["export"]
        )
        executable = owned_executable
        export_ordinal = executable.lookup_export(prepared.case["export"])
    try:
        runtime.dispatch(
            executable,
            export_ordinal,
            prepared.buffers,
            constants=constants,
        )
    finally:
        if owned_executable is not None:
            owned_executable.close()
    runtime.synchronize()

    failures = []
    for descriptor, buffer, default_expected, _initial in prepared.output_records:
        expected = np.asarray(
            expected_by_name.get(descriptor["name"], default_expected)
        ).reshape(-1)
        buffer.invalidate()
        actual = buffer.array.reshape(-1)
        slices = (limits.get("compare_slices") or {}).get(descriptor["name"])
        prefix = (limits.get("compare_prefix") or {}).get(descriptor["name"])
        if slices:
            mask = np.zeros(actual.shape, dtype=bool)
            for start, end in slices:
                start, end = int(start), int(end)
                mask[start:end] = True
                _check_region(
                    descriptor,
                    actual[start:end],
                    expected[start:end],
                    limits,
                    failures,
                    f"[{start}:{end}]",
                )
            if not np.array_equal(actual[~mask], expected[~mask]):
                failures.append(f"{descriptor['name']}: wrote outside its active rows")
        elif prefix is not None:
            prefix = int(prefix)
            _check_region(
                descriptor,
                actual[:prefix],
                expected[:prefix],
                limits,
                failures,
                "",
            )
            if not np.array_equal(actual[prefix:], expected[prefix:]):
                failures.append(f"{descriptor['name']}: wrote past its active rows")
        else:
            _check_region(descriptor, actual, expected, limits, failures, "")
    return failures


def run_package(
    package_root,
    selected_cases=None,
    runtime_factory=HrxRuntime,
    stdout=None,
    lifecycle_rounds=5,
    resident_rounds=10,
    supported_features=None,
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

    required_features = {
        feature for case in cases for feature in case.get("required_features", ())
    }
    runtime = None
    if supported_features is None:
        if required_features:
            runtime = runtime_factory()
            try:
                supported_features = _runtime_supported_features(runtime)
            except Exception:
                runtime.close()
                raise
        else:
            supported_features = set()
    supported_features = set(supported_features)
    skipped_by_name = {
        case["name"]: sorted(
            set(case.get("required_features", ())) - supported_features
        )
        for case in cases
        if set(case.get("required_features", ())) - supported_features
    }
    runnable_cases = [case for case in cases if case["name"] not in skipped_by_name]

    if runnable_cases and runtime is None:
        runtime = runtime_factory()
    failures_by_name = {case["name"]: [] for case in cases}

    def progress(message):
        print(f"[npu-test] {message}", file=stdout, flush=True)

    try:
        for round_index in range(lifecycle_rounds):
            for case in runnable_cases:
                name = case["name"]
                prepared = None
                progress(
                    f"lifecycle {round_index + 1}/{lifecycle_rounds} {name}: preparing"
                )
                try:
                    prepared = _prepare_case(runtime, package_root, case)
                    progress(
                        f"lifecycle {round_index + 1}/{lifecycle_rounds} "
                        f"{name}: dispatching"
                    )
                    failures_by_name[case["name"]].extend(
                        f"lifecycle round {round_index + 1}: {failure}"
                        for failure in _dispatch_and_validate(
                            runtime, prepared, round_index
                        )
                    )
                    progress(
                        f"lifecycle {round_index + 1}/{lifecycle_rounds} "
                        f"{name}: complete"
                    )
                except Exception as exc:
                    failures_by_name[case["name"]].append(
                        f"lifecycle round {round_index + 1}: {exc}"
                    )
                finally:
                    if prepared is not None:
                        prepared.close()

        resident = []
        for case in runnable_cases:
            name = case["name"]
            progress(f"resident setup {name}: preparing")
            try:
                resident.append(_prepare_case(runtime, package_root, case))
                progress(f"resident setup {name}: complete")
            except Exception as exc:
                failures_by_name[case["name"]].append(f"resident setup: {exc}")
        try:
            for round_index in range(resident_rounds):
                for prepared in resident:
                    name = prepared.case["name"]
                    progress(
                        f"resident {round_index + 1}/{resident_rounds} "
                        f"{name}: dispatching"
                    )
                    try:
                        failures_by_name[name].extend(
                            f"resident round {round_index + 1}: {failure}"
                            for failure in _dispatch_and_validate(
                                runtime, prepared, round_index
                            )
                        )
                        progress(
                            f"resident {round_index + 1}/{resident_rounds} "
                            f"{name}: complete"
                        )
                    except Exception as exc:
                        failures_by_name[name].append(
                            f"resident round {round_index + 1}: {exc}"
                        )
        finally:
            for prepared in reversed(resident):
                prepared.close()
    finally:
        if runtime is not None:
            runtime.close()
    for case in cases:
        name = case["name"]
        if name in skipped_by_name:
            print(
                f"{name}: SKIP (missing features: {', '.join(skipped_by_name[name])})",
                file=stdout,
            )
            continue
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
