# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recipe execution, including the staging a chained case depends on.

The point of a multi-stage case is that a later stage reads the buffer an earlier
stage wrote, with no host round trip. That is invisible in a passing result on
hardware, so it is pinned here against a fake runtime that reports what each
dispatch actually saw.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from rocm_npu_qualification import xadx
from rocm_npu_qualification.contracts import KERNEL_RECIPE_SCHEMA_VERSION

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]

SIZE = 8
XADX_DATA = b"stand-in"
XADX_SHA256 = hashlib.sha256(XADX_DATA).hexdigest()
TRANSACTION_DATA = b"transaction"
TRANSACTION_SHA256 = hashlib.sha256(TRANSACTION_DATA).hexdigest()


def test_importing_xadx_does_not_execute_numeric_extensions():
    python_path = str(Path(xadx.__file__).resolve().parents[1])
    inherited_path = os.environ.get("PYTHONPATH")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import rocm_npu_qualification.xadx; "
                "assert 'numpy' not in sys.modules; "
                "assert 'ml_dtypes' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": (
                python_path
                if not inherited_path
                else os.pathsep.join((python_path, inherited_path))
            ),
        },
        text=True,
    )
    assert result.returncode == 0, result.stderr
GENERATOR_VALUES = {"swapped": b"\x11" * SIZE}


def _bf16_bytes(*values):
    return np.asarray(values, dtype=ml_dtypes.bfloat16).view(np.uint16).tobytes()


class StagingRuntime:
    """Records what every dispatch read, and lets a stage write a signature."""

    def __init__(self, architecture="AMD_AIE2_NPU2", writes=None, corrupt=None):
        self.architecture_name = architecture
        self.writes = writes or {}
        self.corrupt = corrupt or {}
        self.resources = []
        self.dispatches = []
        self.observed = []
        self.events = []
        self.payload_inspections = []
        self.inspection_dispatch_counts = []
        self.payload_reads = []

    def architecture(self):
        return self.architecture_name

    def load(self, data, export):
        self.resources.append(("executable", len(self.resources)))
        self.events.append(f"load:{export}")
        return f"executable-{export}", 0

    def allocate(self, data):
        self.resources.append(("buffer", len(self.resources)))
        handle = f"handle-{len(self.resources)}"
        backing = bytearray([xadx._GUARD_VALUE]) * (
            len(data) + 2 * xadx._GUARD_BYTES
        )
        storage = memoryview(backing)[
            xadx._GUARD_BYTES : xadx._GUARD_BYTES + len(data)
        ]
        storage[:] = data
        return xadx._GuardedBuffer(
            handle, storage, backing, xadx._GUARD_BYTES
        )

    def run(self, executable, ordinal, buffers):
        self.events.append(f"run:{executable}")
        self.dispatches.append(executable)
        self.observed.append([bytes(storage) for _, storage in buffers])
        payload = self.writes.get(executable)
        if payload is not None:
            buffers[0][1][: len(payload)] = payload
        corruption = self.corrupt.get(executable)
        if corruption == "leading":
            buffers[0].backing[0] ^= 0xFF
        elif corruption == "trailing":
            buffers[0].backing[-1] ^= 0xFF
        elif corruption == "input":
            buffers[-1][1][0] ^= 0xFF

    def read(self, buffer):
        payload = bytes(buffer[1])
        self.payload_reads.append(payload)
        return payload

    def inspect(self, buffer):
        end = buffer.offset + len(buffer.storage)
        self.events.append("inspect")
        self.payload_inspections.append(bytes(buffer.storage))
        self.inspection_dispatch_counts.append(
            (bytes(buffer.storage), len(self.dispatches))
        )
        return (
            bytes(buffer.backing[: buffer.offset]),
            bytes(buffer.storage),
            bytes(buffer.backing[end:]),
        )

    def inspect_guards(self, buffer):
        end = buffer.offset + len(buffer.storage)
        return (
            bytes(buffer.backing[: buffer.offset]),
            bytes(buffer.backing[end:]),
        )

    def close(self):
        self.resources.clear()

    def _check(self, status, operation):
        if status:
            raise xadx.XadxError(operation)

    def _flush(self, handle, offset, length):
        return 0


class NoncoherentRuntime(StagingRuntime):
    """Keeps device writes hidden until an explicit payload inspection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device_payloads = {}

    def run(self, executable, ordinal, buffers):
        self.events.append(f"run:{executable}")
        self.dispatches.append(executable)
        self.observed.append(
            [
                self.device_payloads.get(id(allocation), bytes(allocation[1]))
                for allocation in buffers
            ]
        )
        payload = self.writes.get(executable)
        if payload is not None:
            self.device_payloads[id(buffers[0])] = payload
        if self.corrupt.get(executable) == "input":
            allocation = buffers[-1]
            payload = bytearray(
                self.device_payloads.get(id(allocation), bytes(allocation[1]))
            )
            payload[0] ^= 0xFF
            self.device_payloads[id(allocation)] = bytes(payload)

    def inspect(self, buffer):
        payload = self.device_payloads.get(id(buffer))
        if payload is not None:
            buffer[1][:] = payload
        return super().inspect(buffer)


class FailingPreparationRuntime(StagingRuntime):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allocation_attempts = 0
        self.resource_counts_at_dispatch = []

    def allocate(self, data):
        self.allocation_attempts += 1
        if self.allocation_attempts == 2:
            raise xadx.XadxError("simulated resident allocation failure")
        return super().allocate(data)

    def run(self, executable, ordinal, buffers):
        self.resource_counts_at_dispatch.append(len(self.resources))
        return super().run(executable, ordinal, buffers)


def generator(id="zero", **parameters):
    size = int(parameters.get("size", SIZE))
    data = GENERATOR_VALUES.get(id, bytes(size))
    generator_id = id
    if id == "swapped":
        generator_id = "q4nx-packed"
        parameters = {
            "m": 32,
            "k": 128,
            "profile": "shared-subset",
            "row_block": 32,
            "col_block": 128,
            "parallel": 16,
        }
    elif id == "boom":
        generator_id = "mm-weight"
        parameters = {"m": 1, "k": 1, "profile": "shared-subset"}
    elif id == "short":
        generator_id = "gemm-reference"
        parameters = {
            "m": 1,
            "k": 1,
            "n": 1,
            "profile": "shared-subset",
            "salt": 0,
        }
    return {
        "id": generator_id,
        "parameters": {"size": SIZE, **parameters}
        if generator_id == "zero"
        else parameters,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def implementation_id(alias):
    digest = hashlib.sha256(f"xclbin-{alias}".encode()).hexdigest()
    return f"dequant-{digest[:16]}"


class Generators:
    """Stands in for recipe_generators, keyed by generator id."""

    def __init__(self, values=None):
        self.values = GENERATOR_VALUES if values is None else values

    def materialize(self, id, parameters):
        test_id = {
            "q4nx-packed": "swapped",
            "mm-weight": "boom",
            "gemm-reference": "short",
        }.get(id, id)
        if test_id == "boom":
            raise ValueError("this generator refuses")
        if test_id in self.values:
            return self.values[test_id]
        return bytes(int(parameters["size"]))

def recipe(cases, implementations=("first", "second")):
    return {
        "schema_version": KERNEL_RECIPE_SCHEMA_VERSION,
        "product": "rocm-npu-fastflowlm-kernel-recipe",
        "target": "npu2",
        "generator_version": "kernel-recipe-generator-v1",
        "coverage_limits": [],
        "implementations": [
            {
                "id": implementation_id(alias),
                "kernel": "dequant",
                "builder": alias,
                "shape": {"m": 1, "k": 1, "n": 1},
                "families": ["test"],
                "xclbin_sha256": hashlib.sha256(
                    f"xclbin-{alias}".encode()
                ).hexdigest(),
                "xclbin_path": f"xclbins/{alias}.xclbin",
                "transaction": "transactions/kernel.bin",
                "transaction_sha256": TRANSACTION_SHA256,
                "xadx": "xadx/kernel.xadx",
                "xadx_sha256": XADX_SHA256,
                "export": alias,
            }
            for alias in implementations
        ],
        "cases": cases,
    }


def stage(
    name,
    implementation,
    *,
    expected=None,
    preserved=None,
    bind=None,
    buffers=(),
):
    records = [
        {
            "name": "output",
            "direction": "output",
            "size": SIZE,
            **(
                {"expected": expected, "comparison": {"kind": "exact"}}
                if expected is not None
                else {}
            ),
            **({} if preserved is None else {"preserved": preserved}),
        }
    ]
    if bind is not None:
        records.append(
            {
                "name": "weight",
                "direction": "input",
                "size": SIZE,
                "source": {"stage_output": bind},
            }
        )
    records.extend(buffers)
    return {
        "name": name,
        "implementation": implementation_id(implementation),
        "buffers": records,
    }


def run(
    tmp_path,
    cases,
    *,
    runtime=None,
    runtime_type=StagingRuntime,
    generators=None,
    **rounds,
):
    root = tmp_path
    (root / "xadx").mkdir(parents=True, exist_ok=True)
    (root / "xadx" / "kernel.xadx").write_bytes(XADX_DATA)
    (root / "transactions").mkdir(parents=True, exist_ok=True)
    (root / "transactions" / "kernel.bin").write_bytes(TRANSACTION_DATA)
    runtimes = []

    def factory():
        runtimes.append(runtime_type(**(runtime or {})))
        return runtimes[-1]

    report = xadx.run_recipe(
        recipe(cases),
        root,
        runtime_factory=factory,
        generators=generators or Generators(),
        lifecycle_rounds=rounds.get("lifecycle_rounds", 1),
        resident_rounds=rounds.get("resident_rounds", 0),
        readback_rounds=rounds.get("readback_rounds", 0),
        target_family="npu2",
    )
    return report, runtimes


@pytest.mark.parametrize("runtime_type", [StagingRuntime, NoncoherentRuntime])
def test_a_later_stage_reads_what_the_earlier_stage_wrote(
    tmp_path, runtime_type
):
    """The whole point of a chain: no host round trip between the stages."""
    signature = GENERATOR_VALUES["swapped"]
    cases = [
        {
            "name": "chained",
            "stages": [
                stage("dequant", "first", preserved=generator("swapped")),
                stage("mm", "second", expected=generator(), bind="dequant.output"),
            ],
        }
    ]
    report, runtimes = run(
        tmp_path,
        cases,
        runtime_type=runtime_type,
        runtime={"writes": {"executable-first": signature}},
    )
    # The second dispatch saw the first stage's bytes in its weight buffer,
    # which is the binding a chained case depends on.
    observed = runtimes[0].observed
    assert len(observed) == 2
    assert observed[0][0] == bytes(SIZE)
    assert observed[1][1] == signature
    assert signature in runtimes[0].payload_inspections
    assert signature not in runtimes[0].payload_reads
    assert (signature, 2) in runtimes[0].inspection_dispatch_counts
    assert report["status"] == "pass", report["failures"]


def test_a_later_stage_may_not_modify_its_chained_input(tmp_path):
    cases = [
        {
            "name": "chained",
            "stages": [
                stage("dequant", "first", preserved=generator("swapped")),
                stage("mm", "second", expected=generator(), bind="dequant.output"),
            ],
        }
    ]
    report, _ = run(
        tmp_path,
        cases,
        runtime={
            "writes": {"executable-first": GENERATOR_VALUES["swapped"]},
            "corrupt": {"executable-second": "input"},
        },
    )
    assert report["status"] == "fail"
    assert any(
        "input was modified" in failure["message"]
        for failure in report["failures"]
    )


def test_the_compared_buffer_is_the_stage_that_declares_the_expectation(tmp_path):
    """An unexpected first-stage result must not be read as the case result."""
    cases = [
        {
            "name": "chained",
            "stages": [
                stage("dequant", "first", preserved=generator("swapped")),
                stage("mm", "second", expected=generator(), bind="dequant.output"),
            ],
        }
    ]
    # Only the first stage writes, and it writes something non-zero. The case
    # still passes because the second stage's output is what is compared.
    report, _ = run(
        tmp_path,
        cases,
        runtime={"writes": {"executable-first": GENERATOR_VALUES["swapped"]}},
    )
    assert report["status"] == "pass", report["failures"]


@pytest.mark.parametrize("corruption", ["leading", "trailing"])
def test_device_writes_outside_a_declared_buffer_are_rejected(
    tmp_path, corruption
):
    cases = [
        {
            "name": "guarded",
            "stages": [stage("only", "first", expected=generator())],
        }
    ]
    report, _ = run(
        tmp_path,
        cases,
        runtime={"corrupt": {"executable-first": corruption}},
    )
    assert report["status"] == "fail"
    assert any(
        f"{corruption} guard" in failure["message"]
        for failure in report["failures"]
    )


def test_device_modification_of_an_input_buffer_is_rejected(tmp_path):
    source = generator()
    cases = [
        {
            "name": "immutable",
            "stages": [
                stage(
                    "only",
                    "first",
                    expected=generator(),
                    buffers=[
                        {
                            "name": "input",
                            "direction": "input",
                            "size": SIZE,
                            "source": source,
                        }
                    ],
                )
            ],
        }
    ]
    report, _ = run(
        tmp_path,
        cases,
        runtime={"corrupt": {"executable-first": "input"}},
    )
    assert report["status"] == "fail"
    assert any(
        "input was modified" in failure["message"]
        for failure in report["failures"]
    )


def test_binding_a_stage_output_that_was_never_written_is_refused(tmp_path):
    cases = [
        {
            "name": "chained",
            "stages": [stage("mm", "second", expected=generator(), bind="absent.output")],
        }
    ]
    with pytest.raises(xadx.XadxError, match="which no earlier stage wrote"):
        run(tmp_path, cases)


def test_binding_a_buffer_of_the_wrong_size_is_refused(tmp_path):
    """A chain only means anything if both stages agree on the buffer."""
    second = stage("mm", "second", expected=generator(), bind="dequant.output")
    second["buffers"][1]["size"] = SIZE * 2
    cases = [{"name": "chained", "stages": [stage("dequant", "first"), second]}]
    with pytest.raises(xadx.XadxError, match="into a"):
        run(tmp_path, cases)


def test_a_case_with_no_expectation_is_refused(tmp_path):
    cases = [{"name": "silent", "stages": [stage("only", "first")]}]
    with pytest.raises(xadx.XadxError, match="exactly one expectation"):
        run(tmp_path, cases)


def test_a_case_naming_an_undescribed_implementation_is_refused(tmp_path):
    cases = [{"name": "stray", "stages": [stage("only", "third", expected=generator())]}]
    with pytest.raises(xadx.XadxError, match="does not describe"):
        run(tmp_path, cases)


def test_a_case_naming_an_undescribed_execution_variant_is_refused(tmp_path):
    selected = stage("only", "first", expected=generator())
    selected["execution"] = "production"
    cases = [{"name": "stray-variant", "stages": [selected]}]
    with pytest.raises(xadx.XadxError, match="unknown execution variant"):
        run(tmp_path, cases)


def test_a_recipe_of_another_schema_version_is_refused(tmp_path):
    body = recipe([])
    body["schema_version"] = KERNEL_RECIPE_SCHEMA_VERSION + 1
    with pytest.raises(xadx.XadxError, match="required constant"):
        xadx.run_recipe(body, tmp_path, runtime_factory=StagingRuntime)


def test_duplicate_implementation_ids_are_refused_before_device_open(tmp_path):
    body = recipe([{"name": "case", "stages": [stage("only", "first", expected=generator())]}])
    body["implementations"].append(dict(body["implementations"][0]))
    with pytest.raises(xadx.XadxError, match="implementation IDs"):
        xadx.run_recipe(body, tmp_path, runtime_factory=StagingRuntime)


def test_duplicate_case_names_are_refused_before_device_open(tmp_path):
    case = {"name": "same", "stages": [stage("only", "first", expected=generator())]}
    with pytest.raises(xadx.XadxError, match="case names"):
        xadx.run_recipe(recipe([case, case]), tmp_path, runtime_factory=StagingRuntime)


def test_duplicate_stage_and_buffer_names_are_refused(tmp_path):
    first = stage("same", "first")
    second = stage("same", "second", expected=generator())
    with pytest.raises(xadx.XadxError, match="duplicate stage"):
        run(tmp_path, [{"name": "case", "stages": [first, second]}])
    duplicate = stage("only", "first", expected=generator())
    duplicate["buffers"].append(dict(duplicate["buffers"][0]))
    with pytest.raises(xadx.XadxError, match="buffer names"):
        run(tmp_path, [{"name": "case", "stages": [duplicate]}])


def test_oversized_allocation_is_refused_before_device_open(tmp_path):
    only = stage("only", "first", expected=generator())
    only["buffers"][0]["size"] = 134217729
    with pytest.raises(xadx.XadxError, match="exceeds its maximum"):
        run(tmp_path, [{"name": "case", "stages": [only]}])


def test_multiple_expectations_are_refused(tmp_path):
    only = stage(
        "only",
        "first",
        expected=generator(),
        buffers=[
            {
                "name": "another-output",
                "direction": "output",
                "size": SIZE,
                "expected": generator(),
                "comparison": {"kind": "exact"},
            }
        ],
    )
    with pytest.raises(xadx.XadxError, match="exactly one expectation"):
        run(tmp_path, [{"name": "case", "stages": [only]}])


def test_wrong_xadx_digest_is_refused(tmp_path):
    body = recipe(
        [{"name": "case", "stages": [stage("only", "first", expected=generator())]}]
    )
    body["implementations"][0]["xadx_sha256"] = "0" * 64
    (tmp_path / "xadx").mkdir()
    (tmp_path / "xadx" / "kernel.xadx").write_bytes(XADX_DATA)
    (tmp_path / "transactions").mkdir()
    (tmp_path / "transactions" / "kernel.bin").write_bytes(TRANSACTION_DATA)
    with pytest.raises(xadx.XadxError, match="XADX digest mismatch"):
        xadx.run_recipe(
            body,
            tmp_path,
            runtime_factory=StagingRuntime,
            generators=Generators(),
            lifecycle_rounds=1,
            target_family="npu2",
        )


def test_wrong_transaction_digest_is_refused(tmp_path):
    body = recipe(
        [{"name": "case", "stages": [stage("only", "first", expected=generator())]}]
    )
    body["implementations"][0]["transaction_sha256"] = "0" * 64
    (tmp_path / "xadx").mkdir()
    (tmp_path / "xadx" / "kernel.xadx").write_bytes(XADX_DATA)
    (tmp_path / "transactions").mkdir()
    (tmp_path / "transactions" / "kernel.bin").write_bytes(TRANSACTION_DATA)
    with pytest.raises(xadx.XadxError, match="transaction digest mismatch"):
        xadx.run_recipe(
            body,
            tmp_path,
            runtime_factory=StagingRuntime,
            generators=Generators(),
            lifecycle_rounds=1,
            target_family="npu2",
        )


def test_wrong_generated_oracle_digest_is_refused(tmp_path):
    expected = generator()
    expected["sha256"] = "0" * 64
    with pytest.raises(xadx.XadxError, match="produced.*expected"):
        run(
            tmp_path,
            [{"name": "case", "stages": [stage("only", "first", expected=expected)]}],
        )


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ((1.0,), (float("nan"),)),
        ((1.0,), (float("inf"),)),
        ((float("inf"),), (float("-inf"),)),
        ((float("nan"),), (1.0,)),
    ],
)
def test_bf16_comparison_rejects_finite_nonfinite_disagreement(actual, expected):
    assert xadx._compare(
        _bf16_bytes(*actual),
        _bf16_bytes(*expected),
        "bf16-tolerance",
        1.0,
        1.0,
    )


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ((float("nan"),), (float("nan"),)),
        ((float("inf"),), (float("inf"),)),
        ((float("-inf"),), (float("-inf"),)),
    ],
)
def test_bf16_comparison_accepts_matching_nonfinite_values(actual, expected):
    assert (
        xadx._compare(
            _bf16_bytes(*actual),
            _bf16_bytes(*expected),
            "bf16-tolerance",
            0.0,
            0.0,
        )
        is None
    )


def test_a_generator_that_disagrees_with_the_declared_size_is_refused(tmp_path):
    """A recipe fault must not arrive disguised as a failed round."""
    cases = [
        {
            "name": "mismatched",
            "stages": [stage("only", "first", expected=generator("short"))],
        }
    ]
    with pytest.raises(xadx.XadxError, match="size mismatch"):
        run(tmp_path, cases, generators=Generators({"short": b"\x00"}))


def test_a_failing_generator_is_reported_as_a_recipe_fault(tmp_path):
    cases = [
        {
            "name": "broken",
            "stages": [stage("only", "first", expected=generator("boom"))],
        }
    ]
    with pytest.raises(xadx.XadxError, match="cannot materialize recipe buffer"):
        run(tmp_path, cases)


def test_buffers_are_resolved_before_any_dispatch(tmp_path):
    """The faulty case must be caught before the device is asked to do work."""
    cases = [
        {"name": "sound", "stages": [stage("only", "first", expected=generator())]},
        {
            "name": "broken",
            "stages": [stage("only", "first", expected=generator("boom"))],
        },
    ]
    runtimes = []

    def factory():
        runtimes.append(StagingRuntime())
        return runtimes[-1]

    (tmp_path / "xadx").mkdir(parents=True, exist_ok=True)
    (tmp_path / "xadx" / "kernel.xadx").write_bytes(XADX_DATA)
    (tmp_path / "transactions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "transactions" / "kernel.bin").write_bytes(TRANSACTION_DATA)
    with pytest.raises(xadx.XadxError):
        xadx.run_recipe(
            recipe(cases),
            tmp_path,
            runtime_factory=factory,
            generators=Generators(),
            lifecycle_rounds=1,
            target_family="npu2",
        )
    assert not any(runtime.dispatches for runtime in runtimes)


def test_aggregate_materialization_is_bounded_before_allocation_or_device(tmp_path):
    class CountingGenerators(Generators):
        def __init__(self):
            super().__init__()
            self.materializations = 0

        def predicted_size(self, id, parameters):
            return parameters["size"]

        def materialize(self, id, parameters):
            self.materializations += 1
            return super().materialize(id, parameters)

    cases = []
    for index in range(9):
        size = 16 * 1024 * 1024 - index
        selected = stage(
            f"stage-{index}",
            "first",
            expected={
                "id": "zero",
                "parameters": {"size": size},
                "sha256": "0" * 64,
            },
        )
        selected["buffers"][0]["size"] = size
        cases.append({"name": f"case-{index}", "stages": [selected]})
    generators = CountingGenerators()
    runtime_calls = 0

    def factory():
        nonlocal runtime_calls
        runtime_calls += 1
        return StagingRuntime()

    with pytest.raises(xadx.XadxError, match="128 MiB materialization limit"):
        xadx.run_recipe(
            recipe(cases, implementations=("first",)),
            tmp_path,
            runtime_factory=factory,
            generators=generators,
            lifecycle_rounds=0,
            resident_rounds=0,
            readback_rounds=0,
            target_family="npu2",
        )
    assert generators.materializations == 0
    assert runtime_calls == 0


def test_authoritative_unsupported_family_short_circuits_assets_and_generators(
    tmp_path,
):
    class GeneratorsThatMustNotRun:
        def predicted_size(self, id, parameters):
            raise AssertionError("unsupported targets must not run generators")

        def materialize(self, id, parameters):
            raise AssertionError("unsupported targets must not materialize")

    def runtime_factory():
        raise AssertionError("unsupported targets must not open the device")

    cases = [
        {
            "name": "unsupported",
            "stages": [stage("only", "first", expected=generator())],
        }
    ]
    report = xadx.run_recipe(
        recipe(cases, implementations=("first",)),
        tmp_path,
        runtime_factory=runtime_factory,
        generators=GeneratorsThatMustNotRun(),
        target_family="npu1",
    )
    assert report["status"] == "unsupported"


def test_every_stage_output_is_cleared_between_rounds(tmp_path):
    """A resident chain must not read the previous round's leftovers."""
    cases = [
        {
            "name": "chained",
            "stages": [
                stage("dequant", "first", preserved=generator("swapped")),
                stage("mm", "second", expected=generator(), bind="dequant.output"),
            ],
        }
    ]
    report, runtimes = run(
        tmp_path,
        cases,
        runtime={"writes": {"executable-first": GENERATOR_VALUES["swapped"]}},
        lifecycle_rounds=0,
        resident_rounds=3,
    )
    assert report["status"] == "pass", report["failures"]
    # Each round's first dispatch sees a cleared buffer, never the leftovers.
    firsts = runtimes[0].observed[0::2]
    assert len(firsts) == 3
    assert all(seen[0] == bytes(SIZE) for seen in firsts)


def test_the_full_case_inventory_is_loaded_before_resident_dispatch(tmp_path):
    cases = [
        {
            "name": alias,
            "stages": [stage("only", alias, expected=generator())],
        }
        for alias in ("first", "second")
    ]
    report, runtimes = run(
        tmp_path,
        cases,
        lifecycle_rounds=0,
        resident_rounds=1,
    )
    assert report["status"] == "pass", report["failures"]
    assert runtimes[0].events[:2] == ["load:first", "load:second"]


def test_failed_resident_preparation_rolls_back_partial_resources(tmp_path):
    cases = [
        {
            "name": case_name,
            "stages": [stage("only", implementation, expected=generator())],
        }
        for case_name, implementation in (
            ("first", "first"),
            ("second", "second"),
            ("third", "first"),
        )
    ]
    report, runtimes = run(
        tmp_path,
        cases,
        runtime_type=FailingPreparationRuntime,
        lifecycle_rounds=0,
        resident_rounds=1,
    )
    assert report["status"] == "fail"
    assert runtimes[0].resource_counts_at_dispatch == [4, 4]
    assert any(
        failure["case"] == "second"
        and failure["phase"] == "resident-preparation"
        for failure in report["failures"]
    )


def test_readback_rewrites_a_buffer_in_place(tmp_path):
    swapped = b"\x11" * SIZE
    cases = [
        {
            "name": "resident",
            "stages": [
                stage(
                    "mm",
                    "first",
                    expected=generator(),
                    buffers=[
                        {
                            "name": "activation",
                            "direction": "input",
                            "size": SIZE,
                            "source": generator(),
                        }
                    ],
                )
            ],
            "readback_variants": [
                {
                    "stage": "mm",
                    "buffer": "activation",
                    "source": generator("swapped"),
                    "expected": generator(),
                }
            ],
        }
    ]
    report, runtimes = run(
        tmp_path,
        cases,
        generators=Generators({"swapped": swapped}),
        lifecycle_rounds=0,
        resident_rounds=1,
        readback_rounds=2,
    )
    assert report["status"] == "pass", report["failures"]
    # The resident dispatch saw zeros; both readback dispatches saw the swap.
    assert runtimes[0].observed[0][1] == bytes(SIZE)
    assert [seen[1] for seen in runtimes[0].observed[1:]] == [swapped, swapped]


def test_readback_may_not_overwrite_an_output(tmp_path):
    cases = [
        {
            "name": "resident",
            "stages": [stage("mm", "first", expected=generator())],
            "readback_variants": [
                {
                    "stage": "mm",
                    "buffer": "output",
                    "source": generator(),
                    "expected": generator(),
                }
            ],
        }
    ]
    with pytest.raises(xadx.XadxError, match="not a resident input"):
        run(tmp_path, cases, readback_rounds=1)


def test_each_coherence_case_receives_the_full_readback_depth(tmp_path):
    cases = []
    for alias in ("first", "second"):
        only = stage(
            "mm",
            alias,
            expected=generator(),
            buffers=[
                {
                    "name": "activation",
                    "direction": "input",
                    "size": SIZE,
                    "source": generator(),
                }
            ],
        )
        cases.append(
            {
                "name": alias,
                "stages": [only],
                "readback_variants": [
                    {
                        "stage": "mm",
                        "buffer": "activation",
                        "source": generator(),
                        "expected": generator(),
                    }
                ],
            }
        )
    report, _ = run(
        tmp_path,
        cases,
        lifecycle_rounds=0,
        resident_rounds=0,
        readback_rounds=3,
    )
    assert report["status"] == "pass", report["failures"]
    assert report["readback_by_case"] == {"first": 3, "second": 3}


def test_report_binds_exact_recipe_bytes_and_flattened_executables(tmp_path):
    body = recipe(
        [{"name": "case", "stages": [stage("only", "first", expected=generator())]}],
        implementations=("first",),
    )
    raw = (json.dumps(body, indent=3) + "\n").encode()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_bytes(raw)
    (tmp_path / "xadx").mkdir()
    (tmp_path / "xadx" / "kernel.xadx").write_bytes(XADX_DATA)
    (tmp_path / "transactions").mkdir()
    (tmp_path / "transactions" / "kernel.bin").write_bytes(TRANSACTION_DATA)
    report = xadx.run_recipe(
        recipe_path,
        tmp_path,
        runtime_factory=StagingRuntime,
        generators=Generators(),
        lifecycle_rounds=1,
        resident_rounds=0,
        readback_rounds=0,
        target_family="npu2",
    )
    assert report["schema_version"] == 1
    assert report["recipe_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["generator_version"] == "kernel-recipe-generator-v1"
    assert report["executables"] == [
        {
            "id": implementation_id("first"),
            "xclbin_path": "xclbins/first.xclbin",
            "xclbin_sha256": hashlib.sha256(b"xclbin-first").hexdigest(),
            "transaction": "transactions/kernel.bin",
            "transaction_sha256": TRANSACTION_SHA256,
            "xadx": "xadx/kernel.xadx",
            "xadx_sha256": XADX_SHA256,
        }
    ]


def test_unsupported_report_satisfies_v1_branch(tmp_path):
    cases = [
        {"name": "case", "stages": [stage("only", "first", expected=generator())]}
    ]
    report, _ = run(tmp_path, cases, runtime={"architecture": "unknown"})
    assert xadx.validate_xadx_report(report) is report
    assert report["report_kind"] == "execution"
    assert report["status"] == "unsupported"
    assert report["case_results"][0]["status"] == "unsupported"
    assert report["case_results"][0]["dispatches"] == 0
    assert report["failures"] == []
    assert "reason" in report


def test_failed_case_does_not_taint_passing_case_row(tmp_path):
    cases = [
        {"name": "bad", "stages": [stage("only", "second", expected=generator())]},
        {
            "name": "bad extra",
            "stages": [stage("only", "first", expected=generator())],
        },
    ]
    report, _ = run(
        tmp_path,
        cases,
        runtime={"writes": {"executable-first": b"\x01" * SIZE}},
    )
    rows = {row["name"]: row for row in report["case_results"]}
    assert report["status"] == "fail"
    assert rows["bad"]["status"] == "pass"
    assert rows["bad"]["failures"] == []
    assert rows["bad"]["validated_bytes"] == SIZE
    assert rows["bad"]["duration_seconds"] > 0
    assert rows["bad extra"]["status"] == "fail"
    assert rows["bad extra"]["mismatch_count"] == SIZE
    assert rows["bad extra"]["first_mismatch"]["index"] == 0
    assert rows["bad extra"]["failures"] == [
        {
            "phase": "lifecycle",
            "round": 1,
            "message": "exact mismatch at byte 0",
        }
    ]
    assert report["failures"] == [
        {
            "case": "bad extra",
            "phase": "lifecycle",
            "round": 1,
            "message": "exact mismatch at byte 0",
        }
    ]
