# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ctypes

import pytest

from build_tools import hrx_runtime as abi

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]


def test_builder_retains_four_distinct_run_records():
    runs = [
        (f"transaction-{index}".encode(), f"payload-{index}".encode())
        for index in range(4)
    ]
    description = abi.AmdxdnaExecutableDescription(
        b"xclbin", None, "MLIR_AIE", runs=runs
    )
    assert description.entry_point.run_count == 4
    assert len(description.transaction_storages) == 4
    assert len(description.data_payload_storages) == 4
    for index, (transaction, payload) in enumerate(runs):
        record = description.entry_point.runs[index]
        assert (
            ctypes.string_at(record.transaction.data, record.transaction.data_length)
            == transaction
        )
        assert (
            ctypes.string_at(record.data_payload.data, record.data_payload.data_length)
            == payload
        )


def test_builder_rejects_empty_or_ambiguous_run_lists():
    with pytest.raises(ValueError):
        abi.AmdxdnaExecutableDescription(b"xclbin", None, "MLIR_AIE", runs=[])
    with pytest.raises(ValueError):
        abi.AmdxdnaExecutableDescription(
            b"xclbin", b"txn", "MLIR_AIE", runs=[(b"other", b"")]
        )


def test_x64_public_record_layout():
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(abi.HrxConstByteSpan) == 16
    assert ctypes.sizeof(abi.HrxAmdxdnaExecutableRun) == 72
    assert ctypes.sizeof(abi.HrxAmdxdnaExecutableEntryPoint) == 88
    assert ctypes.sizeof(abi.HrxAmdxdnaExecutableCreateParams) == 64
    assert abi.HrxAmdxdnaExecutableRun.transaction.offset == 8
    assert abi.HrxAmdxdnaExecutableEntryPoint.runs.offset == 56
    assert abi.HrxAmdxdnaExecutableCreateParams.xclbins.offset == 16
    assert abi.HrxAmdxdnaExecutableCreateParams.entry_points.offset == 32


def test_builder_populates_records_and_retains_storage():
    xclbin = b"xclbin-bytes"
    transaction = b"transaction-bytes"
    description = abi.AmdxdnaExecutableDescription(xclbin, transaction, "MLIR_AIE")

    params = description.params
    assert params.record_length == ctypes.sizeof(params)
    assert params.abi_version == 0
    assert params.xclbin_count == 1
    assert params.entry_point_count == 1
    assert (
        ctypes.string_at(params.xclbins[0].data, params.xclbins[0].data_length)
        == xclbin
    )

    entry = params.entry_points[0]
    assert entry.record_length == ctypes.sizeof(entry)
    assert entry.name.data[: entry.name.size] == b"MLIR_AIE"
    assert entry.context_mode == abi.HRX_AMDXDNA_CONTEXT_MODE_CREATE
    assert entry.run_count == 1

    run = entry.runs[0]
    assert run.record_length == ctypes.sizeof(run)
    assert (
        ctypes.string_at(run.transaction.data, run.transaction.data_length)
        == transaction
    )


def test_builder_populates_pdis_relocations_and_parameters():
    description = abi.AmdxdnaExecutableDescription(
        b"xclbin",
        b"transaction",
        "MLIR_AIE",
        pdis=[b"pdi"],
        run_pdi_relocations=[[(4, 0, -16)]],
        run_control_parameter_relocations=[[(8, 32)]],
        control_parameters=[("input_offset", "i32", 0, 3, 4, 1)],
    )

    assert description.params.abi_version == 1
    assert description.entry_point.abi_version == 1
    assert description.params.pdi_count == 1
    assert (
        ctypes.string_at(
            description.params.pdis[0].data,
            description.params.pdis[0].data_length,
        )
        == b"pdi"
    )
    run = description.entry_point.runs[0]
    assert run.abi_version == 1
    assert run.pdi_relocation_count == 1
    assert run.pdi_relocations[0].addend == -16
    assert run.control_parameter_relocation_count == 1
    assert run.control_parameter_relocations[0].addend == 32
    parameter = description.entry_point.control_parameters[0]
    assert parameter.name.data[: parameter.name.size] == b"input_offset"
    assert parameter.state_table_index == 3


@pytest.mark.parametrize(
    ("xclbin", "transaction", "name"),
    [
        (b"", b"txn", "MLIR_AIE"),
        (b"xclbin", b"", "MLIR_AIE"),
        (b"xclbin", b"txn", ""),
    ],
)
def test_builder_rejects_empty_borrowed_data(xclbin, transaction, name):
    with pytest.raises(ValueError):
        abi.AmdxdnaExecutableDescription(xclbin, transaction, name)
