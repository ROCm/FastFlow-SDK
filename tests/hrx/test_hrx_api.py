# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ctypes
from pathlib import Path

import pytest

from build_tools import hrx_runtime as hrx

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]


@pytest.fixture(scope="module", autouse=True)
def _load_hrx_helper():
    global np
    import numpy as np


class FakeFunction:
    def __init__(self, implementation=None):
        self.implementation = implementation or (lambda *args: None)
        self.calls = []
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.implementation(*args)


class FakeLibrary:
    def __init__(self):
        function_names = (
            "hrx_status_code",
            "hrx_status_to_string",
            "hrx_status_free_message",
            "hrx_status_ignore",
            "hrx_host_allocator_system",
            "hrx_host_allocator_free",
            "hrx_amdxdna_xadx_serialize",
            "hrx_gpu_initialize",
            "hrx_gpu_shutdown",
            "hrx_gpu_device_get",
            "hrx_device_release",
            "hrx_stream_create",
            "hrx_stream_release",
            "hrx_stream_synchronize",
            "hrx_buffer_allocate",
            "hrx_buffer_map_with_mode",
            "hrx_buffer_unmap",
            "hrx_buffer_flush_range",
            "hrx_buffer_invalidate_range",
            "hrx_buffer_release",
            "hrx_executable_load_data",
            "hrx_executable_lookup_export_by_name",
            "hrx_executable_release",
            "hrx_stream_dispatch",
        )
        for name in function_names:
            setattr(self, name, FakeFunction())

        self.serialized = (ctypes.c_uint8 * 4)(0x58, 0x41, 0x44, 0x58)
        self.mapping = (ctypes.c_uint8 * 16)()
        self.hrx_host_allocator_system.implementation = lambda: hrx.HrxHostAllocator(
            0xA1, 0xA2
        )
        self.hrx_amdxdna_xadx_serialize.implementation = self._serialize
        self.hrx_gpu_device_get.implementation = lambda index, out: self._set_handle(
            out, 0x100
        )
        self.hrx_stream_create.implementation = lambda device, flags, out: (
            self._set_handle(out, 0x200)
        )
        self.hrx_buffer_allocate.implementation = (
            lambda stream, size, memory_type, usage, out: self._set_handle(out, 0x300)
        )
        self.hrx_buffer_map_with_mode.implementation = self._map
        self.hrx_executable_load_data.implementation = (
            lambda device, data, size, target_family, target_key, out: self._set_handle(
                out, 0x400
            )
        )
        self.hrx_executable_lookup_export_by_name.implementation = self._lookup

    def _serialize(self, params, allocator, out_data, out_length):
        ctypes.cast(out_data, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)))[0] = (
            ctypes.cast(self.serialized, ctypes.POINTER(ctypes.c_uint8))
        )
        ctypes.cast(out_length, ctypes.POINTER(ctypes.c_size_t))[0] = len(
            self.serialized
        )

    @staticmethod
    def _set_handle(output, value):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = value

    def _map(self, buffer, mode, flags, offset, size, output):
        pointer = ctypes.cast(self.mapping, ctypes.c_void_p).value
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = pointer

    @staticmethod
    def _lookup(executable, name, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint32))[0] = 7


def test_x64_public_record_layouts():
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(hrx.HrxHostAllocator) == 16
    assert ctypes.sizeof(hrx.HrxConstByteSpan) == 16
    assert ctypes.sizeof(hrx.HrxAmdxdnaExecutableRun) == 40
    assert ctypes.sizeof(hrx.HrxAmdxdnaExecutableEntryPoint) == 72
    assert ctypes.sizeof(hrx.HrxAmdxdnaExecutableCreateParams) == 48
    assert hrx.HrxAmdxdnaExecutableEntryPoint.runs.offset == 56
    assert hrx.HrxAmdxdnaExecutableCreateParams.entry_points.offset == 32
    assert ctypes.sizeof(hrx.HrxDispatchConfig) == 28
    assert ctypes.sizeof(hrx.HrxBufferRef) == 24


def test_description_owns_and_populates_borrowed_storage():
    description = hrx.AmdxdnaExecutableDescription(
        b"xclbin", b"transaction", "MLIR_AIE", b"payload"
    )
    params = description.params
    assert params.record_length == ctypes.sizeof(params)
    assert params.xclbin_count == 1
    assert params.entry_point_count == 1
    assert (
        ctypes.string_at(params.xclbins[0].data, params.xclbins[0].data_length)
        == b"xclbin"
    )
    entry = params.entry_points[0]
    assert entry.name.data[: entry.name.size] == b"MLIR_AIE"
    assert entry.context_mode == hrx.HRX_AMDXDNA_CONTEXT_MODE_CREATE
    run = entry.runs[0]
    assert run.record_length == ctypes.sizeof(run)
    assert (
        ctypes.string_at(run.transaction.data, run.transaction.data_length)
        == b"transaction"
    )
    assert (
        ctypes.string_at(run.data_payload.data, run.data_payload.data_length)
        == b"payload"
    )


def test_library_candidate_order_for_windows_and_linux():
    environment = {
        "HRX_LIBHRX": "C:/explicit/hrx.dll",
        "LIBHRX_DIR": "C:/libhrx",
        "HRX_INSTALL": "C:/hrx",
    }
    windows = hrx.library_candidates(environment, "nt")
    assert windows[0] == "C:/explicit/hrx.dll"
    assert windows[1] == str(Path("C:/libhrx") / "hrx.dll")
    assert str(Path("C:/hrx") / "bin" / "hrx.dll") in windows
    assert windows[-1] == "hrx.dll"

    linux = hrx.library_candidates(
        {"LIBHRX_DIR": "/opt/hrx/lib", "HRX_INSTALL": "/opt/hrx"}, "posix"
    )
    assert linux[0] == str(Path("/opt/hrx/lib") / "libhrx.so")
    assert linux[-1] == "libhrx.so"


def test_full_api_reuses_shared_probe_bindings():
    library = FakeLibrary()
    api = hrx.HrxApi(library)

    assert api.status_code is library.hrx_status_code
    assert api.status_to_string is library.hrx_status_to_string
    assert api.gpu_initialize is library.hrx_gpu_initialize
    assert api.gpu_shutdown is library.hrx_gpu_shutdown
    assert library.hrx_status_code.argtypes == [ctypes.c_void_p]


def test_serialize_uses_system_allocator_and_frees_result():
    library = FakeLibrary()
    api = hrx.HrxApi(library)

    assert api.serialize_xadx(b"xclbin", b"transaction", "MLIR_AIE") == b"XADX"
    assert len(library.hrx_host_allocator_system.calls) == 1
    assert len(library.hrx_amdxdna_xadx_serialize.calls) == 1
    assert len(library.hrx_host_allocator_free.calls) == 1
    allocator, pointer = library.hrx_host_allocator_free.calls[0]
    assert allocator.self == 0xA1
    assert pointer


def test_runtime_loads_maps_dispatches_and_releases():
    library = FakeLibrary()
    api = hrx.HrxApi(library)
    runtime = hrx.HrxRuntime(api)
    executable = runtime.load_xadx(b"serialized-xadx")
    buffer = runtime.allocate_buffer((4,), np.int32)

    buffer.array[:] = [1, 2, 3, 4]
    buffer.flush()
    ordinal = executable.lookup_export("MLIR_AIE")
    runtime.dispatch(executable, ordinal, [buffer])
    runtime.synchronize()
    buffer.invalidate()

    load_call = library.hrx_executable_load_data.calls[0]
    assert load_call[3] == hrx.HRX_AMDXDNA_EXECUTABLE_TARGET_FAMILY
    assert load_call[4] == hrx.HRX_AMDXDNA_EXECUTABLE_TARGET_KEY
    allocate_call = library.hrx_buffer_allocate.calls[0]
    assert allocate_call[1] == 16
    assert (
        allocate_call[2]
        == hrx.HRX_MEMORY_TYPE_HOST_LOCAL | hrx.HRX_MEMORY_TYPE_DEVICE_VISIBLE
    )
    map_call = library.hrx_buffer_map_with_mode.calls[0]
    assert map_call[1] == hrx.HRX_MAPPING_MODE_PERSISTENT
    assert map_call[2] == hrx.HRX_MAP_READ | hrx.HRX_MAP_WRITE
    dispatch_call = library.hrx_stream_dispatch.calls[0]
    assert dispatch_call[2] == 7
    assert dispatch_call[7] == 1

    runtime.close()
    assert len(library.hrx_buffer_unmap.calls) == 1
    assert len(library.hrx_buffer_release.calls) == 1
    assert len(library.hrx_executable_release.calls) == 1
    assert len(library.hrx_stream_release.calls) == 1
    assert len(library.hrx_device_release.calls) == 1
    assert len(library.hrx_gpu_shutdown.calls) == 1
