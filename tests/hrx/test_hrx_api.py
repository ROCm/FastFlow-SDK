# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import ctypes
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "tests" / "npu" / "hrx_api.py"
SPEC = importlib.util.spec_from_file_location("test_hrx_api_helper", HELPER_PATH)
hrx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hrx)


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
        self.hrx_stream_create.implementation = (
            lambda device, flags, out: self._set_handle(out, 0x200)
        )
        self.hrx_buffer_allocate.implementation = (
            lambda stream, size, memory_type, usage, out: self._set_handle(out, 0x300)
        )
        self.hrx_buffer_map_with_mode.implementation = self._map
        self.hrx_executable_load_data.implementation = (
            lambda device, data, size, executable_format, out: self._set_handle(
                out, 0x400
            )
        )
        self.hrx_executable_lookup_export_by_name.implementation = self._lookup

    def _serialize(self, params, allocator, out_data, out_length):
        ctypes.cast(
            out_data, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8))
        )[0] = ctypes.cast(self.serialized, ctypes.POINTER(ctypes.c_uint8))
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


class HrxApiTest(unittest.TestCase):
    def test_x64_public_record_layouts(self):
        self.assertEqual(ctypes.sizeof(ctypes.c_void_p), 8)
        self.assertEqual(ctypes.sizeof(hrx.HrxHostAllocator), 16)
        self.assertEqual(ctypes.sizeof(hrx.HrxConstByteSpan), 16)
        self.assertEqual(ctypes.sizeof(hrx.HrxAmdxdnaExecutableRun), 40)
        self.assertEqual(
            ctypes.sizeof(hrx.HrxAmdxdnaExecutableEntryPoint), 72
        )
        self.assertEqual(
            ctypes.sizeof(hrx.HrxAmdxdnaExecutableCreateParams), 48
        )
        self.assertEqual(hrx.HrxAmdxdnaExecutableEntryPoint.runs.offset, 56)
        self.assertEqual(
            hrx.HrxAmdxdnaExecutableCreateParams.entry_points.offset, 32
        )
        self.assertEqual(ctypes.sizeof(hrx.HrxDispatchConfig), 28)
        self.assertEqual(ctypes.sizeof(hrx.HrxBufferRef), 24)

    def test_description_owns_and_populates_borrowed_storage(self):
        description = hrx.AmdxdnaExecutableDescription(
            b"xclbin", b"transaction", "MLIR_AIE", b"payload"
        )
        params = description.params
        self.assertEqual(params.record_length, ctypes.sizeof(params))
        self.assertEqual(params.xclbin_count, 1)
        self.assertEqual(params.entry_point_count, 1)
        self.assertEqual(
            ctypes.string_at(
                params.xclbins[0].data, params.xclbins[0].data_length
            ),
            b"xclbin",
        )
        entry = params.entry_points[0]
        self.assertEqual(entry.name.data[: entry.name.size], b"MLIR_AIE")
        self.assertEqual(entry.context_mode, hrx.HRX_AMDXDNA_CONTEXT_MODE_CREATE)
        run = entry.runs[0]
        self.assertEqual(run.record_length, ctypes.sizeof(run))
        self.assertEqual(
            ctypes.string_at(run.transaction.data, run.transaction.data_length),
            b"transaction",
        )
        self.assertEqual(
            ctypes.string_at(run.data_payload.data, run.data_payload.data_length),
            b"payload",
        )

    def test_library_candidate_order_for_windows_and_linux(self):
        environment = {
            "HRX_LIBHRX": "C:/explicit/hrx.dll",
            "LIBHRX_DIR": "C:/libhrx",
            "HRX_INSTALL": "C:/hrx",
        }
        windows = hrx.library_candidates(environment, "nt")
        self.assertEqual(windows[0], "C:/explicit/hrx.dll")
        self.assertEqual(windows[1], str(Path("C:/libhrx") / "hrx.dll"))
        self.assertIn(str(Path("C:/hrx") / "bin" / "hrx.dll"), windows)
        self.assertEqual(windows[-1], "hrx.dll")

        linux = hrx.library_candidates(
            {"LIBHRX_DIR": "/opt/hrx/lib", "HRX_INSTALL": "/opt/hrx"}, "posix"
        )
        self.assertEqual(linux[0], str(Path("/opt/hrx/lib") / "libhrx.so"))
        self.assertEqual(linux[-1], "libhrx.so")

    def test_serialize_uses_system_allocator_and_frees_result(self):
        library = FakeLibrary()
        api = hrx.HrxApi(library)

        self.assertEqual(
            api.serialize_xadx(b"xclbin", b"transaction", "MLIR_AIE"), b"XADX"
        )
        self.assertEqual(len(library.hrx_host_allocator_system.calls), 1)
        self.assertEqual(len(library.hrx_amdxdna_xadx_serialize.calls), 1)
        self.assertEqual(len(library.hrx_host_allocator_free.calls), 1)
        allocator, pointer = library.hrx_host_allocator_free.calls[0]
        self.assertEqual(allocator.self, 0xA1)
        self.assertTrue(pointer)

    def test_runtime_loads_maps_dispatches_and_releases(self):
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
        self.assertEqual(load_call[3], b"amdxdna-xclbin-fb")
        allocate_call = library.hrx_buffer_allocate.calls[0]
        self.assertEqual(allocate_call[1], 16)
        self.assertEqual(
            allocate_call[2],
            hrx.HRX_MEMORY_TYPE_HOST_LOCAL | hrx.HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        )
        map_call = library.hrx_buffer_map_with_mode.calls[0]
        self.assertEqual(map_call[1], hrx.HRX_MAPPING_MODE_PERSISTENT)
        self.assertEqual(map_call[2], hrx.HRX_MAP_READ | hrx.HRX_MAP_WRITE)
        dispatch_call = library.hrx_stream_dispatch.calls[0]
        self.assertEqual(dispatch_call[2], 7)
        self.assertEqual(dispatch_call[7], 1)

        runtime.close()
        self.assertEqual(len(library.hrx_buffer_unmap.calls), 1)
        self.assertEqual(len(library.hrx_buffer_release.calls), 1)
        self.assertEqual(len(library.hrx_executable_release.calls), 1)
        self.assertEqual(len(library.hrx_stream_release.calls), 1)
        self.assertEqual(len(library.hrx_device_release.calls), 1)
        self.assertEqual(len(library.hrx_gpu_shutdown.calls), 1)


if __name__ == "__main__":
    unittest.main()

