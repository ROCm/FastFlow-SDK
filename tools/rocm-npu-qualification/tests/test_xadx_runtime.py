# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""XADX HRX open/close error reporting.

Pool qualification was failing package-validation with
``hrx_gpu_shutdown failed (status 3)``. Status 3 is HRX_STATUS_INVALID_ARGUMENT,
returned when shutdown runs against a GPU that was never initialized (or when
profiling teardown maps to the same code). That is cleanup masking, not an NPU
firmware verdict: ``_Runtime.close()`` used to call shutdown unconditionally,
and a close failure replaced the real initialize / device_get error.
"""

from __future__ import annotations

import ctypes

import pytest

from rocm_npu_qualification import xadx

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]

_FAILED_STATUS = 0x51
_SHUTDOWN_STATUS = 3


class FakeFunction:
    def __init__(self, implementation=None):
        self.implementation = implementation or (lambda *args: None)
        self.calls = []
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.implementation(*args)


class FakeXadxLibrary:
    _FUNCTIONS = (
        "hrx_status_code",
        "hrx_status_ignore",
        "hrx_status_to_string",
        "hrx_status_free_message",
        "hrx_gpu_initialize",
        "hrx_gpu_shutdown",
        "hrx_gpu_device_get",
        "hrx_device_release",
        "hrx_device_get_property",
        "hrx_stream_create",
        "hrx_stream_release",
        "hrx_stream_synchronize",
        "hrx_buffer_allocate",
        "hrx_buffer_map_with_mode",
        "hrx_buffer_unmap",
        "hrx_buffer_release",
        "hrx_buffer_flush_range",
        "hrx_buffer_invalidate_range",
        "hrx_executable_load_data",
        "hrx_executable_lookup_export_by_name",
        "hrx_executable_release",
        "hrx_stream_dispatch",
    )

    def __init__(self):
        for name in self._FUNCTIONS:
            setattr(self, name, FakeFunction())
        self._status_code_value = 0
        self._status_message = b"ok"
        self._message_buffer = None
        self.hrx_status_code.implementation = lambda status: self._status_code_value
        self.hrx_status_to_string.implementation = self._to_string
        self.hrx_gpu_device_get.implementation = lambda index, out: self._set_handle(
            out, 0x100
        )
        self.hrx_stream_create.implementation = lambda device, flags, out: (
            self._set_handle(out, 0x200)
        )

    def fail(self, name, code, message):
        payload = message if isinstance(message, bytes) else message.encode()

        def implementation(*_args):
            self._status_code_value = code
            self._status_message = payload
            return _FAILED_STATUS

        getattr(self, name).implementation = implementation

    def _to_string(self, status, message_out, length_out):
        self._message_buffer = ctypes.create_string_buffer(self._status_message)
        ctypes.cast(message_out, ctypes.POINTER(ctypes.c_char_p))[0] = ctypes.cast(
            self._message_buffer, ctypes.c_char_p
        )
        ctypes.cast(length_out, ctypes.POINTER(ctypes.c_size_t))[0] = len(
            self._status_message
        )

    @staticmethod
    def _set_handle(output, value):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = value


def _bound_runtime(library):
    runtime = xadx._Runtime.__new__(xadx._Runtime)
    runtime._bind_library(library)
    return runtime


def test_close_without_init_does_not_shut_down():
    library = FakeXadxLibrary()
    runtime = _bound_runtime(library)
    runtime.close()
    assert library.hrx_gpu_initialize.calls == []
    assert library.hrx_gpu_shutdown.calls == []


def test_double_close_shuts_down_once():
    library = FakeXadxLibrary()
    runtime = xadx._Runtime(library)
    runtime.close()
    runtime.close()
    assert len(library.hrx_gpu_initialize.calls) == 1
    assert len(library.hrx_gpu_shutdown.calls) == 1
    assert len(library.hrx_device_release.calls) == 1
    assert len(library.hrx_stream_release.calls) == 1


def test_init_failure_then_close_reports_initialize_not_shutdown():
    library = FakeXadxLibrary()
    library.fail("hrx_gpu_initialize", 7, b"driver missing")
    library.fail(
        "hrx_gpu_shutdown",
        _SHUTDOWN_STATUS,
        b"GPU accelerator not initialized",
    )
    with pytest.raises(
        xadx.XadxError,
        match=r"hrx_gpu_initialize failed \(status 7\): driver missing",
    ) as caught:
        xadx._Runtime(library)
    assert caught.value.code == 7
    assert caught.value.operation == "hrx_gpu_initialize"
    assert "hrx_gpu_shutdown" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert library.hrx_gpu_shutdown.calls == []
    assert library.hrx_gpu_device_get.calls == []


def test_device_get_failure_is_not_masked_by_shutdown_status_3():
    library = FakeXadxLibrary()
    library.fail("hrx_gpu_device_get", 1, b"no such device")
    library.fail(
        "hrx_gpu_shutdown",
        _SHUTDOWN_STATUS,
        b"GPU accelerator not initialized",
    )
    with pytest.raises(
        xadx.XadxError,
        match=r"hrx_gpu_device_get failed \(status 1\): no such device",
    ) as caught:
        xadx._Runtime(library)
    assert caught.value.code == 1
    assert caught.value.operation == "hrx_gpu_device_get"
    assert "hrx_gpu_shutdown" not in str(caught.value)
    assert isinstance(caught.value.__cause__, xadx.XadxError)
    assert "hrx_gpu_shutdown failed (status 3)" in str(caught.value.__cause__)
    assert "GPU accelerator not initialized" in str(caught.value.__cause__)
    assert len(library.hrx_gpu_initialize.calls) == 1
    assert len(library.hrx_gpu_shutdown.calls) == 1
