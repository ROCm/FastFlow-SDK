# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compiler-free FastFlowLM XADX qualification runner.

This module itself needs only the standard library and the public HRX C ABI, but
that is a property of the dispatch code rather than a constraint on the package:
buffers and oracles are computed with NumPy, which a qualification install
carries. What an install genuinely never needs is FastFlowLM or a compiler, so
the asset producer stays separate from the runner.
"""

from __future__ import annotations

import builtins
import ctypes
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path, PurePosixPath

from .contracts import (
    XADX_REPORT_SCHEMA_VERSION,
    XADX_ROUNDS,
    XADX_UNIQUE_XCLBIN_CONTEXT_SOAK,
)
from .numeric_runtime import NUMERIC_DISTRIBUTIONS
from .recipe_contract import recipe_materialization_requests
from .schema import SchemaValidationError, load_packaged_schema, validate_schema

_Handle = ctypes.c_void_p
_Status = ctypes.c_void_p
_TARGET_FAMILY = b"amdxdna"
_TARGET_KEY = b"amdxdna"
_MEMORY_TYPE = (
    0x00000002 | 0x00000008 | 0x00000010
)  # HOST_VISIBLE|HOST_CACHED|DEVICE_VISIBLE


def _numeric_runtime():
    return {
        distribution: importlib.metadata.version(distribution)
        for distribution in NUMERIC_DISTRIBUTIONS
    }


def validate_xadx_report(report):
    """Validate a raw XADX report against its packaged wire contract."""
    try:
        validate_schema(report, load_packaged_schema("xadx-report-v1.schema.json"))
    except (SchemaValidationError, OSError, json.JSONDecodeError) as error:
        raise XadxError(
            f"XADX report does not satisfy the v1 schema: {error}"
        ) from error
    if report.get("schema_version") != XADX_REPORT_SCHEMA_VERSION:
        raise XadxError("XADX report schema version is invalid")
    return report


_BUFFER_USAGE = 0x00000C03 | 0x02000000
_MAPPING_MODE = 0x00000002
_ARCHITECTURE_PROPERTY = 1
_GUARD_BYTES = 4096
_GUARD_VALUE = 0xA5
_MAX_BUFFER_BYTES = 16 * 1024 * 1024
_MAX_MATERIALIZED_BYTES = 128 * 1024 * 1024
_MAX_RESIDENT_BYTES = 256 * 1024 * 1024
# IREE/HRX UNAVAILABLE. Some amdxdna kernels return EAGAIN from the first
# GET_INFO / hrx_gpu_initialize after runtime autosuspend because firmware
# resume (aie2_hw_start) is not ready; a later open succeeds. We cannot patch
# those drivers from qualification, so the runner retries this status only.
# Delays are a first-cut backoff; reports record attempts so a padded host
# is visible.
_HRX_UNAVAILABLE = 14
_OPEN_RETRY_DELAYS = (0.1, 0.4, 1.0)
np = None
ml_dtypes = None


class XadxError(RuntimeError):
    def __init__(self, message, *, code=None, operation=None, runtime_open=None):
        super().__init__(message)
        self.code = code
        self.operation = operation
        self.runtime_open = runtime_open


def _ensure_numeric_runtime():
    global np, ml_dtypes
    if np is None:
        import importlib

        np = importlib.import_module("numpy")
        ml_dtypes = importlib.import_module("ml_dtypes")
    return np, ml_dtypes


class _DispatchConfig(ctypes.Structure):
    _fields_ = [
        ("workgroup_count", ctypes.c_uint32 * 3),
        ("workgroup_size", ctypes.c_uint32 * 3),
        ("subgroup_size", ctypes.c_uint32),
    ]


class _BufferRef(ctypes.Structure):
    _fields_ = [
        ("buffer", _Handle),
        ("offset", ctypes.c_size_t),
        ("length", ctypes.c_size_t),
    ]


class _GuardedBuffer:
    """A device binding into the guarded middle of a larger allocation."""

    def __init__(self, handle, storage, backing, offset):
        self.handle = handle
        self.storage = storage
        self.backing = backing
        self.offset = offset

    def __iter__(self):
        return iter((self.handle, self.storage))

    def __getitem__(self, index):
        return (self.handle, self.storage)[index]


def _bind(library, name, result, arguments):
    try:
        function = getattr(library, name)
    except AttributeError as error:
        raise XadxError(f"public HRX API unavailable: {name}") from error
    function.restype = result
    function.argtypes = arguments
    return function


def _load_library(environ=None):
    environ = os.environ if environ is None else environ
    filename = "hrx.dll" if os.name == "nt" else "libhrx.so"
    authoritative = environ.get("ROCM_NPU_RUNTIME_PATH")
    if authoritative:
        try:
            return ctypes.CDLL(
                authoritative,
                **({} if os.name == "nt" else {"mode": ctypes.RTLD_GLOBAL}),
            )
        except OSError as error:
            raise XadxError(
                "could not load authoritative ROCM_NPU_RUNTIME_PATH "
                f"{authoritative}: {error}"
            ) from error
    candidates = [environ.get("HRX_LIBHRX")]
    if environ.get("LIBHRX_DIR"):
        candidates.append(str(Path(environ["LIBHRX_DIR"]) / filename))
    if environ.get("HRX_INSTALL"):
        candidates.extend(
            str(Path(environ["HRX_INSTALL"]) / part / filename)
            for part in (("bin", "lib") if os.name == "nt" else ("lib", "bin"))
        )
    candidates.append(filename)
    failures = []
    for candidate in dict.fromkeys(value for value in candidates if value):
        try:
            return ctypes.CDLL(
                candidate, **({} if os.name == "nt" else {"mode": ctypes.RTLD_GLOBAL})
            )
        except OSError as error:
            failures.append(f"{candidate}: {error}")
    raise XadxError("could not load HRX; tried " + "; ".join(failures))


class _Runtime:
    def __init__(self, library=None, device_index=0):
        library = _load_library() if library is None else library
        self._bind_library(library)
        try:
            self._check(self._initialize(0), "hrx_gpu_initialize")
            self._initialized = True
            self._check(
                self._device_get(device_index, ctypes.byref(self.device)),
                "hrx_gpu_device_get",
            )
            self._check(
                self._stream_create(self.device, 0, ctypes.byref(self.stream)),
                "hrx_stream_create",
            )
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def _bind_library(self, library):
        self._status_code = _bind(library, "hrx_status_code", ctypes.c_int, [_Status])
        self._status_ignore = _bind(library, "hrx_status_ignore", None, [_Status])
        self._status_to_string = _bind(
            library,
            "hrx_status_to_string",
            _Status,
            [
                _Status,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_size_t),
            ],
        )
        self._status_free_message = _bind(
            library, "hrx_status_free_message", None, [ctypes.c_char_p]
        )
        self._initialize = _bind(
            library, "hrx_gpu_initialize", _Status, [ctypes.c_uint32]
        )
        self._shutdown = _bind(library, "hrx_gpu_shutdown", _Status, [])
        self._device_get = _bind(
            library,
            "hrx_gpu_device_get",
            _Status,
            [ctypes.c_int, ctypes.POINTER(_Handle)],
        )
        self._device_release = _bind(library, "hrx_device_release", None, [_Handle])
        self._device_property = _bind(
            library,
            "hrx_device_get_property",
            _Status,
            [_Handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
        )
        self._stream_create = _bind(
            library,
            "hrx_stream_create",
            _Status,
            [_Handle, ctypes.c_uint32, ctypes.POINTER(_Handle)],
        )
        self._stream_release = _bind(library, "hrx_stream_release", None, [_Handle])
        self._synchronize = _bind(library, "hrx_stream_synchronize", _Status, [_Handle])
        self._allocate = _bind(
            library,
            "hrx_buffer_allocate",
            _Status,
            [
                _Handle,
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(_Handle),
            ],
        )
        self._map = _bind(
            library,
            "hrx_buffer_map_with_mode",
            _Status,
            [
                _Handle,
                ctypes.c_uint32,
                ctypes.c_uint16,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
            ],
        )
        self._unmap = _bind(library, "hrx_buffer_unmap", _Status, [_Handle])
        self._buffer_release = _bind(library, "hrx_buffer_release", None, [_Handle])
        self._flush = _bind(
            library,
            "hrx_buffer_flush_range",
            _Status,
            [_Handle, ctypes.c_size_t, ctypes.c_size_t],
        )
        self._invalidate = _bind(
            library,
            "hrx_buffer_invalidate_range",
            _Status,
            [_Handle, ctypes.c_size_t, ctypes.c_size_t],
        )
        self._load = _bind(
            library,
            "hrx_executable_load_data",
            _Status,
            [
                _Handle,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.POINTER(_Handle),
            ],
        )
        self._lookup = _bind(
            library,
            "hrx_executable_lookup_export_by_name",
            _Status,
            [_Handle, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)],
        )
        self._executable_release = _bind(
            library, "hrx_executable_release", None, [_Handle]
        )
        self._dispatch = _bind(
            library,
            "hrx_stream_dispatch",
            _Status,
            [
                _Handle,
                _Handle,
                ctypes.c_uint32,
                ctypes.POINTER(_DispatchConfig),
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(_BufferRef),
                ctypes.c_size_t,
                ctypes.c_uint32,
            ],
        )
        self.device = _Handle()
        self.stream = _Handle()
        self.resources = []
        self._initialized = False

    def _check(self, status, operation):
        if not status:
            return
        message = ctypes.c_char_p()
        message_length = ctypes.c_size_t()
        secondary = self._status_to_string(
            status, ctypes.byref(message), ctypes.byref(message_length)
        )
        code = self._status_code(status)
        text = (
            ctypes.string_at(message, message_length.value).decode(
                "utf-8", errors="replace"
            )
            if message
            else "unknown error"
        )
        if message:
            self._status_free_message(message)
        if secondary:
            self._status_ignore(secondary)
        self._status_ignore(status)
        raise XadxError(
            f"{operation} failed (status {code}): {text}",
            code=code,
            operation=operation,
        )

    def architecture(self):
        value = ctypes.create_string_buffer(256)
        self._check(
            self._device_property(
                self.device, _ARCHITECTURE_PROPERTY, value, len(value)
            ),
            "hrx_device_get_property(architecture)",
        )
        return value.value.decode("utf-8", "replace")

    def load(self, data, export):
        storage = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        handle = _Handle()
        self._check(
            self._load(
                self.device,
                ctypes.cast(storage, ctypes.c_void_p),
                len(data),
                _TARGET_FAMILY,
                _TARGET_KEY,
                ctypes.byref(handle),
            ),
            "hrx_executable_load_data",
        )
        ordinal = ctypes.c_uint32()
        try:
            self._check(
                self._lookup(handle, export.encode(), ctypes.byref(ordinal)),
                "export lookup",
            )
        except Exception:
            self._executable_release(handle)
            raise
        self.resources.append(("executable", handle, storage))
        return handle, ordinal.value

    def allocate(self, data):
        size = len(data)
        allocation_size = size + 2 * _GUARD_BYTES
        handle = _Handle()
        self._check(
            self._allocate(
                self.stream,
                allocation_size,
                _MEMORY_TYPE,
                _BUFFER_USAGE,
                ctypes.byref(handle),
            ),
            "hrx_buffer_allocate",
        )
        pointer = ctypes.c_void_p()
        mapped = False
        try:
            self._check(
                self._map(
                    handle,
                    _MAPPING_MODE,
                    3,
                    0,
                    allocation_size,
                    ctypes.byref(pointer),
                ),
                "hrx_buffer_map_with_mode",
            )
            mapped = True
            backing = (ctypes.c_uint8 * allocation_size).from_address(pointer.value)
            backing[:] = bytes([_GUARD_VALUE]) * allocation_size
            storage = memoryview(backing).cast("B")[_GUARD_BYTES : _GUARD_BYTES + size]
            storage[:] = data
            self._check(
                self._flush(handle, 0, allocation_size),
                "hrx_buffer_flush_range",
            )
        except Exception:
            if mapped:
                status = self._unmap(handle)
                if status:
                    self._status_ignore(status)
            self._buffer_release(handle)
            raise
        self.resources.append(("buffer", handle, backing))
        return _GuardedBuffer(handle, storage, backing, _GUARD_BYTES)

    def run(self, executable, ordinal, buffers):
        refs = (_BufferRef * len(buffers))(
            *(
                _BufferRef(
                    allocation.handle,
                    allocation.offset,
                    len(allocation.storage),
                )
                if isinstance(allocation, _GuardedBuffer)
                else _BufferRef(allocation[0], 0, len(allocation[1]))
                for allocation in buffers
            )
        )
        config = _DispatchConfig((1, 1, 1), (1, 1, 1), 0)
        self._check(
            self._dispatch(
                self.stream,
                executable,
                ordinal,
                ctypes.byref(config),
                None,
                0,
                refs,
                len(refs),
                0,
            ),
            "hrx_stream_dispatch",
        )
        self._check(self._synchronize(self.stream), "hrx_stream_synchronize")

    def read(self, buffer):
        handle, storage = buffer
        offset = buffer.offset if isinstance(buffer, _GuardedBuffer) else 0
        self._check(
            self._invalidate(handle, offset, len(storage)),
            "hrx_buffer_invalidate_range",
        )
        return bytes(storage)

    def inspect(self, buffer):
        if not isinstance(buffer, _GuardedBuffer):
            return b"", self.read(buffer), b""
        self._check(
            self._invalidate(buffer.handle, 0, len(buffer.backing)),
            "hrx_buffer_invalidate_range",
        )
        end = buffer.offset + len(buffer.storage)
        return (
            bytes(buffer.backing[: buffer.offset]),
            bytes(buffer.storage),
            bytes(buffer.backing[end:]),
        )

    def inspect_guards(self, buffer):
        """Inspect only guard ranges, without invalidating the payload."""
        if not isinstance(buffer, _GuardedBuffer):
            return b"", b""
        if buffer.offset:
            self._check(
                self._invalidate(buffer.handle, 0, buffer.offset),
                "hrx_buffer_invalidate_range",
            )
        end = buffer.offset + len(buffer.storage)
        suffix_size = len(buffer.backing) - end
        if suffix_size:
            self._check(
                self._invalidate(buffer.handle, end, suffix_size),
                "hrx_buffer_invalidate_range",
            )
        return bytes(buffer.backing[: buffer.offset]), bytes(buffer.backing[end:])

    def rollback(self, mark):
        """Release resources allocated after a failed preparation checkpoint."""
        cleanup_error = None
        for kind, handle, _ in reversed(self.resources[mark:]):
            if kind == "buffer":
                try:
                    self._check(self._unmap(handle), "hrx_buffer_unmap")
                except Exception as error:
                    cleanup_error = cleanup_error or error
                self._buffer_release(handle)
            else:
                self._executable_release(handle)
        del self.resources[mark:]
        if cleanup_error is not None:
            raise cleanup_error

    def close(self):
        for kind, handle, _ in reversed(self.resources):
            if kind == "buffer":
                self._check(self._unmap(handle), "hrx_buffer_unmap")
                self._buffer_release(handle)
            else:
                self._executable_release(handle)
        self.resources.clear()
        if self.stream:
            self._stream_release(self.stream)
            self.stream = _Handle()
        if self.device:
            self._device_release(self.device)
            self.device = _Handle()
        if self._initialized:
            self._check(self._shutdown(), "hrx_gpu_shutdown")
            self._initialized = False


def _compare(actual, expected, comparison, atol, rtol):
    return _compare_details(actual, expected, comparison, atol, rtol)["failure"]


def _compare_details(actual, expected, comparison, atol, rtol):
    if len(actual) != len(expected):
        return {
            "failure": f"size mismatch: {len(actual)} != {len(expected)}",
            "mismatch_count": abs(len(actual) - len(expected)) or 1,
            "first_mismatch": {
                "kind": "size",
                "actual": len(actual),
                "expected": len(expected),
            },
        }
    if comparison == "exact":
        if actual == expected:
            return {"failure": None, "mismatch_count": 0, "first_mismatch": None}
        np, _ = _ensure_numeric_runtime()
        mismatch = np.frombuffer(actual, dtype=np.uint8) != np.frombuffer(
            expected, dtype=np.uint8
        )
        mismatch_count = int(np.count_nonzero(mismatch))
        first = int(np.argmax(mismatch))
        return {
            "failure": f"exact mismatch at byte {first}",
            "mismatch_count": mismatch_count,
            "first_mismatch": {
                "kind": "byte",
                "index": first,
                "actual": actual[first],
                "expected": expected[first],
            },
        }
    if comparison != "bf16-tolerance" or len(actual) % 2:
        raise XadxError(f"unsupported comparison contract: {comparison}")
    np, ml_dtypes = _ensure_numeric_runtime()
    got = np.frombuffer(actual, dtype="<u2").view(ml_dtypes.bfloat16).astype(np.float32)
    wanted = (
        np.frombuffer(expected, dtype="<u2").view(ml_dtypes.bfloat16).astype(np.float32)
    )
    # isclose handles infinities by exact signed equality. Equal NaNs are
    # accepted because an expected NaN is a bit-level statement that the result
    # is also undefined; a finite result may never pass against NaN or infinity.
    close = np.isclose(got, wanted, atol=atol, rtol=rtol, equal_nan=True)
    mismatch_indices = np.flatnonzero(~close)
    if not mismatch_indices.size:
        return {"failure": None, "mismatch_count": 0, "first_mismatch": None}
    first = int(mismatch_indices[0])
    actual_value = float(got[first])
    expected_value = float(wanted[first])
    return {
        "failure": (
            f"{mismatch_indices.size} BF16 mismatches; first "
            f"({first}, {actual_value}, {expected_value})"
        ),
        "mismatch_count": int(mismatch_indices.size),
        "first_mismatch": {
            "kind": "bf16",
            "index": first,
            "actual": actual_value if np.isfinite(actual_value) else str(actual_value),
            "expected": (
                expected_value if np.isfinite(expected_value) else str(expected_value)
            ),
        },
    }


def _is_transient_open_error(error):
    """Whether an HRX open failed in the autosuspend-resume EAGAIN window."""
    return (
        isinstance(error, XadxError)
        and error.code == _HRX_UNAVAILABLE
        and error.operation == "hrx_gpu_initialize"
    )


def _runtime_open_payload(record):
    return {
        "attempts": record["attempts"],
        "unavailable_retries": record["unavailable_retries"],
    }


def _open_runtime(factory, record):
    """Open HRX, retrying UNAVAILABLE resumes that some amdxdna drivers flake on."""
    delays = _OPEN_RETRY_DELAYS
    attempts = 1 + len(delays)
    for attempt in range(attempts):
        record["attempts"] += 1
        try:
            return factory()
        except XadxError as error:
            if attempt + 1 == attempts or not _is_transient_open_error(error):
                error.runtime_open = _runtime_open_payload(record)
                raise
            record["unavailable_retries"] += 1
            time.sleep(delays[attempt])


class _Session:
    """Holds the runtime so a lifecycle round can replace a leaking one.

    A case that leaves a resource behind must not carry that state into the next
    round, and the only reliable way to drop it is to close the runtime and open
    a fresh one. Callers reach the current runtime through this rather than
    capturing it, so the replacement is visible to them.
    """

    def __init__(self, factory):
        self.factory = factory
        self.open_record = {"attempts": 0, "unavailable_retries": 0}
        self.runtime = _open_runtime(factory, self.open_record)

    def reset(self):
        self.runtime.close()
        # Same UNAVAILABLE retry as the first open. The throwaway record is so
        # a later reopen is not mixed into the initial-open counts on the report.
        self.runtime = _open_runtime(
            self.factory, {"attempts": 0, "unavailable_retries": 0}
        )

    def close(self):
        self.runtime.close()


# AXLF uniqueId is XRT axlf.m_uniqueId at offset 0x128 (magic 8 + cipher 32 +
# keyBlock 256). Windows keys HAL contexts by full xclbin bytes, so rewriting
# this field is enough to miss the cache without changing PDI or control code.
_AXLF_MAGIC = b"xclbin2\0"
_AXLF_UNIQUE_ID_OFFSET = 0x128
_AXLF_UNIQUE_ID_BASE = 0xA11CE00000000000


def _uniquify_xadx_axlf(data, unique_id):
    patched = bytearray(data)
    found = 0
    start = 0
    token = int(unique_id).to_bytes(8, "little")
    while True:
        offset = patched.find(_AXLF_MAGIC, start)
        if offset < 0:
            break
        uid = offset + _AXLF_UNIQUE_ID_OFFSET
        if uid + 8 > len(patched):
            raise XadxError("embedded AXLF is truncated")
        patched[uid : uid + 8] = token
        found += 1
        start = offset + len(_AXLF_MAGIC)
    if found == 0:
        raise XadxError("XADX does not embed an AXLF/xclbin2 image")
    return bytes(patched)


def _rollback_runtime(runtime, mark):
    if hasattr(runtime, "rollback"):
        runtime.rollback(mark)
        return
    del runtime.resources[mark:]


def _soak_sample(cases, executable_data):
    axlf_items = [
        (key, data) for key, data in executable_data.items() if _AXLF_MAGIC in data
    ]
    if not axlf_items:
        return None
    key, data = min(axlf_items, key=lambda item: len(item[1]))
    impl_id, execution = key
    for case in cases:
        for stage in case["stages"]:
            if (
                stage["implementation"] == impl_id
                and stage.get("execution") == execution
            ):
                return case, key, data
    raise XadxError(
        "unique-xclbin-context-soak found an AXLF image with no recipe case"
    )


def _soak_payload(status, images, duration_seconds, error=None):
    payload = {
        "status": status,
        "images": int(images),
        "duration_seconds": float(duration_seconds),
    }
    if error:
        payload["error"] = error
    return payload


def _soak_unique_xclbin_contexts(
    session, cases, executable_data, *, prepare, unique_xclbin_context_soak
):
    """Load more distinct xclbin images than the Windows HAL cache in one runtime.

    Recipe coverage only has a handful of unique production xclbins, and
    lifecycle rounds reset the device. Windows sizes the context cache at
    architecture-budget-1 (31 on Strix Halo). This clone-and-uniquify pass
    loads 33 distinct images so LRU must evict before CreateContext rather
    than recovering from a 0xc01e0009 remap. The soak is a counted XADX test
    (`unique-xclbin-context-soak`), not a recipe case.
    """
    started = time.monotonic()
    if unique_xclbin_context_soak <= 0:
        return _soak_payload("unsupported", 0, 0.0)
    sample = _soak_sample(cases, executable_data)
    if sample is None:
        return _soak_payload("unsupported", 0, max(time.monotonic() - started, 0.0))
    case, key, original = sample
    runtime = session.runtime
    try:
        for index in range(unique_xclbin_context_soak):
            unique_id = _AXLF_UNIQUE_ID_BASE + index
            uniquified = _uniquify_xadx_axlf(original, unique_id)
            if uniquified == original:
                uniquified = _uniquify_xadx_axlf(original, unique_id ^ (1 << 63))
            if uniquified == original:
                raise XadxError(
                    "unique-xclbin-context-soak could not distinguish AXLF uniqueId"
                )
            executable_data[key] = uniquified
            mark = len(runtime.resources)
            try:
                plan, *_ = prepare(case)
                for executable, ordinal, buffers, _, _ in plan:
                    runtime.run(executable, ordinal, buffers)
            except Exception as error:
                raise XadxError(
                    "unique-xclbin-context-soak image "
                    f"{index + 1}/{unique_xclbin_context_soak} failed: {error}"
                ) from error
            finally:
                try:
                    _rollback_runtime(runtime, mark)
                except Exception as error:
                    raise XadxError(
                        "unique-xclbin-context-soak failed to release image "
                        f"{index + 1}: {error}"
                    ) from error
    finally:
        executable_data[key] = original
    return _soak_payload(
        "pass",
        unique_xclbin_context_soak,
        max(time.monotonic() - started, 1e-9),
    )


def _run_rounds(
    session,
    cases,
    *,
    prepare,
    dispatch,
    readback,
    lifecycle_rounds,
    resident_rounds,
    readback_rounds,
):
    """Drive the lifecycle, resident, and readback rounds over a set of cases.

    This is the qualification contract rather than a detail of any one case
    format, so it is shared: a kernel is torn down and rebuilt, then held
    resident, then has an input rewritten underneath it between dispatches.

    `readback` returns None for a case with nothing to read back, or a callable
    taking the prepared case and the round index.

    Returns the failures and how many readback rounds each case actually
    received, so nothing downstream has to infer that from a case name.
    """
    failures = []

    def record(case, phase, round_number, message):
        failures.append(
            {
                "case": case["name"],
                "phase": phase,
                "round": round_number,
                "message": str(message),
            }
        )

    for round_index in range(lifecycle_rounds):
        for case in cases:
            mark = len(session.runtime.resources)
            try:
                failure = dispatch(case, prepare(case))
                if failure:
                    record(case, "lifecycle", round_index + 1, failure)
            except Exception as error:
                record(case, "lifecycle", round_index + 1, error)
            finally:
                if len(session.runtime.resources) != mark:
                    session.reset()

    # Keep the full case inventory loaded together. Besides testing repeated
    # dispatch through one executable, preserving executable and device-buffer
    # residency pressure across the declared case inventory.
    resident = []
    for case in cases:
        mark = len(session.runtime.resources)
        try:
            resident.append((case, prepare(case), readback(case)))
        except Exception as error:
            record(case, "resident-preparation", None, error)
            if hasattr(session.runtime, "rollback"):
                session.runtime.rollback(mark)
            else:
                del session.runtime.resources[mark:]
    for round_index in range(resident_rounds):
        for case, prepared, _ in resident:
            try:
                failure = dispatch(case, prepared)
                if failure:
                    record(case, "resident", round_index + 1, failure)
            except Exception as error:
                record(case, "resident", round_index + 1, error)

    coherence = [
        (case, prepared, revisit)
        for case, prepared, revisit in resident
        if revisit is not None
    ]
    visits = {case["name"]: 0 for case, _, _ in coherence}
    for round_index in range(readback_rounds):
        for case, prepared, revisit in coherence:
            visits[case["name"]] += 1
            try:
                failure = revisit(prepared, round_index)
                if failure:
                    record(case, "readback", round_index + 1, failure)
            except Exception as error:
                record(case, "readback", round_index + 1, error)
    return failures, visits


def _materialize_recipe_buffers(cases, generators):
    """Bound all unique bytes before allocating any generator output."""
    planned = {}
    resident_bytes = 0

    def plan(specification, expected_size):
        if not 1 <= expected_size <= _MAX_BUFFER_BYTES:
            raise XadxError(
                f"recipe buffer size {expected_size} exceeds the 16 MiB limit"
            )
        try:
            predicted = (
                generators.predicted_size(
                    specification["id"], specification["parameters"]
                )
                if hasattr(generators, "predicted_size")
                else expected_size
            )
        except (TypeError, ValueError, KeyError) as error:
            raise XadxError(f"invalid recipe generator parameters: {error}") from error
        if predicted != expected_size:
            raise XadxError(
                f"recipe generator predicts {predicted} bytes, not {expected_size}"
            )
        key = json.dumps(specification, sort_keys=True, separators=(",", ":"))
        previous = planned.setdefault(key, (specification, predicted))
        if previous[1] != predicted:
            raise XadxError("recipe reuses a generator at conflicting sizes")

    for case in cases:
        for stage in case["stages"]:
            for record in stage["buffers"]:
                if "stage_output" not in record.get("source", {}):
                    resident_bytes += record["size"] + 2 * _GUARD_BYTES
    try:
        requests = recipe_materialization_requests(cases)
    except (KeyError, ValueError) as error:
        raise XadxError(f"invalid recipe materialization contract: {error}") from error
    for specification, size in requests:
        plan(specification, size)

    materialized_bytes = sum(size for _, size in planned.values())
    if materialized_bytes > _MAX_MATERIALIZED_BYTES:
        raise XadxError("recipe exceeds the 128 MiB materialization limit")
    if resident_bytes > _MAX_RESIDENT_BYTES:
        raise XadxError("recipe exceeds the 256 MiB resident allocation limit")

    result = {}
    for key, (specification, expected_size) in planned.items():
        try:
            data = generators.materialize(
                specification["id"], specification["parameters"]
            )
        except (OSError, ValueError, KeyError) as error:
            raise XadxError(f"cannot materialize recipe buffer: {error}") from error
        if len(data) != expected_size:
            raise XadxError(
                f"recipe buffer size mismatch: {len(data)} != {expected_size}"
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest != specification["sha256"]:
            raise XadxError(
                f"recipe generator {specification['id']} produced {digest}; "
                f"expected {specification['sha256']}"
            )
        result[key] = (
            data,
            {
                "source": specification,
                "size": len(data),
                "sha256": digest,
                "contract_sha256": hashlib.sha256(key.encode()).hexdigest(),
            },
        )
    return result


def _architecture_verdict(architecture, target_family):
    """Whether an observed architecture may run NPU2 assets.

    HRX does not distinguish the silicon: on NPU2 parts it reports "amdxdna" for
    both the device name and the architecture, and offers no compute-unit, PCI, or
    clock property to corroborate it. The part is identified upstream instead, by
    the native probe reading the PCI device ID, so accepting the driver's generic
    name requires the probe to have named the family first.
    """
    tokens = {
        token for token in re.split(r"[^a-z0-9]+", architecture.casefold()) if token
    }
    return bool(tokens.intersection({"npu2", "xdna2"})) or (
        target_family == "npu2" and bool(tokens.intersection({"amdxdna"}))
    )


def _executable_identities(implementations):
    identities = []
    for id, entry in implementations.items():
        for execution, executable in [
            (None, entry),
            *[
                (variant["id"], variant)
                for variant in entry.get("execution_variants", ())
            ],
        ]:
            identities.append(
                {
                    "id": id,
                    **({} if execution is None else {"execution": execution}),
                    "xclbin_path": entry["xclbin_path"],
                    "xclbin_sha256": entry["xclbin_sha256"],
                    "xadx": executable["xadx"],
                    "xadx_sha256": executable["xadx_sha256"],
                    "transaction": executable["transaction"],
                    "transaction_sha256": executable["transaction_sha256"],
                }
            )
    return sorted(identities, key=lambda item: (item["id"], item.get("execution", "")))


def _unsupported_report(
    recipe, recipe_bytes, cases, executables, architecture, reason, runtime_open=None
):
    report = {
        "schema_version": XADX_REPORT_SCHEMA_VERSION,
        "report_kind": "execution",
        "status": "unsupported",
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "generator_version": recipe["generator_version"],
        "executables": executables,
        "reason": reason,
        "architecture": architecture,
        "case_names": [case["name"] for case in cases],
        "case_results": [
            {
                "name": case["name"],
                "status": "unsupported",
                "dispatches": 0,
                "readbacks": 0,
                "validated_bytes": 0,
                "duration_seconds": 0.0,
                "mismatch_count": 0,
                "first_mismatch": None,
                "failures": [],
            }
            for case in cases
        ],
        "failures": [],
        "unique_xclbin_context_soak": _soak_payload("unsupported", 0, 0.0),
    }
    if runtime_open is not None:
        report["runtime_open"] = runtime_open
    return validate_xadx_report(report)


def _validate_recipe(recipe):
    """Rejects ambiguous or unsafe recipes before opening the device."""
    try:
        validate_schema(recipe, load_packaged_schema("kernel-recipe-v1.schema.json"))
    except (SchemaValidationError, OSError, json.JSONDecodeError) as error:
        raise XadxError(
            f"kernel recipe does not satisfy the v1 schema: {error}"
        ) from error
    entries = recipe["implementations"]
    cases = recipe["cases"]
    implementations = {}
    for entry in entries:
        id = entry.get("id")
        if not isinstance(id, str) or id in implementations:
            raise XadxError("kernel recipe implementation IDs are invalid")
        xadx_path = entry.get("xadx")
        parsed = PurePosixPath(xadx_path) if isinstance(xadx_path, str) else None
        xclbin_path = entry.get("xclbin_path")
        parsed_xclbin = (
            PurePosixPath(xclbin_path) if isinstance(xclbin_path, str) else None
        )
        transaction_path = entry.get("transaction")
        parsed_transaction = (
            PurePosixPath(transaction_path)
            if isinstance(transaction_path, str)
            else None
        )
        if (
            parsed is None
            or parsed.is_absolute()
            or ".." in parsed.parts
            or "\\" in xadx_path
            or parsed.suffix != ".xadx"
        ):
            raise XadxError(f"implementation {id} has an unsafe XADX path")
        if (
            parsed_xclbin is None
            or parsed_xclbin.is_absolute()
            or ".." in parsed_xclbin.parts
            or "\\" in xclbin_path
            or parsed_xclbin.suffix != ".xclbin"
        ):
            raise XadxError(f"implementation {id} has an unsafe source xclbin path")
        if (
            parsed_transaction is None
            or parsed_transaction.is_absolute()
            or ".." in parsed_transaction.parts
            or "\\" in transaction_path
            or parsed_transaction.suffix != ".bin"
        ):
            raise XadxError(f"implementation {id} has an unsafe transaction path")
        for field in (
            "xclbin_sha256",
            "transaction_sha256",
            "xadx_sha256",
        ):
            value = entry.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise XadxError(f"implementation {id} has an invalid {field}")
        expected_id = f"{entry.get('kernel')}-{entry['xclbin_sha256'][:16]}"
        if id != expected_id:
            raise XadxError(
                f"implementation {id} does not match source identity {expected_id}"
            )
        variants = entry.get("execution_variants", [])
        variant_ids = set()
        if not isinstance(variants, list) or len(variants) > 8:
            raise XadxError(f"implementation {id} has invalid execution variants")
        for variant in variants:
            variant_id = variant.get("id")
            if (
                not isinstance(variant_id, str)
                or not variant_id
                or variant_id in variant_ids
            ):
                raise XadxError(
                    f"implementation {id} has invalid execution variant IDs"
                )
            variant_ids.add(variant_id)
            shape = variant.get("shape")
            if (
                not isinstance(shape, dict)
                or set(shape) != {"m", "k", "n"}
                or any(
                    not isinstance(value, int) or not 0 < value <= 4096
                    for value in shape.values()
                )
            ):
                raise XadxError(f"implementation {id}/{variant_id} has invalid shape")
            for path_field, suffix in (("xadx", ".xadx"), ("transaction", ".bin")):
                value = variant.get(path_field)
                parsed_variant = (
                    PurePosixPath(value) if isinstance(value, str) else None
                )
                if (
                    parsed_variant is None
                    or parsed_variant.is_absolute()
                    or ".." in parsed_variant.parts
                    or "\\" in value
                    or parsed_variant.suffix != suffix
                ):
                    raise XadxError(
                        f"implementation {id}/{variant_id} has unsafe {path_field}"
                    )
            for field in ("transaction_sha256", "xadx_sha256"):
                value = variant.get(field)
                if (
                    not isinstance(value, str)
                    or re.fullmatch(r"[0-9a-f]{64}", value) is None
                ):
                    raise XadxError(
                        f"implementation {id}/{variant_id} has invalid {field}"
                    )
        implementations[id] = entry
    case_names = set()
    for case in cases:
        name = case.get("name")
        if not isinstance(name, str) or not name or name in case_names:
            raise XadxError("kernel recipe case names are invalid")
        case_names.add(name)
        stages = case.get("stages")
        if not isinstance(stages, list) or not stages or len(stages) > 8:
            raise XadxError(f"{name} has an invalid stage inventory")
        staged = {}
        expected_count = 0
        stage_names = set()
        for stage in stages:
            implementation = stage.get("implementation")
            if implementation not in implementations:
                raise XadxError(
                    f"{name} dispatches {implementation}, "
                    "which the recipe does not describe"
                )
            execution = stage.get("execution")
            if execution is not None and execution not in {
                variant["id"]
                for variant in implementations[implementation].get(
                    "execution_variants", ()
                )
            }:
                raise XadxError(
                    f"{name} dispatches unknown execution variant "
                    f"{implementation}/{execution}"
                )
            stage_name = stage.get("name")
            if (
                not isinstance(stage_name, str)
                or not stage_name
                or stage_name in stage_names
            ):
                raise XadxError(f"{name} has duplicate stage name {stage_name!r}")
            stage_names.add(stage_name)
            buffers = stage.get("buffers")
            if not isinstance(buffers, list) or not buffers or len(buffers) > 16:
                raise XadxError(f"{name}/{stage_name} has invalid buffers")
            local_names = set()
            for buffer in buffers:
                buffer_name = buffer.get("name")
                if (
                    not isinstance(buffer_name, str)
                    or not buffer_name
                    or buffer_name in local_names
                ):
                    raise XadxError(
                        f"{name}/{stage_name} has duplicate or invalid buffer names"
                    )
                local_names.add(buffer_name)
                size = buffer.get("size")
                if not isinstance(size, int) or not 0 < size <= _MAX_BUFFER_BYTES:
                    raise XadxError(
                        f"{name}/{stage_name}.{buffer_name} has invalid size"
                    )
                direction = buffer.get("direction")
                source = buffer.get("source")
                expected = buffer.get("expected")
                preserved = buffer.get("preserved")
                comparison = buffer.get("comparison")
                if direction == "input" and source is None:
                    raise XadxError(f"{name}/{stage_name}.{buffer_name} has no source")
                if direction == "output":
                    if source is not None:
                        raise XadxError(
                            f"{name}/{stage_name}.{buffer_name} output has a source"
                        )
                    staged[f"{stage_name}.{buffer_name}"] = {
                        "size": size,
                        "preserved": preserved,
                    }
                elif direction != "input":
                    raise XadxError(
                        f"{name}/{stage_name}.{buffer_name} has invalid direction"
                    )
                if isinstance(source, dict) and "stage_output" in source:
                    alias = source["stage_output"]
                    if alias not in staged:
                        raise XadxError(
                            f"{name} binds {alias}, which no earlier stage wrote"
                        )
                    if staged[alias]["size"] != size:
                        raise XadxError(
                            f"{name} binds {alias} at {staged[alias]['size']} B into a "
                            f"{size} B role"
                        )
                    if staged[alias]["preserved"] is None:
                        raise XadxError(
                            f"{name} binds {alias} without a preserved-output oracle"
                        )
                if expected is not None:
                    expected_count += 1
                    if direction != "output" or not isinstance(comparison, dict):
                        raise XadxError(
                            f"{name}/{stage_name}.{buffer_name} has an invalid "
                            "expectation"
                        )
                elif comparison is not None:
                    raise XadxError(
                        f"{name}/{stage_name}.{buffer_name} compares no expectation"
                    )
                if preserved is not None:
                    if direction != "output":
                        raise XadxError(
                            f"{name}/{stage_name}.{buffer_name} preserves a non-output"
                        )
        if expected_count != 1:
            raise XadxError(f"{name} must declare exactly one expectation")
        for variant in case.get("readback_variants", ()):
            identity = f"{variant.get('stage')}.{variant.get('buffer')}"
            target = next(
                (
                    buffer
                    for stage in stages
                    if stage["name"] == variant.get("stage")
                    for buffer in stage["buffers"]
                    if buffer["name"] == variant.get("buffer")
                ),
                None,
            )
            if not isinstance(variant.get("stage"), str) or target is None:
                raise XadxError(f"{name} readback names unknown buffer {identity!r}")
            if target["direction"] != "input" or "stage_output" in target.get(
                "source", {}
            ):
                raise XadxError(
                    f"{name} readback target {identity!r} is not a resident input"
                )
    return implementations, cases


def run_recipe(
    recipe,
    root,
    *,
    runtime_factory=_Runtime,
    generators=None,
    lifecycle_rounds=XADX_ROUNDS["default"]["lifecycle_rounds"],
    resident_rounds=XADX_ROUNDS["default"]["resident_rounds"],
    readback_rounds=XADX_ROUNDS["default"]["readback_rounds"],
    unique_xclbin_context_soak=XADX_UNIQUE_XCLBIN_CONTEXT_SOAK,
    target_family=None,
):
    """Run a kernel recipe, staging buffers between dispatches within a case.

    A recipe case is a sequence of stages sharing buffers, so a dequant stage can
    hand its output straight to an MM stage without a host round trip. Only the
    stage that declares an expectation is compared; earlier stages are the setup
    the expectation depends on.
    """
    if (
        min(
            lifecycle_rounds,
            resident_rounds,
            readback_rounds,
            unique_xclbin_context_soak,
        )
        < 0
    ):
        raise ValueError("round counts must be nonnegative")
    if (
        max(
            lifecycle_rounds,
            resident_rounds,
            readback_rounds,
            unique_xclbin_context_soak,
        )
        > 1000
    ):
        raise ValueError("round counts exceed the safety bound (1000)")
    root = Path(root)
    if isinstance(recipe, (str, Path)):
        recipe_path = Path(recipe)
        try:
            recipe_bytes = recipe_path.read_bytes()
            recipe = json.loads(recipe_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise XadxError(f"cannot read kernel recipe: {error}") from error
    elif isinstance(recipe, (bytes, bytearray)):
        recipe_bytes = bytes(recipe)
        try:
            recipe = json.loads(recipe_bytes)
        except json.JSONDecodeError as error:
            raise XadxError(f"cannot parse kernel recipe: {error}") from error
    elif isinstance(recipe, dict):
        recipe_bytes = json.dumps(
            recipe, sort_keys=True, separators=(",", ":")
        ).encode()
    else:
        raise TypeError("recipe must be a mapping, bytes, or a path")
    if generators is None:
        from . import recipe_generators as generators

    implementations, cases = _validate_recipe(recipe)
    executable_identities = _executable_identities(implementations)
    if target_family not in (None, "npu2"):
        return _unsupported_report(
            recipe,
            recipe_bytes,
            cases,
            executable_identities,
            "unknown",
            f"platform probe reported unsupported target family {target_family!r}",
        )
    uneven = [
        case["name"]
        for case in cases
        if case.get("readback_variants")
        and readback_rounds % len(case["readback_variants"])
    ]
    if uneven:
        raise XadxError(
            "readback rounds do not evenly cover variants for: " + ", ".join(uneven)
        )
    materialized = _materialize_recipe_buffers(cases, generators)
    executable_data = {}
    for id, entry in implementations.items():
        executions = [
            (None, entry),
            *[
                (variant["id"], variant)
                for variant in entry.get("execution_variants", ())
            ],
        ]
        for execution, executable in executions:
            label = id if execution is None else f"{id}/{execution}"
            try:
                data = (root / executable["xadx"]).read_bytes()
            except OSError as error:
                raise XadxError(f"cannot read {label} XADX: {error}") from error
            digest = hashlib.sha256(data).hexdigest()
            if digest != executable["xadx_sha256"]:
                raise XadxError(
                    f"{label} XADX digest mismatch: {digest} != "
                    f"{executable['xadx_sha256']}"
                )
            executable_data[(id, execution)] = data
            try:
                transaction = (root / executable["transaction"]).read_bytes()
            except OSError as error:
                raise XadxError(f"cannot read {label} transaction: {error}") from error
            transaction_digest = hashlib.sha256(transaction).hexdigest()
            if transaction_digest != executable["transaction_sha256"]:
                raise XadxError(
                    f"{label} transaction digest mismatch: {transaction_digest} "
                    f"!= {executable['transaction_sha256']}"
                )
    session = _Session(runtime_factory)
    try:
        architecture = session.runtime.architecture()
    except Exception as error:
        try:
            session.close()
        except Exception as cleanup_error:
            raise error from cleanup_error
        raise
    if not _architecture_verdict(architecture, target_family):
        session.close()
        return _unsupported_report(
            recipe,
            recipe_bytes,
            cases,
            executable_identities,
            architecture,
            (
                "FastFlowLM kernel recipes require an NPU2/XDNA2 part, named "
                "either by the runtime or by the platform probe; the runtime "
                f"reported architecture {architecture!r} and the probe reported "
                f"target family {target_family!r}"
            ),
            runtime_open=_runtime_open_payload(session.open_record),
        )

    case_metrics = {
        case["name"]: {
            "dispatches": 0,
            "validated_bytes": 0,
            "duration_seconds": 0.0,
            "mismatch_count": 0,
            "first_mismatch": None,
        }
        for case in cases
    }

    def materialize(specification, expected_size):
        key = json.dumps(specification, sort_keys=True, separators=(",", ":"))
        data = materialized[key][0]
        if len(data) != expected_size:
            raise XadxError(
                f"recipe buffer size mismatch: {len(data)} != {expected_size}"
            )
        return data

    def prepare(case):
        """Allocate every stage's buffers, sharing the ones a chain binds."""
        runtime = session.runtime
        staged, plan, comparison, chained_allocations = {}, [], None, set()
        preserved_allocations = {}
        for stage in case["stages"]:
            entry = implementations[stage["implementation"]]
            execution = stage.get("execution")
            executable_entry = (
                entry
                if execution is None
                else next(
                    (
                        variant
                        for variant in entry.get("execution_variants", ())
                        if variant["id"] == execution
                    ),
                    None,
                )
            )
            if executable_entry is None:
                raise XadxError(
                    f"{case['name']} names unavailable execution "
                    f"{entry['id']}/{execution}"
                )
            executable, ordinal = runtime.load(
                executable_data[(entry["id"], execution)],
                executable_entry["export"],
            )
            buffers, outputs = [], []
            for record in stage["buffers"]:
                source = record.get("source")
                if source is not None and "stage_output" in source:
                    staged_output = staged.get(source["stage_output"])
                    if staged_output is None:
                        raise XadxError(
                            f"{case['name']} binds {source['stage_output']}, "
                            "which no earlier stage wrote"
                        )
                    allocation, preserved = staged_output
                    if len(allocation[1]) != record["size"]:
                        raise XadxError(
                            f"{case['name']} binds {source['stage_output']} at "
                            f"{len(allocation[1])} B into a {record['size']} B role"
                        )
                    buffers.append(allocation)
                    chained_allocations.add(builtins.id(allocation))
                    preserved_allocations[builtins.id(allocation)] = (
                        record["name"],
                        preserved,
                        record["size"],
                        allocation,
                    )
                    continue
                if record["direction"] == "output":
                    allocation = runtime.allocate(bytes(record["size"]))
                    outputs.append(allocation)
                    staged[f"{stage['name']}.{record['name']}"] = (
                        allocation,
                        record.get("preserved"),
                    )
                    if "expected" in record:
                        comparison = (
                            allocation,
                            record["expected"],
                            record["size"],
                            record["comparison"],
                        )
                else:
                    allocation = runtime.allocate(materialize(source, record["size"]))
                buffers.append(allocation)
            plan.append((executable, ordinal, buffers, outputs, stage["buffers"]))
        if comparison is None:
            raise XadxError(f"{case['name']} declares no expectation")
        return plan, comparison, chained_allocations, preserved_allocations

    def dispatch(case, prepared, clear_output=True, expected_source=None):
        started = time.monotonic()
        metrics = case_metrics[case["name"]]
        metrics["dispatches"] += 1
        runtime = session.runtime
        (
            plan,
            (output, expected, size, contract),
            chained_allocations,
            preserved_allocations,
        ) = prepared
        if clear_output:
            for _, _, _, outputs, _ in plan:
                for allocation in outputs:
                    handle, storage = allocation
                    storage[:] = bytes(len(storage))
                    offset = (
                        allocation.offset
                        if isinstance(allocation, _GuardedBuffer)
                        else 0
                    )
                    runtime._check(
                        runtime._flush(handle, offset, len(storage)), "output flush"
                    )
        for executable, ordinal, buffers, _, records in plan:
            immutable = {
                builtins.id(allocation): bytes(allocation[1])
                for record, allocation in zip(records, buffers, strict=True)
                if record["direction"] == "input"
                and builtins.id(allocation) not in chained_allocations
            }
            runtime.run(executable, ordinal, buffers)
            for record, allocation in zip(records, buffers, strict=True):
                if builtins.id(allocation) in chained_allocations:
                    if hasattr(runtime, "inspect_guards"):
                        prefix, suffix = runtime.inspect_guards(allocation)
                    else:
                        prefix, suffix = b"", b""
                    payload = None
                elif hasattr(runtime, "inspect"):
                    prefix, payload, suffix = runtime.inspect(allocation)
                else:
                    prefix, payload, suffix = b"", runtime.read(allocation), b""
                if prefix != bytes([_GUARD_VALUE]) * len(prefix):
                    return f"{record['name']} modified its leading guard"
                if suffix != bytes([_GUARD_VALUE]) * len(suffix):
                    return f"{record['name']} modified its trailing guard"
                if (
                    record["direction"] == "input"
                    and payload is not None
                    and payload != immutable[builtins.id(allocation)]
                ):
                    return f"{record['name']} input was modified"
        for (
            name,
            preserved,
            preserved_size,
            allocation,
        ) in preserved_allocations.values():
            if hasattr(runtime, "inspect"):
                prefix, payload, suffix = runtime.inspect(allocation)
            else:
                prefix, payload, suffix = b"", runtime.read(allocation), b""
            if prefix != bytes([_GUARD_VALUE]) * len(prefix):
                return f"{name} modified its leading guard"
            if suffix != bytes([_GUARD_VALUE]) * len(suffix):
                return f"{name} modified its trailing guard"
            if payload != materialize(preserved, preserved_size):
                return f"{name} input was modified"
        wanted = materialize(expected_source or expected, size)
        actual = runtime.read(output)
        comparison_result = _compare_details(
            actual,
            wanted,
            contract["kind"],
            float(contract.get("atol", 0)),
            float(contract.get("rtol", 0)),
        )
        metrics["validated_bytes"] += len(actual)
        metrics["duration_seconds"] += max(time.monotonic() - started, 1e-9)
        metrics["mismatch_count"] += comparison_result["mismatch_count"]
        if metrics["first_mismatch"] is None:
            metrics["first_mismatch"] = comparison_result["first_mismatch"]
        return comparison_result["failure"]

    def readback(case):
        variants = case.get("readback_variants")
        if not variants:
            return None

        def revisit(prepared, round_index):
            runtime = session.runtime
            variant = variants[round_index % len(variants)]
            plan = prepared[0]
            for stage, (_, _, buffers, _, _) in zip(case["stages"], plan, strict=True):
                if stage["name"] != variant["stage"]:
                    continue
                for record, allocation in zip(stage["buffers"], buffers, strict=True):
                    if record["name"] != variant["buffer"]:
                        continue
                    handle, storage = allocation
                    data = materialize(variant["source"], record["size"])
                    if len(data) != len(storage):
                        raise XadxError("readback buffer size mismatch")
                    storage[:] = data
                    offset = (
                        allocation.offset
                        if isinstance(allocation, _GuardedBuffer)
                        else 0
                    )
                    runtime._check(
                        runtime._flush(handle, offset, len(storage)),
                        "readback flush",
                    )
                    return dispatch(
                        case,
                        prepared,
                        clear_output=False,
                        expected_source=variant["expected"],
                    )
            raise XadxError(
                f"{case['name']} has no buffer named "
                f"{variant['stage']}.{variant['buffer']}"
            )

        return revisit

    try:
        soak = _soak_unique_xclbin_contexts(
            session,
            cases,
            executable_data,
            prepare=prepare,
            unique_xclbin_context_soak=unique_xclbin_context_soak,
        )
        failures, readback_by_case = _run_rounds(
            session,
            cases,
            prepare=prepare,
            dispatch=dispatch,
            readback=readback,
            lifecycle_rounds=lifecycle_rounds,
            resident_rounds=resident_rounds,
            readback_rounds=readback_rounds,
        )
    finally:
        session.close()
    case_results = []
    for case in cases:
        name = case["name"]
        metrics = case_metrics[name]
        case_failures = [
            {
                "phase": failure["phase"],
                "round": failure["round"],
                "message": failure["message"],
            }
            for failure in failures
            if failure["case"] == name
        ]
        case_results.append(
            {
                "name": name,
                "status": "fail" if case_failures else "pass",
                "dispatches": metrics["dispatches"],
                "readbacks": readback_by_case.get(name, 0),
                "validated_bytes": metrics["validated_bytes"],
                "duration_seconds": metrics["duration_seconds"],
                "mismatch_count": metrics["mismatch_count"],
                "first_mismatch": metrics["first_mismatch"],
                "failures": case_failures,
            }
        )
    return validate_xadx_report(
        {
            "schema_version": XADX_REPORT_SCHEMA_VERSION,
            "report_kind": "execution",
            "status": "pass" if not failures else "fail",
            "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
            "generator_version": recipe["generator_version"],
            "executables": executable_identities,
            "architecture": architecture,
            "numeric_runtime": _numeric_runtime(),
            "lifecycle_rounds": lifecycle_rounds,
            "resident_rounds": resident_rounds,
            "readback_rounds": readback_rounds,
            "cases": len(cases),
            "case_names": [case["name"] for case in cases],
            "readback_by_case": readback_by_case,
            "case_results": case_results,
            "materialized_buffers": sorted(
                (metadata for _, metadata in materialized.values()),
                key=lambda item: item["contract_sha256"],
            ),
            "runtime_open": _runtime_open_payload(session.open_record),
            "unique_xclbin_context_soak": soak,
            "failures": failures,
        }
    )
