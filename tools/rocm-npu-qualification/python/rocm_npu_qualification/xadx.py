# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compiler-free FastFlowLM XADX qualification runner.

Only the Python standard library and the public HRX C ABI are used here.  The
asset producer is deliberately separate: installed qualification wheels never
need FastFlowLM, NumPy, or a compiler.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
from pathlib import Path

from .contracts import (
    XADX_LIFECYCLE_ROUNDS,
    XADX_READBACK_ROUNDS,
    XADX_RESIDENT_ROUNDS,
)


_Handle = ctypes.c_void_p
_Status = ctypes.c_void_p
XADX_MANIFEST_SCHEMA_VERSION = 1
_FORMAT = b"amdxdna-xclbin-fb"
_MEMORY_TYPE = 0x00000046 | 0x00000010
_BUFFER_USAGE = 0x00000C03 | 0x02000000
_MAPPING_MODE = 0x00000002
_ARCHITECTURE_PROPERTY = 1


class XadxError(RuntimeError):
    pass


def _generator_module():
    try:
        from . import xadx_generators

        return xadx_generators
    except ImportError:
        path = Path(__file__).with_name("xadx_generators.py")
        spec = importlib.util.spec_from_file_location(
            "rocm_npu_qualification_xadx_generators", path
        )
        if spec is None or spec.loader is None:
            raise XadxError(f"cannot load XADX generators from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


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
        self._status_code = _bind(library, "hrx_status_code", ctypes.c_int, [_Status])
        self._status_ignore = _bind(library, "hrx_status_ignore", None, [_Status])
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
        self._check(self._initialize(0), "hrx_gpu_initialize")
        try:
            self._check(
                self._device_get(device_index, ctypes.byref(self.device)),
                "hrx_gpu_device_get",
            )
            self._check(
                self._stream_create(self.device, 0, ctypes.byref(self.stream)),
                "hrx_stream_create",
            )
        except Exception:
            self.close()
            raise

    def _check(self, status, operation):
        if status:
            code = self._status_code(status)
            self._status_ignore(status)
            raise XadxError(f"{operation} failed (status {code})")

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
                _FORMAT,
                ctypes.byref(handle),
            ),
            "hrx_executable_load_data",
        )
        ordinal = ctypes.c_uint32()
        self._check(
            self._lookup(handle, export.encode(), ctypes.byref(ordinal)),
            "export lookup",
        )
        self.resources.append(("executable", handle, storage))
        return handle, ordinal.value

    def allocate(self, data):
        size = len(data)
        handle = _Handle()
        self._check(
            self._allocate(
                self.stream, size, _MEMORY_TYPE, _BUFFER_USAGE, ctypes.byref(handle)
            ),
            "hrx_buffer_allocate",
        )
        pointer = ctypes.c_void_p()
        try:
            self._check(
                self._map(handle, _MAPPING_MODE, 3, 0, size, ctypes.byref(pointer)),
                "hrx_buffer_map_with_mode",
            )
        except Exception:
            self._buffer_release(handle)
            raise
        storage = (ctypes.c_uint8 * size).from_address(pointer.value)
        storage[:] = data
        self._check(self._flush(handle, 0, size), "hrx_buffer_flush_range")
        self.resources.append(("buffer", handle, storage))
        return handle, storage

    def run(self, executable, ordinal, buffers):
        refs = (_BufferRef * len(buffers))(
            *(_BufferRef(handle, 0, len(storage)) for handle, storage in buffers)
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
        self._check(
            self._invalidate(handle, 0, len(storage)), "hrx_buffer_invalidate_range"
        )
        return bytes(storage)

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
        self._check(self._shutdown(), "hrx_gpu_shutdown")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    if (
        manifest.get("schema_version") != XADX_MANIFEST_SCHEMA_VERSION
        or manifest.get("product") != "rocm-npu-fastflowlm-xadx"
        or manifest.get("target") != "npu2"
    ):
        raise XadxError("unsupported FastFlowLM XADX package manifest")
    inventory = manifest.get("files")
    if not isinstance(inventory, list):
        raise XadxError("XADX manifest has no file inventory")
    expected = {item["path"]: item for item in inventory}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(expected) != set(actual):
        raise XadxError("XADX package inventory mismatch")
    for relative, path in actual.items():
        record = expected[relative]
        if path.stat().st_size != record["size"] or _sha256(path) != record["sha256"]:
            raise XadxError(f"XADX package corruption: {relative}")
    return manifest


def _bf16(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _compare(actual, expected, comparison, atol, rtol):
    if len(actual) != len(expected):
        return f"size mismatch: {len(actual)} != {len(expected)}"
    if comparison == "exact":
        if actual == expected:
            return None
        return f"exact mismatch at byte {next(i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])}"
    if comparison != "bf16-tolerance" or len(actual) % 2:
        raise XadxError(f"unsupported comparison contract: {comparison}")
    mismatches = 0
    first = None
    for index, (left, right) in enumerate(
        zip(struct.iter_unpack("<H", actual), struct.iter_unpack("<H", expected))
    ):
        got, wanted = _bf16(left[0]), _bf16(right[0])
        if not math.isfinite(got) or abs(got - wanted) > atol + rtol * abs(wanted):
            mismatches += 1
            first = first or (index, got, wanted)
    return None if not mismatches else f"{mismatches} BF16 mismatches; first {first}"


def run_package(
    root,
    *,
    runtime_factory=_Runtime,
    lifecycle_rounds=XADX_LIFECYCLE_ROUNDS,
    resident_rounds=XADX_RESIDENT_ROUNDS,
    readback_rounds=XADX_READBACK_ROUNDS,
    target_family=None,
):
    """Runs bounded lifecycle, resident, and host-readback/coherence rounds."""
    if min(lifecycle_rounds, resident_rounds, readback_rounds) < 0:
        raise ValueError("round counts must be nonnegative")
    if max(lifecycle_rounds, resident_rounds, readback_rounds) > 1000:
        raise ValueError("round counts exceed the safety bound (1000)")
    root = Path(root)
    manifest = verify_package(root)
    runtime = runtime_factory()
    try:
        architecture = runtime.architecture()
    except Exception:
        runtime.close()
        raise
    architecture_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", architecture.casefold())
        if token
    }
    architecture_matches = bool(
        architecture_tokens.intersection({"npu2", "xdna2"})
    ) or (
        target_family == "npu2"
        and architecture_tokens.intersection({"amdxdna"})
    )
    if not architecture_matches:
        runtime.close()
        return {
            "status": "unsupported",
            "reason": (
                "FastFlowLM XADX assets require a positively identified "
                "NPU2/XDNA2 architecture; "
                f"observed {architecture!r} with target family {target_family!r}"
            ),
            "architecture": architecture,
            "case_names": [case["name"] for case in manifest["cases"]],
        }
    try:
        generator_module = _generator_module()
    except Exception:
        runtime.close()
        raise
    failures = []
    materialized = {}

    def materialize(specification, expected_size):
        key = (
            json.dumps(specification, sort_keys=True)
            if isinstance(specification, dict)
            else str(specification)
        )
        if key not in materialized:
            try:
                data, metadata = generator_module.materialize(specification, root)
            except (OSError, ValueError) as error:
                raise XadxError(f"cannot materialize XADX buffer: {error}") from error
            if len(data) != expected_size:
                raise XadxError(
                    f"materialized buffer size mismatch: {len(data)} != {expected_size}"
                )
            materialized[key] = (data, metadata)
        return materialized[key][0]

    def prepare(case):
        executable, ordinal = runtime.load(
            (root / case["xadx"]).read_bytes(), case["export"]
        )
        buffers = []
        expected = None
        for record in case["buffers"]:
            if record["direction"] == "output":
                data = bytes(record["size"])
                expected = (
                    record,
                    materialize(record["expected"], record["size"]),
                )
            else:
                data = materialize(record["input"], record["size"])
            buffers.append(runtime.allocate(data))
        return executable, ordinal, buffers, expected

    def dispatch(case, prepared, clear_output=True, expected_source=None):
        executable, ordinal, buffers, expected = prepared
        output_record, wanted = expected
        if expected_source is not None:
            wanted = materialize(expected_source, output_record["size"])
        if clear_output:
            buffers[0][1][:] = bytes(len(buffers[0][1]))
            runtime._check(
                runtime._flush(buffers[0][0], 0, len(buffers[0][1])), "output flush"
            )
        runtime.run(executable, ordinal, buffers)
        actual = runtime.read(buffers[0])
        return _compare(
            actual,
            wanted,
            output_record["comparison"],
            float(output_record.get("atol", 0)),
            float(output_record.get("rtol", 0)),
        )

    try:
        cases = manifest["cases"]
        for round_index in range(lifecycle_rounds):
            for case in cases:
                mark = len(runtime.resources)
                try:
                    failure = dispatch(case, prepare(case))
                    if failure:
                        failures.append(
                            f"{case['name']} lifecycle {round_index + 1}: {failure}"
                        )
                except Exception as error:
                    failures.append(
                        f"{case['name']} lifecycle {round_index + 1}: {error}"
                    )
                finally:
                    # Lifecycle resources are released by closing/reopening the runtime.
                    if len(runtime.resources) != mark:
                        runtime.close()
                        runtime = runtime_factory()
        resident = [(case, prepare(case)) for case in cases]
        for round_index in range(resident_rounds):
            for case, prepared in resident:
                failure = dispatch(case, prepared)
                if failure:
                    failures.append(
                        f"{case['name']} resident {round_index + 1}: {failure}"
                    )
        coherence = [
            (case, prepared)
            for case, prepared in resident
            if case.get("readback_variants")
        ]
        for round_index in range(readback_rounds):
            if coherence:
                case, prepared = coherence[round_index % len(coherence)]
                variants = case["readback_variants"]
                variant = variants[round_index % len(variants)]
                activation_index = next(
                    index
                    for index, record in enumerate(case["buffers"])
                    if record["name"] == variant["buffer"]
                )
                activation_record = case["buffers"][activation_index]
                activation = materialize(variant["input"], activation_record["size"])
                handle, storage = prepared[2][activation_index]
                if len(activation) != len(storage):
                    raise XadxError("readback activation size mismatch")
                storage[:] = activation
                runtime._check(
                    runtime._flush(handle, 0, len(storage)),
                    "activation flush",
                )
                failure = dispatch(
                    case,
                    prepared,
                    clear_output=False,
                    expected_source=variant["expected"],
                )
                if failure:
                    failures.append(
                        f"{case['name']} readback {round_index + 1}: {failure}"
                    )
    finally:
        runtime.close()
    return {
        "status": "pass" if not failures else "fail",
        "architecture": architecture,
        "lifecycle_rounds": lifecycle_rounds,
        "resident_rounds": resident_rounds,
        "readback_rounds": readback_rounds,
        "cases": len(manifest["cases"]),
        "case_names": [case["name"] for case in manifest["cases"]],
        "materialized_buffers": [metadata for _, metadata in materialized.values()],
        "failures": failures,
    }
