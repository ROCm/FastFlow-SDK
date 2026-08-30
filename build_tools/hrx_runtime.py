# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared HRX runtime discovery, loader setup, and probe helpers."""

import contextlib
import ctypes
import os
import re
from pathlib import Path

from build_tools import paths


HRX_CTS_TESTS = ("core_tests", "buffer_tests", "command_buffer_tests", "queue_tests")
HRX_CTS_DIR = ("runtime", "src", "iree", "hal", "drivers", "amdxdna", "cts")
# Hermetic iree_cc_test targets under amdxdna/ (host-side; not the on-device cts/ suite).
HRX_AMDXDNA_HOST_TESTS = (
    "allocator_test",
    "xclbin_util_test",
    "direct_command_buffer_chain_cache_test",
    "direct_command_buffer_single_cache_test",
    "context_cache_test",
    "context_cache_capacity_test",
    "direct_command_buffer_planning_test",
    "async_queue_test",
    "completion_queue_test",
    "command_buffer_test",
    "buffer_test",
    "direct_command_buffer_test",
    "device_test",
    "driver_test",
    "executable_test",
    "executable_xclbin_test",
    "nop_executable_cache_test",
    "semaphore_test",
)
HRX_AMDXDNA_HOST_TESTS_WINDOWS = ("native_windows_mcdm_test",)
# Nested MCDM shim tests: (CMake target, CTest name relative to amdxdna/).
HRX_AMDXDNA_MCDM_HOST_TESTS = (
    (
        "iree_hal_drivers_amdxdna_shim_windows_mcdm_kmt_api_test",
        "shim/windows/mcdm/kmt_api_test",
    ),
)


def hrx_amdxdna_host_tests():
    """Returns hermetic amdxdna iree_cc_test names for the host OS."""
    if paths.is_windows():
        return HRX_AMDXDNA_HOST_TESTS + HRX_AMDXDNA_HOST_TESTS_WINDOWS
    return HRX_AMDXDNA_HOST_TESTS


def hrx_amdxdna_host_test_targets():
    """Returns the CMake target names generated for the host tests."""
    targets = [
        f"iree_hal_drivers_amdxdna_{name}" for name in hrx_amdxdna_host_tests()
    ]
    if paths.is_windows():
        targets.extend(target for target, _ in HRX_AMDXDNA_MCDM_HOST_TESTS)
    return tuple(targets)


def hrx_amdxdna_host_test_regex():
    """Returns a ctest regex matching only the selected amdxdna host tests."""
    names = list(hrx_amdxdna_host_tests())
    if paths.is_windows():
        names.extend(relative_name for _, relative_name in HRX_AMDXDNA_MCDM_HOST_TESTS)
    escaped = "|".join(re.escape(name) for name in names)
    return rf"^iree/hal/drivers/amdxdna/({escaped})$"


def _numpy():
    """Import NumPy only for numeric buffer operations."""
    import numpy

    return numpy


def libhrx_path(bundle=None, runtime_package=False):
    if runtime_package:
        try:
            import rocm_npu_runtime

            return Path(rocm_npu_runtime.libhrx_path()).resolve()
        except (ImportError, FileNotFoundError) as error:
            print(f"  [MISS] installed rocm-npu-runtime package: {error}")
            return None
    if bundle:
        relative = "bin/hrx.dll" if paths.is_windows() else "lib/libhrx.so"
        candidate = (Path(bundle) / relative).resolve()
        return candidate if candidate.is_file() else None
    relative = "bin/hrx.dll" if paths.is_windows() else "lib/libhrx.so"
    candidate = paths.HRX_INSTALL / relative
    return candidate if candidate.is_file() else None


def add_hrx_install_env(env, install=None):
    """Expose an installed HRX package to ctypes and native dependencies."""
    selected_install = Path(install or paths.HRX_INSTALL)
    lib_dir = selected_install / ("bin" if paths.is_windows() else "lib")
    env["HRX_INSTALL"] = str(selected_install)
    env["LIBHRX_DIR"] = str(lib_dir)
    loader_var = "PATH" if paths.is_windows() else "LD_LIBRARY_PATH"
    env[loader_var] = (
        str(lib_dir) + os.pathsep + env.get(loader_var, "")
    )
    return env


def hrx_check(lib, status, what):
    """Return ``(ok, message)`` for and consume an ``hrx_status_t``."""
    if status is None:
        return True, "ok"
    message = ctypes.c_char_p()
    length = ctypes.c_size_t()
    lib.hrx_status_ignore(
        lib.hrx_status_to_string(
            status, ctypes.byref(message), ctypes.byref(length)
        )
    )
    code = lib.hrx_status_code(status)
    text = (
        message.value.decode("utf-8", "replace") if message.value else "?"
    )
    if message.value:
        lib.hrx_status_free_message(message)
    lib.hrx_status_ignore(status)
    return False, f"{what} failed (code {code}): {text}"


def bind_libhrx(lib, *, include_device_count=True):
    """Bind the HRX functions shared by probes and the full runtime API."""
    void = ctypes.c_void_p
    pointer = ctypes.POINTER
    signatures = {
        "hrx_status_code": (ctypes.c_int, [void]),
        "hrx_status_ignore": (None, [void]),
        "hrx_status_free_message": (None, [ctypes.c_char_p]),
        "hrx_status_to_string": (
            void,
            [void, pointer(ctypes.c_char_p), pointer(ctypes.c_size_t)],
        ),
        "hrx_gpu_initialize": (void, [ctypes.c_uint32]),
        "hrx_gpu_shutdown": (void, []),
    }
    if include_device_count:
        signatures["hrx_gpu_device_count"] = (void, [pointer(ctypes.c_int)])
    for name, (result, arguments) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = result, arguments
    return lib


def probe_libhrx(bundle=None, runtime_package=False):
    lib_path = libhrx_path(bundle, runtime_package)
    library_name = "hrx.dll" if paths.is_windows() else "libhrx.so"
    if not lib_path:
        print(f"  [MISS] {library_name} - run: python dev.py build hrx")
        return 1
    print(f"  [ok ] {library_name}: {lib_path}")
    with contextlib.ExitStack() as stack:
        if paths.is_windows() and hasattr(os, "add_dll_directory"):
            stack.enter_context(os.add_dll_directory(str(lib_path.parent)))
        load_args = (
            {} if paths.is_windows() else {"mode": ctypes.RTLD_GLOBAL}
        )
        try:
            library = ctypes.CDLL(str(lib_path), **load_args)
        except OSError as error:
            print(f"  [MISS] loading {library_name} failed: {error}")
            return 1
        bind_libhrx(library)
        ok, message = hrx_check(
            library,
            library.hrx_gpu_initialize(0),
            "hrx_gpu_initialize",
        )
        if not ok:
            print(f"  [MISS] {message}")
            return 1
        print("  [ok ] hrx_gpu_initialize")
        rc = 0
        count = ctypes.c_int(0)
        ok, message = hrx_check(
            library,
            library.hrx_gpu_device_count(ctypes.byref(count)),
            "hrx_gpu_device_count",
        )
        if not ok:
            print(f"  [MISS] {message}")
            rc = 1
        else:
            print(
                f"  [{'ok ' if count.value else 'MISS'}] "
                f"hrx gpu devices: {count.value}"
            )
            if count.value == 0:
                rc = 1
        library.hrx_status_ignore(library.hrx_gpu_shutdown())
        return rc


HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0 = 0
HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0 = 0
HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0 = 0
HRX_AMDXDNA_CONTEXT_MODE_CREATE = 0

HRX_MEMORY_TYPE_HOST_LOCAL = 0x00000042
HRX_MEMORY_TYPE_DEVICE_VISIBLE = 0x00000010
HRX_BUFFER_USAGE_DEFAULT = 0x00000C03
HRX_BUFFER_USAGE_MAPPING_PERSISTENT = 0x02000000
HRX_MAP_READ = 0x01
HRX_MAP_WRITE = 0x02
HRX_MAPPING_MODE_PERSISTENT = 0x00000002
HRX_DISPATCH_FLAG_NONE = 0

HRX_AMDXDNA_EXECUTABLE_TARGET_FAMILY = b"amdxdna"
HRX_AMDXDNA_EXECUTABLE_TARGET_KEY = b"amdxdna"

_Handle = ctypes.c_void_p
_Status = ctypes.c_void_p


class HrxError(RuntimeError):
    """An error returned by the HRX C API."""


class HrxHostAllocator(ctypes.Structure):
    _fields_ = [
        ("self", ctypes.c_void_p),
        ("ctl", ctypes.c_void_p),
    ]


class HrxStringView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_char_p),
        ("size", ctypes.c_size_t),
    ]


class HrxConstByteSpan(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("data_length", ctypes.c_size_t),
    ]


class HrxAmdxdnaExecutableRun(ctypes.Structure):
    _fields_ = [
        ("record_length", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("transaction", HrxConstByteSpan),
        ("data_payload", HrxConstByteSpan),
    ]


class HrxAmdxdnaExecutableEntryPoint(ctypes.Structure):
    _fields_ = [
        ("record_length", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("name", HrxStringView),
        ("context_mode", ctypes.c_uint32),
        ("xclbin_ordinal", ctypes.c_uint32),
        ("pdi_ordinal", ctypes.c_uint32),
        ("source_line", ctypes.c_uint32),
        ("source_file", HrxStringView),
        ("runs", ctypes.POINTER(HrxAmdxdnaExecutableRun)),
        ("run_count", ctypes.c_size_t),
    ]


class HrxAmdxdnaExecutableCreateParams(ctypes.Structure):
    _fields_ = [
        ("record_length", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("xclbins", ctypes.POINTER(HrxConstByteSpan)),
        ("xclbin_count", ctypes.c_size_t),
        ("entry_points", ctypes.POINTER(HrxAmdxdnaExecutableEntryPoint)),
        ("entry_point_count", ctypes.c_size_t),
    ]


class HrxDispatchConfig(ctypes.Structure):
    _fields_ = [
        ("workgroup_count", ctypes.c_uint32 * 3),
        ("workgroup_size", ctypes.c_uint32 * 3),
        ("subgroup_size", ctypes.c_uint32),
    ]


class HrxBufferRef(ctypes.Structure):
    _fields_ = [
        ("buffer", _Handle),
        ("offset", ctypes.c_size_t),
        ("length", ctypes.c_size_t),
    ]


class AmdxdnaExecutableDescription:
    """Owns all borrowed storage referenced by an amdxdna description."""

    def __init__(
        self, xclbin, transaction, entry_name, data_payload=b"", *, runs=None
    ):
        if not xclbin:
            raise ValueError("xclbin must not be empty")
        self.name_storage = entry_name.encode("utf-8")
        if not self.name_storage:
            raise ValueError("entry_name must not be empty")
        if runs is None:
            runs = [(transaction, data_payload)]
        elif transaction is not None or data_payload:
            raise ValueError("transaction/data_payload and runs are mutually exclusive")
        runs = list(runs)
        if not runs:
            raise ValueError("runs must not be empty")

        self.xclbin_storage = _byte_storage(xclbin)
        self.xclbins = (HrxConstByteSpan * 1)(
            _span(self.xclbin_storage, len(xclbin))
        )
        self.transaction_storages = []
        self.data_payload_storages = []
        self.runs = (HrxAmdxdnaExecutableRun * len(runs))()
        for index, record in enumerate(runs):
            transaction_bytes, payload_bytes = record
            if not transaction_bytes:
                raise ValueError(f"run {index} transaction must not be empty")
            transaction_storage = _byte_storage(transaction_bytes)
            payload_storage = (
                _byte_storage(payload_bytes) if payload_bytes else None
            )
            self.transaction_storages.append(transaction_storage)
            self.data_payload_storages.append(payload_storage)
            run = self.runs[index]
            run.record_length = ctypes.sizeof(run)
            run.abi_version = HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0
            run.transaction = _span(
                transaction_storage, len(transaction_bytes)
            )
            if payload_storage is not None:
                run.data_payload = _span(payload_storage, len(payload_bytes))
        # Compatibility aliases for callers and tests that inspect a single run.
        self.transaction_storage = self.transaction_storages[0]
        self.data_payload_storage = self.data_payload_storages[0]
        self.run = self.runs[0]

        self.entry_point = HrxAmdxdnaExecutableEntryPoint()
        self.entry_point.record_length = ctypes.sizeof(self.entry_point)
        self.entry_point.abi_version = (
            HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0
        )
        self.entry_point.name = HrxStringView(
            self.name_storage, len(self.name_storage)
        )
        self.entry_point.context_mode = HRX_AMDXDNA_CONTEXT_MODE_CREATE
        self.entry_point.runs = self.runs
        self.entry_point.run_count = len(self.runs)

        self.params = HrxAmdxdnaExecutableCreateParams()
        self.params.record_length = ctypes.sizeof(self.params)
        self.params.abi_version = (
            HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0
        )
        self.params.xclbins = self.xclbins
        self.params.xclbin_count = 1
        self.params.entry_points = ctypes.pointer(self.entry_point)
        self.params.entry_point_count = 1


def _byte_storage(data):
    data = bytes(data)
    return (ctypes.c_uint8 * len(data)).from_buffer_copy(data)


def _span(storage, size):
    return HrxConstByteSpan(ctypes.cast(storage, ctypes.c_void_p), size)


def library_candidates(environ=None, platform_name=None):
    """Returns HRX library candidates in discovery order."""
    environ = os.environ if environ is None else environ
    platform_name = os.name if platform_name is None else platform_name
    filename = "hrx.dll" if platform_name == "nt" else "libhrx.so"
    candidates = []

    explicit = environ.get("HRX_LIBHRX")
    if explicit:
        candidates.append(explicit)

    lib_dir = environ.get("LIBHRX_DIR")
    if lib_dir:
        candidates.append(str(Path(lib_dir) / filename))

    install = environ.get("HRX_INSTALL")
    if install:
        subdirs = ("bin", "lib") if platform_name == "nt" else ("lib", "bin")
        candidates.extend(str(Path(install) / subdir / filename) for subdir in subdirs)

    candidates.append(filename)
    return list(dict.fromkeys(candidates))


def load_libhrx(environ=None, platform_name=None):
    """Loads HRX using environment overrides followed by loader defaults."""
    candidates = library_candidates(environ, platform_name)
    errors = []
    for candidate in candidates:
        try:
            kwargs = {}
            if (os.name if platform_name is None else platform_name) != "nt":
                kwargs["mode"] = ctypes.RTLD_GLOBAL
            return ctypes.CDLL(candidate, **kwargs)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise HrxError("Could not load HRX; tried " + "; ".join(errors))


def _declare(library, name, restype, argtypes):
    function = getattr(library, name)
    function.restype = restype
    function.argtypes = argtypes
    return function


class HrxApi:
    """Bound functions from the public HRX runtime and amdxdna headers."""

    def __init__(self, library=None):
        self.library = load_libhrx() if library is None else library
        self._bind()

    def _bind(self):
        def d(name, result, args):
            return _declare(self.library, name, result, args)

        bind_libhrx(self.library, include_device_count=False)
        self.status_code = self.library.hrx_status_code
        self.status_to_string = self.library.hrx_status_to_string
        self.status_free_message = self.library.hrx_status_free_message
        self.status_ignore = self.library.hrx_status_ignore

        self.host_allocator_system = d(
            "hrx_host_allocator_system", HrxHostAllocator, []
        )
        self.host_allocator_free = d(
            "hrx_host_allocator_free",
            None,
            [HrxHostAllocator, ctypes.c_void_p],
        )
        self.amdxdna_xadx_serialize = d(
            "hrx_amdxdna_xadx_serialize",
            _Status,
            [
                ctypes.POINTER(HrxAmdxdnaExecutableCreateParams),
                HrxHostAllocator,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.POINTER(ctypes.c_size_t),
            ],
        )

        self.gpu_initialize = self.library.hrx_gpu_initialize
        self.gpu_shutdown = self.library.hrx_gpu_shutdown
        self.gpu_device_get = d(
            "hrx_gpu_device_get", _Status, [ctypes.c_int, ctypes.POINTER(_Handle)]
        )
        self.device_release = d("hrx_device_release", None, [_Handle])
        self.stream_create = d(
            "hrx_stream_create",
            _Status,
            [_Handle, ctypes.c_uint32, ctypes.POINTER(_Handle)],
        )
        self.stream_release = d("hrx_stream_release", None, [_Handle])
        self.stream_synchronize = d(
            "hrx_stream_synchronize", _Status, [_Handle]
        )

        self.buffer_allocate = d(
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
        self.buffer_map_with_mode = d(
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
        self.buffer_unmap = d("hrx_buffer_unmap", _Status, [_Handle])
        self.buffer_flush_range = d(
            "hrx_buffer_flush_range",
            _Status,
            [_Handle, ctypes.c_size_t, ctypes.c_size_t],
        )
        self.buffer_invalidate_range = d(
            "hrx_buffer_invalidate_range",
            _Status,
            [_Handle, ctypes.c_size_t, ctypes.c_size_t],
        )
        self.buffer_release = d("hrx_buffer_release", None, [_Handle])

        self.executable_load_data = d(
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
        self.executable_lookup_export_by_name = d(
            "hrx_executable_lookup_export_by_name",
            _Status,
            [_Handle, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)],
        )
        self.executable_release = d("hrx_executable_release", None, [_Handle])
        self.stream_dispatch = d(
            "hrx_stream_dispatch",
            _Status,
            [
                _Handle,
                _Handle,
                ctypes.c_uint32,
                ctypes.POINTER(HrxDispatchConfig),
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(HrxBufferRef),
                ctypes.c_size_t,
                ctypes.c_uint32,
            ],
        )

    def check(self, status, operation):
        if not status:
            return
        message = ctypes.c_char_p()
        message_length = ctypes.c_size_t()
        secondary = self.status_to_string(
            status, ctypes.byref(message), ctypes.byref(message_length)
        )
        code = self.status_code(status)
        text = (
            ctypes.string_at(message, message_length.value).decode(
                "utf-8", errors="replace"
            )
            if message
            else "unknown error"
        )
        if message:
            self.status_free_message(message)
        if secondary:
            self.status_ignore(secondary)
        self.status_ignore(status)
        raise HrxError(f"{operation} failed (status {code}): {text}")

    def serialize_xadx(
        self, xclbin, transaction=None, entry_name="MLIR_AIE", data_payload=b"",
        *, runs=None
    ):
        """Serializes one or more runs into an owned XADX byte string."""
        description = AmdxdnaExecutableDescription(
            xclbin, transaction, entry_name, data_payload, runs=runs
        )
        allocator = self.host_allocator_system()
        output = ctypes.POINTER(ctypes.c_uint8)()
        output_length = ctypes.c_size_t()
        self.check(
            self.amdxdna_xadx_serialize(
                ctypes.byref(description.params),
                allocator,
                ctypes.byref(output),
                ctypes.byref(output_length),
            ),
            "hrx_amdxdna_xadx_serialize",
        )
        try:
            return ctypes.string_at(output, output_length.value)
        finally:
            self.host_allocator_free(allocator, output)


class HrxExecutable:
    def __init__(self, runtime, handle, storage):
        self._runtime = runtime
        self.handle = handle
        self._storage = storage

    def lookup_export(self, name):
        ordinal = ctypes.c_uint32()
        self._runtime.api.check(
            self._runtime.api.executable_lookup_export_by_name(
                self.handle, name.encode("utf-8"), ctypes.byref(ordinal)
            ),
            "hrx_executable_lookup_export_by_name",
        )
        return ordinal.value

    def close(self):
        if self.handle:
            self._runtime.api.executable_release(self.handle)
            self.handle = _Handle()
            self._runtime._forget(self)


class HrxBuffer:
    def __init__(self, runtime, handle, mapped_pointer, shape, dtype):
        np = _numpy()
        self._runtime = runtime
        self.handle = handle
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        self._mapped_pointer = mapped_pointer
        self._storage = (ctypes.c_uint8 * self.nbytes).from_address(mapped_pointer)
        self.array = np.ctypeslib.as_array(self._storage).view(self.dtype).reshape(
            self.shape
        )

    def flush(self, offset=0, size=None):
        size = self.nbytes - offset if size is None else size
        self._runtime.api.check(
            self._runtime.api.buffer_flush_range(self.handle, offset, size),
            "hrx_buffer_flush_range",
        )

    def invalidate(self, offset=0, size=None):
        size = self.nbytes - offset if size is None else size
        self._runtime.api.check(
            self._runtime.api.buffer_invalidate_range(self.handle, offset, size),
            "hrx_buffer_invalidate_range",
        )

    def close(self):
        if self.handle:
            self._runtime.api.check(
                self._runtime.api.buffer_unmap(self.handle), "hrx_buffer_unmap"
            )
            self._runtime.api.buffer_release(self.handle)
            self.handle = _Handle()
            self._runtime._forget(self)


class HrxRuntime:
    """Owns one NPU device and stream for direct XADX dispatch."""

    def __init__(self, api=None, device_index=0):
        self.api = HrxApi() if api is None else api
        self.device = _Handle()
        self.stream = _Handle()
        self._resources = []
        self._initialized = False
        self.api.check(self.api.gpu_initialize(0), "hrx_gpu_initialize")
        self._initialized = True
        try:
            self.api.check(
                self.api.gpu_device_get(device_index, ctypes.byref(self.device)),
                "hrx_gpu_device_get",
            )
            self.api.check(
                self.api.stream_create(
                    self.device, 0, ctypes.byref(self.stream)
                ),
                "hrx_stream_create",
            )
        except Exception:
            self.close()
            raise

    def _forget(self, resource):
        if resource in self._resources:
            self._resources.remove(resource)

    def load_xadx(self, xadx):
        """Loads an already serialized amdxdna XADX executable."""
        storage = _byte_storage(xadx)
        executable = _Handle()
        self.api.check(
            self.api.executable_load_data(
                self.device,
                ctypes.cast(storage, ctypes.c_void_p),
                len(xadx),
                HRX_AMDXDNA_EXECUTABLE_TARGET_FAMILY,
                HRX_AMDXDNA_EXECUTABLE_TARGET_KEY,
                ctypes.byref(executable),
            ),
            "hrx_executable_load_data",
        )
        result = HrxExecutable(self, executable, storage)
        self._resources.append(result)
        return result

    def allocate_buffer(self, shape, dtype=None):
        """Allocates a persistently mapped, device-visible NumPy buffer."""
        np = _numpy()
        dtype = np.uint8 if dtype is None else dtype
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if nbytes <= 0:
            raise ValueError("buffer size must be positive")

        handle = _Handle()
        self.api.check(
            self.api.buffer_allocate(
                self.stream,
                nbytes,
                HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
                ctypes.byref(handle),
            ),
            "hrx_buffer_allocate",
        )
        pointer = ctypes.c_void_p()
        try:
            self.api.check(
                self.api.buffer_map_with_mode(
                    handle,
                    HRX_MAPPING_MODE_PERSISTENT,
                    HRX_MAP_READ | HRX_MAP_WRITE,
                    0,
                    nbytes,
                    ctypes.byref(pointer),
                ),
                "hrx_buffer_map_with_mode",
            )
        except Exception:
            self.api.buffer_release(handle)
            raise

        result = HrxBuffer(self, handle, pointer.value, shape, dtype)
        self._resources.append(result)
        return result

    def dispatch(self, executable, export_ordinal, buffers, constants=b""):
        bindings = (HrxBufferRef * len(buffers))(
            *(HrxBufferRef(buffer.handle, 0, buffer.nbytes) for buffer in buffers)
        )
        config = HrxDispatchConfig(
            (ctypes.c_uint32 * 3)(1, 1, 1),
            (ctypes.c_uint32 * 3)(1, 1, 1),
            0,
        )
        constant_storage = _byte_storage(constants) if constants else None
        constant_pointer = (
            ctypes.cast(constant_storage, ctypes.c_void_p)
            if constant_storage is not None
            else None
        )
        self.api.check(
            self.api.stream_dispatch(
                self.stream,
                executable.handle,
                export_ordinal,
                ctypes.byref(config),
                constant_pointer,
                len(constants),
                bindings,
                len(buffers),
                HRX_DISPATCH_FLAG_NONE,
            ),
            "hrx_stream_dispatch",
        )

    def synchronize(self):
        self.api.check(
            self.api.stream_synchronize(self.stream), "hrx_stream_synchronize"
        )

    def close(self):
        for resource in reversed(self._resources[:]):
            resource.close()
        if self.stream:
            self.api.stream_release(self.stream)
            self.stream = _Handle()
        if self.device:
            self.api.device_release(self.device)
            self.device = _Handle()
        if self._initialized:
            self.api.check(self.api.gpu_shutdown(), "hrx_gpu_shutdown")
            self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
