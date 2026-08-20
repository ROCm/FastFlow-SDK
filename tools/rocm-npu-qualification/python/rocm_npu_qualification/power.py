# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host power-management guards for uninterrupted qualification runs."""

from __future__ import annotations

import ctypes
import os
from collections.abc import Iterator
from contextlib import contextmanager


_ES_SYSTEM_REQUIRED = 0x00000001
_ES_CONTINUOUS = 0x80000000


def _set_thread_execution_state(flags: int) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint]
    function.restype = ctypes.c_uint
    return int(function(flags))


@contextmanager
def inhibit_system_sleep() -> Iterator[None]:
    """Keeps Windows awake while hardware qualification is in progress."""
    if os.name != "nt":
        yield
        return

    if not _set_thread_execution_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED):
        error = ctypes.get_last_error()
        raise OSError(error, "failed to inhibit Windows system sleep")
    try:
        yield
    except BaseException as error:
        if not _set_thread_execution_state(_ES_CONTINUOUS):
            error.add_note("failed to restore Windows execution state")
        raise
    if not _set_thread_execution_state(_ES_CONTINUOUS):
        error = ctypes.get_last_error()
        raise OSError(error, "failed to restore Windows execution state")
