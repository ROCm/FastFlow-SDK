# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Small protocol fakes shared by HRX runtime tests."""

from __future__ import annotations

from collections.abc import Callable


class FakeExecutable:
    def __init__(
        self,
        *,
        export_ordinal: int,
        name: str | None = None,
        on_lookup: Callable[[str], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ):
        self.export_ordinal = export_ordinal
        self.name = name
        self.on_lookup = on_lookup
        self.on_close = on_close
        self.closed = False

    def lookup_export(self, name: str) -> int:
        if self.on_lookup is not None:
            self.on_lookup(name)
        return self.export_ordinal

    def close(self) -> None:
        self.closed = True
        if self.on_close is not None:
            self.on_close()


class FakeBuffer:
    def __init__(
        self,
        shape,
        dtype,
        *,
        zeros: bool = False,
        on_flush: Callable[[], None] | None = None,
        on_invalidate: Callable[[], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ):
        import numpy as np

        factory = np.zeros if zeros else np.empty
        self.array = factory(shape, dtype=dtype)
        self.on_flush = on_flush
        self.on_invalidate = on_invalidate
        self.on_close = on_close
        self.flush_count = 0
        self.invalidate_count = 0
        self.closed = False

    def flush(self) -> None:
        self.flush_count += 1
        if self.on_flush is not None:
            self.on_flush()

    def invalidate(self) -> None:
        self.invalidate_count += 1
        if self.on_invalidate is not None:
            self.on_invalidate()

    def close(self) -> None:
        self.closed = True
        if self.on_close is not None:
            self.on_close()
