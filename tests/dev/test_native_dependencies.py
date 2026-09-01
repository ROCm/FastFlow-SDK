# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import shutil
import struct
import sys
from pathlib import Path

import pytest

from build_tools.native_dependencies import (
    elf_needed,
    elf_rpath,
    is_elf,
    is_loadable_elf,
    is_pe,
    pe_imports,
)

pytestmark = pytest.mark.presubmit


def _write_pe(path: Path, *, malformed_delay_va: bool = False) -> Path:
    data = bytearray(0x600)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    coff = 0x84
    struct.pack_into("<H", data, coff, 0x8664)
    struct.pack_into("<H", data, coff + 2, 1)
    struct.pack_into("<H", data, coff + 16, 240)
    optional = coff + 20
    struct.pack_into("<H", data, optional, 0x20B)
    image_base = 0x140000000
    struct.pack_into("<Q", data, optional + 24, image_base)
    struct.pack_into("<I", data, optional + 108, 16)
    directory = optional + 112
    struct.pack_into("<II", data, directory + 1 * 8, 0x1000, 40)
    struct.pack_into("<II", data, directory + 13 * 8, 0x1040, 64)
    section = optional + 240
    data[section : section + 8] = b".rdata\0\0"
    struct.pack_into("<IIII", data, section + 8, 0x400, 0x1000, 0x400, 0x200)

    normal = 0x200
    struct.pack_into("<I", data, normal + 12, 0x1090)
    delay = 0x240
    struct.pack_into("<I", data, delay, 0 if malformed_delay_va else 1)
    delay_name = 1 if malformed_delay_va else 0x10A0
    struct.pack_into("<I", data, delay + 4, delay_name)
    data[0x290:0x29D] = b"KERNEL32.dll\0"
    data[0x2A0:0x2AA] = b"delay.dll\0"
    path.write_bytes(data)
    return path


def test_pe_identification_and_normal_plus_delay_imports(tmp_path):
    binary = _write_pe(tmp_path / "fixture.exe")
    assert is_pe(binary)
    assert pe_imports(binary) == ["delay.dll", "kernel32.dll"]


def test_pe_parser_rejects_malformed_headers(tmp_path):
    truncated = tmp_path / "truncated.exe"
    truncated.write_bytes(b"MZ" + b"\0" * 61)
    with pytest.raises(RuntimeError, match="not a PE binary"):
        pe_imports(truncated)

    bad_delay = _write_pe(tmp_path / "bad-delay.exe", malformed_delay_va=True)
    with pytest.raises(RuntimeError, match="invalid PE delay-import VA"):
        pe_imports(bad_delay)


def test_elf_identification_contract(tmp_path):
    relocatable = bytearray(18)
    relocatable[:4] = b"\x7fELF"
    relocatable[5] = 1
    relocatable[16:18] = (1).to_bytes(2, "little")
    path = tmp_path / "object.o"
    path.write_bytes(relocatable)
    assert is_elf(path)
    assert not is_loadable_elf(path)
    relocatable[16:18] = (3).to_bytes(2, "little")
    path.write_bytes(relocatable)
    assert is_loadable_elf(path)


def test_elf_needed_and_rpath_contracts_when_patchelf_is_available():
    patchelf = shutil.which("patchelf")
    executable = Path(sys.executable)
    if not patchelf or not is_loadable_elf(executable):
        pytest.skip("requires patchelf and an ELF Python executable")
    assert isinstance(elf_needed(executable, patchelf=patchelf), list)
    assert isinstance(elf_rpath(executable, patchelf=patchelf), str)
