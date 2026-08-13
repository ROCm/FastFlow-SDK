# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Platform-neutral contracts for PE and ELF native inspection."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path


def is_pe(path: Path) -> bool:
    try:
        with Path(path).open("rb") as stream:
            return stream.read(2) == b"MZ"
    except OSError:
        return False


def pe_imports(path: Path) -> list[str]:
    """Returns normal and delay-load PE imports using an in-process parser."""
    path = Path(path)
    data = path.read_bytes()

    def require(offset: int, size: int, description: str) -> None:
        if offset < 0 or size < 0 or offset + size > len(data):
            raise RuntimeError(f"truncated PE {description}: {path}")

    if len(data) < 0x40 or data[:2] != b"MZ":
        raise RuntimeError(f"not a PE binary: {path}")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    require(pe_offset, 24, "header")
    if data[pe_offset:pe_offset + 4] != b"PE\0\0":
        raise RuntimeError(f"invalid PE header: {path}")
    coff = pe_offset + 4
    section_count = struct.unpack_from("<H", data, coff + 2)[0]
    optional_size = struct.unpack_from("<H", data, coff + 16)[0]
    optional = coff + 20
    require(optional, optional_size, "optional header")
    require(optional, 2, "optional-header magic")
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic == 0x20B:
        directory_offset = optional + 112
        number_of_directories_offset = optional + 108
        require(optional + 24, 8, "image base")
        image_base = struct.unpack_from("<Q", data, optional + 24)[0]
    elif magic == 0x10B:
        directory_offset = optional + 96
        number_of_directories_offset = optional + 92
        require(optional + 28, 4, "image base")
        image_base = struct.unpack_from("<I", data, optional + 28)[0]
    else:
        raise RuntimeError(f"unsupported PE optional-header magic: {path}")
    if number_of_directories_offset + 4 > optional + optional_size:
        raise RuntimeError(f"truncated PE data-directory count: {path}")
    directory_count = struct.unpack_from("<I", data, number_of_directories_offset)[0]

    section_offset = optional + optional_size
    sections = []
    for index in range(section_count):
        entry = section_offset + index * 40
        require(entry, 40, "section table")
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, entry + 8
        )
        if raw_size:
            require(raw_offset, raw_size, "section data")
        sections.append(
            (virtual_address, max(virtual_size, raw_size), raw_offset, raw_size)
        )

    def rva_offset(rva: int, size: int = 1) -> int:
        for virtual_address, virtual_span, raw_offset, raw_size in sections:
            if virtual_address <= rva and rva + size <= virtual_address + virtual_span:
                delta = rva - virtual_address
                if delta + size > raw_size:
                    break
                result = raw_offset + delta
                require(result, size, f"RVA 0x{rva:x}")
                return result
        raise RuntimeError(f"PE RVA 0x{rva:x} is not backed by file data: {path}")

    def c_string(rva: int) -> str:
        offset = rva_offset(rva)
        end = data.find(b"\0", offset)
        if end < 0:
            raise RuntimeError(f"unterminated PE import name: {path}")
        try:
            return data[offset:end].decode("ascii").lower()
        except UnicodeDecodeError as error:
            raise RuntimeError(f"non-ASCII PE import name: {path}") from error

    imports = set()
    directories = ((1, 20, 12, False), (13, 32, 4, True))
    for directory_index, descriptor_size, name_field, is_delay in directories:
        if directory_index >= directory_count:
            continue
        entry = directory_offset + directory_index * 8
        if entry + 8 > optional + optional_size:
            raise RuntimeError(f"truncated PE data directory {directory_index}: {path}")
        table_rva, table_size = struct.unpack_from("<II", data, entry)
        if not table_rva and not table_size:
            continue
        if not table_rva or table_size < descriptor_size:
            raise RuntimeError(f"malformed PE import directory {directory_index}: {path}")
        offset = rva_offset(table_rva, table_size)
        limit = offset + table_size
        terminated = False
        while offset + descriptor_size <= limit:
            descriptor = data[offset:offset + descriptor_size]
            if not any(descriptor):
                terminated = True
                break
            name_rva = struct.unpack_from("<I", descriptor, name_field)[0]
            if not name_rva:
                raise RuntimeError(f"PE import descriptor has no name: {path}")
            if is_delay and not (struct.unpack_from("<I", descriptor, 0)[0] & 1):
                if name_rva < image_base:
                    raise RuntimeError(f"invalid PE delay-import VA: {path}")
                name_rva -= image_base
            imports.add(c_string(name_rva))
            offset += descriptor_size
        if not terminated:
            raise RuntimeError(f"unterminated PE import directory {directory_index}: {path}")
    return sorted(imports)


def is_elf(path: Path) -> bool:
    try:
        with Path(path).open("rb") as stream:
            return stream.read(4) == b"\x7fELF"
    except OSError:
        return False


def is_loadable_elf(path: Path) -> bool:
    """Identifies ELF executables/shared objects while excluding ET_REL objects."""
    try:
        with Path(path).open("rb") as stream:
            header = stream.read(18)
    except OSError:
        return False
    if len(header) < 18 or header[:4] != b"\x7fELF":
        return False
    byteorder = {1: "little", 2: "big"}.get(header[5])
    if byteorder is None:
        return False
    return int.from_bytes(header[16:18], byteorder) in (2, 3)


def require_patchelf() -> str:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required for ELF inspection")
    return patchelf


def elf_needed(path: Path, *, patchelf: str | None = None) -> list[str]:
    tool = patchelf or require_patchelf()
    return subprocess.run(
        [tool, "--print-needed", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def elf_rpath(path: Path, *, patchelf: str | None = None) -> str:
    tool = patchelf or require_patchelf()
    return subprocess.run(
        [tool, "--print-rpath", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def elf_soname(path: Path, *, patchelf: str | None = None) -> str | None:
    tool = patchelf or require_patchelf()
    result = subprocess.run(
        [tool, "--print-soname", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def set_elf_rpath(path: Path, rpath: str, *, patchelf: str | None = None) -> None:
    tool = patchelf or require_patchelf()
    subprocess.run([tool, "--set-rpath", rpath, str(path)], check=True)
