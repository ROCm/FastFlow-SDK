# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import struct
from unittest.mock import ANY

import pytest

from build_tools import aiecc_elf_xadx

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]


def _make_test_elf(
    duplicate_pdi=None,
    *,
    pdi_numbers=(1,),
    relocated_pdi_number=1,
    extra_control=False,
    scratchpad_in_extra_control=False,
):
    control_code = struct.pack("<6BHII", 0, 1, 4, 6, 8, 1, 0, 4, 32)
    control_code += b"\x05\0\0\0" * 4
    pdi_names = [f".pdi.{number}" for number in pdi_numbers]
    dynstr = b"\0" + b"".join(name.encode() + b"\0" for name in pdi_names)
    dynstr += b"scratch-pad-ctrl\0"
    names = {
        **{name: dynstr.index(name.encode()) for name in pdi_names},
        "scratch-pad-ctrl": dynstr.index(b"scratch-pad-ctrl"),
    }
    symbol_names = ("", *pdi_names, "scratch-pad-ctrl")
    scratchpad_section_index = 5 + len(pdi_names) if scratchpad_in_extra_control else 1
    dynsym = b"".join(
        struct.pack(
            "<IIIBBH",
            names.get(name, 0),
            0,
            0,
            0,
            0,
            scratchpad_section_index if name == "scratch-pad-ctrl" else int(bool(name)),
        )
        for name in symbol_names
    )
    pdi_symbol_index = symbol_names.index(f".pdi.{relocated_pdi_number}")
    scratchpad_symbol_index = symbol_names.index("scratch-pad-ctrl")
    rela = b"".join(
        (
            struct.pack("<IIi", 16, (pdi_symbol_index << 8) | 8, -4),
            struct.pack("<IIi", 24, (scratchpad_symbol_index << 8) | 8, 4),
        )
    )
    sections = [
        (".ctrltext.sequence", control_code),
        *(
            (name, f"PDI{number}".encode())
            for number, name in zip(pdi_numbers, pdi_names, strict=True)
        ),
        (".dynstr", dynstr),
        (".dynsym", dynsym),
        (".rela.dyn", rela),
    ]
    if extra_control:
        sections.append((".ctrltext.trailing-helper", b"NOT THE SEQUENCE"))
    if duplicate_pdi is not None:
        sections.insert(2, (".pdi.1", duplicate_pdi))
    shstr = b"\0" + b"".join(
        name.encode() + b"\0" for name in [*(name for name, _ in sections), ".shstrtab"]
    )
    sections.append((".shstrtab", shstr))

    data = bytearray(52)
    data[:6] = b"\x7fELF\x01\x01"
    section_headers = [(0,) * 10]
    shstr_offsets = {name: shstr.index(name.encode()) for name, _ in sections}
    for name, contents in sections:
        while len(data) % 4:
            data.append(0)
        offset = len(data)
        data.extend(contents)
        section_headers.append(
            (shstr_offsets[name], 1, 0, 0, offset, len(contents), 0, 0, 1, 0)
        )
    while len(data) % 4:
        data.append(0)
    section_offset = len(data)
    for header in section_headers:
        data.extend(struct.pack("<IIIIIIIIII", *header))
    struct.pack_into("<I", data, 32, section_offset)
    struct.pack_into("<H", data, 40, 52)
    struct.pack_into("<HHH", data, 46, 40, len(section_headers), len(sections))
    return bytes(data)


def test_extracts_pdi_and_control_parameter_relocations():
    control_code, pdis, pdi_relocations, parameter_relocations = (
        aiecc_elf_xadx.parse_elf32(_make_test_elf())
    )
    assert len(control_code) == 32
    assert pdis == [b"PDI1"]
    assert pdi_relocations == [(16, 0, -4)]
    assert parameter_relocations == [(24, 4)]


def test_accepts_only_identical_duplicate_pdi_sections():
    assert aiecc_elf_xadx.parse_elf32(_make_test_elf(b"PDI1"))[1] == [b"PDI1"]
    with pytest.raises(ValueError, match="conflicting duplicate section"):
        aiecc_elf_xadx.parse_elf32(_make_test_elf(b"OTHER"))


def test_relocations_select_control_section_and_sparse_pdi_number():
    control_code, pdis, pdi_relocations, _ = aiecc_elf_xadx.parse_elf32(
        _make_test_elf(pdi_numbers=(1, 3), relocated_pdi_number=3, extra_control=True)
    )
    assert len(control_code) == 32
    assert pdis == [b"PDI1", b"PDI3"]
    assert pdi_relocations == [(16, 1, -4)]


def test_pdi_relocation_selects_control_section_over_other_scratchpad():
    control_code, _, pdi_relocations, parameter_relocations = (
        aiecc_elf_xadx.parse_elf32(
            _make_test_elf(
                extra_control=True,
                scratchpad_in_extra_control=True,
            )
        )
    )
    assert len(control_code) == 32
    assert pdi_relocations == [(16, 0, -4)]
    assert parameter_relocations == []


def test_rejects_truncated_section_table():
    elf = bytearray(_make_test_elf())
    struct.pack_into("<I", elf, 32, len(elf) - 4)
    with pytest.raises(ValueError, match="section table"):
        aiecc_elf_xadx.parse_elf32(bytes(elf))


def test_parses_typed_parameter_layout():
    assert aiecc_elf_xadx.parse_params(
        "2\nvector_size 0 i32 core\ninput_offset 3 bf16 addr\n"
    ) == [
        ("vector_size", "i32", 0, 0, 4, 0),
        ("input_offset", "bf16", 4, 3, 2, 1),
    ]


def test_parses_parameter_ranges_and_joint_bounds():
    assert aiecc_elf_xadx.parse_params("0\n0\n") == []
    assert aiecc_elf_xadx.parse_params(
        "1\ninput_offset 3 i32 addr 0 1024\n1\ninput_offset vector_size 1 4096\n"
    ) == [("input_offset", "i32", 0, 3, 4, 1)]


def test_rejects_invalid_parameter_metadata():
    with pytest.raises(ValueError, match="count is 2"):
        aiecc_elf_xadx.parse_params("2\nvector_size 0 i32 core\n")
    with pytest.raises(ValueError, match="outside"):
        aiecc_elf_xadx.parse_params("1\ninput_offset 32 i32 addr\n")


def test_conversion_forwards_all_dynamic_metadata():
    class Serializer:
        def __init__(self):
            self.arguments = None

        def serialize_xadx(self, *args, **kwargs):
            self.arguments = (args, kwargs)
            return b"XADX"

    serializer = Serializer()
    result = aiecc_elf_xadx.convert_to_xadx(
        serializer,
        _make_test_elf(),
        b"context-xclbin",
        "1\ninput_offset 3 i32 addr\n",
    )

    assert result == b"XADX"
    args, kwargs = serializer.arguments
    assert args == (b"context-xclbin", ANY, "MLIR_AIE")
    assert kwargs["pdis"] == [b"PDI1"]
    assert kwargs["run_pdi_relocations"] == [[(16, 0, -4)]]
    assert kwargs["run_control_parameter_relocations"] == [[(24, 4)]]
    assert kwargs["control_parameters"] == [("input_offset", "i32", 0, 3, 4, 1)]
