# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Converts an internal aiecc configuration ELF into an HRX XADX artifact."""

import re
import struct


def _range(data, offset, size, description):
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise ValueError(f"{description} is outside the ELF")
    return data[offset : offset + size]


def _string(table, offset):
    if offset < 0 or offset >= len(table):
        raise ValueError("ELF string offset is out of range")
    end = table.find(b"\0", offset)
    if end < 0:
        raise ValueError("unterminated ELF string")
    return table[offset:end].decode("utf-8")


def parse_elf32(elf_data):
    """Extracts control code, PDIs, and address relocations from aiecc ELF32."""
    if len(elf_data) < 52 or elf_data[:6] != b"\x7fELF\x01\x01":
        raise ValueError("expected a little-endian ELF32 file")
    section_offset = struct.unpack_from("<I", elf_data, 32)[0]
    section_entry_size, section_count, string_table_index = struct.unpack_from(
        "<HHH", elf_data, 46
    )
    if section_entry_size != 40:
        raise ValueError(f"unsupported ELF32 section header size {section_entry_size}")
    if not section_count or string_table_index >= section_count:
        raise ValueError("ELF section string table index is out of range")
    section_table = _range(
        elf_data,
        section_offset,
        section_entry_size * section_count,
        "ELF section table",
    )
    section_headers = [
        struct.unpack_from("<IIIIIIIIII", section_table, index * 40)
        for index in range(section_count)
    ]
    string_header = section_headers[string_table_index]
    section_strings = _range(
        elf_data, string_header[4], string_header[5], "ELF section string table"
    )

    sections = {}
    control_sections = {}
    for index, header in enumerate(section_headers):
        name = _string(section_strings, header[0])
        contents = _range(elf_data, header[4], header[5], f"ELF section {name!r}")
        existing = sections.get(name)
        if existing is not None and existing != contents:
            raise ValueError(
                f"ELF has conflicting duplicate section {name!r}; "
                "relocations cannot identify which occurrence to use"
            )
        sections.setdefault(name, contents)
        if name == ".ctrltext.sequence" or name.startswith(".ctrltext."):
            control_sections[index] = (name, contents)
    if not control_sections:
        raise ValueError("ELF has no control-text section")

    try:
        pdi_sections = sorted(
            (
                (int(name.rsplit(".", 1)[1]), name)
                for name in sections
                if name.startswith(".pdi.")
            ),
            key=lambda item: item[0],
        )
    except ValueError as error:
        raise ValueError("ELF has a malformed .pdi.N section name") from error
    if not pdi_sections:
        raise ValueError("ELF has no .pdi.N sections")
    if any(number < 1 for number, _ in pdi_sections):
        raise ValueError("ELF .pdi.N section numbers must be positive")
    pdi_ordinals = {number: ordinal for ordinal, (number, _) in enumerate(pdi_sections)}
    pdis = [sections[name] for _, name in pdi_sections]

    dynstr = sections.get(".dynstr", b"")
    dynsym = sections.get(".dynsym", b"")
    if len(dynsym) % 16:
        raise ValueError("ELF .dynsym size is not a multiple of ELF32 Sym")
    symbol_names = []
    symbol_sections = []
    for offset in range(0, len(dynsym), 16):
        name_offset, section_index = struct.unpack_from("<I10xH", dynsym, offset)
        symbol_names.append(_string(dynstr, name_offset) if name_offset else "")
        symbol_sections.append(section_index)

    pdi_relocations = []
    parameter_relocations = []
    rela = sections.get(".rela.dyn", b"")
    if len(rela) % 12:
        raise ValueError("ELF .rela.dyn size is not a multiple of ELF32 Rela")
    relocations = [
        struct.unpack_from("<IIi", rela, offset) for offset in range(0, len(rela), 12)
    ]
    relocated_control_sections = {
        symbol_sections[info >> 8]
        for _, info, _ in relocations
        if (info & 0xFF) == 8
        and (info >> 8) < len(symbol_names)
        and symbol_names[info >> 8].startswith(".pdi.")
        and symbol_sections[info >> 8] in control_sections
    }
    if len(relocated_control_sections) != 1:
        raise ValueError(
            "ELF PDI address_64 relocations do not identify exactly one "
            "control-text section"
        )
    control_section_index = relocated_control_sections.pop()
    _, control_code = control_sections[control_section_index]

    for offset in range(0, len(rela), 12):
        transaction_offset, info, addend = struct.unpack_from("<IIi", rela, offset)
        if (info & 0xFF) != 8:
            continue
        symbol_index = info >> 8
        if symbol_index >= len(symbol_names):
            raise ValueError("ELF relocation has an invalid symbol index")
        if symbol_sections[symbol_index] != control_section_index:
            continue
        symbol_name = symbol_names[symbol_index]
        if symbol_name == "scratch-pad-ctrl":
            parameter_relocations.append((transaction_offset, addend))
            continue
        if not symbol_name.startswith(".pdi."):
            raise ValueError(
                f"address_64 relocation references non-PDI symbol {symbol_name!r}"
            )
        try:
            pdi_number = int(symbol_name.rsplit(".", 1)[1])
        except ValueError as error:
            raise ValueError(
                f"PDI relocation has malformed symbol {symbol_name!r}"
            ) from error
        pdi_ordinal = pdi_ordinals.get(pdi_number)
        if pdi_ordinal is None:
            raise ValueError(f"PDI relocation references missing {symbol_name}")
        pdi_relocations.append((transaction_offset, pdi_ordinal, addend))
    if not pdi_relocations:
        raise ValueError("ELF has no address_64 PDI relocations")
    return control_code, pdis, pdi_relocations, parameter_relocations


def _scalar_byte_length(scalar_type):
    match = re.fullmatch(r"i([0-9]+)", scalar_type)
    if match:
        bit_width = int(match.group(1))
    elif scalar_type in ("f16", "bf16"):
        bit_width = 16
    elif scalar_type == "f32":
        bit_width = 32
    else:
        raise ValueError(f"unsupported scratchpad scalar type {scalar_type!r}")
    if bit_width == 0 or bit_width > 32 or bit_width % 8:
        raise ValueError(
            f"scratchpad scalar type {scalar_type!r} is not a 1-4 byte scalar"
        )
    return bit_width // 8


def parse_params(params_text):
    """Parses aiecc scratchpad metadata into HRX control-parameter records."""
    lines = [line.strip() for line in params_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("scratchpad parameter metadata is empty")
    try:
        expected_count = int(lines[0])
    except ValueError as error:
        raise ValueError("scratchpad parameter count is not an integer") from error
    if expected_count != len(lines) - 1:
        raise ValueError(
            f"scratchpad parameter count is {expected_count}, "
            f"but {len(lines) - 1} records follow"
        )
    parameters = []
    constant_offset = 0
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"malformed scratchpad parameter record {line!r}")
        name, state_table_index_text, scalar_type, kind_text = fields
        try:
            state_table_index = int(state_table_index_text)
        except ValueError as error:
            raise ValueError(
                f"scratchpad parameter {name!r} has a non-integer slot"
            ) from error
        if not 0 <= state_table_index < 32:
            raise ValueError(
                f"scratchpad parameter {name!r} slot {state_table_index} "
                "is outside [0, 32)"
            )
        if kind_text not in ("core", "addr"):
            raise ValueError(
                f"scratchpad parameter {name!r} has unknown kind {kind_text!r}"
            )
        byte_length = _scalar_byte_length(scalar_type)
        parameters.append(
            (
                name,
                scalar_type,
                constant_offset,
                state_table_index,
                byte_length,
                1 if kind_text == "addr" else 0,
            )
        )
        constant_offset += byte_length
    return parameters


def convert_to_xadx(api, elf_data, context_xclbin, params_text, entry_name="MLIR_AIE"):
    """Converts build-time aiecc outputs into a self-contained XADX."""
    control_code, pdis, pdi_relocations, parameter_relocations = parse_elf32(elf_data)
    parameters = parse_params(params_text) if params_text is not None else []
    if parameter_relocations and not parameters:
        raise ValueError(
            "scratch-pad-ctrl relocation requires control-parameter metadata"
        )
    return api.serialize_xadx(
        context_xclbin,
        control_code,
        entry_name,
        pdis=pdis,
        run_pdi_relocations=[pdi_relocations],
        run_control_parameter_relocations=[parameter_relocations],
        control_parameters=parameters,
    )
