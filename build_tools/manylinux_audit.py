# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Structured auditwheel analysis shared by Linux wheel products."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

AUDITWHEEL_API_VERSION = "6.7.0"


def _auditwheel_policy_diagnostics(wheel_info, target_policy_name: str) -> str:
    """Formats the structured reasons a wheel misses its claimed policy."""
    target = wheel_info.policies.get_policy_by_name(target_policy_name)
    components = (
        ("external references", wheel_info.ref_policy),
        ("versioned symbols", wheel_info.sym_policy),
        ("UCS-2 symbols", wheel_info.ucs_policy),
        ("PyFPE symbols", wheel_info.pyfpe_policy),
        ("blacklisted symbols", wheel_info.blacklist_policy),
        ("machine ISA", wheel_info.machine_policy),
    )
    lines = [
        f"auditwheel policy diagnostics for {target_policy_name}:",
        f"  overall: {wheel_info.overall_policy.name}",
    ]
    for label, policy in components:
        marker = "LIMITING" if policy < target else "ok"
        lines.append(f"  {label}: {policy.name} ({marker})")

    target_refs = wheel_info.external_refs[target.name]
    if target_refs.libs:
        lines.append("  external libraries:")
        lines.extend(
            f"    {name}: {path if path is not None else '<unresolved>'}"
            for name, path in sorted(target_refs.libs.items())
        )
    if target_refs.blacklist:
        lines.append("  blacklisted symbol references:")
        lines.extend(
            f"    {name}: {', '.join(sorted(symbols))}"
            for name, symbols in sorted(target_refs.blacklist.items())
        )

    if wheel_info.sym_policy < target:
        allowed = {
            f"{prefix}_{version}"
            for prefix, versions in target.symbol_versions.items()
            for version in versions
        }
        required = {
            symbol
            for symbols in wheel_info.versioned_symbols.values()
            for symbol in symbols
        }
        unsupported = sorted(required - allowed)
        if unsupported:
            lines.append("  unsupported versioned symbols:")
            lines.extend(f"    {symbol}" for symbol in unsupported)
    return "\n".join(lines)


def auditwheel_policy(
    wheel: Path,
    *,
    exclude: frozenset[str] = frozenset(),
) -> str:
    """Returns auditwheel's structured overall policy for a wheel."""
    try:
        installed_version = distribution_version("auditwheel")
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"auditwheel {AUDITWHEEL_API_VERSION} is required to audit Linux wheels"
        ) from error
    if installed_version != AUDITWHEEL_API_VERSION:
        raise RuntimeError(
            "unsupported auditwheel structured API version: "
            f"{installed_version}; expected {AUDITWHEEL_API_VERSION}"
        )
    from auditwheel.error import NonPlatformWheelError, WheelToolsError
    from auditwheel.wheel_abi import analyze_wheel_abi
    from auditwheel.wheeltools import get_wheel_architecture, get_wheel_libc

    try:
        architecture = get_wheel_architecture(wheel.name)
    except (WheelToolsError, NonPlatformWheelError):
        architecture = None
    try:
        libc = get_wheel_libc(wheel.name)
    except WheelToolsError:
        libc = None
    wheel_info = analyze_wheel_abi(
        libc,
        architecture,
        wheel,
        exclude,
        disable_isa_ext_check=False,
        allow_graft=False,
    )
    claimed_policy = re.search(r"manylinux_\d+_\d+_x86_64", wheel.name)
    if claimed_policy:
        target = wheel_info.policies.get_policy_by_name(claimed_policy.group())
        if wheel_info.overall_policy < target:
            print(_auditwheel_policy_diagnostics(wheel_info, target.name))
    return wheel_info.overall_policy.name


def validate_manylinux_policy(
    policy: str, *, maximum_glibc: tuple[int, int]
) -> str:
    """Requires an x86-64 manylinux policy no newer than the given glibc."""
    manylinux = re.fullmatch(r"manylinux_(\d+)_(\d+)_x86_64", policy)
    if not manylinux or tuple(map(int, manylinux.groups())) > maximum_glibc:
        maximum = "_".join(map(str, maximum_glibc))
        raise RuntimeError(
            f"wheel is not manylinux_{maximum}_x86_64 compatible: {policy}"
        )
    return policy
