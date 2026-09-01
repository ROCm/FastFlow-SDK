# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Validated loaders for repository release and numeric-runtime policy."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_POLICY_PATH = ROOT / "release.toml"
NUMERIC_RUNTIME_CONTRACT_PATH = (
    ROOT
    / "tools"
    / "rocm-npu-qualification"
    / "python"
    / "rocm_npu_qualification"
    / "numeric-runtime.json"
)
COMPILER_STABLE_ABI_MINOR = "3.12"
_PACKAGE_VERSION_RE = re.compile(
    r"^[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+|\.post[0-9]+|\.dev[0-9]+)?"
    r"(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?$"
)


def _wheel_release(path: Path) -> dict[str, object]:
    path = Path(path)
    with path.open("rb") as policy_file:
        policy = tomllib.load(policy_file)
    if policy.get("schema_version") != 1:
        raise RuntimeError(f"unsupported release policy schema: {path}")
    release = policy.get("wheel_release")
    if not isinstance(release, dict):
        raise RuntimeError(  # noqa: TRY004
            f"release policy wheel_release table is missing: {path}"
        )
    return release


def release_base_version(policy_path: Path = RELEASE_POLICY_PATH) -> str:
    """Loads the stable version targeted by the current development train."""
    base_version = _wheel_release(policy_path).get("base_version")
    if (
        not isinstance(base_version, str)
        or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", base_version)
        or not _PACKAGE_VERSION_RE.fullmatch(base_version)
    ):
        raise RuntimeError(
            "wheel_release.base_version must be a stable numeric version: "
            f"{policy_path}"
        )
    return base_version


def compiler_python_versions(
    policy_path: Path = RELEASE_POLICY_PATH,
) -> tuple[str, ...]:
    """Loads and validates the supported compiler CPython minor versions."""
    versions = _wheel_release(policy_path).get("compiler_python_versions")
    if not isinstance(versions, list) or not versions:
        raise RuntimeError(
            "wheel_release.compiler_python_versions must be a non-empty list: "
            f"{policy_path}"
        )
    result = tuple(str(version) for version in versions)
    if any(not re.fullmatch(r"3\.[0-9]+", version) for version in result):
        invalid = next(
            version for version in result if not re.fullmatch(r"3\.[0-9]+", version)
        )
        raise RuntimeError(f"invalid compiler Python minor in {policy_path}: {invalid}")
    if len(set(result)) != len(result):
        raise RuntimeError(f"duplicate compiler Python minor in {policy_path}")
    return result


def compiler_python_stable_abi(
    policy_path: Path = RELEASE_POLICY_PATH,
) -> str:
    """Loads and validates the compiler wheel's pinned stable ABI floor."""
    stable_abi = _wheel_release(policy_path).get("compiler_python_stable_abi")
    if (
        not isinstance(stable_abi, str)
        or not re.fullmatch(r"3\.[0-9]+", stable_abi)
        or stable_abi != COMPILER_STABLE_ABI_MINOR
        or stable_abi not in compiler_python_versions(policy_path)
    ):
        raise RuntimeError(
            "wheel_release.compiler_python_stable_abi must match the pinned "
            f"MLIR stable ABI floor {COMPILER_STABLE_ABI_MINOR}: {policy_path}"
        )
    return stable_abi


def compiler_requires_python(
    policy_path: Path = RELEASE_POLICY_PATH,
) -> str:
    """Derives Requires-Python from the validated compiler minor range."""
    versions = compiler_python_versions(policy_path)
    parsed = [tuple(map(int, version.split("."))) for version in versions]
    expected = [(parsed[0][0], parsed[0][1] + offset) for offset in range(len(parsed))]
    if parsed != expected or versions[0] != compiler_python_stable_abi(policy_path):
        raise RuntimeError(
            "wheel_release.compiler_python_versions must be a contiguous, "
            f"ascending range beginning at {COMPILER_STABLE_ABI_MINOR}: {policy_path}"
        )
    return f">={versions[0]},<{parsed[-1][0]}.{parsed[-1][1] + 1}"


def numeric_runtime_contract(
    path: Path = NUMERIC_RUNTIME_CONTRACT_PATH,
) -> dict[str, object]:
    """Loads and validates the qualification numeric-runtime JSON contract."""
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid numeric runtime contract: {path}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise RuntimeError(f"invalid numeric runtime contract: {path}")
    packages = document.get("packages")
    python_abis = document.get("python_abis")
    requires_python = document.get("requires_python")
    requires_match = (
        re.fullmatch(r">=3\.([0-9]+),<3\.([0-9]+)", requires_python)
        if isinstance(requires_python, str)
        else None
    )
    abi_minors = (
        [int(abi[3:]) for abi in python_abis]
        if isinstance(python_abis, list)
        and all(
            isinstance(abi, str) and re.fullmatch(r"cp3[0-9]{2}", abi)
            for abi in python_abis
        )
        else []
    )
    if (
        not isinstance(packages, dict)
        or not packages
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not _PACKAGE_VERSION_RE.fullmatch(version)
            for name, version in packages.items()
        )
        or not isinstance(python_abis, list)
        or not python_abis
        or not abi_minors
        or len(set(python_abis)) != len(python_abis)
        or abi_minors != list(range(abi_minors[0], abi_minors[-1] + 1))
        or requires_match is None
        or (int(requires_match[1]), int(requires_match[2]))
        != (abi_minors[0], abi_minors[-1] + 1)
    ):
        raise RuntimeError(f"invalid numeric runtime contract: {path}")
    return {
        "schema_version": 1,
        "requires_python": requires_python,
        "python_abis": list(python_abis),
        "packages": dict(packages),
    }


def numeric_package_versions(
    path: Path = NUMERIC_RUNTIME_CONTRACT_PATH,
) -> dict[str, str]:
    return dict(numeric_runtime_contract(path)["packages"])


def numeric_python_abis(
    path: Path = NUMERIC_RUNTIME_CONTRACT_PATH,
) -> tuple[str, ...]:
    return tuple(numeric_runtime_contract(path)["python_abis"])


def numeric_requires_python(path: Path = NUMERIC_RUNTIME_CONTRACT_PATH) -> str:
    return str(numeric_runtime_contract(path)["requires_python"])
