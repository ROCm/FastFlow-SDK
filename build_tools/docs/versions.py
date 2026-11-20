# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The component versions page, from ``manifests/versions.toml``."""

from __future__ import annotations

import inspect
import tomllib
from pathlib import Path

from build_tools.fastflow_sdk_wheel_common import build_wheel
from build_tools.release_policy import compiler_requires_python

VERSIONS = Path("manifests") / "versions.toml"
RELEASE = Path("release.toml")

_NAMES = {
    "llvm": "ROCm LLVM",
    "iron_tools": "iron-tools (MLIR-AIE)",
    "aiebu": "aiebu",
    "iron_ops": "IRON operator library",
    "peano": "Peano (llvm-aie)",
    "hrx": "HRX runtime",
    "fastflowlm": "FastFlowLM",
    "eudsl": "eudsl-python-extras",
}


def load(root: Path) -> tuple[dict, dict]:
    versions = tomllib.loads((root / VERSIONS).read_text(encoding="utf-8"))
    release = tomllib.loads((root / RELEASE).read_text(encoding="utf-8"))
    return versions, release


def pinned_commit(versions: dict, component: str) -> str | None:
    return (versions.get(component) or {}).get("commit")


def _pin(entry: dict) -> str:
    repo, commit = entry.get("repo"), entry.get("commit")
    if repo and commit:
        return f"[`{commit[:12]}`](https://{repo}/commit/{commit})"
    if entry.get("wheel"):
        return f"`{entry['wheel']}`"
    if entry.get("version"):
        return f"`{entry.get('package', '')}=={entry['version']}`"
    return ""


def versions_page(versions: dict, release: dict, ref: str = "main") -> str:
    rows = []
    for key, entry in versions.items():
        if key == "meta" or not isinstance(entry, dict):
            continue
        repo = entry.get("repo")
        source = f"[{repo.removeprefix('github.com/')}](https://{repo})" if repo else ""
        if not source and entry.get("index"):
            source = f"[wheel index]({entry['index']})"
        if not source and entry.get("find_links"):
            source = f"[find-links]({entry['find_links']})"
        provides = ", ".join(f"`{p}`" for p in entry.get("provides") or [])
        branch = f"`{entry['branch']}`" if entry.get("branch") else ""
        rows.append(
            f"| {_NAMES.get(key, key)} | {source} | {_pin(entry)} | {branch} | {provides} |"
        )
    wheel = release.get("wheel_release") or {}
    python = ", ".join(wheel.get("compiler_python_versions") or [])
    abi = "cp" + str(wheel.get("compiler_python_stable_abi", "")).replace(".", "")
    runtime_python = (
        inspect.signature(build_wheel).parameters["requires_python"].default
    )
    return f"""# Component versions

This site documents FastFlow SDK at `{ref}`, built from source at these pins,
recorded in
[`manifests/versions.toml`](https://github.com/ROCm/FastFlow-SDK/blob/{ref}/manifests/versions.toml),
which also explains each pin and its patches. Upstream documentation, such as
the MLIR-AIE programming guide and examples, describes its default branch; use
the pinned revision below when an example must match what the SDK ships.

| Component | Source | Pinned revision | Branch | Provides |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Python

| Wheels | Requires-Python |
| --- | --- |
| Compiler (`fastflow-sdk-compiler`, IRON operators) | `{compiler_requires_python()}`: CPython {python}, one `{abi}-abi3` build |
| Runtime and FastFlowLM | `{runtime_python}` |

The release base version is `{wheel.get("base_version", "")}`.
"""
