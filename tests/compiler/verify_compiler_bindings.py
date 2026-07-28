# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Runtime conformance checks for an installed compiler bindings wheel."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


def _extension_module_name(aie_root: Path, extension: Path) -> str:
    relative = extension.relative_to(aie_root)
    if extension.name.endswith(".pyd"):
        basename = extension.name.removesuffix(".pyd").removesuffix(".abi3")
    elif extension.name.endswith(".abi3.so"):
        basename = extension.name.removesuffix(".abi3.so")
    else:
        raise RuntimeError(f"not an abi3 extension: {extension}")
    return ".".join(("aie", *relative.parent.parts, basename))


def verify(expected_python: str) -> dict[str, object]:
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        raise RuntimeError(
            f"bindings test expected Python {expected_python}, got {actual_python}"
        )

    runtime_dependencies = {}
    for distribution, module in (
        ("aiofiles", "aiofiles"),
        ("cloudpickle", "cloudpickle"),
        ("ml-dtypes", "ml_dtypes"),
        ("numpy", "numpy"),
        ("rich", "rich"),
    ):
        importlib.import_module(module)
        runtime_dependencies[distribution] = importlib.metadata.version(distribution)

    import aie
    import aie.dialects.aie
    import aie.dialects.aiex
    import aie.dialects.func
    import aie.dialects.transform.interpreter
    import aie.extras.ast.util
    import aie.extras.context
    import aie.iron
    from aie.ir import Context, Module
    from aie.passmanager import PassManager

    aie_root = Path(aie.__file__).resolve().parent
    extensions = sorted(
        path
        for path in (aie_root / "_mlir_libs").rglob("*")
        if path.is_file()
        and path.name.endswith((".pyd", ".abi3.so"))
    )
    if not extensions:
        raise RuntimeError(
            f"installed aie package contains no abi3 extensions: {aie_root}"
        )

    imported_extensions = []
    for extension in extensions:
        module_name = _extension_module_name(aie_root, extension)
        importlib.import_module(module_name)
        imported_extensions.append(module_name)

    with Context() as context:
        context.allow_unregistered_dialects = False
        module = Module.parse("module {}")
        pipeline = PassManager.parse("builtin.module(canonicalize)")
        pipeline.run(module.operation)
        if "module" not in str(module):
            raise RuntimeError("MLIR module round trip produced unexpected output")

    return {
        "python": actual_python,
        "aie_root": str(aie_root),
        "extensions": imported_extensions,
        "runtime_dependencies": runtime_dependencies,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-python", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.expected_python), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
