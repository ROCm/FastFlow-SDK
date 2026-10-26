# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the ``fastflow-sdk`` meta wheel.

The wheel carries no code. It pins the FastFlow SDK products to one release so
an install is:

    pip install fastflow-sdk==<version>

which pulls ``fastflow-sdk-fastflowlm``, ``fastflow-sdk-compiler``, and
``fastflow-sdk-ops`` at the same version.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from build_tools import compiler_product, paths
from build_tools.artifact import git_source_state, reset_directory
from build_tools.fastflow_sdk_wheel_common import (
    COMPILER_DISTRIBUTION,
    FASTFLOWLM_DISTRIBUTION,
    OPS_DISTRIBUTION,
    SDK_DISTRIBUTION,
    SDK_PACKAGE,
    build_wheel,
    remove_distribution_wheels,
    verify_wheel_checksums,
)
from build_tools.package_version import default_package_version
from build_tools.release_policy import compiler_requires_python
from build_tools.wheel_payload import write_file, write_package_manifest

SDK_WHEEL_TAG = "py3-none-any"
SDK_SUMMARY = "FastFlow SDK"
SDK_WHEEL_ARTIFACTS = compiler_product.COMPILER_ARTIFACTS
SDK_PRODUCTS = (FASTFLOWLM_DISTRIBUTION, COMPILER_DISTRIBUTION, OPS_DISTRIBUTION)


def sdk_requires(version: str) -> tuple[str, ...]:
    return tuple(f"{distribution} =={version}" for distribution in SDK_PRODUCTS)


def stage_sdk_tree(destination: Path, *, version: str) -> Path:
    destination = Path(destination)
    reset_directory(destination, destination.parent)
    package_root = destination / SDK_PACKAGE
    write_file(package_root / "__init__.py", "")
    licenses = package_root / "licenses"
    licenses.mkdir(parents=True)
    shutil.copy2(paths.ROOT / "LICENSE", licenses / "fastflow-sdk-LICENSE")
    write_package_manifest(
        package_root,
        {
            "schema_version": 1,
            "product": SDK_DISTRIBUTION,
            "artifact": SDK_DISTRIBUTION,
            "version": version,
            "wheel_tag": SDK_WHEEL_TAG,
            "requires": list(sdk_requires(version)),
            "sources": {"fastflow-sdk": git_source_state(paths.ROOT)},
            "dependencies": {
                distribution: {"version": version} for distribution in SDK_PRODUCTS
            },
        },
    )
    return destination


def build_sdk_wheel(*, output_directory: Path, version: str | None = None) -> Path:
    version = version or default_package_version(paths.ROOT)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    remove_distribution_wheels(output_directory, (SDK_DISTRIBUTION,))
    paths.BUILD.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="fastflow-sdk-meta-wheel-stage-", dir=paths.BUILD
    ) as temporary:
        tree = stage_sdk_tree(Path(temporary) / "tree", version=version)
        wheel = build_wheel(
            tree,
            SDK_DISTRIBUTION,
            version,
            output_directory,
            summary=SDK_SUMMARY,
            tag=SDK_WHEEL_TAG,
            requires=sdk_requires(version),
            requires_python=compiler_requires_python(),
            root_is_purelib=True,
        )
    verify_wheel_checksums({SDK_DISTRIBUTION: wheel})
    return wheel


def cmd_package_sdk_wheel(args):
    try:
        wheel = build_sdk_wheel(output_directory=Path(args.dir), version=args.version)
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"package sdk wheel: ok -> {wheel}")
