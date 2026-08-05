# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the platform-neutral ROCm NPU qualification plugin wheel."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from build_tools.fastflowlm_kernel_tests import verify_xadx_assets
from build_tools.rocm_npu_wheel_common import (
    FASTFLOWLM_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION,
    QUALIFICATION_PACKAGE,
    _write_package_manifest,
    build_wheel,
    default_package_version,
    verify_wheel_checksums,
)


QUALIFICATION_WHEEL_TAG = "py3-none-any"
ROOT = Path(__file__).resolve().parents[1]


def stage_qualification_tree(
    root: Path, destination: Path, *, version: str, xadx_assets: Path
) -> Path:
    root = Path(root)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    package_root = destination / QUALIFICATION_PACKAGE
    shutil.copytree(
        root / "tools" / "rocm-npu-qualification" / "python" / QUALIFICATION_PACKAGE,
        package_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    xadx_assets = Path(xadx_assets)
    xadx_manifest = verify_xadx_assets(xadx_assets)
    if (
        len(xadx_manifest.get("executables", [])) != 4
        or len(xadx_manifest.get("cases", [])) != 5
    ):
        raise RuntimeError(
            "qualification wheel requires four production XADX executables "
            "and five stress cases"
        )
    shutil.copytree(xadx_assets, package_root / "xadx_assets")
    licenses = package_root / "licenses"
    licenses.mkdir()
    shutil.copy2(root / "LICENSE", licenses / "rocm-npu-LICENSE")
    qualification_licenses = root / "tools" / "rocm-npu-qualification" / "licenses"
    fastflowlm_licenses = root / "projects" / "FastFlowLM"
    for source_name, packaged_name in (
        ("LICENSE_RUNTIME.txt", "FastFlowLM-LICENSE_RUNTIME.txt"),
        ("TERMS.md", "FastFlowLM-TERMS.md"),
    ):
        source = qualification_licenses / packaged_name
        upstream = fastflowlm_licenses / source_name
        if upstream.is_file() and [
            line.rstrip() for line in source.read_text(encoding="utf-8").splitlines()
        ] != [
            line.rstrip() for line in upstream.read_text(encoding="utf-8").splitlines()
        ]:
            raise RuntimeError(
                f"qualification license snapshot differs from FastFlowLM: {source_name}"
            )
        shutil.copy2(source, licenses / packaged_name)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    requires = (f"{FASTFLOWLM_DISTRIBUTION} =={version}",)
    _write_package_manifest(
        package_root,
        {
            "schema_version": 1,
            "product": QUALIFICATION_DISTRIBUTION,
            "artifact": QUALIFICATION_DISTRIBUTION,
            "version": version,
            "wheel_tag": QUALIFICATION_WHEEL_TAG,
            "requires": list(requires),
            "sources": {
                "rocm-npu-staging": {
                    "revision": revision,
                }
            },
            "dependencies": {
                FASTFLOWLM_DISTRIBUTION: {
                    "version": version,
                }
            },
            "embedded_binaries": {
                "fastflowlm_xadx": {
                    "manifest": "xadx_assets/manifest.json",
                    "target": xadx_manifest["target"],
                    "serializer": xadx_manifest["serializer"],
                    "case_contract_sha256": xadx_manifest["case_contract_sha256"],
                    "executables": [
                        {
                            key: executable[key]
                            for key in (
                                "xadx",
                                "xadx_sha256",
                                "source_xclbin",
                                "source_xclbin_sha256",
                                "transaction_sha256",
                            )
                        }
                        for executable in xadx_manifest["executables"]
                    ],
                    "license_expression": (
                        "LicenseRef-FastFlowLM-Proprietary-Kernel-Terms"
                    ),
                }
            },
            "toolchain": {"python": "standard-library"},
        },
    )
    return destination


def build_qualification_wheel(
    root: Path,
    *,
    output_directory: Path,
    xadx_assets: Path,
    version: str | None = None,
) -> Path:
    root = Path(root)
    version = version or default_package_version(root)
    build_root = root / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="rocm-npu-qualification-wheel-stage-", dir=build_root
    ) as temporary:
        tree = stage_qualification_tree(
            root,
            Path(temporary),
            version=version,
            xadx_assets=xadx_assets,
        )
        wheel = build_wheel(
            tree,
            QUALIFICATION_DISTRIBUTION,
            version,
            Path(output_directory),
            summary="Installed-product model qualification for ROCm NPU",
            tag=QUALIFICATION_WHEEL_TAG,
            requires=(f"{FASTFLOWLM_DISTRIBUTION} =={version}",),
            console_scripts={
                "rocm-npu-qualify": f"{QUALIFICATION_PACKAGE}.cli:main",
            },
            root_is_purelib=True,
        )
    verify_wheel_checksums({QUALIFICATION_DISTRIBUTION: wheel})
    return wheel


def cmd_package_qualification_wheel(args):
    wheel = build_qualification_wheel(
        ROOT,
        output_directory=Path(args.dir),
        xadx_assets=Path(args.xadx_assets),
        version=args.version,
    )
    print(f"package qualification wheel: ok -> {wheel}")
    return wheel
