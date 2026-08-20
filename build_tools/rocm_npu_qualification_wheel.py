# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the platform-neutral ROCm NPU qualification plugin wheel."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from build_tools.fastflowlm_xadx_assets import verify_kernel_recipe_assets
from build_tools.package_version import default_package_version
from build_tools.release_policy import (
    numeric_package_versions,
    numeric_requires_python,
)
from build_tools.rocm_npu_wheel_common import (
    FASTFLOWLM_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION,
    QUALIFICATION_PACKAGE,
    build_wheel,
    verify_wheel_checksums,
)
from build_tools.wheel_payload import write_package_manifest as _write_package_manifest

QUALIFICATION_WHEEL_TAG = "py3-none-any"
ROOT = Path(__file__).resolve().parents[1]

# The ASR probe's audio fixture is copied from the pinned FastFlowLM submodule
# (see manifests/versions.toml) at build time rather than vendored in git.
# FastFlowLM ships this clip only in its source tree -- never in an installed
# package and never via ``flm pull`` -- so a qualification host cannot fetch it,
# but the wheel-build host (the FastFlowLM HRX jobs) already has the checkout.
# The byte hash is pinned so an upstream clip change fails the build loudly
# instead of silently drifting the paired reference transcript.
FASTFLOWLM_ASR_SAMPLE = Path("projects/FastFlowLM/src/tb_files/Recording.wav")
ASR_SAMPLE_SHA256 = (
    "c0f16ef6b0a9053d9abccb603d1aa1445c62b2b35bbbb4eaa3996422c68df97a"
)


def _stage_asr_sample(root: Path, package_root: Path) -> None:
    """Vendor the ASR fixture from the pinned FastFlowLM checkout into the wheel.

    When FastFlowLM is not checked out (e.g. presubmit builds that skip
    submodules) the fixture is omitted; the ASR serve-probe then fails closed at
    runtime rather than pretending to transcribe. Released wheels are built in
    the FastFlowLM HRX jobs, where the submodule is always present, so they
    always carry the clip.
    """
    upstream = root / FASTFLOWLM_ASR_SAMPLE
    if not upstream.is_file():
        print(
            "package-qualification-wheel: FastFlowLM checkout absent; omitting "
            "assets/asr_sample.wav (ASR probe will fail closed at runtime)",
            file=sys.stderr,
        )
        return
    digest = hashlib.sha256(upstream.read_bytes()).hexdigest()
    if digest != ASR_SAMPLE_SHA256:
        raise RuntimeError(
            "FastFlowLM ASR fixture Recording.wav does not match the pinned "
            f"hash: expected {ASR_SAMPLE_SHA256}, got {digest}. If the upstream "
            "clip changed intentionally, update ASR_SAMPLE_SHA256 and the "
            "assets/asr_sample.reference.txt transcript together."
        )
    shutil.copy2(upstream, package_root / "assets" / "asr_sample.wav")

# The package computes its own qualification buffers and BF16 oracles rather
# than shipping them. Keep these exact: qualification evidence must mean the
# same arithmetic in an offline install as it did in CI.
NUMERIC_RUNTIME = numeric_package_versions()
QUALIFICATION_REQUIRES_PYTHON = numeric_requires_python()
QUALIFICATION_REQUIRES = tuple(
    f"{distribution} =={version}"
    for distribution, version in sorted(NUMERIC_RUNTIME.items())
)


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
    xadx_manifest, recipe = verify_kernel_recipe_assets(xadx_assets)
    capabilities = json.loads(
        (xadx_assets / "capability-manifest.json").read_text(encoding="utf-8")
    )
    if (
        {entry["id"] for entry in recipe.get("implementations", [])}
        != set(capabilities["implementations"]["entries"])
        or {entry["id"] for entry in recipe.get("coverage_limits", [])}
        != (
            set(capabilities["implementations"]["unreached"])
            | {
                id
                for id, facts in capabilities["implementations"]["entries"].items()
                if facts.get("numeric_limit") is not None
            }
        )
    ):
        raise RuntimeError(
            "qualification recipe coverage differs from its capability manifest"
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
    _stage_asr_sample(root, package_root)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    requires = (f"{FASTFLOWLM_DISTRIBUTION} =={version}", *QUALIFICATION_REQUIRES)
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
                    "recipe": "xadx_assets/recipe.json",
                    "recipe_sha256": xadx_manifest["recipe"]["sha256"],
                    "executables": [
                        {
                            "xadx": execution.get("xadx"),
                            "xadx_sha256": execution.get("xadx_sha256"),
                            "xclbin_path": implementation.get("xclbin_path"),
                            "xclbin_sha256": implementation.get(
                                "xclbin_sha256"
                            ),
                            "transaction": execution.get("transaction"),
                            "transaction_sha256": execution.get(
                                "transaction_sha256"
                            ),
                            **(
                                {}
                                if execution_id is None
                                else {"execution": execution_id}
                            ),
                        }
                        for implementation in recipe["implementations"]
                        for execution_id, execution in [
                            (None, implementation),
                            *[
                                (variant["id"], variant)
                                for variant in implementation.get(
                                    "execution_variants", ()
                                )
                            ],
                        ]
                    ],
                    "license_expression": (
                        "LicenseRef-FastFlowLM-Proprietary-Kernel-Terms"
                    ),
                }
            },
            "toolchain": {
                "python": "cpython",
                **NUMERIC_RUNTIME,
            },
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
            requires=(f"{FASTFLOWLM_DISTRIBUTION} =={version}", *QUALIFICATION_REQUIRES),
            console_scripts={
                "rocm-npu-qualify": f"{QUALIFICATION_PACKAGE}.cli:main",
            },
            requires_python=QUALIFICATION_REQUIRES_PYTHON,
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
