# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""FastFlowLM bundle and wheel packaging orchestration."""

import shutil
import subprocess
import sys
from pathlib import Path

from build_tools import environment, paths
from build_tools.artifact import (
    sha256_file,
    validate_directory_target,
    verify_sha256_sidecar,
)
from build_tools.fastflowlm_artifact import (
    create_archive,
    extract_archive,
    stage_runtime_bundle,
    validate_source_policy,
    verify_bundle,
    verify_fastflowlm_fresh_installs,
)
from build_tools.fastflowlm_deps import require_fastflowlm_linux_packaging_tools
from build_tools.manylinux_audit import auditwheel_policy
from build_tools.package_version import default_package_version
from build_tools.rocm_npu_linux_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_LINUX_WHEEL_TAG,
    LINUX_WHEEL_TAG,
    RUNTIME_DISTRIBUTION,
    build_runtime_fastflowlm_wheels,
    find_rocm_npu_wheels,
    validate_auditwheel_policy,
    verify_wheels_do_not_have_text_path_references,
)
from build_tools.rocm_npu_wheel_common import verify_wheel_checksums

ROOT = paths.ROOT
PROJ = paths.PROJ
BUILD = paths.BUILD
HRX_INSTALL = paths.HRX_INSTALL
FASTFLOWLM_INSTALL = paths.FASTFLOWLM_INSTALL
FASTFLOWLM_BUNDLE_STAGE = paths.BUILD / "FastFlowLM-bundle"
FASTFLOWLM_ARTIFACTS = paths.BUILD / "artifacts"
FASTFLOWLM_EXTRACTED = paths.BUILD / "FastFlowLM-extracted"
FASTFLOWLM_WHEEL_STAGE = paths.BUILD / "rocm-npu-wheel-stage"
FASTFLOWLM_WHEEL_VENV = paths.BUILD / "rocm-npu-wheel-venv"
FRESH_BUILD_MARKER = ".rns-fresh-build.json"
IS_WINDOWS = paths.is_windows()
_EXE = ".exe" if IS_WINDOWS else ""
sh = environment.sh

def _verify_fastflowlm_fresh_installs():
    try:
        return verify_fastflowlm_fresh_installs(
            hrx_install=HRX_INSTALL,
            hrx_source=PROJ / "hrx",
            fastflowlm_install=FASTFLOWLM_INSTALL,
            fastflowlm_source=PROJ / "FastFlowLM",
            marker_name=FRESH_BUILD_MARKER,
        )
    except RuntimeError as error:
        sys.exit(str(error))


def cmd_package_fastflowlm(args):
    """Create and independently re-extract the relocatable FLM + HRX runtime."""
    require_fastflowlm_linux_packaging_tools(archive=True)
    _verify_fastflowlm_fresh_installs()
    bundle = stage_runtime_bundle(ROOT, FASTFLOWLM_INSTALL, HRX_INSTALL,
                                  FASTFLOWLM_BUNDLE_STAGE)
    archive = create_archive(bundle, FASTFLOWLM_ARTIFACTS)
    extracted = extract_archive(
        archive, FASTFLOWLM_EXTRACTED, allowed_root=BUILD)
    print("package fastflowlm: ok")
    print("  archive:  ", archive)
    print("  extracted:", extracted)


def cmd_package_fastflowlm_wheels(args):
    """Build split rocm-npu-runtime and rocm-npu-fastflowlm Linux wheels."""
    if IS_WINDOWS:
        sys.exit("rocm-npu runtime/FastFlowLM wheel packaging is Linux-only for now")
    require_fastflowlm_linux_packaging_tools()
    _verify_fastflowlm_fresh_installs()
    version = args.version or default_package_version(ROOT)
    bundle = stage_runtime_bundle(ROOT, FASTFLOWLM_INSTALL, HRX_INSTALL,
                                  FASTFLOWLM_BUNDLE_STAGE)
    wheels = build_runtime_fastflowlm_wheels(
        bundle, FASTFLOWLM_WHEEL_STAGE, FASTFLOWLM_ARTIFACTS, version)
    print("package fastflowlm wheels: ok")
    for product in (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION):
        print(f"  {product}: {wheels[product]}")


def _python_executable_tool(name):
    tool = shutil.which(name)
    if tool:
        return tool
    candidate = Path(sys.executable).resolve().parent / (f"{name}.exe" if IS_WINDOWS else name)
    return str(candidate) if candidate.is_file() else None


def cmd_audit_fastflowlm_wheels(args):
    if IS_WINDOWS:
        sys.exit("audit-fastflowlm-wheels is Linux-only")
    wheels = find_rocm_npu_wheels(Path(args.dir))
    verify_wheel_checksums(wheels)
    expected_tags = {
        RUNTIME_DISTRIBUTION: LINUX_WHEEL_TAG,
        FASTFLOWLM_DISTRIBUTION: FASTFLOWLM_LINUX_WHEEL_TAG,
    }
    for product in (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION):
        print(f"== auditwheel analyze: {wheels[product].name} ==")
        if not wheels[product].name.endswith(f"-{expected_tags[product]}.whl"):
            raise RuntimeError(
                f"{product} wheel does not use expected tag {expected_tags[product]}: "
                f"{wheels[product].name}"
            )
        policy = auditwheel_policy(wheels[product])
        print(f"auditwheel overall policy: {policy}")
        validate_auditwheel_policy(product, policy)


def _venv_python(venv_dir):
    return Path(venv_dir) / ("Scripts" if IS_WINDOWS else "bin") / f"python{_EXE}"


def cmd_verify_fastflowlm_wheels(args):
    wheel_directory = Path(args.dir)
    wheels = find_rocm_npu_wheels(wheel_directory)
    verify_wheel_checksums(wheels)
    verify_wheels_do_not_have_text_path_references(wheels, (ROOT,))

    venv_dir = Path(args.venv)
    validate_directory_target(venv_dir, BUILD)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    sh([sys.executable, "-m", "venv", venv_dir])
    py = _venv_python(venv_dir)
    sh(
        [
            py,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links",
            wheel_directory,
            wheels[FASTFLOWLM_DISTRIBUTION],
        ]
    )
    sh([py, "-m", "pip", "check"])
    smoke = (
        "import os, sysconfig; "
        "from pathlib import Path; "
        "import rocm_npu_runtime as r, rocm_npu_fastflowlm as f; "
        "print('runtime', r.self_test(load=True)); "
        "print('flm', f.flm_path()); "
        "script = Path(sysconfig.get_path('scripts')) / ('flm.exe' if os.name == 'nt' else 'flm'); "
        "print('flm script', script); "
        "assert script.is_file(), script; "
        "print('config', f.config_path()); "
        "print('xclbins', f.xclbin_path())"
    )
    sh([py, "-c", smoke])
    print(f"verify fastflowlm wheels: ok -> {venv_dir}")


def _find_fastflowlm_archive(directory):
    directory = Path(directory)
    archives = sorted([*directory.glob("fastflowlm-hrx-*.zip"),
                       *directory.glob("fastflowlm-hrx-*.tar.zst")])
    if len(archives) != 1:
        sys.exit(f"expected exactly one FastFlowLM artifact in {directory}, found {len(archives)}")
    archive = archives[0]
    try:
        verify_sha256_sidecar(archive)
    except RuntimeError as error:
        sys.exit(str(error))
    return archive


def cmd_unpack_fastflowlm(args):
    archive = _find_fastflowlm_archive(args.dir)
    bundle = extract_archive(
        archive, FASTFLOWLM_EXTRACTED, allowed_root=BUILD)
    print("unpack fastflowlm: ok ->", bundle)


def cmd_verify_fastflowlm(args):
    location = Path(args.dir)
    if not location.is_dir():
        sys.exit(f"extracted FastFlowLM bundle directory not found: {location}")
    candidates = ([location] if (location / "manifest.json").is_file() else
                  [path for path in location.iterdir()
                   if path.is_dir() and (path / "manifest.json").is_file()])
    if len(candidates) != 1:
        sys.exit(f"expected exactly one extracted FastFlowLM bundle in {location}")
    manifest = verify_bundle(candidates[0])
    try:
        validate_source_policy(manifest, ROOT)
    except RuntimeError as error:
        sys.exit(str(error))
    current_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    source_revision = (manifest.get("sources", {})
                       .get("rocm-npu-staging", {}).get("revision"))
    if source_revision != current_revision:
        sys.exit("FastFlowLM artifact source revision does not match this checkout")
    for record in manifest.get("patches", []):
        patch_path = ROOT / record["path"]
        if (not patch_path.is_file() or patch_path.stat().st_size != record["size"] or
                sha256_file(patch_path) != record["sha256"]):
            sys.exit(f"FastFlowLM artifact patch identity mismatch: {record['path']}")
    print(f"verify fastflowlm: ok ({len(manifest['files'])} files) -> {candidates[0]}")
