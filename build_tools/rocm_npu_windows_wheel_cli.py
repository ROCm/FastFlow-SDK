# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build, verify, and exercise Windows ROCm NPU runtime wheels."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_tools.artifact import validate_directory_target  # noqa: E402
from build_tools.fastflowlm_artifact import (  # noqa: E402
    stage_runtime_bundle,
    verify_fastflowlm_fresh_installs,
)
from build_tools.rocm_npu_wheel_common import (  # noqa: E402
    PACKAGE_VERSION_ENV,
    default_package_version,
    verify_wheel_checksums,
)
from build_tools.rocm_npu_windows_wheels import (  # noqa: E402
    FASTFLOWLM_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
    build_runtime_fastflowlm_wheels,
    find_rocm_npu_wheels,
    verify_wheels_do_not_have_text_path_references,
)
from build_tools.windows_npu import probe_runtime  # noqa: E402

PROJECTS = ROOT / "projects"
BUILD = ROOT / "build"
HRX_INSTALL = BUILD / "hrx-install"
FASTFLOWLM_INSTALL = BUILD / "FastFlowLM-install"
FASTFLOWLM_MODEL_CACHE = BUILD / "FastFlowLM-models"
FASTFLOWLM_BUNDLE_STAGE = BUILD / "FastFlowLM-bundle"
FASTFLOWLM_ARTIFACTS = BUILD / "artifacts"
FRESH_BUILD_MARKER = ".rns-fresh-build.json"
WHEEL_STAGE = BUILD / "rocm-npu-windows-wheel-stage"
WHEEL_VENV = BUILD / "rocm-npu-windows-wheel-venv"


def _verify_fresh_installs() -> None:
    verify_fastflowlm_fresh_installs(
        hrx_install=HRX_INSTALL,
        hrx_source=PROJECTS / "hrx",
        fastflowlm_install=FASTFLOWLM_INSTALL,
        fastflowlm_source=PROJECTS / "FastFlowLM",
        marker_name=FRESH_BUILD_MARKER,
    )


def package_wheels(version: str | None) -> None:
    _verify_fresh_installs()
    version = version or default_package_version(ROOT)
    bundle = stage_runtime_bundle(
        ROOT,
        FASTFLOWLM_INSTALL,
        HRX_INSTALL,
        FASTFLOWLM_BUNDLE_STAGE,
    )
    wheels = build_runtime_fastflowlm_wheels(
        bundle, WHEEL_STAGE, FASTFLOWLM_ARTIFACTS, version
    )
    for product in (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION):
        print(f"{product}: {wheels[product]}")


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe"


def verify_wheels(directory: Path, venv_dir: Path) -> None:
    wheels = find_rocm_npu_wheels(directory)
    verify_wheel_checksums(wheels)
    verify_wheels_do_not_have_text_path_references(wheels, (ROOT,))
    validate_directory_target(venv_dir, BUILD)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    python = _venv_python(venv_dir)
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links",
            directory,
            wheels[FASTFLOWLM_DISTRIBUTION],
        ],
        check=True,
    )
    subprocess.run([python, "-m", "pip", "check"], check=True)
    smoke = (
        "import sysconfig; from pathlib import Path; "
        "import rocm_npu_runtime as r, rocm_npu_fastflowlm as f; "
        "print('runtime', r.self_test(load=True)); "
        "print('flm', f.flm_path()); "
        "script = Path(sysconfig.get_path('scripts')) / 'flm.exe'; "
        "assert script.is_file(), script; "
        "print('flm script', script); "
        "print('config', f.config_path()); "
        "print('xclbins', f.xclbin_path())"
    )
    subprocess.run([python, "-c", smoke], check=True)
    print(f"verify Windows wheels: ok -> {venv_dir}")


def probe_runtime_package() -> int:
    import rocm_npu_runtime

    return probe_runtime(rocm_npu_runtime.libhrx_path())


def smoke_fastflowlm(iterations: int) -> int:
    import rocm_npu_fastflowlm
    import rocm_npu_runtime

    env = os.environ.copy()
    env["FLM_BIN"] = str(Path(sysconfig.get_path("scripts")) / "flm.exe")
    env["HRX_LIBHRX"] = str(rocm_npu_runtime.libhrx_path())
    env["FLM_CONFIG_PATH"] = str(rocm_npu_fastflowlm.config_path())
    env["FLM_XCLBIN_PATH"] = str(rocm_npu_fastflowlm.xclbin_path())
    env.setdefault("FLM_MODEL_PATH", str(FASTFLOWLM_MODEL_CACHE))
    result = subprocess.run(
        [
            sys.executable,
            ROOT / "tests" / "fastflowlm" / "run_fastflowlm_hrx_smoke.py",
            "--inference-iterations",
            str(iterations),
        ],
        env=env,
    )
    return result.returncode


def main() -> int:
    if os.name != "nt":
        raise RuntimeError("Windows ROCm NPU wheel commands require a Windows host")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    package = subparsers.add_parser("package")
    package.add_argument(
        "--version",
        help=f"PEP 440 package version (or ${PACKAGE_VERSION_ENV})",
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--dir", type=Path, default=FASTFLOWLM_ARTIFACTS)
    verify.add_argument("--venv", type=Path, default=WHEEL_VENV)
    subparsers.add_parser("probe")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--inference-iterations", type=int, default=1)
    args = parser.parse_args()

    if args.command == "package":
        package_wheels(args.version)
    elif args.command == "verify":
        verify_wheels(args.dir, args.venv)
    elif args.command == "probe":
        return probe_runtime_package()
    elif args.command == "smoke":
        return smoke_fastflowlm(args.inference_iterations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
