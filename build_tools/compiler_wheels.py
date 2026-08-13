# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""ROCm NPU compiler wheel packaging and verification orchestration."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from build_tools import compiler_product, environment, paths
from build_tools.artifact import git_source_state, validate_directory_target
from build_tools.manylinux_audit import auditwheel_policy
from build_tools.package_version import default_package_version
from build_tools.rocm_npu_compiler_wheels import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    build_compiler_wheel_family,
    compiler_core_shared_library_names,
    compiler_product_report,
    validate_compiler_auditwheel_policy,
    verify_compiler_binding_stable_abi,
    verify_compiler_wheel_path_references,
    verify_linux_compiler_wheels,
    verify_windows_compiler_wheels,
)
from build_tools.rocm_npu_wheel_common import (
    COMPILER_PACKAGE,
    verify_wheel_checksums,
)

ROOT = paths.ROOT
BUILD = paths.BUILD
IRON_TOOLS_INSTALL = paths.IRON_TOOLS_INSTALL
COMPILER_WHEEL_ARTIFACTS = compiler_product.COMPILER_ARTIFACTS
COMPILER_WHEEL_STAGE = paths.BUILD / "rocm-npu-compiler-wheel-stage"
COMPILER_WHEEL_VENV = paths.BUILD / "rocm-npu-compiler-wheel-venv"
IS_WINDOWS = paths.is_windows()
_VBIN = "Scripts" if IS_WINDOWS else "bin"
_EXE = ".exe" if IS_WINDOWS else ""
cmake_bin = environment.cmake_bin

def _compiler_binding_root(value):
    root = Path(value) if value else IRON_TOOLS_INSTALL / "python"
    if not (root / "aie").is_dir():
        raise RuntimeError(f"compiler binding aie package missing: {root}")
    return root


def _compiler_wheel_files(directory):
    directory = Path(directory)
    prefixes = tuple(
        distribution.replace("-", "_") + "-"
        for distribution in (
            COMPILER_DISTRIBUTION,
            COMPILER_CORE_DISTRIBUTION,
            COMPILER_BINDINGS_DISTRIBUTION,
        )
    )
    wheels = sorted(
        path for path in directory.glob("*.whl")
        if path.name.startswith(prefixes)
    )
    counts = {
        distribution: len([
            wheel for wheel in wheels
            if wheel.name.startswith(distribution.replace("-", "_") + "-")
        ])
        for distribution in (
            COMPILER_DISTRIBUTION,
            COMPILER_CORE_DISTRIBUTION,
            COMPILER_BINDINGS_DISTRIBUTION,
        )
    }
    if counts[COMPILER_DISTRIBUTION] != 1 or counts[COMPILER_CORE_DISTRIBUTION] != 1:
        raise RuntimeError(
            f"expected one compiler meta and core wheel in {directory}: {counts}"
        )
    if counts[COMPILER_BINDINGS_DISTRIBUTION] != 1:
        raise RuntimeError(
            f"expected one compiler bindings wheel in {directory}: {counts}"
        )
    return wheels


def verify_compiler_wheel_source_revision(wheels, expected_revision):
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            manifests = [
                name
                for name in archive.namelist()
                if name.count("/") == 1 and name.endswith("/manifest.json")
            ]
            if len(manifests) != 1:
                raise RuntimeError(
                    f"compiler wheel must contain one package manifest: {wheel}"
                )
            manifest = json.loads(archive.read(manifests[0]))
        revision = (
            manifest.get("sources", {})
            .get("rocm-npu-staging", {})
            .get("revision")
        )
        if revision != expected_revision:
            raise RuntimeError(
                "compiler wheel source revision does not match checkout: "
                f"{wheel}: embedded={revision!r}, expected={expected_revision!r}"
            )


def cmd_package_compiler_wheels(args):
    version = args.version or default_package_version(ROOT)
    try:
        if args.product:
            product = compiler_product.resolve_product_root(args.product)
            compiler_product.verify_product_inventory(product)
        else:
            product, _ = compiler_product.stage_product()
        binding_root = _compiler_binding_root(args.binding)
        platform_tag = "win_amd64" if IS_WINDOWS else "manylinux_2_28_x86_64"
        report = compiler_product_report(product)
        report_path = (
            COMPILER_WHEEL_ARTIFACTS
            / "rocm-npu-compiler-wheel-report.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        wheels = build_compiler_wheel_family(
            product,
            binding_root,
            COMPILER_WHEEL_STAGE,
            COMPILER_WHEEL_ARTIFACTS,
            version=version,
            platform_tag=platform_tag,
            include_meta=not args.no_meta,
        )
        verify_compiler_binding_stable_abi(
            wheels[COMPILER_BINDINGS_DISTRIBUTION]
        )
    except RuntimeError as error:
        sys.exit(str(error))
    for name, wheel in wheels.items():
        print(f"{name}: {wheel}")
    print("compiler wheel report:", report_path)


def cmd_audit_compiler_wheels(args):
    if IS_WINDOWS:
        sys.exit("audit-compiler-wheels is Linux-only")
    wheels = _compiler_wheel_files(args.dir)
    verify_linux_compiler_wheels(wheels)
    core_wheels = [
        wheel for wheel in wheels
        if wheel.name.startswith(
            COMPILER_CORE_DISTRIBUTION.replace("-", "_") + "-"
        )
    ]
    if len(core_wheels) != 1:
        sys.exit("audit-compiler-wheels requires exactly one compiler core wheel")
    core_libraries = compiler_core_shared_library_names(core_wheels[0])
    for wheel in wheels:
        if wheel.name.startswith(COMPILER_DISTRIBUTION.replace("-", "_") + "-"):
            continue
        distribution = (
            COMPILER_CORE_DISTRIBUTION
            if wheel.name.startswith(
                COMPILER_CORE_DISTRIBUTION.replace("-", "_") + "-"
            )
            else COMPILER_BINDINGS_DISTRIBUTION
        )
        policy = auditwheel_policy(
            wheel,
            exclude=(
                core_libraries
                if distribution == COMPILER_BINDINGS_DISTRIBUTION
                else frozenset()
            ),
        )
        validate_compiler_auditwheel_policy(distribution, policy)
        if "manylinux_2_28_x86_64" in wheel.name and not policy.startswith(
            "manylinux_"
        ):
            raise RuntimeError(
                f"{wheel.name} claims manylinux_2_28 but auditwheel reports {policy}"
            )
        print(f"{wheel.name}: {policy}")


def _compiler_installed_prefix(python):
    result = subprocess.run(
        [
            python,
            "-c",
            "import rocm_npu_compiler as c; print(c.compiler_prefix())",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _verify_installed_compiler_bindings(python, env=None):
    expected_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    subprocess.run(
        [
            python,
            ROOT / "tests" / "compiler" / "verify_compiler_bindings.py",
            "--expected-python",
            expected_python,
        ],
        check=True,
        env=env,
    )


def _compiler_e2e_env(prefix, env=None):
    result = dict(os.environ if env is None else env)
    result["ROCM_NPU_COMPILER_PREFIX"] = str(prefix)
    result["ROCM_NPU_CMAKE"] = cmake_bin()
    return result


def _hardlink_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _compiler_site_packages(python):
    result = subprocess.run(
        [
            python,
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _export_compiler_consumer_root(python, destination):
    """Hard-links the installed compiler packages for cross-minor ABI checks."""
    destination = Path(destination)
    validate_directory_target(destination, BUILD)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    site_packages = _compiler_site_packages(python)
    names = {
        "aie",
        COMPILER_DISTRIBUTION.replace("-", "_"),
        COMPILER_CORE_DISTRIBUTION.replace("-", "_"),
        COMPILER_BINDINGS_DISTRIBUTION.replace("-", "_"),
    }
    sources = [
        path for path in site_packages.iterdir()
        if path.name in names
        or any(path.name.startswith(name + "-") and path.name.endswith(".dist-info")
               for name in names)
    ]
    if not any(path.name == "aie" for path in sources):
        raise RuntimeError(f"installed compiler bindings missing from {site_packages}")
    for source in sources:
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                copy_function=_hardlink_or_copy,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        else:
            _hardlink_or_copy(source, target)
    return destination


def _verify_compiler_wheel_consumer(args):
    if not args.consumer_root:
        raise RuntimeError("--consumer-only requires --consumer-root")
    consumer_root = Path(args.consumer_root)
    if not (consumer_root / "aie").is_dir():
        raise RuntimeError(
            f"shared compiler consumer root is missing or incomplete: {consumer_root}"
        )
    venv_dir = Path(args.venv)
    validate_directory_target(venv_dir, BUILD)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    python = venv_dir / _VBIN / f"python{_EXE}"
    environment.python_install(python, *COMPILER_BINDINGS_RUNTIME_REQUIREMENTS)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(consumer_root.resolve()), env.get("PYTHONPATH", "")]
    )
    _verify_installed_compiler_bindings(python, env=env)
    subprocess.run(
        [
            python,
            "-c",
            "import aie.dialects.transform.interpreter; "
            "import rocm_npu_compiler as c; "
            "print(c.compiler_prefix()); print(c.native_library_dir())",
        ],
        check=True,
        env=env,
    )
    if getattr(args, "consumer_e2e", False):
        prefix = consumer_root / COMPILER_PACKAGE
        subprocess.run(
            [
                python,
                ROOT / "tests" / "e2e" / "run_e2e.py",
                "--compiler-prefix",
                prefix,
            ],
            check=True,
            env=_compiler_e2e_env(prefix, env),
        )
    print(f"verify compiler wheel consumer: ok -> {venv_dir}")


def cmd_verify_compiler_wheels(args):
    if getattr(args, "consumer_only", False):
        try:
            _verify_compiler_wheel_consumer(args)
        except RuntimeError as error:
            sys.exit(str(error))
        return
    wheels = _compiler_wheel_files(args.dir)
    version = args.version
    if not version:
        meta = next(
            wheel for wheel in wheels
            if wheel.name.startswith(COMPILER_DISTRIBUTION.replace("-", "_") + "-")
        )
        version = meta.name[
            len(COMPILER_DISTRIBUTION.replace("-", "_")) + 1:
        ].split("-py3-none-any.whl", 1)[0]
    verify_wheel_checksums({
        f"compiler-{index}": wheel for index, wheel in enumerate(wheels)
    })
    verify_compiler_wheel_source_revision(
        wheels, git_source_state(ROOT)["revision"]
    )
    verify_compiler_wheel_path_references(wheels, (ROOT, BUILD))
    if IS_WINDOWS:
        graph = verify_windows_compiler_wheels(wheels)
    else:
        graph = verify_linux_compiler_wheels(wheels)
    venv_dir = Path(args.venv)
    validate_directory_target(venv_dir, BUILD)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    python = venv_dir / _VBIN / f"python{_EXE}"
    environment.python_install(
        python,
        "--find-links",
        str(Path(args.dir).resolve()),
        f"{COMPILER_DISTRIBUTION}=={version}",
        refresh_packages=(
            COMPILER_DISTRIBUTION,
            COMPILER_CORE_DISTRIBUTION,
            COMPILER_BINDINGS_DISTRIBUTION,
        ),
    )
    subprocess.run([python, "-m", "pip", "check"], check=True)
    _verify_installed_compiler_bindings(python)
    subprocess.run(
        [
            python,
            "-c",
            "import aie.dialects.transform.interpreter; "
            "import rocm_npu_compiler as c; "
            "print(c.compiler_prefix()); print(c.native_library_dir())",
        ],
        check=True,
    )
    scripts = venv_dir / _VBIN
    for tool, argument in (("aie-opt", "--version"), ("aiecc", "--help")):
        subprocess.run([scripts / f"{tool}{_EXE}", argument], check=True)
    prefix = _compiler_installed_prefix(python)
    subprocess.run(
        [
            python,
            ROOT / "tests" / "e2e" / "run_e2e.py",
            "--compiler-prefix",
            prefix,
        ],
        check=True,
        env=_compiler_e2e_env(prefix),
    )
    if getattr(args, "consumer_root", None):
        exported = _export_compiler_consumer_root(python, args.consumer_root)
        print("compiler wheel consumer root:", exported)
    print(
        f"verify compiler wheels: ok ({len(graph)} native dependency records) "
        f"-> {venv_dir}"
    )
