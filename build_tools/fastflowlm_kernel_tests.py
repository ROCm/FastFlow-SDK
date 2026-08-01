# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build, package, verify, and run the FastFlowLM production-xclbin smoke."""

import json
import math
import os
import platform
import re
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from build_tools.dev import environment, paths
from build_tools.dev.build import HRX_INSTALL
from build_tools.fastflowlm_artifact import sha256_file


BUILD = paths.BUILD / "fastflowlm-kernel-smoke"
INSTALL = paths.BUILD / "fastflowlm-kernel-smoke-install"
STAGE = paths.BUILD / "fastflowlm-kernel-smoke-stage"
EXTRACTED = paths.BUILD / "fastflowlm-kernel-smoke-extracted"
ARTIFACTS = paths.BUILD / "artifacts" / "fastflowlm-kernel-smoke"
ARCHIVE_NAME = "fastflowlm-kernel-smoke.tar.gz"
PRODUCT = "rocm-npu-fastflowlm-kernel-smoke"
CAPABILITIES_RELATIVE = Path("share/fastflowlm-kernel-smoke/kernel_capabilities.json")
CAPABILITIES_SOURCE = paths.ROOT / "tests" / "fastflowlm" / "kernel_capabilities.json"


def load_capabilities(path=CAPABILITIES_SOURCE):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "rocm-npu-staging-fastflowlm-kernel-test-capabilities"
        or result.get("provenance", {}).get("public_fastflowlm_abi") is not False
        or result.get("shape") != {"m": 1024, "k": 1024, "n": 128}
        or not result.get("families")
    ):
        raise RuntimeError("invalid FastFlowLM kernel capability manifest")
    names = set()
    for family in result["families"]:
        paths_to_check = [
            family.get("xclbin_directory", ""),
            *family.get("xclbins", {}).values(),
        ]
        parsed_paths = [PurePosixPath(value) for value in paths_to_check]
        if (
            not family.get("name")
            or family["name"] in names
            or family.get("nibble_semantics")
            not in {"unsigned", "signed-twos-complement"}
            or family.get("zero_point_semantics") not in {"add-after-scale", "ignored"}
            or any(
                not value or parsed.is_absolute() or ".." in parsed.parts
                for value, parsed in zip(paths_to_check, parsed_paths)
            )
        ):
            raise RuntimeError("invalid FastFlowLM family capability")
        names.add(family["name"])
    return result


def _representatives(families, key):
    seen = set()
    result = []
    for family in families:
        value = key(family)
        if value not in seen:
            seen.add(value)
            result.append(family)
    return result


def _expected_matrix(capabilities):
    families = capabilities["families"]
    shape = capabilities["shape"]
    policy = capabilities["test_policy"]
    dequant_shape = [shape["m"], shape["k"]]
    mm_shape = [shape["m"], shape["k"], shape["n"]]
    dequant_bytes = 2 * shape["m"] * shape["k"]
    mm_bytes = 2 * shape["m"] * shape["n"]

    def groups(family):
        return family["implementation_groups"]

    def nibble(family):
        return family["nibble_semantics"]

    def zero_point(family):
        return family["zero_point_semantics"]

    dequant_reps = _representatives(families, lambda f: groups(f)["dequant"])
    mm_reps = _representatives(families, lambda f: groups(f)["mm"])
    nibble_dequant_reps = _representatives(
        families, lambda f: (groups(f)["dequant"], nibble(f))
    )
    affine_dequant_reps = _representatives(
        families,
        lambda f: (groups(f)["dequant"], nibble(f), zero_point(f)),
    )
    structured_mm_reps = _representatives(
        families, lambda f: (groups(f)["mm"], groups(f)["shared_subset"])
    )
    nibble_mm_reps = _representatives(families, lambda f: (groups(f)["mm"], nibble(f)))
    affine_mm_reps = _representatives(
        families, lambda f: (groups(f)["mm"], nibble(f), zero_point(f))
    )
    cases = []
    implementation_groups = {}

    def add(family, kernel, vector, record):
        name = f"{family['name']}/{kernel}/{vector}"
        cases.append((name, *record))
        implementation_groups[name] = groups(family)[
            "dequant" if kernel == "dequant" else "mm"
        ]

    exact_zero_dequant = (
        dequant_shape,
        dequant_bytes,
        "synthetic-zero-q4nx",
        "exact-zero",
        "exact-ordered",
        0.0,
        0.0,
    )
    for family in dequant_reps:
        add(family, "dequant", "zero-baseline", exact_zero_dequant)
    for family in mm_reps:
        add(
            family,
            "mm",
            "zero-baseline",
            (
                mm_shape,
                mm_bytes,
                "synthetic-zero-bf16",
                "exact-zero",
                "exact-ordered",
                0.0,
                0.0,
            ),
        )
    for family in dequant_reps:
        add(
            family,
            "dequant",
            "structured-baseline",
            (
                dequant_shape,
                dequant_bytes,
                "generated-q4nx",
                "cpu-mm-layout",
                "exact-ordered",
                0.0,
                0.0,
            ),
        )
    for family in nibble_dequant_reps:
        reference = (
            "cpu-unsigned-nibble-mm-layout"
            if nibble(family) == "unsigned"
            else "cpu-signed-nibble-mm-layout"
        )
        add(
            family,
            "dequant",
            "full-nibble",
            (
                dequant_shape,
                dequant_bytes,
                "generated-full-nibble-q4nx",
                reference,
                "exact-ordered",
                0.0,
                0.0,
            ),
        )
    for family in affine_dequant_reps:
        reference = (
            "cpu-qwen-q-times-scale-plus-zp-mm-layout"
            if zero_point(family) == "add-after-scale"
            else "cpu-lfm-signed-nibble-zp-ignored-mm-layout"
        )
        add(
            family,
            "dequant",
            "full-affine",
            (
                dequant_shape,
                dequant_bytes,
                "generated-full-affine-q4nx",
                reference,
                "exact-ordered",
                0.0,
                0.0,
            ),
        )
    structured = (
        mm_shape,
        mm_bytes,
        "cpu-generated-mm-layout",
        "cpu-logical-matmul",
        "tolerance",
        policy["structured_absolute_tolerance"],
        policy["relative_tolerance"],
    )
    for family in mm_reps:
        add(family, "mm", "isolated-structured", structured)
    integrated_structured = (
        mm_shape,
        mm_bytes,
        "npu-dequant-preflight",
        "cpu-logical-matmul",
        "tolerance",
        policy["structured_absolute_tolerance"],
        policy["relative_tolerance"],
    )
    for family in structured_mm_reps:
        add(family, "mm", "structured-baseline", integrated_structured)
        add(family, "mm", "structured-alternate", integrated_structured)
    for family in nibble_mm_reps:
        gemm = family["gemm_policy"]
        add(
            family,
            "mm",
            "full-nibble",
            (
                mm_shape,
                mm_bytes,
                "npu-dequant-preflight",
                "cpu-family-nibble-kernel-chain",
                "tolerance",
                gemm["full_nibble_absolute_tolerance"],
                policy["relative_tolerance"],
            ),
        )
    for family in affine_mm_reps:
        gemm = family["gemm_policy"]
        add(
            family,
            "mm",
            "full-affine",
            (
                mm_shape,
                mm_bytes,
                "npu-dequant-preflight",
                "cpu-family-affine-kernel-chain",
                "tolerance",
                gemm["full_affine_absolute_tolerance"],
                policy["relative_tolerance"],
            ),
        )
    preflight_dispatches = (
        len(dequant_reps) + len(nibble_dequant_reps) + len(affine_dequant_reps)
    )
    return tuple(cases), implementation_groups, preflight_dispatches


CAPABILITIES = load_capabilities()
EXPECTED_CASES, EXPECTED_IMPLEMENTATION_GROUPS, EXPECTED_PREFLIGHT_DISPATCHES = (
    _expected_matrix(CAPABILITIES)
)


def executable_name():
    return paths.executable_name("fastflowlm-hrx-kernel-smoke")


def source_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=paths.ROOT, text=True
    ).strip()


def build_helper(fresh=False):
    source = paths.ROOT / "tests" / "fastflowlm"
    fastflowlm_source = paths.PROJ / "FastFlowLM" / "src"
    hrx_config = HRX_INSTALL / "lib" / "cmake" / "hrx" / "hrx-config.cmake"
    if not hrx_config.is_file():
        raise RuntimeError(f"missing installed HRX package: {hrx_config}")
    if not (fastflowlm_source / "lib").is_dir():
        raise RuntimeError(
            "missing FastFlowLM engine libraries; run `dev.py build fastflowlm`"
        )
    if fresh:
        shutil.rmtree(BUILD, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--fresh",
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(source),
            "-B",
            environment.cmake_path(BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={environment.cmake_path(HRX_INSTALL)}",
            f"-DFASTFLOWLM_SOURCE_DIR={environment.cmake_path(fastflowlm_source)}",
            f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(INSTALL)}",
        ],
        env=environment.cmake_env(),
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(BUILD),
            "--target",
            "fastflowlm-hrx-kernel-smoke",
            "-j",
            os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4"),
        ],
        env=environment.cmake_env(),
    )
    shutil.rmtree(INSTALL, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--install",
            environment.cmake_path(BUILD),
        ],
        env=environment.cmake_env(),
    )
    executable = INSTALL / "bin" / executable_name()
    if not executable.is_file():
        raise RuntimeError(f"kernel smoke build did not produce {executable}")
    return executable


def test_native_helper():
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(BUILD),
            "--target",
            "fastflowlm-kernel-native-tests",
            "fastflowlm-kernel-shape-probe",
            "-j",
            os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4"),
        ],
        env=environment.cmake_env(),
    )
    env = environment.cmake_env()
    search_key = "PATH" if paths.is_windows() else "LD_LIBRARY_PATH"
    search_dirs = [
        paths.PROJ / "FastFlowLM" / "src" / "lib",
        HRX_INSTALL / ("bin" if paths.is_windows() else "lib"),
    ]
    entries = [str(path) for path in search_dirs if path.is_dir()]
    if env.get(search_key):
        entries.append(env[search_key])
    env[search_key] = os.pathsep.join(entries)
    environment.sh(
        [
            "ctest",
            "--test-dir",
            environment.cmake_path(BUILD),
            "--output-on-failure",
        ],
        env=env,
    )


def _write_manifest(root):
    executable = root / "bin" / executable_name()
    capabilities = root / CAPABILITIES_RELATIVE
    manifest = {
        "schema_version": 1,
        "product": PRODUCT,
        "source_revision": source_revision(),
        "platform": {
            "system": platform.system().lower(),
            "machine": platform.machine().lower(),
        },
        "executable": {
            "path": executable.relative_to(root).as_posix(),
            "size": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
        "capabilities": {
            "path": capabilities.relative_to(root).as_posix(),
            "size": capabilities.stat().st_size,
            "sha256": sha256_file(capabilities),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_tree(root, *, require_checkout=True):
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing helper manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("product") != PRODUCT:
        raise RuntimeError("invalid FastFlowLM kernel helper manifest")
    if manifest.get("platform", {}).get("system") != platform.system().lower():
        raise RuntimeError("kernel helper platform does not match this runner")
    if require_checkout and manifest.get("source_revision") != source_revision():
        raise RuntimeError("kernel helper source revision does not match this checkout")
    record = manifest.get("executable", {})
    executable = root / record.get("path", "")
    if (
        not executable.is_file()
        or executable.stat().st_size != record.get("size")
        or sha256_file(executable) != record.get("sha256")
    ):
        raise RuntimeError("kernel helper executable identity mismatch")
    capability_record = manifest.get("capabilities", {})
    capability_path = root / capability_record.get("path", "")
    if (
        not capability_path.is_file()
        or capability_path.stat().st_size != capability_record.get("size")
        or sha256_file(capability_path) != capability_record.get("sha256")
    ):
        raise RuntimeError("kernel helper capability manifest identity mismatch")
    load_capabilities(capability_path)
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != {"manifest.json", record["path"], capability_record["path"]}:
        raise RuntimeError(
            f"kernel helper contains unexpected files: {sorted(actual_files)}"
        )
    root_bytes = str(paths.ROOT).encode()
    if root_bytes in executable.read_bytes():
        raise RuntimeError("kernel helper embeds its build checkout path")
    return executable


def package_helper(args):
    executable = build_helper(getattr(args, "fresh", False))
    test_native_helper()
    shutil.rmtree(STAGE, ignore_errors=True)
    (STAGE / "bin").mkdir(parents=True)
    shutil.copy2(executable, STAGE / "bin" / executable.name)
    capabilities = STAGE / CAPABILITIES_RELATIVE
    capabilities.parent.mkdir(parents=True)
    shutil.copy2(INSTALL / CAPABILITIES_RELATIVE, capabilities)
    _write_manifest(STAGE)
    verify_tree(STAGE)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    archive = ARTIFACTS / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz", compresslevel=6) as output:
        for path in sorted(STAGE.rglob("*")):
            output.add(path, arcname=path.relative_to(STAGE))
    print(f"package FastFlowLM kernel smoke: ok -> {archive}")


def extract_archive(directory):
    directory = Path(directory)
    archive = directory if directory.is_file() else directory / ARCHIVE_NAME
    if not archive.is_file():
        raise RuntimeError(f"missing FastFlowLM kernel helper archive: {archive}")
    shutil.rmtree(EXTRACTED, ignore_errors=True)
    EXTRACTED.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as source:
        if any(member.issym() or member.islnk() for member in source.getmembers()):
            raise RuntimeError("kernel helper archive contains links")
        source.extractall(EXTRACTED, filter="data")
    return EXTRACTED


def verify_helper(args):
    root = extract_archive(args.dir)
    verify_tree(root)
    print(f"verify FastFlowLM kernel smoke: ok -> {root}")


def _wheel_paths():
    try:
        import rocm_npu_fastflowlm
        import rocm_npu_runtime
    except ImportError as error:
        raise RuntimeError(
            f"installed rocm-npu wheel package is unavailable: {error}"
        ) from error
    flm_root = Path(rocm_npu_fastflowlm.flm_path()).resolve().parents[1]
    runtime = Path(rocm_npu_runtime.libhrx_path()).resolve()
    return {
        "xclbin_root": Path(rocm_npu_fastflowlm.xclbin_path()) / "xclbins",
        "search_dirs": [
            flm_root / "bin",
            flm_root / "lib" / "flm",
            flm_root / "lib",
            runtime.parent,
        ],
    }


def validate_report(report, capabilities=CAPABILITIES):
    expected_cases, expected_groups, expected_preflight = _expected_matrix(capabilities)
    cases = report.get("cases", [])
    resident_rounds = report.get("resident_rounds")
    expected_dispatches = (
        expected_preflight + len(expected_cases) * (5 + resident_rounds)
        if isinstance(resident_rounds, int)
        else None
    )
    if (
        report.get("status") != "pass"
        or report.get("failures") != 0
        or report.get("capability_contract_id") != capabilities["contract_id"]
        or report.get("implementation_id_algorithm") != "fnv1a64"
        or [case.get("name") for case in cases]
        != [record[0] for record in expected_cases]
        or len(cases) != len(expected_cases)
        or report.get("lifecycle_rounds") != 5
        or not isinstance(resident_rounds, int)
        or resident_rounds <= 0
        or report.get("iterations") != resident_rounds
        or report.get("preflight_dispatches") != expected_preflight
        or report.get("dispatches") != expected_dispatches
        or not all(
            math.isfinite(report.get(field, math.nan)) and report[field] > 0
            for field in (
                "elapsed_seconds",
                "device_region_seconds",
                "process_wall_seconds",
            )
        )
    ):
        raise RuntimeError(f"unexpected kernel smoke report: {report}")
    implementation_ids = {case.get("implementation_id") for case in cases}
    ids_by_group = {}
    for case in cases:
        group = expected_groups.get(case.get("name"))
        ids_by_group.setdefault(group, set()).add(case.get("implementation_id"))
    if (
        not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value)
            for value in implementation_ids
        )
        or set(ids_by_group) != set(expected_groups.values())
        or any(len(ids) != 1 for ids in ids_by_group.values())
        or len({next(iter(ids)) for ids in ids_by_group.values()}) != len(ids_by_group)
    ):
        raise RuntimeError("kernel smoke xclbin capability identity mismatch")
    if any(
        case.get("validated_bytes") != expected_bytes
        or case.get("failures") != 0
        or case.get("mismatches") != 0
        or not math.isfinite(case.get("maximum_absolute_error", math.nan))
        or not math.isfinite(case.get("maximum_relative_error", math.nan))
        or not math.isfinite(case.get("maximum_tolerance_ratio", math.nan))
        or case.get("maximum_tolerance_ratio") > 1.000001
        or not math.isclose(
            case.get("absolute_tolerance", math.nan),
            absolute_tolerance,
            abs_tol=1e-9,
        )
        or not math.isclose(
            case.get("relative_tolerance", math.nan),
            relative_tolerance,
            abs_tol=1e-9,
        )
        or case.get("weight_provenance") != weight_provenance
        or case.get("reference_kind") != reference_kind
        or case.get("comparison") != comparison
        or case.get("device_seconds", 0) <= 0
        or case.get("dimensions") != expected_dimensions
        or case.get("lifecycle_rounds") != 5
        or case.get("resident_rounds") != report["resident_rounds"]
        or case.get("name")
        != (f"{case.get('family')}/{case.get('kernel')}/{case.get('vector_id')}")
        for case, (
            _,
            expected_dimensions,
            expected_bytes,
            weight_provenance,
            reference_kind,
            comparison,
            absolute_tolerance,
            relative_tolerance,
        ) in zip(cases, expected_cases)
    ):
        raise RuntimeError("kernel smoke report contains an empty numeric validation")
    return report


def run_helper(args):
    if args.dir:
        root = extract_archive(args.dir)
        executable = verify_tree(root)
        capability_path = root / CAPABILITIES_RELATIVE
    else:
        executable = INSTALL / "bin" / executable_name()
        if not executable.is_file():
            executable = build_helper(False)
        capability_path = INSTALL / CAPABILITIES_RELATIVE
    capabilities = load_capabilities(capability_path)
    if args.wheel_package:
        runtime = _wheel_paths()
    else:
        runtime = {
            "xclbin_root": paths.PROJ / "FastFlowLM" / "src" / "xclbins",
            "search_dirs": [
                paths.PROJ / "FastFlowLM" / "src" / "lib",
                HRX_INSTALL / ("bin" if paths.is_windows() else "lib"),
            ],
        }
    env = os.environ.copy()
    env.setdefault("HRX_GPU_DRIVER", "amdxdna")
    search_key = "PATH" if paths.is_windows() else "LD_LIBRARY_PATH"
    entries = [str(path) for path in runtime["search_dirs"] if path.is_dir()]
    if env.get(search_key):
        entries.append(env[search_key])
    env[search_key] = os.pathsep.join(entries)
    command = [
        str(executable),
        "--xclbin-root",
        str(runtime["xclbin_root"]),
        "--capabilities",
        str(capability_path),
        "--iterations",
        str(args.iterations),
        "--json",
    ]
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    print(completed.stdout, end="")
    report = json.loads(completed.stdout.splitlines()[-1])
    return validate_report(report, capabilities)
