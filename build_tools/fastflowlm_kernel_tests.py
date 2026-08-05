# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build, package, verify, and run the FastFlowLM production-xclbin smoke."""

import gzip
import importlib.util
import io
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
XADX_STAGE = paths.BUILD / "fastflowlm-xadx-assets"
XADX_PRODUCT = "rocm-npu-fastflowlm-xadx"
XADX_GENERATOR_TARGET = "fastflowlm-xadx-asset-generator"
ARCHIVE_NAME = "fastflowlm-kernel-smoke.tar.gz"
PRODUCT = "rocm-npu-fastflowlm-kernel-smoke"
CAPABILITIES_RELATIVE = Path("share/fastflowlm-kernel-smoke/kernel_capabilities.json")
CAPABILITIES_SOURCE = paths.ROOT / "tests" / "fastflowlm" / "kernel_capabilities.json"
XADX_CASES_SOURCE = (
    paths.ROOT
    / "tools"
    / "rocm-npu-qualification"
    / "python"
    / "rocm_npu_qualification"
    / "xadx_cases.json"
)
# Role -> CMake target. "kernel-smoke" validates the production xclbins
# numerically; "readback-smoke" validates that the runtime keeps a host mapping
# coherent across back-to-back device writes.
TARGETS = {
    "kernel-smoke": "fastflowlm-hrx-kernel-smoke",
    "readback-smoke": "fastflowlm-hrx-readback-smoke",
}


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


def executable_name(role="kernel-smoke"):
    return paths.executable_name(TARGETS[role])


def source_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=paths.ROOT, text=True
    ).strip()


def _installed_executables():
    return {role: INSTALL / "bin" / executable_name(role) for role in TARGETS}


def _configure_helper(*, fresh=False, build_testing=True):
    source = paths.ROOT / "tests" / "fastflowlm"
    fastflowlm_source = paths.PROJ / "FastFlowLM" / "src"
    hrx_config = HRX_INSTALL / "lib" / "cmake" / "hrx" / "hrx-config.cmake"
    if not hrx_config.is_file():
        raise RuntimeError(f"missing installed HRX package: {hrx_config}")
    required_fastflowlm = (
        fastflowlm_source / "include",
        fastflowlm_source / "lib",
        fastflowlm_source / "xclbins",
    )
    missing = [str(path) for path in required_fastflowlm if not path.is_dir()]
    if missing:
        raise RuntimeError(
            "FastFlowLM source is not initialized or is incomplete; missing "
            + ", ".join(missing)
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
            f"-DBUILD_TESTING={'ON' if build_testing else 'OFF'}",
            f"-DCMAKE_PREFIX_PATH={environment.cmake_path(HRX_INSTALL)}",
            f"-DFASTFLOWLM_SOURCE_DIR={environment.cmake_path(fastflowlm_source)}",
            f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(INSTALL)}",
        ],
        env=environment.cmake_env(),
    )
    return fastflowlm_source


def build_xadx_generator(fresh=False):
    """Builds only the host-side serializer; the FastFlowLM app is not required."""
    _configure_helper(fresh=fresh, build_testing=False)
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(BUILD),
            "--target",
            XADX_GENERATOR_TARGET,
            "-j",
            os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4"),
        ],
        env=environment.cmake_env(),
    )
    generator = BUILD / paths.executable_name(XADX_GENERATOR_TARGET)
    if not generator.is_file():
        raise RuntimeError(f"XADX build did not produce {generator}")
    return generator


def build_helper(fresh=False):
    _configure_helper(fresh=fresh, build_testing=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(BUILD),
            "--target",
            *sorted(TARGETS.values()),
            XADX_GENERATOR_TARGET,
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
    executables = _installed_executables()
    for executable in executables.values():
        if not executable.is_file():
            raise RuntimeError(f"kernel smoke build did not produce {executable}")
    return executables


def _file_record(root, path):
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _assert_runtime_generator_parity(destination, specifications):
    generator_path = XADX_CASES_SOURCE.with_name("xadx_generators.py")
    module_spec = importlib.util.spec_from_file_location(
        "rocm_npu_qualification_xadx_generator_parity", generator_path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load qualification XADX runtime generators")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    for relative, specification in specifications.items():
        generated = (Path(destination) / relative).read_bytes()
        materialized, _ = module.materialize(specification, Path(destination))
        if materialized != generated:
            raise RuntimeError(
                f"qualification XADX runtime generator disagrees with C++: {relative}"
            )


def _deterministic_gzip(data):
    stream = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=stream,
        mtime=0,
    ) as archive:
        archive.write(data)
    return stream.getvalue()


def generate_xadx_assets(xclbin_root, destination=XADX_STAGE):
    """Generates deterministic NPU2 XADX qualification assets without an NPU."""
    destination = Path(destination)
    generator = BUILD / paths.executable_name(XADX_GENERATOR_TARGET)
    if not generator.is_file():
        build_xadx_generator(False)
    if not generator.is_file():
        raise RuntimeError(f"missing FastFlowLM XADX asset generator: {generator}")
    xclbin_root = Path(xclbin_root)
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
            str(generator),
            str(xclbin_root),
            str(CAPABILITIES_SOURCE),
            str(destination),
        ],
        env=env,
    )
    draft_path = destination / ".generation.json"
    if not draft_path.is_file():
        raise RuntimeError("FastFlowLM XADX generator produced no metadata")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if (
        draft.get("serializer_api") != "hrx_amdxdna_xadx_serialize"
        or draft.get("serializer_revision") != "hrx-amdxdna-v0"
    ):
        raise RuntimeError("FastFlowLM XADX generator used an unknown serializer")
    transactions = {}
    for kernel, relative in draft["transactions"].items():
        path = destination / relative
        transactions[kernel] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    capabilities = destination / "capability-manifest.json"
    shutil.copy2(CAPABILITIES_SOURCE, capabilities)
    cases = []
    executables = []
    coherence_executable = None
    shape = draft["shape"]

    def zero(size):
        return {
            "kind": "generator",
            "id": "zero",
            "parameters": {"size": size},
        }

    mm_weight = {
        "kind": "generator",
        "id": "mm-weight-bf16",
        "parameters": {
            "m": shape["m"],
            "k": shape["k"],
            "salt": 41,
            "padding": 4096,
        },
    }

    def mm_activation(salt):
        return {
            "kind": "generator",
            "id": "mm-activation-bf16",
            "parameters": {"k": shape["k"], "n": shape["n"], "salt": salt},
        }

    _assert_runtime_generator_parity(
        destination,
        {
            "data/mm-weight.bin": mm_weight,
            "data/mm-activation-a.bin": mm_activation(41),
            "data/mm-activation-b.bin": mm_activation(54),
        },
    )

    for record in draft["executables"]:
        source = xclbin_root / record["source_xclbin"]
        xadx = destination / record["xadx"]
        if not source.is_file() or not xadx.is_file():
            raise RuntimeError(f"missing XADX source or output for {record['id']}")
        executable = {
            **record,
            "source_xclbin_size": source.stat().st_size,
            "source_xclbin_sha256": sha256_file(source),
            "transaction_size": transactions[record["kernel"]]["size"],
            "transaction_sha256": transactions[record["kernel"]]["sha256"],
            "xadx_size": xadx.stat().st_size,
            "xadx_sha256": sha256_file(xadx),
        }
        executables.append(executable)
        if record["kernel"] == "dequant":
            cases.append(
                {
                    "name": f"{record['family']}/dequant/zero-baseline",
                    "xadx": record["xadx"],
                    "export": "MLIR_AIE",
                    "buffers": [
                        {
                            "name": "output",
                            "direction": "output",
                            "size": 2 * shape["m"] * shape["k"],
                            "expected": zero(2 * shape["m"] * shape["k"]),
                            "comparison": "exact",
                        },
                        {
                            "name": "weight",
                            "direction": "input",
                            "size": (destination / "data/dequant-weight-zero.bin")
                            .stat()
                            .st_size,
                            "input": zero(
                                (destination / "data/dequant-weight-zero.bin")
                                .stat()
                                .st_size
                            ),
                        },
                        {
                            "name": "parameters",
                            "direction": "input",
                            "size": 4096,
                            "input": zero(4096),
                        },
                    ],
                }
            )
        else:
            cases.append(
                {
                    "name": f"{record['family']}/mm/structured",
                    "xadx": record["xadx"],
                    "export": "MLIR_AIE",
                    "buffers": [
                        {
                            "name": "output",
                            "direction": "output",
                            "size": 2 * shape["m"] * shape["n"],
                            "expected": {
                                "kind": "gzip",
                                "path": "data/mm-expected-a.bin.gz",
                            },
                            "comparison": "bf16-tolerance",
                            "atol": draft["tolerances"]["atol"],
                            "rtol": draft["tolerances"]["rtol"],
                        },
                        {
                            "name": "weight",
                            "direction": "input",
                            "size": (destination / "data/mm-weight.bin").stat().st_size,
                            "input": mm_weight,
                        },
                        {
                            "name": "activation",
                            "direction": "input",
                            "size": (destination / "data/mm-activation-a.bin")
                            .stat()
                            .st_size,
                            "input": mm_activation(41),
                        },
                    ],
                }
            )
            if coherence_executable is None:
                coherence_executable = record
    if coherence_executable is None:
        raise RuntimeError("FastFlowLM XADX generation has no MM coherence kernel")
    cases.append(
        {
            "name": "host-readback/coherence",
            "xadx": coherence_executable["xadx"],
            "export": "MLIR_AIE",
            "buffers": [
                {
                    "name": "output",
                    "direction": "output",
                    "size": 2 * shape["m"] * shape["n"],
                    "expected": {
                        "kind": "gzip",
                        "path": "data/mm-expected-a.bin.gz",
                    },
                    "comparison": "bf16-tolerance",
                    "atol": draft["tolerances"]["atol"],
                    "rtol": draft["tolerances"]["rtol"],
                },
                {
                    "name": "weight",
                    "direction": "input",
                    "size": (destination / "data/mm-weight.bin").stat().st_size,
                    "input": mm_weight,
                },
                {
                    "name": "activation",
                    "direction": "input",
                    "size": (destination / "data/mm-activation-a.bin").stat().st_size,
                    "input": mm_activation(41),
                },
            ],
            "readback_variants": [
                {
                    "buffer": "activation",
                    "input": mm_activation(41),
                    "expected": {
                        "kind": "gzip",
                        "path": "data/mm-expected-a.bin.gz",
                    },
                },
                {
                    "buffer": "activation",
                    "input": mm_activation(54),
                    "expected": {
                        "kind": "gzip",
                        "path": "data/mm-expected-b.bin.gz",
                    },
                },
            ],
        }
    )
    case_contract = json.loads(XADX_CASES_SOURCE.read_text(encoding="utf-8"))
    expected_case_names = [
        f"{family}/{suffix}"
        for family in case_contract["families"]
        for suffix in case_contract["case_suffixes"]
    ] + case_contract["global_cases"]
    if (
        case_contract.get("schema_version") != 1
        or case_contract.get("shape") != shape
        or [case["name"] for case in cases] != expected_case_names
    ):
        raise RuntimeError("generated XADX cases disagree with qualification contract")
    for name in ("mm-expected-a.bin", "mm-expected-b.bin"):
        source = destination / "data" / name
        source.with_suffix(source.suffix + ".gz").write_bytes(
            _deterministic_gzip(source.read_bytes())
        )
    for path in (destination / "data").glob("*.bin"):
        path.unlink()
    shutil.rmtree(destination / ".transactions")
    draft_path.unlink()
    files = [
        _file_record(destination, path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "schema_version": 1,
        "product": XADX_PRODUCT,
        "target": "npu2",
        "serializer": {
            "api": draft["serializer_api"],
            "revision": draft["serializer_revision"],
        },
        "case_contract_sha256": sha256_file(XADX_CASES_SOURCE),
        "capabilities": _file_record(destination, capabilities),
        "shape": shape,
        "executables": executables,
        "cases": cases,
        "files": files,
    }
    (destination / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    verify_xadx_assets(destination)
    return destination


def cmd_generate_xadx_assets(args):
    build_xadx_generator(args.fresh)
    return generate_xadx_assets(
        paths.PROJ / "FastFlowLM" / "src" / "xclbins",
        destination=args.dir,
    )


def verify_xadx_assets(root):
    """Rejects missing, extra, corrupted, or provenance-incomplete XADX assets."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("product") != XADX_PRODUCT
        or manifest.get("target") != "npu2"
        or manifest.get("serializer")
        != {"api": "hrx_amdxdna_xadx_serialize", "revision": "hrx-amdxdna-v0"}
        or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("case_contract_sha256", ""))
    ):
        raise RuntimeError("invalid FastFlowLM XADX package manifest")
    expected = {record["path"]: record for record in manifest.get("files", [])}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(expected) != set(actual):
        raise RuntimeError("FastFlowLM XADX package inventory mismatch")
    for relative, path in actual.items():
        record = expected[relative]
        if (
            path.stat().st_size != record["size"]
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"FastFlowLM XADX package corruption: {relative}")
    for executable in manifest.get("executables", []):
        xadx = root / executable["xadx"]
        if (
            not executable.get("source_xclbin_sha256")
            or not executable.get("transaction_sha256")
            or xadx.stat().st_size != executable.get("xadx_size")
            or sha256_file(xadx) != executable.get("xadx_sha256")
        ):
            raise RuntimeError(f"incomplete XADX provenance: {executable.get('id')}")
    return manifest


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


def _identity(root, path):
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_manifest(root):
    capabilities = root / CAPABILITIES_RELATIVE
    manifest = {
        "schema_version": 2,
        "product": PRODUCT,
        "source_revision": source_revision(),
        "platform": {
            "system": platform.system().lower(),
            "machine": platform.machine().lower(),
        },
        "executables": {
            role: _identity(root, root / "bin" / executable_name(role))
            for role in sorted(TARGETS)
        },
        "capabilities": _identity(root, capabilities),
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
    if manifest.get("schema_version") != 2 or manifest.get("product") != PRODUCT:
        raise RuntimeError("invalid FastFlowLM kernel helper manifest")
    if manifest.get("platform", {}).get("system") != platform.system().lower():
        raise RuntimeError("kernel helper platform does not match this runner")
    if require_checkout and manifest.get("source_revision") != source_revision():
        raise RuntimeError("kernel helper source revision does not match this checkout")
    records = manifest.get("executables", {})
    if set(records) != set(TARGETS):
        raise RuntimeError("kernel helper manifest does not cover every helper")
    executables = {}
    for role, record in records.items():
        executable = root / record.get("path", "")
        if (
            not executable.is_file()
            or executable.stat().st_size != record.get("size")
            or sha256_file(executable) != record.get("sha256")
        ):
            raise RuntimeError(f"kernel helper {role} executable identity mismatch")
        executables[role] = executable
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
    expected_files = {"manifest.json", capability_record["path"]}
    expected_files.update(record["path"] for record in records.values())
    if actual_files != expected_files:
        raise RuntimeError(
            f"kernel helper contains unexpected files: {sorted(actual_files)}"
        )
    root_bytes = str(paths.ROOT).encode()
    for role, executable in executables.items():
        if root_bytes in executable.read_bytes():
            raise RuntimeError(f"kernel helper {role} embeds its build checkout path")
    return executables


def package_helper(args):
    executables = build_helper(getattr(args, "fresh", False))
    test_native_helper()
    generate_xadx_assets(paths.PROJ / "FastFlowLM" / "src" / "xclbins")
    shutil.rmtree(STAGE, ignore_errors=True)
    (STAGE / "bin").mkdir(parents=True)
    for executable in executables.values():
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
        executables = verify_tree(root)
        capability_path = root / CAPABILITIES_RELATIVE
    else:
        executables = _installed_executables()
        if not all(executable.is_file() for executable in executables.values()):
            executables = build_helper(False)
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

    def run(role, *extra):
        completed = subprocess.run(
            [
                str(executables[role]),
                "--xclbin-root",
                str(runtime["xclbin_root"]),
                "--capabilities",
                str(capability_path),
                *extra,
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        print(completed.stdout, end="")
        return completed.stdout

    output = run("kernel-smoke", "--iterations", str(args.iterations), "--json")
    report = validate_report(json.loads(output.splitlines()[-1]), capabilities)
    run("readback-smoke", "--iterations", str(args.readback_iterations))
    return report
