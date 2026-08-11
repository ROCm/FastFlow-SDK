# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared build and parity infrastructure owned by qualification."""

import importlib.util
import json
import os
import shutil
from pathlib import Path, PurePosixPath

from build_tools.dev import environment, paths
from build_tools.dev.build import HRX_INSTALL
from build_tools.fastflowlm_artifact import sha256_file


QUALIFICATION_BUILD = paths.BUILD / "fastflowlm-qualification-tools"
XADX_STAGE = paths.BUILD / "fastflowlm-xadx-assets"
XADX_PRODUCT = "rocm-npu-fastflowlm-xadx"
XADX_IMPLEMENTATION_PROBE_TARGET = "fastflowlm-xadx-implementation-probe"
VECTOR_DUMP_TARGET = "fastflowlm-q4nx-vector-dump"
VECTORS = paths.BUILD / "fastflowlm-q4nx-vectors"
CAPABILITIES_SOURCE = (
    paths.ROOT
    / "tools/rocm-npu-qualification/python/rocm_npu_qualification"
    / "fastflowlm-kernel-capabilities-v1.json"
)
KERNEL_RECIPE_SCHEMA = (
    paths.ROOT
    / "tools"
    / "rocm-npu-qualification"
    / "python"
    / "rocm_npu_qualification"
    / "schemas"
    / "kernel-recipe-v1.schema.json"
)
def load_capabilities(path=CAPABILITIES_SOURCE):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != 1
        or result.get("kind")
        != "rocm-npu-fastflowlm-kernel-qualification-capabilities"
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


def configure_qualification_tools(*, fresh=False):
    source = paths.ROOT / "tools" / "rocm-npu-qualification" / "native"
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
        shutil.rmtree(QUALIFICATION_BUILD, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--fresh",
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(source),
            "-B",
            environment.cmake_path(QUALIFICATION_BUILD),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={environment.cmake_path(HRX_INSTALL)}",
            f"-DFASTFLOWLM_SOURCE_DIR={environment.cmake_path(fastflowlm_source)}",
        ],
        env=environment.cmake_env(),
    )
    return fastflowlm_source


def file_record(root, path):
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def recipe_generators():
    path = CAPABILITIES_SOURCE.with_name("recipe_generators.py")
    module_spec = importlib.util.spec_from_file_location(
        "rocm_npu_qualification_recipe_generator_parity", path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load qualification recipe generators")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def verify_recipe_generator_parity(destination=VECTORS):
    """Verify each recipe generator against the strongest available C++ authority.

    Default 32x256 Q4NX vectors are produced by FastFlowLM's converter.
    Parameterized 32x128 and 16x128 vectors are C++/Python interpreter agreement
    over measured layout facts; on-device qualification remains their independent
    authority.
    """
    import numpy as np

    configure_qualification_tools(fresh=False)
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(QUALIFICATION_BUILD),
            "--target",
            VECTOR_DUMP_TARGET,
            "-j",
            os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4"),
        ],
        env=environment.cmake_env(),
    )
    destination = Path(destination)
    shutil.rmtree(destination, ignore_errors=True)
    env = environment.cmake_env()
    search_key = "PATH" if paths.is_windows() else "LD_LIBRARY_PATH"
    entries = [
        str(path)
        for path in (
            paths.PROJ / "FastFlowLM" / "src" / "lib",
            HRX_INSTALL / ("bin" if paths.is_windows() else "lib"),
        )
        if path.is_dir()
    ]
    if env.get(search_key):
        entries.append(env[search_key])
    env[search_key] = os.pathsep.join(entries)
    environment.sh(
        [
            str(QUALIFICATION_BUILD / paths.executable_name(VECTOR_DUMP_TARGET)),
            str(destination),
        ],
        env=env,
    )

    generators = recipe_generators()
    shape = json.loads((destination / "index.json").read_text(encoding="utf-8"))["shape"]
    m, k, n = shape["m"], shape["k"], shape["n"]
    checked = 0
    for profile in generators.PROFILES:
        tensors = generators.source_tensors(m, k, profile)
        expected = {
            f"{profile}-packed.bin": generators.pack_q4nx(*tensors),
            f"{profile}-mm-layout.bin": generators.dequant_expected(
                *tensors, "logical", "mm"
            ),
            f"{profile}-gpt-oss-layout.bin": generators.dequant_expected(
                *tensors, "logical", "gpt-oss"
            ),
            f"{profile}-gemma-text-layout.bin": generators.dequant_expected(
                *tensors, "logical", "gemma-text"
            ),
        }
        for semantics in ("logical", "signed-nibble", "qwen-affine", "lfm-affine"):
            expected[f"{profile}-{semantics}.bin"] = generators.dequantize(
                *tensors, semantics
            ).tobytes()
        expected[f"{profile}-matmul-29-128.bin"] = generators.gemm_reference(
            m, k, 128, profile, 29
        )
        expected[f"{profile}-matmul-29-512.bin"] = generators.gemm_reference(
            m, k, 512, profile, 29
        )
        for semantics, order in (
            ("lfm-affine", "mm"),
            ("qwen-affine", "mm"),
            ("qwen-affine", "gemma-text"),
        ):
            expected[
                f"{profile}-chain-{semantics}-{order}-512.bin"
            ] = generators.chained_reference(
                m, k, 512, profile, 29, semantics, order
            )
        for salt in (29, 41, 46, 54):
            expected[f"activation-{salt}-logical.bin"] = generators.activation(
                k, n, salt
            ).tobytes()
            expected[f"activation-{salt}-physical.bin"] = (
                generators.mm_activation_layout(k, n, salt, "gemm")
            )
            expected[f"activation-{salt}-gemm-512.bin"] = (
                generators.mm_activation_layout(k, 512, salt, "gemm")
            )
            expected[f"activation-{salt}-gemma-gemm-512.bin"] = (
                generators.mm_activation_layout(k, 512, salt, "gemma-gemm")
            )
        expected["matmul-shared-subset-29-29.bin"] = generators.gemm_reference(
            m, k, n, "shared-subset", 29
        )
        expected["bf16-edge-values.bin"] = generators.bf16(
            np.asarray(
                [
                    0.0,
                    -0.0,
                    np.finfo(np.float32).smallest_subnormal,
                    np.inf,
                    -np.inf,
                    np.nan,
                    1.00390625,
                    1.01171875,
                    -1.00390625,
                    -1.01171875,
                ],
                dtype=np.float32,
            )
        ).tobytes()
        for relative, data in expected.items():
            if (destination / relative).read_bytes() != data:
                raise RuntimeError(
                    f"qualification recipe generator disagrees with C++: {relative}"
                )
            checked += 1
    blocks = m * k // generators.GROUP_COLUMNS
    for shift in range(5):
        coordinate = (
            np.arange(blocks, dtype=np.int64) & 0x7F
            if shift % 2 == 0
            else np.arange(blocks, dtype=np.int64) >> 7
        )
        scale = generators.bf16(coordinate).reshape(m, k // 32)
        zero_point = coordinate.reshape(m, k // 32)
        quantized = ((np.arange(m * k, dtype=np.int64) >> (4 * shift)) & 0xF).reshape(
            m, k
        )
        for row_block, col_block, parallel, prefix in (
            (32, 256, 16, "probe"),
            (32, 128, 16, "probe-32x128x16"),
            (16, 128, 16, "probe-16x128x16"),
        ):
            expected = generators.pack_q4nx(
                quantized,
                generators.widen(scale),
                zero_point.astype("int64"),
                row_block,
                col_block,
                parallel,
            )
            relative = f"{prefix}-nibble-shift{shift}.bin"
            if (destination / relative).read_bytes() != expected:
                authority = (
                    "converter-backed"
                    if (row_block, col_block) == (32, 256)
                    else "measured-fact C++/Python interpreter"
                )
                raise RuntimeError(
                    f"qualification {authority} Q4NX packing disagrees: {relative}"
                )
            checked += 1
    return checked
