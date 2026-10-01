# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Artifact helpers for compiler-free NPU XADX test packages.

CI NPU e2e unpacks this archive and runs ``dev.py test-npu-package``. That is
not ``test-npu-iron-ops`` (IRON pytest, local only).
"""

import json
import os
import platform
import shutil
import sys
from pathlib import Path

from build_tools import (
    aiecc_elf_xadx,
    hrx_runtime,
    iron,
    npu_prereqs,
    npu_test_cases,
    paths,
)
from build_tools import archive as archive_tools
from build_tools.archive import (
    create_deterministic_tar_gz,
    create_deterministic_tar_zstd_cli,
)
from build_tools.artifact import (
    file_inventory,
    git_revision,
    verify_file_inventory,
)

ARCHIVE_STEM = "npu-test-package.tar"
FULL_ELF_LOAD_PDI_FEATURE = "amdxdna-full-elf-load-pdi"
KNOWN_REQUIRED_FEATURES = {FULL_ELF_LOAD_PDI_FEATURE}


def find_archive(directory):
    directory = Path(directory)
    for extension in ("zst", "gz"):
        candidate = directory / f"{ARCHIVE_STEM}.{extension}"
        if candidate.is_file():
            return candidate
    return None


def write_case(stage, generated, xadx, source, required_features=()):
    """Writes one generated numeric case and returns its manifest record."""
    import numpy as np

    stage = Path(stage)
    case_dir = stage / "cases" / generated["name"]
    case_dir.mkdir(parents=True)
    xadx_path = case_dir / "kernel.xadx"
    xadx_path.write_bytes(xadx)
    buffers = []
    for descriptor in generated["buffers"]:
        record = {
            key: (list(value) if key == "shape" else value)
            for key, value in descriptor.items()
            if key not in ("data", "expected")
        }
        storage_dtype = np.dtype(record["dtype"])
        record["dtype"] = storage_dtype.str
        if descriptor["direction"] == "input":
            field, value = "input", descriptor["data"]
        else:
            field, value = "expected", descriptor["expected"]
        data_path = case_dir / f"{descriptor['name']}.bin"
        np.asarray(value, dtype=storage_dtype).tofile(data_path)
        record[field] = data_path.relative_to(stage).as_posix()
        buffers.append(record)
    case = {
        "name": generated["name"],
        "xadx": xadx_path.relative_to(stage).as_posix(),
        "export": "MLIR_AIE",
        "source": source,
        "buffers": buffers,
    }
    if required_features:
        case["required_features"] = list(required_features)
    dispatches = []
    for index, dispatch in enumerate(generated.get("dispatches", ())):
        dispatch_name = dispatch["name"]
        constants_path = case_dir / f"dispatch-{index}-{dispatch_name}-constants.bin"
        constants_path.write_bytes(dispatch["constants"])
        expected_paths = {}
        for descriptor in generated["buffers"]:
            if descriptor["direction"] != "output":
                continue
            buffer_name = descriptor["name"]
            expected_path = (
                case_dir / f"dispatch-{index}-{dispatch_name}-{buffer_name}.bin"
            )
            np.asarray(
                dispatch["expected"][buffer_name],
                dtype=np.dtype(descriptor["dtype"]),
            ).tofile(expected_path)
            expected_paths[buffer_name] = expected_path.relative_to(stage).as_posix()
        dispatches.append(
            {
                "name": dispatch_name,
                "constants": constants_path.relative_to(stage).as_posix(),
                "expected": expected_paths,
            }
        )
    if dispatches:
        case["dispatches"] = dispatches
    return case


def write_manifest(stage, cases, source_revision, system, machine):
    """Inventories a staged package and writes its identity manifest."""
    stage = Path(stage)
    manifest = {
        "schema_version": 1,
        "product": "rocm-npu-test-package",
        "source_revision": source_revision,
        "platform": {"system": system, "machine": machine},
        "cases": cases,
        "files": file_inventory(stage),
    }
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def create_archive(stage, output_directory, codec):
    """Creates the platform-selected compressed tar archive."""
    stage = Path(stage)
    output_directory = Path(output_directory)
    archive = output_directory / f"{ARCHIVE_STEM}.{codec}"
    output_directory.mkdir(parents=True, exist_ok=True)
    for other in output_directory.glob(f"{ARCHIVE_STEM}.*"):
        if other != archive:
            other.unlink()
    if codec == "zst":
        create_deterministic_tar_zstd_cli(stage, archive, arcname=".", level=6)
    elif codec == "gz":
        create_deterministic_tar_gz(stage, archive, arcname=".", compresslevel=6)
    else:
        raise ValueError(f"unsupported NPU test package codec: {codec}")
    return archive


def extract_archive(archive, destination):
    """Safely extracts a link-free test package into an empty destination."""
    archive = Path(archive)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive_tools.extract_archive(archive, destination)


def verify_tree(package, expected_cases, source_revision, system):
    """Verifies identity, complete inventory, case data, and XADX containers."""
    package = Path(package)
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing NPU test package manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported NPU test package schema")
    if manifest.get("product") != "rocm-npu-test-package":
        raise RuntimeError("not a rocm-npu-test-package manifest")
    if manifest.get("source_revision") != source_revision:
        raise RuntimeError(
            "NPU test package source revision does not match this checkout"
        )
    if manifest.get("platform", {}).get("system") != system:
        raise RuntimeError("NPU test package platform does not match this runner")

    expected_cases = set(expected_cases)
    cases = manifest.get("cases", [])
    case_names = {case.get("name") for case in cases}
    if case_names != expected_cases or len(cases) != len(expected_cases):
        raise RuntimeError(
            "NPU test package cases do not match the declared suite: "
            f"expected {sorted(expected_cases)}, got {sorted(case_names)}"
        )

    try:
        verify_file_inventory(
            package,
            manifest.get("files"),
            exclude_paths=(manifest_path,),
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"NPU test package inventory verification failed: {error}"
        ) from error

    for case in cases:
        required_features = case.get("required_features", [])
        if (
            not isinstance(required_features, list)
            or len(required_features) != len(set(required_features))
            or not set(required_features) <= KNOWN_REQUIRED_FEATURES
        ):
            raise RuntimeError(
                f"{case['name']}: invalid required_features: {required_features!r}"
            )
        data = (package / case["xadx"]).read_bytes()
        if len(data) < 8 or data[4:8] != b"XADX":
            raise RuntimeError(f"{case['name']}: malformed XADX container")
        if case["name"] == "vector_scalar_add_chain4":
            source = case.get("source", {})
            if (
                source.get("name") != "vector_scalar_add"
                or source.get("run_count") != 4
            ):
                raise RuntimeError(
                    "vector_scalar_add_chain4 source identity must declare "
                    "vector_scalar_add with run_count 4"
                )
        for buffer in case["buffers"]:
            field = "input" if buffer["direction"] == "input" else "expected"
            if not (package / buffer[field]).is_file():
                raise RuntimeError(
                    f"{case['name']}.{buffer['name']}: missing {field} data"
                )
        dispatch_names = set()
        output_names = {
            buffer["name"]
            for buffer in case["buffers"]
            if buffer["direction"] == "output"
        }
        for dispatch in case.get("dispatches", ()):
            dispatch_name = dispatch.get("name")
            if not dispatch_name or dispatch_name in dispatch_names:
                raise RuntimeError(
                    f"{case['name']}: dispatch names must be non-empty and unique"
                )
            dispatch_names.add(dispatch_name)
            if not (package / dispatch["constants"]).is_file():
                raise RuntimeError(
                    f"{case['name']}.{dispatch_name}: missing dispatch constants"
                )
            expected = dispatch.get("expected", {})
            if set(expected) != output_names:
                raise RuntimeError(
                    f"{case['name']}.{dispatch_name}: expected outputs must be "
                    f"{sorted(output_names)}, got {sorted(expected)}"
                )
            for buffer_name, relative_path in expected.items():
                if not (package / relative_path).is_file():
                    raise RuntimeError(
                        f"{case['name']}.{dispatch_name}.{buffer_name}: "
                        "missing expected data"
                    )
    return manifest


ROOT = paths.ROOT
PROJ = paths.PROJ
OUT = paths.OUT
NPU_TEST_PACKAGE_STAGE = paths.BUILD / "npu-test-package"
NPU_TEST_PACKAGE_EXTRACTED = paths.BUILD / "npu-test-package-extracted"
IS_WINDOWS = paths.is_windows()


def _single_artifact(directory, pattern, description):
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise RuntimeError(
            f"{directory}: expected one {description} ({pattern}), found {len(matches)}: {names}"
        )
    return matches[0]


def _serialize_full_elf_xadx(elf, context_xclbin, params, serializer):
    """Converts one internal aiecc ELF into a self-contained XADX."""
    return aiecc_elf_xadx.convert_to_xadx(
        serializer,
        elf.read_bytes(),
        context_xclbin.read_bytes(),
        params.read_text("utf-8"),
    )


def cmd_package_npu_test_package(args):
    """Build execute-only XADX kernels and deterministic numeric cases for NPU CI."""
    libhrx_env = hrx_runtime.add_hrx_install_env(os.environ.copy())
    serializer = hrx_runtime.HrxApi(
        hrx_runtime.load_libhrx(libhrx_env, "nt" if IS_WINDOWS else "posix")
    )

    if NPU_TEST_PACKAGE_STAGE.exists():
        shutil.rmtree(NPU_TEST_PACKAGE_STAGE)
    NPU_TEST_PACKAGE_STAGE.mkdir(parents=True)
    cases = []

    # The on-device single-dispatch gate uses vector_scalar_add, which differs
    # from the matrix_scalar_add compiler/HAL contract gate run earlier.
    design = "vector_scalar_add"
    design_dir = PROJ / "iron-tools" / "programming_examples" / "basic" / design
    artifacts = iron.build_design(
        design_dir, iron.NPU_TARGETS["npu2"]["device"], [], iron.iron_env()
    )
    iron.require_txn_devgen(artifacts.insts, "npu2")
    vector_xclbin = artifacts.xclbin.read_bytes()
    vector_transaction = artifacts.insts.read_bytes()
    cases.append(
        write_case(
            NPU_TEST_PACKAGE_STAGE,
            npu_test_cases.generate_case(design),
            serializer.serialize_xadx(vector_xclbin, vector_transaction, "MLIR_AIE"),
            {"kind": "iron-tools-design", "name": design, "device": "npu2"},
        )
    )
    cases.append(
        write_case(
            NPU_TEST_PACKAGE_STAGE,
            npu_test_cases.generate_case("vector_scalar_add_chain4"),
            serializer.serialize_xadx(
                vector_xclbin,
                entry_name="MLIR_AIE",
                runs=[(vector_transaction, b"")] * 4,
            ),
            {
                "kind": "iron-tools-design",
                "name": design,
                "device": "npu2",
                "run_count": 4,
            },
        )
    )

    for sequence in iron.SEQUENCES:
        build_dir = OUT / "iron-ops" / sequence.name
        full_elf_dir = build_dir / "full-elf"
        context_dir = build_dir / "context"
        elf = _single_artifact(full_elf_dir, "*.elf", "sequence full ELF")
        params = _single_artifact(
            full_elf_dir,
            "**/params.txt",
            "sequence scratchpad parameter metadata",
        )
        context_xclbin = _single_artifact(
            context_dir, "*.xclbin", "sequence context xclbin"
        )
        cases.append(
            write_case(
                NPU_TEST_PACKAGE_STAGE,
                npu_test_cases.generate_case(sequence.name),
                _serialize_full_elf_xadx(
                    elf,
                    context_xclbin,
                    params,
                    serializer,
                ),
                {
                    "kind": "iron-ops-full-elf-sequence",
                    "name": sequence.name,
                    "device": "npu2",
                    "size": sequence.size,
                    "tile_size": sequence.tile_size,
                    "num_aie_columns": sequence.num_aie_columns,
                },
                required_features=(FULL_ELF_LOAD_PDI_FEATURE,),
            )
        )

    for operator in iron.OPERATORS:
        if operator.name not in npu_test_cases.CASE_GENERATORS:
            continue
        build_dir = OUT / "iron-ops" / operator.name
        if operator.full_elf:
            full_elf_dir = build_dir / "full-elf"
            context_dir = build_dir / "context"
            elf = _single_artifact(full_elf_dir, "*.elf", "operator full ELF")
            params = _single_artifact(
                full_elf_dir,
                "**/params.txt",
                "scratchpad parameter metadata",
            )
            context_xclbin = _single_artifact(context_dir, "*.xclbin", "context xclbin")
            xadx = _serialize_full_elf_xadx(
                elf,
                context_xclbin,
                params,
                serializer,
            )
            cases.append(
                write_case(
                    NPU_TEST_PACKAGE_STAGE,
                    npu_test_cases.generate_case(operator.name),
                    xadx,
                    {
                        "kind": "iron-ops-full-elf",
                        "name": operator.name,
                        "device": operator.device,
                        "build_kwargs": operator.build_kwargs,
                    },
                    required_features=(FULL_ELF_LOAD_PDI_FEATURE,),
                )
            )
            continue
        xclbin = _single_artifact(build_dir, "*.xclbin", "operator xclbin")
        insts = _single_artifact(build_dir, "*.bin", "operator transaction")
        iron.require_txn_devgen(insts, operator.device)
        cases.append(
            write_case(
                NPU_TEST_PACKAGE_STAGE,
                npu_test_cases.generate_case(operator.name),
                serializer.serialize_xadx(
                    xclbin.read_bytes(), insts.read_bytes(), "MLIR_AIE"
                ),
                {
                    "kind": "iron-ops",
                    "name": operator.name,
                    "device": operator.device,
                    "build_kwargs": operator.build_kwargs,
                },
            )
        )

    write_manifest(
        NPU_TEST_PACKAGE_STAGE,
        cases,
        _source_revision(),
        platform.system().lower(),
        platform.machine().lower(),
    )
    verify_npu_test_package(NPU_TEST_PACKAGE_STAGE)
    archive = create_archive(
        NPU_TEST_PACKAGE_STAGE,
        npu_prereqs.NPU_PREREQ_DIR,
        npu_prereqs.archive_codec(),
    )
    print(
        f"package-npu-test-package: ok ({len(cases)} XADX cases) -> "
        f"{archive} ({archive.stat().st_size / 1e6:.1f} MB)"
    )


def _source_revision():
    return git_revision(ROOT)


def verify_npu_test_package(package):
    expected_cases = [
        "vector_scalar_add",
        "vector_scalar_add_chain4",
        *(sequence.name for sequence in iron.SEQUENCES),
        *(
            operator.name
            for operator in iron.OPERATORS
            if operator.name in npu_test_cases.CASE_GENERATORS
        ),
    ]
    return verify_tree(
        package, expected_cases, _source_revision(), platform.system().lower()
    )


def cmd_verify_npu_test_package(args):
    if args.extracted:
        package = NPU_TEST_PACKAGE_EXTRACTED
        try:
            manifest = verify_npu_test_package(package)
        except RuntimeError as error:
            sys.exit(str(error))
        print(f"verify-npu-test-package: ok ({len(manifest['cases'])} XADX cases)")
        return

    archive = find_archive(npu_prereqs.NPU_PREREQ_DIR)
    if archive is None:
        sys.exit(
            "no npu-test-package.tar.* under "
            f"{npu_prereqs.NPU_PREREQ_DIR} — "
            "run: python dev.py package-npu-test-package"
        )
    try:
        extract_archive(archive, NPU_TEST_PACKAGE_EXTRACTED)
        manifest = verify_npu_test_package(NPU_TEST_PACKAGE_EXTRACTED)
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"verify-npu-test-package: ok ({len(manifest['cases'])} XADX cases)")


def cmd_unpack_npu_test_package(args):
    archive = find_archive(args.dir)
    if archive is None:
        sys.exit(f"no npu-test-package.tar.* under {args.dir}")
    try:
        extract_archive(archive, NPU_TEST_PACKAGE_EXTRACTED)
    except RuntimeError as error:
        sys.exit(str(error))
    print(
        f"unpack-npu-test-package: ok ({archive.name} -> {NPU_TEST_PACKAGE_EXTRACTED})"
    )
