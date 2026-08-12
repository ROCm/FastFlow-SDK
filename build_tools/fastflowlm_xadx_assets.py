# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qualification-owned shipping XADX asset generation."""

import json
import os
import shutil
from pathlib import Path

from build_tools.dev import environment, paths
from build_tools.dev.build import HRX_INSTALL
from build_tools.fastflowlm_artifact import sha256_file
from build_tools.fastflowlm_kernel_recipe import (
    attest_recipe_generators,
    build_recipe,
    validate_recipe,
)
from build_tools.fastflowlm_qualification_common import (
    CAPABILITIES_SOURCE,
    KERNEL_RECIPE_SCHEMA,
    QUALIFICATION_BUILD,
    XADX_IMPLEMENTATION_PROBE_TARGET,
    XADX_PRODUCT,
    XADX_STAGE,
    configure_qualification_tools,
    file_record,
    load_capabilities,
    recipe_generators,
    run_native_qualification_tests,
    verify_recipe_generator_parity,
)

def build_xadx_generator(fresh=False):
    """Builds the implementation/transaction probe used by recipe generation."""
    configure_qualification_tools(fresh=fresh)
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(QUALIFICATION_BUILD),
            "--target",
            XADX_IMPLEMENTATION_PROBE_TARGET,
            "-j",
            os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4"),
        ],
        env=environment.cmake_env(),
    )
    probe = QUALIFICATION_BUILD / paths.executable_name(
        XADX_IMPLEMENTATION_PROBE_TARGET
    )
    if not probe.is_file():
        raise RuntimeError(f"XADX build did not produce {probe}")
    return probe



def _implementation_catalogue(xclbin_root):
    """Returns one stable record per distinct shipped dequant/MM xclbin."""
    unique = {}
    for directory in sorted(Path(xclbin_root).iterdir()):
        if not directory.is_dir():
            continue
        for kernel in ("dequant", "mm"):
            source = directory / f"{kernel}.xclbin"
            if not source.is_file():
                continue
            digest = sha256_file(source)
            key = (kernel, digest)
            if key not in unique:
                unique[key] = {
                    "id": f"{kernel}-{digest[:16]}",
                    "kernel": kernel,
                    "representative": directory.name,
                    "families": [],
                    "xclbin_bytes": source.stat().st_size,
                    "xclbin_sha256": digest,
                    "xclbin_path": source.relative_to(xclbin_root).as_posix(),
                }
            unique[key]["families"].append(directory.name)
    records = sorted(unique.values(), key=lambda record: record["id"])
    if len({record["id"] for record in records}) != len(records):
        raise RuntimeError("source xclbin SHA-256 prefixes collide")
    return records


def _recipe_materializer():
    generators = recipe_generators()

    def materialize(specification, expected_size):
        data = generators.materialize(
            specification["id"], specification["parameters"]
        )
        if len(data) != expected_size:
            raise RuntimeError(
                f"{specification['id']} produced {len(data)} bytes; "
                f"expected {expected_size}"
            )
        return data

    return materialize


def verify_kernel_recipe_assets(root):
    """Rejects incomplete, corrupted, or semantically invalid recipe assets."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    recipe_path = root / "recipe.json"
    if not manifest_path.is_file() or not recipe_path.is_file():
        raise RuntimeError("kernel recipe asset manifest or recipe is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("product") != XADX_PRODUCT
        or manifest.get("target") != "npu2"
        or manifest.get("serializer")
        != {
            "api": "hrx_amdxdna_xadx_serialize",
            "revision": "hrx-amdxdna-v0",
        }
        or manifest.get("recipe")
        != file_record(root, recipe_path)
    ):
        raise RuntimeError("invalid kernel recipe asset manifest")
    validate_recipe(recipe, KERNEL_RECIPE_SCHEMA)
    expected_files = {
        record["path"]: record for record in manifest.get("files", [])
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(expected_files) != actual_files:
        raise RuntimeError(
            "kernel recipe asset file inventory differs: "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    for relative, record in expected_files.items():
        path = root / relative
        if file_record(root, path) != record:
            raise RuntimeError(f"kernel recipe asset is corrupted: {relative}")
    for implementation in recipe["implementations"]:
        executions = [
            (None, implementation),
            *[
                (variant["id"], variant)
                for variant in implementation.get("execution_variants", ())
            ],
        ]
        for execution, executable in executions:
            xadx_path = root / executable["xadx"]
            transaction_path = root / executable["transaction"]
            if sha256_file(xadx_path) != executable["xadx_sha256"]:
                suffix = "" if execution is None else f"/{execution}"
                raise RuntimeError(
                    "recipe XADX digest differs: "
                    f"{implementation['id']}{suffix}"
                )
            if (
                sha256_file(transaction_path)
                != executable["transaction_sha256"]
            ):
                suffix = "" if execution is None else f"/{execution}"
                raise RuntimeError(
                    "recipe transaction digest differs: "
                    f"{implementation['id']}{suffix}"
                )
    return manifest, recipe


def generate_xadx_assets(xclbin_root, destination=XADX_STAGE):
    """Builds the attested implementation-indexed kernel recipe asset."""
    destination = Path(destination)
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    probe = QUALIFICATION_BUILD / paths.executable_name(
        XADX_IMPLEMENTATION_PROBE_TARGET
    )
    if not probe.is_file():
        build_xadx_generator(False)
    if not probe.is_file():
        raise RuntimeError(f"missing XADX implementation probe: {probe}")

    capabilities = load_capabilities()
    catalogue = _implementation_catalogue(xclbin_root)
    records = {record["id"]: record for record in catalogue}
    measured = capabilities["implementations"]["entries"]
    missing = set(measured) - set(records)
    unknown_limits = set(capabilities["implementations"]["unreached"]) - set(records)
    if missing or unknown_limits:
        raise RuntimeError(
            "measured implementation identities do not match shipped xclbins: "
            f"missing={sorted(missing)}, limits={sorted(unknown_limits)}"
        )

    env = environment.cmake_env()
    search_key = "PATH" if paths.is_windows() else "LD_LIBRARY_PATH"
    # FLM's prebuilt engine libs (dequant/gemm/q4_npu_eXpress) live under the
    # per-backend lib/hrx, while the flat lib/ carries their third-party runtime
    # deps, so both belong on the loader path.
    search_dirs = [
        paths.PROJ / "FastFlowLM" / "src" / "lib" / "hrx",
        paths.PROJ / "FastFlowLM" / "src" / "lib",
        HRX_INSTALL / ("bin" if paths.is_windows() else "lib"),
    ]
    entries = [str(path) for path in search_dirs if path.is_dir()]
    if env.get(search_key):
        entries.append(env[search_key])
    env[search_key] = os.pathsep.join(entries)

    probe_root = destination / ".probe"
    xadx_root = destination / "xadx"
    transaction_root = destination / "transactions"
    xadx_root.mkdir()
    transaction_root.mkdir()
    runs = {}

    def executions(facts):
        return [
            {
                "id": None,
                "builder": facts["builder"],
                "shape": facts["shape"],
            },
            *facts.get("execution_variants", ()),
        ]

    for id, facts in measured.items():
        for execution in executions(facts):
            shape = (
                capabilities["shape"]
                if execution["shape"] == "production"
                else capabilities["implementations"]["wide_shape"]
            )
            key = (
                execution["builder"],
                shape["m"],
                shape["k"],
                shape["n"],
            )
            if key in runs:
                continue
            environment.sh(
                [
                    str(probe),
                    str(xclbin_root),
                    str(probe_root),
                    execution["builder"],
                    f"{shape['m']},{shape['k']},{shape['n']}",
                ],
                env=env,
            )
            variant = (
                f"{execution['builder']}-"
                f"{shape['m']}x{shape['k']}x{shape['n']}"
            )
            index = json.loads(
                (probe_root / f"index-{variant}.json").read_text(encoding="utf-8")
            )
            runs[key] = {
                record["representative"]: record for record in index
            }

    digests = {}
    for id, facts in measured.items():
        catalogue_record = records[id]
        digests[id] = {
            "xclbin_sha256": catalogue_record["xclbin_sha256"],
            "xclbin_path": catalogue_record["xclbin_path"],
            "execution_variants": {},
        }
        for execution in executions(facts):
            shape = (
                capabilities["shape"]
                if execution["shape"] == "production"
                else capabilities["implementations"]["wide_shape"]
            )
            key = (
                execution["builder"],
                shape["m"],
                shape["k"],
                shape["n"],
            )
            try:
                probe_record = runs[key][catalogue_record["representative"]]
            except KeyError as error:
                raise RuntimeError(
                    f"{id} did not accept measured builder/shape {key}"
                ) from error
            source_xadx = probe_root / probe_record["xadx"]
            suffix = (
                "" if execution["id"] is None else f"-{execution['id']}"
            )
            packaged_xadx = xadx_root / f"{id}{suffix}.xadx"
            shutil.copy2(source_xadx, packaged_xadx)
            transaction = probe_root / probe_record["transaction"]
            transaction_sha256 = sha256_file(transaction)
            packaged_transaction = (
                transaction_root / f"{transaction_sha256}.bin"
            )
            if not packaged_transaction.exists():
                shutil.copy2(transaction, packaged_transaction)
            identity = {
                "transaction": packaged_transaction.relative_to(
                    destination
                ).as_posix(),
                "transaction_sha256": transaction_sha256,
                "xadx": packaged_xadx.relative_to(destination).as_posix(),
                "xadx_sha256": sha256_file(packaged_xadx),
            }
            if execution["id"] is None:
                digests[id].update(identity)
            else:
                digests[id]["execution_variants"][execution["id"]] = identity

    recipe = build_recipe(capabilities, catalogue, digests)
    attest_recipe_generators(recipe, _recipe_materializer())
    validate_recipe(recipe, KERNEL_RECIPE_SCHEMA)
    recipe_path = destination / "recipe.json"
    recipe_path.write_bytes(
        (json.dumps(recipe, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    capability_path = destination / "capability-manifest.json"
    shutil.copy2(CAPABILITIES_SOURCE, capability_path)
    shutil.rmtree(probe_root)
    files = [
        file_record(destination, path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "schema_version": 1,
        "product": XADX_PRODUCT,
        "target": "npu2",
        "serializer": {
            "api": "hrx_amdxdna_xadx_serialize",
            "revision": "hrx-amdxdna-v0",
        },
        "recipe": file_record(destination, recipe_path),
        "capabilities": file_record(destination, capability_path),
        "files": files,
    }
    (destination / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    verify_kernel_recipe_assets(destination)
    return destination


def cmd_generate_xadx_assets(args):
    build_xadx_generator(args.fresh)
    run_native_qualification_tests(configure=False)
    verify_recipe_generator_parity(configure=False)
    return generate_xadx_assets(
        paths.PROJ / "FastFlowLM" / "src" / "xclbins",
        destination=args.dir,
    )
