#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""iron-ops BUILD gate (no NPU run): build every operator in the suite.

Proves the chain the operator library actually walks:

    IRON operator -> MLIR -> Peano (the operator's C++ kernel AND the cores)
                  -> bootgen PDI -> hrx-xclbinutil(.xclbin) + insts.bin
                     or an iron.Graph -> aiebu-asm full ELF + parameter table

`run_e2e.py` proves the same toolchain against an iron-tools programming example. This
proves it against projects/iron-ops, and the difference is the point: iron-ops is developed
against the mlir_aie wheel, while this stack builds the aie package from SOURCE at the
[iron_tools] pin. When the two drift, an operator stops compiling -- and it should say so
here, on a CPU runner, not an hour later on the one machine that has an NPU. Kernel sources
come from mlir-aie's aie_kernels, which the examples' @iron.jit path never compiles on its own.

Which operators, at what size: build_tools/iron.py. Runs on Linux and Windows.

Must be run BY the stack's venv python, WITH the IRON toolchain env (PYTHONPATH ->
projects/iron-ops plus the selected compiler's Python package, PEANO_INSTALL_DIR,
tools-bin on PATH). `dev.py test-iron-ops` provides both layers.
"""

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

sys.path.insert(0, str(ROOT))
from build_tools import iron

# This gate must never open an NPU: it runs on the CPU build runners, and the device it
# targets is a declared input, not something to detect. IRON_RUNTIME=cpu makes that
# structural -- aie.utils then has no default NPU runtime at all, so a stray device access
# fails loudly here rather than quietly working on the one machine that has one. It also
# makes Linux (where `build hrx` leaves a libhrx that IRON_RUNTIME=auto would select) behave
# exactly like Windows (where there is none).
os.environ.setdefault("IRON_RUNTIME", "cpu")

DEVICES = ("npu1", "npu2")
# mlir-aie's compile cache, under the build dir. It is read when aie.utils.compile is
# first imported, so it is set before any IRON import.
CACHE = ".npu-cache"


class BuildError(RuntimeError):
    """A fatal failure. main() turns this into a non-zero exit."""


def device_for(name: str):
    from aie.iron.device import NPU2, NPU1Col1

    return {"npu2": NPU2, "npu1": NPU1Col1}[name]()


def check_runtime_images(staged: list[Path], kind: str, device: str) -> None:
    """Require a full ELF, xclbin+insts, or xclbin+dispatch library."""
    try:
        for path in staged:
            if kind == "dispatch" and path is staged[1]:
                iron.require_host_library(path)
            else:
                iron.require_runtime_image(path, elf=(kind == "elf"))
            print(f"  ok: {path.name} ({path.stat().st_size} bytes)")
    except RuntimeError as error:
        raise BuildError(str(error)) from error
    if kind in ("xclbin", "dispatch"):
        iron.check_sections(
            iron.require_xclbinutil(iron.compiler_prefix_from_env()),
            staged[0],
        )
    if kind == "xclbin":
        iron.require_txn_devgen(staged[1], device)


def fresh_dir(build_dir: Path) -> Path:
    # Start from nothing: a leftover image from an earlier run must never be what we
    # validate.
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    return build_dir


def stage(compiled: iron.Compiled, build_dir: Path, device: str) -> None:
    try:
        kind, staged = iron.stage_compiled_images(compiled, build_dir)
    except RuntimeError as error:
        raise BuildError(str(error)) from error
    check_runtime_images(staged, kind, device)


def build(operator_spec: iron.Operator, device: str, build_root: Path) -> None:
    """Build one operator and check what came out. Raises BuildError on any failure."""
    build_dir = fresh_dir(build_root / operator_spec.name)
    shape = ", ".join(f"{k}={v}" for k, v in operator_spec.build_kwargs.items())
    print(f"\n== iron-ops: {operator_spec.name} ({shape}, {device}) ==")
    try:
        compiled = iron.compile_operator(operator_spec, device_for(device))
    except (ImportError, RuntimeError) as error:
        raise BuildError(f"{operator_spec.name}: {error}") from error
    stage(compiled, build_dir, device)


def build_sequence(sequence_spec: iron.Sequence, device: str, build_root: Path) -> None:
    """Build one fused add-to-ReLU sequence to a full ELF."""
    build_dir = fresh_dir(build_root / sequence_spec.name)
    print(
        f"\n== iron-ops: {sequence_spec.name} "
        f"(size={sequence_spec.size}, tile_size={sequence_spec.tile_size}, "
        f"num_aie_columns={sequence_spec.num_aie_columns}, {device}) =="
    )
    stage(iron.compile_sequence(sequence_spec, device_for(device)), build_dir, device)


def _compile_one_subprocess(
    name: str, device: str | None, build_dir: Path
) -> tuple[str, subprocess.CompletedProcess]:
    """One operator in its own process: aie.utils device state is process-global."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--operator",
        name,
        "--build-dir",
        str(build_dir),
        "--jobs",
        "1",
        "--keep-cache",
    ]
    if device:
        command += ["--device", device]
    return name, subprocess.run(command, capture_output=True, text=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator", help="build only this one (default: every operator in the suite)"
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        help="override the NPU generation the suite pins for each operator",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "out" / "iron-ops",
        help="where the artifacts land (default: out/iron-ops)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel operator compiles (0 = min(8, CPUs); 1 = this process)",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help=argparse.SUPPRESS,  # a parallel run's children share its fresh cache
    )
    args = parser.parse_args(argv)

    # A cache hit would skip the very compile this gate exists to run.
    cache = args.build_dir / CACHE
    if not args.keep_cache and cache.exists():
        shutil.rmtree(cache)
    os.environ["NPU_CACHE_HOME"] = str(cache)

    try:
        if args.operator:
            sequences = tuple(
                sequence
                for sequence in iron.SEQUENCES
                if sequence.name == args.operator
            )
            if sequences:
                selected = ()
                operators = ()
            else:
                selected = iron.select(args.operator)
                operators = iron.host_compile_suite(selected)
        else:
            sequences = iron.SEQUENCES
            selected = iron.select(None)
            operators = iron.host_compile_suite(selected)
        for skipped in selected:
            if not skipped.host_compile:
                why = (
                    "local test-npu-iron-ops still collects this operator"
                    if skipped.hrx_pytest_case
                    else "not built on this stack"
                )
                print(f"{skipped.name}: skipped host compile ({why})")
        if not operators and not sequences:
            print("\niron-ops build: PASS (0 host-compile items)")
            return 0
        items = [
            *(("operator", operator.name, operator.device) for operator in operators),
            *(("sequence", sequence.name, "npu2") for sequence in sequences),
        ]
        jobs = iron.compile_job_count(len(items), args.jobs)
        if jobs == 1:
            for operator_spec in operators:
                build(
                    operator_spec,
                    args.device or operator_spec.device,
                    args.build_dir,
                )
            for sequence_spec in sequences:
                build_sequence(
                    sequence_spec,
                    args.device or "npu2",
                    args.build_dir,
                )
        else:
            print(f"iron-ops build: {len(items)} items, jobs={jobs}")
            failures = []
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = [
                    pool.submit(
                        _compile_one_subprocess,
                        name,
                        args.device or device,
                        args.build_dir,
                    )
                    for _, name, device in items
                ]
                for future in as_completed(futures):
                    name, completed = future.result()
                    if completed.stdout:
                        sys.stdout.write(completed.stdout)
                    if completed.stderr:
                        sys.stderr.write(completed.stderr)
                    if completed.returncode != 0:
                        failures.append(name)
            if failures:
                print(
                    f"IRON-OPS BUILD FAIL: {', '.join(failures)}",
                    file=sys.stderr,
                )
                return 1
    except (BuildError, iron.E2EError) as e:
        print(f"IRON-OPS BUILD FAIL: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        # KeyError stringifies to its repr ("'...'"); the message is the argument.
        print(f"IRON-OPS BUILD FAIL: {e.args[0]}", file=sys.stderr)
        return 1

    names = ", ".join(name for _, name, _ in items)
    print(f"\niron-ops build: PASS ({len(items)} item(s): {names})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
