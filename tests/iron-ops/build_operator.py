#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""iron-ops BUILD gate (no NPU run): build every operator in the suite.

Proves the chain the operator library actually walks:

    IRON operator -> MLIR -> Peano (the operator's C++ kernel AND the cores)
                  -> bootgen PDI -> hrx-xclbinutil(.xclbin) + insts.bin

`run_e2e.py` proves the same toolchain against an iron-tools programming example. This
proves it against projects/iron-ops, and the difference is the point: iron-ops is developed
against the mlir_aie WHEEL (v1.3.4), while this stack builds the aie package from SOURCE at
a nightly pin. When the two drift, an operator stops compiling -- and it should say so here,
on a CPU runner, not an hour later on the one machine that has an NPU. This also compiles
the operator's own aie_kernels/*.cc through Peano, which the examples' @iron.jit path never
does.

Which operators, at what size: tests/iron-ops/suite.py. Runs on Linux and Windows.

Must be run BY the stack's venv python, WITH the IRON toolchain env (PYTHONPATH ->
build/iron-tools/python, PEANO_INSTALL_DIR, tools-bin on PATH). `dev.py test-iron-ops` does
both; it takes that env from run_e2e.toolchain_env(), the single definition.
"""

import argparse
import importlib
import inspect
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

sys.path[:0] = [str(HERE), str(ROOT / "tests" / "e2e"), str(ROOT)]
import dev  # noqa: E402   (TXN DevGen check + the NPU generation table)
import run_e2e  # noqa: E402   (toolchain paths + the xclbin section check)
import suite  # noqa: E402   (the operator table)

# This gate must never open an NPU: it runs on the CPU build runners, and the device it
# targets is a declared input, not something to detect. IRON_RUNTIME=cpu makes that
# structural -- aie.utils then has no default NPU runtime at all, so a stray device access
# fails loudly here rather than quietly working on the one machine that has one. It also
# makes Linux (where `build hrx` leaves a libhrx that IRON_RUNTIME=auto would select) behave
# exactly like Windows (where there is none).
os.environ.setdefault("IRON_RUNTIME", "cpu")

import aie.utils as aie_utils  # noqa: E402   (must follow IRON_RUNTIME)
from aie.iron.device import NPU1Col1, NPU2  # noqa: E402
from iron.common import AIEContext, AIEOperatorBase  # noqa: E402
from iron.common.compilation import InstsBinArtifact, XclbinArtifact  # noqa: E402

DEVICES = {"npu2": NPU2, "npu1": NPU1Col1}

class BuildError(RuntimeError):
    """A fatal failure. main() turns this into a non-zero exit."""


def find_operator_class(name: str) -> type:
    """The AIEOperatorBase subclass defined in iron.operators.<name>.op.

    Resolved by inspection rather than by a name table: the directory-to-class mapping is
    not mechanical (rms_norm -> RMSNorm, mha -> MHA, rope -> RoPE), and a table here would
    be one more thing to keep in step with the submodule.
    """
    try:
        module = importlib.import_module(f"iron.operators.{name}.op")
    except ModuleNotFoundError as e:
        raise BuildError(f"{name}: no such operator in iron.operators ({e})") from e

    found = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and issubclass(value, AIEOperatorBase)
        and value.__module__ == module.__name__
    ]
    if len(found) != 1:
        names = ", ".join(sorted(c.__name__ for c in found)) or "none"
        raise BuildError(
            f"{name}: expected exactly one operator class in {module.__name__}, found "
            f"{len(found)} ({names}). Teach this driver which one to build."
        )
    return found[0]


def instantiate(
    cls: type, operator_spec: suite.Operator, context: AIEContext
) -> AIEOperatorBase:
    """The operator shape declared in suite.py."""
    params = inspect.signature(cls.__init__).parameters
    unknown = [p for p in operator_spec.build_kwargs if p not in params]
    if unknown:
        takes = ", ".join(p for p in params if p != "self")
        raise BuildError(
            f"{operator_spec.name}: suite.py passes unsupported constructor arg(s) "
            f"{', '.join(unknown)} to {cls.__name__}; it takes ({takes})."
        )
    return cls(**operator_spec.build_kwargs, context=context)


def artifact_of(operator: AIEOperatorBase, kind: type) -> Path:
    """The single compiled artifact of type `kind`.

    By artifact TYPE, not by filename: IRON names the instruction stream
    `<Operator>_sz..._npu2.bin`, so matching on an `insts.bin` suffix silently finds nothing.
    """
    found = [Path(a.filename) for a in operator.artifacts if isinstance(a, kind)]
    if len(found) != 1:
        got = ", ".join(str(f) for f in found) or "none"
        raise BuildError(f"expected exactly one {kind.__name__}, got {len(found)}: {got}")
    return found[0]


def build(operator_spec: suite.Operator, device: str, build_root: Path) -> None:
    """Build one operator and check what came out. Raises BuildError on any failure."""
    build_dir = build_root / operator_spec.name
    # Start from nothing: a leftover xclbin from an earlier run must never be what we
    # validate. IRON's artifact graph skips any rule whose output is already available.
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    # Bind the device explicitly. Everything downstream keys off it -- which aie_kernels
    # subdir the .cc comes from, the core Peano targets, the DevGen in the emitted TXN --
    # and with no NPU present there is nothing to infer it from.
    aie_utils.set_current_device(DEVICES[device]())
    shape = ", ".join(f"{k}={v}" for k, v in operator_spec.build_kwargs.items())
    print(f"\n== iron-ops: {operator_spec.name} ({shape}, {device}) ==")

    operator = instantiate(
        find_operator_class(operator_spec.name),
        operator_spec,
        AIEContext(build_dir=build_dir),
    )
    operator.compile()

    xclbin = artifact_of(operator, XclbinArtifact)
    insts = artifact_of(operator, InstsBinArtifact)
    for produced in (xclbin, insts):
        if not produced.is_file():
            raise BuildError(f"{produced} was not produced by the build")
        print(f"  ok: {produced.name} ({produced.stat().st_size} bytes)")

    # The same two checks run_e2e makes of an iron-tools design: hrx-xclbinutil really
    # packaged the NPU sections, and the instruction stream targets the device we asked for
    # (a mismatched DevGen is accepted by the driver and then never completes).
    run_e2e.check_sections(
        run_e2e.require_xclbinutil(run_e2e.compiler_prefix_from_env()),
        xclbin,
    )
    dev._require_txn_devgen(insts, device)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator", help="build only this one (default: every operator in the suite)"
    )
    parser.add_argument(
        "--device",
        choices=sorted(DEVICES),
        help="override the NPU generation the suite pins for each operator",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "out" / "iron-ops",
        help="where the artifacts land (default: out/iron-ops)",
    )
    args = parser.parse_args(argv)

    try:
        operators = suite.select(args.operator)
        for operator_spec in operators:
            build(operator_spec, args.device or operator_spec.device, args.build_dir)
    except (BuildError, run_e2e.E2EError) as e:
        print(f"IRON-OPS BUILD FAIL: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        # KeyError stringifies to its repr ("'...'"); the message is the argument.
        print(f"IRON-OPS BUILD FAIL: {e.args[0]}", file=sys.stderr)
        return 1

    names = ", ".join(op.name for op in operators)
    print(f"\niron-ops build: PASS ({len(operators)} operator(s): {names})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
