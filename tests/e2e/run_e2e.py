#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""rocm-npu-staging e2e BUILD test (no NPU run).

Proves the chain:

    IRON emit -> Peano -> bootgen PDI -> hrx-xclbinutil(.xclbin) -> [HAL extracts PDI]

Runs on Linux and Windows. Exits non-zero on any failure.

The design and its -d device default to `[iron_tools] e2e_design`/`e2e_device` in
manifests/versions.toml. DESIGN overrides it: a bare name resolves beside the manifest's
design, a path is taken relative to projects/iron-tools. It must be Peano-capable
and use makefile-common's `jit_xclbin` rule -- `jit_xclbin_elf` designs also emit
insts.elf, which needs aiebu-asm (absent from this stack).

We invoke the design's @iron.jit script directly rather than `make`. That IS the
jit_xclbin recipe (`python3 <design>.py -d <dev> --xclbin-path=... --insts-path=...`),
it keeps the full chain, and it drops make -- which Git Bash does not ship. Designs
whose Makefile passes extra `aieargs` need them after `--`.

Must be run BY the stack's venv python (`dev.py test` does this): every tool location is
derived from `sys.executable`, so there is no venv layout to guess at.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from build_tools import iron

ROOT = iron.ROOT
PROJ = iron.PROJ
OUT = iron.OUT
EXE = iron.EXE
E2EError = iron.E2EError
LOG_TAIL_CHARS = iron.LOG_TAIL_CHARS
build_design = iron.build_design
check_sections = iron.check_sections
cmake_exe = iron.cmake_exe
cmake_path = iron.cmake_path
compiler_prefix_from_env = iron.compiler_prefix_from_env
installed_aie_python_root = iron.installed_aie_python_root
read_e2e_pin = iron.read_e2e_pin
require_xclbinutil = iron.require_xclbinutil
resolve_design_dir = iron.resolve_design_dir
run_process = iron.run_process
toolchain_env = iron.toolchain_env


def build_loader(env: dict[str, str]) -> Path:
    """Step 3a: build hal_pdi_loader against the HAL. CMake owns the link line (archive
    naming, the group flags the circular iree base deps need, system libs)."""
    hrx = PROJ / "hrx"
    hrx_build = hrx / "build" / "cmake"
    hal_dir = hrx_build / "runtime" / "src" / "iree" / "hal" / "drivers" / "amdxdna"
    if not hal_dir.is_dir():
        raise E2EError(
            "projects/hrx not built (run: python dev.py build hrx -- or all)"
        )

    build_dir = OUT / "hal-contract"
    # hal_pdi_loader is a C target; honour $CC when the caller pinned one (the Windows
    # CI pins clang-cl) and otherwise let CMake pick, as dev.py's own builds do.
    cc = os.environ.get("CC")
    configure = [
        cmake_exe(),
        "-G",
        "Ninja",
        "-S",
        cmake_path(HERE.parent / "hal-contract"),
        "-B",
        cmake_path(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        *([f"-DCMAKE_C_COMPILER={cc}"] if cc else []),
        f"-DHRX_SRC={cmake_path(hrx)}",
        f"-DHRX_BUILD={cmake_path(hrx_build)}",
    ]
    log = OUT / "e2e-loader-build.log"
    with log.open("wb") as log_file:
        completed = run_process(
            configure, env=env, stdout=log_file, stderr=subprocess.STDOUT
        )
        if completed.returncode == 0:
            completed = run_process(
                [cmake_exe(), "--build", cmake_path(build_dir)],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
    if completed.returncode != 0:
        sys.stderr.write(log.read_text(errors="replace")[-LOG_TAIL_CHARS:])
        raise E2EError("HAL PDI loader build failed")

    loader = build_dir / f"hal_pdi_loader{EXE}"
    if not loader.exists():
        raise E2EError(f"loader not produced ({loader})")
    return loader


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rocm-npu-staging e2e build test")
    parser.add_argument(
        "design",
        nargs="?",
        help="design to build: a bare name resolves beside the manifest's "
        "e2e_design, a path is relative to projects/iron-tools "
        "(default: the manifest's e2e_design)",
    )
    parser.add_argument(
        "design_args",
        nargs="*",
        help="extra aieargs for the design script, passed after `--`",
    )
    parser.add_argument(
        "--compiler-prefix",
        type=Path,
        default=compiler_prefix_from_env(),
        help="relocated rocm-npu compiler product prefix to use instead of build trees",
    )
    args = parser.parse_args(argv)
    if args.design and args.design.startswith("-"):
        parser.error("DESIGN must precede `--`; extra design args go after it")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        pin = read_e2e_pin()
        design_dir = resolve_design_dir(pin, args.design)

        OUT.mkdir(exist_ok=True)
        compiler_prefix = (
            args.compiler_prefix.resolve() if args.compiler_prefix else None
        )
        xclbinutil = require_xclbinutil(compiler_prefix)
        env = toolchain_env(xclbinutil, compiler_prefix)
        artifacts = build_design(
            design_dir=design_dir,
            device=pin.device,
            design_args=args.design_args,
            env=env,
        )
        check_sections(xclbinutil=xclbinutil, xclbin=artifacts.xclbin, env=env)

        print("== HAL artifact-contract check ==")
        loader = build_loader(env)
        extracted = run_process([loader, artifacts.xclbin, artifacts.pdi], env=env)
        if extracted.returncode != 0:
            raise E2EError("HAL did not extract matching PDI")
    except E2EError as error:
        print(f"E2E FAIL: {error}", file=sys.stderr)
        return 1

    print(f"E2E PASS: {design_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
