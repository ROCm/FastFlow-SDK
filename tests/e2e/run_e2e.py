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
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROJ = ROOT / "projects"
OUT = ROOT / "out"

EXE = ".exe" if os.name == "nt" else ""  # the one irreducible platform difference
VENV_BIN = Path(sys.executable).parent  # .venv/bin or .venv/Scripts
SITE = Path(sysconfig.get_paths()["purelib"])  # this interpreter's site-packages

# Sections hrx-xclbinutil must have packaged into the .xclbin.
REQUIRED_SECTIONS = ("MEM_TOPOLOGY", "AIE_PARTITION", "EMBEDDED_METADATA")
# How much of a failed sub-build's log to echo before giving up.
LOG_TAIL_CHARS = 2000


class E2EError(RuntimeError):
    """A fatal e2e failure. main() turns this into a non-zero exit."""


class E2EPin(NamedTuple):
    """The design under test, as pinned by the manifest.

    Attributes:
        design: design directory, relative to projects/iron-tools
        device: value passed to the design script's -d flag
    """

    design: Path
    device: str


class DesignArtifacts(NamedTuple):
    """What building a design produces.

    Attributes:
        xclbin: the packaged .xclbin
        insts: the raw XAie transaction consumed by the host runtime
        pdi: bootgen's main.pdi, which the HAL must extract byte-for-byte
    """

    xclbin: Path
    insts: Path
    pdi: Path


def run_process(cmd: list[str | Path], **kwargs: object) -> subprocess.CompletedProcess:
    """Run cmd, streaming output unless redirected. Never raises; callers check."""
    return subprocess.run([str(c) for c in cmd], **kwargs)


def cmake_path(path: Path) -> str:
    """CMake parses forward slashes on every platform.

    CMake re-emits some cache values into generated CMake code, where a Windows
    backslash reads as an escape. as_posix() keeps the drive letter and flips the
    separators.
    """
    return path.as_posix()


def cmake_candidate() -> Path:
    override = os.environ.get("ROCM_NPU_CMAKE")
    return Path(override) if override else VENV_BIN / f"cmake{EXE}"


def cmake_exe() -> Path:
    """The explicitly supplied test CMake, or the one beside this interpreter."""
    cmake = cmake_candidate()
    if not cmake.exists():
        raise E2EError(f"cmake not in the venv (run: python dev.py setup)\n  {cmake}")
    return cmake


def read_e2e_pin() -> E2EPin:
    """Read the design under test from the manifest, the pin authority.

    Reading it here rather than duplicating the values keeps `[iron_tools] e2e_design`
    load-bearing instead of a comment that rots.
    """
    manifest = tomllib.loads((ROOT / "manifests" / "versions.toml").read_text())
    iron_tools = manifest.get("iron_tools", {})
    design = iron_tools.get("e2e_design")
    device = iron_tools.get("e2e_device")
    if not design or not device:
        raise E2EError(
            "manifests/versions.toml [iron_tools] needs e2e_design and e2e_device"
        )
    return E2EPin(design=Path(design), device=device)


def resolve_design_dir(pin: E2EPin, design: str | None) -> Path:
    """A bare name resolves beside the manifest's design; a path is used as given."""
    relative = pin.design
    if design is not None:
        candidate = Path(design)
        relative = candidate if len(candidate.parts) > 1 else pin.design.parent / design
    return PROJ / "iron-tools" / relative


def compiler_prefix_from_env() -> Path | None:
    value = os.environ.get("ROCM_NPU_COMPILER_PREFIX")
    return Path(value).resolve() if value else None


def installed_aie_python_root() -> Path | None:
    """Find the Python root containing an installed or PYTHONPATH-provided aie."""
    spec = importlib.util.find_spec("aie")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))).resolve().parent


def require_xclbinutil(compiler_prefix: Path | None = None) -> Path:
    """aiecc invokes plain `xclbinutil`; `dev.py build hrx-xclbinutil` publishes ours
    under that name in tools-bin/. Depend on it rather than re-publish it here: the
    test should not mutate shared state, and duplicating the symlink-vs-copy logic
    invites drift.
    """
    xclbinutil = ((compiler_prefix / "bin" / f"xclbinutil{EXE}")
                  if compiler_prefix else
                  ROOT / "tools-bin" / f"xclbinutil{EXE}")
    if not xclbinutil.exists():
        raise E2EError(
            "xclbinutil not published (run: python dev.py build hrx-xclbinutil)\n"
            f"  {xclbinutil}"
        )
    return xclbinutil


def toolchain_env(xclbinutil: Path,
                  compiler_prefix: Path | None = None) -> dict[str, str]:
    """IRON toolchain env from either build trees or a relocated rocm-npu-compiler product."""
    if compiler_prefix:
        aie_bin = compiler_prefix / "bin"
        product_aie_python = compiler_prefix / "python"
        aie_python = (
            product_aie_python
            if (product_aie_python / "aie").is_dir()
            else installed_aie_python_root() or product_aie_python
        )
        peano = compiler_prefix / "peano" / "llvm-aie"
        aieopt_dir = compiler_prefix / "third_party" / "aie_api"
    else:
        aie_bin = ROOT / "build" / "iron-tools" / "bin"
        aie_python = ROOT / "build" / "iron-tools" / "python"
        peano = SITE / "llvm-aie"
        aieopt_dir = PROJ / "iron-tools" / "third_party" / "aie_api"
    if not (aie_bin / f"aie-opt{EXE}").exists():
        raise E2EError(
            "iron-tools not built from source (run: python dev.py build iron-tools)"
        )
    if not (aie_python / "aie" / "compiler" / "aiecc" / "main.py").exists():
        raise E2EError(
            "iron-tools python bindings not built (run: python dev.py build iron-tools)"
        )
    if not peano.is_dir():
        raise E2EError(f"Peano not available to this toolchain\n  {peano}")
    if not aieopt_dir.is_dir():
        raise E2EError(f"aie_api not available to this toolchain\n  {aieopt_dir}")

    env = dict(os.environ)
    path_entries = [xclbinutil.parent, aie_bin, VENV_BIN, cmake_candidate().parent]
    if compiler_prefix and os.name == "nt":
        path_entries.append(compiler_prefix / "lib")
    env["PATH"] = os.pathsep.join([*(str(entry) for entry in path_entries),
                                   env.get("PATH", "")])
    if compiler_prefix and os.name != "nt":
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(compiler_prefix / "lib"), env.get("LD_LIBRARY_PATH", "")]
        )
    env["PYTHONPATH"] = str(aie_python)
    env["PEANO_INSTALL_DIR"] = str(peano)
    env["AIEOPT_DIR"] = str(aieopt_dir)
    if compiler_prefix:
        # A verified product is immutable input. Importing its Python modules
        # must not add unmanifested __pycache__ files to the product tree.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["ROCM_NPU_COMPILER_PREFIX"] = str(compiler_prefix)
        print(f"== toolchain: relocated rocm-npu compiler product ({compiler_prefix}) ==")
    else:
        print(
            "== toolchain: SOURCE iron-tools (built against ROCm amd-llvm) + Peano wheel =="
        )
    return env


def build_design(
    design_dir: Path,
    device: str,
    design_args: list[str],
    env: dict[str, str],
) -> DesignArtifacts:
    """Step 1: the design's @iron.jit compile() -> build/final.xclbin, plus the aiecc
    project dir build/final.prj/ holding bootgen's main.pdi (used by step 3)."""
    if not design_dir.is_dir():
        raise E2EError(f"no design {design_dir}")
    script = design_dir / f"{design_dir.name}.py"
    if not script.exists():
        raise E2EError(f"design has no {script.name} (not an @iron.jit design?)")

    # Start from nothing: a leftover final.xclbin/final.prj from an earlier run must
    # never be what we validate. @iron.jit does rebuild unconditionally today, but the
    # test should not depend on that staying true.
    build_dir = design_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    print(f"== building {design_dir.name} ==")
    log = OUT / "e2e-design.log"
    with log.open("wb") as log_file:
        completed = run_process(
            [
                sys.executable,
                script.name,
                "-d",
                device,
                "--xclbin-path=build/final.xclbin",
                "--insts-path=build/insts.bin",
                *design_args,
            ],
            cwd=design_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        sys.stderr.write(log.read_text(errors="replace")[-LOG_TAIL_CHARS:])
        raise E2EError("design build failed")

    xclbin = build_dir / "final.xclbin"
    insts = build_dir / "insts.bin"
    pdi = build_dir / "final.prj" / "main.pdi"
    if not xclbin.is_file():
        raise E2EError("no .xclbin produced")
    if not insts.is_file():
        raise E2EError("no instruction transaction produced")
    if not pdi.is_file():
        raise E2EError(f"no bootgen PDI produced ({pdi})")

    artifacts_dir = OUT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xclbin, artifacts_dir / f"{design_dir.name}.xclbin")
    print(f"  ok: {design_dir.name}.xclbin ({xclbin.stat().st_size} B)")
    return DesignArtifacts(xclbin=xclbin, insts=insts, pdi=pdi)


def check_sections(
    xclbinutil: Path,
    xclbin: Path,
    env: dict[str, str] | None = None,
) -> None:
    """Step 2: the xclbin carries the NPU sections hrx-xclbinutil packaged.

    xclbinutil is passed absolute: on Windows CreateProcess resolves a bare program name
    against the PARENT's PATH, not the `env` handed to the child, so relying on the
    tools-bin entry in `env` would only work for our grandchildren.
    """
    completed = run_process(
        [xclbinutil, "--info", "--input", xclbin],
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise E2EError(f"xclbinutil --info failed on {xclbin}")
    missing = [s for s in REQUIRED_SECTIONS if s not in completed.stdout]
    if missing:
        raise E2EError(f"xclbin missing section(s): {', '.join(missing)}")
    print(f"  ok: sections present ({'/'.join(REQUIRED_SECTIONS)})")


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
        compiler_prefix = args.compiler_prefix.resolve() if args.compiler_prefix else None
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
