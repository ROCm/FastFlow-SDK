# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared IRON suite, target, toolchain, and design-build policy."""

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path
from typing import NamedTuple

from build_tools import paths

ROOT = paths.ROOT
PROJ = paths.PROJ
OUT = paths.OUT
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
    xclbinutil = (
        (compiler_prefix / "bin" / f"xclbinutil{EXE}")
        if compiler_prefix
        else ROOT / "tools-bin" / f"xclbinutil{EXE}"
    )
    if not xclbinutil.exists():
        raise E2EError(
            "xclbinutil not published (run: python dev.py build hrx-xclbinutil)\n"
            f"  {xclbinutil}"
        )
    return xclbinutil


def toolchain_env(
    xclbinutil: Path, compiler_prefix: Path | None = None
) -> dict[str, str]:
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
    # aiecc is a native bin/ tool now (checked above via aie-opt); the Python bindings
    # are marked by the assembled aie package (aie/ir.py), not a python aiecc/main.py.
    if not (aie_python / "aie" / "ir.py").exists():
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
    env["PATH"] = os.pathsep.join(
        [*(str(entry) for entry in path_entries), env.get("PATH", "")]
    )
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
        print(
            f"== toolchain: relocated rocm-npu compiler product ({compiler_prefix}) =="
        )
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


def require_runtime_image(path, *, elf=False):
    """A produced image must exist, be non-empty, and if ELF, start with ELF magic.

    Full-ELF operators skip xclbin section / TXN DevGen checks; an empty file
    would otherwise pass a boolean is_file() gate. AIE ELFs use ELF magic on
    every host, including Windows.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{path} was not produced by the build")
    if elf:
        magic = path.read_bytes()[:4]
        if magic != b"\x7fELF":
            raise RuntimeError(f"{path} is not an ELF image (magic={magic!r})")


def classify_compiled_images(artifacts):
    """Return ``('elf', [elf])`` or ``('xclbin', [xclbin, insts])``."""
    from iron.common.compilation import (
        FullElfArtifact,
        InstsBinArtifact,
        XclbinArtifact,
    )

    elves = [
        Path(artifact.filename)
        for artifact in artifacts
        if isinstance(artifact, FullElfArtifact)
    ]
    xclbins = [
        Path(artifact.filename)
        for artifact in artifacts
        if isinstance(artifact, XclbinArtifact)
    ]
    insts = [
        Path(artifact.filename)
        for artifact in artifacts
        if isinstance(artifact, InstsBinArtifact)
    ]
    if len(elves) == 1 and not xclbins and not insts:
        return "elf", elves
    if len(xclbins) == 1 and len(insts) == 1 and not elves:
        return "xclbin", xclbins + insts
    raise RuntimeError(
        "expected xclbin+insts or one full ELF, got "
        f"elf={elves} xclbin={xclbins} insts={insts}"
    )


NPU_TARGETS = {
    "npu2": {"device": "npu2", "devgen": 4, "arch": "AIE2P (Strix)"},
    "npu1": {"device": "npu", "devgen": 3, "arch": "AIE2IPU (Phoenix)"},
}


class Operator(NamedTuple):
    """One operator under test.

    Attributes:
        name: directory under iron-ops' iron/operators/
        build_kwargs: constructor kwargs for the CPU-side compile gate
        device: NPU generation to build for (dev.py's NPU_TARGETS)
        hrx_pytest_filter: pytest -k for local ``dev.py test-npu-iron-ops``.
            None means that CLI must not collect test.py. CI NPU e2e ignores
            this field; it runs ``test-npu-package`` on CASE_GENERATORS.
        host_compile: if False, skip test-iron-ops / verify-ops-wheels compile.
            The operator stays in the catalog for leftover-closed-set. Use this
            when CPU compile is known-broken (do not shim the IRON runtime in
            staging).
    """

    name: str
    build_kwargs: dict[str, object]
    device: str = "npu2"
    hrx_pytest_filter: str | None = None
    host_compile: bool = True


def _channeled_unary(name, *, extra=None, filter_extra=""):
    """One-column, one-channel, size=tile_size=2048 — the non-extensive unary shape."""
    kwargs = {
        "size": 2048,
        "num_aie_columns": 1,
        "num_channels": 1,
        "tile_size": 2048,
    }
    if extra:
        kwargs.update(extra)
    pytest_filter = (
        f"test_{name} and input_length_2048 and num_aie_columns_1 and "
        f"num_channels_1 and tile_size_2048{filter_extra}"
    )
    return Operator(name=name, build_kwargs=kwargs, hrx_pytest_filter=pytest_filter)


def _binary(name, *, extra=None, filter_extra=""):
    """One-column, size=tile_size=2048 — the non-extensive binary-elementwise shape."""
    kwargs = {"size": 2048, "num_aie_columns": 1, "tile_size": 2048}
    if extra:
        kwargs.update(extra)
    pytest_filter = (
        f"test_{name} and input_length_2048 and num_aie_columns_1 and "
        f"tile_size_2048{filter_extra}"
    )
    return Operator(name=name, build_kwargs=kwargs, hrx_pytest_filter=pytest_filter)


# One catalog, three independent consumers:
#   host_compile=True  -> CPU CI (`test-iron-ops`, `verify-ops-wheels`)
#   CASE_GENERATORS    -> NPU CI (`package-npu-test-package` / `test-npu-package`)
#   hrx_pytest_filter  -> local `test-npu-iron-ops` only (not GitHub Actions)
# swiglu_prefill_stream stays out: stream-dse imports torch and onnx at module scope.
OPERATORS = (
    _binary("elementwise_add"),
    _binary("elementwise_mul"),
    _binary("axpy", extra={"scalar_factor": 3.0}),
    _channeled_unary("relu"),
    _channeled_unary("gelu"),
    _channeled_unary("leaky_relu"),
    _channeled_unary("sigmoid"),
    _channeled_unary("tanh"),
    _channeled_unary("layer_norm"),
    Operator(
        name="silu",
        build_kwargs={"size": 2048, "num_aie_columns": 1, "tile_size": 2048},
        hrx_pytest_filter=(
            "test_silu and input_length_2048 and num_aie_columns_1 and tile_size_2048"
        ),
    ),
    Operator(
        name="rms_norm",
        build_kwargs={
            "size": 2048,
            "num_aie_columns": 1,
            "num_channels": 1,
            "tile_size": 2048,
            "weighted": False,
        },
        hrx_pytest_filter=(
            "test_rms_norm and input_length_2048 and num_aie_columns_1 and "
            "num_channels_1 and tile_size_2048"
        ),
    ),
    Operator(
        name="dequant",
        build_kwargs={
            "size": 2048,
            "num_aie_columns": 1,
            "num_channels": 1,
            "tile_size": 2048,
        },
        hrx_pytest_filter=(
            "test_dequant and input_length_2048 and num_aie_columns_1 and "
            "num_channels_1 and tile_size_2048"
        ),
    ),
    Operator(
        name="mem_copy",
        build_kwargs={
            "size": 2048,
            "num_cores": 1,
            "num_channels": 1,
            "bypass": False,
            "tile_size": 2048,
        },
        hrx_pytest_filter=(
            "test_mem_copy and input_length_2048 and num_cores_1 and "
            "num_channels_1 and tile_size_2048"
        ),
    ),
    Operator(
        name="gemv",
        build_kwargs={
            "M": 128,
            "K": 128,
            "num_aie_columns": 1,
            "tile_size_input": 32,
            "tile_size_output": 128,
        },
        hrx_pytest_filter="test_gemv and M_128 and K_128 and num_aie_columns_1",
    ),
    Operator(
        name="gemm",
        build_kwargs={
            "M": 256,
            "K": 64,
            "N": 64,
            "num_aie_columns": 1,
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 64,
        },
        # CPU compile 256x64x64; NPU CI runs that shape via CASE_GENERATORS.
        # IRON pytest has no such case (regular is 2048^3). Do not collect
        # test.py: a fake -k still looks like coverage when pytest exits 0.
    ),
    Operator(
        name="softmax",
        # aie2p softmax.cc loops vector_size / SM_VEC_LEN with SM_VEC_LEN=64.
        # cols=32 compiles but runs 0 iterations and writes zeros.
        build_kwargs={
            "rows": 32,
            "cols": 64,
            "num_aie_columns": 1,
            "num_channels": 1,
        },
        # IRON pytest only has length 32768. NPU CI uses this 32x64 package case.
    ),
    Operator(
        name="rope",
        build_kwargs={
            "rows": 32,
            "cols": 512,
            "angle_rows": 8,
            "num_aie_columns": 1,
            "method_type": 0,
        },
        hrx_pytest_filter=(
            "test_rope and rows_32 and cols_512 and angle_rows_8 and "
            "aie_columns_1 and method_type_0"
        ),
    ),
    Operator(
        name="transpose",
        build_kwargs={
            "M": 64,
            "N": 64,
            "num_aie_columns": 1,
            "num_channels": 1,
            "m": 64,
            "n": 64,
            "s": 8,
            "num_batches": 1,
        },
        # Compile 64x64 is extensive in IRON pytest; -m not extensive skips it.
        # NPU CI runs this shape via CASE_GENERATORS.
    ),
    Operator(
        name="repeat",
        # cols must exceed 1023 so the design's DMA split is 4-byte aligned for BF16.
        # CPU compile and the packaged numeric case are 32x1024x2. IRON's
        # non-extensive pytest row is 4x1024x2 (ids rows_4-cols_1024-...), not
        # this compile shape.
        build_kwargs={"rows": 32, "cols": 1024, "repeat": 2},
    ),
    Operator(
        name="strided_copy",
        # CPU compile is 1-channel contiguous. NPU CI packages that identity
        # gather only; do not add two_channels (amd/IRON#158 hang). Local
        # pytest still collects two_channels; other pytest ids stay extensive.
        build_kwargs={
            "input_sizes": [1024],
            "input_strides": [1],
            "input_offset": 0,
            "input_buffer_size": 1024,
            "output_sizes": [1024],
            "output_strides": [1],
            "output_offset": 0,
            "output_buffer_size": 1024,
            "num_aie_channels": 1,
        },
        hrx_pytest_filter="test_strided_copy and (contiguous or two_channels)",
    ),
    Operator(
        name="mha",
        # Only one non-extensive IRON pytest case. CPU CI compiles this shape
        # (patch 0001: get_current_device). NPU CI has no MHA generator, so it
        # does not dispatch this xclbin. Local test-npu-iron-ops would compile
        # again inside run_test; CPU CI is the drift gate, not an artifact cache.
        build_kwargs={
            "num_heads": 1,
            "seq_len": 16384,
            "d": 64,
            "num_KV_heads": 0,
            "num_of_pipelines": 8,
        },
        hrx_pytest_filter="test_mha",
    ),
    Operator(
        name="swiglu_decode",
        # OperatorSequence fused ELF needs aiebu-asm, which this stack does not
        # ship. HRX cannot dispatch it yet either. Keep in the catalog for the
        # leftover-closed-set; do not host-compile or collect test.py.
        build_kwargs={"embedding_dim": 2048, "hidden_dim": 2048},
        host_compile=False,
    ),
    Operator(
        name="swiglu_prefill",
        # Same fused-ELF host path as swiglu_decode.
        build_kwargs={
            "seq_len": 256,
            "embedding_dim": 2048,
            "hidden_dim": 2048,
            "prio_accuracy": False,
        },
        host_compile=False,
    ),
)

DEFAULT = OPERATORS[0].name


def select(name: str | None) -> tuple[Operator, ...]:
    """The operators to gate on: all of them, or the one named."""
    if name is None:
        return OPERATORS
    chosen = [op for op in OPERATORS if op.name == name]
    if not chosen:
        known = ", ".join(op.name for op in OPERATORS)
        raise KeyError(
            f"{name}: not in the iron-ops suite ({known}) -- add a row to OPERATORS"
        )
    return tuple(chosen)


def host_compile_suite(
    operators: tuple[Operator, ...] | None = None,
) -> tuple[Operator, ...]:
    """Operators the CPU compile gate actually builds."""
    chosen = OPERATORS if operators is None else operators
    return tuple(operator for operator in chosen if operator.host_compile)


def compile_job_count(n_operators: int, jobs: int = 0) -> int:
    """How many operator compiles to run at once.

    ``jobs <= 0`` means auto: ``min(8, os.cpu_count(), n_operators)``.
    ``aie.utils`` device state is process-global, so callers must isolate each
    compile in its own process when this is greater than 1.
    """
    n = max(0, int(n_operators))
    if n <= 1:
        return 1
    if jobs and jobs > 0:
        return min(int(jobs), n)
    cpu = os.cpu_count() or 1
    return max(1, min(8, cpu, n))


def iron_env(compiler_prefix=None):
    prefix = (
        Path(compiler_prefix).resolve()
        if compiler_prefix
        else compiler_prefix_from_env()
    )
    try:
        return toolchain_env(require_xclbinutil(prefix), prefix)
    except E2EError as error:
        sys.exit(str(error))


def iron_ops_env(compiler_prefix=None):
    """Extend the selected IRON toolchain environment with iron-ops."""
    env = iron_env(compiler_prefix)
    env["PYTHONPATH"] = os.pathsep.join([str(PROJ / "iron-ops"), env["PYTHONPATH"]])
    return env


def require_txn_devgen(insts, target):
    want = NPU_TARGETS[target]["devgen"]
    data = insts.read_bytes()
    if len(data) < 3:
        sys.exit(f"{insts}: too small for a TXN header")
    got = data[2]
    if got != want:
        by_generation = {
            value["devgen"]: f"{key} / {value['arch']}"
            for key, value in NPU_TARGETS.items()
        }
        sys.exit(
            f"{insts}: TXN DevGen={got} "
            f"({by_generation.get(got, 'unknown')}) but --device={target} "
            f"needs DevGen={want} ({by_generation[want]})"
        )
    print(f"insts.bin: TXN DevGen={got} (matches --device={target})")
