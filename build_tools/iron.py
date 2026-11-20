# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared IRON suite, target, toolchain, and design-build policy."""

import importlib
import importlib.util
import json
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
    override = os.environ.get("FASTFLOW_SDK_CMAKE")
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
    value = os.environ.get("FASTFLOW_SDK_COMPILER_PREFIX")
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
    """IRON toolchain env from either build trees or a relocated fastflow-sdk-compiler product."""
    if compiler_prefix:
        aie_bin = compiler_prefix / "bin"
        aiebu_bin = aie_bin
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
        aiebu_bin = paths.AIEBU_INSTALL / "bin"
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
    if not (aiebu_bin / f"aiebu-asm{EXE}").is_file():
        raise E2EError(
            f"aiebu-asm not built (run: python dev.py build aiebu)\n  {aiebu_bin}"
        )

    env = dict(os.environ)
    path_entries = [
        xclbinutil.parent,
        aie_bin,
        aiebu_bin,
        VENV_BIN,
        cmake_candidate().parent,
    ]
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
        env["FASTFLOW_SDK_COMPILER_PREFIX"] = str(compiler_prefix)
        print(
            f"== toolchain: relocated fastflow compiler product ({compiler_prefix}) =="
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


def require_host_library(path):
    """A dispatch library must exist and be a host image: PE on Windows, ELF elsewhere."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{path} was not produced by the build")
    magic = path.read_bytes()[:4]
    if os.name == "nt":
        if magic[:2] != b"MZ":
            raise RuntimeError(f"{path} is not a Windows library (magic={magic[:2]!r})")
        return
    if magic != b"\x7fELF":
        raise RuntimeError(f"{path} is not an ELF library (magic={magic!r})")


def classify_compiled_images(artifacts):
    """Return the runtime images an IRON ``Artifacts`` record names.

    ``('elf', [elf])`` is one full ELF. ``('xclbin', [xclbin, insts])`` is a
    fixed instruction stream. ``('dispatch', [xclbin, library])`` is an xclbin
    whose instruction stream a host library generates per call, from the
    operator's per-call values.
    """
    image = Path(artifacts.image)
    if artifacts.full_elf:
        return "elf", [image]
    entry = artifacts.entry
    library = None if entry is None else entry.dispatch_library
    if library is not None:
        return "dispatch", [image, Path(library)]
    if artifacts.insts is not None:
        return "xclbin", [image, Path(artifacts.insts)]
    raise RuntimeError(
        f"{image}: an xclbin with neither an instruction stream nor a dispatch library"
    )


STAGED_NAMES = {
    "elf": ("final.elf",),
    "xclbin": ("final.xclbin", "insts.bin"),
    "dispatch": ("final.xclbin", "dispatch.dll" if os.name == "nt" else "dispatch.so"),
}


class Compiled(NamedTuple):
    """One compile's IRON ``Artifacts`` and, for a full ELF, its arguments.

    ``arguments`` is the bytes of each runtime-sequence argument by buffer
    kind, in binding order (``ArgumentSizes.arguments()``); None for an xclbin.
    """

    artifacts: object
    arguments: dict | None = None


def stage_compiled_images(compiled, directory):
    """Copy the runtime images out of mlir-aie's compile cache into ``directory``.

    The cache is keyed by content and shared across operators; the NPU package
    reads each operator's images from its own directory. A full ELF also stages
    its scratchpad parameter table (``params.txt``) and its ``layout.json``:
    ``arguments`` (bytes per argument kind, in binding order) and ``buffers``
    (name to argument kind, byte offset, bytes). Returns ``(kind, staged)``.
    """
    artifacts = compiled.artifacts
    kind, produced = classify_compiled_images(artifacts)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    staged = []
    for source, name in zip(produced, STAGED_NAMES[kind], strict=True):
        target = directory / name
        shutil.copy2(source, target)
        staged.append(target)
    if kind == "elf":
        if artifacts.params is None:
            raise RuntimeError(f"{produced[0]}: a full ELF without a parameter table")
        if compiled.arguments is None:
            raise RuntimeError(f"{produced[0]}: a full ELF without its argument sizes")
        shutil.copy2(artifacts.params, directory / "params.txt")
        layout = {
            "arguments": dict(compiled.arguments),
            "buffers": {name: list(where) for name, where in artifacts.buffers.items()},
        }
        (directory / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    return kind, staged


def operator_class(module, class_name=None):
    """The IRON operator class ``module`` (dotted under ``iron.operators``) defines.

    A package keeps its operators in its ``op`` module (``flm.gemm``); a flat
    module (``relu``) defines them itself. ``class_name`` picks one where the
    module defines several (``elementwise_add``'s ``ElementwiseAdd`` and
    ``RowwiseAdd``).
    """
    from iron.common import Operator

    found_module = importlib.import_module(f"iron.operators.{module}")
    if hasattr(found_module, "__path__"):
        found_module = importlib.import_module(f"{found_module.__name__}.op")
    found = [
        value
        for value in vars(found_module).values()
        if isinstance(value, type)
        and issubclass(value, Operator)
        and value.__module__ == found_module.__name__
        and not value.__name__.startswith("_")
    ]
    if class_name:
        found = [value for value in found if value.__name__ == class_name]
    if len(found) != 1:
        names = ", ".join(sorted(value.__name__ for value in found)) or "none"
        raise RuntimeError(
            f"{module}: expected exactly one operator class in {found_module.__name__}, "
            f"found {len(found)} ({names}); name one with class_name"
        )
    return found[0]


def construct_operator(cls, kwargs):
    """``cls(**kwargs)``, refusing a keyword the operator does not declare."""
    declared = list(cls.declaration.params)
    unknown = sorted(set(kwargs) - set(declared))
    if unknown:
        raise RuntimeError(
            f"{cls.__name__} declares no {', '.join(unknown)}; it takes "
            f"({', '.join(declared)})"
        )
    return cls(**kwargs)


def add_relu_graph(size, tile_size, num_aie_columns):
    """``relu(a + b)`` over ``size`` elements, both steps in one full ELF."""
    import iron as graphs
    from iron.operators import ElementwiseAdd, ReLU

    class AddRelu(graphs.Graph):
        def body(self, a, b):
            added = ElementwiseAdd.step(
                a, b, tile_size=tile_size, num_aie_columns=num_aie_columns
            )
            return ReLU.step(
                added,
                tile_size=tile_size,
                num_aie_columns=num_aie_columns,
                num_channels=1,
            )

    return AddRelu(), {"a": (size,), "b": (size,)}


def offset_copy_graph(input_buffer_size, length):
    """Copy ``length`` elements of the input from a per-call offset, in one full ELF.

    The offset is the scratchpad word ``input_offset``; one image serves every
    offset that keeps the copy inside the input.
    """
    import iron as graphs
    import numpy as np
    from iron.operators import Copy

    class OffsetCopy(graphs.Graph):
        def body(self, x, *, input_offset: graphs.Scratchpad[np.int32]):
            return Copy.step(x[input_offset : input_offset + length])

    return OffsetCopy(), {"x": (input_buffer_size,)}


def require_compile_cache():
    """mlir-aie compiles into the ``NPU_CACHE_HOME`` this process was given.

    It reads the variable once, when ``aie.utils.compile`` is first imported;
    one imported earlier would compile into, or find a stale image in, another
    cache.
    """
    from aie.utils.compile import NPU_CACHE_HOME

    wanted = os.environ.get("NPU_CACHE_HOME")
    if wanted and Path(NPU_CACHE_HOME).resolve() != Path(wanted).resolve():
        raise RuntimeError(
            f"mlir-aie's compile cache is {NPU_CACHE_HOME}, not NPU_CACHE_HOME={wanted}: "
            "aie.utils.compile was imported before NPU_CACHE_HOME was set"
        )


# A full-ELF catalog entry's graph, by name: its build_kwargs are the builder's.
FULL_ELF_GRAPHS = {"strided_copy": offset_copy_graph}


def compile_full_elf(graph, shapes, device):
    """Compile ``graph`` at ``shapes`` to one full ELF; return its ``Compiled``."""
    compiled = graph.compile(dev=device, full_elf=True, **shapes)
    return Compiled(compiled.artifacts, compiled.sequence.buffer_sizes.arguments())


def compile_operator(operator, device):
    """Compile one catalog entry for ``device`` and return its ``Compiled``.

    ``device`` is an ``aie.iron.device`` instance, bound for this compile
    alone: with no NPU present there is nothing to infer it from.
    """
    import aie.utils as aie_utils

    require_compile_cache()
    kwargs = resolve_build_kwargs(operator.build_kwargs)
    if operator.full_elf:
        graph, shapes = FULL_ELF_GRAPHS[operator.name](**kwargs)
        return compile_full_elf(graph, shapes, device)
    cls = operator_class(operator.module or operator.name, operator.class_name)
    with aie_utils.set_current_device(device):
        return Compiled(construct_operator(cls, kwargs).resolved(device).compile())


def compile_sequence(sequence, device):
    """Compile one fused sequence to a full ELF and return its ``Compiled``."""
    require_compile_cache()
    graph, shapes = add_relu_graph(
        sequence.size, sequence.tile_size, sequence.num_aie_columns
    )
    return compile_full_elf(graph, shapes, device)


NPU_TARGETS = {
    "npu2": {"device": "npu2", "devgen": 4, "arch": "AIE2P (Strix)"},
    "npu1": {"device": "npu", "devgen": 3, "arch": "AIE2IPU (Phoenix)"},
}


class Operator(NamedTuple):
    """One operator under test.

    Attributes:
        name: the suite's name for it; also its module under ``iron.operators``
            unless ``module`` says otherwise
        build_kwargs: constructor kwargs for the CPU-side compile gate
        device: NPU generation to build for (dev.py's NPU_TARGETS)
        hrx_pytest_case: the id of IRON's ``test_catalog.py::test_operator``
            case that runs this shape, for local ``dev.py test-npu-iron-ops``.
            None means IRON declares no case at this shape and that CLI skips
            it. CI NPU e2e ignores this field; it runs ``test-npu-package``
            on CASE_GENERATORS.
        host_compile: if False, skip test-iron-ops / verify-ops-wheels compile.
            The operator stays in the catalog for leftover-closed-set. Use this
            when CPU compile is known-broken (do not shim the IRON runtime in
            staging).
        full_elf: compile through an ``iron.Graph`` to one full ELF and its
            scratchpad parameter table, the pair a dynamic IRON ELF converts
            to XADX from.
        module: dotted module under ``iron.operators`` when it differs from
            ``name``. A package's operators live in its ``op`` module.
        class_name: operator class to build when ``module`` defines more than one.
        context: for a full ELF, the catalog operator whose xclbin creates
            the hardware context its XADX runs in (the ELF loads its own
            PDIs, but HRX takes the partition from an xclbin).
    """

    name: str
    build_kwargs: dict[str, object]
    device: str = "npu2"
    hrx_pytest_case: str | None = None
    host_compile: bool = True
    full_elf: bool = False
    module: str | None = None
    class_name: str | None = None
    context: str | None = None


class Sequence(NamedTuple):
    """One fused multi-operator sequence compiled for packaged NPU coverage.

    ``context`` is as on ``Operator``.
    """

    name: str
    size: int
    tile_size: int
    num_aie_columns: int
    context: str


SEQUENCES = (
    Sequence(
        name="add_relu_fused",
        size=4096,
        tile_size=1024,
        num_aie_columns=4,
        context="elementwise_add",
    ),
)


def _case_id(class_name, kwargs, device="npu2"):
    """IRON's catalog case id: device, class, then each kwarg as ``key_value``."""
    label = "-".join(f"{key}_{value}" for key, value in kwargs.items())
    return f"{device}-{class_name}-{label}"


_UNARY_2048 = {
    "size": 2048,
    "num_aie_columns": 1,
    "num_channels": 1,
    "tile_size": 2048,
}


def _channeled_unary(name, class_name):
    """One-column, one-channel, size=tile_size=2048 — IRON's first unary case."""
    return Operator(
        name=name,
        build_kwargs=dict(_UNARY_2048),
        hrx_pytest_case=_case_id(class_name, _UNARY_2048),
        class_name=class_name,
    )


def _declared(name, class_name, kwargs, *, label=None, module=None):
    """An operator at one of IRON's declared cases, which local NPU runs name.

    ``label`` is the case's name where IRON names it rather than listing its
    kwargs.
    """
    case_id = f"npu2-{class_name}-{label}" if label else _case_id(class_name, kwargs)
    return Operator(
        name=name,
        build_kwargs=kwargs,
        hrx_pytest_case=case_id,
        module=module,
        class_name=class_name,
    )


def _binary(name, class_name, *, extra=None):
    """One-column, size=tile_size=2048 — IRON's first binary-elementwise case."""
    kwargs = {"size": 2048, "num_aie_columns": 1, "tile_size": 2048, **(extra or {})}
    return Operator(
        name=name,
        build_kwargs=kwargs,
        hrx_pytest_case=_case_id(class_name, kwargs),
        class_name=class_name,
    )


def _rows(name, class_name):
    """One 2048-wide row, one column and channel — IRON's first norm case."""
    kwargs = {
        "rows": 1,
        "num_aie_columns": 1,
        "num_channels": 1,
        "tile_size": 2048,
    }
    return Operator(
        name=name,
        build_kwargs=kwargs,
        hrx_pytest_case=_case_id(class_name, kwargs),
        class_name=class_name,
    )


# One catalog, three independent consumers:
#   host_compile=True  -> CPU CI (`test-iron-ops`, `verify-ops-wheels`)
#   CASE_GENERATORS    -> NumPy oracles in the prebuilt NPU package
#   flm_npu_cases      -> the flm ports, appended to that same package
#   hrx_pytest_case    -> local `test-npu-iron-ops` only (not GitHub Actions)
OPERATORS = (
    _binary("elementwise_add", "ElementwiseAdd"),
    _binary("elementwise_mul", "ElementwiseMul"),
    _binary("axpy", "AXPY", extra={"scalar_factor": 3.0}),
    _channeled_unary("relu", "ReLU"),
    _channeled_unary("gelu", "GELU"),
    _channeled_unary("sigmoid", "Sigmoid"),
    _channeled_unary("tanh", "Tanh"),
    _rows("layer_norm", "LayerNorm"),
    Operator(
        name="silu",
        class_name="SiLU",
        build_kwargs={"size": 2048, "num_aie_columns": 1, "tile_size": 2048},
        hrx_pytest_case=_case_id(
            "SiLU", {"size": 2048, "num_aie_columns": 1, "tile_size": 2048}
        ),
    ),
    _rows("rms_norm", "RMSNorm"),
    Operator(
        name="copy",
        class_name="Copy",
        build_kwargs={
            "input_buffer_size": 2048,
            "num_channels": 1,
            "tile_size": 2048,
        },
        # IRON's Copy cases are named by layout (contiguous, slot5, ...),
        # not by shape; none is this one.
    ),
    Operator(
        name="gemv",
        class_name="GEMV",
        build_kwargs={
            "M": 128,
            "K": 128,
            "num_aie_columns": 1,
            "tile_size_input": 32,
            "tile_size_output": 128,
        },
        hrx_pytest_case=_case_id(
            "GEMV",
            {
                "M": 128,
                "K": 128,
                "num_aie_columns": 1,
                "tile_size_input": 32,
                "tile_size_output": 128,
            },
        ),
    ),
    Operator(
        name="gemm",
        class_name="GEMM",
        build_kwargs={
            "M": 256,
            "K": 64,
            "N": 64,
            "num_aie_columns": 1,
            "tile_m": 64,
            "tile_k": 64,
            "tile_n": 64,
            "precision": "highest",
        },
        # CPU compile 256x64x64; NPU CI runs that shape via CASE_GENERATORS.
        # IRON's smallest GEMM case is 2048^3.
    ),
    Operator(
        name="softmax",
        class_name="Softmax",
        # aie2p softmax.cc loops vector_size / SM_VEC_LEN with SM_VEC_LEN=64.
        # cols=32 compiles but runs 0 iterations and writes zeros.
        build_kwargs={
            "rows": 32,
            "cols": 64,
            "num_aie_columns": 1,
            "num_channels": 1,
        },
        # IRON's smallest Softmax case is 32x1024 on two columns. NPU CI uses
        # this 32x64 package case.
    ),
    Operator(
        name="rope",
        class_name="RoPE",
        # two_halves (its default) rotates the halves of each row, as
        # Hugging Face does.
        build_kwargs={
            "rows": 32,
            "cols": 512,
            "num_aie_columns": 1,
            "angle_rows": 8,
        },
        hrx_pytest_case=_case_id(
            "RoPE", {"rows": 32, "cols": 512, "num_aie_columns": 1, "angle_rows": 8}
        ),
    ),
    Operator(
        name="strided_copy",
        module="copy",
        class_name="Copy",
        # A Copy from a per-call offset, compiled through an iron.Graph to one
        # full ELF. NPU CI alternates the input_offset scratchpad word between
        # the two 1024-element halves of one 2048-element input while keeping
        # the executable resident.
        build_kwargs={"input_buffer_size": 2048, "length": 1024},
        # No IRON catalog case has this dynamic-offset shape. The packaged
        # XADX case is the execution gate.
        full_elf=True,
        context="copy",
    ),
    Operator(
        name="mha",
        class_name="MHA",
        # CPU CI compiles this shape. NPU CI has no MHA generator, so it does
        # not dispatch this xclbin.
        build_kwargs={"num_heads": 1, "seq_len": 16384, "num_pipelines": 8},
        hrx_pytest_case=_case_id(
            "MHA", {"num_heads": 1, "seq_len": 16384, "num_pipelines": 8}
        ),
    ),
    # The rest of IRON's catalog, each at its first declared case: compiled on
    # CPU CI and run locally by that case; no packaged NPU numeric case yet.
    _declared("clamp", "Clamp", {**_UNARY_2048, "low": -0.75, "high": 1.25}),
    _declared(
        "depthwise_conv1d",
        "DepthwiseConv1d",
        {"rows": 64, "channels": 64, "num_aie_columns": 1},
    ),
    _declared("emit", "Emit", {"slots": 32, "carried": 1}, label="one_carried"),
    _declared("limbs", "Limbs", {"rows": 64, "line": 160, "num_aie_columns": 1}),
    _declared("log", "Log", dict(_UNARY_2048)),
    _declared(
        "magnitude", "Magnitude", {"rows": 64, "width": 640, "num_aie_columns": 1}
    ),
    _declared(
        "merge",
        "Merge",
        {"rows": 2048, "placeholders": ((258881, 2048), (258880, 2560))},
    ),
    _declared(
        "patch_positions",
        "PatchPositions",
        {"rows": 2560, "out_height": 672, "out_width": 912},
        module="resample",
    ),
    _declared(
        "resample",
        "Resample",
        {"height": 3024, "width": 4032, "out_height": 672, "out_width": 912},
    ),
    _declared(
        "resample_taps",
        "ResampleTaps",
        {"in_size": 4032, "out_size": 912},
        module="resample",
    ),
    _declared(
        "rowwise_add",
        "RowwiseAdd",
        {"rows": 1, "num_aie_columns": 1, "num_channels": 1, "tile_size": 2048},
        module="elementwise_add",
    ),
    _declared(
        "rowwise_mul",
        "RowwiseMul",
        {"rows": 1, "num_aie_columns": 1, "num_channels": 1, "tile_size": 2048},
        module="elementwise_mul",
    ),
    _declared(
        "sample", "Sample", {"vocab": 4096, "cores": 4, "steps": 3}, label="small"
    ),
    _declared(
        "selective_scan",
        "SelectiveScan",
        {"rows": 5, "heads": 2, "head_dim": 16, "state_size": 16, "columns": 1},
    ),
    # FastFlowLM ports, at their smallest declared shapes. They are
    # host-compiled; flm_npu_cases packages them for NPU CI.
    Operator(
        name="flm.gemm",
        class_name="ProjectionGEMM",
        # npu2's smallest full column sweep. ProjectionGEMM rounds per K tile,
        # so "high" is the most it offers.
        build_kwargs={"M": 256, "K": 512, "N": 1024, "precision": "high"},
    ),
    Operator(
        name="flm.dequant",
        class_name="DequantBFP",
        build_kwargs={"K": 512, "N": 128},
        hrx_pytest_case=_case_id("flm.DequantBFP", {"K": 512, "N": 128}),
    ),
    Operator(
        name="flm.layer",
        class_name="DecodeLayer",
        # E2B is the non-extensive geometry. The string is resolved to
        # aie.iron.kernels.FLM_GEMMA4_E2B_DECODE at compile time.
        build_kwargs={"geometry": "FLM_GEMMA4_E2B_DECODE", "layer_type": "global"},
    ),
    Operator(
        name="flm.lm_head",
        class_name="LMHead",
        # Gemma 4 E2B hidden size. vocab=4096 is one or two NPU2 rounds
        # (cols * rows * 64) and is the projection test's vocabulary.
        build_kwargs={"dim": 1536, "vocab": 4096, "softcap": 1000.0},
    ),
    Operator(
        name="flm.prefill_attn",
        class_name="PrefillAttention",
        build_kwargs={"max_context": 1024, "num_heads": 8, "num_kv_heads": 1},
    ),
    Operator(
        name="flm.prefill_sliding_attn",
        module="flm.prefill_attn",
        class_name="PrefillSlidingAttention",
        build_kwargs={
            "max_context": 2048,
            "window": 512,
            "num_heads": 8,
            "num_kv_heads": 1,
        },
    ),
)


DEFAULT = OPERATORS[0].name

# IRON operators with no shape of their own to compile: they run only as a
# step of a graph (Copy's gathered rows), so OPERATORS has no row for them.
GRAPH_ONLY = frozenset({"Gather", "GatherWords"})


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


def resolve_build_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    """Turn catalog values the constructor cannot take literally into objects.

    ``flm.layer`` stores ``geometry`` as the ``aie.iron.kernels`` attribute
    name so importing this module does not import the compiler.
    """
    resolved = dict(kwargs)
    geometry = resolved.get("geometry")
    if isinstance(geometry, str):
        kernels = importlib.import_module("aie.iron.kernels")
        try:
            resolved["geometry"] = getattr(kernels, geometry)
        except AttributeError as error:
            raise RuntimeError(f"unknown flm geometry {geometry!r}") from error
    return resolved


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
