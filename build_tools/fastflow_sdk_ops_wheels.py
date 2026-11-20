# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and verify the platform-neutral FastFlow SDK operator-library wheel.

The importable package is still ``iron``. The distribution name is
``fastflow-sdk-ops`` so a release install is:

    pip install fastflow-sdk-ops==<version>

which pulls ``fastflow-sdk-compiler`` at the same version. The wheel keeps the
source-tree layout at site-packages root. It ships no kernel sources: the
kernel factories compile from the compiler's own ``include/aie_kernels``
(``aie.utils.config.aie_kernels_dir()``).
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from build_tools import compiler_product, environment, paths
from build_tools.artifact import (
    file_inventory,
    git_source_state,
    reset_directory,
    validate_directory_target,
)
from build_tools.fastflow_sdk_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    OPS_DISTRIBUTION,
    OPS_PACKAGE,
    build_wheel,
    find_distribution_wheels,
    remove_distribution_wheels,
    verify_wheel_checksums,
    verify_wheel_source_revision,
    verify_wheel_text_path_references,
)
from build_tools.package_version import default_package_version
from build_tools.release_policy import compiler_requires_python
from build_tools.wheel_payload import copy_file, write_file, write_package_manifest

OPS_WHEEL_TAG = "py3-none-any"
OPS_SUMMARY = "IRON operator library for FastFlow SDK"
OPS_WHEEL_VENV = paths.BUILD / "fastflow-sdk-ops-wheel-venv"
OPS_WHEEL_ARTIFACTS = compiler_product.COMPILER_ARTIFACTS
IRON_OPS_SOURCE = paths.PROJ / "iron-ops"
_SKIP_DIR_NAMES = frozenset({"tests", "applications", "__pycache__"})
_SKIP_FILE_NAMES = frozenset({"test.py", "conftest.py"})
# IRON's Gemma 4 engine override, under iron.operators.flm. The installed ops
# wheel ships its sources and nothing else there: its Makefile clones
# FastFlowLM into build/, and the SDK example does not.
_GEMMA4_ENGINE = Path("operators") / "flm" / "gemma4_engine"
_GEMMA4_ENGINE_FILES = frozenset({"build.py", "overrides.hpp", "pli_weights.py"})
IS_WINDOWS = paths.is_windows()
_VBIN = "Scripts" if IS_WINDOWS else "bin"
_EXE = ".exe" if IS_WINDOWS else ""


def ops_requires(version: str) -> tuple[str, ...]:
    return (f"{COMPILER_DISTRIBUTION} =={version}",)


def _skip_relative(relative: Path) -> bool:
    if relative.is_relative_to(_GEMMA4_ENGINE):
        return not (
            relative.parent == _GEMMA4_ENGINE and relative.name in _GEMMA4_ENGINE_FILES
        )
    return any(part in _SKIP_DIR_NAMES for part in relative.parts) or (
        relative.name in _SKIP_FILE_NAMES or relative.suffix == ".pyc"
    )


def _copy_tree_filtered(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise RuntimeError(f"required IRON ops payload missing: {source}")
    copied = 0
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = path.relative_to(source)
        if _skip_relative(relative):
            continue
        copy_file(path, destination / relative)
        copied += 1
    if copied == 0:
        raise RuntimeError(f"IRON ops payload copied no files: {source}")


def _source_states(source_root: Path) -> dict[str, object]:
    sources = {"fastflow-sdk": git_source_state(paths.ROOT)}
    git_marker = source_root / ".git"
    if git_marker.exists():
        sources["iron-ops"] = git_source_state(source_root)
    return sources


def _ops_metadata_module() -> str:
    return '''"""Metadata for the packaged FastFlow SDK operator library.

The importable operator API is ``iron``. This package is the wheel's
identity surface: layout next to ``iron/``, plus the package manifest used by
release assembly.
"""
from __future__ import annotations

import json
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def install_root() -> Path:
    """Directory that contains ``iron/`` (site-packages)."""
    return package_root().parent


def kernels_root() -> Path:
    """The compiler's ``aie_kernels/``, which IRON's kernel factories compile from."""
    from aie.utils.config import aie_kernels_dir

    return Path(aie_kernels_dir())


def gemma4_engine_root() -> Path:
    """Installed ``overrides.hpp`` and ``build.py`` for the Gemma 4 engine override."""
    return install_root() / "iron" / "operators" / "flm" / "gemma4_engine"


def manifest() -> dict:
    return json.loads(
        (package_root() / "manifest.json").read_text(encoding="utf-8")
    )
'''


def stage_ops_tree(
    source_root: Path,
    destination: Path,
    *,
    version: str,
) -> Path:
    """Stage ``iron/`` and the ``fastflow_sdk_ops`` metadata package."""
    source_root = Path(source_root)
    destination = Path(destination)
    iron_root = source_root / "iron"
    if not iron_root.is_dir():
        raise RuntimeError(
            "projects/iron-ops is missing iron/; run: python dev.py submodules iron-ops"
        )
    reset_directory(destination, destination.parent)
    _copy_tree_filtered(iron_root, destination / "iron")
    gemma4_engine = destination / "iron" / _GEMMA4_ENGINE
    missing_gemma4 = [
        name
        for name in sorted(_GEMMA4_ENGINE_FILES)
        if not (gemma4_engine / name).is_file()
    ]
    if missing_gemma4:
        raise RuntimeError(
            "staged IRON ops wheel is missing gemma4_engine files: "
            + ", ".join(missing_gemma4)
        )
    if not (destination / "iron" / "operators" / "__init__.py").is_file():
        raise RuntimeError("staged IRON ops wheel is missing iron.operators")

    package_root = destination / OPS_PACKAGE
    write_file(package_root / "__init__.py", _ops_metadata_module())
    licenses = package_root / "licenses"
    licenses.mkdir(parents=True)
    shutil.copy2(paths.ROOT / "LICENSE", licenses / "fastflow-sdk-LICENSE")
    iron_license = source_root / "LICENSE"
    if not iron_license.is_file():
        raise RuntimeError(f"IRON LICENSE missing: {iron_license}")
    shutil.copy2(iron_license, licenses / "IRON-LICENSE")

    requires = list(ops_requires(version))
    manifest_path = package_root / "manifest.json"
    write_package_manifest(
        package_root,
        {
            "schema_version": 1,
            "product": OPS_DISTRIBUTION,
            "artifact": OPS_DISTRIBUTION,
            "version": version,
            "wheel_tag": OPS_WHEEL_TAG,
            "requires": requires,
            "sources": _source_states(source_root),
            "dependencies": {COMPILER_DISTRIBUTION: {"version": version}},
        },
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = file_inventory(destination, exclude_paths=(manifest_path,))
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return destination


def build_ops_wheel(
    source_root: Path,
    *,
    output_directory: Path,
    version: str | None = None,
) -> Path:
    source_root = Path(source_root)
    version = version or default_package_version(paths.ROOT)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    remove_distribution_wheels(output_directory, (OPS_DISTRIBUTION,))
    build_root = paths.BUILD
    build_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="fastflow-sdk-ops-wheel-stage-", dir=build_root
    ) as temporary:
        tree = stage_ops_tree(source_root, Path(temporary), version=version)
        wheel = build_wheel(
            tree,
            OPS_DISTRIBUTION,
            version,
            output_directory,
            summary=OPS_SUMMARY,
            tag=OPS_WHEEL_TAG,
            requires=ops_requires(version),
            requires_python=compiler_requires_python(),
            root_is_purelib=True,
        )
    verify_wheel_checksums({OPS_DISTRIBUTION: wheel})
    return wheel


def cmd_package_ops_wheels(args):
    try:
        wheel = build_ops_wheel(
            IRON_OPS_SOURCE,
            output_directory=Path(args.dir),
            version=args.version,
        )
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"package ops wheel: ok -> {wheel}")


def probe_operator_modules(operators):
    """Import every module of the ``iron.operators`` package; none may need torch.

    Discovery walks ``*.py`` under the package, through each directory that is
    itself a package (has ``__init__.py``), so the Gemma 4 engine's scripts are
    left out. ``test.py`` and ``conftest.py`` do not ship. Every name in
    ``CATALOG`` must then be a class from an imported module. IRON keeps torch
    off the import path of every operator; an import that pulls it in fails.
    """
    import importlib
    import sys
    from pathlib import Path

    operators_root = Path(operators.__file__).resolve().parent
    imported = []
    for path in sorted(operators_root.rglob("*.py")):
        relative = path.relative_to(operators_root)
        if path.name in ("test.py", "conftest.py") or any(
            not (
                operators_root.joinpath(*relative.parts[:depth]) / "__init__.py"
            ).is_file()
            for depth in range(1, len(relative.parts))
        ):
            continue
        parts = relative.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join((operators.__name__, *parts))
        importlib.import_module(name)
        imported.append(name)
    if "torch" in sys.modules:
        raise RuntimeError("importing iron.operators imported torch")
    for label, cls in operators.CATALOG.items():
        if not isinstance(cls, type) or cls.__module__ not in imported:
            raise RuntimeError(f"iron.operators.CATALOG[{label!r}] resolved to {cls!r}")
    return imported


def _installed_ops_probe() -> str:
    return (
        """
import importlib.metadata
from pathlib import Path

import iron.operators as operators

"""
        + inspect.getsource(probe_operator_modules)
        + """
assert importlib.metadata.version("fastflow-sdk-ops")
import fastflow_sdk_ops
kernel = fastflow_sdk_ops.kernels_root() / "eltwise" / "add.cc"
assert kernel.is_file(), kernel
assert fastflow_sdk_ops.manifest()["product"] == "fastflow-sdk-ops"
assert (fastflow_sdk_ops.gemma4_engine_root() / "overrides.hpp").is_file()
imported = probe_operator_modules(operators)
assert "iron.operators.elementwise_add" in imported
print(f"ops import: ok ({len(imported)} modules, {len(operators.CATALOG)} operators, {kernel})")
"""
    )


def _ops_compile_specs() -> tuple[
    tuple[dict[str, object], ...], tuple[dict[str, object], ...]
]:
    """Catalog records compiled from an installed ops wheel.

    Same catalog as ``dev.py test-iron-ops``: ``host_compile_suite()`` and
    ``SEQUENCES``, as the fields of their records.
    """
    from build_tools.iron import SEQUENCES, host_compile_suite

    return (
        tuple(operator._asdict() for operator in host_compile_suite()),
        tuple(sequence._asdict() for sequence in SEQUENCES),
    )


def _installed_ops_compile(build_dir: Path) -> str:
    """Python the wheel venv runs: subprocess-per-operator, like test-iron-ops.

    The venv has ``iron`` and the compiler, not ``build_tools``. The catalog
    records and the compile and staging helpers are copied from this tree, so
    an installed wheel compiles exactly what ``test-iron-ops`` does.
    """
    from build_tools import iron

    operator_specs, sequence_specs = _ops_compile_specs()
    helpers = "\n\n".join(
        inspect.getsource(helper)
        for helper in (
            iron.Operator,
            iron.Sequence,
            iron.require_runtime_image,
            iron.require_host_library,
            iron.classify_compiled_images,
            iron.Compiled,
            iron.stage_compiled_images,
            iron.operator_class,
            iron.construct_operator,
            iron.add_relu_graph,
            iron.offset_copy_graph,
            iron.require_compile_cache,
            iron.compile_full_elf,
            iron.compile_operator,
            iron.compile_sequence,
            iron.compile_job_count,
            iron.resolve_build_kwargs,
        )
    )
    graphs = ", ".join(
        f"{name!r}: {builder.__name__}"
        for name, builder in iron.FULL_ELF_GRAPHS.items()
    )
    return (
        """
import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("IRON_RUNTIME", "cpu")

import fastflow_sdk_compiler as compiler

compiler.activate_toolchain_env()
compiler.activate_native_libraries()
prefix = Path(compiler.compiler_prefix())
if not (prefix / "include").is_dir():
    raise SystemExit(f"compiler prefix has no include/: {prefix}")

"""
        + f"STAGED_NAMES = {iron.STAGED_NAMES!r}\n\n\n"
        + helpers
        + f"""

FULL_ELF_GRAPHS = {{{graphs}}}
OPERATORS = tuple(Operator(**spec) for spec in {operator_specs!r})
SEQUENCES = tuple(Sequence(**spec) for spec in {sequence_specs!r})
DEFAULT_BUILD_DIR = Path({str(build_dir)!r})


def _device(name):
    from aie.iron.device import NPU1Col1, NPU2

    return {{"npu1": NPU1Col1, "npu2": NPU2}}[name]()


def compile_one(name, build_root):
    op_dir = Path(build_root) / name
    if op_dir.exists():
        shutil.rmtree(op_dir)
    sequence = next((s for s in SEQUENCES if s.name == name), None)
    try:
        if sequence is not None:
            compiled = compile_sequence(sequence, _device("npu2"))
            want = "elf"
        else:
            operator = next(o for o in OPERATORS if o.name == name)
            compiled = compile_operator(operator, _device(operator.device))
            want = "elf" if operator.full_elf else None
        kind, staged = stage_compiled_images(compiled, op_dir)
        if want is not None and kind != want:
            raise RuntimeError(f"built {{kind}}, not {{want}}")
        for path in staged:
            if kind == "dispatch" and path == staged[1]:
                require_host_library(path)
            else:
                require_runtime_image(path, elf=(kind == "elf"))
    except (ImportError, RuntimeError) as error:
        raise SystemExit(f"{{name}}: {{error}}") from error
    print("ops compile: ok", name, *(path.name for path in staged), flush=True)
    return name


def _spawn(name, build_root):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--operator",
        name,
        "--jobs",
        "1",
        "--build-dir",
        str(build_root),
        "--keep-cache",
    ]
    return name, subprocess.run(command, capture_output=True, text=True)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args(argv)
    names = [o.name for o in OPERATORS] + [s.name for s in SEQUENCES]
    selected = [name for name in names if args.operator in (None, name)]
    if args.operator and not selected:
        raise SystemExit(f"unknown operator: {{args.operator}}")
    if not selected:
        print("ops compile: ok 0 operators")
        return 0
    build_root = args.build_dir
    build_root.mkdir(parents=True, exist_ok=True)
    # mlir-aie's compile cache, read at its first import: fresh, so every
    # operator is compiled rather than found.
    cache = build_root / ".npu-cache"
    if not args.keep_cache and cache.exists():
        shutil.rmtree(cache)
    os.environ["NPU_CACHE_HOME"] = str(cache)
    workers = compile_job_count(len(selected), args.jobs)
    print(f"ops compile: {{len(selected)}} operators, jobs={{workers}}", flush=True)
    if workers == 1:
        compiled = [compile_one(name, build_root) for name in selected]
    else:
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_spawn, name, build_root) for name in selected]
            for future in as_completed(futures):
                name, completed = future.result()
                if completed.stdout:
                    sys.stdout.write(completed.stdout)
                if completed.stderr:
                    sys.stderr.write(completed.stderr)
                if completed.returncode != 0:
                    failures.append(name)
        if failures:
            raise SystemExit("ops compile failed: " + ", ".join(failures))
        compiled = selected
    print("ops compile: ok", len(compiled), "operators")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
"""
    )


def _ops_refresh_packages(*, reuse: bool) -> tuple[str, ...]:
    if reuse:
        return (OPS_DISTRIBUTION,)
    return (
        OPS_DISTRIBUTION,
        COMPILER_DISTRIBUTION,
        COMPILER_CORE_DISTRIBUTION,
        COMPILER_BINDINGS_DISTRIBUTION,
    )


def _ops_verify_python(venv_dir: Path, *, reuse: bool) -> Path:
    validate_directory_target(venv_dir, paths.BUILD)
    python = venv_dir / _VBIN / f"python{_EXE}"
    if reuse:
        if not python.is_file():
            raise RuntimeError(
                f"--reuse-venv requires an existing interpreter: {python}"
            )
        return python
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    return python


def cmd_verify_ops_wheels(args):
    compile_installed = not args.skip_compile
    reuse_venv = bool(getattr(args, "reuse_venv", False))
    wheel_dir = Path(args.dir)
    try:
        wheels = find_distribution_wheels(wheel_dir, (OPS_DISTRIBUTION,))
        verify_wheel_checksums(wheels)
        verify_wheel_source_revision(
            wheels.values(),
            git_source_state(paths.ROOT)["revision"],
            product="ops wheel",
        )
        verify_wheel_text_path_references(
            wheels,
            (paths.ROOT, paths.BUILD),
            product="ops wheel",
        )
        version = args.version
        if not version:
            wheel = wheels[OPS_DISTRIBUTION]
            prefix = OPS_DISTRIBUTION.replace("-", "_") + "-"
            version = wheel.name[len(prefix) :].split("-py3-none-any.whl", 1)[0]
        venv_dir = Path(args.venv)
        python = _ops_verify_python(venv_dir, reuse=reuse_venv)
        environment.python_install(
            python,
            "--find-links",
            str(wheel_dir.resolve()),
            f"{OPS_DISTRIBUTION}=={version}",
            refresh_packages=_ops_refresh_packages(reuse=reuse_venv),
        )
        subprocess.run([python, "-m", "pip", "check"], check=True)
        env = dict(os.environ)
        env.setdefault("IRON_RUNTIME", "cpu")
        subprocess.run(
            [python, "-c", _installed_ops_probe()],
            check=True,
            env=env,
        )
        if compile_installed:
            compile_dir = venv_dir / "ops-compile"
            compile_dir.mkdir(parents=True, exist_ok=True)
            script_path = compile_dir / "compile_installed_operators.py"
            jobs = int(getattr(args, "jobs", 0) or 0)
            script_path.write_text(
                _installed_ops_compile(compile_dir),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    python,
                    str(script_path),
                    "--jobs",
                    str(jobs),
                    "--build-dir",
                    str(compile_dir),
                ],
                check=True,
                env=env,
            )
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"verify ops wheels: ok -> {venv_dir}")
