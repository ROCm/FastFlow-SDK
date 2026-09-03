# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and verify the platform-neutral ROCm NPU operator-library wheel.

The importable package is still ``iron`` (and its ``aie_kernels`` sibling). The
distribution name is ``rocm-npu-ops`` so a release install is:

    pip install rocm-npu-ops==<version>

which pulls ``rocm-npu-compiler`` at the same version. IRON looks up kernel
sources as ``AIEContext.base_dir / "aie_kernels"``, and ``base_dir`` is the
parent of the ``iron`` package, so the wheel keeps that source-tree layout at
site-packages root.
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
from build_tools.package_version import default_package_version
from build_tools.release_policy import compiler_requires_python
from build_tools.rocm_npu_wheel_common import (
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
from build_tools.wheel_payload import copy_file, write_file, write_package_manifest

OPS_WHEEL_TAG = "py3-none-any"
OPS_SUMMARY = "IRON operator library for ROCm NPU"
OPS_WHEEL_VENV = paths.BUILD / "rocm-npu-ops-wheel-venv"
OPS_WHEEL_ARTIFACTS = compiler_product.COMPILER_ARTIFACTS
IRON_OPS_SOURCE = paths.PROJ / "iron-ops"
_SKIP_DIR_NAMES = frozenset({"tests", "applications", "__pycache__"})
_SKIP_FILE_NAMES = frozenset({"test.py", "conftest.py"})
IS_WINDOWS = paths.is_windows()
_VBIN = "Scripts" if IS_WINDOWS else "bin"
_EXE = ".exe" if IS_WINDOWS else ""


def ops_requires(version: str) -> tuple[str, ...]:
    return (f"{COMPILER_DISTRIBUTION} =={version}",)


def _skip_relative(relative: Path) -> bool:
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
    sources = {"rocm-npu-staging": git_source_state(paths.ROOT)}
    git_marker = source_root / ".git"
    if git_marker.exists():
        sources["iron-ops"] = git_source_state(source_root)
    return sources


def _ops_metadata_module() -> str:
    return '''"""Metadata for the packaged ROCm NPU operator library.

The importable operator API is ``iron``. This package is the wheel's
identity surface: layout next to ``iron/`` and ``aie_kernels/``, plus the
package manifest used by release assembly.
"""
from __future__ import annotations

import json
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def install_root() -> Path:
    """Directory that contains ``iron/`` and ``aie_kernels/`` (site-packages)."""
    return package_root().parent


def kernels_root() -> Path:
    return install_root() / "aie_kernels"


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
    """Stage ``iron/``, ``aie_kernels/``, and the ``rocm_npu_ops`` metadata package."""
    source_root = Path(source_root)
    destination = Path(destination)
    iron_root = source_root / "iron"
    kernels_root = source_root / "aie_kernels"
    if not iron_root.is_dir() or not kernels_root.is_dir():
        raise RuntimeError(
            "projects/iron-ops is missing iron/ or aie_kernels/; "
            "run: python dev.py submodules iron-ops"
        )
    reset_directory(destination, destination.parent)
    _copy_tree_filtered(iron_root, destination / "iron")
    _copy_tree_filtered(kernels_root, destination / "aie_kernels")
    if not (destination / "iron" / "operators" / "__init__.py").is_file():
        raise RuntimeError("staged IRON ops wheel is missing iron.operators")
    if not (destination / "aie_kernels" / "generic" / "add.cc").is_file():
        raise RuntimeError(
            "staged IRON ops wheel is missing aie_kernels/generic/add.cc"
        )

    package_root = destination / OPS_PACKAGE
    write_file(package_root / "__init__.py", _ops_metadata_module())
    licenses = package_root / "licenses"
    licenses.mkdir(parents=True)
    shutil.copy2(paths.ROOT / "LICENSE", licenses / "rocm-npu-LICENSE")
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
        prefix="rocm-npu-ops-wheel-stage-", dir=build_root
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


# Operator packages whose import graph still needs torch at module scope.
# Stream-dse binds torch/onnx in iron.common.stream.ops. MHA and the hand-written
# SwiGLU sequences compile without torch (IRON #164/#165). An unused skip is a
# hard failure so a pin bump cannot leave a torch regression silently ignored.
_TORCH_AT_IMPORT = frozenset({"swiglu_prefill_stream"})


def _torch_is_missing(error: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModuleNotFoundError) and current.name == "torch":
            return True
        current = current.__cause__ or current.__context__
    return False


def probe_operator_modules(operators, *, torch_at_import: frozenset[str]):
    """Import every ``iron.operators.<pkg>.op``; skip only ``torch_at_import``.

    Discovery is ``*/op.py`` next to ``iron.operators`` (IRON operator packages
    have no ``__init__.py``, so ``pkgutil.iter_modules`` would miss them). The
    discovered set must equal imported ∪ skipped. Every skip-list package must
    be present and must fail with missing torch. A skip-list package that
    imports is a stale allowlist (drop it). Lazy ``__all__`` exports are
    resolved through ``getattr`` after the package walk.
    """
    import importlib
    import sys
    from pathlib import Path

    operators_root = Path(operators.__file__).resolve().parent
    discovered = []
    imported = []
    skipped = []
    for child in sorted(operators_root.iterdir()):
        if not child.is_dir() or not (child / "op.py").is_file():
            continue
        discovered.append(child.name)
        name = f"{operators.__name__}.{child.name}.op"
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            if child.name in torch_at_import and _torch_is_missing(error):
                skipped.append(name)
                continue
            raise
        imported.append(name)

    discovered_set = set(discovered)
    missing_skips = sorted(torch_at_import - discovered_set)
    if missing_skips:
        raise RuntimeError(
            "ops wheel is missing torch-at-import operator packages: "
            + ", ".join(missing_skips)
        )
    unused_skips = sorted(
        pkg for pkg in torch_at_import if f"{operators.__name__}.{pkg}.op" in imported
    )
    if unused_skips:
        raise RuntimeError(
            "torch-at-import skip list is stale; drop: " + ", ".join(unused_skips)
        )
    expected_skipped = {f"{operators.__name__}.{pkg}.op" for pkg in torch_at_import}
    if set(skipped) != expected_skipped:
        raise RuntimeError(
            f"torch-at-import skips mismatch: skipped={skipped} "
            f"expected={sorted(expected_skipped)}"
        )
    expected_imported = {
        f"{operators.__name__}.{pkg}.op"
        for pkg in discovered_set - set(torch_at_import)
    }
    if set(imported) != expected_imported:
        raise RuntimeError(
            f"operator import set mismatch: imported={imported} "
            f"expected={sorted(expected_imported)}"
        )

    for export in operators.__all__:
        try:
            value = getattr(operators, export)
        except ModuleNotFoundError as error:
            if not _torch_is_missing(error):
                raise
            if not skipped:
                raise
            imported_modules = [
                sys.modules[name] for name in imported if name in sys.modules
            ]
            if any(
                getattr(module, export, None) is not None for module in imported_modules
            ):
                raise
            continue
        if getattr(value, "__name__", None) != export:
            raise RuntimeError(f"iron.operators.{export} resolved to {value!r}")

    return imported, skipped


def _installed_ops_probe() -> str:
    return (
        """
import importlib.metadata
from pathlib import Path

import iron.operators as operators
from iron.common.context import AIEContext

"""
        + inspect.getsource(_torch_is_missing)
        + "\n"
        + inspect.getsource(probe_operator_modules)
        + f"""
assert importlib.metadata.version("rocm-npu-ops")
import rocm_npu_ops
kernel = AIEContext.base_dir / "aie_kernels" / "generic" / "add.cc"
assert kernel.is_file(), kernel
assert (rocm_npu_ops.kernels_root() / "generic" / "add.cc").is_file()
assert rocm_npu_ops.manifest()["product"] == "rocm-npu-ops"
imported, skipped = probe_operator_modules(
    operators, torch_at_import=frozenset({sorted(_TORCH_AT_IMPORT)!r})
)
assert "iron.operators.elementwise_add.op" in imported
print(
    f"ops import: ok ({{len(imported)}} modules, skipped {{skipped or ['none']}}, {{kernel}})"
)
"""
    )


def _ops_compile_specs() -> tuple[dict[str, object], ...]:
    """Shapes compiled from an installed ops wheel.

    Same catalog as ``dev.py test-iron-ops`` (``host_compile_suite()``).
    Torch-at-import packages stay out of that list. ``host_compile=False``
    operators stay in ``OPERATORS`` for leftover-closed-set.
    """
    from build_tools.iron import OPERATORS, host_compile_suite

    overlap = sorted({operator.name for operator in OPERATORS} & _TORCH_AT_IMPORT)
    if overlap:
        raise RuntimeError(
            "OPERATORS includes torch-at-import packages: " + ", ".join(overlap)
        )
    return tuple(
        {
            "name": operator.name,
            "kwargs": dict(operator.build_kwargs),
            "device": operator.device,
        }
        for operator in host_compile_suite()
    )


def _installed_ops_compile(build_dir: Path) -> str:
    """Python the wheel venv runs: subprocess-per-operator, like test-iron-ops.

    The venv has ``iron`` and the compiler, not ``build_tools``. ELF helpers are
    copied from this tree. The driver itself is a script (argparse + re-exec),
    not pickled ``build_tools`` callables.
    """
    from build_tools.iron import (
        classify_compiled_images,
        compile_job_count,
        require_runtime_image,
    )

    specs = _ops_compile_specs()
    return (
        """
import argparse
import importlib
import inspect
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("IRON_RUNTIME", "cpu")

import rocm_npu_compiler as compiler

compiler.activate_toolchain_env()
compiler.activate_native_libraries()
prefix = Path(compiler.compiler_prefix())
if not (prefix / "include").is_dir():
    raise SystemExit(f"compiler prefix has no include/: {prefix}")

"""
        + inspect.getsource(require_runtime_image)
        + "\n"
        + inspect.getsource(classify_compiled_images)
        + "\n"
        + inspect.getsource(compile_job_count)
        + f"""

SPECS = {specs!r}
DEFAULT_BUILD_DIR = Path({str(build_dir)!r})
DEVICES = None


def _devices():
    global DEVICES
    if DEVICES is None:
        from aie.iron.device import NPU1Col1, NPU2

        DEVICES = {{"npu1": NPU1Col1, "npu2": NPU2}}
    return DEVICES


def compile_one(spec, build_root):
    import aie.utils as aie_utils
    from iron.common import AIEContext, AIEOperatorBase

    devices = _devices()
    name = spec["name"]
    device = spec["device"]
    if device not in devices:
        raise SystemExit(f"{{name}}: unknown device {{device!r}}")
    aie_utils.set_current_device(devices[device]())
    op_dir = Path(build_root) / name
    if op_dir.exists():
        shutil.rmtree(op_dir)
    op_dir.mkdir(parents=True)
    module = importlib.import_module(f"iron.operators.{{name}}.op")
    found = [
        value
        for value in vars(module).values()
        if inspect.isclass(value)
        and issubclass(value, AIEOperatorBase)
        and value.__module__ == module.__name__
    ]
    if len(found) != 1:
        names = ", ".join(sorted(cls.__name__ for cls in found)) or "none"
        raise SystemExit(
            f"{{name}}: expected exactly one operator class, found "
            f"{{len(found)}} ({{names}})"
        )
    cls = found[0]
    params = inspect.signature(cls.__init__).parameters
    unknown = [key for key in spec["kwargs"] if key not in params]
    if unknown:
        raise SystemExit(f"{{name}}: unsupported constructor args {{unknown}}")
    operator = cls(**spec["kwargs"], context=AIEContext(build_dir=op_dir))
    operator.compile()
    try:
        kind, produced = classify_compiled_images(operator.artifacts)
        for path in produced:
            require_runtime_image(path, elf=(kind == "elf"))
    except RuntimeError as error:
        raise SystemExit(f"{{name}}: {{error}}") from error
    print("ops compile: ok", name, *(path.name for path in produced), flush=True)
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
    ]
    return name, subprocess.run(command, capture_output=True, text=True)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    args = parser.parse_args(argv)
    selected = [
        spec
        for spec in SPECS
        if args.operator is None or spec["name"] == args.operator
    ]
    if args.operator and not selected:
        raise SystemExit(f"unknown operator: {{args.operator}}")
    if not selected:
        print("ops compile: ok 0 operators")
        return 0
    workers = compile_job_count(len(selected), args.jobs)
    print(f"ops compile: {{len(selected)}} operators, jobs={{workers}}", flush=True)
    build_root = args.build_dir
    build_root.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        compiled = [compile_one(spec, build_root) for spec in selected]
    else:
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_spawn, spec["name"], build_root) for spec in selected
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
            raise SystemExit("ops compile failed: " + ", ".join(failures))
        compiled = [spec["name"] for spec in selected]
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
