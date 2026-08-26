# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Builds the split ROCm NPU compiler wheel family."""

from __future__ import annotations

import email.parser
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from build_tools import release_policy
from build_tools.archive import extract_archive
from build_tools.artifact import reset_directory
from build_tools.compiler_native import (
    linux_system_runtime,
    relocate_linux_compiler_rpaths,
    windows_system_runtime,
)
from build_tools.manylinux_audit import validate_manylinux_policy
from build_tools.native_dependencies import (
    elf_needed,
    elf_rpath,
    is_loadable_elf,
    is_pe,
    pe_imports,
)
from build_tools.rocm_npu_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_PACKAGE,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    COMPILER_PACKAGE,
    build_wheel,
    find_wheel_path_references,
    remove_distribution_wheels,
)
from build_tools.wheel_payload import (
    copy_file as _copy_file,
)
from build_tools.wheel_payload import (
    write_file as _write_file,
)
from build_tools.wheel_payload import (
    write_package_manifest as _write_package_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
COMPILER_DISTRIBUTIONS = (
    COMPILER_DISTRIBUTION,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_BINDINGS_DISTRIBUTION,
)


def validate_compiler_auditwheel_policy(distribution: str, policy: str) -> str:
    """Checks compiler wheel audit policy against the compiler release contract."""
    if distribution not in (
        COMPILER_CORE_DISTRIBUTION,
        COMPILER_BINDINGS_DISTRIBUTION,
    ):
        raise RuntimeError(f"unexpected Linux compiler wheel distribution: {distribution}")
    try:
        return validate_manylinux_policy(policy, maximum_glibc=(2, 28))
    except RuntimeError as error:
        raise RuntimeError(
            f"{distribution} is not manylinux_2_28 compatible: {policy}"
        ) from error


COMPILER_TOOLS = (
    "aiecc",
    "aie-opt",
    "aie-translate",
    "bootgen",
    "target_model",
    "target_model_rtti",
    "register_database",
    "xclbinutil",
)
# The compiler payload is dominated by large native binaries. Level 1 retains
# deterministic DEFLATE wheels while avoiding most of zlib's single-threaded
# search cost; artifact upload already skips a second compression pass.
COMPILER_WHEEL_COMPRESSION_LEVEL = 1
COMPILER_BINDINGS_RUNTIME_REQUIREMENTS = (
    "aiofiles ==25.1.0",
    "cloudpickle ==3.1.2",
    "ml-dtypes ==0.6.0",
    "numpy ==2.4.6",
    "rich ==15.0.0",
)
CORE_AIE_NATIVE_PREFIXES = (
    "AIEAggregateCAPI",
    "libAIEAggregateCAPI",
    "libmlir_apfloat_wrappers",
    "mlir_async_runtime",
    "mlir_c_runner_utils",
    "mlir_float16_utils",
    "mlir_runner_utils",
    "libmlir_async_runtime",
    "libmlir_c_runner_utils",
    "libmlir_float16_utils",
    "libmlir_runner_utils",
)


def python_abi_tag() -> str:
    return "cp" + release_policy.compiler_python_stable_abi().replace(".", "")


def binding_wheel_tag(platform_tag: str) -> str:
    return f"{python_abi_tag()}-abi3-{platform_tag}"


def compiler_core_wheel_tag(platform_tag: str) -> str:
    return f"py3-none-{platform_tag}"


def compiler_core_shared_library_names(wheel: Path) -> frozenset[str]:
    """Returns Linux shared-library names supplied by the compiler core wheel."""
    with zipfile.ZipFile(wheel) as archive:
        return frozenset(
            PurePosixPath(member).name
            for member in archive.namelist()
            if re.search(r"\.so(?:\.|$)", PurePosixPath(member).name)
        )


def _load_product_manifest(source_product: Path) -> dict[str, object]:
    manifest_path = source_product / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"compiler product manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("product") != "rocm-npu-compiler":
        raise RuntimeError(f"not a rocm-npu-compiler product: {source_product}")
    return manifest


def compiler_product_report(source_product: Path) -> dict[str, object]:
    """Reports the size and ABI split used as the packaging feasibility gate."""
    source_product = Path(source_product)
    _load_product_manifest(source_product)
    files = sorted(path for path in source_product.rglob("*") if path.is_file())
    python_root = source_product / "python"
    binding_files = [
        path for path in files
        if path.is_relative_to(python_root) and not _is_core_aie_native(path)
    ]
    binding_set = set(binding_files)
    core_files = [path for path in files if path not in binding_set]
    extensions = [
        path.relative_to(source_product).as_posix()
        for path in binding_files
        if path.suffix.casefold() in (".pyd", ".so")
    ]
    return {
        "schema_version": 1,
        "product": source_product.name,
        "total": {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        },
        "core": {
            "files": len(core_files),
            "bytes": sum(path.stat().st_size for path in core_files),
        },
        "bindings": {
            "files": len(binding_files),
            "bytes": sum(path.stat().st_size for path in binding_files),
            "extensions": extensions,
        },
        "maximum_relative_path": max(
            (len(path.relative_to(source_product).as_posix()) for path in files),
            default=0,
        ),
        "largest_files": [
            {
                "path": path.relative_to(source_product).as_posix(),
                "size": path.stat().st_size,
            }
            for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True)[:20]
        ],
    }


def _stage_file(source: Path, destination: Path) -> None:
    """Hard-links immutable staging inputs, falling back across filesystems."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        resolved = source.resolve()
        if not resolved.is_file():
            raise RuntimeError(
                f"compiler wheel payload symlink does not resolve to a file: {source}"
            )
        source = resolved
    try:
        os.link(source, destination)
    except OSError:
        _copy_file(source, destination)


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    hardlink: bool = True,
) -> None:
    if not source.is_dir():
        raise RuntimeError(f"compiler wheel source directory missing: {source}")
    stage = _stage_file if hardlink else _copy_file
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        stage(path, destination / path.relative_to(source))


def _copy_product_licenses(source_product: Path, package_root: Path) -> None:
    licenses = source_product / "licenses"
    if not licenses.is_dir() or not any(licenses.rglob("*")):
        raise RuntimeError(f"compiler product license material missing: {licenses}")
    _copy_tree(licenses, package_root / "licenses")


def _core_module() -> str:
    functions = "\n\n".join(
        f"def {name.replace('-', '_')}() -> int:\n"
        f"    return _run_tool({name!r})"
        for name in COMPILER_TOOLS
    )
    return f'''"""Paths and launchers for the packaged ROCm NPU compiler."""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

_dll_directory_handle = None
_preloaded_libraries = []


def compiler_prefix() -> Path:
    return Path(__file__).resolve().parent


def toolchain_env(base: dict[str, str] | None = None) -> dict[str, str]:
    prefix = compiler_prefix()
    env = dict(os.environ if base is None else base)
    path_entries = [prefix / "bin"]
    if os.name == "nt":
        path_entries.append(prefix / "lib")
    env["PATH"] = os.pathsep.join(
        [*(str(path) for path in path_entries), env.get("PATH", "")]
    )
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(prefix / "lib"), env.get("LD_LIBRARY_PATH", "")]
        )
    env["PEANO_INSTALL_DIR"] = str(prefix / "peano" / "llvm-aie")
    env["AIEOPT_DIR"] = str(prefix / "third_party" / "aie_api")
    env["ROCM_NPU_COMPILER_PREFIX"] = str(prefix)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def activate_toolchain_env() -> None:
    """Apply toolchain_env() to this process, keeping values already set.

    The in-process IRON JIT resolves PEANO_INSTALL_DIR when `aie` is imported,
    which no subprocess environment can reach.
    """
    prefix = compiler_prefix()
    path_entries = [prefix / "bin"]
    if os.name == "nt":
        path_entries.append(prefix / "lib")
    os.environ["PATH"] = os.pathsep.join(
        [*(str(path) for path in path_entries), os.environ.get("PATH", "")]
    )
    if os.name != "nt":
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(prefix / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]
        )
    os.environ.setdefault("PEANO_INSTALL_DIR", str(prefix / "peano" / "llvm-aie"))
    os.environ.setdefault("AIEOPT_DIR", str(prefix / "third_party" / "aie_api"))
    os.environ.setdefault("ROCM_NPU_COMPILER_PREFIX", str(prefix))


def native_library_dir() -> Path:
    return compiler_prefix() / "lib" / "aie"


def activate_native_libraries() -> Path:
    global _dll_directory_handle
    library_dir = native_library_dir()
    if os.name == "nt":
        if _dll_directory_handle is None:
            _dll_directory_handle = os.add_dll_directory(str(library_dir))
    elif not _preloaded_libraries:
        for library in sorted(library_dir.glob("*.so*")):
            _preloaded_libraries.append(
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            )
    return library_dir


def _tool_path(name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    path = compiler_prefix() / "bin" / f"{{name}}{{suffix}}"
    if not path.is_file():
        raise RuntimeError(f"ROCm NPU compiler tool missing: {{path}}")
    return path


def _run_tool(name: str) -> int:
    activate_native_libraries()
    return subprocess.call([str(_tool_path(name)), *os.sys.argv[1:]], env=toolchain_env())


{functions}
'''


def _is_core_aie_native(path: Path) -> bool:
    name = path.name
    return (
        path.suffix.casefold() == ".dll" or ".so" in name
    ) and name.startswith(
        CORE_AIE_NATIVE_PREFIXES
    )


def _binding_bootstrap() -> str:
    return (
        "\n# ROCm NPU compiler split-wheel native activation.\n"
        "from rocm_npu_compiler import (\n"
        "    activate_native_libraries as _rocm_npu_activate_native_libraries,\n"
        "    activate_toolchain_env as _rocm_npu_activate_toolchain_env,\n"
        ")\n"
        "_rocm_npu_activate_native_libraries()\n"
        "_rocm_npu_activate_toolchain_env()\n"
        "try:\n"
        "    from rocm_npu_runtime import (\n"
        "        activate_iron_host as _rocm_npu_activate_iron_host,\n"
        "    )\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    _rocm_npu_activate_iron_host()\n"
    )


def _common_manifest(
    source_manifest: dict[str, object],
    *,
    product: str,
    artifact: str,
    wheel_tag: str,
    version: str,
    requires: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "product": product,
        "artifact": artifact,
        "version": version,
        "wheel_tag": wheel_tag,
        "requires": list(requires),
        "sources": source_manifest.get("sources", {}),
        "dependencies": source_manifest.get("dependencies", {}),
        "toolchain": source_manifest.get("toolchain", {}),
        "peano": source_manifest.get("peano", {}),
    }


def stage_compiler_core(
    source_product: Path,
    destination: Path,
    *,
    version: str,
    platform_tag: str,
) -> Path:
    source_product = Path(source_product)
    source_manifest = _load_product_manifest(source_product)
    tree = Path(destination)
    reset_directory(tree, tree.parent)
    package_root = tree / COMPILER_PACKAGE
    package_root.mkdir(parents=True)
    is_linux = platform_tag.startswith("manylinux_")
    stage = _copy_file if is_linux else _stage_file
    for child in sorted(source_product.iterdir()):
        if child.name in ("python", "manifest.json"):
            continue
        if child.is_dir():
            _copy_tree(
                child,
                package_root / child.name,
                hardlink=not is_linux,
            )
        elif child.is_file():
            stage(child, package_root / child.name)
    source_mlir_libs = source_product / "python" / "aie" / "_mlir_libs"
    if source_mlir_libs.is_dir():
        for path in sorted(item for item in source_mlir_libs.iterdir() if item.is_file()):
            if _is_core_aie_native(path):
                stage(path, package_root / "lib" / "aie" / path.name)
    _write_file(package_root / "__init__.py", _core_module())
    tag = compiler_core_wheel_tag(platform_tag)
    _write_package_manifest(
        package_root,
        _common_manifest(
            source_manifest,
            product=COMPILER_CORE_DISTRIBUTION,
            artifact=f"rocm-npu-compiler-core-{platform_tag}",
            wheel_tag=tag,
            version=version,
            requires=(),
        ),
    )
    if is_linux:
        # AIE support libraries move from python/aie/_mlir_libs in the product
        # to lib/aie in the core wheel. Recalculate RPATHs after that move so
        # auditwheel and the dynamic loader resolve the wheel-internal closure.
        relocate_linux_compiler_rpaths(package_root)
    return tree


def _binding_extensions(aie_root: Path) -> list[Path]:
    return [
        path for path in aie_root.rglob("*")
        if path.is_file()
        and (
            path.suffix.casefold() == ".pyd"
            or (
                path.suffix.casefold() == ".so"
                and not path.name.startswith("lib")
                and not _is_core_aie_native(path)
            )
        )
    ]


def _validate_binding_abi(aie_root: Path) -> None:
    extensions = _binding_extensions(aie_root)
    if not extensions:
        raise RuntimeError(f"compiler bindings contain no native extensions: {aie_root}")
    non_abi3 = []
    for path in extensions:
        name = path.name
        if path.suffix.casefold() == ".so" and not name.endswith(".abi3.so"):
            non_abi3.append(name)
        elif path.suffix.casefold() == ".pyd" and re.search(
            r"(?:cp|cpython-)[0-9]{2,3}", name
        ):
            non_abi3.append(name)
    if non_abi3:
        raise RuntimeError(
            "compiler binding extensions are not stable-ABI modules: "
            + ", ".join(sorted(non_abi3)[:5])
        )


def stage_compiler_bindings(
    source_product: Path,
    python_root: Path,
    destination: Path,
    *,
    version: str,
    platform_tag: str,
) -> Path:
    source_product = Path(source_product)
    source_manifest = _load_product_manifest(source_product)
    source_aie = Path(python_root) / "aie"
    _validate_binding_abi(source_aie)
    tree = Path(destination)
    reset_directory(tree, tree.parent)
    for path in sorted(item for item in source_aie.rglob("*") if item.is_file()):
        relative = path.relative_to(source_aie)
        if path.suffix.casefold() in (".a", ".lib") or _is_core_aie_native(path):
            continue
        destination = tree / "aie" / relative
        # This file receives the wheel bootstrap below; all other staged inputs
        # remain immutable and can safely share storage with the source product.
        if relative == Path("__init__.py"):
            _copy_file(path, destination)
        else:
            _stage_file(path, destination)
    init = tree / "aie" / "__init__.py"
    init.write_text(
        (init.read_text(encoding="utf-8") if init.is_file() else "")
        + _binding_bootstrap(),
        encoding="utf-8",
    )
    metadata_root = tree / COMPILER_BINDINGS_PACKAGE
    metadata_root.mkdir(parents=True)
    _write_file(metadata_root / "__init__.py", "")
    _copy_product_licenses(source_product, metadata_root)
    tag = binding_wheel_tag(platform_tag)
    requirements = (
        f"{COMPILER_CORE_DISTRIBUTION} =={version}",
        *COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    )
    _write_package_manifest(
        metadata_root,
        _common_manifest(
            source_manifest,
            product=COMPILER_BINDINGS_DISTRIBUTION,
            artifact=f"rocm-npu-compiler-bindings-abi3-{platform_tag}",
            wheel_tag=tag,
            version=version,
            requires=requirements,
        )
        | {
            "python_stable_abi": release_policy.compiler_python_stable_abi(),
            "supported_python_minors": list(
                release_policy.compiler_python_versions()
            ),
        },
    )
    return tree


def stage_compiler_meta(
    source_product: Path,
    destination: Path,
    *,
    version: str,
) -> Path:
    source_product = Path(source_product)
    source_manifest = _load_product_manifest(source_product)
    tree = Path(destination)
    reset_directory(tree, tree.parent)
    package_root = tree / "rocm_npu_compiler_meta"
    package_root.mkdir(parents=True)
    _write_file(package_root / "__init__.py", "")
    _copy_product_licenses(source_product, package_root)
    requires = (
        f"{COMPILER_CORE_DISTRIBUTION} =={version}",
        f"{COMPILER_BINDINGS_DISTRIBUTION} =={version}",
    )
    _write_package_manifest(
        package_root,
        _common_manifest(
            source_manifest,
            product=COMPILER_DISTRIBUTION,
            artifact="rocm-npu-compiler-meta",
            wheel_tag="py3-none-any",
            version=version,
            requires=requires,
        ),
    )
    return tree


def build_compiler_wheel_family(
    source_product: Path,
    binding_python_root: Path,
    stage_directory: Path,
    output_directory: Path,
    *,
    version: str,
    platform_tag: str,
    include_shared: bool = True,
    include_meta: bool = True,
) -> dict[str, Path]:
    """Builds a platform core plus one CPython 3.12 stable-ABI bindings wheel."""
    started = time.perf_counter()
    binding_python_root = Path(binding_python_root)
    if not (binding_python_root / "aie").is_dir():
        raise RuntimeError(
            f"compiler binding aie package missing: {binding_python_root}"
        )
    output_directory = Path(output_directory)
    remove_distribution_wheels(
        output_directory,
        (
            *((COMPILER_DISTRIBUTION, COMPILER_CORE_DISTRIBUTION) if include_shared else ()),
            *((COMPILER_DISTRIBUTION,) if include_meta and not include_shared else ()),
            COMPILER_BINDINGS_DISTRIBUTION,
        ),
    )
    built: dict[str, Path] = {}
    core_tree = None
    if include_shared:
        stage_started = time.perf_counter()
        core_tree = stage_compiler_core(
            source_product,
            Path(stage_directory) / "core",
            version=version,
            platform_tag=platform_tag,
        )
        print(f"package compiler core staging: "
              f"{time.perf_counter() - stage_started:.2f}s")
    meta_tree = None
    if include_meta:
        stage_started = time.perf_counter()
        meta_tree = stage_compiler_meta(
            source_product, Path(stage_directory) / "meta", version=version
        )
        print(f"package compiler metadata staging: "
              f"{time.perf_counter() - stage_started:.2f}s")
    stage_started = time.perf_counter()
    tree = stage_compiler_bindings(
        source_product,
        binding_python_root,
        Path(stage_directory) / "bindings-abi3",
        version=version,
        platform_tag=platform_tag,
    )
    print(f"package compiler bindings staging: "
          f"{time.perf_counter() - stage_started:.2f}s")
    requirements = (
        f"{COMPILER_CORE_DISTRIBUTION} =={version}",
        *COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    )
    build_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        core_future = None
        if core_tree is not None:
            core_scripts = {
                tool: f"{COMPILER_PACKAGE}:{tool.replace('-', '_')}"
                for tool in COMPILER_TOOLS
            }
            core_future = executor.submit(
                build_wheel,
                core_tree,
                COMPILER_CORE_DISTRIBUTION,
                version,
                output_directory,
                summary="ROCm NPU compiler toolchain core",
                tag=compiler_core_wheel_tag(platform_tag),
                console_scripts=core_scripts,
                requires_python=release_policy.compiler_requires_python(),
                compression_level=COMPILER_WHEEL_COMPRESSION_LEVEL,
            )
        bindings_future = executor.submit(
            build_wheel,
            tree,
            COMPILER_BINDINGS_DISTRIBUTION,
            version,
            output_directory,
            summary="ROCm NPU compiler bindings for CPython 3.12+",
            tag=binding_wheel_tag(platform_tag),
            requires=requirements,
            requires_python=release_policy.compiler_requires_python(),
            compression_level=COMPILER_WHEEL_COMPRESSION_LEVEL,
        )
        if meta_tree is not None:
            meta_requires = (
                f"{COMPILER_CORE_DISTRIBUTION} =={version}",
                f"{COMPILER_BINDINGS_DISTRIBUTION} =={version}",
            )
            built[COMPILER_DISTRIBUTION] = build_wheel(
                meta_tree,
                COMPILER_DISTRIBUTION,
                version,
                output_directory,
                summary="ROCm NPU compiler",
                tag="py3-none-any",
                requires=meta_requires,
                requires_python=release_policy.compiler_requires_python(),
                root_is_purelib=True,
            )
        if core_future is not None:
            built[COMPILER_CORE_DISTRIBUTION] = core_future.result()
        built[COMPILER_BINDINGS_DISTRIBUTION] = bindings_future.result()
    print(f"package compiler wheel compression: "
          f"{time.perf_counter() - build_started:.2f}s")
    print(f"package compiler wheels elapsed: {time.perf_counter() - started:.2f}s")
    return built


def verify_compiler_binding_stable_abi(wheel: Path) -> None:
    """Uses abi3audit to verify exported CPython symbols against the 3.12 ABI."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "abi3audit",
            "--strict",
            "--summary",
            "--assume-minimum-abi3",
            release_policy.compiler_python_stable_abi(),
            str(wheel),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        details = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"compiler bindings fail abi3audit:\n{details}")


def current_python_minor() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _wheel_distribution(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise RuntimeError(f"expected one METADATA file in wheel: {wheel}")
        return str(
            email.parser.Parser()
            .parsestr(archive.read(names[0]).decode("utf-8"))["Name"]
        )


def _extract_wheel(wheel: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    extract_archive(wheel, destination)
    return destination


def verify_windows_compiler_wheels(
    wheels: list[Path],
) -> dict[str, list[str]]:
    """Requires every compiler PE import to resolve from core/bindings or Windows."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extracted = []
        for index, wheel in enumerate(wheels):
            distribution = _wheel_distribution(wheel)
            extracted.append(
                (distribution, _extract_wheel(wheel, root / str(index)))
            )
        core_roots = [
            tree for distribution, tree in extracted
            if distribution == COMPILER_CORE_DISTRIBUTION
        ]
        if len(core_roots) != 1:
            raise RuntimeError("Windows compiler wheel validation requires one core wheel")
        core = core_roots[0]
        core_names = {
            path.name.casefold()
            for path in core.rglob("*")
            if path.is_file() and is_pe(path)
        }
        graph: dict[str, list[str]] = {}
        missing = []
        versioned_python_imports = []
        stable_python_imports = []
        for distribution, tree in extracted:
            if distribution not in (
                COMPILER_CORE_DISTRIBUTION,
                COMPILER_BINDINGS_DISTRIBUTION,
            ):
                continue
            own_names = {
                path.name.casefold()
                for path in tree.rglob("*")
                if path.is_file() and is_pe(path)
            }
            allowed = own_names | core_names
            for path in sorted(
                item for item in tree.rglob("*")
                if item.is_file() and is_pe(item)
            ):
                imports = pe_imports(path)
                key = f"{distribution}/{path.relative_to(tree).as_posix()}"
                graph[key] = imports
                if distribution == COMPILER_BINDINGS_DISTRIBUTION:
                    versioned_python_imports.extend(
                        f"{key} -> {name}" for name in imports
                        if re.fullmatch(r"python3[0-9]{1,2}\.dll", name.casefold())
                    )
                    stable_python_imports.extend(
                        f"{key} -> {name}" for name in imports
                        if name.casefold() == "python3.dll"
                    )
                missing.extend(
                    f"{key} -> {name}" for name in imports
                    if name.casefold() not in allowed
                    and not windows_system_runtime(name)
                )
        if versioned_python_imports:
            raise RuntimeError(
                "ROCm NPU compiler bindings import a versioned CPython DLL:\n  "
                + "\n  ".join(sorted(versioned_python_imports))
            )
        if not stable_python_imports:
            raise RuntimeError(
                "ROCm NPU compiler bindings do not import stable-ABI python3.dll"
            )
        if missing:
            raise RuntimeError(
                "ROCm NPU compiler Windows dependency closure is incomplete:\n  "
                + "\n  ".join(sorted(missing))
            )
        return graph


def verify_linux_compiler_wheels(
    wheels: list[Path],
) -> dict[str, list[str]]:
    """Checks compiler ELF closure and rejects absolute runtime search paths."""
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to verify Linux compiler wheels")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extracted = []
        for index, wheel in enumerate(wheels):
            distribution = _wheel_distribution(wheel)
            extracted.append(
                (distribution, _extract_wheel(wheel, root / str(index)))
            )
        core_roots = [
            tree for distribution, tree in extracted
            if distribution == COMPILER_CORE_DISTRIBUTION
        ]
        if len(core_roots) != 1:
            raise RuntimeError("Linux compiler wheel validation requires one core wheel")
        core_names = {
            path.name for path in core_roots[0].rglob("*")
            if path.is_file() and is_loadable_elf(path)
        }
        graph: dict[str, list[str]] = {}
        missing = []
        absolute_rpaths = []
        for distribution, tree in extracted:
            if distribution not in (
                COMPILER_CORE_DISTRIBUTION,
                COMPILER_BINDINGS_DISTRIBUTION,
            ):
                continue
            own_names = {
                path.name for path in tree.rglob("*")
                if path.is_file() and is_loadable_elf(path)
            }
            allowed = own_names | core_names
            for path in sorted(
                item for item in tree.rglob("*")
                if item.is_file() and is_loadable_elf(item)
            ):
                key = f"{distribution}/{path.relative_to(tree).as_posix()}"
                needed = elf_needed(path, patchelf=patchelf)
                rpath = elf_rpath(path, patchelf=patchelf)
                graph[key] = sorted(needed)
                absolute_rpaths.extend(
                    f"{key} -> {entry}"
                    for entry in rpath.split(":")
                    if entry.startswith("/")
                )
                missing.extend(
                    f"{key} -> {name}" for name in needed
                    if name not in allowed and not linux_system_runtime(name)
                )
        if absolute_rpaths:
            raise RuntimeError(
                "ROCm NPU compiler wheels contain absolute ELF RPATHs:\n  "
                + "\n  ".join(sorted(absolute_rpaths))
            )
        if missing:
            raise RuntimeError(
                "ROCm NPU compiler Linux dependency closure is incomplete:\n  "
                + "\n  ".join(sorted(missing))
            )
        return graph


def verify_compiler_wheel_path_references(
    wheels: list[Path], forbidden_paths: tuple[Path, ...]
) -> None:
    references = find_wheel_path_references(
        {str(index): wheel for index, wheel in enumerate(wheels)},
        forbidden_paths,
    )
    text_suffixes = (
        ".cmake",
        ".json",
        ".md",
        ".pc",
        ".py",
        ".pth",
        ".txt",
        ".yaml",
        ".yml",
    )
    text_references = [
        reference for reference in references
        if reference.member.casefold().endswith(text_suffixes)
    ]
    if text_references:
        details = "\n  ".join(
            f"{item.wheel.name}:{item.member} -> {item.forbidden_path}"
            for item in text_references
        )
        raise RuntimeError(
            "ROCm NPU compiler wheel text payload contains local paths:\n  "
            + details
        )
