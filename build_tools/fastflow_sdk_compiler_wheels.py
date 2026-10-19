# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Builds the split FastFlow SDK compiler wheel family."""

from __future__ import annotations

import email.parser
import filecmp
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
from build_tools.fastflow_sdk_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_PACKAGE,
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    COMPILER_PACKAGE,
    build_wheel,
    find_wheel_path_references,
    remove_distribution_wheels,
)
from build_tools.manylinux_audit import validate_manylinux_policy
from build_tools.native_dependencies import (
    elf_needed,
    elf_rpath,
    is_loadable_elf,
    is_pe,
    pe_imports,
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
        raise RuntimeError(
            f"unexpected Linux compiler wheel distribution: {distribution}"
        )
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
# The compiler payload is dominated by large native binaries. Maximum DEFLATE
# materially reduces release and PyPI storage while retaining deterministic
# wheels; artifact upload already skips a second compression pass.
COMPILER_WHEEL_COMPRESSION_LEVEL = 9
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
WINDOWS_PEANO_ALIAS_FORWARDING = {
    "clang.exe": "clang++.exe",
    "clang-cl.exe": "clang++.exe",
    "clang-cpp.exe": "clang++.exe",
    "ld.lld.exe": "lld.exe",
    "ld64.lld.exe": "lld.exe",
    "lld-link.exe": "lld.exe",
    "wasm-ld.exe": "lld.exe",
    "llvm-readelf.exe": "llvm-readobj.exe",
}
LINUX_PEANO_ALIAS_FORWARDING = {
    "ld.lld": ("lld", ("-flavor", "gnu")),
    "ld64.lld": ("lld", ("-flavor", "darwin")),
    "lld-link": ("lld", ("-flavor", "link")),
    "wasm-ld": ("lld", ("-flavor", "wasm")),
    "llvm-readelf": ("llvm-readobj", ("--elf-output-style=GNU",)),
}
WINDOWS_TOOL_ALIAS_LAUNCHER_SOURCE = (
    Path(__file__).with_name("native") / "compiler_tool_alias_launcher.cpp"
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
    if (
        not isinstance(manifest, dict)
        or manifest.get("product") != "fastflow-sdk-compiler"
    ):
        raise RuntimeError(f"not a fastflow-sdk-compiler product: {source_product}")
    return manifest


def compiler_product_report(source_product: Path) -> dict[str, object]:
    """Reports the size and ABI split used as the packaging feasibility gate."""
    source_product = Path(source_product)
    _load_product_manifest(source_product)
    files = sorted(path for path in source_product.rglob("*") if path.is_file())
    python_root = source_product / "python"
    binding_files = [
        path
        for path in files
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
            for path in sorted(
                files, key=lambda item: item.stat().st_size, reverse=True
            )[:20]
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


def _compile_windows_tool_alias_launcher(output: Path) -> None:
    compiler = os.environ.get("CXX") or shutil.which("clang-cl")
    if not compiler:
        raise RuntimeError(
            "CXX or clang-cl is required to build Windows compiler tool aliases"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    object_file = output.with_suffix(".obj")
    result = subprocess.run(
        [
            compiler.strip('"'),
            "/nologo",
            "/std:c++17",
            "/O1",
            "/EHsc",
            "/MT",
            "/Brepro",
            str(WINDOWS_TOOL_ALIAS_LAUNCHER_SOURCE),
            f"/Fo{object_file}",
            f"/Fe{output}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode or not output.is_file() or not is_pe(output):
        details = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"failed to build Windows compiler tool alias:\n{details}")


def _forward_windows_peano_aliases(package_root: Path) -> tuple[str, ...]:
    """Replaces byte-identical LLVM aliases with flavor-preserving launchers."""
    binary_directory = package_root / "peano" / "llvm-aie" / "bin"
    replacements = []
    for alias_name, canonical_name in WINDOWS_PEANO_ALIAS_FORWARDING.items():
        alias = binary_directory / alias_name
        if not alias.is_file():
            continue
        canonical = binary_directory / canonical_name
        if not canonical.is_file():
            raise RuntimeError(
                f"Peano alias {alias_name} has no canonical {canonical_name}"
            )
        if not filecmp.cmp(alias, canonical, shallow=False):
            raise RuntimeError(
                f"Peano alias {alias_name} differs from canonical {canonical_name}"
            )
        replacements.append(alias)
    if not replacements:
        return ()
    with tempfile.TemporaryDirectory(
        prefix="fastflow-sdk-tool-alias-", dir=package_root.parent
    ) as temporary:
        launcher = Path(temporary) / "fastflow-sdk-tool-alias.exe"
        _compile_windows_tool_alias_launcher(launcher)
        for alias in replacements:
            alias.unlink()
            _copy_file(launcher, alias)
    return tuple(alias.relative_to(package_root).as_posix() for alias in replacements)


def _forward_linux_peano_aliases(package_root: Path) -> tuple[str, ...]:
    """Replaces byte-identical LLVM aliases with flavor-preserving scripts."""
    binary_directory = package_root / "peano" / "llvm-aie" / "bin"
    forwarded = []
    for alias_name, (canonical_name, arguments) in LINUX_PEANO_ALIAS_FORWARDING.items():
        alias = binary_directory / alias_name
        if not alias.is_file():
            continue
        canonical = binary_directory / canonical_name
        if not canonical.is_file():
            raise RuntimeError(
                f"Peano alias {alias_name} has no canonical {canonical_name}"
            )
        if not filecmp.cmp(alias, canonical, shallow=False):
            raise RuntimeError(
                f"Peano alias {alias_name} differs from canonical {canonical_name}"
            )
        argument_text = " ".join(arguments)
        alias.write_text(
            "#!/bin/sh\n"
            'launcher=$(readlink -f -- "$0") || exit 127\n'
            'directory=$(CDPATH= cd -- "$(dirname -- "$launcher")" && pwd)\n'
            f'exec "$directory/{canonical_name}" {argument_text} "$@"\n',
            encoding="utf-8",
        )
        alias.chmod(0o755)
        forwarded.append(alias.relative_to(package_root).as_posix())
    return tuple(forwarded)


def _remove_redundant_linux_llvm_library(package_root: Path) -> tuple[str, ...]:
    library_directory = package_root / "peano" / "llvm-aie" / "lib"
    canonical = library_directory / "libLLVM.so"
    redundant_libraries = sorted(
        path
        for path in library_directory.glob("libLLVM-*.so")
        if path.is_file()
        and re.fullmatch(r"libLLVM-[0-9]+(?:\.[0-9]+)*\.so", path.name)
    )
    if not redundant_libraries:
        return ()
    if not canonical.is_file():
        names = ", ".join(path.name for path in redundant_libraries)
        verb = "has" if len(redundant_libraries) == 1 else "have"
        raise RuntimeError(f"{names} {verb} no canonical libLLVM.so")
    for redundant in redundant_libraries:
        if not filecmp.cmp(redundant, canonical, shallow=False):
            raise RuntimeError(f"{redundant.name} differs from canonical libLLVM.so")
    for redundant in redundant_libraries:
        redundant.unlink()
    return tuple(
        path.relative_to(package_root).as_posix() for path in redundant_libraries
    )


def _strip_linux_compiler(package_root: Path) -> dict[str, int]:
    targets = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file() and is_loadable_elf(path)
    )
    if not targets:
        return {"files": 0, "bytes_before": 0, "bytes_after": 0}
    strip = shutil.which("strip")
    if not strip:
        raise RuntimeError("strip is required to prepare Linux compiler wheels")
    bytes_before = sum(path.stat().st_size for path in targets)
    result = subprocess.run(
        [strip, "--strip-unneeded", *[str(path) for path in targets]],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        details = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"failed to strip Linux compiler payload:\n{details}")
    return {
        "files": len(targets),
        "bytes_before": bytes_before,
        "bytes_after": sum(path.stat().st_size for path in targets),
    }


def _core_module() -> str:
    functions = "\n\n".join(
        f"def {name.replace('-', '_')}() -> int:\n    return _run_tool({name!r})"
        for name in COMPILER_TOOLS
    )
    return f'''"""Paths and launchers for the packaged FastFlow SDK compiler."""
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
    env["FASTFLOW_SDK_COMPILER_PREFIX"] = str(prefix)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def activate_toolchain_env() -> None:
    """Apply toolchain_env() to this process, keeping values already set."""
    env = toolchain_env()
    for name in ("PATH", "LD_LIBRARY_PATH"):
        if name in env:
            os.environ[name] = env[name]
    for name in ("PEANO_INSTALL_DIR", "AIEOPT_DIR", "FASTFLOW_SDK_COMPILER_PREFIX"):
        os.environ.setdefault(name, env[name])


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
        raise RuntimeError(f"FastFlow SDK compiler tool missing: {{path}}")
    return path


def _run_tool(name: str) -> int:
    activate_native_libraries()
    return subprocess.call([str(_tool_path(name)), *os.sys.argv[1:]], env=toolchain_env())


{functions}
'''


def _is_core_aie_native(path: Path) -> bool:
    name = path.name
    return (path.suffix.casefold() == ".dll" or ".so" in name) and name.startswith(
        CORE_AIE_NATIVE_PREFIXES
    )


def _binding_bootstrap() -> str:
    return (
        "\n# FastFlow SDK compiler split-wheel native activation.\n"
        "from fastflow_sdk_compiler import (\n"
        "    activate_native_libraries as _fastflow_sdk_activate_native_libraries,\n"
        "    activate_toolchain_env as _fastflow_sdk_activate_toolchain_env,\n"
        ")\n"
        "_fastflow_sdk_activate_native_libraries()\n"
        "_fastflow_sdk_activate_toolchain_env()\n"
        "try:\n"
        "    from fastflow_sdk_runtime import (\n"
        "        activate_iron_host as _fastflow_sdk_activate_iron_host,\n"
        "    )\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    _fastflow_sdk_activate_iron_host()\n"
    )


def _rewrite_configure_install_path(configure_py: Path) -> None:
    """Make ``aie.utils.configure`` safe to ship in the bindings wheel.

    Upstream walks three directories up from ``aie/utils``, which is the
    mlir-aie install prefix when ``aie`` lives at ``$PREFIX/python/aie``. The
    FastFlow SDK bindings wheel puts ``aie`` at site-packages root, so that walk
    lands on the venv ``Lib`` / ``lib/pythonX.Y`` directory. C++ kernel
    compiles then fail looking for headers in ``<venv>/Lib/include``. Headers
    actually live in ``fastflow_sdk_compiler/include``, which
    ``FASTFLOW_SDK_COMPILER_PREFIX`` already names.

    mlir-aie v1.4.4 also records ``LLVM_TOOLS_BINARY_DIR`` in this file. That
    is the build machine's ``build/llvm/bin``. The installed compiler already
    ships ``llvm-nm`` and ``llvm-objcopy`` in its own ``bin``, and a missing
    directory is skipped, so the wheel stores an empty directory instead.
    """
    text = configure_py.read_text(encoding="utf-8").replace("\r\n", "\n")
    if 'os.getenv("FASTFLOW_SDK_COMPILER_PREFIX")' not in text:
        needle = "def install_path():\n"
        if needle not in text:
            raise RuntimeError(
                "cannot rewrite aie.utils.configure.install_path: "
                f"function missing in {configure_py}"
            )
        replacement = (
            "def install_path():\n"
            '    prefix = os.getenv("FASTFLOW_SDK_COMPILER_PREFIX")\n'
            '    if prefix and os.path.isdir(os.path.join(prefix, "include")):\n'
            "        return os.path.realpath(prefix)\n"
        )
        text = text.replace(needle, replacement, 1)
    text = re.sub(
        r'(?m)^llvm_tools_binary_dir\s*=\s*r?"""[\s\S]*?"""\s*$',
        'llvm_tools_binary_dir = ""',
        text,
        count=1,
    )
    configure_py.write_text(text, encoding="utf-8")


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
        for path in sorted(
            item for item in source_mlir_libs.iterdir() if item.is_file()
        ):
            if _is_core_aie_native(path):
                stage(path, package_root / "lib" / "aie" / path.name)
    removed_redundant_files = ()
    strip_report = {"files": 0, "bytes_before": 0, "bytes_after": 0}
    if is_linux:
        # AIE support libraries move from python/aie/_mlir_libs in the product
        # to lib/aie in the core wheel. Recalculate RPATHs after that move so
        # auditwheel and the dynamic loader resolve the wheel-internal closure.
        relocate_linux_compiler_rpaths(package_root)
        removed_redundant_files = _remove_redundant_linux_llvm_library(package_root)
        forwarded_aliases = _forward_linux_peano_aliases(package_root)
        strip_report = _strip_linux_compiler(package_root)
    else:
        forwarded_aliases = _forward_windows_peano_aliases(package_root)
    _write_file(package_root / "__init__.py", _core_module())
    tag = compiler_core_wheel_tag(platform_tag)
    _write_package_manifest(
        package_root,
        _common_manifest(
            source_manifest,
            product=COMPILER_CORE_DISTRIBUTION,
            artifact=f"fastflow-sdk-compiler-core-{platform_tag}",
            wheel_tag=tag,
            version=version,
            requires=(),
        )
        | {
            "forwarded_peano_aliases": list(forwarded_aliases),
            "removed_redundant_files": list(removed_redundant_files),
            "strip": strip_report,
        },
    )
    return tree


def _binding_extensions(aie_root: Path) -> list[Path]:
    return [
        path
        for path in aie_root.rglob("*")
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
        raise RuntimeError(
            f"compiler bindings contain no native extensions: {aie_root}"
        )
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
        # These files are rewritten after staging; copy so the source product
        # (and hardlinked siblings) stay immutable.
        if relative in (Path("__init__.py"), Path("utils") / "configure.py"):
            _copy_file(path, destination)
        else:
            _stage_file(path, destination)
    configure_py = tree / "aie" / "utils" / "configure.py"
    if configure_py.is_file():
        _rewrite_configure_install_path(configure_py)
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
            artifact=f"fastflow-sdk-compiler-bindings-abi3-{platform_tag}",
            wheel_tag=tag,
            version=version,
            requires=requirements,
        )
        | {
            "python_stable_abi": release_policy.compiler_python_stable_abi(),
            "supported_python_minors": list(release_policy.compiler_python_versions()),
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
    package_root = tree / "fastflow_sdk_compiler_meta"
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
            artifact="fastflow-sdk-compiler-meta",
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
            *(
                (COMPILER_DISTRIBUTION, COMPILER_CORE_DISTRIBUTION)
                if include_shared
                else ()
            ),
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
        print(
            f"package compiler core staging: {time.perf_counter() - stage_started:.2f}s"
        )
    meta_tree = None
    if include_meta:
        stage_started = time.perf_counter()
        meta_tree = stage_compiler_meta(
            source_product, Path(stage_directory) / "meta", version=version
        )
        print(
            f"package compiler metadata staging: "
            f"{time.perf_counter() - stage_started:.2f}s"
        )
    stage_started = time.perf_counter()
    tree = stage_compiler_bindings(
        source_product,
        binding_python_root,
        Path(stage_directory) / "bindings-abi3",
        version=version,
        platform_tag=platform_tag,
    )
    print(
        f"package compiler bindings staging: {time.perf_counter() - stage_started:.2f}s"
    )
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
                summary="FastFlow SDK compiler toolchain core",
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
            summary="FastFlow SDK compiler bindings for CPython 3.12+",
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
                summary="FastFlow SDK compiler",
                tag="py3-none-any",
                requires=meta_requires,
                requires_python=release_policy.compiler_requires_python(),
                root_is_purelib=True,
            )
        if core_future is not None:
            built[COMPILER_CORE_DISTRIBUTION] = core_future.result()
        built[COMPILER_BINDINGS_DISTRIBUTION] = bindings_future.result()
    print(
        f"package compiler wheel compression: "
        f"{time.perf_counter() - build_started:.2f}s"
    )
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
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise RuntimeError(f"expected one METADATA file in wheel: {wheel}")
        return str(
            email.parser.Parser().parsestr(archive.read(names[0]).decode("utf-8"))[
                "Name"
            ]
        )


def _extract_wheel(wheel: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    extract_archive(wheel, destination)
    return destination


def _verify_windows_tool_alias_launchers(core: Path) -> None:
    package_root = core / COMPILER_PACKAGE
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aliases = manifest.get("forwarded_peano_aliases", ())
    if not isinstance(aliases, list):
        raise RuntimeError("forwarded_peano_aliases must be a list")
    binary_directory = package_root / "peano" / "llvm-aie" / "bin"

    def run(tool: str, *arguments: str, cwd: Path | None = None) -> str:
        path = binary_directory / tool
        result = subprocess.run(
            [path.resolve(), *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            details = (result.stdout + result.stderr).strip()
            raise RuntimeError(f"forwarded Peano alias failed: {tool}\n{details}")
        return result.stdout + result.stderr

    alias_names = {
        PurePosixPath(str(relative)).name
        for relative in aliases
        if str(relative).startswith("peano/llvm-aie/bin/")
    }
    expected_aliases = set(WINDOWS_PEANO_ALIAS_FORWARDING)
    if alias_names != expected_aliases:
        raise RuntimeError(
            "Windows compiler wheel does not contain the expected forwarded aliases: "
            f"expected {sorted(expected_aliases)}, found {sorted(alias_names)}"
        )
    for alias_name in sorted(alias_names):
        if not (binary_directory / alias_name).is_file():
            raise RuntimeError(f"forwarded Peano alias is missing: {alias_name}")
        run(alias_name, "--version")
    run("clang.exe", "-cc1", "-help")
    run("clang.exe", "-cc1as", "-help")

    with tempfile.TemporaryDirectory(prefix="fastflow-sdk-windows-alias-") as temporary:
        working_directory = Path(temporary)
        source = working_directory / "alias source.c"
        source.write_text("int fastflow_alias_test(void) { return 0; }\n")
        run(
            "clang.exe",
            "--target=x86_64-linux-gnu",
            "-c",
            source.name,
            "-o",
            "main.o",
            cwd=working_directory,
        )
        run(
            "ld.lld.exe",
            "-e",
            "fastflow_alias_test",
            "main.o",
            "-o",
            "main",
            cwd=working_directory,
        )
        readelf = run("llvm-readelf.exe", "-h", "main", cwd=working_directory)
        if "ELF Header" not in readelf:
            raise RuntimeError("forwarded llvm-readelf did not emit GNU ELF output")
        preprocessed = run("clang-cpp.exe", source.name, cwd=working_directory)
        if "fastflow_alias_test" not in preprocessed:
            raise RuntimeError("forwarded clang-cpp did not preprocess C source")
        run("clang-cl.exe", "/c", source.name, "/Fomain.obj", cwd=working_directory)
        run(
            "lld-link.exe",
            "/entry:fastflow_alias_test",
            "/subsystem:console",
            "/nodefaultlib",
            "/out:main.exe",
            "main.obj",
            cwd=working_directory,
        )
    if "-platform_version" not in run("ld64.lld.exe", "--help"):
        raise RuntimeError("forwarded ld64.lld did not select the Darwin flavor")
    if "--export-table" not in run("wasm-ld.exe", "--help"):
        raise RuntimeError("forwarded wasm-ld did not select the WebAssembly flavor")


def _verify_linux_tool_alias_launchers(core: Path) -> None:
    package_root = core / COMPILER_PACKAGE
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aliases = manifest.get("forwarded_peano_aliases", ())
    if not isinstance(aliases, list):
        raise RuntimeError("forwarded_peano_aliases must be a list")
    binary_directory = package_root / "peano" / "llvm-aie" / "bin"

    def run(tool: str, *arguments: str, cwd: Path | None = None) -> str:
        path = binary_directory / tool
        result = subprocess.run(
            [path.resolve(), *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            details = (result.stdout + result.stderr).strip()
            raise RuntimeError(f"forwarded Peano alias failed: {tool}\n{details}")
        return result.stdout + result.stderr

    alias_names = {
        PurePosixPath(str(relative)).name
        for relative in aliases
        if str(relative).startswith("peano/llvm-aie/bin/")
    }
    expected_aliases = set(LINUX_PEANO_ALIAS_FORWARDING)
    if alias_names != expected_aliases:
        raise RuntimeError(
            "Linux compiler wheel does not contain the expected forwarded aliases: "
            f"expected {sorted(expected_aliases)}, found {sorted(alias_names)}"
        )
    for alias_name in sorted(alias_names):
        if not (binary_directory / alias_name).is_file():
            raise RuntimeError(f"forwarded Peano alias is missing: {alias_name}")
        run(alias_name, "--version")

    with tempfile.TemporaryDirectory(prefix="fastflow-sdk-linux-alias-") as temporary:
        working_directory = Path(temporary)
        external_directory = working_directory / "external"
        external_directory.mkdir()
        for alias_name in sorted(alias_names):
            external_alias = external_directory / alias_name
            external_alias.symlink_to(binary_directory / alias_name)
            result = subprocess.run(
                [external_alias, "--version"],
                cwd=working_directory,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                details = (result.stdout + result.stderr).strip()
                raise RuntimeError(
                    "forwarded Peano alias failed through an external symlink: "
                    f"{alias_name}\n{details}"
                )
        source = working_directory / "main.c"
        source.write_text("int main(void) { return 0; }\n")
        run(
            "clang",
            "--target=x86_64-linux-gnu",
            "-c",
            source.name,
            "-o",
            "main.o",
            cwd=working_directory,
        )
        run("ld.lld", "-e", "main", "main.o", "-o", "main", cwd=working_directory)
        readelf = run("llvm-readelf", "-h", "main", cwd=working_directory)
        if "ELF Header" not in readelf:
            raise RuntimeError("forwarded llvm-readelf did not emit GNU ELF output")
        run("clang-cl", "/c", source.name, "/Fomain.obj", cwd=working_directory)
        run(
            "lld-link",
            "/entry:main",
            "/subsystem:console",
            "/nodefaultlib",
            "/out:main.exe",
            "main.obj",
            cwd=working_directory,
        )
    if "-platform_version" not in run("ld64.lld", "--help"):
        raise RuntimeError("forwarded ld64.lld did not select the Darwin flavor")
    if "--export-table" not in run("wasm-ld", "--help"):
        raise RuntimeError("forwarded wasm-ld did not select the WebAssembly flavor")


def verify_windows_compiler_wheels(
    wheels: list[Path],
) -> dict[str, list[str]]:
    """Requires every compiler PE import to resolve from core/bindings or Windows."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        extracted = []
        for index, wheel in enumerate(wheels):
            distribution = _wheel_distribution(wheel)
            extracted.append((distribution, _extract_wheel(wheel, root / str(index))))
        core_roots = [
            tree
            for distribution, tree in extracted
            if distribution == COMPILER_CORE_DISTRIBUTION
        ]
        if len(core_roots) != 1:
            raise RuntimeError(
                "Windows compiler wheel validation requires one core wheel"
            )
        core = core_roots[0]
        _verify_windows_tool_alias_launchers(core)
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
                item for item in tree.rglob("*") if item.is_file() and is_pe(item)
            ):
                imports = pe_imports(path)
                key = f"{distribution}/{path.relative_to(tree).as_posix()}"
                graph[key] = imports
                if distribution == COMPILER_BINDINGS_DISTRIBUTION:
                    versioned_python_imports.extend(
                        f"{key} -> {name}"
                        for name in imports
                        if re.fullmatch(r"python3[0-9]{1,2}\.dll", name.casefold())
                    )
                    stable_python_imports.extend(
                        f"{key} -> {name}"
                        for name in imports
                        if name.casefold() == "python3.dll"
                    )
                missing.extend(
                    f"{key} -> {name}"
                    for name in imports
                    if name.casefold() not in allowed
                    and not windows_system_runtime(name)
                )
        if versioned_python_imports:
            raise RuntimeError(
                "FastFlow SDK compiler bindings import a versioned CPython DLL:\n  "
                + "\n  ".join(sorted(versioned_python_imports))
            )
        if not stable_python_imports:
            raise RuntimeError(
                "FastFlow SDK compiler bindings do not import stable-ABI python3.dll"
            )
        if missing:
            raise RuntimeError(
                "FastFlow SDK compiler Windows dependency closure is incomplete:\n  "
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
            extracted.append((distribution, _extract_wheel(wheel, root / str(index))))
        core_roots = [
            tree
            for distribution, tree in extracted
            if distribution == COMPILER_CORE_DISTRIBUTION
        ]
        if len(core_roots) != 1:
            raise RuntimeError(
                "Linux compiler wheel validation requires one core wheel"
            )
        _verify_linux_tool_alias_launchers(core_roots[0])
        core_names = {
            path.name
            for path in core_roots[0].rglob("*")
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
                path.name
                for path in tree.rglob("*")
                if path.is_file() and is_loadable_elf(path)
            }
            allowed = own_names | core_names
            for path in sorted(
                item
                for item in tree.rglob("*")
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
                    f"{key} -> {name}"
                    for name in needed
                    if name not in allowed and not linux_system_runtime(name)
                )
        if absolute_rpaths:
            raise RuntimeError(
                "FastFlow SDK compiler wheels contain absolute ELF RPATHs:\n  "
                + "\n  ".join(sorted(absolute_rpaths))
            )
        if missing:
            raise RuntimeError(
                "FastFlow SDK compiler Linux dependency closure is incomplete:\n  "
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
        reference
        for reference in references
        if reference.member.casefold().endswith(text_suffixes)
    ]
    if text_references:
        details = "\n  ".join(
            f"{item.wheel.name}:{item.member} -> {item.forbidden_path}"
            for item in text_references
        )
        raise RuntimeError(
            "FastFlow SDK compiler wheel text payload contains local paths:\n  "
            + details
        )
