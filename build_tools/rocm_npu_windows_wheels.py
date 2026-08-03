# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build Windows rocm-npu runtime and FastFlowLM wheels."""

from __future__ import annotations

import os
import platform
import zipfile
from pathlib import Path

from build_tools.fastflowlm_artifact import (
    pe_imports,
    reset_directory,
    verify_bundle,
)
from build_tools.rocm_npu_wheel_common import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    build_wheel as _build_wheel,
    fastflowlm_artifact_name,
    find_distribution_wheels,
    find_wheel_path_references,
    remove_distribution_wheels,
    runtime_artifact_name,
    validate_version,
    _copy_dependency_licenses,
    _copy_file,
    _copy_license,
    _copy_license_as,
    _copy_relative_file,
    _copy_relative_tree,
    _dependency_license_packages as _dependency_license_packages,
    _write_file,
    _write_package_manifest,
)


WINDOWS_WHEEL_TAG = "py3-none-win_amd64"


def wheel_tag(system: str | None = None, machine: str | None = None) -> str:
    system = (system or platform.system()).lower()
    if system != "windows":
        raise RuntimeError(f"rocm-npu wheels are unsupported on {system}")
    machine = (machine or platform.machine()).lower()
    if machine not in ("amd64", "x86_64"):
        raise RuntimeError(
            f"rocm-npu Windows wheels require an x86-64 host, got {machine}"
        )
    return WINDOWS_WHEEL_TAG


def _write_runtime_module(package_root: Path) -> None:
    _write_file(package_root / "__init__.py", '''"""Package-local HRX runtime helpers."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

_dll_directory_handle = None
_runtime_library = None


def package_root() -> Path:
    return Path(__file__).resolve().parent


def binary_dir() -> Path:
    return package_root() / "bin"


def rocm_npu_path() -> Path:
    path = binary_dir() / "rocm-npu.exe"
    if not path.is_file():
        raise FileNotFoundError(f"rocm-npu executable was not found: {path}")
    return path


def libhrx_path() -> Path:
    path = binary_dir() / "hrx.dll"
    if not path.is_file():
        raise FileNotFoundError(f"HRX runtime was not found: {path}")
    return path


def initialize_process():
    global _dll_directory_handle, _runtime_library
    if _dll_directory_handle is None:
        _dll_directory_handle = os.add_dll_directory(str(binary_dir()))
    if _runtime_library is None:
        _runtime_library = ctypes.WinDLL(str(libhrx_path()))
    return _runtime_library


def self_test(*, load: bool = False) -> Path:
    path = libhrx_path()
    if load:
        initialize_process()
    return path


def main() -> int:
    environment = os.environ.copy()
    scripts = str(Path(sys.executable).resolve().parent)
    existing = environment.get("ROCM_NPU_PLUGIN_PATH")
    environment["ROCM_NPU_PLUGIN_PATH"] = (
        scripts if not existing else scripts + os.pathsep + existing
    )
    return subprocess.run(
        [str(rocm_npu_path()), *sys.argv[1:]],
        env=environment,
        check=False,
    ).returncode
''')
    _write_file(package_root / "__main__.py", '''from . import main

raise SystemExit(main())
''')


def _write_fastflowlm_module(package_root: Path) -> None:
    _write_file(package_root / "__init__.py", '''"""Package-local FastFlowLM runtime helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def flm_path() -> Path:
    path = package_root() / "bin" / "flm.exe"
    if not path.is_file():
        raise FileNotFoundError(f"FastFlowLM executable was not found: {path}")
    return path


def config_path() -> Path:
    path = package_root() / "share" / "flm" / "model_list.json"
    if not path.is_file():
        raise FileNotFoundError(f"FastFlowLM model list was not found: {path}")
    return path


def xclbin_path() -> Path:
    path = package_root() / "share" / "flm"
    if not (path / "xclbins").is_dir():
        raise FileNotFoundError(f"FastFlowLM xclbins were not found: {path / 'xclbins'}")
    return path


def main(argv: list[str] | None = None) -> int:
    executable = str(flm_path())
    args = list(sys.argv[1:] if argv is None else argv)
    env = dict(os.environ)
    env.setdefault("HRX_GPU_DRIVER", "amdxdna")
    env.setdefault("FLM_CONFIG_PATH", str(config_path()))
    env.setdefault("FLM_XCLBIN_PATH", str(xclbin_path()))
    from rocm_npu_runtime import binary_dir as runtime_binary_dir
    env["PATH"] = os.pathsep.join(
        (str(package_root() / "bin"), str(runtime_binary_dir()), env.get("PATH", ""))
    )
    return subprocess.call([executable, *args], env=env)
''')
    _write_file(package_root / "__main__.py", '''from __future__ import annotations

from . import main


if __name__ == "__main__":
    main()
''')


def _is_pe(path: Path) -> bool:
    try:
        return path.read_bytes()[:2] == b"MZ"
    except OSError:
        return False


def _pe_files(package_root: Path) -> list[Path]:
    return sorted(path for path in package_root.rglob("*")
                  if path.is_file() and _is_pe(path))


def _is_windows_system_runtime(name: str) -> bool:
    name = name.lower()
    if name.startswith(("api-ms-", "ext-ms-")):
        return True
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return (system32 / name).is_file()


def _copy_windows_runtime_closure(source_bundle: Path, runtime_root: Path) -> None:
    candidates = {
        path.name.lower(): path
        for path in sorted((source_bundle / "bin").glob("*.dll"))
        if path.is_file()
    }
    queue = _pe_files(runtime_root)
    while queue:
        path = queue.pop(0)
        for needed in pe_imports(path):
            destination = runtime_root / "bin" / needed
            if destination.is_file() or _is_windows_system_runtime(needed):
                continue
            source = candidates.get(needed)
            if not source:
                relative = path.relative_to(runtime_root).as_posix()
                raise RuntimeError(
                    f"{RUNTIME_PACKAGE} dependency is not present in the staged "
                    f"FastFlowLM bundle: {relative} -> {needed}")
            _copy_file(source, destination)
            queue.append(destination)


def _copy_windows_fastflowlm_payload(source_bundle: Path, fastflowlm_root: Path,
                                     runtime_root: Path) -> None:
    runtime_names = {
        path.name.lower() for path in (runtime_root / "bin").glob("*.dll")
    }
    _copy_relative_file(source_bundle, "bin/flm.exe", fastflowlm_root)
    for source in sorted((source_bundle / "bin").glob("*.dll")):
        if source.name.lower() not in runtime_names:
            _copy_file(source, fastflowlm_root / "bin" / source.name)


def _copy_microsoft_runtime_notice(
    source_bundle: Path,
    source_manifest: dict[str, object],
    package_root: Path,
) -> None:
    dependencies = source_manifest.get("dependencies")
    if not isinstance(dependencies, dict):
        return
    vcpkg = (
        dependencies.get("vcpkg")
        if dependencies.get("provider") == "vcpkg+system"
        else dependencies
    )
    if not isinstance(vcpkg, dict):
        return
    runtime_files = vcpkg.get("microsoft_runtime_files")
    if not isinstance(runtime_files, list):
        return
    shipped_names = {
        path.name.casefold() for path in package_root.rglob("*") if path.is_file()
    }
    if not any(str(name).casefold() in shipped_names for name in runtime_files):
        return
    _copy_license(
        source_bundle,
        "licenses/Microsoft-Visual-Cpp-Runtime-NOTICE.txt",
        package_root,
    )


def _verify_windows_products(package_roots: list[Path]) -> dict[str, list[str]]:
    package_files = {
        package_root: {path.name.lower() for path in _pe_files(package_root)}
        for package_root in package_roots
    }
    runtime_names = next(
        (names for root, names in package_files.items()
         if root.name == RUNTIME_PACKAGE),
        set(),
    )
    graph = {}
    missing = []
    for package_root in package_roots:
        allowed_names = set(package_files[package_root])
        if package_root.name == FASTFLOWLM_PACKAGE:
            allowed_names.update(runtime_names)
        for path in _pe_files(package_root):
            imports = pe_imports(path)
            relative = f"{package_root.name}/{path.relative_to(package_root).as_posix()}"
            graph[relative] = imports
            missing.extend(
                f"{relative} -> {name}" for name in imports
                if name not in allowed_names and not _is_windows_system_runtime(name)
            )
    if missing:
        raise RuntimeError(
            "rocm-npu wheel Windows dependency closure is incomplete within "
            "package boundaries:\n  " + "\n  ".join(sorted(missing)))
    return graph


def stage_wheel_products(source_bundle: Path, destination: Path,
                         version: str) -> dict[str, Path]:
    """Stage split package roots from the combined FastFlowLM+HRX bundle."""
    if os.name != "nt":
        raise RuntimeError("Windows rocm-npu wheel staging requires a Windows host")
    version = validate_version(version)
    source_manifest = verify_bundle(source_bundle)
    reset_directory(destination, destination.parent)

    runtime_tree = destination / "runtime"
    fastflowlm_tree = destination / "fastflowlm"
    runtime_root = runtime_tree / RUNTIME_PACKAGE
    fastflowlm_root = fastflowlm_tree / FASTFLOWLM_PACKAGE
    runtime_root.mkdir(parents=True)
    fastflowlm_root.mkdir(parents=True)

    _write_runtime_module(runtime_root)
    _write_fastflowlm_module(fastflowlm_root)
    _copy_relative_file(source_bundle, "bin/hrx.dll", runtime_root)
    _copy_relative_file(source_bundle, "bin/rocm-npu.exe", runtime_root)
    _copy_relative_tree(source_bundle, "share/rocm-npu/self-test", runtime_root)
    _copy_windows_runtime_closure(source_bundle, runtime_root)
    _copy_windows_fastflowlm_payload(
        source_bundle, fastflowlm_root, runtime_root)
    _copy_relative_tree(source_bundle, "share/flm", fastflowlm_root)

    _copy_license_as(source_bundle, "licenses/rocm-npu-staging-LICENSE",
                     runtime_root, "licenses/rocm-npu-LICENSE")
    _copy_license(source_bundle, "licenses/hrx-LICENSE", runtime_root)
    _copy_license(source_bundle, "licenses/nlohmann-json-LICENSE.MIT",
                  runtime_root)
    _copy_license(source_bundle, "licenses/THIRD_PARTY_NOTICES.md",
                  runtime_root)
    _copy_license_as(source_bundle, "licenses/rocm-npu-staging-LICENSE",
                     fastflowlm_root, "licenses/rocm-npu-LICENSE")
    _copy_license(source_bundle, "licenses/FastFlowLM-LICENSE_RUNTIME.txt", fastflowlm_root)
    _copy_license(source_bundle, "licenses/FastFlowLM-TERMS.md", fastflowlm_root)

    runtime_names = {path.name for path in runtime_root.rglob("*") if path.is_file()}
    fastflowlm_names = {path.name for path in fastflowlm_root.rglob("*") if path.is_file()}
    _copy_dependency_licenses(source_bundle, source_manifest, runtime_root, runtime_names)
    _copy_dependency_licenses(source_bundle, source_manifest, fastflowlm_root, fastflowlm_names)
    _copy_microsoft_runtime_notice(source_bundle, source_manifest, runtime_root)
    _copy_microsoft_runtime_notice(source_bundle, source_manifest, fastflowlm_root)

    binary_dependencies = _verify_windows_products(
        [runtime_root, fastflowlm_root])

    common_manifest = {
        "schema_version": 1,
        "version": version,
        "platform": source_manifest.get("platform", {}),
        "sources": source_manifest.get("sources", {}),
        "patches": source_manifest.get("patches", []),
        "dependencies": source_manifest.get("dependencies", {}),
        "toolchain": source_manifest.get("toolchain", {}),
        "source_artifact": source_manifest.get("artifact"),
        "source_product": source_manifest.get("product"),
        "wheel_tag": wheel_tag(),
        "strip": {"stripped": 0},
    }
    _write_package_manifest(runtime_root, {
        **common_manifest,
        "product": RUNTIME_DISTRIBUTION,
        "artifact": runtime_artifact_name(),
        "components": source_manifest.get("components", []),
        "binary_dependencies": {
            name: needed for name, needed in binary_dependencies.items()
            if name.startswith(f"{RUNTIME_PACKAGE}/")
        },
    })
    _write_package_manifest(fastflowlm_root, {
        **common_manifest,
        "product": FASTFLOWLM_DISTRIBUTION,
        "artifact": fastflowlm_artifact_name(),
        "components": [],
        "requires": [f"{RUNTIME_DISTRIBUTION} =={version}"],
        "binary_dependencies": {
            name: needed for name, needed in binary_dependencies.items()
            if name.startswith(f"{FASTFLOWLM_PACKAGE}/")
        },
    })
    return {
        RUNTIME_DISTRIBUTION: runtime_tree,
        FASTFLOWLM_DISTRIBUTION: fastflowlm_tree,
    }


def build_wheel(tree_root: Path, distribution: str, version: str, output_directory: Path,
                *, summary: str, requires: tuple[str, ...] = (),
                console_scripts: dict[str, str] | None = None,
                tag: str | None = None) -> Path:
    return _build_wheel(
        tree_root,
        distribution,
        version,
        output_directory,
        summary=summary,
        tag=tag or wheel_tag(),
        requires=requires,
        console_scripts=console_scripts,
    )


def build_runtime_fastflowlm_wheels(source_bundle: Path, stage_directory: Path,
                                    output_directory: Path,
                                    version: str) -> dict[str, Path]:
    products = stage_wheel_products(source_bundle, stage_directory, version)
    remove_distribution_wheels(
        output_directory, (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION)
    )
    runtime_wheel = build_wheel(
        products[RUNTIME_DISTRIBUTION],
        RUNTIME_DISTRIBUTION,
        version,
        output_directory,
        summary="ROCm NPU HRX amdxdna runtime",
        console_scripts={"rocm-npu": f"{RUNTIME_PACKAGE}:main"},
    )
    fastflowlm_wheel = build_wheel(
        products[FASTFLOWLM_DISTRIBUTION],
        FASTFLOWLM_DISTRIBUTION,
        version,
        output_directory,
        summary="ROCm NPU FastFlowLM runtime assets",
        requires=(f"{RUNTIME_DISTRIBUTION} =={version}",),
        console_scripts={"flm": f"{FASTFLOWLM_PACKAGE}:main"},
    )
    return {
        RUNTIME_DISTRIBUTION: runtime_wheel,
        FASTFLOWLM_DISTRIBUTION: fastflowlm_wheel,
    }


def find_rocm_npu_wheels(directory: Path) -> dict[str, Path]:
    return find_distribution_wheels(
        directory, (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION)
    )


def verify_wheels_do_not_have_text_path_references(
    wheels: dict[str, Path], forbidden_paths: tuple[Path, ...]
) -> None:
    """Rejects local paths in text payloads and audits opaque native payloads.

    Native strings are not a sound relocation signal: compilers and third-party
    libraries record source paths in diagnostics and several debug formats. PE
    import closure and clean-install runtime tests enforce native relocation.
    """
    references = find_wheel_path_references(wheels, forbidden_paths)
    text_references = []
    native_references = []
    for reference in references:
        with zipfile.ZipFile(reference.wheel) as archive:
            with archive.open(reference.member) as member:
                is_native = member.read(2) == b"MZ"
        (native_references if is_native else text_references).append(reference)
    if native_references:
        print(
            "wheel path audit: "
            f"{len(native_references)} native member(s) contain local path strings; "
            "native relocation is enforced by PE import and runtime tests"
        )
    if text_references:
        details = "\n  ".join(
            f"{reference.wheel.name}:{reference.member} -> "
            f"{reference.forbidden_path}"
            for reference in text_references
        )
        raise RuntimeError(
            "wheel text payload contains local checkout/build paths:\n  "
            + details
        )
