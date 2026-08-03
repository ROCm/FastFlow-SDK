# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build Linux rocm-npu runtime and FastFlowLM wheels from staged artifacts."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import zipfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

from build_tools.fastflowlm_artifact import (
    _is_linux_system_runtime,
    reset_directory,
    verify_bundle,
)
from build_tools.rocm_npu_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_CORE_DISTRIBUTION,
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
    _write_file,
    _write_package_manifest,
)


LINUX_WHEEL_TAG = "py3-none-manylinux_2_28_x86_64"
FASTFLOWLM_LINUX_WHEEL_TAG = "py3-none-linux_x86_64"
AUDITWHEEL_API_VERSION = "6.7.0"


def _auditwheel_policy_diagnostics(wheel_info, target_policy_name: str) -> str:
    """Formats the structured reasons a wheel misses its claimed policy."""
    target = wheel_info.policies.get_policy_by_name(target_policy_name)
    components = (
        ("external references", wheel_info.ref_policy),
        ("versioned symbols", wheel_info.sym_policy),
        ("UCS-2 symbols", wheel_info.ucs_policy),
        ("PyFPE symbols", wheel_info.pyfpe_policy),
        ("blacklisted symbols", wheel_info.blacklist_policy),
        ("machine ISA", wheel_info.machine_policy),
    )
    lines = [
        f"auditwheel policy diagnostics for {target_policy_name}:",
        f"  overall: {wheel_info.overall_policy.name}",
    ]
    for label, policy in components:
        marker = "LIMITING" if policy < target else "ok"
        lines.append(f"  {label}: {policy.name} ({marker})")

    target_refs = wheel_info.external_refs[target.name]
    if target_refs.libs:
        lines.append("  external libraries:")
        lines.extend(
            f"    {name}: {path if path is not None else '<unresolved>'}"
            for name, path in sorted(target_refs.libs.items())
        )
    if target_refs.blacklist:
        lines.append("  blacklisted symbol references:")
        lines.extend(
            f"    {name}: {', '.join(sorted(symbols))}"
            for name, symbols in sorted(target_refs.blacklist.items())
        )

    if wheel_info.sym_policy < target:
        allowed = {
            f"{prefix}_{version}"
            for prefix, versions in target.symbol_versions.items()
            for version in versions
        }
        required = {
            symbol
            for symbols in wheel_info.versioned_symbols.values()
            for symbol in symbols
        }
        unsupported = sorted(required - allowed)
        if unsupported:
            lines.append("  unsupported versioned symbols:")
            lines.extend(f"    {symbol}" for symbol in unsupported)
    return "\n".join(lines)


def wheel_tag(system: str | None = None, machine: str | None = None) -> str:
    system = (system or platform.system()).lower()
    if system != "linux":
        raise RuntimeError(f"rocm-npu Linux wheels are unsupported on {system}")
    machine = (machine or platform.machine()).lower()
    if machine not in ("amd64", "x86_64"):
        raise RuntimeError(
            f"rocm-npu Linux wheels require an x86-64 host, got {machine}"
        )
    return LINUX_WHEEL_TAG


def fastflowlm_wheel_tag(
    system: str | None = None, machine: str | None = None
) -> str:
    wheel_tag(system, machine)
    return FASTFLOWLM_LINUX_WHEEL_TAG


def auditwheel_policy(
    wheel: Path,
    *,
    exclude: frozenset[str] = frozenset(),
) -> str:
    """Returns auditwheel's structured overall policy for a wheel."""
    try:
        installed_version = distribution_version("auditwheel")
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"auditwheel {AUDITWHEEL_API_VERSION} is required to audit Linux wheels"
        ) from error
    if installed_version != AUDITWHEEL_API_VERSION:
        raise RuntimeError(
            "unsupported auditwheel structured API version: "
            f"{installed_version}; expected {AUDITWHEEL_API_VERSION}"
        )
    from auditwheel.error import NonPlatformWheelError, WheelToolsError
    from auditwheel.wheel_abi import analyze_wheel_abi
    from auditwheel.wheeltools import get_wheel_architecture, get_wheel_libc

    try:
        architecture = get_wheel_architecture(wheel.name)
    except (WheelToolsError, NonPlatformWheelError):
        architecture = None
    try:
        libc = get_wheel_libc(wheel.name)
    except WheelToolsError:
        libc = None
    wheel_info = analyze_wheel_abi(
        libc,
        architecture,
        wheel,
        exclude,
        disable_isa_ext_check=False,
        allow_graft=False,
    )
    claimed_policy = re.search(r"manylinux_\d+_\d+_x86_64", wheel.name)
    if claimed_policy:
        target = wheel_info.policies.get_policy_by_name(claimed_policy.group())
        if wheel_info.overall_policy < target:
            print(_auditwheel_policy_diagnostics(wheel_info, target.name))
    return wheel_info.overall_policy.name


def validate_auditwheel_policy(distribution: str, policy: str) -> str:
    """Checks auditwheel's structured policy against the claimed wheel policy."""
    if distribution in (
        RUNTIME_DISTRIBUTION,
        COMPILER_CORE_DISTRIBUTION,
        COMPILER_BINDINGS_DISTRIBUTION,
    ):
        manylinux = re.fullmatch(r"manylinux_(\d+)_(\d+)_x86_64", policy)
        if not manylinux or tuple(map(int, manylinux.groups())) > (2, 28):
            raise RuntimeError(
                f"{distribution} is not manylinux_2_28 compatible: {policy}"
            )
    elif distribution == FASTFLOWLM_DISTRIBUTION:
        if policy != "linux_x86_64" and not re.fullmatch(
            r"manylinux_\d+_\d+_x86_64", policy
        ):
            raise RuntimeError(
                f"{distribution} has an unexpected auditwheel policy: {policy}"
            )
    else:
        raise RuntimeError(f"unexpected Linux wheel distribution: {distribution}")
    return policy


def _write_runtime_module(package_root: Path) -> None:
    _write_file(package_root / "__init__.py", '''"""Package-local HRX runtime helpers."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def rocm_npu_path() -> Path:
    path = package_root() / "bin" / "rocm-npu"
    if not path.is_file():
        raise FileNotFoundError(f"rocm-npu executable was not found: {path}")
    return path


def libhrx_path() -> Path:
    direct = package_root() / "lib" / "libhrx.so"
    if direct.is_file():
        return direct
    candidates = sorted((package_root() / "lib").glob("libhrx.so*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"libhrx.so was not found under {package_root() / 'lib'}")


def initialize_process():
    return ctypes.CDLL(str(libhrx_path()), mode=getattr(ctypes, "RTLD_GLOBAL", 0))


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
import sys
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def flm_path() -> Path:
    path = package_root() / "bin" / "flm"
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
    os.execvpe(executable, [executable, *args], env)
    raise RuntimeError("unreachable")
''')
    _write_file(package_root / "__main__.py", '''from __future__ import annotations

from . import main


if __name__ == "__main__":
    main()
''')


def _is_elf(path: Path) -> bool:
    try:
        return path.read_bytes()[:4] == b"\x7fELF"
    except OSError:
        return False


def _elf_files(package_root: Path) -> list[Path]:
    return sorted(path for path in package_root.rglob("*")
                  if path.is_file() and _is_elf(path))


def _print_soname(patchelf: str, path: Path) -> str | None:
    result = subprocess.run([patchelf, "--print-soname", str(path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _copy_hrx_soname(source_bundle: Path, runtime_root: Path) -> str:
    """Copies only the runtime SONAME, omitting linker and version aliases."""
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to prepare Linux rocm-npu wheels")
    libraries = sorted((source_bundle / "lib").glob("libhrx.so*"))
    if not libraries:
        raise RuntimeError(
            f"staged runtime bundle contains no libhrx.so* under "
            f"{source_bundle / 'lib'}"
        )
    sonames = {
        soname for library in libraries
        if (soname := _print_soname(patchelf, library))
    }
    if len(sonames) != 1:
        raise RuntimeError(
            "staged HRX libraries must expose exactly one ELF SONAME, found: "
            + ", ".join(sorted(sonames))
        )
    soname = next(iter(sonames))
    source = next(
        (library for library in libraries if library.name == soname),
        libraries[0],
    )
    _copy_file(source, runtime_root / "lib" / soname)
    return soname


def _print_needed(patchelf: str, path: Path) -> list[str]:
    return subprocess.run(
        [patchelf, "--print-needed", str(path)], check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()


def _print_rpath(patchelf: str, path: Path) -> str:
    return subprocess.run(
        [patchelf, "--print-rpath", str(path)], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _provided_library_filenames(package_root: Path) -> set[str]:
    return {path.name for path in _elf_files(package_root)}


def _source_linux_library_candidates(source_bundle: Path, patchelf: str) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    for path in sorted((source_bundle / "lib").rglob("*.so*")):
        if not path.is_file() or not _is_elf(path):
            continue
        candidates.setdefault(path.name, path)
        soname = _print_soname(patchelf, path)
        if soname:
            candidates.setdefault(soname, path)
    return candidates


def _copy_native_closure(source_bundle: Path, package_root: Path,
                         external_roots: tuple[Path, ...] = ()) -> set[str]:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to prepare Linux rocm-npu wheels")
    candidates = _source_linux_library_candidates(source_bundle, patchelf)
    license_names = set()
    queue = _elf_files(package_root)
    while queue:
        path = queue.pop(0)
        for needed in _print_needed(patchelf, path):
            provided = _provided_library_filenames(package_root)
            for external_root in external_roots:
                provided.update(_provided_library_filenames(external_root))
            if needed in provided:
                continue
            if _is_linux_system_runtime(needed):
                continue
            source = candidates.get(needed)
            if not source:
                relative = path.relative_to(package_root).as_posix()
                raise RuntimeError(
                    f"{package_root.name} dependency is not present in the staged "
                    f"FastFlowLM bundle: {relative} -> {needed}")
            destination = package_root / "lib" / needed
            if destination.exists():
                continue
            _copy_file(source, destination)
            license_names.update({source.name, needed})
            queue.append(destination)
    return license_names


def _copy_runtime_native_closure(source_bundle: Path, runtime_root: Path) -> set[str]:
    return _copy_native_closure(source_bundle, runtime_root)


def _copy_fastflowlm_native_closure(source_bundle: Path, fastflowlm_root: Path,
                                    runtime_root: Path) -> set[str]:
    return _copy_native_closure(source_bundle, fastflowlm_root, (runtime_root,))


def _rpath_for(path: Path, package_root: Path) -> str:
    relative = path.relative_to(package_root)
    if package_root.name == RUNTIME_PACKAGE:
        return "$ORIGIN"
    if relative.parts[0] == "bin":
        return "$ORIGIN/../lib/flm:$ORIGIN/../lib:$ORIGIN/../../rocm_npu_runtime/lib"
    if len(relative.parts) >= 3 and relative.parts[0:2] == ("lib", "flm"):
        return "$ORIGIN:$ORIGIN/..:$ORIGIN/../../../rocm_npu_runtime/lib"
    if relative.parts[0] == "lib":
        return "$ORIGIN:$ORIGIN/flm:$ORIGIN/../../rocm_npu_runtime/lib"
    return "$ORIGIN"


def _strip_debug(package_roots: list[Path]) -> dict[str, int]:
    strip = shutil.which("strip")
    if not strip:
        raise RuntimeError("strip is required to prepare Linux rocm-npu wheels")
    targets = [path for package_root in package_roots for path in _elf_files(package_root)]
    if targets:
        subprocess.run([strip, "--strip-debug", *[str(path) for path in targets]], check=True)
    return {"stripped": len(targets)}


def _patch_and_verify_linux_products(package_roots: list[Path]) -> dict[str, list[str]]:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to prepare Linux rocm-npu wheels")
    package_elf_files = {package_root: _elf_files(package_root) for package_root in package_roots}
    if not any(package_elf_files.values()):
        raise RuntimeError("rocm-npu wheel payload contains no ELF files")

    for package_root, elf_files in package_elf_files.items():
        for path in elf_files:
            subprocess.run([patchelf, "--set-rpath", _rpath_for(path, package_root),
                            str(path)], check=True)
            absolute_entries = [
                entry for entry in _print_rpath(patchelf, path).split(":")
                if entry.startswith("/")
            ]
            if absolute_entries:
                relative = f"{package_root.name}/{path.relative_to(package_root).as_posix()}"
                raise RuntimeError(
                    "rocm-npu wheel ELF RPATH/RUNPATH contains absolute paths: "
                    f"{relative} -> {':'.join(absolute_entries)}")

    provided_names = {
        package_root: _provided_library_filenames(package_root)
        for package_root in package_roots
    }
    runtime_names = set()
    for package_root, names in provided_names.items():
        if package_root.name == RUNTIME_PACKAGE:
            runtime_names = names
            break

    graph = {}
    missing = []
    for package_root, elf_files in package_elf_files.items():
        allowed_names = set(provided_names[package_root])
        if package_root.name == FASTFLOWLM_PACKAGE:
            allowed_names.update(runtime_names)
        for path in elf_files:
            needed = _print_needed(patchelf, path)
            relative = f"{package_root.name}/{path.relative_to(package_root).as_posix()}"
            graph[relative] = sorted(needed)
            missing.extend(f"{relative} -> {name}" for name in needed
                           if name not in allowed_names and
                           not _is_linux_system_runtime(name))
    if missing:
        raise RuntimeError("rocm-npu wheel Linux dependency closure is incomplete "
                           "within package boundaries:\n  " +
                           "\n  ".join(sorted(missing)))
    return graph


def stage_wheel_products(source_bundle: Path, destination: Path,
                         version: str) -> dict[str, Path]:
    """Stage split package roots from the combined FastFlowLM+HRX bundle."""
    if os.name == "nt":
        raise RuntimeError("rocm-npu runtime/FastFlowLM wheel staging is Linux-only for now")
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
    _copy_hrx_soname(source_bundle, runtime_root)
    _copy_relative_file(source_bundle, "bin/rocm-npu", runtime_root)
    _copy_relative_tree(source_bundle, "share/rocm-npu/self-test", runtime_root)
    runtime_license_names = _copy_runtime_native_closure(source_bundle, runtime_root)

    _write_fastflowlm_module(fastflowlm_root)
    _copy_relative_file(source_bundle, "bin/flm", fastflowlm_root)
    _copy_relative_tree(source_bundle, "share/flm", fastflowlm_root)
    flm_lib_dir = source_bundle / "lib" / "flm"
    if flm_lib_dir.is_dir():
        _copy_relative_tree(source_bundle, "lib/flm", fastflowlm_root)
    fastflowlm_license_names = _copy_fastflowlm_native_closure(
        source_bundle, fastflowlm_root, runtime_root)

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
    runtime_names.update(runtime_license_names)
    fastflowlm_names = {path.name for path in fastflowlm_root.rglob("*") if path.is_file()}
    fastflowlm_names.update(fastflowlm_license_names)
    _copy_dependency_licenses(source_bundle, source_manifest, runtime_root, runtime_names)
    _copy_dependency_licenses(source_bundle, source_manifest, fastflowlm_root, fastflowlm_names)

    strip_report = _strip_debug([runtime_root, fastflowlm_root])
    binary_dependencies = _patch_and_verify_linux_products([runtime_root, fastflowlm_root])

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
        "strip": strip_report,
    }
    _write_package_manifest(runtime_root, {
        **common_manifest,
        "product": RUNTIME_DISTRIBUTION,
        "artifact": runtime_artifact_name(),
        "components": source_manifest.get("components", []),
        "wheel_tag": wheel_tag(),
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
        "wheel_tag": fastflowlm_wheel_tag(),
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
        tag=fastflowlm_wheel_tag(),
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


def _zip_member_is_elf(data: bytes) -> bool:
    return data.startswith(b"\x7fELF")


def verify_wheels_do_not_have_text_path_references(wheels: dict[str, Path],
                                                   forbidden_paths: tuple[Path, ...]) -> None:
    """Rejects local paths in text payloads and audits opaque native payloads."""
    references = find_wheel_path_references(wheels, forbidden_paths)
    text_references = []
    native_references = []
    for reference in references:
        with zipfile.ZipFile(reference.wheel) as archive:
            with archive.open(reference.member) as member:
                is_native = _zip_member_is_elf(member.read(4))
        (native_references if is_native else text_references).append(reference)
    if native_references:
        print(
            "wheel path audit: "
            f"{len(native_references)} native member(s) contain local path strings; "
            "native relocation is enforced by ELF RPATH/RUNPATH and runtime tests"
        )
    if text_references:
        details = "\n  ".join(
            f"{reference.wheel.name}:{reference.member} -> "
            f"{reference.forbidden_path}"
            for reference in text_references
        )
        raise RuntimeError(
            "wheel text payload contains local checkout/build paths:\n  " +
            details
        )
