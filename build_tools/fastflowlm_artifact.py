# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and verify relocatable FastFlowLM + HRX runtime artifacts."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

from build_tools.archive import (
    create_deterministic_tar_zstd_cli,
    create_deterministic_zip,
    extract_archive_atomic,
)
from build_tools.artifact import (
    directory_state,
    file_inventory,
    git_source_state,
    reset_directory,
    sha256_file,
    validate_directory_target,
    verify_file_inventory,
    write_sha256_sidecar,
)
from build_tools.artifact import (
    git_revision as _git_revision,
)
from build_tools.fastflowlm_deps import dependency_paths
from build_tools.native_dependencies import (
    elf_needed,
    elf_soname,
    pe_imports,
    set_elf_rpath,
)
from build_tools.package_version import default_package_version

SCHEMA_VERSION = 1


def _run_text(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def verify_fastflowlm_fresh_installs(
    *,
    hrx_install: Path,
    hrx_source: Path,
    fastflowlm_install: Path,
    fastflowlm_source: Path,
    marker_name: str,
) -> dict[str, dict[str, object]]:
    """Verifies that fresh HRX and FastFlowLM installs still match their sources."""
    installed_flm = (
        fastflowlm_install / "bin" / ("flm.exe" if os.name == "nt" else "flm")
    )
    if not installed_flm.is_file():
        raise RuntimeError(
            "FastFlowLM is not installed; run: python dev.py build fastflowlm --fresh"
        )

    markers = {}
    for name, install, source in (
        ("HRX", hrx_install, hrx_source),
        ("FastFlowLM", fastflowlm_install, fastflowlm_source),
    ):
        marker = install / marker_name
        if not marker.is_file():
            raise RuntimeError(f"{name} was not built with --fresh")
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("source") != git_source_state(source):
            raise RuntimeError(
                f"{name} sources changed after the fresh build; rebuild with --fresh"
            )
        if recorded.get("install") != directory_state(
            install, exclude_names=(marker_name,)
        ):
            raise RuntimeError(
                f"{name} install changed after the fresh build; rebuild with --fresh"
            )
        markers[name] = recorded
    if markers["FastFlowLM"].get("hrx") != markers["HRX"]:
        raise RuntimeError(
            "FastFlowLM was built against a different HRX install; rebuild with --fresh"
        )
    return markers


def verify_cli_fresh_install(
    *,
    root: Path,
    cli_install: Path,
    hrx_install: Path,
    hrx_source: Path,
    marker_name: str,
) -> dict[str, object]:
    """Verifies that the installed native CLI matches this source checkout."""
    cli_name = "rocm-npu.exe" if os.name == "nt" else "rocm-npu"
    if not (cli_install / "bin" / cli_name).is_file():
        raise RuntimeError(
            "rocm-npu CLI is not installed; run: "
            "python dev.py build rocm-npu-cli --fresh"
        )
    marker = cli_install / marker_name
    if not marker.is_file():
        raise RuntimeError("rocm-npu CLI was not built with --fresh")
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    if recorded.get("source") != git_source_state(root):
        raise RuntimeError(
            "rocm-npu CLI sources changed after the fresh build; rebuild with --fresh"
        )
    hrx_marker_path = hrx_install / marker_name
    if not hrx_marker_path.is_file():
        raise RuntimeError("HRX was not built with --fresh")
    hrx_marker = json.loads(hrx_marker_path.read_text(encoding="utf-8"))
    hrx_headers = git_source_state(hrx_source)
    if (
        recorded.get("hrx_headers") != hrx_headers
        or hrx_marker.get("source") != hrx_headers
    ):
        raise RuntimeError(
            "rocm-npu CLI was built against different HRX headers; "
            "rebuild HRX and the CLI with --fresh"
        )
    if hrx_marker.get("install") != directory_state(
        hrx_install, exclude_names=(marker_name,)
    ):
        raise RuntimeError("HRX install changed since its fresh build")
    if recorded.get("version") != default_package_version(root):
        raise RuntimeError("rocm-npu CLI version does not match the package version")
    if recorded.get("install") != directory_state(
        cli_install, exclude_names=(marker_name,)
    ):
        raise RuntimeError(
            "rocm-npu CLI install changed after the fresh build; rebuild with --fresh"
        )
    return recorded


def validate_source_policy(manifest: dict[str, object], root: Path) -> None:
    versions = tomllib.loads(
        (root / "manifests" / "versions.toml").read_text(encoding="utf-8")
    )
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("artifact source provenance is missing")
    expected = {
        "rocm-npu-staging": _git_revision(root),
        "hrx": versions["hrx"]["commit"],
        "FastFlowLM": versions["fastflowlm"]["commit"],
    }
    for component, revision in expected.items():
        state = sources.get(component)
        if not isinstance(state, dict) or state.get("revision") != revision:
            raise RuntimeError(
                f"artifact {component} revision does not match the pinned revision"
            )
    staging_state = sources["rocm-npu-staging"]
    if staging_state.get("dirty"):
        raise RuntimeError("artifact was produced from a dirty staging checkout")


def _copy_checked(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"artifact overlay collision: {destination}")
        return
    shutil.copy2(source, destination)


def _parse_control_file(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    records = []
    for paragraph in path.read_text(encoding="utf-8", errors="replace").split("\n\n"):
        record = {}
        last_key = None
        for line in paragraph.splitlines():
            if line.startswith((" ", "\t")) and last_key:
                record[last_key] += "\n" + line[1:]
            elif ": " in line:
                key, value = line.split(": ", 1)
                last_key = key.lower().replace("-", "_")
                record[last_key] = value
        if record.get("status") == "install ok installed":
            records.append(record)
    return records


def _windows_pe_dependency_graph(bundle: Path) -> dict[str, list[str]]:
    binaries = sorted(
        [*(bundle / "bin").glob("*.exe"), *(bundle / "bin").glob("*.dll")]
    )
    bundled = {path.name.lower() for path in binaries}
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    graph = {}
    missing = []
    for binary in binaries:
        imports = pe_imports(binary)
        graph[binary.relative_to(bundle).as_posix()] = imports
        for name in imports:
            is_api_set = name.startswith(("api-ms-", "ext-ms-"))
            if (
                name not in bundled
                and not is_api_set
                and not (system32 / name).is_file()
            ):
                missing.append(f"{binary.name} -> {name}")
    if missing:
        raise RuntimeError(
            "Windows PE dependency closure is incomplete:\n  "
            + "\n  ".join(sorted(missing))
        )
    return graph


_MSVC_REDISTRIBUTABLE_RE = re.compile(
    r"^(?:concrt|msvcp|vcomp|vcruntime)\d[^/\\]*\.dll$",
    re.IGNORECASE,
)


def _copy_windows_redistributable_closure(bundle: Path) -> None:
    """Copies imported MSVC/OpenMP runtimes from the VS redistributable tree."""
    redist_root_value = os.environ.get("VCToolsRedistDir")
    redist_root = Path(redist_root_value) if redist_root_value else None
    candidates: dict[str, Path] | None = None
    queue = sorted([*(bundle / "bin").glob("*.exe"), *(bundle / "bin").glob("*.dll")])
    bundled = {path.name.casefold() for path in queue}
    while queue:
        binary = queue.pop(0)
        for needed in pe_imports(binary):
            normalized = needed.casefold()
            if normalized in bundled or not _MSVC_REDISTRIBUTABLE_RE.fullmatch(needed):
                continue
            if redist_root is None or not redist_root.is_dir():
                raise RuntimeError(
                    f"{binary.name} imports {needed}, but VCToolsRedistDir does not "
                    "identify a Visual Studio redistributable tree"
                )
            if candidates is None:
                candidates = {}
                x64_dlls = (
                    path
                    for path in redist_root.rglob("*.dll")
                    if "x64" in {part.casefold() for part in path.parts}
                )
                for path in sorted(
                    x64_dlls,
                    key=lambda item: (
                        "spectre" in {part.casefold() for part in item.parts},
                        len(item.parts),
                        str(item).casefold(),
                    ),
                ):
                    candidates.setdefault(path.name.casefold(), path)
            source = candidates.get(normalized)
            if source is None:
                raise RuntimeError(
                    f"{binary.name} imports {needed}, but it was not found under "
                    f"VCToolsRedistDir={redist_root}"
                )
            destination = bundle / "bin" / needed
            _copy_checked(source, destination)
            bundled.add(normalized)
            queue.append(destination)


def _vcpkg_dependencies_and_licenses(
    root: Path, bundle: Path, vcpkg_root: Path
) -> dict[str, object]:
    packages = _parse_control_file(vcpkg_root.parent / "vcpkg" / "status")
    if not packages:
        packages = _parse_control_file(
            root / ".cache" / "vcpkg-installed" / "vcpkg" / "status"
        )
    if not packages:
        raise RuntimeError("vcpkg dependency inventory is missing")

    licensed_packages = set()
    for copyright_file in sorted((vcpkg_root / "share").glob("*/copyright")):
        package = copyright_file.parent.name
        _copy_checked(
            copyright_file, bundle / "licenses" / "vcpkg" / package / "copyright"
        )
        licensed_packages.add(package)
    missing = sorted(
        {
            record["package"]
            for record in packages
            if record.get("package") not in licensed_packages
            and not record.get("package", "").startswith("vcpkg-")
        }
    )
    runtime_packages = {}
    info_root = vcpkg_root.parent / "vcpkg" / "info"
    if os.name == "nt":
        runtime_files = {path.name.lower() for path in (bundle / "bin").glob("*.dll")}
    else:
        runtime_files = {
            path.name
            for pattern in ("lib/*.so*", "lib/flm/*.so*")
            for path in bundle.glob(pattern)
        }
    for record in packages:
        package = record.get("package")
        if not package:
            continue
        for listing in info_root.glob(f"{package}_*.list"):
            for line in listing.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                name = Path(line.replace("/", os.sep)).name
                if os.name == "nt":
                    name = name.lower()
                if name in runtime_files:
                    runtime_packages[name] = package
    triplet_name = vcpkg_root.name
    triplet = root / ".github" / "vcpkg" / "triplets" / f"{triplet_name}.cmake"
    resolved_package_names = sorted(
        {record["package"] for record in packages if record.get("package")}
    )
    shipped_runtime_package_names = sorted(set(runtime_packages.values()))
    packages_without_shipped_runtime_files = sorted(
        set(resolved_package_names) - set(shipped_runtime_package_names)
    )
    result = {
        "provider": "vcpkg",
        # The policy-addressed install root contains exactly one manifest
        # resolution. Keep the complete resolution for license/build provenance,
        # and identify the subset represented by shipped dynamic libraries.
        "resolved_packages": packages,
        "shipped_runtime_packages": shipped_runtime_package_names,
        "packages_without_shipped_runtime_files": packages_without_shipped_runtime_files,
        "vcpkg_revision": _git_revision(root / ".cache" / "vcpkg-src"),
        "triplet": triplet.relative_to(root).as_posix(),
        "triplet_sha256": sha256_file(triplet),
        "runtime_file_packages": dict(sorted(runtime_packages.items())),
        "packages_without_copyright_file": missing,
    }
    if os.name != "nt":
        runtime_without_license = sorted(
            set(runtime_packages.values()) - licensed_packages
        )
        if runtime_without_license:
            raise RuntimeError(
                "runtime vcpkg packages are missing license records: "
                + ", ".join(runtime_without_license)
            )
        result["runtime_packages_without_copyright_file"] = runtime_without_license
        return result

    bundle_dlls = runtime_files
    # The prebuilt FLM engine DLLs (dequant/gemm/q4_npu_eXpress and the per-model
    # *_npu.dll) live under src/lib and the per-backend src/lib/hrx. Both sets are
    # project-owned, so treat them as such rather than flagging them as
    # unclassified external runtime dependencies.
    fastflowlm_lib = root / "projects" / "FastFlowLM" / "src" / "lib"
    project_dlls = {
        path.name.lower()
        for lib_dir in (fastflowlm_lib, fastflowlm_lib / "hrx")
        for path in lib_dir.glob("*.dll")
    }
    project_dlls.update({"flm.exe", "hrx.dll"})
    microsoft_runtime = sorted(
        name
        for name in bundle_dlls
        if re.match(r"^(msvcp|vcruntime|concrt|ucrtbase|vcomp)", name)
    )
    notice = bundle / "licenses" / "Microsoft-Visual-Cpp-Runtime-NOTICE.txt"
    if microsoft_runtime:
        notice.write_text(
            "The listed Microsoft Visual C++ runtime files are redistributed under "
            "the license terms applicable to the Visual Studio toolchain used to "
            "build this artifact:\n  " + "\n  ".join(microsoft_runtime) + "\n",
            encoding="utf-8",
        )
    classified = set(runtime_packages) | project_dlls | set(microsoft_runtime)
    unclassified = sorted(bundle_dlls - classified)
    if unclassified:
        raise RuntimeError(
            "unclassified Windows runtime dependencies: " + ", ".join(unclassified)
        )
    runtime_without_license = sorted(set(runtime_packages.values()) - licensed_packages)
    if runtime_without_license:
        raise RuntimeError(
            "runtime vcpkg packages are missing license records: "
            + ", ".join(runtime_without_license)
        )
    result.update(
        {
            "microsoft_runtime_files": microsoft_runtime,
            "microsoft_runtime_license": "https://visualstudio.microsoft.com/license-terms/",
            "unclassified_runtime_files": unclassified,
            "runtime_packages_without_copyright_file": runtime_without_license,
        }
    )
    return result


def _linux_elf_files(bundle: Path) -> list[Path]:
    candidates = [
        bundle / "bin" / "flm",
        *bundle.glob("lib/*.so*"),
        *bundle.glob("lib/flm/*.so*"),
    ]
    return sorted(path for path in candidates if path.is_file())


def _is_linux_license_document(path: str, package: str) -> bool:
    normalized = path.replace("\\", "/")
    if "/usr/share/licenses/" in normalized:
        return True
    doc_marker = f"/usr/share/doc/{package}/"
    if doc_marker not in normalized:
        return False
    name = PurePosixPath(normalized).name.lower()
    return name.startswith(("copying", "copyright", "licence", "license", "notice"))


def _system_license_destination(
    bundle: Path, package: str, source: Path, source_package: str | None = None
) -> Path:
    normalized = source.as_posix()
    marker = "/usr/share/licenses/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        source_package_name = PurePosixPath(relative).parts[0]
        destination = bundle / "licenses" / "system" / package
        if source_package_name != package:
            destination /= source_package_name
        relative = PurePosixPath(relative).relative_to(source_package_name).as_posix()
        return destination / relative
    doc_package = source_package or package
    doc_marker = f"/usr/share/doc/{doc_package}/"
    if doc_marker in normalized:
        destination = bundle / "licenses" / "system" / package
        if doc_package != package:
            destination /= doc_package
        return destination / normalized.split(doc_marker, 1)[1]
    return bundle / "licenses" / "system" / package / source.name


_RPM_LICENSE_METADATA_QUERY = (
    "Name: %{NAME}\n"
    "Version: %{VERSION}\n"
    "Release: %{RELEASE}\n"
    "Architecture: %{ARCH}\n"
    "License: %{LICENSE}\n"
    "Source RPM: %{SOURCERPM}\n"
    "URL: %{URL}\n"
    "Summary: %{SUMMARY}\n"
)


def _rpm_license_metadata(package: str) -> str | None:
    result = subprocess.run(
        ["rpm", "-q", "--qf", _RPM_LICENSE_METADATA_QUERY, package],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return (
        "This RPM-owned runtime package did not install a standalone license "
        "file in a location known to the artifact packager. The RPM database "
        "reported this license metadata at packaging time:\n\n"
        f"{result.stdout.strip()}\n"
    )


def _rpm_license_entries(package: str) -> list[tuple[Path, str]]:
    listing = subprocess.run(
        ["rpm", "-ql", package], check=True, capture_output=True, text=True
    ).stdout
    return [
        (Path(line), package)
        for line in listing.splitlines()
        if _is_linux_license_document(line, package)
    ]


def _rpm_query_field(package: str, field: str) -> str | None:
    result = subprocess.run(
        ["rpm", "-q", "--qf", f"%{{{field}}}", package], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value and value != "(none)" else None


def _rpm_packages_from_source(source_rpm: str) -> list[str]:
    result = subprocess.run(
        ["rpm", "-qa", "--qf", "%{NAME}\t%{SOURCERPM}\n"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    packages = []
    for line in result.stdout.splitlines():
        try:
            package, source = line.split("\t", 1)
        except ValueError:
            continue
        if source == source_rpm:
            packages.append(package)
    return sorted(set(packages))


def _linux_soname(path: Path) -> str | None:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        return None
    return elf_soname(path, patchelf=patchelf)


def _linux_package_owner_candidates(path: Path) -> list[Path]:
    candidates = [path]
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return candidates
    if resolved != path:
        candidates.append(resolved)
    return candidates


def _is_linux_system_runtime(name: str) -> bool:
    return bool(
        re.match(r"^(ld-linux|linux-vdso|lib(c|dl|m|pthread|rt)\.so)([.-].*)?$", name)
    )


def _normalize_and_verify_linux_elf(bundle: Path) -> dict[str, list[str]]:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to package the Linux runtime closure")
    elf_files = _linux_elf_files(bundle)
    if not elf_files:
        raise RuntimeError("Linux runtime bundle contains no ELF payloads")

    for path in elf_files:
        relative = path.relative_to(bundle)
        if relative.parts[0] == "bin":
            rpath = "$ORIGIN/../lib/flm:$ORIGIN/../lib"
        elif relative.parent.as_posix() == "lib/flm":
            rpath = "$ORIGIN:$ORIGIN/.."
        else:
            rpath = "$ORIGIN:$ORIGIN/flm"
        set_elf_rpath(path, rpath, patchelf=patchelf)

    bundled_names = {path.name for path in elf_files}
    graph = {}
    missing = []
    for path in elf_files:
        needed = elf_needed(path, patchelf=patchelf)
        graph[path.relative_to(bundle).as_posix()] = sorted(needed)
        missing.extend(
            f"{path.name} -> {name}"
            for name in needed
            if name not in bundled_names and not _is_linux_system_runtime(name)
        )
    if missing:
        raise RuntimeError(
            "Linux runtime dependency closure is incomplete:\n  "
            + "\n  ".join(sorted(missing))
        )
    return graph


def _linux_dependencies_and_licenses(
    bundle: Path,
    vcpkg_runtime_files: set[str],
    internal_runtime_files: set[str] | None = None,
) -> dict[str, object]:
    ldconfig = shutil.which("ldconfig")
    library_sources = {}
    if ldconfig:
        output = subprocess.run(
            [ldconfig, "-p"], check=True, capture_output=True, text=True
        ).stdout
        for line in output.splitlines():
            match = re.match(r"\s*(\S+)\s+.*=>\s+(\S+)$", line)
            if match:
                library_sources.setdefault(match.group(1), Path(match.group(2)))

    package_files: dict[str, list[tuple[Path, str]]] = {}
    package_versions: dict[str, str] = {}
    package_license_source_packages: dict[str, list[str]] = {}
    rpm = shutil.which("rpm")
    dpkg_query = shutil.which("dpkg-query")
    mapped_libraries = {}
    unmapped_libraries = []
    internal_runtime_files = internal_runtime_files or set()
    for library in sorted(bundle.glob("lib/*.so*")):
        if (
            library.name in vcpkg_runtime_files
            or library.name in internal_runtime_files
        ):
            continue
        source = library_sources.get(library.name)
        if not source:
            soname = _linux_soname(library)
            if soname:
                source = library_sources.get(soname)
        package = None
        if source and rpm:
            for candidate in _linux_package_owner_candidates(source):
                query = subprocess.run(
                    [rpm, "-qf", "--qf", "%{NAME}\n", str(candidate)],
                    capture_output=True,
                    text=True,
                )
                if query.returncode == 0:
                    package = query.stdout.strip()
                    package_versions[package] = _run_text([rpm, "-q", package])
                    package_files[package] = _rpm_license_entries(package)
                    if not package_files[package]:
                        source_rpm = _rpm_query_field(package, "SOURCERPM")
                        if source_rpm:
                            for source_package in _rpm_packages_from_source(source_rpm):
                                if source_package == package:
                                    continue
                                entries = _rpm_license_entries(source_package)
                                if entries:
                                    package_files[package].extend(entries)
                                    package_license_source_packages.setdefault(
                                        package, []
                                    ).append(source_package)
                    break
        elif source and dpkg_query:
            for candidate in _linux_package_owner_candidates(source):
                query = subprocess.run(
                    [dpkg_query, "-S", str(candidate)], capture_output=True, text=True
                )
                if query.returncode == 0:
                    # `dpkg -S` can prefix diversion lines (e.g. for libreadline
                    # on t64 hosts) and may list several comma-separated packages
                    # before the ": <path>" owner. Skip diversions and take the
                    # first real package so dpkg-query -W is not handed a bogus
                    # name like "diversion by libreadline8t64 from".
                    package = None
                    for entry in query.stdout.splitlines():
                        if entry.startswith("diversion by "):
                            continue
                        owner = entry.split(":", 1)[0].strip()
                        if owner:
                            package = owner.split(",", 1)[0].strip()
                            break
                    if not package:
                        continue
                    package_versions[package] = _run_text(
                        [dpkg_query, "-W", "-f=${Package}=${Version}", package]
                    )
                    copyright_file = Path("/usr/share/doc") / package / "copyright"
                    package_files[package] = [(copyright_file, package)]
                    break
        if package:
            mapped_libraries[library.name] = package
        else:
            unmapped_libraries.append(library.name)

    copied_licenses = 0
    packages_with_licenses = set()
    for package, files in sorted(package_files.items()):
        for source, source_package in files:
            if source.is_file():
                destination = _system_license_destination(
                    bundle, package, source, source_package
                )
                _copy_checked(source, destination)
                copied_licenses += 1
                packages_with_licenses.add(package)
    generated_license_records = {}
    if rpm:
        for package in sorted(set(package_versions) - packages_with_licenses):
            metadata = _rpm_license_metadata(package)
            if not metadata:
                continue
            destination = (
                bundle / "licenses" / "system" / package / "RPM-LICENSE-METADATA.txt"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(metadata, encoding="utf-8")
            copied_licenses += 1
            packages_with_licenses.add(package)
            generated_license_records[package] = destination.relative_to(
                bundle
            ).as_posix()
    if package_versions and not copied_licenses:
        raise RuntimeError("no licenses found for bundled Linux system packages")
    missing_package_licenses = sorted(set(package_versions) - packages_with_licenses)
    if missing_package_licenses:
        raise RuntimeError(
            "bundled Linux packages are missing license records: "
            + ", ".join(missing_package_licenses)
        )
    unexpected_unmapped = sorted(unmapped_libraries)
    if unexpected_unmapped:
        raise RuntimeError(
            "bundled Linux libraries have no package owner: "
            + ", ".join(unexpected_unmapped)
        )
    return {
        "provider": "rpm" if rpm else "dpkg" if dpkg_query else "unknown",
        "packages": [
            {"name": name, "version": version}
            for name, version in sorted(package_versions.items())
        ],
        "library_packages": mapped_libraries,
        "unmapped_libraries": unmapped_libraries,
        "packages_without_license_files": missing_package_licenses,
        "license_source_packages": {
            package: sorted(set(source_packages))
            for package, source_packages in sorted(
                package_license_source_packages.items()
            )
        },
        "generated_license_records": generated_license_records,
    }


def artifact_name(system: str | None = None, machine: str | None = None) -> str:
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower().replace("amd64", "x86_64")
    return f"rocm-npu-{system}-{machine}"


def _patch_manifest(root: Path) -> list[dict[str, object]]:
    records = []
    for component in ("hrx", "FastFlowLM"):
        for patch in sorted((root / "patches" / component).glob("*.patch")):
            records.append(
                {
                    "path": patch.relative_to(root).as_posix(),
                    "size": patch.stat().st_size,
                    "sha256": sha256_file(patch),
                }
            )
    return records


def _tool_version(command: list[str]) -> str | None:
    try:
        return _run_text(command).splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _cmake_cache_values(cache: Path, names: list[str]) -> dict[str, str]:
    if not cache.is_file():
        return {}
    wanted = set(names)
    values = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if (
            line.startswith("//")
            or line.startswith("#")
            or "=" not in line
            or ":" not in line
        ):
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key in wanted:
            values[key] = value
    return values


def _cmake_configuration(cache: Path, root: Path) -> dict[str, str]:
    if not cache.is_file():
        return {}
    exact = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_C_COMPILER",
        "CMAKE_C_COMPILER_ID",
        "CMAKE_C_COMPILER_VERSION",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_COMPILER_ID",
        "CMAKE_CXX_COMPILER_VERSION",
        "CMAKE_GENERATOR",
        "CMAKE_GENERATOR_PLATFORM",
        "CMAKE_INSTALL_PREFIX",
        "CMAKE_MAKE_PROGRAM",
        "CMAKE_MSVC_RUNTIME_LIBRARY",
        "CMAKE_PREFIX_PATH",
        "CMAKE_TOOLCHAIN_FILE",
        "VCPKG_INSTALLED_DIR",
        "VCPKG_MANIFEST_DIR",
        "VCPKG_MANIFEST_INSTALL",
        "VCPKG_TARGET_TRIPLET",
    }
    prefixes = (
        "CMAKE_C_FLAGS",
        "CMAKE_CXX_FLAGS",
        "CMAKE_EXE_LINKER_FLAGS",
        "CMAKE_SHARED_LINKER_FLAGS",
        "FASTFLOWLM_",
        "FLM_",
        "HRX_",
        "IREE_HAL_DRIVER_",
        "LIBHRX_",
        "LOOM_",
        "NPU_",
    )
    values = {}
    root_text = str(root.resolve()).replace("\\", "/")
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("//", "#")) or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if key not in exact and not key.startswith(prefixes):
            continue
        values[key] = value.replace("\\", "/").replace(root_text, "${SOURCE_ROOT}")
    return dict(sorted(values.items()))


def stage_runtime_bundle(
    root: Path, flm_install: Path, hrx_install: Path, destination: Path
) -> Path:
    """Creates the bundle root and writes its complete hash manifest."""
    cli_install = root / "build" / "rocm-npu-cli-install"
    cli_marker = verify_cli_fresh_install(
        root=root,
        cli_install=cli_install,
        hrx_install=hrx_install,
        hrx_source=root / "projects" / "hrx",
        marker_name=".rns-fresh-build.json",
    )
    bundle = destination / artifact_name()
    reset_directory(destination, root / "build")
    shutil.copytree(flm_install, bundle)
    build_marker = bundle / ".rns-fresh-build.json"
    if build_marker.exists():
        build_marker.unlink()

    binary_dependencies = None
    if os.name == "nt":
        _copy_checked(hrx_install / "bin" / "hrx.dll", bundle / "bin" / "hrx.dll")
        cli_name = "rocm-npu.exe"
    else:
        cli_name = "rocm-npu"
        hrx_libraries = sorted((hrx_install / "lib").glob("libhrx.so*"))
        if not hrx_libraries:
            raise RuntimeError(
                f"installed HRX runtime not found under {hrx_install / 'lib'}"
            )
        for library in hrx_libraries:
            _copy_checked(library, bundle / "lib" / library.name)

    _copy_checked(cli_install / "bin" / cli_name, bundle / "bin" / cli_name)
    fixture_source = cli_install / "share" / "rocm-npu" / "self-test"
    if not fixture_source.is_dir():
        raise RuntimeError(
            f"installed rocm-npu self-test data not found: {fixture_source}"
        )
    shutil.copytree(
        fixture_source,
        bundle / "share" / "rocm-npu" / "self-test",
        dirs_exist_ok=True,
    )
    if os.name == "nt":
        # Include imports introduced uniquely by the native CLI.
        _copy_windows_redistributable_closure(bundle)
        binary_dependencies = _windows_pe_dependency_graph(bundle)
    else:
        binary_dependencies = _normalize_and_verify_linux_elf(bundle)

    license_sources = [
        (root / "LICENSE", "rocm-npu-staging-LICENSE"),
        (root / "projects" / "hrx" / "LICENSE", "hrx-LICENSE"),
        (
            root / "projects" / "FastFlowLM" / "LICENSE_RUNTIME.txt",
            "FastFlowLM-LICENSE_RUNTIME.txt",
        ),
        (root / "projects" / "FastFlowLM" / "TERMS.md", "FastFlowLM-TERMS.md"),
        (
            root / "third_party" / "nlohmann_json" / "LICENSE.MIT",
            "nlohmann-json-LICENSE.MIT",
        ),
        (root / "docs" / "third-party-notices.md", "THIRD_PARTY_NOTICES.md"),
    ]
    for source, name in license_sources:
        if not source.is_file():
            raise RuntimeError(f"required license file missing: {source}")
        _copy_checked(source, bundle / "licenses" / name)

    vcpkg_root = Path(dependency_paths(root)["triplet_root"])
    vcpkg_dependencies = _vcpkg_dependencies_and_licenses(root, bundle, vcpkg_root)
    if os.name == "nt":
        dependencies = vcpkg_dependencies
    else:
        internal_runtime_files = {
            path.name for path in (bundle / "lib" / "flm").glob("*.so*")
        }
        internal_runtime_files.update(
            path.name for path in (bundle / "lib").glob("libhrx.so*")
        )
        system_dependencies = _linux_dependencies_and_licenses(
            bundle,
            set(vcpkg_dependencies["runtime_file_packages"]),
            internal_runtime_files,
        )
        dependencies = {
            "provider": "vcpkg+system",
            "vcpkg": vcpkg_dependencies,
            "system": system_dependencies,
        }

    files = file_inventory(bundle)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "version": default_package_version(root),
        "product": "FastFlowLM + HRX amdxdna runtime",
        "artifact": artifact_name(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "sources": {
            "rocm-npu-staging": git_source_state(root),
            "hrx": git_source_state(root / "projects" / "hrx"),
            "FastFlowLM": git_source_state(root / "projects" / "FastFlowLM"),
        },
        "cli_build": cli_marker,
        "patches": _patch_manifest(root),
        "dependencies": dependencies,
        "components": [
            {
                "name": "nlohmann/json",
                "version": "3.12.0",
                "upstream": "https://github.com/nlohmann/json",
                "license": "MIT",
            }
        ],
        "binary_dependencies": binary_dependencies,
        "toolchain": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": _tool_version(["git", "--version"]),
            "cmake": _tool_version(["cmake", "--version"]),
            "ninja": _tool_version(["ninja", "--version"]),
            "rustc": _tool_version(["rustc", "--version"]),
            "cargo": _tool_version(["cargo", "--version"]),
            "zstd": _tool_version(["zstd", "--version"]),
            "patchelf": _tool_version(["patchelf", "--version"]),
            "ccache": _tool_version(["ccache", "--version"]),
            "cc": os.environ.get("CC"),
            "cxx": os.environ.get("CXX"),
            "build_image": os.environ.get("RNS_BUILD_IMAGE"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
            "runner_image_os": os.environ.get("ImageOS"),
            "runner_image_version": os.environ.get("ImageVersion"),
            "hrx_cmake": _cmake_cache_values(
                root / "projects" / "hrx" / "build" / "cmake" / "CMakeCache.txt",
                [
                    "CMAKE_C_COMPILER",
                    "CMAKE_C_COMPILER_ID",
                    "CMAKE_C_COMPILER_VERSION",
                    "CMAKE_CXX_COMPILER",
                    "CMAKE_CXX_COMPILER_ID",
                    "CMAKE_CXX_COMPILER_VERSION",
                ],
            ),
            "fastflowlm_cmake": _cmake_cache_values(
                root / "build" / "FastFlowLM" / "CMakeCache.txt",
                [
                    "CMAKE_CXX_COMPILER",
                    "CMAKE_CXX_COMPILER_ID",
                    "CMAKE_CXX_COMPILER_VERSION",
                ],
            ),
            "hrx_configuration": _cmake_configuration(
                root / "projects" / "hrx" / "build" / "cmake" / "CMakeCache.txt", root
            ),
            "fastflowlm_configuration": _cmake_configuration(
                root / "build" / "FastFlowLM" / "CMakeCache.txt", root
            ),
            "effective_install_prefixes": {
                "hrx": str(hrx_install.resolve())
                .replace("\\", "/")
                .replace(str(root.resolve()).replace("\\", "/"), "${SOURCE_ROOT}"),
                "fastflowlm": str(flm_install.resolve())
                .replace("\\", "/")
                .replace(str(root.resolve()).replace("\\", "/"), "${SOURCE_ROOT}"),
            },
        },
        "files": files,
    }
    validate_source_policy(manifest, root)
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle


def verify_bundle(bundle: Path) -> dict[str, object]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"bundle manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported bundle manifest schema: {manifest.get('schema_version')}"
        )

    verify_file_inventory(bundle, manifest.get("files"), exclude_paths=(manifest_path,))
    return manifest


def create_archive(bundle: Path, output_directory: Path) -> Path:
    verify_bundle(bundle)
    output_directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        archive = output_directory / f"{bundle.name}.zip"
        create_deterministic_zip(
            bundle,
            archive,
            arcname=bundle.name,
            compresslevel=9,
            executable=lambda path: path.name == "flm.exe",
        )
    else:
        archive = output_directory / f"{bundle.name}.tar.zst"
        create_deterministic_tar_zstd_cli(
            bundle, archive, arcname=bundle.name, level=19
        )
    archive_digest = sha256_file(archive)
    write_sha256_sidecar(archive, digest=archive_digest)
    archive.with_name(archive.name + ".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archive": archive.name,
                "archive_sha256": archive_digest,
                "bundle": json.loads(
                    (bundle / "manifest.json").read_text(encoding="utf-8")
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return archive


def extract_archive(archive: Path, destination: Path, *, allowed_root: Path) -> Path:
    """Verifies in a temporary sibling and promotes only a complete bundle."""
    validate_directory_target(destination, allowed_root)
    return extract_archive_atomic(
        archive,
        destination,
        allowed_root=allowed_root,
        verify_root=verify_bundle,
    )
