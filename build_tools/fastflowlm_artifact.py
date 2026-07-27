# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and verify relocatable FastFlowLM + HRX runtime artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

from build_tools.fastflowlm_deps import dependency_paths


SCHEMA_VERSION = 1
FIXED_MTIME = 946684800  # 2000-01-01T00:00:00Z


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zstandard_module():
    # dev.py imports this module before `dev.py setup` creates the venv, so keep
    # the optional codec import at the point where an archive actually needs it.
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError(
            "the zstandard Python package is required for .tar.zst artifacts; "
            "run: python dev.py setup"
        ) from error
    return zstandard


def _run_text(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _git_revision(path: Path) -> str:
    return _run_text(["git", "rev-parse", "HEAD"], cwd=path)


def git_source_state(path: Path) -> dict[str, object]:
    """Describes the exact tracked and non-ignored source state used to build."""
    diff = subprocess.run(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD"], cwd=path,
        check=True, capture_output=True,
    ).stdout
    tracked_output = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACDMRTUXB", "-z", "HEAD"],
        cwd=path, check=True, capture_output=True,
    ).stdout
    untracked_output = subprocess.run(
        ["git", "ls-files", "-o", "--exclude-standard", "-z"],
        cwd=path, check=True, capture_output=True,
    ).stdout
    changed_paths = sorted({
        item.decode("utf-8", errors="surrogateescape")
        for output in (tracked_output, untracked_output)
        for item in output.split(b"\0") if item
    })
    changed_files = []
    for relative in changed_paths:
        source = path / relative
        record: dict[str, object] = {"path": relative.replace("\\", "/")}
        if source.is_file():
            record.update({"size": source.stat().st_size,
                           "sha256": sha256_file(source)})
        else:
            record["deleted"] = True
        changed_files.append(record)
    return {
        "revision": _git_revision(path),
        "dirty": bool(changed_files),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "changed_files": changed_files,
    }


def directory_state(path: Path, *, exclude_names: tuple[str, ...] = ()) -> dict[str, object]:
    files = []
    for source in sorted(item for item in path.rglob("*") if item.is_file()):
        if source.name in exclude_names:
            continue
        files.append({"path": source.relative_to(path).as_posix(),
                      "size": source.stat().st_size,
                      "sha256": sha256_file(source)})
    digest = hashlib.sha256()
    for record in files:
        digest.update(json.dumps(record, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return {"sha256": digest.hexdigest(), "file_count": len(files)}


def verify_fastflowlm_fresh_installs(
    *,
    hrx_install: Path,
    hrx_source: Path,
    fastflowlm_install: Path,
    fastflowlm_source: Path,
    marker_name: str,
) -> dict[str, dict[str, object]]:
    """Verifies that fresh HRX and FastFlowLM installs still match their sources."""
    installed_flm = fastflowlm_install / "bin" / (
        "flm.exe" if os.name == "nt" else "flm"
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


def validate_source_policy(manifest: dict[str, object], root: Path) -> None:
    versions = tomllib.loads(
        (root / "manifests" / "versions.toml").read_text(encoding="utf-8"))
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
                f"artifact {component} revision does not match the pinned revision")
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


def validate_directory_target(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"refusing to remove directory outside {root}: {resolved}")


def reset_directory(path: Path, allowed_root: Path) -> None:
    validate_directory_target(path, allowed_root)
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


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


def pe_imports(path: Path) -> list[str]:
    data = path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        raise RuntimeError(f"not a PE binary: {path}")
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if pe_offset + 24 > len(data) or data[pe_offset:pe_offset + 4] != b"PE\0\0":
        raise RuntimeError(f"invalid PE header: {path}")
    coff = pe_offset + 4
    section_count = struct.unpack_from("<H", data, coff + 2)[0]
    optional_size = struct.unpack_from("<H", data, coff + 16)[0]
    optional = coff + 20
    if optional + optional_size > len(data):
        raise RuntimeError(f"truncated PE optional header: {path}")
    magic = struct.unpack_from("<H", data, optional)[0]
    if magic == 0x20B:
        directory_offset = optional + 112
        image_base = struct.unpack_from("<Q", data, optional + 24)[0]
    elif magic == 0x10B:
        directory_offset = optional + 96
        image_base = struct.unpack_from("<I", data, optional + 28)[0]
    else:
        raise RuntimeError(f"unsupported PE optional-header magic: {path}")
    section_offset = optional + optional_size
    sections = []
    for index in range(section_count):
        entry = section_offset + index * 40
        if entry + 40 > len(data):
            raise RuntimeError(f"truncated PE section table: {path}")
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, entry + 8)
        sections.append((virtual_address, max(virtual_size, raw_size), raw_offset))

    def rva_offset(rva: int) -> int:
        for virtual_address, size, raw_offset in sections:
            if virtual_address <= rva < virtual_address + size:
                result = raw_offset + rva - virtual_address
                if result >= len(data):
                    break
                return result
        raise RuntimeError(f"PE RVA 0x{rva:x} is not backed by a section: {path}")

    def c_string(rva: int) -> str:
        offset = rva_offset(rva)
        end = data.find(b"\0", offset)
        if end < 0:
            raise RuntimeError(f"unterminated PE import name: {path}")
        return data[offset:end].decode("ascii", errors="strict").lower()

    imports = set()
    directories = ((1, 20, 12, False), (13, 32, 4, True))
    for directory_index, descriptor_size, name_field, is_delay in directories:
        entry = directory_offset + directory_index * 8
        if entry + 8 > optional + optional_size:
            continue
        table_rva, table_size = struct.unpack_from("<II", data, entry)
        if not table_rva or not table_size:
            continue
        offset = rva_offset(table_rva)
        limit = min(len(data), offset + table_size)
        while offset + descriptor_size <= limit:
            descriptor = data[offset:offset + descriptor_size]
            if not any(descriptor):
                break
            name_rva = struct.unpack_from("<I", descriptor, name_field)[0]
            if is_delay and not (struct.unpack_from("<I", descriptor, 0)[0] & 1):
                if name_rva < image_base:
                    raise RuntimeError(f"invalid delay-import VA: {path}")
                name_rva -= image_base
            imports.add(c_string(name_rva))
            offset += descriptor_size
    return sorted(imports)


def _windows_pe_dependency_graph(bundle: Path) -> dict[str, list[str]]:
    binaries = sorted([*(bundle / "bin").glob("*.exe"),
                       *(bundle / "bin").glob("*.dll")])
    bundled = {path.name.lower() for path in binaries}
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    graph = {}
    missing = []
    for binary in binaries:
        imports = pe_imports(binary)
        graph[binary.relative_to(bundle).as_posix()] = imports
        for name in imports:
            is_api_set = name.startswith(("api-ms-", "ext-ms-"))
            if name not in bundled and not is_api_set and not (system32 / name).is_file():
                missing.append(f"{binary.name} -> {name}")
    if missing:
        raise RuntimeError("Windows PE dependency closure is incomplete:\n  " +
                           "\n  ".join(sorted(missing)))
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
                    path for path in redist_root.rglob("*.dll")
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


def _vcpkg_dependencies_and_licenses(root: Path, bundle: Path,
                                     vcpkg_root: Path) -> dict[str, object]:
    packages = _parse_control_file(vcpkg_root.parent / "vcpkg" / "status")
    if not packages:
        packages = _parse_control_file(root / ".cache" / "vcpkg-installed" /
                                       "vcpkg" / "status")
    if not packages:
        raise RuntimeError("vcpkg dependency inventory is missing")

    licensed_packages = set()
    for copyright_file in sorted((vcpkg_root / "share").glob("*/copyright")):
        package = copyright_file.parent.name
        _copy_checked(copyright_file,
                      bundle / "licenses" / "vcpkg" / package / "copyright")
        licensed_packages.add(package)
    missing = sorted({record["package"] for record in packages
                      if record.get("package") not in licensed_packages and
                      not record.get("package", "").startswith("vcpkg-")})
    runtime_packages = {}
    info_root = vcpkg_root.parent / "vcpkg" / "info"
    if os.name == "nt":
        runtime_files = {path.name.lower() for path in (bundle / "bin").glob("*.dll")}
    else:
        runtime_files = {
            path.name for pattern in ("lib/*.so*", "lib/flm/*.so*")
            for path in bundle.glob(pattern)
        }
    for record in packages:
        package = record.get("package")
        if not package:
            continue
        for listing in info_root.glob(f"{package}_*.list"):
            for line in listing.read_text(encoding="utf-8", errors="replace").splitlines():
                name = Path(line.replace("/", os.sep)).name
                if os.name == "nt":
                    name = name.lower()
                if name in runtime_files:
                    runtime_packages[name] = package
    triplet_name = vcpkg_root.name
    triplet = root / ".github" / "vcpkg" / "triplets" / f"{triplet_name}.cmake"
    resolved_package_names = sorted({
        record["package"] for record in packages if record.get("package")
    })
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
        "packages_without_shipped_runtime_files":
            packages_without_shipped_runtime_files,
        "vcpkg_revision": _git_revision(root / ".cache" / "vcpkg-src"),
        "triplet": triplet.relative_to(root).as_posix(),
        "triplet_sha256": sha256_file(triplet),
        "runtime_file_packages": dict(sorted(runtime_packages.items())),
        "packages_without_copyright_file": missing,
    }
    if os.name != "nt":
        runtime_without_license = sorted(set(runtime_packages.values()) - licensed_packages)
        if runtime_without_license:
            raise RuntimeError("runtime vcpkg packages are missing license records: " +
                               ", ".join(runtime_without_license))
        result["runtime_packages_without_copyright_file"] = runtime_without_license
        return result

    bundle_dlls = runtime_files
    project_dlls = {
        path.name.lower() for path in (root / "projects" / "FastFlowLM" / "src" / "lib").glob("*.dll")
    }
    project_dlls.update({"flm.exe", "hrx.dll"})
    microsoft_runtime = sorted(
        name for name in bundle_dlls
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
        raise RuntimeError("unclassified Windows runtime dependencies: " +
                           ", ".join(unclassified))
    runtime_without_license = sorted(
        set(runtime_packages.values()) - licensed_packages)
    if runtime_without_license:
        raise RuntimeError("runtime vcpkg packages are missing license records: " +
                           ", ".join(runtime_without_license))
    result.update({
        "microsoft_runtime_files": microsoft_runtime,
        "microsoft_runtime_license": "https://visualstudio.microsoft.com/license-terms/",
        "unclassified_runtime_files": unclassified,
        "runtime_packages_without_copyright_file": runtime_without_license,
    })
    return result


def _linux_elf_files(bundle: Path) -> list[Path]:
    candidates = [bundle / "bin" / "flm", *bundle.glob("lib/*.so*"),
                  *bundle.glob("lib/flm/*.so*")]
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


def _system_license_destination(bundle: Path, package: str, source: Path,
                                source_package: str | None = None) -> Path:
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
        capture_output=True, text=True,
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
    listing = subprocess.run(["rpm", "-ql", package], check=True,
                             capture_output=True, text=True).stdout
    return [(Path(line), package) for line in listing.splitlines()
            if _is_linux_license_document(line, package)]


def _rpm_query_field(package: str, field: str) -> str | None:
    result = subprocess.run(["rpm", "-q", "--qf", f"%{{{field}}}", package],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value and value != "(none)" else None


def _rpm_packages_from_source(source_rpm: str) -> list[str]:
    result = subprocess.run(["rpm", "-qa", "--qf", "%{NAME}\t%{SOURCERPM}\n"],
                            capture_output=True, text=True)
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
    result = subprocess.run([patchelf, "--print-soname", str(path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None
    soname = result.stdout.strip()
    return soname or None


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
    return bool(re.match(
        r"^(ld-linux|linux-vdso|lib(c|dl|m|pthread|rt)\.so)([.-].*)?$", name))


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
        subprocess.run([patchelf, "--set-rpath", rpath, str(path)], check=True)

    bundled_names = {path.name for path in elf_files}
    graph = {}
    missing = []
    for path in elf_files:
        needed = subprocess.run(
            [patchelf, "--print-needed", str(path)], check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        graph[path.relative_to(bundle).as_posix()] = sorted(needed)
        missing.extend(f"{path.name} -> {name}" for name in needed
                       if name not in bundled_names and
                       not _is_linux_system_runtime(name))
    if missing:
        raise RuntimeError("Linux runtime dependency closure is incomplete:\n  " +
                           "\n  ".join(sorted(missing)))
    return graph


def _linux_dependencies_and_licenses(
        bundle: Path, vcpkg_runtime_files: set[str],
        internal_runtime_files: set[str] | None = None) -> dict[str, object]:
    ldconfig = shutil.which("ldconfig")
    library_sources = {}
    if ldconfig:
        output = subprocess.run([ldconfig, "-p"], check=True, capture_output=True,
                                text=True).stdout
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
        if library.name in vcpkg_runtime_files or library.name in internal_runtime_files:
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
                    capture_output=True, text=True)
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
                                        package, []).append(source_package)
                    break
        elif source and dpkg_query:
            for candidate in _linux_package_owner_candidates(source):
                query = subprocess.run([dpkg_query, "-S", str(candidate)],
                                       capture_output=True, text=True)
                if query.returncode == 0:
                    package = query.stdout.split(":", 1)[0]
                    package_versions[package] = _run_text(
                        [dpkg_query, "-W", "-f=${Package}=${Version}", package])
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
                    bundle, package, source, source_package)
                _copy_checked(source, destination)
                copied_licenses += 1
                packages_with_licenses.add(package)
    generated_license_records = {}
    if rpm:
        for package in sorted(set(package_versions) - packages_with_licenses):
            metadata = _rpm_license_metadata(package)
            if not metadata:
                continue
            destination = (bundle / "licenses" / "system" / package /
                           "RPM-LICENSE-METADATA.txt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(metadata, encoding="utf-8")
            copied_licenses += 1
            packages_with_licenses.add(package)
            generated_license_records[package] = destination.relative_to(
                bundle).as_posix()
    if package_versions and not copied_licenses:
        raise RuntimeError("no licenses found for bundled Linux system packages")
    missing_package_licenses = sorted(set(package_versions) - packages_with_licenses)
    if missing_package_licenses:
        raise RuntimeError("bundled Linux packages are missing license records: " +
                           ", ".join(missing_package_licenses))
    unexpected_unmapped = sorted(unmapped_libraries)
    if unexpected_unmapped:
        raise RuntimeError("bundled Linux libraries have no package owner: " +
                           ", ".join(unexpected_unmapped))
    return {
        "provider": "rpm" if rpm else "dpkg" if dpkg_query else "unknown",
        "packages": [{"name": name, "version": version}
                     for name, version in sorted(package_versions.items())],
        "library_packages": mapped_libraries,
        "unmapped_libraries": unmapped_libraries,
        "packages_without_license_files": missing_package_licenses,
        "license_source_packages": {
            package: sorted(set(source_packages))
            for package, source_packages in sorted(
                package_license_source_packages.items())
        },
        "generated_license_records": generated_license_records,
    }


def artifact_name(system: str | None = None, machine: str | None = None) -> str:
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower().replace("amd64", "x86_64")
    return f"fastflowlm-hrx-{system}-{machine}"


def _patch_manifest(root: Path) -> list[dict[str, object]]:
    records = []
    for component in ("hrx", "FastFlowLM"):
        for patch in sorted((root / "patches" / component).glob("*.patch")):
            records.append({
                "path": patch.relative_to(root).as_posix(),
                "size": patch.stat().st_size,
                "sha256": sha256_file(patch),
            })
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
        if line.startswith("//") or line.startswith("#") or "=" not in line or ":" not in line:
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
        "CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_C_COMPILER_ID",
        "CMAKE_C_COMPILER_VERSION", "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION",
        "CMAKE_GENERATOR", "CMAKE_GENERATOR_PLATFORM", "CMAKE_INSTALL_PREFIX",
        "CMAKE_MAKE_PROGRAM", "CMAKE_MSVC_RUNTIME_LIBRARY",
        "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE", "VCPKG_INSTALLED_DIR",
        "VCPKG_MANIFEST_DIR", "VCPKG_MANIFEST_INSTALL", "VCPKG_TARGET_TRIPLET",
    }
    prefixes = ("CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_EXE_LINKER_FLAGS",
                "CMAKE_SHARED_LINKER_FLAGS", "FASTFLOWLM_", "FLM_", "HRX_",
                "IREE_HAL_DRIVER_",
                "LIBHRX_", "LOOM_", "NPU_")
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


def stage_runtime_bundle(root: Path, flm_install: Path, hrx_install: Path,
                         destination: Path) -> Path:
    """Creates the bundle root and writes its complete hash manifest."""
    bundle = destination / artifact_name()
    reset_directory(destination, root / "build")
    shutil.copytree(flm_install, bundle)
    build_marker = bundle / ".rns-fresh-build.json"
    if build_marker.exists():
        build_marker.unlink()

    binary_dependencies = None
    if os.name == "nt":
        _copy_checked(hrx_install / "bin" / "hrx.dll", bundle / "bin" / "hrx.dll")
        _copy_windows_redistributable_closure(bundle)
        binary_dependencies = _windows_pe_dependency_graph(bundle)
    else:
        hrx_libraries = sorted((hrx_install / "lib").glob("libhrx.so*"))
        if not hrx_libraries:
            raise RuntimeError(f"installed HRX runtime not found under {hrx_install / 'lib'}")
        for library in hrx_libraries:
            _copy_checked(library, bundle / "lib" / library.name)
        binary_dependencies = _normalize_and_verify_linux_elf(bundle)

    license_sources = [
        (root / "LICENSE", "rocm-npu-staging-LICENSE"),
        (root / "projects" / "hrx" / "LICENSE", "hrx-LICENSE"),
        (root / "projects" / "FastFlowLM" / "LICENSE_RUNTIME.txt", "FastFlowLM-LICENSE_RUNTIME.txt"),
        (root / "projects" / "FastFlowLM" / "TERMS.md", "FastFlowLM-TERMS.md"),
    ]
    for source, name in license_sources:
        if not source.is_file():
            raise RuntimeError(f"required license file missing: {source}")
        _copy_checked(source, bundle / "licenses" / name)

    vcpkg_root = Path(dependency_paths(root)["triplet_root"])
    vcpkg_dependencies = _vcpkg_dependencies_and_licenses(
        root, bundle, vcpkg_root)
    if os.name == "nt":
        dependencies = vcpkg_dependencies
    else:
        internal_runtime_files = {
            path.name for path in (bundle / "lib" / "flm").glob("*.so*")
        }
        internal_runtime_files.update(
            path.name for path in (bundle / "lib").glob("libhrx.so*"))
        system_dependencies = _linux_dependencies_and_licenses(
            bundle, set(vcpkg_dependencies["runtime_file_packages"]),
            internal_runtime_files)
        dependencies = {
            "provider": "vcpkg+system",
            "vcpkg": vcpkg_dependencies,
            "system": system_dependencies,
        }

    files = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        relative = path.relative_to(bundle).as_posix()
        files.append({"path": relative, "size": path.stat().st_size,
                      "sha256": sha256_file(path)})

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "product": "FastFlowLM + HRX amdxdna runtime",
        "artifact": artifact_name(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "sources": {
            "rocm-npu-staging": git_source_state(root),
            "hrx": git_source_state(root / "projects" / "hrx"),
            "FastFlowLM": git_source_state(root / "projects" / "FastFlowLM"),
        },
        "patches": _patch_manifest(root),
        "dependencies": dependencies,
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
                ["CMAKE_C_COMPILER", "CMAKE_C_COMPILER_ID", "CMAKE_C_COMPILER_VERSION",
                 "CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION"]),
            "fastflowlm_cmake": _cmake_cache_values(
                root / "build" / "FastFlowLM" / "CMakeCache.txt",
                ["CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION"]),
            "hrx_configuration": _cmake_configuration(
                root / "projects" / "hrx" / "build" / "cmake" / "CMakeCache.txt", root),
            "fastflowlm_configuration": _cmake_configuration(
                root / "build" / "FastFlowLM" / "CMakeCache.txt", root),
            "effective_install_prefixes": {
                "hrx": str(hrx_install.resolve()).replace("\\", "/").replace(
                    str(root.resolve()).replace("\\", "/"), "${SOURCE_ROOT}"),
                "fastflowlm": str(flm_install.resolve()).replace("\\", "/").replace(
                    str(root.resolve()).replace("\\", "/"), "${SOURCE_ROOT}"),
            },
        },
        "files": files,
    }
    validate_source_policy(manifest, root)
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def verify_bundle(bundle: Path) -> dict[str, object]:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"bundle manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported bundle manifest schema: {manifest.get('schema_version')}")

    file_records = manifest.get("files")
    if not isinstance(file_records, list):
        raise RuntimeError("bundle manifest files must be a list")
    expected = {}
    for entry in file_records:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise RuntimeError("bundle manifest contains an invalid file record")
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) != entry["path"]:
            raise RuntimeError(f"bundle manifest contains an unsafe path: {entry['path']}")
        if entry["path"] in expected:
            raise RuntimeError(f"bundle manifest contains a duplicate path: {entry['path']}")
        if not isinstance(entry.get("size"), int) or entry["size"] < 0 or not re.fullmatch(
                r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise RuntimeError(f"bundle manifest contains invalid metadata: {entry['path']}")
        expected[entry["path"]] = entry
    actual = {
        path.relative_to(bundle).as_posix(): path
        for path in bundle.rglob("*") if path.is_file() and path != manifest_path
    }
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise RuntimeError(f"bundle inventory mismatch; missing={missing}, unexpected={unexpected}")
    for relative, entry in expected.items():
        path = actual[relative]
        if path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"bundle file verification failed: {relative}")
    return manifest


def _zip_bundle(bundle: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
            relative = (PurePosixPath(bundle.name) / path.relative_to(bundle).as_posix()).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            if path.name == "flm.exe":
                info.external_attr = 0o100755 << 16
            output.writestr(info, path.read_bytes(), compresslevel=9)


def _tar_zst_bundle(bundle: Path, archive: Path) -> None:
    zstd = shutil.which("zstd")
    if not zstd:
        raise RuntimeError("zstd is required to create the Linux .tar.zst artifact")
    with tempfile.TemporaryDirectory(prefix="fastflowlm-tar-") as temporary:
        tar_path = Path(temporary) / "bundle.tar"
        with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as output:
            for path in sorted([bundle, *bundle.rglob("*")]):
                relative = PurePosixPath(bundle.name) / path.relative_to(bundle).as_posix()
                info = tarfile.TarInfo(relative.as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = FIXED_MTIME
                if path.is_file():
                    if path.is_symlink():
                        resolved = path.resolve()
                        bundle_root = bundle.resolve()
                        if bundle_root not in resolved.parents or not resolved.is_file():
                            raise RuntimeError(
                                f"artifact symlink escapes bundle or is not a file: {path}")
                    info.type = tarfile.REGTYPE
                    info.size = path.stat().st_size
                    mode = stat.S_IMODE(path.stat().st_mode)
                    info.mode = 0o755 if mode & 0o111 else 0o644
                    with path.open("rb") as stream:
                        output.addfile(info, stream)
                elif path.is_dir() and not path.is_symlink():
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    output.addfile(info)
                else:
                    raise RuntimeError(f"unsupported artifact filesystem entry: {path}")
        subprocess.run([zstd, "-q", "-19", "-T0", "-f", str(tar_path), "-o", str(archive)], check=True)


def create_archive(bundle: Path, output_directory: Path) -> Path:
    verify_bundle(bundle)
    output_directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        archive = output_directory / f"{bundle.name}.zip"
        _zip_bundle(bundle, archive)
    else:
        archive = output_directory / f"{bundle.name}.tar.zst"
        _tar_zst_bundle(bundle, archive)
    archive.with_name(archive.name + ".sha256").write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="ascii")
    return archive


def _safe_member_path(destination: Path, name: str) -> Path:
    path = (destination / Path(PurePosixPath(name))).resolve()
    destination = destination.resolve()
    if path != destination and destination not in path.parents:
        raise RuntimeError(f"archive path escapes extraction root: {name}")
    return path


def _extract_archive_contents(archive: Path, destination: Path) -> Path:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as source:
            names = [member.filename for member in source.infolist()]
            if len(names) != len(set(names)):
                raise RuntimeError(f"duplicate paths are not allowed in runtime artifacts: {archive}")
            targets = set()
            for member in source.infolist():
                target = _safe_member_path(destination, member.filename)
                normalized = os.path.normcase(str(target))
                if normalized in targets:
                    raise RuntimeError(
                        f"duplicate normalized path in runtime artifact: {member.filename}")
                targets.add(normalized)
                mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise RuntimeError(
                        f"links are not allowed in runtime artifacts: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as input_stream, target.open("wb") as output_stream:
                        shutil.copyfileobj(input_stream, output_stream)
    elif archive.name.endswith((".tar.gz", ".tar.zst")):
        raw_stream = archive.open("rb")
        decoded_stream = None
        zstd_process = None
        try:
            if archive.name.endswith(".tar.zst"):
                try:
                    decoded_stream = _zstandard_module().ZstdDecompressor().stream_reader(
                        raw_stream)
                except RuntimeError as module_error:
                    zstd = shutil.which("zstd")
                    if not zstd:
                        raise module_error
                    zstd_process = subprocess.Popen(
                        [zstd, "-q", "-d", "-c", str(archive)],
                        stdout=subprocess.PIPE,
                    )
                    decoded_stream = zstd_process.stdout
                tar_stream = decoded_stream
                mode = "r|"
            else:
                tar_stream = raw_stream
                mode = "r:gz"
            with tarfile.open(fileobj=tar_stream, mode=mode) as source:
                names = set()
                targets = set()
                for member in source:
                    if member.name in names:
                        raise RuntimeError(
                            f"duplicate paths are not allowed in runtime artifacts: {archive}")
                    names.add(member.name)
                    target = _safe_member_path(destination, member.name)
                    normalized = os.path.normcase(str(target))
                    if normalized in targets:
                        raise RuntimeError(
                            f"duplicate normalized path in runtime artifact: {member.name}")
                    targets.add(normalized)
                    if not (member.isfile() or member.isdir()):
                        raise RuntimeError(
                            f"special files are not allowed in runtime artifacts: {member.name}")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        input_stream = source.extractfile(member)
                        if input_stream is None:
                            raise RuntimeError(f"cannot read archive member: {member.name}")
                        with input_stream, target.open("wb") as output_stream:
                            shutil.copyfileobj(input_stream, output_stream)
                        target.chmod(member.mode & 0o777)
        finally:
            if decoded_stream is not None:
                decoded_stream.close()
            raw_stream.close()
            if zstd_process is not None and zstd_process.wait() != 0:
                raise RuntimeError(
                    f"zstd failed to extract {archive} (exit {zstd_process.returncode})")
    else:
        raise RuntimeError(f"unsupported artifact format: {archive}")

    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"artifact must contain exactly one root directory: {archive}")
    verify_bundle(roots[0])
    return roots[0]


def extract_archive(archive: Path, destination: Path, *, allowed_root: Path) -> Path:
    """Verifies in a temporary sibling and promotes only a complete bundle."""
    validate_directory_target(destination, allowed_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}-extract-", dir=destination.parent))
    try:
        extracted = _extract_archive_contents(archive, temporary)
        root_name = extracted.name
        if destination.is_symlink():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return destination / root_name
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
