# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check the standalone runtime archive's HRX CMake package and wheel split."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_tools.fastflowlm_artifact import extract_archive

_PROBE_CMAKE = r"""
cmake_minimum_required(VERSION 3.20)
project(hrx_archive_probe NONE)
if(NOT DEFINED ARCHIVE_ROOT)
  message(FATAL_ERROR "ARCHIVE_ROOT is required")
endif()
find_package(hrx CONFIG REQUIRED)
if(NOT TARGET hrx::hrx)
  message(FATAL_ERROR "hrx::hrx was not defined")
endif()
get_filename_component(_root "${ARCHIVE_ROOT}" ABSOLUTE)
string(REPLACE "\\" "/" _root "${_root}")
if(_root MATCHES "/$")
  string(REGEX REPLACE "/$" "" _root "${_root}")
endif()
set(_library_paths "")
get_target_property(_configs hrx::hrx IMPORTED_CONFIGURATIONS)
if(NOT _configs)
  set(_configs "")
endif()
foreach(_cfg ${_configs})
  string(TOUPPER "${_cfg}" _CFG)
  foreach(_prop IMPORTED_IMPLIB_${_CFG} IMPORTED_LOCATION_${_CFG})
    get_target_property(_path hrx::hrx "${_prop}")
    if(_path AND NOT _path MATCHES "-NOTFOUND$")
      list(APPEND _library_paths "${_path}")
    endif()
  endforeach()
endforeach()
if(NOT _library_paths)
  message(FATAL_ERROR "hrx::hrx has no imported library path")
endif()
foreach(_path ${_library_paths})
  get_filename_component(_path "${_path}" ABSOLUTE)
  string(REPLACE "\\" "/" _path "${_path}")
  string(FIND "${_path}" "${_root}/" _idx)
  if(NOT _idx EQUAL 0)
    message(FATAL_ERROR "HRX library is outside the archive: ${_path}")
  endif()
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR "HRX library does not exist: ${_path}")
  endif()
  message(STATUS "HRX library ${_path}")
endforeach()
get_target_property(_includes hrx::hrx INTERFACE_INCLUDE_DIRECTORIES)
if(NOT _includes)
  message(FATAL_ERROR "hrx::hrx has no include directory")
endif()
set(_saw_hrx_include FALSE)
foreach(_include ${_includes})
  get_filename_component(_include "${_include}" ABSOLUTE)
  string(REPLACE "\\" "/" _include "${_include}")
  string(FIND "${_include}" "${_root}/" _idx)
  if(NOT _idx EQUAL 0)
    message(FATAL_ERROR "HRX include is outside the archive: ${_include}")
  endif()
  if(_include STREQUAL "${_root}/include/hrx")
    set(_saw_hrx_include TRUE)
  endif()
endforeach()
if(NOT _saw_hrx_include)
  message(FATAL_ERROR "hrx::hrx does not export ${_root}/include/hrx")
endif()
message(STATUS "HRX includes ${_includes}")
"""


def _one_match(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        found = ", ".join(path.name for path in paths) or "none"
        raise RuntimeError(f"expected one {description}, found: {found}")
    return paths[0]


def find_runtime_artifacts(directory: Path) -> tuple[Path, Path, list[Path]]:
    """Returns the runtime archive, runtime wheel, and FastFlowLM wheels."""
    directory = Path(directory)
    archives = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.name.startswith("fastflow-sdk-")
        and path.name.endswith((".zip", ".tar.zst"))
    )
    runtime_wheels = sorted(directory.glob("fastflow_sdk_runtime-*.whl"))
    fastflowlm_wheels = sorted(directory.glob("fastflow_sdk_fastflowlm-*.whl"))
    return (
        _one_match(archives, "standalone runtime archive"),
        _one_match(runtime_wheels, "fastflow-sdk-runtime wheel"),
        fastflowlm_wheels,
    )


def verify_hrx_prefix(prefix: Path, *, cmake: str | None = None) -> None:
    """Configures find_package(hrx) using only ``prefix``."""
    prefix = Path(prefix)
    config = prefix / "lib" / "cmake" / "hrx" / "hrx-config.cmake"
    if not config.is_file():
        raise RuntimeError(f"HRX CMake package missing: {config}")
    missing_headers = [
        name
        for name in ("hrx_runtime.h", "hrx_amdxdna.h")
        if not (prefix / "include" / "hrx" / name).is_file()
    ]
    if missing_headers:
        raise RuntimeError("HRX public headers missing: " + ", ".join(missing_headers))
    if os.name == "nt":
        for relative in ("lib/hrx.lib", "bin/hrx.dll"):
            if not (prefix / relative).is_file():
                raise RuntimeError(f"HRX link file missing: {prefix / relative}")
    elif not any((prefix / "lib").glob("libhrx.so*")):
        raise RuntimeError(f"HRX SONAME missing under {prefix / 'lib'}")

    cmake = cmake or shutil.which("cmake")
    if not cmake:
        raise RuntimeError("cmake is required to verify the HRX package")
    with tempfile.TemporaryDirectory(prefix="hrx-cmake-probe-") as temporary:
        probe = Path(temporary)
        (probe / "CMakeLists.txt").write_text(_PROBE_CMAKE, encoding="utf-8")
        build = probe / "build"
        result = subprocess.run(
            [
                cmake,
                "-S",
                str(probe),
                "-B",
                str(build),
                f"-DARCHIVE_ROOT={prefix.resolve().as_posix()}",
                f"-DCMAKE_PREFIX_PATH={prefix.resolve().as_posix()}",
            ],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        detail = "\n".join(
            line
            for line in (result.stdout + result.stderr).splitlines()
            if line.strip()
        )
        raise RuntimeError(
            "find_package(hrx) failed for the runtime archive:\n" + detail
        )


def verify_runtime_wheel(wheel: Path, *, require_runtime: bool) -> None:
    """Rejects HRX development files inside a runtime or FastFlowLM wheel."""
    wheel = Path(wheel)
    with zipfile.ZipFile(wheel) as package:
        names = [name.replace("\\", "/") for name in package.namelist()]
    development = [
        name
        for name in names
        if "/include/hrx/" in f"/{name}"
        or "/lib/cmake/hrx/" in f"/{name}"
        or name.endswith("/lib/hrx.lib")
        or name.endswith("lib/hrx.lib")
    ]
    if development:
        raise RuntimeError(
            f"{wheel.name} contains HRX development files:\n  "
            + "\n  ".join(sorted(development))
        )
    if not require_runtime:
        return
    has_fastflow = any(
        name.endswith("/bin/fastflow") or name.endswith("/bin/fastflow.exe")
        for name in names
    )
    has_hrx = any(
        name.endswith("/bin/hrx.dll") or "/libhrx.so" in f"/{name}" for name in names
    )
    if not has_fastflow or not has_hrx:
        raise RuntimeError(
            f"{wheel.name} is missing the runtime fastflow executable or HRX "
            "shared library"
        )


def _runtime_archives(directory: Path) -> list[Path]:
    return sorted(
        [
            *directory.glob("fastflow-sdk-linux-*.zip"),
            *directory.glob("fastflow-sdk-linux-*.tar.zst"),
            *directory.glob("fastflow-sdk-windows-*.zip"),
            *directory.glob("fastflow-sdk-windows-*.tar.zst"),
        ]
    )


def verify_extracted_archive(archive: Path, workspace: Path) -> None:
    """Extracts one runtime archive and configures find_package(hrx)."""
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    try:
        extracted = extract_archive(
            archive, workspace / "extracted", allowed_root=workspace.resolve()
        )
        verify_hrx_prefix(extracted)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def verify_hrx_package_split(directory: Path, wheels: dict[str, Path]) -> None:
    """Checks wheel contents, and the archive when it sits beside the wheels."""
    from build_tools.fastflow_sdk_wheel_common import (
        FASTFLOWLM_DISTRIBUTION,
        RUNTIME_DISTRIBUTION,
    )

    verify_runtime_wheel(wheels[RUNTIME_DISTRIBUTION], require_runtime=True)
    verify_runtime_wheel(wheels[FASTFLOWLM_DISTRIBUTION], require_runtime=False)
    archives = _runtime_archives(directory)
    if len(archives) > 1:
        found = ", ".join(path.name for path in archives)
        raise RuntimeError(
            f"expected one runtime archive in {directory}, found: {found}"
        )
    if not archives:
        return
    verify_extracted_archive(archives[0], directory / ".hrx-cmake-verify")
    print(f"runtime archive HRX CMake package: ok ({archives[0].name})")


def verify_artifact_dir(directory: Path) -> None:
    archive, runtime_wheel, fastflowlm_wheels = find_runtime_artifacts(directory)
    work = directory / ".hrx-cmake-verify"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    try:
        extracted = extract_archive(
            archive, work / "extracted", allowed_root=work.resolve()
        )
        verify_hrx_prefix(extracted)
        verify_runtime_wheel(runtime_wheel, require_runtime=True)
        for wheel in fastflowlm_wheels:
            verify_runtime_wheel(wheel, require_runtime=False)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"runtime archive HRX CMake package: ok ({archive.name})")
    print(f"runtime wheel excludes HRX development files: ok ({runtime_wheel.name})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifacts",
        type=Path,
        help="directory containing the standalone runtime archive and wheels",
    )
    args = parser.parse_args(argv)
    try:
        verify_artifact_dir(args.artifacts)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
