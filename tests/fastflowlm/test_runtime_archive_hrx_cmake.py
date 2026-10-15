# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import shutil
import zipfile
from pathlib import Path

import pytest

from build_tools.fastflow_sdk_archive import _stage_hrx_development_files
from build_tools.verify_runtime_archive_hrx_cmake import (
    verify_hrx_prefix,
    verify_runtime_wheel,
    verify_runtime_wheel_prefix,
)

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


def _write_install(install: Path, *, library_cmake: str) -> None:
    include_dir = install / "include" / "hrx"
    include_dir.mkdir(parents=True)
    (include_dir / "hrx_runtime.h").write_text("runtime\n", encoding="utf-8")
    (include_dir / "hrx_amdxdna.h").write_text("amdxdna\n", encoding="utf-8")
    package = install / "lib" / "cmake" / "hrx"
    package.mkdir(parents=True)
    (package / "hrx-config.cmake").write_text(
        'include("${CMAKE_CURRENT_LIST_DIR}/hrx-targets.cmake")\n',
        encoding="utf-8",
    )
    (package / "hrx-targets.cmake").write_text(library_cmake, encoding="utf-8")
    if os.name == "nt":
        library = install / "lib" / "hrx.lib"
        library.parent.mkdir(parents=True, exist_ok=True)
        library.write_bytes(b"implib")


def test_packaged_hrx_config_resolves_inside_the_archive(tmp_path):
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required")
    install = tmp_path / "build" / "hrx-install"
    prefix = install.resolve().as_posix()
    if os.name == "nt":
        library_cmake = (
            "add_library(hrx::hrx SHARED IMPORTED)\n"
            "set_property(TARGET hrx::hrx PROPERTY IMPORTED_CONFIGURATIONS RELEASE)\n"
            "set_target_properties(hrx::hrx PROPERTIES\n"
            f'  IMPORTED_IMPLIB_RELEASE "{prefix}/lib/hrx.lib"\n'
            f'  IMPORTED_LOCATION_RELEASE "{prefix}/bin/hrx.dll"\n'
            f'  INTERFACE_INCLUDE_DIRECTORIES "{prefix}/include/hrx"\n'
            ")\n"
        )
    else:
        library_cmake = (
            "add_library(hrx::hrx SHARED IMPORTED)\n"
            "set_property(TARGET hrx::hrx PROPERTY IMPORTED_CONFIGURATIONS RELEASE)\n"
            "set_target_properties(hrx::hrx PROPERTIES\n"
            f'  IMPORTED_LOCATION_RELEASE "{prefix}/lib/libhrx.so"\n'
            f'  INTERFACE_INCLUDE_DIRECTORIES "{prefix}/include/hrx"\n'
            ")\n"
        )
    _write_install(install, library_cmake=library_cmake)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _stage_hrx_development_files(install, bundle)
    if os.name == "nt":
        dll = bundle / "bin" / "hrx.dll"
        dll.parent.mkdir()
        dll.write_bytes(b"MZ")
    else:
        soname = bundle / "lib" / "libhrx.so"
        soname.parent.mkdir(parents=True, exist_ok=True)
        soname.write_bytes(b"elf")

    verify_hrx_prefix(bundle)

    packaged = (bundle / "lib" / "cmake" / "hrx" / "hrx-targets.cmake").read_text(
        encoding="utf-8"
    )
    assert prefix not in packaged
    assert "${CMAKE_CURRENT_LIST_DIR}/../../../" in packaged


def _write_runtime_wheel(path: Path, prefix: Path) -> None:
    fastflow = "fastflow.exe" if os.name == "nt" else "fastflow"
    with zipfile.ZipFile(path, "w") as package:
        package.writestr(f"fastflow_sdk_runtime/bin/{fastflow}", b"cli")
        for member in sorted(item for item in prefix.rglob("*") if item.is_file()):
            package.write(
                member,
                f"fastflow_sdk_runtime/{member.relative_to(prefix).as_posix()}",
            )


def test_runtime_wheel_requires_hrx_development_files(tmp_path):
    wheel = tmp_path / "fastflow_sdk_runtime-1.2.3-py3-none-any.whl"
    fastflow = "fastflow.exe" if os.name == "nt" else "fastflow"
    library = "bin/hrx.dll" if os.name == "nt" else "lib/libhrx.so.0"
    with zipfile.ZipFile(wheel, "w") as package:
        package.writestr(f"fastflow_sdk_runtime/bin/{fastflow}", b"cli")
        package.writestr(f"fastflow_sdk_runtime/{library}", b"hrx")
    with pytest.raises(RuntimeError, match="missing HRX development files"):
        verify_runtime_wheel(wheel, require_runtime=True)

    payload = tmp_path / "fastflow_sdk_fastflowlm-1.2.3-py3-none-any.whl"
    with zipfile.ZipFile(payload, "w") as package:
        package.writestr("fastflow_sdk_fastflowlm/include/hrx/hrx_runtime.h", b"header")
    with pytest.raises(RuntimeError, match="contains HRX development files"):
        verify_runtime_wheel(payload, require_runtime=False)


def test_runtime_wheel_prefix_resolves_find_package(tmp_path):
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required")
    install = tmp_path / "build" / "hrx-install"
    prefix = install.resolve().as_posix()
    if os.name == "nt":
        library_cmake = (
            "add_library(hrx::hrx SHARED IMPORTED)\n"
            "set_property(TARGET hrx::hrx PROPERTY IMPORTED_CONFIGURATIONS RELEASE)\n"
            "set_target_properties(hrx::hrx PROPERTIES\n"
            f'  IMPORTED_IMPLIB_RELEASE "{prefix}/lib/hrx.lib"\n'
            f'  IMPORTED_LOCATION_RELEASE "{prefix}/bin/hrx.dll"\n'
            f'  INTERFACE_INCLUDE_DIRECTORIES "{prefix}/include/hrx"\n'
            ")\n"
        )
    else:
        library_cmake = (
            "add_library(hrx::hrx SHARED IMPORTED)\n"
            "set_property(TARGET hrx::hrx PROPERTY IMPORTED_CONFIGURATIONS RELEASE)\n"
            "set_target_properties(hrx::hrx PROPERTIES\n"
            f'  IMPORTED_LOCATION_RELEASE "{prefix}/lib/libhrx.so"\n'
            f'  INTERFACE_INCLUDE_DIRECTORIES "{prefix}/include/hrx"\n'
            ")\n"
        )
    _write_install(install, library_cmake=library_cmake)
    package_root = tmp_path / "package"
    package_root.mkdir()
    _stage_hrx_development_files(install, package_root)
    if os.name == "nt":
        dll = package_root / "bin" / "hrx.dll"
        dll.parent.mkdir()
        dll.write_bytes(b"MZ")
    else:
        soname = package_root / "lib" / "libhrx.so"
        soname.parent.mkdir(parents=True, exist_ok=True)
        soname.write_bytes(b"elf")
    wheel = tmp_path / "fastflow_sdk_runtime-1.2.3-py3-none-any.whl"
    _write_runtime_wheel(wheel, package_root)

    verify_runtime_wheel(wheel, require_runtime=True)
    verify_runtime_wheel_prefix(wheel)
