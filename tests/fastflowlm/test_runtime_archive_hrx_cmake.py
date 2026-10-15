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


def test_runtime_wheel_keeps_run_files_and_drops_hrx_development_files(tmp_path):
    wheel = tmp_path / "fastflow_sdk_runtime-1.2.3-py3-none-any.whl"
    fastflow = "fastflow.exe" if os.name == "nt" else "fastflow"
    library = "bin/hrx.dll" if os.name == "nt" else "lib/libhrx.so.0"
    with zipfile.ZipFile(wheel, "w") as package:
        package.writestr(f"fastflow_sdk_runtime/bin/{fastflow}", b"cli")
        package.writestr(f"fastflow_sdk_runtime/{library}", b"hrx")

    verify_runtime_wheel(wheel, require_runtime=True)

    contaminated = tmp_path / "fastflow_sdk_runtime-dev.whl"
    with zipfile.ZipFile(contaminated, "w") as package:
        package.writestr(f"fastflow_sdk_runtime/bin/{fastflow}", b"cli")
        package.writestr(f"fastflow_sdk_runtime/{library}", b"hrx")
        package.writestr(
            "fastflow_sdk_runtime/lib/cmake/hrx/hrx-config.cmake", b"config"
        )
        package.writestr("fastflow_sdk_runtime/include/hrx/hrx_runtime.h", b"header")
    with pytest.raises(RuntimeError, match="HRX development files"):
        verify_runtime_wheel(contaminated, require_runtime=True)
