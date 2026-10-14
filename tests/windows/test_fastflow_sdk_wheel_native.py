# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

from build_tools.artifact import sha256_file
from build_tools.fastflow_sdk_windows_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    _copy_microsoft_runtime_notice,
    stage_wheel_products,
)
from build_tools.runtime_wheel_staging import _write_runtime_module

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.windows,
]


class TestFastflowSdkWheelNative:
    def test_stage_products_partitions_payload_and_dependency_licenses(self, tmp_path):
        source = tmp_path / "bundle"
        self._write_pe(source / "bin" / "hrx.dll")
        self._write_pe(source / "bin" / "hrx-amdxdna-resource-pressure.exe")
        self._write_pe(source / "bin" / "Runtime.DLL")
        self._write_pe(source / "bin" / "flm.exe")
        self._write_pe(source / "bin" / "fastflow.exe")
        fixture = source / "share" / "fastflow-sdk" / "self-test" / "case"
        fixture.mkdir(parents=True)
        (fixture / "manifest.json").write_text("{}", encoding="utf-8")
        (fixture / "kernel.xadx").write_bytes(b"xadx")
        (source / "share" / "flm" / "xclbins").mkdir(parents=True)
        (source / "share" / "flm" / "model_list.json").write_text(
            "{}", encoding="utf-8"
        )
        (source / "share" / "flm" / "xclbins" / "kernel.xclbin").write_bytes(b"xclbin")
        licenses = {
            "licenses/fastflow-sdk-LICENSE": "staging",
            "licenses/hrx-LICENSE/LICENSE": "hrx",
            "licenses/FastFlowLM-LICENSE_RUNTIME.txt": "fastflowlm",
            "licenses/FastFlowLM-TERMS.md": "terms",
            "licenses/nlohmann-json-LICENSE.MIT": "nlohmann",
            "licenses/THIRD_PARTY_NOTICES.md": "notices",
            "licenses/vcpkg/runtime-package/copyright": "dependency",
        }
        for relative, contents in licenses.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        headers = source / "include" / "hrx"
        headers.mkdir(parents=True)
        (headers / "hrx_runtime.h").write_text("runtime\n", encoding="utf-8")
        (headers / "hrx_amdxdna.h").write_text("amdxdna\n", encoding="utf-8")
        cmake = source / "lib" / "cmake" / "hrx"
        cmake.mkdir(parents=True)
        (cmake / "hrx-config.cmake").write_text("find_package\n", encoding="utf-8")
        (source / "lib" / "hrx.lib").write_bytes(b"implib")
        manifest = {
            "schema_version": 1,
            "sources": {"FastFlowLM": {"revision": "abc123"}},
            "components": [
                {
                    "name": "nlohmann/json",
                    "version": "3.12.0",
                    "upstream": "https://github.com/nlohmann/json",
                    "license": "MIT",
                }
            ],
            "toolchain": {"cmake": "cmake version 4.0"},
            "dependencies": {
                "provider": "vcpkg",
                "runtime_file_packages": {"runtime.dll": "runtime-package"},
            },
            "files": [
                {
                    "path": path.relative_to(source).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(item for item in source.rglob("*") if item.is_file())
            ],
        }
        (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        imports = {
            "hrx.dll": ["runtime.dll"],
            "hrx-amdxdna-resource-pressure.exe": ["hrx.dll"],
            "runtime.dll": ["kernel32.dll"],
            "flm.exe": ["runtime.dll"],
            "fastflow.exe": ["kernel32.dll"],
        }
        with (
            mock.patch(
                "build_tools.fastflow_sdk_windows_wheels.pe_imports",
                side_effect=lambda path: imports[Path(path).name.casefold()],
            ),
            mock.patch(
                "build_tools.fastflow_sdk_windows_wheels._is_windows_system_runtime",
                side_effect=lambda name: name == "kernel32.dll",
            ),
        ):
            products = stage_wheel_products(source, tmp_path / "stage", "1.2.3")

        runtime = products[RUNTIME_DISTRIBUTION] / RUNTIME_PACKAGE
        fastflowlm = products[FASTFLOWLM_DISTRIBUTION] / FASTFLOWLM_PACKAGE
        assert (runtime / "bin" / "Runtime.DLL").is_file()
        assert (runtime / "bin" / "hrx-amdxdna-resource-pressure.exe").is_file()
        assert (runtime / "bin" / "hrx.dll").is_file()
        assert (runtime / "bin" / "fastflow.exe").is_file()
        for package in (runtime, fastflowlm):
            assert not (package / "include" / "hrx").exists()
            assert not (package / "lib" / "cmake" / "hrx").exists()
            assert not (package / "lib" / "hrx.lib").exists()
        assert (
            runtime / "share" / "fastflow-sdk" / "self-test" / "case" / "kernel.xadx"
        ).is_file()
        assert (runtime / "licenses" / "nlohmann-json-LICENSE.MIT").is_file()
        assert not (fastflowlm / "bin" / "Runtime.DLL").exists()
        assert (
            runtime / "licenses" / "vcpkg" / "runtime-package" / "copyright"
        ).is_file()
        assert (fastflowlm / "bin" / "flm.exe").is_file()
        assert (fastflowlm / "share" / "flm" / "xclbins" / "kernel.xclbin").is_file()
        runtime_manifest = json.loads(
            (runtime / "manifest.json").read_text(encoding="utf-8")
        )
        assert runtime_manifest["dependencies"] == manifest["dependencies"]
        assert runtime_manifest["toolchain"] == manifest["toolchain"]
        assert runtime_manifest["components"] == manifest["components"]
        fastflowlm_manifest = json.loads(
            (fastflowlm / "manifest.json").read_text(encoding="utf-8")
        )
        assert fastflowlm_manifest["components"] == []

    def test_microsoft_runtime_notice_follows_shipped_runtime(self, tmp_path):
        source = tmp_path / "bundle"
        notice = source / "licenses" / "Microsoft-Visual-Cpp-Runtime-NOTICE.txt"
        notice.parent.mkdir(parents=True)
        notice.write_text("Microsoft runtime terms\n", encoding="utf-8")
        package = tmp_path / RUNTIME_PACKAGE
        self._write_pe(package / "bin" / "msvcp140.dll")
        manifest = {
            "dependencies": {
                "provider": "vcpkg",
                "microsoft_runtime_files": ["msvcp140.dll"],
            }
        }

        _copy_microsoft_runtime_notice(source, manifest, package)

        assert (
            package / "licenses" / "Microsoft-Visual-Cpp-Runtime-NOTICE.txt"
        ).read_text(encoding="utf-8") == "Microsoft runtime terms\n"

    def test_generated_runtime_keeps_dll_directory_and_library_alive(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("NPU_RUNTIME", raising=False)
        monkeypatch.delenv("IRON_RUNTIME", raising=False)
        monkeypatch.delenv("HRX_LIBHRX", raising=False)
        package = tmp_path / RUNTIME_PACKAGE
        (package / "bin").mkdir(parents=True)
        (package / "bin" / "hrx.dll").write_bytes(b"MZ")
        _write_runtime_module(package)
        spec = importlib.util.spec_from_file_location(
            "test_generated_fastflow_sdk_runtime", package / "__init__.py"
        )
        module = importlib.util.module_from_spec(spec)
        handle = object()
        library = object()
        with (
            mock.patch("os.add_dll_directory", return_value=handle) as add_dir,
            mock.patch("ctypes.WinDLL", return_value=library) as load_library,
        ):
            assert spec.loader is not None
            spec.loader.exec_module(module)
            assert module.initialize_process() is library
            assert module.initialize_process() is library
        add_dir.assert_called_once_with(str(package / "bin"))
        load_library.assert_called_once_with(str(package / "bin" / "hrx.dll"))

    def _write_pe(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"MZfake")
