# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.rocm_npu_windows_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    WINDOWS_WHEEL_TAG,
    _copy_dependency_licenses,
    _copy_windows_runtime_closure,
    _dependency_license_packages,
    _verify_windows_products,
    _write_runtime_module,
    build_wheel,
    find_rocm_npu_wheels,
    stage_wheel_products,
    verify_wheels_do_not_have_text_path_references,
    wheel_tag,
)
from build_tools.fastflowlm_artifact import sha256_file
from build_tools.rocm_npu_wheel_common import verify_wheel_checksums


class RocmNpuWheelTest(unittest.TestCase):
    def test_find_rocm_npu_wheels_requires_one_of_each_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = build_wheel(
                self._package_tree(root, "rocm_npu_runtime"),
                RUNTIME_DISTRIBUTION,
                "0.0.0.dev1",
                root / "dist",
                summary="runtime",
                tag=WINDOWS_WHEEL_TAG,
            )
            fastflowlm = build_wheel(
                self._package_tree(root, "rocm_npu_fastflowlm"),
                FASTFLOWLM_DISTRIBUTION,
                "0.0.0.dev1",
                root / "dist",
                summary="fastflowlm",
                requires=("rocm-npu-runtime ==0.0.0.dev1",),
                tag=WINDOWS_WHEEL_TAG,
            )

            wheels = find_rocm_npu_wheels(root / "dist")
            self.assertEqual(wheels[RUNTIME_DISTRIBUTION], runtime)
            self.assertEqual(wheels[FASTFLOWLM_DISTRIBUTION], fastflowlm)
            verify_wheel_checksums(wheels)

    def test_path_reference_check_fails_on_text_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "tree" / "rocm_npu_runtime"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                f"PATH = {root!s}\n", encoding="utf-8")
            wheel = build_wheel(
                root / "tree",
                RUNTIME_DISTRIBUTION,
                "0.0.0.dev1",
                root / "dist",
                summary="runtime",
                tag=WINDOWS_WHEEL_TAG,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "wheel text payload contains local checkout/build paths",
            ):
                verify_wheels_do_not_have_text_path_references(
                    {RUNTIME_DISTRIBUTION: wheel}, (root,))

    def test_path_reference_check_audits_pe_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "tree" / "rocm_npu_runtime" / "bin"
            package.mkdir(parents=True)
            (package / "payload.dll").write_bytes(
                b"MZ" + bytes(str(root), "utf-8"))
            wheel = build_wheel(
                root / "tree",
                RUNTIME_DISTRIBUTION,
                "0.0.0.dev1",
                root / "dist",
                summary="runtime",
                tag=WINDOWS_WHEEL_TAG,
            )

            with mock.patch("builtins.print") as output:
                verify_wheels_do_not_have_text_path_references(
                    {RUNTIME_DISTRIBUTION: wheel}, (root,))
            output.assert_called_once_with(
                "wheel path audit: 1 native member(s) contain local path strings; "
                "native relocation is enforced by PE import and runtime tests"
            )

    def test_windows_wheel_tag(self):
        self.assertEqual(wheel_tag("Windows"), WINDOWS_WHEEL_TAG)
        with self.assertRaisesRegex(RuntimeError, "x86-64"):
            wheel_tag("Windows", "arm64")

    def test_dependency_license_matching_is_case_insensitive(self):
        manifest = {
            "dependencies": {
                "provider": "vcpkg",
                "runtime_file_packages": {"runtime.dll": "runtime-package"},
            }
        }
        self.assertEqual(
            _dependency_license_packages(manifest, {"Runtime.DLL"}),
            ({"runtime-package"}, set()),
        )

    def test_missing_dependency_license_is_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "dependencies": {
                    "provider": "vcpkg",
                    "runtime_file_packages": {"runtime.dll": "runtime-package"},
                }
            }
            with self.assertRaisesRegex(
                RuntimeError, "runtime-package.*missing license material"
            ):
                _copy_dependency_licenses(
                    root, manifest, root / "package", {"Runtime.DLL"}
                )

    def test_runtime_closure_copies_transitive_dlls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            runtime = root / RUNTIME_PACKAGE
            self._write_pe(source / "bin" / "first.dll")
            self._write_pe(source / "bin" / "second.dll")
            self._write_pe(runtime / "bin" / "hrx.dll")
            imports = {
                "hrx.dll": ["first.dll"],
                "first.dll": ["second.dll"],
                "second.dll": ["kernel32.dll"],
            }
            with mock.patch(
                "build_tools.rocm_npu_windows_wheels.pe_imports",
                side_effect=lambda path: imports[Path(path).name],
            ), mock.patch(
                "build_tools.rocm_npu_windows_wheels._is_windows_system_runtime",
                side_effect=lambda name: name == "kernel32.dll",
            ):
                _copy_windows_runtime_closure(source, runtime)
            self.assertTrue((runtime / "bin" / "first.dll").is_file())
            self.assertTrue((runtime / "bin" / "second.dll").is_file())

    @unittest.skipUnless(os.name == "nt", "staging requires a Windows host")
    def test_stage_products_partitions_payload_and_dependency_licenses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "bundle"
            self._write_pe(source / "bin" / "hrx.dll")
            self._write_pe(source / "bin" / "Runtime.DLL")
            self._write_pe(source / "bin" / "flm.exe")
            (source / "share" / "flm" / "xclbins").mkdir(parents=True)
            (source / "share" / "flm" / "model_list.json").write_text(
                "{}", encoding="utf-8"
            )
            (source / "share" / "flm" / "xclbins" / "kernel.xclbin").write_bytes(
                b"xclbin"
            )
            licenses = {
                "licenses/rocm-npu-staging-LICENSE": "staging",
                "licenses/hrx-LICENSE/LICENSE": "hrx",
                "licenses/FastFlowLM-LICENSE_RUNTIME.txt": "fastflowlm",
                "licenses/FastFlowLM-TERMS.md": "terms",
                "licenses/vcpkg/runtime-package/copyright": "dependency",
            }
            for relative, contents in licenses.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "dependencies": {
                    "provider": "vcpkg",
                    "runtime_file_packages": {
                        "runtime.dll": "runtime-package"
                    },
                },
                "files": [
                    {
                        "path": path.relative_to(source).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in sorted(
                        item for item in source.rglob("*") if item.is_file()
                    )
                ],
            }
            (source / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            imports = {
                "hrx.dll": ["runtime.dll"],
                "runtime.dll": ["kernel32.dll"],
                "flm.exe": ["runtime.dll"],
            }
            with mock.patch(
                "build_tools.rocm_npu_windows_wheels.pe_imports",
                side_effect=lambda path: imports[Path(path).name.casefold()],
            ), mock.patch(
                "build_tools.rocm_npu_windows_wheels._is_windows_system_runtime",
                side_effect=lambda name: name == "kernel32.dll",
            ):
                products = stage_wheel_products(source, root / "stage", "1.2.3")

            runtime = products[RUNTIME_DISTRIBUTION] / RUNTIME_PACKAGE
            fastflowlm = (
                products[FASTFLOWLM_DISTRIBUTION] / FASTFLOWLM_PACKAGE
            )
            self.assertTrue((runtime / "bin" / "Runtime.DLL").is_file())
            self.assertFalse((fastflowlm / "bin" / "Runtime.DLL").exists())
            self.assertTrue(
                (
                    runtime
                    / "licenses"
                    / "vcpkg"
                    / "runtime-package"
                    / "copyright"
                ).is_file()
            )
            self.assertTrue((fastflowlm / "bin" / "flm.exe").is_file())
            self.assertTrue(
                (fastflowlm / "share" / "flm" / "xclbins" / "kernel.xclbin").is_file()
            )

    @unittest.skipUnless(os.name == "nt", "Windows DLL APIs require Windows")
    def test_generated_runtime_keeps_dll_directory_and_library_alive(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / RUNTIME_PACKAGE
            (package / "bin").mkdir(parents=True)
            (package / "bin" / "hrx.dll").write_bytes(b"MZ")
            _write_runtime_module(package)
            spec = importlib.util.spec_from_file_location(
                "test_generated_rocm_npu_runtime", package / "__init__.py"
            )
            module = importlib.util.module_from_spec(spec)
            handle = object()
            library = object()
            with mock.patch("os.add_dll_directory", return_value=handle) as add_dir, \
                 mock.patch("ctypes.WinDLL", return_value=library) as load_library:
                assert spec.loader is not None
                spec.loader.exec_module(module)
                self.assertIs(module.initialize_process(), library)
                self.assertIs(module.initialize_process(), library)
            add_dir.assert_called_once_with(str(package / "bin"))
            load_library.assert_called_once_with(str(package / "bin" / "hrx.dll"))

    def test_windows_fastflowlm_product_can_depend_on_runtime_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_root = root / RUNTIME_PACKAGE
            fastflowlm_root = root / FASTFLOWLM_PACKAGE
            self._write_pe(runtime_root / "bin" / "hrx.dll")
            self._write_pe(fastflowlm_root / "bin" / "flm.exe")

            imports = {
                "hrx.dll": ["kernel32.dll"],
                "flm.exe": ["hrx.dll"],
            }
            with mock.patch(
                "build_tools.rocm_npu_windows_wheels.pe_imports",
                side_effect=lambda path: imports[Path(path).name],
            ), mock.patch(
                "build_tools.rocm_npu_windows_wheels._is_windows_system_runtime",
                side_effect=lambda name: name == "kernel32.dll",
            ):
                graph = _verify_windows_products(
                    [runtime_root, fastflowlm_root])

            self.assertEqual(
                graph[f"{FASTFLOWLM_PACKAGE}/bin/flm.exe"],
                ["hrx.dll"],
            )

    def test_windows_dependency_closure_is_checked_per_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_root = root / RUNTIME_PACKAGE
            fastflowlm_root = root / FASTFLOWLM_PACKAGE
            self._write_pe(runtime_root / "bin" / "hrx.dll")
            self._write_pe(fastflowlm_root / "bin" / "private.dll")

            with mock.patch(
                "build_tools.rocm_npu_windows_wheels.pe_imports",
                side_effect=lambda path: (
                    ["private.dll"] if Path(path).name == "hrx.dll" else []
                ),
            ), mock.patch(
                "build_tools.rocm_npu_windows_wheels._is_windows_system_runtime",
                return_value=False,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "rocm_npu_runtime/bin/hrx.dll -> private.dll",
                ):
                    _verify_windows_products([runtime_root, fastflowlm_root])

    def _package_tree(self, root, name):
        tree = root / f"tree-{name}"
        package = tree / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        return tree

    def _write_pe(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"MZfake")


if __name__ == "__main__":
    unittest.main()
