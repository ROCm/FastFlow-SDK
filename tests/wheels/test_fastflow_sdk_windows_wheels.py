# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from unittest import mock

import pytest

from build_tools.fastflow_sdk_wheel_common import verify_wheel_checksums
from build_tools.fastflow_sdk_windows_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    WINDOWS_WHEEL_TAG,
    _copy_windows_runtime_closure,
    _verify_windows_products,
    build_wheel,
    find_fastflow_sdk_wheels,
    verify_wheels_do_not_have_text_path_references,
    wheel_tag,
)
from build_tools.wheel_payload import (
    _dependency_license_packages,
)
from build_tools.wheel_payload import (
    copy_dependency_licenses as _copy_dependency_licenses,
)
from tests.support.wheels import make_package_tree

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


class TestFastflowSdkWheel:
    def test_find_fastflow_sdk_wheels_requires_one_of_each_product(self, tmp_path):
        root = tmp_path
        runtime = build_wheel(
            make_package_tree(root, "fastflow_sdk_runtime"),
            RUNTIME_DISTRIBUTION,
            "0.0.0.dev1",
            root / "dist",
            summary="runtime",
            tag=WINDOWS_WHEEL_TAG,
        )
        fastflowlm = build_wheel(
            make_package_tree(root, "fastflow_sdk_fastflowlm"),
            FASTFLOWLM_DISTRIBUTION,
            "0.0.0.dev1",
            root / "dist",
            summary="fastflowlm",
            requires=("fastflow-sdk-runtime ==0.0.0.dev1",),
            tag=WINDOWS_WHEEL_TAG,
        )

        wheels = find_fastflow_sdk_wheels(root / "dist")
        assert wheels[RUNTIME_DISTRIBUTION] == runtime
        assert wheels[FASTFLOWLM_DISTRIBUTION] == fastflowlm
        verify_wheel_checksums(wheels)

    def test_path_reference_check_fails_on_text_payload(self, tmp_path):
        root = tmp_path
        package = root / "tree" / "fastflow_sdk_runtime"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"PATH = {root!s}\n", encoding="utf-8")
        wheel = build_wheel(
            root / "tree",
            RUNTIME_DISTRIBUTION,
            "0.0.0.dev1",
            root / "dist",
            summary="runtime",
            tag=WINDOWS_WHEEL_TAG,
        )

        with pytest.raises(
            RuntimeError, match="wheel text payload contains local checkout/build paths"
        ):
            verify_wheels_do_not_have_text_path_references(
                {RUNTIME_DISTRIBUTION: wheel}, (root,)
            )

    def test_path_reference_check_audits_pe_payload(self, tmp_path):
        root = tmp_path
        package = root / "tree" / "fastflow_sdk_runtime" / "bin"
        package.mkdir(parents=True)
        (package / "payload.dll").write_bytes(b"MZ" + bytes(str(root), "utf-8"))
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
                {RUNTIME_DISTRIBUTION: wheel}, (root,)
            )
        output.assert_called_once_with(
            "wheel path audit: 1 native member(s) contain local path strings; "
            "native relocation is enforced by PE import and runtime tests"
        )

    def test_windows_wheel_tag(self):
        assert wheel_tag("Windows") == WINDOWS_WHEEL_TAG
        with pytest.raises(RuntimeError, match="x86-64"):
            wheel_tag("Windows", "arm64")

    def test_dependency_license_matching_is_case_insensitive(self):
        manifest = {
            "dependencies": {
                "provider": "vcpkg",
                "runtime_file_packages": {"runtime.dll": "runtime-package"},
            }
        }
        assert _dependency_license_packages(manifest, {"Runtime.DLL"}) == (
            {"runtime-package"},
            set(),
        )

    def test_missing_dependency_license_is_fatal(self, tmp_path):
        root = tmp_path
        manifest = {
            "dependencies": {
                "provider": "vcpkg",
                "runtime_file_packages": {"runtime.dll": "runtime-package"},
            }
        }
        with pytest.raises(
            RuntimeError, match="runtime-package.*missing license material"
        ):
            _copy_dependency_licenses(root, manifest, root / "package", {"Runtime.DLL"})

    def test_runtime_closure_copies_transitive_dlls(self, tmp_path):
        root = tmp_path
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
        with (
            mock.patch(
                "build_tools.fastflow_sdk_windows_wheels.pe_imports",
                side_effect=lambda path: imports[Path(path).name],
            ),
            mock.patch(
                "build_tools.fastflow_sdk_windows_wheels._is_windows_system_runtime",
                side_effect=lambda name: name == "kernel32.dll",
            ),
        ):
            _copy_windows_runtime_closure(source, runtime)
        assert (runtime / "bin" / "first.dll").is_file()
        assert (runtime / "bin" / "second.dll").is_file()

    def test_windows_fastflowlm_product_can_depend_on_runtime_product(self, tmp_path):
        root = tmp_path
        runtime_root = root / RUNTIME_PACKAGE
        fastflowlm_root = root / FASTFLOWLM_PACKAGE
        self._write_pe(runtime_root / "bin" / "hrx.dll")
        self._write_pe(fastflowlm_root / "bin" / "flm.exe")

        imports = {
            "hrx.dll": ["kernel32.dll"],
            "flm.exe": ["hrx.dll"],
        }
        with (
            mock.patch(
                "build_tools.fastflow_sdk_windows_wheels.pe_imports",
                side_effect=lambda path: imports[Path(path).name],
            ),
            mock.patch(
                "build_tools.fastflow_sdk_windows_wheels._is_windows_system_runtime",
                side_effect=lambda name: name == "kernel32.dll",
            ),
        ):
            graph = _verify_windows_products([runtime_root, fastflowlm_root])

        assert graph[f"{FASTFLOWLM_PACKAGE}/bin/flm.exe"] == ["hrx.dll"]

    def test_windows_dependency_closure_is_checked_per_product(self, tmp_path):
        root = tmp_path
        runtime_root = root / RUNTIME_PACKAGE
        fastflowlm_root = root / FASTFLOWLM_PACKAGE
        self._write_pe(runtime_root / "bin" / "hrx.dll")
        self._write_pe(fastflowlm_root / "bin" / "private.dll")

        with (
            mock.patch(
                "build_tools.fastflow_sdk_windows_wheels.pe_imports",
                side_effect=lambda path: (
                    ["private.dll"] if Path(path).name == "hrx.dll" else []
                ),
            ),
            mock.patch(
                "build_tools.fastflow_sdk_windows_wheels._is_windows_system_runtime",
                return_value=False,
            ),
        ):
            with pytest.raises(
                RuntimeError, match="fastflow_sdk_runtime/bin/hrx.dll -> private.dll"
            ):
                _verify_windows_products([runtime_root, fastflowlm_root])

    def _write_pe(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"MZfake")
