# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import email.parser
import json
import re
import subprocess
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from build_tools.compiler_native import (
    linux_system_runtime,
    relocate_linux_compiler_rpaths,
    windows_system_runtime,
)
from build_tools.compiler_wheels import verify_compiler_wheel_source_revision
from build_tools.native_dependencies import is_loadable_elf
from build_tools.release_policy import (
    compiler_python_stable_abi,
    compiler_python_versions,
    compiler_requires_python,
)
from build_tools.rocm_npu_compiler_wheels import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    _is_core_aie_native,
    binding_wheel_tag,
    build_compiler_wheel_family,
    compiler_core_shared_library_names,
    compiler_product_report,
    stage_compiler_bindings,
    stage_compiler_core,
    verify_compiler_binding_stable_abi,
    verify_compiler_wheel_path_references,
    verify_windows_compiler_wheels,
)
from build_tools.rocm_npu_wheel_release import _copy_release_wheel

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


class TestRocmNpuCompilerWheels:
    @staticmethod
    def requirement_name(requirement: str) -> str:
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if not match:
            raise AssertionError(f"cannot parse requirement: {requirement!r}")
        return re.sub(r"[-_.]+", "-", match.group(0)).lower()

    def test_binding_runtime_requirements_match_upstream_contract(self):
        requirements_path = (
            Path(__file__).parents[2]
            / "projects"
            / "iron-tools"
            / "python"
            / "requirements.txt"
        )
        if not requirements_path.is_file():
            pytest.skip("iron-tools source is not checked out for this CI profile")
        upstream = {
            self.requirement_name(line)
            for raw_line in requirements_path.read_text(encoding="utf-8").splitlines()
            if (line := raw_line.split("#", 1)[0].strip()) and not line.startswith("-")
        }
        upstream.remove("eudsl-python-extras")
        declared = {
            self.requirement_name(requirement)
            for requirement in COMPILER_BINDINGS_RUNTIME_REQUIREMENTS
        }
        assert declared == upstream

    def make_product(self, root: Path) -> Path:
        product = root / "rocm-npu-compiler-windows-x86_64"
        for relative, data in {
            "bin/aie-opt.exe": b"MZtool",
            "bin/aiecc.exe": b"MZtool",
            "python/aie/_mlir_libs/AIEAggregateCAPI.dll": b"MZlibrary",
            "include/aie_api/aie.hpp": b"header",
            "peano/llvm-aie/bin/clang.exe": b"MZclang",
            "third_party/aie_api/LICENSE": b"Apache\n",
            "licenses/rocm-npu-staging-LICENSE": b"MIT\n",
            "licenses/llvm-LICENSE.txt": b"Apache LLVM exception\n",
        }.items():
            path = product / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (product / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "product": "rocm-npu-compiler",
                    "artifact": product.name,
                    "sources": {
                        "rocm-npu-staging": {
                            "revision": "abc",
                            "dirty": False,
                        }
                    },
                    "dependencies": {},
                    "toolchain": {"python": "3.12"},
                    "peano": {"wheel": "llvm-aie==1"},
                    "files": [],
                }
            ),
            encoding="utf-8",
        )
        return product

    def make_python_root(self, root: Path) -> Path:
        python_root = root / "python-abi3"
        aie = python_root / "aie"
        (aie / "_mlir_libs").mkdir(parents=True)
        (aie / "__init__.py").write_text("", encoding="utf-8")
        (aie / "_mlir_libs" / "_mlir.pyd").write_bytes(b"MZbinding")
        (aie / "_mlir_libs" / "AIEAggregateCAPI.dll").write_bytes(b"MZlibrary")
        (aie / "_mlir_libs" / "MLIRPythonSupport-mlir.dll").write_bytes(b"MZsupport")
        (aie / "_mlir_libs" / "MLIRPythonSupport-mlir.lib").write_bytes(b"archive")
        return python_root

    def metadata(self, wheel: Path):
        with zipfile.ZipFile(wheel) as archive:
            name = next(
                item
                for item in archive.namelist()
                if item.endswith(".dist-info/METADATA")
            )
            return email.parser.Parser().parsestr(archive.read(name).decode("utf-8"))

    def wheel_tags(self, wheel: Path) -> tuple[str, ...]:
        with zipfile.ZipFile(wheel) as archive:
            name = next(
                item for item in archive.namelist() if item.endswith(".dist-info/WHEEL")
            )
            metadata = email.parser.Parser().parsestr(
                archive.read(name).decode("utf-8")
            )
            return tuple(metadata.get_all("Tag") or ())

    def package_manifest(self, wheel: Path) -> dict[str, object]:
        with zipfile.ZipFile(wheel) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.count("/") == 1 and name.endswith("/manifest.json")
            ]
            assert len(names) == 1
            return json.loads(archive.read(names[0]))

    def test_release_policy_declares_supported_python_minors(self):
        assert compiler_python_versions() == ("3.12", "3.13", "3.14")
        assert compiler_python_stable_abi() == "3.12"
        assert compiler_requires_python() == ">=3.12,<3.15"
        assert binding_wheel_tag("win_amd64") == "cp312-abi3-win_amd64"

    def test_requires_python_is_derived_from_supported_minors(self, tmp_path):
        policy = tmp_path / "release.toml"
        policy.write_text(
            "schema_version = 1\n"
            "[wheel_release]\n"
            'compiler_python_stable_abi = "3.12"\n'
            'compiler_python_versions = ["3.12", "3.13", "3.14", "3.15"]\n',
            encoding="utf-8",
        )
        assert compiler_requires_python(policy) == ">=3.12,<3.16"

    def test_core_shared_library_names_are_derived_from_wheel_contents(self, tmp_path):
        wheel = tmp_path / "core.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                "rocm_npu_compiler/lib/aie/libAIEAggregateCAPI.so",
                b"ELF",
            )
            archive.writestr(
                "rocm_npu_compiler/lib/libLLVM.so.23.0",
                b"ELF",
            )
            archive.writestr("rocm_npu_compiler/bin/aie-opt", b"ELF")
        assert compiler_core_shared_library_names(wheel) == frozenset(
            {"libAIEAggregateCAPI.so", "libLLVM.so.23.0"}
        )

    def test_apfloat_wrapper_is_owned_by_compiler_core(self):
        assert _is_core_aie_native(Path("libmlir_apfloat_wrappers.so.23.0git"))

    def test_zlib_is_a_manylinux_system_dependency(self):
        assert linux_system_runtime("libz.so.1")

    def test_windows_compiler_system_runtime_classification(
        self, monkeypatch, tmp_path
    ):
        system32 = tmp_path / "System32"
        system32.mkdir()
        (system32 / "kernel32.dll").write_bytes(b"MZ")
        monkeypatch.setenv("SystemRoot", str(tmp_path))

        assert windows_system_runtime("python3.dll")
        assert windows_system_runtime("api-ms-win-core-file-l1-1-0.dll")
        assert windows_system_runtime("kernel32.dll")
        assert not windows_system_runtime("python312.dll")
        assert not windows_system_runtime("private.dll")

    def test_runtime_closure_ignores_relocatable_elf_objects(self, tmp_path):
        root = tmp_path

        def write_elf(name: str, elf_type: int) -> Path:
            path = root / name
            header = bytearray(18)
            header[:4] = b"\x7fELF"
            header[5] = 1
            header[16:18] = elf_type.to_bytes(2, "little")
            path.write_bytes(header)
            return path

        assert not is_loadable_elf(write_elf("object.o", 1))
        assert is_loadable_elf(write_elf("executable", 2))
        assert is_loadable_elf(write_elf("library.so", 3))

    def test_linux_compiler_rpaths_are_relocated_into_product(self, tmp_path):
        product = tmp_path / "product"
        executable = product / "bin" / "aie-opt"
        executable.parent.mkdir(parents=True)
        (product / "lib" / "aie").mkdir(parents=True)
        header = bytearray(18)
        header[:4] = b"\x7fELF"
        header[5] = 1
        header[16:18] = (2).to_bytes(2, "little")
        executable.write_bytes(header)
        bootgen = product / "bin" / "bootgen"
        bootgen.write_bytes(header)
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if command[1] == "--print-rpath":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "/checkout/build/llvm/lib:$ORIGIN/local\n"
                        if Path(command[-1]).name == "aie-opt"
                        else ""
                    ),
                )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch(
                "build_tools.compiler_native.shutil.which",
                return_value="/usr/bin/patchelf",
            ),
            mock.patch(
                "build_tools.native_dependencies.subprocess.run",
                side_effect=run,
            ),
        ):
            rewritten = relocate_linux_compiler_rpaths(product)

        expected = "$ORIGIN/local:$ORIGIN/../lib:$ORIGIN/../lib/aie"
        bootgen_expected = "$ORIGIN/../lib:$ORIGIN/../lib/aie"
        assert rewritten == {"bin/aie-opt": expected, "bin/bootgen": bootgen_expected}
        assert [
            "/usr/bin/patchelf",
            "--set-rpath",
            expected,
            str(executable),
        ] in commands
        assert [
            "/usr/bin/patchelf",
            "--set-rpath",
            bootgen_expected,
            str(bootgen),
        ] in commands

    def test_linux_core_relocates_after_staging_without_mutating_product(
        self, tmp_path
    ):
        root = tmp_path
        product = self.make_product(root)
        source = product / "bin" / "aie-opt.exe"
        with mock.patch(
            "build_tools.rocm_npu_compiler_wheels.relocate_linux_compiler_rpaths"
        ) as relocate:
            tree = stage_compiler_core(
                product,
                root / "stage",
                version="0.1.0.dev1",
                platform_tag="manylinux_2_28_x86_64",
            )

        package = tree / "rocm_npu_compiler"
        relocate.assert_called_once_with(package)
        staged = package / "bin" / "aie-opt.exe"
        staged.write_bytes(b"changed")
        assert source.read_bytes() == b"MZtool"

    def test_product_report_splits_python_payload(self, tmp_path):
        product = self.make_product(tmp_path)
        binding = product / "python" / "aie" / "_mlir_libs" / ("_mlir.pyd")
        binding.parent.mkdir(parents=True, exist_ok=True)
        binding.write_bytes(b"MZbinding")

        report = compiler_product_report(product)

        assert report["bindings"]["files"] == 1
        assert report["bindings"]["bytes"] == len(b"MZbinding")
        assert report["core"]["files"] > 1
        assert "python/aie/_mlir_libs/_mlir.pyd" in report["bindings"]["extensions"]

    def test_builds_meta_core_and_stable_abi_bindings(self, tmp_path):
        root = tmp_path
        product = self.make_product(root)
        python_root = self.make_python_root(root)

        wheels = build_compiler_wheel_family(
            product,
            python_root,
            root / "stage",
            root / "dist",
            version="0.1.0.dev1",
            platform_tag="win_amd64",
        )

        assert len(wheels) == 3
        verify_compiler_wheel_source_revision(wheels.values(), "abc")
        with pytest.raises(
            RuntimeError,
            match="embedded='abc', expected='wrong'",
        ):
            verify_compiler_wheel_source_revision(wheels.values(), "wrong")
        for wheel in wheels.values():
            assert self.package_manifest(wheel)["sources"] == {
                "rocm-npu-staging": {
                    "revision": "abc",
                    "dirty": False,
                }
            }
        release = root / "release"
        release.mkdir()
        for wheel in wheels.values():
            _copy_release_wheel(
                wheel,
                release,
                expected_source_sha="abc",
            )
        meta = wheels[COMPILER_DISTRIBUTION]
        core = wheels[COMPILER_CORE_DISTRIBUTION]
        assert self.wheel_tags(meta) == ("py3-none-any",)
        assert self.wheel_tags(core) == ("py3-none-win_amd64",)
        with zipfile.ZipFile(core) as archive:
            assert (
                "rocm_npu_compiler/lib/aie/AIEAggregateCAPI.dll" in archive.namelist()
            )
        assert set(self.metadata(meta).get_all("Requires-Dist") or ()) == {
            "rocm-npu-compiler-core ==0.1.0.dev1",
            "rocm-npu-compiler-bindings ==0.1.0.dev1",
        }
        wheel = wheels[COMPILER_BINDINGS_DISTRIBUTION]
        assert self.wheel_tags(wheel) == (binding_wheel_tag("win_amd64"),)
        metadata = self.metadata(wheel)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            assert not any((name.endswith("AIEAggregateCAPI.dll") for name in names))
            assert not any((name.endswith(".lib") for name in names))
        assert metadata["Requires-Python"] == ">=3.12,<3.15"
        assert set(metadata.get_all("Requires-Dist") or ()) == {
            "aiofiles ==25.1.0",
            "cloudpickle ==3.1.2",
            "ml-dtypes ==0.5.4",
            "numpy ==2.4.6",
            "rich ==15.0.0",
            "rocm-npu-compiler-core ==0.1.0.dev1",
        }
        assert (python_root / "aie" / "__init__.py").read_text(encoding="utf-8") == ""

    def test_binding_stage_rejects_cpython_specific_extension(self, tmp_path):
        root = tmp_path
        product = self.make_product(root)
        python_root = self.make_python_root(root)
        extension = python_root / "aie" / "_mlir_libs" / "_mlir.pyd"
        extension.rename(extension.with_name("_mlir.cp312-win_amd64.pyd"))
        with pytest.raises(RuntimeError, match="not stable-ABI"):
            stage_compiler_bindings(
                product,
                python_root,
                root / "stage",
                version="0.1.0.dev1",
                platform_tag="win_amd64",
            )

    def test_stable_abi_audit_is_strict_and_uses_python_312_floor(self):
        with mock.patch(
            "build_tools.rocm_npu_compiler_wheels.subprocess.run",
            return_value=mock.Mock(returncode=0, stdout="", stderr=""),
        ) as run:
            verify_compiler_binding_stable_abi(Path("bindings.whl"))

        command = run.call_args.args[0]
        assert "--strict" in command
        assert command[command.index("--assume-minimum-abi3") + 1] == "3.12"

    def test_windows_closure_allows_bindings_to_resolve_core_dll(self, tmp_path):
        root = tmp_path
        product = self.make_product(root)
        wheels = build_compiler_wheel_family(
            product,
            self.make_python_root(root),
            root / "stage",
            root / "dist",
            version="0.1.0.dev1",
            platform_tag="win_amd64",
        )
        selected = [
            wheels[COMPILER_CORE_DISTRIBUTION],
            wheels[COMPILER_BINDINGS_DISTRIBUTION],
        ]

        def imports(path: Path):
            if (
                path.name.startswith("_mlir.")
                or path.name == "MLIRPythonSupport-mlir.dll"
            ):
                return ["aieaggregatecapi.dll", "python3.dll", "kernel32.dll"]
            return ["kernel32.dll"]

        with (
            mock.patch(
                "build_tools.rocm_npu_compiler_wheels.pe_imports",
                side_effect=imports,
            ),
            mock.patch(
                "build_tools.rocm_npu_compiler_wheels.windows_system_runtime",
                side_effect=lambda name: name in ("kernel32.dll", "python3.dll"),
            ),
        ):
            graph = verify_windows_compiler_wheels(selected)
        assert any((name.endswith("_mlir.pyd") for name in graph))

    def test_windows_closure_rejects_versioned_cpython_runtime(self, tmp_path):
        root = tmp_path
        wheels = build_compiler_wheel_family(
            self.make_product(root),
            self.make_python_root(root),
            root / "stage",
            root / "dist",
            version="0.1.0.dev1",
            platform_tag="win_amd64",
        )
        selected = [
            wheels[COMPILER_CORE_DISTRIBUTION],
            wheels[COMPILER_BINDINGS_DISTRIBUTION],
        ]

        with (
            mock.patch(
                "build_tools.rocm_npu_compiler_wheels.pe_imports",
                side_effect=lambda path: (
                    ["python312.dll"]
                    if path.name == "MLIRPythonSupport-mlir.dll"
                    else ["kernel32.dll"]
                ),
            ),
            mock.patch(
                "build_tools.rocm_npu_compiler_wheels.windows_system_runtime",
                side_effect=lambda name: name == "kernel32.dll",
            ),
            pytest.raises(RuntimeError, match="versioned CPython DLL"),
        ):
            verify_windows_compiler_wheels(selected)

    def test_text_path_references_are_rejected(self, tmp_path):
        root = tmp_path
        product = self.make_product(root)
        (product / "include" / "build-path.txt").write_text(str(root), encoding="utf-8")
        wheels = build_compiler_wheel_family(
            product,
            self.make_python_root(root),
            root / "stage",
            root / "dist",
            version="0.1.0.dev1",
            platform_tag="win_amd64",
        )
        with pytest.raises(RuntimeError, match="contains local paths"):
            verify_compiler_wheel_path_references(list(wheels.values()), (root,))
