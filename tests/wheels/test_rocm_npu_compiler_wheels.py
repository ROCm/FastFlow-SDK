# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import email.parser
import json
import re
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from build_tools.rocm_npu_compiler_wheels import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    _is_core_aie_native,
    _is_loadable_elf,
    _linux_system_runtime,
    binding_wheel_tag,
    build_compiler_wheel_family,
    compiler_core_shared_library_names,
    compiler_product_report,
    compiler_requires_python,
    compiler_python_stable_abi,
    compiler_python_versions,
    relocate_linux_compiler_rpaths,
    stage_compiler_bindings,
    stage_compiler_core,
    verify_compiler_binding_stable_abi,
    verify_compiler_wheel_path_references,
    verify_windows_compiler_wheels,
)


class RocmNpuCompilerWheelsTest(unittest.TestCase):
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
            self.skipTest("iron-tools source is not checked out for this CI profile")
        upstream = {
            self.requirement_name(line)
            for raw_line in requirements_path.read_text(encoding="utf-8").splitlines()
            if (line := raw_line.split("#", 1)[0].strip())
            and not line.startswith("-")
        }
        upstream.remove("eudsl-python-extras")
        declared = {
            self.requirement_name(requirement)
            for requirement in COMPILER_BINDINGS_RUNTIME_REQUIREMENTS
        }
        self.assertEqual(declared, upstream)

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
        (aie / "_mlir_libs" / "_mlir.pyd").write_bytes(
            b"MZbinding"
        )
        (aie / "_mlir_libs" / "AIEAggregateCAPI.dll").write_bytes(b"MZlibrary")
        (aie / "_mlir_libs" / "MLIRPythonSupport-mlir.dll").write_bytes(
            b"MZsupport"
        )
        (aie / "_mlir_libs" / "MLIRPythonSupport-mlir.lib").write_bytes(b"archive")
        return python_root

    def metadata(self, wheel: Path):
        with zipfile.ZipFile(wheel) as archive:
            name = next(
                item for item in archive.namelist()
                if item.endswith(".dist-info/METADATA")
            )
            return email.parser.Parser().parsestr(
                archive.read(name).decode("utf-8")
            )

    def wheel_tags(self, wheel: Path) -> tuple[str, ...]:
        with zipfile.ZipFile(wheel) as archive:
            name = next(
                item for item in archive.namelist()
                if item.endswith(".dist-info/WHEEL")
            )
            metadata = email.parser.Parser().parsestr(
                archive.read(name).decode("utf-8")
            )
            return tuple(metadata.get_all("Tag") or ())

    def test_release_policy_declares_supported_python_minors(self):
        self.assertEqual(compiler_python_versions(), ("3.12", "3.13", "3.14"))
        self.assertEqual(compiler_python_stable_abi(), "3.12")
        self.assertEqual(compiler_requires_python(), ">=3.12,<3.15")
        self.assertEqual(
            binding_wheel_tag("win_amd64"),
            "cp312-abi3-win_amd64",
        )

    def test_requires_python_is_derived_from_supported_minors(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = Path(temporary) / "release.toml"
            policy.write_text(
                "schema_version = 1\n"
                "[wheel_release]\n"
                'compiler_python_stable_abi = "3.12"\n'
                'compiler_python_versions = ["3.12", "3.13", "3.14", "3.15"]\n',
                encoding="utf-8",
            )
            self.assertEqual(
                compiler_requires_python(policy),
                ">=3.12,<3.16",
            )

    def test_core_shared_library_names_are_derived_from_wheel_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "core.whl"
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
            self.assertEqual(
                compiler_core_shared_library_names(wheel),
                frozenset({"libAIEAggregateCAPI.so", "libLLVM.so.23.0"}),
            )

    def test_apfloat_wrapper_is_owned_by_compiler_core(self):
        self.assertTrue(
            _is_core_aie_native(Path("libmlir_apfloat_wrappers.so.23.0git"))
        )

    def test_zlib_is_a_manylinux_system_dependency(self):
        self.assertTrue(_linux_system_runtime("libz.so.1"))

    def test_runtime_closure_ignores_relocatable_elf_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_elf(name: str, elf_type: int) -> Path:
                path = root / name
                header = bytearray(18)
                header[:4] = b"\x7fELF"
                header[5] = 1
                header[16:18] = elf_type.to_bytes(2, "little")
                path.write_bytes(header)
                return path

            self.assertFalse(_is_loadable_elf(write_elf("object.o", 1)))
            self.assertTrue(_is_loadable_elf(write_elf("executable", 2)))
            self.assertTrue(_is_loadable_elf(write_elf("library.so", 3)))

    def test_linux_compiler_rpaths_are_relocated_into_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary) / "product"
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
                    "build_tools.rocm_npu_compiler_wheels.shutil.which",
                    return_value="/usr/bin/patchelf",
                ),
                mock.patch(
                    "build_tools.rocm_npu_compiler_wheels.subprocess.run",
                    side_effect=run,
                ),
            ):
                rewritten = relocate_linux_compiler_rpaths(product)

            expected = "$ORIGIN/local:$ORIGIN/../lib:$ORIGIN/../lib/aie"
            bootgen_expected = "$ORIGIN/../lib:$ORIGIN/../lib/aie"
            self.assertEqual(
                rewritten,
                {
                    "bin/aie-opt": expected,
                    "bin/bootgen": bootgen_expected,
                },
            )
            self.assertIn(
                ["/usr/bin/patchelf", "--set-rpath", expected, str(executable)],
                commands,
            )
            self.assertIn(
                [
                    "/usr/bin/patchelf",
                    "--set-rpath",
                    bootgen_expected,
                    str(bootgen),
                ],
                commands,
            )

    def test_linux_core_relocates_after_staging_without_mutating_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = self.make_product(root)
            source = product / "bin" / "aie-opt.exe"
            with mock.patch(
                "build_tools.rocm_npu_compiler_wheels."
                "relocate_linux_compiler_rpaths"
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
            self.assertEqual(source.read_bytes(), b"MZtool")

    def test_product_report_splits_python_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = self.make_product(Path(temporary))
            binding = product / "python" / "aie" / "_mlir_libs" / (
                "_mlir.pyd"
            )
            binding.parent.mkdir(parents=True, exist_ok=True)
            binding.write_bytes(b"MZbinding")

            report = compiler_product_report(product)

            self.assertEqual(report["bindings"]["files"], 1)
            self.assertEqual(report["bindings"]["bytes"], len(b"MZbinding"))
            self.assertGreater(report["core"]["files"], 1)
            self.assertIn(
                "python/aie/_mlir_libs/_mlir.pyd",
                report["bindings"]["extensions"],
            )

    def test_builds_meta_core_and_stable_abi_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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

            self.assertEqual(len(wheels), 3)
            meta = wheels[COMPILER_DISTRIBUTION]
            core = wheels[COMPILER_CORE_DISTRIBUTION]
            self.assertEqual(self.wheel_tags(meta), ("py3-none-any",))
            self.assertEqual(self.wheel_tags(core), ("py3-none-win_amd64",))
            with zipfile.ZipFile(core) as archive:
                self.assertIn(
                    "rocm_npu_compiler/lib/aie/AIEAggregateCAPI.dll",
                    archive.namelist(),
                )
            self.assertEqual(
                set(self.metadata(meta).get_all("Requires-Dist") or ()),
                {
                    "rocm-npu-compiler-core ==0.1.0.dev1",
                    "rocm-npu-compiler-bindings ==0.1.0.dev1",
                },
            )
            wheel = wheels[COMPILER_BINDINGS_DISTRIBUTION]
            self.assertEqual(
                self.wheel_tags(wheel),
                (binding_wheel_tag("win_amd64"),),
            )
            metadata = self.metadata(wheel)
            with zipfile.ZipFile(wheel) as archive:
                names = archive.namelist()
                self.assertFalse(
                    any(name.endswith("AIEAggregateCAPI.dll") for name in names)
                )
                self.assertFalse(any(name.endswith(".lib") for name in names))
            self.assertEqual(metadata["Requires-Python"], ">=3.12,<3.15")
            self.assertEqual(
                set(metadata.get_all("Requires-Dist") or ()),
                {
                    "aiofiles ==25.1.0",
                    "cloudpickle ==3.1.2",
                    "ml-dtypes ==0.5.4",
                    "numpy ==2.4.6",
                    "rich ==15.0.0",
                    "rocm-npu-compiler-core ==0.1.0.dev1",
                },
            )
            self.assertEqual(
                (python_root / "aie" / "__init__.py").read_text(encoding="utf-8"),
                "",
            )

    def test_binding_stage_rejects_cpython_specific_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = self.make_product(root)
            python_root = self.make_python_root(root)
            extension = python_root / "aie" / "_mlir_libs" / "_mlir.pyd"
            extension.rename(extension.with_name("_mlir.cp312-win_amd64.pyd"))
            with self.assertRaisesRegex(RuntimeError, "not stable-ABI"):
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
        self.assertIn("--strict", command)
        self.assertEqual(
            command[command.index("--assume-minimum-abi3") + 1],
            "3.12",
        )

    def test_windows_closure_allows_bindings_to_resolve_core_dll(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
                    "build_tools.rocm_npu_compiler_wheels._windows_system_runtime",
                    side_effect=lambda name: name in ("kernel32.dll", "python3.dll"),
                ),
            ):
                graph = verify_windows_compiler_wheels(selected)
            self.assertTrue(
                any(name.endswith("_mlir.pyd") for name in graph)
            )

    def test_windows_closure_rejects_versioned_cpython_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
                    "build_tools.rocm_npu_compiler_wheels._windows_system_runtime",
                    side_effect=lambda name: name == "kernel32.dll",
                ),
                self.assertRaisesRegex(RuntimeError, "versioned CPython DLL"),
            ):
                verify_windows_compiler_wheels(selected)

    def test_text_path_references_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = self.make_product(root)
            (product / "include" / "build-path.txt").write_text(
                str(root), encoding="utf-8"
            )
            wheels = build_compiler_wheel_family(
                product,
                self.make_python_root(root),
                root / "stage",
                root / "dist",
                version="0.1.0.dev1",
                platform_tag="win_amd64",
            )
            with self.assertRaisesRegex(RuntimeError, "contains local paths"):
                verify_compiler_wheel_path_references(list(wheels.values()), (root,))


if __name__ == "__main__":
    unittest.main()
