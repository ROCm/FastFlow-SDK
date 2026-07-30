# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "tests" / "e2e"),
    str(ROOT / "tests" / "compiler"),
]

import dev  # noqa: E402
import run_e2e  # noqa: E402
import verify_compiler_bindings  # noqa: E402


class CompilerProductTest(unittest.TestCase):
    def write_file(self, path: Path, data: bytes = b"x", executable: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if executable and os.name != "nt":
            path.chmod(0o755)

    def make_product(
        self,
        root: Path,
        artifact: str = "rocm-npu-compiler-linux-x86_64",
        include_build_metadata: bool = False,
    ) -> Path:
        product = root / artifact
        required_tools = []
        for name in [*dev.NPU_PREREQ_TOOLS, "xclbinutil"]:
            tool = f"{name}{dev._EXE}"
            self.write_file(product / "bin" / tool, executable=True)
            required_tools.append(f"bin/{tool}")
        for relative in (
            "python/aie/ir.py",
            "python/aie/compiler/aiecc/main.py",
            "include/aie_api/aie.hpp",
            "aie_runtime_lib/AIE2P/vec_math.h",
            "peano/llvm-aie/README.txt",
            "third_party/aie_api/README.txt",
            "lib/.keep",
            *(f"licenses/{name}" for name in dev.COMPILER_PRODUCT_LICENSES),
        ):
            self.write_file(product / relative)
        if include_build_metadata:
            self.write_file(product / "python" / "CMakeFiles" / "target.dir" / "object.obj")

        files = []
        for path in sorted(p for p in product.rglob("*") if p.is_file()):
            files.append({"path": path.relative_to(product).as_posix(),
                          "size": path.stat().st_size,
                          "sha256": dev.sha256_file(path)})
        (product / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "product": "rocm-npu-compiler",
            "artifact": artifact,
            "required_tools": required_tools,
            "files": files,
        }), encoding="utf-8")
        return product

    def test_product_name_is_public_and_platform_normalized(self):
        self.assertEqual(
            dev._compiler_product_name("Linux", "AMD64"),
            "rocm-npu-compiler-linux-x86_64",
        )
        self.assertNotIn("staging", dev._compiler_product_name("Windows", "x86_64"))

    def test_iron_tools_build_does_not_resolve_virtualenv_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            venv_python = root / "venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.write_bytes(b"")

            bdir = root / "iron-tools"
            install_dir = root / "iron-tools-install"
            original_resolve = Path.resolve

            def reject_python_resolve(path, *args, **kwargs):
                if path == venv_python:
                    raise AssertionError("virtualenv Python path must not be resolved")
                return original_resolve(path, *args, **kwargs)

            with (
                mock.patch.object(dev, "BUILD", root),
                mock.patch.object(dev, "_eudsl_cache_ok", return_value=True),
                mock.patch.object(dev, "_compilers", return_value=("cc", "c++")),
                mock.patch.object(dev, "_cc_flags", return_value=[]),
                mock.patch.object(dev, "_ccache_flags", return_value=[]),
                mock.patch.object(dev, "cmake_env", return_value={}),
                mock.patch.object(dev, "cmake_bin", return_value="cmake"),
                mock.patch.object(dev, "sh") as run,
                mock.patch.object(Path, "resolve", reject_python_resolve),
            ):
                dev._build_iron_tools(venv_python, bdir, install_dir)

            configure_command = run.call_args_list[0].args[0]
            self.assertIn(
                f"-DPython3_EXECUTABLE={dev.cmake_path(venv_python.absolute())}",
                configure_command,
            )
            self.assertIn(
                f"-DPython_EXECUTABLE={dev.cmake_path(venv_python.absolute())}",
                configure_command,
            )
            self.assertIn("-DMLIR_ENABLE_PYTHON_STABLE_ABI=ON", configure_command)
            self.assertIn("-DAIE_ENABLE_PYTHON_PASSES=OFF", configure_command)
            self.assertIn("-DAIE_ENABLE_XRT_PYTHON_BINDINGS=OFF", configure_command)

    def test_binding_conformance_discovers_abi3_extension_modules(self):
        aie_root = Path("site-packages") / "aie"
        self.assertEqual(
            verify_compiler_bindings._extension_module_name(
                aie_root,
                aie_root / "_mlir_libs" / "_mlir.pyd",
            ),
            "aie._mlir_libs._mlir",
        )
        self.assertEqual(
            verify_compiler_bindings._extension_module_name(
                aie_root,
                aie_root / "_mlir_libs" / "_aie.abi3.so",
            ),
            "aie._mlir_libs._aie",
        )

    def test_wheel_verification_runs_binding_conformance_with_current_minor(self):
        with mock.patch.object(dev.subprocess, "run") as run:
            dev._verify_installed_compiler_bindings(Path("venv-python"))

        command = run.call_args.args[0]
        self.assertEqual(command[0], Path("venv-python"))
        self.assertTrue(str(command[1]).endswith("verify_compiler_bindings.py"))
        self.assertEqual(
            command[-1],
            f"{sys.version_info.major}.{sys.version_info.minor}",
        )

    def test_consumer_only_verification_skips_static_wheel_scans(self):
        args = SimpleNamespace(consumer_only=True)
        with (
            mock.patch.object(dev, "_verify_compiler_wheel_consumer") as consumer,
            mock.patch.object(dev, "_compiler_wheel_files") as static_scan,
        ):
            dev.cmd_verify_compiler_wheels(args)

        consumer.assert_called_once_with(args)
        static_scan.assert_not_called()

    def test_consumer_root_exports_only_cross_minor_compiler_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site_packages = root / "site-packages"
            self.write_file(site_packages / "aie" / "__init__.py", b"")
            self.write_file(
                site_packages / "aie" / "__pycache__" / "bad.cpython-312.pyc"
            )
            self.write_file(
                site_packages / "rocm_npu_compiler" / "__init__.py", b""
            )
            self.write_file(
                site_packages
                / "rocm_npu_compiler_core-1.0.dist-info"
                / "METADATA"
            )
            self.write_file(site_packages / "numpy" / "__init__.py", b"")
            destination = root / "build" / "consumer"
            with (
                mock.patch.object(dev, "BUILD", root / "build"),
                mock.patch.object(
                    dev, "_compiler_site_packages", return_value=site_packages
                ),
            ):
                dev._export_compiler_consumer_root(
                    Path("python"), destination)

            self.assertTrue((destination / "aie" / "__init__.py").is_file())
            self.assertTrue(
                (destination / "rocm_npu_compiler" / "__init__.py").is_file()
            )
            self.assertFalse((destination / "aie" / "__pycache__").exists())
            self.assertFalse((destination / "numpy").exists())

    def test_compiler_site_packages_uses_platform_purelib(self):
        self.assertEqual(
            dev._compiler_site_packages(Path(sys.executable)),
            Path(sysconfig.get_path("purelib")),
        )

    def test_latest_minor_consumer_can_run_one_e2e(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            consumer_root = root / "consumer"
            self.write_file(consumer_root / "aie" / "__init__.py", b"")
            self.write_file(
                consumer_root / "rocm_npu_compiler" / "__init__.py", b""
            )
            args = SimpleNamespace(
                consumer_root=str(consumer_root),
                consumer_e2e=True,
                venv=str(root / "build" / "venv"),
            )
            with (
                mock.patch.object(dev, "BUILD", root / "build"),
                mock.patch.object(dev, "_python_install"),
                mock.patch.object(dev, "_verify_installed_compiler_bindings"),
                mock.patch.object(dev.subprocess, "run") as run,
            ):
                dev._verify_compiler_wheel_consumer(args)

            e2e = run.call_args_list[-1]
            self.assertTrue(str(e2e.args[0][1]).endswith("run_e2e.py"))
            self.assertEqual(
                e2e.kwargs["env"]["ROCM_NPU_COMPILER_PREFIX"],
                str(consumer_root / "rocm_npu_compiler"),
            )
            self.assertEqual(
                e2e.kwargs["env"]["ROCM_NPU_CMAKE"],
                dev.cmake_bin(),
            )

    def test_compiler_product_can_extract_to_explicit_short_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            short_root = Path(temporary) / "rnc"
            archive = Path(temporary) / "compiler.zip"
            with (
                mock.patch.object(dev, "_find_compiler_product_archive",
                                  return_value=archive),
                mock.patch.object(
                    dev, "extract_fastflowlm_archive",
                    return_value=short_root / "rocm-npu-compiler-windows-x86_64",
                ) as extract,
            ):
                dev.cmd_unpack_compiler_product(SimpleNamespace(
                    dir=Path(temporary), output_dir=str(short_root)))

            extract.assert_called_once_with(
                archive, short_root, allowed_root=short_root.parent)

    def test_inventory_accepts_complete_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = self.make_product(Path(temporary))
            manifest = dev._verify_compiler_product_inventory(product)
            self.assertEqual(manifest["product"], "rocm-npu-compiler")

    def test_inventory_rejects_staging_artifact_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = self.make_product(
                Path(temporary), "rocm-npu-staging-compiler-linux-x86_64")
            with self.assertRaisesRegex(RuntimeError, "must not include staging"):
                dev._verify_compiler_product_inventory(product)

    def test_inventory_rejects_build_system_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = self.make_product(
                Path(temporary), include_build_metadata=True)
            with self.assertRaisesRegex(RuntimeError, "build-system metadata"):
                dev._verify_compiler_product_inventory(product)

    def test_runtime_staging_excludes_build_intermediates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            self.write_file(source / "lib" / "runtime.lib")
            self.write_file(source / "lib" / "crt0.o")
            self.write_file(source / "include" / "runtime.h")
            self.write_file(source / "CMakeFiles" / "target.dir" / "runtime.obj")
            self.write_file(
                source / "lib" / "objects-Release" / "obj.AIECAPI" / "Dialects.cpp.o"
            )
            self.write_file(source / "lib" / "stray.pdb")
            self.write_file(source / "__pycache__" / "module.cpython-312.pyc")
            self.write_file(
                source
                / "runtime_lib"
                / "xaiengineTmp"
                / "x86_64"
                / "tmp"
                / "xaiengine-cache-Release.cmake"
            )

            destination = root / "destination"
            dev._stage_runtime_tree(source, destination)

            self.assertTrue((destination / "lib" / "runtime.lib").is_file())
            self.assertTrue((destination / "lib" / "crt0.o").is_file())
            self.assertTrue((destination / "include" / "runtime.h").is_file())
            self.assertFalse((destination / "CMakeFiles").exists())
            self.assertFalse((destination / "lib" / "objects-Release").exists())
            self.assertFalse((destination / "lib" / "stray.pdb").exists())
            self.assertFalse((destination / "__pycache__").exists())
            self.assertFalse(
                (destination / "runtime_lib" / "xaiengineTmp").exists()
            )

    def test_ci_runs_one_deep_doctor_and_uses_content_addressed_caches(self):
        workflows = ROOT / ".github" / "workflows"
        linux = (workflows / "build_rocm_npu_staging_linux.yml").read_text()
        windows = (workflows / "build_rocm_npu_staging_windows.yml").read_text()

        self.assertIn("dev.py doctor compiler-stack --deep", linux)
        self.assertNotIn("dev.py doctor compiler-stack --deep", windows)
        self.assertNotIn(
            "key: ccache-${{ runner.os }}-${{ github.run_id }}", linux + windows
        )
        self.assertNotIn(
            "key: vcpkg-${{ runner.os }}-${{ hashFiles("
            "'.github/vcpkg/triplets/*.cmake') }}-${{ github.run_id }}",
            windows,
        )
        cache_inputs = (
            "hashFiles('manifests/versions.toml', 'patches/**/*.patch', "
            "'dev.py', 'build_tools/dev/**/*.py')"
        )
        self.assertIn(cache_inputs, linux)
        self.assertIn(cache_inputs, windows)

    def test_npu_prerequisites_exclude_separate_compiler_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hrx_install = root / "hrx-install"
            self.write_file(
                hrx_install / ("bin/hrx.dll" if os.name == "nt" else "lib/libhrx.so"))
            self.write_file(hrx_install / "include/hrx/hrx_runtime.h")
            self.write_file(hrx_install / "include/hrx/hrx_amdxdna.h")
            cts_source = root / "cts"
            for name in dev.HRX_CTS_TESTS:
                self.write_file(
                    cts_source / f"{name}{dev._EXE}", executable=True)

            stage = root / "stage"
            dev._stage_npu_runtime_prereqs(stage, hrx_install, cts_source)

            self.assertTrue(
                (stage / "build/hrx-install/include/hrx/hrx_runtime.h").is_file())
            cts_stage = stage.joinpath(
                "projects", "hrx", "build", "cmake", *dev.HRX_CTS_DIR)
            for name in dev.HRX_CTS_TESTS:
                self.assertTrue((cts_stage / f"{name}{dev._EXE}").is_file())
            self.assertFalse((stage / "build/iron-tools").exists())
            self.assertFalse((stage / "tools-bin").exists())

    def test_npu_test_package_contains_only_xadx_and_numeric_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            _, test_cases = dev._npu_test_modules()
            case_records = []
            for name in ("vector_scalar_add", "elementwise_add", "relu", "gemv"):
                case_records.append(
                    dev.npu_test_package.write_case(
                        stage,
                        test_cases.generate_case(name),
                        b"\x00\x00\x00\x00XADX",
                        {"kind": "test", "name": name},
                    )
                )

            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            system = dev.platform.system().lower()
            dev.npu_test_package.write_manifest(
                stage, case_records, revision, system, "x86_64"
            )
            manifest = dev.npu_test_package.verify_tree(
                stage,
                ("vector_scalar_add", "elementwise_add", "relu", "gemv"),
                revision,
                system,
            )

            self.assertEqual(len(manifest["cases"]), 4)
            packaged = {
                path.suffix for path in stage.rglob("*") if path.is_file()
            }
            self.assertNotIn(".xclbin", packaged)
            self.assertNotIn(".mlir", packaged)

    def test_windows_tar_zst_is_deterministic_and_extracts_valid_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = self.make_product(
                root, artifact="rocm-npu-compiler-windows-x86_64")
            with mock.patch.object(dev, "IS_WINDOWS", True):
                first = dev._create_compiler_product_archive(
                    product, root / "first")
                second = dev._create_compiler_product_archive(
                    product, root / "second")

            self.assertTrue(first.name.endswith(".tar.zst"))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertTrue(first.with_name(first.name + ".sha256").is_file())
            extracted = dev.extract_fastflowlm_archive(
                first, root / "extracted", allowed_root=root)
            self.assertEqual(
                (extracted / "python" / "aie" / "ir.py").read_bytes(), b"x")
            dev._verify_compiler_product_inventory(extracted)

    def test_product_smoke_reports_python_extension_import_failure(self):
        aieopt_result = mock.Mock(returncode=0, stderr=b"")
        import_result = mock.Mock(
            returncode=1, stderr="ImportError: DLL filename is too long")
        with (
            mock.patch.object(dev, "_verify_compiler_product_inventory",
                              return_value={"files": []}),
            mock.patch.object(dev, "_compiler_product_env", return_value={}),
            mock.patch.object(dev, "sh"),
            mock.patch.object(dev.subprocess, "run",
                              side_effect=[aieopt_result, import_result]) as run,
            self.assertRaisesRegex(RuntimeError, "DLL filename is too long"),
        ):
            dev._smoke_compiler_product(Path("compiler"))

        self.assertEqual(
            run.call_args_list[1].args[0][-1],
            "import aie.dialects.transform.interpreter")

    def test_windows_compiler_product_stages_non_system_dll_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = root / "product"
            dependency_root = root / "openssl"
            system_root = root / "windows"
            self.write_file(product / "bin" / "bootgen.exe", b"MZbootgen")
            self.write_file(
                product / "python" / "aie" / "_mlir_libs" / "nanobind-abi3-mlir.dll",
                b"MZnanobind",
            )
            self.write_file(
                product / "python" / "aie" / "_mlir_libs" / "MLIRPythonSupport-mlir.dll",
                b"MZsupport",
            )
            self.write_file(
                dependency_root / "bin" / "libcrypto-3-x64.dll",
                b"MZcrypto",
            )
            self.write_file(
                system_root / "System32" / "kernel32.dll",
                b"MZsystem",
            )
            imports = {
                "bootgen.exe": ["libcrypto-3-x64.dll"],
                "libcrypto-3-x64.dll": ["kernel32.dll"],
                "nanobind-abi3-mlir.dll": ["python3.dll"],
                "MLIRPythonSupport-mlir.dll": ["python3.dll"],
            }
            with (
                mock.patch.object(dev, "IS_WINDOWS", True),
                mock.patch.dict(
                    os.environ,
                    {
                        "OPENSSL_ROOT_DIR": str(dependency_root),
                        "SystemRoot": str(system_root),
                    },
                    clear=False,
                ),
                mock.patch.object(
                    dev,
                    "pe_imports",
                    side_effect=lambda path: imports[Path(path).name],
                ),
            ):
                copied = dev._stage_compiler_native_dependencies(product)
            self.assertTrue((product / "bin" / "libcrypto-3-x64.dll").is_file())
            self.assertIn("libcrypto-3-x64.dll", copied)
            self.assertFalse((product / "bin" / "python3.dll").exists())

    def test_windows_compiler_product_rejects_versioned_cpython_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary) / "product"
            support = (
                product
                / "python"
                / "aie"
                / "_mlir_libs"
                / "MLIRPythonSupport-mlir.dll"
            )
            self.write_file(support, b"MZsupport")
            with (
                mock.patch.object(dev, "IS_WINDOWS", True),
                mock.patch.object(
                    dev,
                    "pe_imports",
                    return_value=["python312.dll"],
                ),
                self.assertRaisesRegex(RuntimeError, "versioned CPython DLL"),
            ):
                dev._stage_compiler_native_dependencies(product)

    def test_windows_product_verifier_accepts_stable_abi_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary) / "product"
            support = product / "python" / "MLIRPythonSupport-mlir.dll"
            self.write_file(support, b"MZsupport")
            with (
                mock.patch.object(dev, "IS_WINDOWS", True),
                mock.patch.object(dev, "pe_imports", return_value=["python3.dll"]),
            ):
                graph = dev._verify_compiler_native_dependencies(product)
            self.assertEqual(
                graph["python/MLIRPythonSupport-mlir.dll"],
                ["python3.dll"],
            )

    def test_windows_product_verifier_rejects_versioned_cpython_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary) / "product"
            support = product / "python" / "MLIRPythonSupport-mlir.dll"
            self.write_file(support, b"MZsupport")
            with (
                mock.patch.object(dev, "IS_WINDOWS", True),
                mock.patch.object(
                    dev,
                    "pe_imports",
                    return_value=["python312.dll"],
                ),
                self.assertRaisesRegex(RuntimeError, "versioned CPython DLL"),
            ):
                dev._verify_compiler_native_dependencies(product)

    def test_linux_compiler_product_bundles_non_manylinux_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = root / "product"
            openssl = root / "openssl"

            def write_elf(path):
                header = bytearray(18)
                header[:4] = b"\x7fELF"
                header[5] = 1
                header[16:18] = (3).to_bytes(2, "little")
                self.write_file(path, bytes(header))

            write_elf(product / "bin" / "bootgen")
            write_elf(openssl / "lib" / "libssl.so.1.1")
            write_elf(openssl / "lib" / "libcrypto.so.1.1")
            self.write_file(
                openssl / "share" / "openssl" / "copyright",
                b"OpenSSL license",
            )
            needed = {
                "bootgen": ["libssl.so.1.1", "libz.so.1"],
                "libssl.so.1.1": ["libcrypto.so.1.1"],
                "libcrypto.so.1.1": ["libc.so.6"],
            }

            def run(command, **kwargs):
                path = Path(command[-1])
                output = "\n".join(needed[path.name])
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=output + "\n",
                )

            with (
                mock.patch.object(dev, "IS_WINDOWS", False),
                mock.patch.dict(
                    os.environ,
                    {"OPENSSL_ROOT_DIR": str(openssl)},
                    clear=False,
                ),
                mock.patch.object(
                    dev.shutil,
                    "which",
                    return_value="/usr/bin/patchelf",
                ),
                mock.patch.object(dev.subprocess, "run", side_effect=run),
            ):
                copied = dev._stage_compiler_native_dependencies(product)

            self.assertEqual(
                set(copied),
                {"libssl.so.1.1", "libcrypto.so.1.1"},
            )
            self.assertTrue((product / "lib" / "libssl.so.1.1").is_file())
            self.assertTrue((product / "lib" / "libcrypto.so.1.1").is_file())
            self.assertTrue(
                (product / "licenses" / "openssl-copyright").is_file()
            )

    @unittest.skipIf(os.name == "nt", "Linux archive format test")
    def test_linux_archive_is_tar_gz_without_zstd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            product = self.make_product(root / "stage")
            with mock.patch("build_tools.fastflowlm_artifact.shutil.which",
                            lambda name: None):
                archive = dev._create_compiler_product_archive(product, root / "artifacts")

            self.assertTrue(archive.name.endswith(".tar.gz"))
            self.assertTrue(archive.with_name(archive.name + ".sha256").is_file())
            extracted = dev.extract_fastflowlm_archive(
                archive, root / "extracted", allowed_root=root)
            dev._verify_compiler_product_inventory(extracted)
            self.assertTrue(os.access(extracted / "bin" / "xclbinutil", os.X_OK))

    def test_relocated_toolchain_env_uses_product_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            self.write_file(prefix / "bin" / f"aie-opt{run_e2e.EXE}", executable=True)
            self.write_file(prefix / "bin" / f"xclbinutil{run_e2e.EXE}", executable=True)
            self.write_file(prefix / "python" / "aie" / "compiler" / "aiecc" / "main.py")
            self.write_file(prefix / "python" / "probe.py", b"VALUE = 1\n")
            (prefix / "peano" / "llvm-aie").mkdir(parents=True)
            (prefix / "third_party" / "aie_api").mkdir(parents=True)
            (prefix / "lib").mkdir()

            env = run_e2e.toolchain_env(
                prefix / "bin" / f"xclbinutil{run_e2e.EXE}", prefix)

            self.assertEqual(env["PYTHONPATH"], str(prefix / "python"))
            self.assertEqual(env["PEANO_INSTALL_DIR"], str(prefix / "peano" / "llvm-aie"))
            self.assertEqual(env["AIEOPT_DIR"], str(prefix / "third_party" / "aie_api"))
            self.assertEqual(env["ROCM_NPU_COMPILER_PREFIX"], str(prefix))
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
            subprocess.run(
                [sys.executable, "-c", "import probe"], env=env, check=True)
            self.assertFalse((prefix / "python" / "__pycache__").exists())
            if os.name == "nt":
                self.assertIn(str(prefix / "lib"), env["PATH"].split(os.pathsep))
            else:
                self.assertEqual(env["LD_LIBRARY_PATH"].split(os.pathsep)[0],
                                 str(prefix / "lib"))

    def test_relocated_toolchain_env_uses_installed_wheel_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site_packages = root / "site-packages"
            prefix = site_packages / "rocm_npu_compiler"
            self.write_file(prefix / "bin" / f"aie-opt{run_e2e.EXE}", executable=True)
            self.write_file(prefix / "bin" / f"xclbinutil{run_e2e.EXE}", executable=True)
            self.write_file(
                site_packages / "aie" / "compiler" / "aiecc" / "main.py"
            )
            (prefix / "peano" / "llvm-aie").mkdir(parents=True)
            (prefix / "third_party" / "aie_api").mkdir(parents=True)
            (prefix / "lib").mkdir()

            with mock.patch.object(
                run_e2e,
                "installed_aie_python_root",
                return_value=site_packages,
            ):
                env = run_e2e.toolchain_env(
                    prefix / "bin" / f"xclbinutil{run_e2e.EXE}", prefix
                )

            self.assertEqual(env["PYTHONPATH"], str(site_packages))


if __name__ == "__main__":
    unittest.main()
