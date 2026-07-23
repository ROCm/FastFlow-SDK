# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests" / "e2e")]

import dev  # noqa: E402
import run_e2e  # noqa: E402


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
            self.write_file(source / "include" / "runtime.h")
            self.write_file(source / "CMakeFiles" / "target.dir" / "runtime.obj")
            self.write_file(source / "lib" / "stray.pdb")
            self.write_file(source / "__pycache__" / "module.cpython-312.pyc")

            destination = root / "destination"
            dev._stage_runtime_tree(source, destination)

            self.assertTrue((destination / "lib" / "runtime.lib").is_file())
            self.assertTrue((destination / "include" / "runtime.h").is_file())
            self.assertFalse((destination / "CMakeFiles").exists())
            self.assertFalse((destination / "lib" / "stray.pdb").exists())
            self.assertFalse((destination / "__pycache__").exists())

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


if __name__ == "__main__":
    unittest.main()
