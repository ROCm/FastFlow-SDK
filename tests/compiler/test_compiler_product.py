# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
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

    def make_product(self, root: Path, artifact: str = "rocm-npu-compiler-linux-x86_64") -> Path:
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

    def test_inventory_accepts_complete_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = self.make_product(Path(temporary))
            manifest = dev._verify_compiler_product_inventory(product)
            self.assertEqual(manifest["product"], "rocm-npu-compiler")

    def test_distribution_audit_runs_payload_checks_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = self.make_product(Path(temporary))

            with mock.patch.object(dev, "_verify_compiler_product_linux_elf",
                                   return_value={"elf_files": 0}) as linux_elf, \
                 mock.patch.object(dev, "_verify_compiler_product_no_local_paths",
                                   return_value={"local_paths": []}) as local_paths:
                audit = dev._verify_compiler_product_distribution(product)

            self.assertEqual(audit["manifest"]["product"], "rocm-npu-compiler")
            linux_elf.assert_called_once_with(product)
            local_paths.assert_called_once_with(product)

    def test_inventory_rejects_staging_artifact_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = self.make_product(
                Path(temporary), "rocm-npu-staging-compiler-linux-x86_64")
            with self.assertRaisesRegex(RuntimeError, "must not include staging"):
                dev._verify_compiler_product_inventory(product)

    def test_python_bytecode_is_pruned_from_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary)
            self.write_file(product / "python" / "aie" / "__pycache__" / "ir.pyc")
            self.write_file(product / "python" / "aie" / "module.pyo")

            dev._prune_python_bytecode(product)

            self.assertFalse((product / "python" / "aie" / "__pycache__").exists())
            self.assertFalse((product / "python" / "aie" / "module.pyo").exists())

    def test_build_metadata_is_pruned_from_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary)
            self.write_file(product / "lib" / "CMakeFiles" / "obj.dir" / "payload.o")
            self.write_file(product / "runtime_lib" / "xaiengineTmp" / "cfgcmd.txt")
            self.write_file(product / "runtime_lib" / "x86_64" / "xaiengine" / "build.ninja")
            self.write_file(product / "include" / "aie" / "cmake_install.cmake")
            self.write_file(product / "lib" / "cmake" / "aie" / "AIEConfig.cmake")
            self.write_file(product / "lib" / "libAIE.a")

            removed = dev._prune_compiler_product_build_metadata(product)

            self.assertIn("lib/CMakeFiles/", removed)
            self.assertFalse((product / "lib" / "CMakeFiles").exists())
            self.assertFalse((product / "runtime_lib" / "xaiengineTmp").exists())
            self.assertFalse((product / "runtime_lib" / "x86_64" / "xaiengine" / "build.ninja").exists())
            self.assertFalse((product / "include" / "aie" / "cmake_install.cmake").exists())
            self.assertFalse((product / "lib" / "cmake" / "aie").exists())
            self.assertTrue((product / "lib" / "libAIE.a").exists())

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

    def test_relocated_toolchain_resolves_product_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            self.write_file(prefix / "bin" / f"aie-opt{run_e2e.EXE}", executable=True)
            self.write_file(prefix / "bin" / f"xclbinutil{run_e2e.EXE}", executable=True)
            self.write_file(prefix / "python" / "aie" / "compiler" / "aiecc" / "main.py")
            (prefix / "peano" / "llvm-aie").mkdir(parents=True)
            (prefix / "third_party" / "aie_api").mkdir(parents=True)
            (prefix / "lib").mkdir()

            toolchain = run_e2e.resolve_toolchain(prefix)
            env = toolchain.process_env({"PATH": "ambient-bin"})

            self.assertEqual(toolchain.compiler_prefix, prefix)
            self.assertEqual(toolchain.xclbinutil,
                             prefix / "bin" / f"xclbinutil{run_e2e.EXE}")
            self.assertEqual(toolchain.aie_opt, prefix / "bin" / f"aie-opt{run_e2e.EXE}")
            self.assertEqual(env["PYTHONPATH"], str(prefix / "python"))
            self.assertEqual(env["PEANO_INSTALL_DIR"], str(prefix / "peano" / "llvm-aie"))
            self.assertEqual(env["AIEOPT_DIR"], str(prefix / "third_party" / "aie_api"))
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(env["ROCM_NPU_COMPILER_PREFIX"], str(prefix))
            self.assertEqual(env["PATH"].split(os.pathsep)[-1], "ambient-bin")
            self.assertNotIn("LD_LIBRARY_PATH", env)
            if os.name == "nt":
                self.assertIn(str(prefix / "lib"), env["PATH"].split(os.pathsep))

    @unittest.skipIf(os.name == "nt", "POSIX compiler flag test")
    def test_posix_build_flags_map_local_source_paths(self):
        flags = dev._compiler_product_source_path_map_flags()
        joined = " ".join(flags)
        self.assertIn("-ffile-prefix-map=", joined)
        self.assertIn("-fdebug-prefix-map=", joined)
        self.assertIn("-fmacro-prefix-map=", joined)
        self.assertIn(dev.ROOT.resolve().as_posix(), joined)

    @unittest.skipIf(os.name == "nt", "Linux ELF RPATH test")
    def test_compiler_product_rpath_repair_drops_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary) / "rocm-npu-compiler-linux-x86_64"
            bin_tool = product / "bin" / "aie-opt"
            py_ext = product / "python" / "aie" / "_mlir_libs" / "_mlir.so"

            self.assertEqual(
                dev._compiler_product_repaired_rpath(
                    product, bin_tool, "$ORIGIN/../lib:/tmp/build/llvm/lib"),
                "$ORIGIN/../lib",
            )
            self.assertEqual(
                dev._compiler_product_repaired_rpath(
                    product, py_ext, "$ORIGIN/../lib:/tmp/build/iron-tools/python"),
                "$ORIGIN:$ORIGIN/../../../lib:$ORIGIN/../lib",
            )

    @unittest.skipIf(os.name == "nt", "Linux ELF audit test")
    def test_compiler_product_audit_rejects_absolute_runpath(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary) / "rocm-npu-compiler-linux-x86_64"
            payload = product / "bin" / "aie-opt"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"\x7fELF")

            with mock.patch.object(dev, "_compiler_product_linux_elf_files",
                                   lambda root: [payload]), \
                 mock.patch.object(dev, "_readelf_rpath",
                                   lambda path, readelf=None: "$ORIGIN/../lib:/tmp/build/llvm/lib"), \
                 mock.patch("dev.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(
                    ["ldd", str(payload)], 0,
                    stdout="\tlibc.so.6 => /lib64/libc.so.6 (0x1)\n",
                )
                with self.assertRaisesRegex(RuntimeError, "absolute RPATH/RUNPATH"):
                    dev._verify_compiler_product_linux_elf(product)

    @unittest.skipIf(os.name == "nt", "Linux tool diagnostic test")
    def test_linux_tool_probe_reports_missing_package_tools(self):
        with mock.patch("dev.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "binutils patchelf"):
                dev._compiler_product_linux_tools(["strip", "patchelf"])

    @unittest.skipIf(os.name == "nt", "Linux local path audit test")
    def test_compiler_product_audit_rejects_local_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary)
            self.write_file(product / "lib" / "cmake" / "aie" / "AIEConfig.cmake",
                            b'set(AIE_BINARY_DIR "/tmp/build/iron-tools")')

            with mock.patch.object(dev, "_compiler_product_local_path_markers",
                                   lambda: ["/tmp/build"]):
                with self.assertRaisesRegex(RuntimeError, "local build paths"):
                    dev._verify_compiler_product_no_local_paths(product)

    def test_local_path_audit_rejects_windows_marker_spelling(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = Path(temporary)
            self.write_file(product / "bin" / "tool.exe",
                            br'compiled from C:\a\repo\build\llvm\tools\mlir')

            with self.assertRaisesRegex(RuntimeError, "local build paths"):
                dev._verify_compiler_product_no_local_paths(
                    product, markers=["C:/a/repo/build"])

    def test_local_path_marker_variants_cover_native_and_slash_forms(self):
        markers = dev._compiler_product_local_path_marker_variants(r"C:\a\repo\build")

        self.assertIn(r"C:\a\repo\build", markers)
        self.assertIn("C:/a/repo/build", markers)


if __name__ == "__main__":
    unittest.main()
