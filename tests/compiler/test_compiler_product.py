# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
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

    def test_inventory_rejects_staging_artifact_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            product = self.make_product(
                Path(temporary), "rocm-npu-staging-compiler-linux-x86_64")
            with self.assertRaisesRegex(RuntimeError, "must not include staging"):
                dev._verify_compiler_product_inventory(product)

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
            (prefix / "peano" / "llvm-aie").mkdir(parents=True)
            (prefix / "third_party" / "aie_api").mkdir(parents=True)
            (prefix / "lib").mkdir()

            env = run_e2e.toolchain_env(
                prefix / "bin" / f"xclbinutil{run_e2e.EXE}", prefix)

            self.assertEqual(env["PYTHONPATH"], str(prefix / "python"))
            self.assertEqual(env["PEANO_INSTALL_DIR"], str(prefix / "peano" / "llvm-aie"))
            self.assertEqual(env["AIEOPT_DIR"], str(prefix / "third_party" / "aie_api"))
            self.assertEqual(env["ROCM_NPU_COMPILER_PREFIX"], str(prefix))
            if os.name == "nt":
                self.assertIn(str(prefix / "lib"), env["PATH"].split(os.pathsep))
            else:
                self.assertEqual(env["LD_LIBRARY_PATH"].split(os.pathsep)[0],
                                 str(prefix / "lib"))


if __name__ == "__main__":
    unittest.main()
