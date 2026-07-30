# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from build_tools import (
    compiler_product,
    compiler_wheels,
    fastflowlm_package,
    npu_prereqs,
    npu_test_package,
)
from build_tools.dev import build, iron, paths, sources, testing


class PathOwnershipTest(unittest.TestCase):
    def test_repository_paths_expose_only_repository_layout(self):
        path_constants = {
            name
            for name, value in vars(paths).items()
            if name.isupper() and isinstance(value, Path)
        }
        self.assertEqual(
            path_constants,
            {
                "ROOT",
                "PROJ",
                "BUILD",
                "OUT",
                "VENV",
                "PATCHES",
                "FETCH_CACHE",
            },
        )

    def test_domain_paths_are_derived_by_their_owners(self):
        self.assertEqual(build.HRX_INSTALL, paths.BUILD / "hrx-install")
        self.assertEqual(
            fastflowlm_package.FASTFLOWLM_WHEEL_VENV,
            paths.BUILD / "rocm-npu-wheel-venv",
        )
        self.assertEqual(
            compiler_product.COMPILER_PRODUCT_STAGE,
            paths.BUILD / "rocm-npu-compiler-product",
        )
        self.assertEqual(
            compiler_wheels.COMPILER_WHEEL_STAGE,
            paths.BUILD / "rocm-npu-compiler-wheel-stage",
        )
        self.assertEqual(
            npu_prereqs.NPU_PREREQ_DIR,
            paths.BUILD / "artifacts",
        )
        self.assertEqual(
            npu_test_package.NPU_TEST_PACKAGE_STAGE,
            paths.BUILD / "npu-test-package",
        )


class BuildDomainTest(unittest.TestCase):
    def test_compiler_flags_require_a_complete_pair(self):
        self.assertEqual(build._cc_flags(None, None), [])
        self.assertEqual(build._cc_flags("cc", None), [])
        self.assertEqual(
            build._cc_flags("cc", "c++"),
            ["-DCMAKE_C_COMPILER=cc", "-DCMAKE_CXX_COMPILER=c++"],
        )

    def test_fresh_tree_rejects_unknown_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError, "unexpected build directory"
            ):
                build._prepare_fresh_cmake_tree(Path(directory))

    def test_vendored_xclbinutil_build_includes_smoke_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = (
                root
                / "projects"
                / "iron-tools"
                / "third_party"
                / "hrx-xclbinutil"
            )
            (source / "CMakeLists.txt").parent.mkdir(parents=True)
            (source / "CMakeLists.txt").touch()
            with (
                mock.patch.object(build.paths, "PROJ", root / "projects"),
                mock.patch.object(build, "TOOLS_BIN", root / "tools-bin"),
                mock.patch.object(
                    build.environment, "cmake_bin", return_value="cmake"
                ),
                mock.patch.object(
                    build.environment,
                    "cmake_path",
                    side_effect=lambda path: str(path),
                ),
                mock.patch.object(
                    build.environment, "cmake_env", return_value={}
                ),
                mock.patch.object(build.environment, "sh") as run,
                mock.patch.object(Path, "symlink_to"),
                mock.patch.object(build.paths, "is_windows", return_value=False),
            ):
                build.cmd_build_hrx_xclbinutil(SimpleNamespace())

            configure, compile_command = (
                call.args[0] for call in run.call_args_list
            )
            self.assertIn(str(source), configure)
            self.assertIn("hrx-xclbinutil", compile_command)
            self.assertIn("pdi_transform_test", compile_command)

    def test_llvm_build_includes_objcopy_for_relocated_products(self):
        self.assertIn("llvm-objcopy", build.LLVM_BUILD_TARGETS)


class SourceDomainTest(unittest.TestCase):
    def test_submodules_groups_flat_and_recursive_updates(self):
        args = SimpleNamespace(
            names=["llvm-project", "iron-tools"],
            reset=True,
            shallow=True,
            clean=[],
        )
        with mock.patch.object(sources.environment, "sh") as run:
            sources.cmd_submodules(args)
        self.assertEqual(run.call_count, 2)
        self.assertNotIn("--recursive", run.call_args_list[0].args[0])
        self.assertIn("--recursive", run.call_args_list[1].args[0])
        self.assertTrue(
            all(call.kwargs["cwd"] == paths.ROOT for call in run.call_args_list)
        )


class TestingDomainTest(unittest.TestCase):
    def test_gtest_summary_counts_passes_and_skips(self):
        output = "[  PASSED  ] 12 tests.\n[  SKIPPED ] 3 tests.\n"
        self.assertEqual(testing._gtest_counts(output), (12, 3))

    def test_windows_npu_probe_routes_to_windows_owner(self):
        args = SimpleNamespace(bundle=None, runtime_package=False)
        library = Path("hrx.dll")
        with (
            mock.patch.object(testing.paths, "is_windows", return_value=True),
            mock.patch.object(
                testing.hrx_runtime, "libhrx_path", return_value=library
            ),
            mock.patch.object(
                testing, "probe_windows_npu_runtime", return_value=7
            ) as probe,
        ):
            self.assertEqual(testing.cmd_npu_probe(args), 7)
        probe.assert_called_once_with(library)

    def test_linux_npu_probe_stops_without_a_device(self):
        args = SimpleNamespace(bundle=None, runtime_package=False)
        with (
            mock.patch.object(testing.paths, "is_windows", return_value=False),
            mock.patch.object(testing.linux_npu, "print_npu_driver_info"),
            mock.patch.object(
                testing.linux_npu,
                "probe_npu_devnodes",
                return_value=(0, []),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(testing.cmd_npu_probe(args), 1)


class IronPolicyTest(unittest.TestCase):
    def test_transaction_generation_matches_target(self):
        with tempfile.TemporaryDirectory() as directory:
            transaction = Path(directory) / "insts.bin"
            transaction.write_bytes(b"\x00\x00\x04")
            with contextlib.redirect_stdout(io.StringIO()):
                iron.require_txn_devgen(transaction, "npu2")


class NpuTestPackageCommandTest(unittest.TestCase):
    def test_extracted_verification_uses_package_owner(self):
        args = SimpleNamespace(extracted=True)
        manifest = {"cases": [{"name": "vector_scalar_add"}]}
        with (
            mock.patch.object(
                npu_test_package,
                "verify_npu_test_package",
                return_value=manifest,
            ) as verify,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            npu_test_package.cmd_verify_npu_test_package(args)
        verify.assert_called_once_with(
            npu_test_package.NPU_TEST_PACKAGE_EXTRACTED
        )


if __name__ == "__main__":
    unittest.main()
