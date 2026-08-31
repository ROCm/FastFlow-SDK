# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from build_tools import (
    compiler_product,
    compiler_wheels,
    environment,
    fastflowlm_package,
    iron,
    npu_prereqs,
    npu_test_package,
    paths,
    rocm_npu_ops_wheels,
)
from build_tools.dev import compiler_build, device_testing, runtime_build, sources

pytestmark = pytest.mark.presubmit


class TestPathOwnership:
    def test_repository_paths_are_a_shared_foundation(self):
        assert paths.__name__ == "build_tools.paths"
        assert not (paths.ROOT / "build_tools" / "dev" / "paths.py").exists()

    def test_repository_paths_expose_shared_build_layout(self):
        path_constants = {
            name
            for name, value in vars(paths).items()
            if name.isupper() and isinstance(value, Path)
        }
        assert {
            "ROOT",
            "PROJ",
            "BUILD",
            "OUT",
            "VENV",
            "PATCHES",
            "FETCH_CACHE",
            "LLVM_BUILD",
            "IRON_TOOLS_BUILD",
            "IRON_TOOLS_INSTALL",
            "TOOLS_BIN",
            "HRX_CMAKE_BUILD",
            "HRX_INSTALL",
            "HRX_PACKAGE_SMOKE_BUILD",
            "ROCM_NPU_CLI_BUILD",
            "ROCM_NPU_CLI_INSTALL",
            "FASTFLOWLM_BUILD",
            "FASTFLOWLM_INSTALL",
            "FASTFLOWLM_MODEL_CACHE",
        } <= path_constants

    def test_domain_paths_are_derived_by_their_owners(self):
        assert paths.HRX_INSTALL == paths.BUILD / "hrx-install"
        assert fastflowlm_package.FASTFLOWLM_WHEEL_VENV == (
            paths.BUILD / "rocm-npu-wheel-venv"
        )
        assert compiler_product.COMPILER_PRODUCT_STAGE == (
            paths.BUILD / "rocm-npu-compiler-product"
        )
        assert compiler_wheels.COMPILER_WHEEL_STAGE == (
            paths.BUILD / "rocm-npu-compiler-wheel-stage"
        )
        assert rocm_npu_ops_wheels.OPS_WHEEL_VENV == (
            paths.BUILD / "rocm-npu-ops-wheel-venv"
        )
        assert rocm_npu_ops_wheels.OPS_WHEEL_ARTIFACTS == (
            compiler_product.COMPILER_ARTIFACTS
        )
        assert npu_prereqs.NPU_PREREQ_DIR == paths.BUILD / "artifacts"
        assert npu_test_package.NPU_TEST_PACKAGE_STAGE == (
            paths.BUILD / "npu-test-package"
        )


class TestBuildDomain:
    def test_compiler_flags_require_a_complete_pair(self):
        assert environment.cmake_compiler_flags(None, None) == []
        assert environment.cmake_compiler_flags("cc", None) == []
        assert environment.cmake_compiler_flags("cc", "c++") == [
            "-DCMAKE_C_COMPILER=cc",
            "-DCMAKE_CXX_COMPILER=c++",
        ]

    def test_fresh_tree_rejects_unknown_directory(self, tmp_path):
        with pytest.raises(RuntimeError, match="unexpected build directory"):
            runtime_build._prepare_fresh_cmake_tree(tmp_path)

    def test_vendored_xclbinutil_build_includes_smoke_helper(self, tmp_path):
        source = tmp_path / "projects" / "iron-tools" / "third_party" / "hrx-xclbinutil"
        (source / "CMakeLists.txt").parent.mkdir(parents=True)
        (source / "CMakeLists.txt").touch()
        with (
            mock.patch.object(compiler_build.paths, "PROJ", tmp_path / "projects"),
            mock.patch.object(compiler_build.paths, "TOOLS_BIN", tmp_path / "tools-bin"),
            mock.patch.object(compiler_build.environment, "cmake_bin", return_value="cmake"),
            mock.patch.object(
                compiler_build.environment,
                "cmake_path",
                side_effect=lambda path: str(path),
            ),
            mock.patch.object(compiler_build.environment, "cmake_env", return_value={}),
            mock.patch.object(compiler_build.environment, "sh") as run,
            mock.patch.object(Path, "symlink_to"),
            mock.patch.object(compiler_build.paths, "is_windows", return_value=False),
        ):
            compiler_build.cmd_build_hrx_xclbinutil(SimpleNamespace())

        configure, compile_command = (call.args[0] for call in run.call_args_list)
        assert str(source) in configure
        assert "hrx-xclbinutil" in compile_command
        assert "pdi_transform_test" in compile_command

    def test_llvm_build_includes_objcopy_for_relocated_products(self):
        assert "llvm-objcopy" in compiler_build.LLVM_BUILD_TARGETS


class TestSourceDomain:
    def test_submodules_groups_flat_and_recursive_updates(self):
        args = SimpleNamespace(
            names=["llvm-project", "iron-tools"],
            reset=True,
            shallow=True,
            clean=[],
        )
        with mock.patch.object(sources.environment, "sh") as run:
            sources.cmd_submodules(args)
        commands = [call.args[0] for call in run.call_args_list]
        assert commands == [
            ["git", "submodule", "sync", "projects/llvm-project"],
            ["git", "submodule", "update", "--init", "--force",
             "--depth", "1", "--jobs", "4", "projects/llvm-project"],
            ["git", "submodule", "sync", "--recursive", "projects/iron-tools"],
            ["git", "submodule", "update", "--init", "--force", "--recursive",
             "--depth", "1", "--jobs", "4", "projects/iron-tools"],
        ]
        assert all(call.kwargs["cwd"] == paths.ROOT for call in run.call_args_list)

    def test_submodules_syncs_repointed_remotes_before_fetching(self):
        """A repointed .gitmodules URL must reach an already-initialized clone."""
        args = SimpleNamespace(
            names=["llvm-project"], reset=False, shallow=False, clean=[]
        )
        with mock.patch.object(sources.environment, "sh") as run:
            sources.cmd_submodules(args)
        commands = [call.args[0] for call in run.call_args_list]
        assert commands[0][:3] == ["git", "submodule", "sync"]
        assert commands[1][:3] == ["git", "submodule", "update"]


class TestTestingDomain:
    def test_gtest_summary_counts_passes_and_skips(self):
        output = "[  PASSED  ] 12 tests.\n[  SKIPPED ] 3 tests.\n"
        assert device_testing._gtest_counts(output) == (12, 3)

    def test_hrx_cts_timeout_preserves_diagnostics_and_summary(self, tmp_path):
        executable = tmp_path.joinpath(
            *device_testing.hrx_runtime.HRX_CTS_DIR,
            paths.executable_name("core_tests"),
        )
        executable.parent.mkdir(parents=True)
        executable.touch()
        timeout = device_testing.subprocess.TimeoutExpired(
            ["core_tests"],
            device_testing.HRX_CTS_TIMEOUT_SECONDS,
            output="[  PASSED  ] 2 tests.\n",
            stderr="partial diagnostic\n",
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(
                device_testing.paths, "HRX_CMAKE_BUILD", tmp_path
            ),
            mock.patch.object(
                device_testing.environment, "run", side_effect=timeout
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = device_testing.cmd_test_hrx_cts(
                SimpleNamespace(suite="core_tests")
            )

        assert result == 1
        assert "2 passed" in stdout.getvalue()
        assert "partial diagnostic" in stderr.getvalue()
        assert "timed out after" in stderr.getvalue()

    def test_windows_npu_probe_routes_to_windows_owner(self):
        args = SimpleNamespace(bundle=None, runtime_package=False)
        library = Path("hrx.dll")
        with (
            mock.patch.object(device_testing.paths, "is_windows", return_value=True),
            mock.patch.object(device_testing.hrx_runtime, "libhrx_path", return_value=library),
            mock.patch.object(
                device_testing, "probe_windows_npu_runtime", return_value=7
            ) as probe,
        ):
            assert device_testing.cmd_npu_probe(args) == 7
        probe.assert_called_once_with(library)

    def test_linux_npu_probe_stops_without_a_device(self):
        args = SimpleNamespace(bundle=None, runtime_package=False)
        with (
            mock.patch.object(device_testing.paths, "is_windows", return_value=False),
            mock.patch.object(device_testing.linux_npu, "print_npu_driver_info"),
            mock.patch.object(
                device_testing.linux_npu,
                "probe_npu_devnodes",
                return_value=(0, []),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            assert device_testing.cmd_npu_probe(args) == 1


class TestIronPolicy:
    def test_iron_ops_environment_extends_shared_toolchain(self):
        base = {"PYTHONPATH": "toolchain"}
        with mock.patch.object(iron, "iron_env", return_value=base.copy()):
            env = iron.iron_ops_env()
        assert env["PYTHONPATH"] == os.pathsep.join(
            [str(paths.PROJ / "iron-ops"), "toolchain"]
        )

    def test_transaction_generation_matches_target(self, tmp_path):
        transaction = tmp_path / "insts.bin"
        transaction.write_bytes(b"\x00\x00\x04")
        with contextlib.redirect_stdout(io.StringIO()):
            iron.require_txn_devgen(transaction, "npu2")


class TestNpuTestPackageCommand:
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
        verify.assert_called_once_with(npu_test_package.NPU_TEST_PACKAGE_EXTRACTED)
