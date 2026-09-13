# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from build_tools import hrx_runtime
from build_tools.dev import build, cli, device_testing, host_testing

pytestmark = pytest.mark.presubmit


def _public_subparsers():
    return next(
        action
        for action in cli.create_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    ).choices


class TestDevCli:
    @pytest.mark.parametrize(
        "arguments",
        [
            ["--help"],
            ["setup", "--help"],
            ["clang-format", "--help"],
            ["ruff", "--help"],
            ["format", "--help"],
        ],
    )
    def test_bootstrap_help_does_not_require_site_packages(self, arguments):
        root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [sys.executable, "-S", root / "dev.py", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    def test_root_module_is_executable_shim(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "dev.py").read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert functions == []
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        assert [
            (node.module, [alias.name for alias in node.names]) for node in imports
        ] == [("build_tools.dev.cli", ["main"])]

    def test_presubmit_runs_dev_core_tests(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github" / "workflows" / "ci.yml").read_text()
        assert "python -m pytest --basetemp=.pytest-tmp -m presubmit" in workflow

    def test_every_public_command_owns_dispatch_metadata(self):
        choices = _public_subparsers()
        assert len(choices) > 20
        assert all("_handler" in parser._defaults for parser in choices.values())
        assert all("_exit_on_return" in parser._defaults for parser in choices.values())

    def test_commands_register_actual_owner_functions(self):
        choices = _public_subparsers()
        expected_test_handlers = {
            "test": host_testing.cmd_test,
            "test-fastflowlm": host_testing.cmd_test_fastflowlm,
            "test-iron-ops": host_testing.cmd_test_iron_ops,
            "smoke": host_testing.cmd_smoke,
            "test-npu-package": device_testing.cmd_test_npu_package,
            "npu-probe": device_testing.cmd_npu_probe,
            "test-hrx-cts": device_testing.cmd_test_hrx_cts,
            "test-npu": device_testing.cmd_test_npu,
            "test-npu-jit": device_testing.cmd_test_npu_jit,
            "test-npu-iron-ops": device_testing.cmd_test_npu_iron_ops,
        }
        for name, handler in expected_test_handlers.items():
            assert choices[name]._defaults["_handler"] is handler
        assert all(
            parser._defaults["_handler"].__module__ != cli.__name__
            for name, parser in choices.items()
            if name != "build"
        )
        assert (
            choices["doctor"]._defaults["_handler"].__module__
            == "build_tools.dev.doctor"
        )
        assert (
            choices["clang-format"]._defaults["_handler"].__module__
            == "build_tools.dev.clang_format"
        )
        assert (
            choices["ruff"]._defaults["_handler"].__module__
            == "build_tools.dev.format_tools"
        )
        assert (
            choices["format"]._defaults["_handler"].__module__
            == "build_tools.dev.format_tools"
        )
        assert (
            choices["setup"]._defaults["_handler"].__module__ == "build_tools.dev.setup"
        )
        assert (
            choices["package-ops-wheels"]._defaults["_handler"].__module__
            == "build_tools.rocm_npu_ops_wheels"
        )
        assert (
            choices["install-compiler-wheels"]._defaults["_handler"].__module__
            == "build_tools.compiler_wheels"
        )
        assert (
            choices["verify-ops-wheels"]._defaults["_handler"].__module__
            == "build_tools.rocm_npu_ops_wheels"
        )
        verify_ops = choices["verify-ops-wheels"]
        assert any(option.dest == "reuse_venv" for option in verify_ops._actions)
        assert (
            choices["fetch"]._defaults["_handler"].__module__
            == "build_tools.dev.patches"
        )
        assert (
            choices["submodules"]._defaults["_handler"].__module__
            == "build_tools.dev.sources"
        )
        assert choices["build"]._defaults["_handler"] is cli._dispatch_build

    def test_build_registry_has_direct_domain_handlers(self):
        expected_modules = {
            "llvm": "build_tools.dev.compiler_build",
            "iron-tools": "build_tools.dev.compiler_build",
            "hrx-xclbinutil": "build_tools.dev.compiler_build",
            "hrx": "build_tools.dev.runtime_build",
            "rocm-npu-cli": "build_tools.dev.runtime_build",
            "fastflowlm": "build_tools.dev.fastflowlm_build",
            "fastflowlm-xrt": "build_tools.dev.fastflowlm_build",
            "all": "build_tools.dev.build",
        }
        assert {
            name: handler.__module__ for name, handler in build.BUILD_HANDLERS.items()
        } == expected_modules
        parsed = cli.create_parser().parse_args(["build", "llvm"])
        assert set(parsed._handler.__globals__["dev_build"].BUILD_HANDLERS) == set(
            expected_modules
        )

    def test_build_all_preserves_compiler_stack_sequence(self):
        calls = []
        names = (
            (build.runtime_build, "cmd_build_rocm_npu_cli"),
            (build.compiler_build, "cmd_build_hrx_xclbinutil"),
            (build.compiler_build, "cmd_build_llvm"),
            (build.compiler_build, "cmd_build_iron_tools"),
            (build.runtime_build, "cmd_build_hrx"),
        )
        patches = [
            mock.patch.object(
                owner, name, side_effect=lambda _, name=name: calls.append(name)
            )
            for owner, name in names
        ]
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            build.cmd_build_all(SimpleNamespace(fresh=False))
        assert calls == [name for _, name in names]

    def test_help_discovers_every_registered_command(self, capsys):
        parser = cli.create_parser()
        with pytest.raises(SystemExit) as raised:
            parser.parse_args(["--help"])
        assert raised.value.code == 0
        help_text = capsys.readouterr().out
        assert all(name in help_text for name in _public_subparsers())

    def test_exit_commands_preserve_exit_dispatch(self):
        assert {
            name
            for name, parser in _public_subparsers().items()
            if parser._defaults["_exit_on_return"]
        } == {
            "doctor",
            "clang-format",
            "ruff",
            "format",
            "smoke",
            "verify-npu-prereqs",
            "npu-probe",
            "test-hrx-cts",
        }

    def test_dispatch_exit_contract(self):
        regular = mock.Mock(return_value=7)
        assert (
            cli.dispatch(SimpleNamespace(_handler=regular, _exit_on_return=False)) == 7
        )
        exiting = mock.Mock(return_value=3)
        with pytest.raises(SystemExit) as raised:
            cli.dispatch(SimpleNamespace(_handler=exiting, _exit_on_return=True))
        assert raised.value.code == 3

    def test_parser_defaults_are_preserved(self):
        parser = cli.create_parser()
        setup = parser.parse_args(["setup"])
        assert setup.profile == "compiler-stack"
        assert not setup.iron_ops_tests
        doctor = parser.parse_args(["doctor", "fastflowlm", "--deep"])
        assert doctor.profile == "fastflowlm" and doctor.deep
        fastflowlm = parser.parse_args(["test-fastflowlm"])
        assert fastflowlm.inference_iterations == 1
        package = parser.parse_args(["test-npu-package"])
        assert package.lifecycle_rounds == 5 and package.resident_rounds == 10
        hrx = parser.parse_args(["build", "hrx", "--fresh", "--include-tests"])
        assert hrx.what == "hrx" and hrx.fresh and hrx.include_tests
        hrx_cts = parser.parse_args(["build", "hrx", "--include-cts"])
        assert hrx_cts.what == "hrx" and hrx_cts.include_cts
        iron_ops = parser.parse_args(["test-iron-ops"])
        assert iron_ops.jobs == 0
        verify_ops = parser.parse_args(["verify-ops-wheels"])
        assert verify_ops.jobs == 0
        assert parser.parse_args(["test-iron-ops", "--jobs", "4"]).jobs == 4
        assert parser.parse_args(["verify-ops-wheels", "--jobs", "4"]).jobs == 4
        jit = parser.parse_args(["test-npu-jit"])
        assert not jit.build_only
        install = parser.parse_args(["install-compiler-wheels", "--dir", "wheels"])
        assert install.dir == "wheels"

    @pytest.mark.parametrize(
        ("include_tests", "include_cts", "enabled"),
        [
            (False, False, "OFF"),
            (True, False, "ON"),
            (False, True, "ON"),
            (True, True, "ON"),
        ],
    )
    def test_hrx_tests_configure_required_benchmark_support(
        self, include_tests, include_cts, enabled
    ):
        assert build.runtime_build._hrx_test_cmake_options(
            include_tests, include_cts
        ) == (
            f"-DIREE_BUILD_TESTS={enabled}",
            f"-DIREE_BUILD_BENCHMARKS={enabled}",
            f"-DIREE_HAL_DRIVER_LOCAL_SYNC={enabled}",
        )

    def test_hrx_host_test_names_map_to_scoped_cmake_and_ctest_names(self):
        with mock.patch.object(hrx_runtime.paths, "is_windows", return_value=False):
            tests = hrx_runtime.hrx_amdxdna_host_tests()
            targets = hrx_runtime.hrx_amdxdna_host_test_targets()
            pattern = re.compile(hrx_runtime.hrx_amdxdna_host_test_regex())
        assert "native_windows_mcdm_test" not in tests
        assert targets == tuple(f"iree_hal_drivers_amdxdna_{name}" for name in tests)
        assert pattern.fullmatch("iree/hal/drivers/amdxdna/allocator_test")
        assert not pattern.fullmatch("iree/base/allocator_test")

        with mock.patch.object(hrx_runtime.paths, "is_windows", return_value=True):
            assert "native_windows_mcdm_test" in hrx_runtime.hrx_amdxdna_host_tests()

    def test_hrx_host_tests_exist_in_amdxdna_cmake(self):
        cmake_path = (
            hrx_runtime.paths.PROJ
            / "hrx"
            / "runtime"
            / "src"
            / "iree"
            / "hal"
            / "drivers"
            / "amdxdna"
            / "CMakeLists.txt"
        )
        if not cmake_path.is_file():
            pytest.skip("HRX submodule is not checked out")
        declared = set(
            re.findall(
                r"iree_cc_test\(\s*NAME\s+(\S+)",
                cmake_path.read_text(encoding="utf-8"),
            )
        )
        missing = [
            name
            for name in hrx_runtime.HRX_AMDXDNA_HOST_TESTS
            + hrx_runtime.HRX_AMDXDNA_HOST_TESTS_WINDOWS
            if name not in declared
        ]
        assert missing == []

    def test_windows_context_reclaim_is_covered_by_hrx_host_tests(self):
        """Lock the HAL tests that catch process-lifetime unique-xclbin leaks.

        Qualify --fast soaks 33 distinct XADX images in one runtime (past the
        Windows cache of architecture-budget-1). These hermetic host tests
        still lock the HAL classifier, budget-1 cache size, and lease-evict
        policy via ``dev.py build all --include-tests``.
        """
        amdxdna = (
            hrx_runtime.paths.PROJ
            / "hrx"
            / "runtime"
            / "src"
            / "iree"
            / "hal"
            / "drivers"
            / "amdxdna"
        )
        cache_test = amdxdna / "context_cache_test.cc"
        mcdm_test = amdxdna / "native_windows_mcdm_test.cc"
        mcdm_cc = amdxdna / "native_windows_mcdm.cc"
        if not cache_test.is_file() or not mcdm_test.is_file() or not mcdm_cc.is_file():
            pytest.skip("HRX submodule is not checked out")
        cache = cache_test.read_text(encoding="utf-8")
        mcdm = mcdm_test.read_text(encoding="utf-8")
        windows = mcdm_cc.read_text(encoding="utf-8")
        assert "CapacityFullOfLeasedEntriesForceEvicts" in cache
        assert "ClassifiesIpuRemappedCreateContextFailure" in mcdm
        assert "SizesCacheOneBelowArchitectureBudget" in mcdm
        assert "0xC01E0009" in mcdm
        assert "kWindowsHardwareContextCacheCap" not in windows
        assert (
            "iree_hal_amdxdna_native_windows_hardware_context_cache_capacity" in windows
        )

    @pytest.mark.parametrize("value", ["0", "-1", "abc"])
    def test_positive_integer_rejects_invalid_values(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.argparse_positive_int(value)
