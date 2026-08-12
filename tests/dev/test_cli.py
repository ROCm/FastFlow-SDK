# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from build_tools.dev import build, cli, device_testing, host_testing

pytestmark = pytest.mark.presubmit


def _public_subparsers():
    return next(
        action
        for action in cli.create_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    ).choices


class TestDevCli:
    @pytest.mark.parametrize("arguments", [["--help"], ["setup", "--help"]])
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
            choices["setup"]._defaults["_handler"].__module__ == "build_tools.dev.setup"
        )
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
            "hrx-cts": "build_tools.dev.runtime_build",
            "rocm-npu-cli": "build_tools.dev.runtime_build",
            "fastflowlm": "build_tools.dev.fastflowlm_build",
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
        } == {"doctor", "smoke", "verify-npu-prereqs", "npu-probe", "test-hrx-cts"}

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

    @pytest.mark.parametrize("value", ["0", "-1", "abc"])
    def test_positive_integer_rejects_invalid_values(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.argparse_positive_int(value)
