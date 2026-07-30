# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import dev

pytestmark = pytest.mark.presubmit


def _public_subparser_cases():
    parser = dev.create_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, dev.argparse._SubParsersAction)
    )
    return list(subparsers.choices.items())


class TestDevCli:
    def test_dev_module_is_a_structural_cli_facade(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "dev.py").read_text(encoding="utf-8"))
        allowed_functions = {
            "argparse_positive_int",
            "_dispatch_build",
            "_add_command",
            "dispatch",
            "create_parser",
            "main",
        }
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("cmd_"):
                assert len(node.body) == 1, node.name
                assert isinstance(node.body[0], ast.Return), node.name
                assert isinstance(node.body[0].value, ast.Call), node.name
            else:
                assert node.name in allowed_functions

        imported = {
            alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert imported == {"argparse", "sys"}

    def test_presubmit_runs_dev_core_tests(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github" / "workflows" / "ci.yml").read_text()
        assert "python -m pytest -m presubmit" in workflow

    @pytest.mark.parametrize(
        ("name", "command_parser"),
        _public_subparser_cases(),
        ids=lambda value: value if isinstance(value, str) else None,
    )
    def test_every_public_subparser_owns_dispatch_metadata(self, name, command_parser):
        assert len(_public_subparser_cases()) > 20
        assert "_handler" in command_parser._defaults
        assert "_exit_on_return" in command_parser._defaults
        expected_name = (
            "_dispatch_build" if name == "build" else f"cmd_{name.replace('-', '_')}"
        )
        assert command_parser._defaults["_handler"] is getattr(dev, expected_name)

    def test_exit_commands_preserve_exit_dispatch(self):
        choices = dict(_public_subparser_cases())
        exit_commands = {
            name
            for name, command_parser in choices.items()
            if command_parser._defaults["_exit_on_return"]
        }
        assert exit_commands == {
            "doctor",
            "smoke",
            "verify-npu-prereqs",
            "npu-probe",
            "test-hrx-cts",
        }

    def test_core_command_wrappers_delegate_to_owning_modules(self):
        args = SimpleNamespace()
        with (
            mock.patch.object(
                dev.dev_patches, "cmd_fetch", return_value="fetch"
            ) as fetch,
            mock.patch.object(
                dev.dev_doctor, "cmd_doctor", return_value="doctor"
            ) as doctor,
            mock.patch.object(
                dev.dev_setup, "cmd_setup", return_value="setup"
            ) as setup,
            mock.patch.object(
                dev.dev_build, "cmd_build_llvm", return_value="build"
            ) as build,
            mock.patch.object(
                dev.dev_sources, "cmd_submodules", return_value="sources"
            ) as sources,
            mock.patch.object(
                dev.dev_testing, "cmd_smoke", return_value="smoke"
            ) as smoke,
        ):
            assert dev.cmd_fetch(args, {"hrx"}) == "fetch"
            assert dev.cmd_doctor(args) == "doctor"
            assert dev.cmd_setup(args) == "setup"
            assert dev.cmd_build_llvm(args) == "build"
            assert dev.cmd_submodules(args) == "sources"
            assert dev.cmd_smoke(args) == "smoke"
        fetch.assert_called_once_with(args, {"hrx"})
        doctor.assert_called_once_with(args)
        setup.assert_called_once_with(args)
        build.assert_called_once_with(args)
        sources.assert_called_once_with(args)
        smoke.assert_called_once_with(args)

    @pytest.mark.parametrize(
        ("wrapper_name", "owner", "owner_name"),
        [
            (
                "cmd_package_fastflowlm",
                dev.fastflowlm_package,
                "cmd_package_fastflowlm",
            ),
            (
                "cmd_package_compiler_product",
                dev.compiler_product,
                "cmd_package_compiler_product",
            ),
            (
                "cmd_verify_compiler_wheels",
                dev.compiler_wheels,
                "cmd_verify_compiler_wheels",
            ),
            (
                "cmd_package_npu_prereqs",
                dev.npu_prereqs,
                "cmd_package_npu_prereqs",
            ),
            (
                "cmd_package_npu_test_package",
                dev.npu_test_package,
                "cmd_package_npu_test_package",
            ),
        ],
    )
    def test_artifact_wrappers_delegate_to_product_owners(
        self, wrapper_name, owner, owner_name
    ):
        args = SimpleNamespace()
        with mock.patch.object(
            owner, owner_name, return_value=wrapper_name
        ) as operation:
            assert getattr(dev, wrapper_name)(args) == wrapper_name
        operation.assert_called_once_with(args)

    def test_dispatch_returns_for_regular_commands(self):
        handler = mock.Mock(return_value=7)
        args = SimpleNamespace(cmd="setup", _handler=handler, _exit_on_return=False)
        assert dev.dispatch(args) == 7
        handler.assert_called_once_with(args)

    def test_dispatch_preserves_exit_commands(self):
        handler = mock.Mock(return_value=3)
        args = SimpleNamespace(cmd="doctor", _handler=handler, _exit_on_return=True)
        with pytest.raises(SystemExit) as raised:
            dev.dispatch(args)
        assert raised.value.code == 3

    def test_setup_parser_defaults_are_preserved(self):
        with (
            mock.patch.object(dev.sys, "argv", ["dev.py", "setup"]),
            mock.patch.object(dev, "dispatch") as dispatch,
        ):
            dev.main()
        args = dispatch.call_args.args[0]
        assert args.cmd == "setup"
        assert args.profile == "compiler-stack"
        assert not args.iron_ops_tests

    def test_doctor_parser_accepts_deep_profile(self):
        with (
            mock.patch.object(
                dev.sys,
                "argv",
                ["dev.py", "doctor", "fastflowlm", "--deep"],
            ),
            mock.patch.object(dev, "dispatch") as dispatch,
        ):
            dev.main()
        args = dispatch.call_args.args[0]
        assert args.profile == "fastflowlm"
        assert args.deep

    def test_artifact_defaults_come_from_owner_modules(self):
        parser = dev.create_parser()
        fastflowlm = parser.parse_args(["verify-fastflowlm-wheels"])
        assert fastflowlm.dir == str(dev.fastflowlm_package.FASTFLOWLM_ARTIFACTS)
        assert fastflowlm.venv == str(dev.fastflowlm_package.FASTFLOWLM_WHEEL_VENV)

        compiler = parser.parse_args(["verify-compiler-wheels"])
        assert compiler.dir == str(dev.compiler_wheels.COMPILER_WHEEL_ARTIFACTS)
        assert compiler.venv == str(dev.compiler_wheels.COMPILER_WHEEL_VENV)
