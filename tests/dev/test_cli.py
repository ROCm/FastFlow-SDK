# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import dev


class DevCliTest(unittest.TestCase):
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
                self.assertEqual(len(node.body), 1, node.name)
                self.assertIsInstance(node.body[0], ast.Return, node.name)
                self.assertIsInstance(node.body[0].value, ast.Call, node.name)
            else:
                self.assertIn(node.name, allowed_functions)

        imported = {
            alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertEqual(imported, {"argparse", "sys"})

    def test_presubmit_runs_dev_core_tests(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github" / "workflows" / "ci.yml").read_text()
        self.assertIn(
            "python -m unittest discover -s tests/dev -p 'test_*.py'",
            workflow,
        )

    def test_every_public_subparser_owns_dispatch_metadata(self):
        parser = dev.create_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, dev.argparse._SubParsersAction)
        )
        self.assertGreater(len(subparsers.choices), 20)
        for name, command_parser in subparsers.choices.items():
            with self.subTest(command=name):
                self.assertIn("_handler", command_parser._defaults)
                self.assertIn("_exit_on_return", command_parser._defaults)
                expected_name = (
                    "_dispatch_build"
                    if name == "build"
                    else f"cmd_{name.replace('-', '_')}"
                )
                self.assertIs(
                    command_parser._defaults["_handler"],
                    getattr(dev, expected_name),
                )

        exit_commands = {
            name
            for name, command_parser in subparsers.choices.items()
            if command_parser._defaults["_exit_on_return"]
        }
        self.assertEqual(
            exit_commands,
            {
                "doctor",
                "smoke",
                "verify-npu-prereqs",
                "npu-probe",
                "test-hrx-cts",
            },
        )

    def test_core_command_wrappers_delegate_to_owning_modules(self):
        args = SimpleNamespace()
        with (
            mock.patch.object(dev.dev_patches, "cmd_fetch", return_value="fetch") as fetch,
            mock.patch.object(dev.dev_doctor, "cmd_doctor", return_value="doctor") as doctor,
            mock.patch.object(dev.dev_setup, "cmd_setup", return_value="setup") as setup,
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
            self.assertEqual(dev.cmd_fetch(args, {"hrx"}), "fetch")
            self.assertEqual(dev.cmd_doctor(args), "doctor")
            self.assertEqual(dev.cmd_setup(args), "setup")
            self.assertEqual(dev.cmd_build_llvm(args), "build")
            self.assertEqual(dev.cmd_submodules(args), "sources")
            self.assertEqual(dev.cmd_smoke(args), "smoke")
        fetch.assert_called_once_with(args, {"hrx"})
        doctor.assert_called_once_with(args)
        setup.assert_called_once_with(args)
        build.assert_called_once_with(args)
        sources.assert_called_once_with(args)
        smoke.assert_called_once_with(args)

    def test_artifact_wrappers_delegate_to_product_owners(self):
        args = SimpleNamespace()
        routes = {
            "cmd_package_fastflowlm": (
                dev.fastflowlm_package,
                "cmd_package_fastflowlm",
            ),
            "cmd_package_compiler_product": (
                dev.compiler_product,
                "cmd_package_compiler_product",
            ),
            "cmd_verify_compiler_wheels": (
                dev.compiler_wheels,
                "cmd_verify_compiler_wheels",
            ),
            "cmd_package_npu_prereqs": (
                dev.npu_prereqs,
                "cmd_package_npu_prereqs",
            ),
            "cmd_package_npu_test_package": (
                dev.npu_test_package,
                "cmd_package_npu_test_package",
            ),
        }
        for wrapper_name, (owner, owner_name) in routes.items():
            with self.subTest(wrapper=wrapper_name):
                with mock.patch.object(
                    owner, owner_name, return_value=wrapper_name
                ) as operation:
                    self.assertEqual(
                        getattr(dev, wrapper_name)(args), wrapper_name
                    )
                operation.assert_called_once_with(args)

    def test_dispatch_returns_for_regular_commands(self):
        handler = mock.Mock(return_value=7)
        args = SimpleNamespace(
            cmd="setup", _handler=handler, _exit_on_return=False
        )
        self.assertEqual(dev.dispatch(args), 7)
        handler.assert_called_once_with(args)

    def test_dispatch_preserves_exit_commands(self):
        handler = mock.Mock(return_value=3)
        args = SimpleNamespace(
            cmd="doctor", _handler=handler, _exit_on_return=True
        )
        with self.assertRaises(SystemExit) as raised:
            dev.dispatch(args)
        self.assertEqual(raised.exception.code, 3)

    def test_setup_parser_defaults_are_preserved(self):
        with (
            mock.patch.object(dev.sys, "argv", ["dev.py", "setup"]),
            mock.patch.object(dev, "dispatch") as dispatch,
        ):
            dev.main()
        args = dispatch.call_args.args[0]
        self.assertEqual(args.cmd, "setup")
        self.assertEqual(args.profile, "compiler-stack")
        self.assertFalse(args.iron_ops_tests)

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
        self.assertEqual(args.profile, "fastflowlm")
        self.assertTrue(args.deep)

    def test_artifact_defaults_come_from_owner_modules(self):
        parser = dev.create_parser()
        fastflowlm = parser.parse_args(["verify-fastflowlm-wheels"])
        self.assertEqual(
            fastflowlm.dir,
            str(dev.fastflowlm_package.FASTFLOWLM_ARTIFACTS),
        )
        self.assertEqual(
            fastflowlm.venv,
            str(dev.fastflowlm_package.FASTFLOWLM_WHEEL_VENV),
        )

        compiler = parser.parse_args(["verify-compiler-wheels"])
        self.assertEqual(
            compiler.dir,
            str(dev.compiler_wheels.COMPILER_WHEEL_ARTIFACTS),
        )
        self.assertEqual(
            compiler.venv,
            str(dev.compiler_wheels.COMPILER_WHEEL_VENV),
        )


if __name__ == "__main__":
    unittest.main()
