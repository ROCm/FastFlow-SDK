# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import dev


class DevCliTest(unittest.TestCase):
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
        ):
            self.assertEqual(dev.cmd_fetch(args, {"hrx"}), "fetch")
            self.assertEqual(dev.cmd_doctor(args), "doctor")
            self.assertEqual(dev.cmd_setup(args), "setup")
        fetch.assert_called_once_with(args, {"hrx"})
        doctor.assert_called_once_with(args)
        setup.assert_called_once_with(args)

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


if __name__ == "__main__":
    unittest.main()
