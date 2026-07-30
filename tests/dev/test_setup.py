# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from build_tools.dev import paths, setup


class RequirementsFileTest(unittest.TestCase):
    def test_profile_requirements_are_explicit(self):
        root = Path("root")
        self.assertEqual(
            setup.requirements_file("fastflowlm", root=root, is_windows=True),
            (root / "requirements-fastflowlm.lock", True),
        )
        self.assertEqual(
            setup.requirements_file("npu-runtime", root=root, is_windows=False),
            (root / "requirements-npu-runtime.txt", False),
        )
        self.assertEqual(
            setup.requirements_file("compiler-stack", root=root, is_windows=True),
            (root / "requirements.txt", False),
        )

    def test_linux_compiler_stack_prefers_existing_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.lock").touch()
            self.assertEqual(
                setup.requirements_file(
                    "compiler-stack", root=root, is_windows=False
                ),
                (root / "requirements.lock", False),
            )

    def test_setup_plan_resolves_profile_actions(self):
        root = Path("root")
        expected = {
            "compiler-stack": (True, True, False),
            "fastflowlm": (False, False, True),
            "npu-e2e": (False, True, False),
            "npu-runtime": (False, False, False),
        }
        for profile_name, actions in expected.items():
            with self.subTest(profile=profile_name):
                plan = setup.build_setup_plan(
                    profile_name,
                    iron_ops_tests=True,
                    root=root,
                    is_windows=True,
                )
                self.assertEqual(
                    (
                        plan.check_system,
                        plan.install_iron_ops,
                        plan.install_fastflowlm_native,
                    ),
                    actions,
                )
                self.assertEqual(
                    plan.install_iron_ops_tests, plan.install_iron_ops
                )


class SetupCommandTest(unittest.TestCase):
    def test_compiler_stack_orders_environment_fetch_and_iron_ops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv = root / ".venv"
            python = venv / "Scripts" / "python.exe"
            uv = venv / "Scripts" / "uv.exe"
            args = SimpleNamespace(
                profile="compiler-stack",
                iron_ops_tests=True,
            )
            events = []

            def run(command, **_kwargs):
                events.append(("run", [str(item) for item in command]))

            with (
                mock.patch.object(paths, "OUT", root / "out"),
                mock.patch.object(paths, "VENV", venv),
                mock.patch.object(setup.environment, "venv_py", return_value=python),
                mock.patch.object(setup.environment, "uv_bin", return_value=uv),
                mock.patch.object(setup.environment, "sh", side_effect=run),
                mock.patch.object(
                    setup.doctor, "check_system", return_value=(True, [])
                ),
                mock.patch.object(
                    setup,
                    "requirements_file",
                    return_value=(root / "requirements.lock", False),
                ),
                mock.patch.object(
                    setup.patches,
                    "cmd_fetch",
                    side_effect=lambda *_args: events.append(("fetch", None)),
                ),
                mock.patch.object(
                    setup,
                    "install_iron_ops",
                    side_effect=lambda *_args, **_kwargs: events.append(
                        ("iron-ops", _kwargs["with_tests"])
                    ),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                setup.cmd_setup(args)

            self.assertEqual(events[0][0], "run")
            self.assertEqual(events[0][1][:3], [
                str(Path(setup.sys.executable)),
                "-m",
                "venv",
            ])
            self.assertEqual(events[-2:], [("fetch", None), ("iron-ops", True)])

    def test_fastflowlm_uses_hashes_and_native_dependency_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(profile="fastflowlm", iron_ops_tests=False)
            python = Path("venv/python")
            uv = Path("venv/uv")
            commands = []
            with (
                mock.patch.object(paths, "OUT", Path(directory) / "out"),
                mock.patch.object(
                    setup.environment, "venv_py", return_value=python
                ),
                mock.patch.object(
                    setup.environment, "uv_bin", return_value=uv
                ),
                mock.patch.object(
                    setup.environment,
                    "sh",
                    side_effect=lambda command, **_kwargs: commands.append(
                        command
                    ),
                ),
                mock.patch.object(
                    setup,
                    "requirements_file",
                    return_value=(
                        Path("requirements-fastflowlm.lock"),
                        True,
                    ),
                ),
                mock.patch.object(setup.patches, "cmd_fetch"),
                mock.patch.object(setup, "install_iron_ops") as install_iron_ops,
                mock.patch.object(
                    setup, "_install_fastflowlm_deps"
                ) as install_native,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                setup.cmd_setup(args)

        self.assertIn("--require-hashes", commands[-1])
        install_iron_ops.assert_not_called()
        install_native.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
