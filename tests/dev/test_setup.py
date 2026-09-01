# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from build_tools import paths
from build_tools.dev import setup

pytestmark = pytest.mark.presubmit


class TestRequirementsFile:
    def test_profile_requirements_are_explicit(self):
        root = Path("root")
        assert setup.requirements_file("fastflowlm", root=root, is_windows=True) == (
            root / "requirements/fastflowlm.lock",
            True,
        )
        assert setup.requirements_file(
            "npu-test-runtime", root=root, is_windows=False
        ) == (
            root / "requirements/npu-test-runtime.lock",
            True,
        )
        assert setup.requirements_file(
            "compiler-stack", root=root, is_windows=True
        ) == (root / "requirements/compiler-stack.txt", False)

    def test_linux_compiler_stack_prefers_existing_lock(self, tmp_path):
        lock = tmp_path / "requirements" / "compiler-stack.lock"
        lock.parent.mkdir()
        lock.touch()
        assert setup.requirements_file(
            "compiler-stack", root=tmp_path, is_windows=False
        ) == (lock, False)

    def test_compiler_lock_preserves_non_registry_sources(self):
        source = (paths.ROOT / "requirements" / "compiler-stack.txt").read_text(
            encoding="utf-8"
        )
        lock = (paths.ROOT / "requirements" / "compiler-stack.lock").read_text(
            encoding="utf-8"
        )
        find_links = {
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("--find-links ")
        }
        assert find_links
        assert find_links.issubset(set(lock.splitlines()))

    @pytest.mark.parametrize(
        ("profile_name", "actions"),
        [
            ("compiler-stack", (True, True, False)),
            ("fastflowlm", (False, False, True)),
            ("npu-e2e", (False, True, False)),
            ("npu-test-runtime", (False, False, False)),
        ],
    )
    def test_setup_plan_resolves_profile_actions(self, profile_name, actions):
        root = Path("root")
        plan = setup.build_setup_plan(
            profile_name,
            iron_ops_tests=True,
            root=root,
            is_windows=True,
        )
        assert (
            plan.check_system,
            plan.install_iron_ops,
            plan.install_fastflowlm_native,
        ) == actions
        assert plan.install_iron_ops_tests == plan.install_iron_ops


class TestSetupCommand:
    def test_compiler_stack_orders_environment_fetch_and_iron_ops(self, tmp_path):
        root = tmp_path
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
            mock.patch.object(setup.doctor, "check_system", return_value=(True, [])),
            mock.patch.object(
                setup,
                "requirements_file",
                return_value=(
                    root / "requirements" / "compiler-stack.lock",
                    False,
                ),
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

        assert events[0][0] == "run"
        assert events[0][1][:3] == [
            str(Path(setup.sys.executable)),
            "-m",
            "venv",
        ]
        assert events[-2:] == [("fetch", None), ("iron-ops", True)]
        format_packages = setup.doctor.format_pip_packages()
        assert any(
            event[0] == "run" and event[1][-2:] == format_packages for event in events
        )

    def test_fastflowlm_uses_hashes_and_native_dependency_setup(self, tmp_path):
        args = SimpleNamespace(profile="fastflowlm", iron_ops_tests=False)
        python = Path("venv/python")
        uv = Path("venv/uv")
        commands = []
        with (
            mock.patch.object(paths, "OUT", tmp_path / "out"),
            mock.patch.object(setup.environment, "venv_py", return_value=python),
            mock.patch.object(setup.environment, "uv_bin", return_value=uv),
            mock.patch.object(
                setup.environment,
                "sh",
                side_effect=lambda command, **_kwargs: commands.append(command),
            ),
            mock.patch.object(
                setup,
                "requirements_file",
                return_value=(
                    Path("requirements/fastflowlm.lock"),
                    True,
                ),
            ),
            mock.patch.object(setup.patches, "cmd_fetch"),
            mock.patch.object(setup, "install_iron_ops") as install_iron_ops,
            mock.patch.object(
                setup.fastflowlm_deps, "install_managed_vcpkg"
            ) as install_native,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            setup.cmd_setup(args)

        assert "--require-hashes" in commands[-2]
        assert commands[-1][-2:] == [
            f"ruff=={setup.doctor.ruff_pinned_version()}",
            f"clang-format=={setup.doctor.clang_format.PINNED_VERSION}",
        ]
        install_iron_ops.assert_not_called()
        install_native.assert_called_once_with(paths.ROOT)
