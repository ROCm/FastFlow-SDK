# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression checks for the repository-wide pytest migration."""

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.presubmit

ROOT = Path(__file__).resolve().parents[2]


def test_suite_does_not_use_unittest_testcase_apis():
    violations = []
    test_roots = (
        ROOT / "tests",
        ROOT / "tools/rocm-npu-qualification/tests",
    )
    for test_root in test_roots:
        for path in sorted(test_root.rglob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative_path = path.relative_to(ROOT)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "unittest":
                            violations.append(
                                f"{relative_path}:{node.lineno}: import unittest"
                            )
                if not isinstance(node, ast.Attribute):
                    continue
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and (
                        node.attr.startswith("assert")
                        or node.attr in {"skipTest", "subTest"}
                    )
                ):
                    violations.append(
                        f"{relative_path}:{node.lineno}: self.{node.attr}"
                    )
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "unittest"
                    and node.attr
                    in {"TestCase", "main", "skip", "skipIf", "skipUnless"}
                ):
                    violations.append(
                        f"{relative_path}:{node.lineno}: unittest.{node.attr}"
                    )

    assert not violations, "legacy unittest APIs remain:\n  " + "\n  ".join(violations)


def test_presubmit_workflow_uses_marker_selection():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python -m pytest -m presubmit" in workflow
    pytest_config = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "tools/rocm-npu-qualification/tests" in pytest_config


def test_test_profile_covers_presubmit_runtime_dependencies():
    source = (ROOT / "requirements/test.txt").read_text(encoding="utf-8")
    entries = {
        line.partition("#")[0].strip()
        for line in source.splitlines()
        if line.partition("#")[0].strip()
    }
    assert entries == {
        "jsonschema==4.26.0",
            "ml-dtypes==0.5.4",
            "numpy==2.4.6",
        "pytest==8.4.2",
        "zstandard==0.25.0",
    }


@pytest.mark.parametrize(
    "requirements",
    ("requirements/compiler-stack.txt", "requirements/fastflowlm.txt"),
)
def test_build_requirements_do_not_include_test_tools(requirements):
    source = (ROOT / requirements).read_text(encoding="utf-8")
    entries = [
        line.partition("#")[0].strip()
        for line in source.splitlines()
        if line.partition("#")[0].strip()
    ]
    assert not any("requirements/test" in entry for entry in entries)
    assert not any(entry.startswith("pytest") for entry in entries)


@pytest.mark.parametrize(
    ("workflow_path", "python"),
    (
        (".github/workflows/build_rocm_npu_staging_linux.yml", ".venv/bin/python"),
        (
            ".github/workflows/build_rocm_npu_staging_windows.yml",
            ".venv/Scripts/python.exe",
        ),
    ),
)
def test_compiler_workflows_install_runner_and_use_marker_selection(
    workflow_path, python
):
    workflow = (ROOT / workflow_path).read_text(encoding="utf-8")
    assert f"{python} -m pip install" in workflow
    assert "-r requirements/test.lock" in workflow
    assert "hashFiles('requirements/**/*.txt', 'requirements/**/*.lock'" in workflow
    assert f'{python} -m pytest -m "compiler_stack or presubmit"' in workflow


@pytest.mark.parametrize(
    ("workflow_path", "python"),
    (
        (".github/workflows/fastflowlm_hrx_linux.yml", ".venv/bin/python"),
        (
            ".github/workflows/fastflowlm_hrx_windows.yml",
            ".venv/Scripts/python.exe",
        ),
    ),
)
def test_fastflowlm_workflows_install_runner_and_use_marker_selection(
    workflow_path, python
):
    workflow = (ROOT / workflow_path).read_text(encoding="utf-8")
    assert f"{python} -m pip install" in workflow
    assert "-r requirements/test.lock" in workflow
    assert f"{python} -m pytest -m fastflowlm" in workflow
