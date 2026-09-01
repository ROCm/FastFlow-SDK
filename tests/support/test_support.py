# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from tests.support.modules import load_module
from tests.support.wheels import make_package_tree
from tests.support.workflows import workflow_job

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]


def test_workflow_job_returns_only_requested_top_level_job():
    workflow = (
        "jobs:\n  first:\n    runs-on: linux\n  second-job:\n    runs-on: windows\n"
    )

    assert workflow_job(workflow, "first") == ("  first:\n    runs-on: linux\n")
    assert workflow_job(workflow, "second-job") == (
        "  second-job:\n    runs-on: windows\n"
    )
    with pytest.raises(AssertionError, match="workflow job is missing: third"):
        workflow_job(workflow, "third")


def test_package_tree_runs_optional_initializer(tmp_path):
    initialized = []
    tree = make_package_tree(
        tmp_path,
        "example_package",
        initialize=lambda package: initialized.append(package),
    )

    package = tree / "example_package"
    assert (package / "__init__.py").read_text(encoding="utf-8") == ""
    assert initialized == [package]


def test_load_module_executes_repository_style_script(tmp_path):
    script = tmp_path / "helper.py"
    script.write_text("VALUE = 42\n", encoding="utf-8")

    assert load_module("support_test_helper", script).VALUE == 42
