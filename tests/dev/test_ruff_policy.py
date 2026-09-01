# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Presubmit Python format and lint policy."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

from tests.support.workflows import read_workflow, workflow_job

pytestmark = pytest.mark.presubmit

ROOT = Path(__file__).resolve().parents[2]
FIRST_PARTY_FILES = frozenset({"dev.py", "conftest.py"})
FIRST_PARTY_PREFIXES = (
    "build_tools/",
    "tests/",
    "tools/",
    ".github/scripts/",
)


def test_ruff_config_covers_first_party_python_only():
    config = tomllib.loads((ROOT / "ruff.toml").read_text(encoding="utf-8"))
    assert config["required-version"] == "==0.15.21"
    assert config["target-version"] == "py312"
    assert "projects" in config["extend-exclude"]
    assert config["lint"]["select"] == [
        "E4",
        "E7",
        "E9",
        "F",
        "I",
        "UP",
        "B",
        "RUF100",
    ]
    assert "E402" in config["lint"]["ignore"]
    assert "build_tools" in config["lint"]["isort"]["known-first-party"]
    assert "rocm_npu_qualification" in config["lint"]["isort"]["known-first-party"]
    assert f"ruff=={config['required-version'].lstrip('=')}" in (
        ROOT / "requirements" / "test.txt"
    ).read_text(encoding="utf-8")


def test_presubmit_runs_config_driven_ruff_check_and_format():
    presubmit = workflow_job(read_workflow("ci.yml"), "presubmit")
    assert "requirements/test.lock" in presubmit
    assert "python -m ruff check" in presubmit
    assert "python -m ruff format --check" in presubmit
    assert "--select" not in presubmit
    assert "ruff check --select F821" not in presubmit


@pytest.mark.parametrize(
    "workflow_name",
    (
        "build_rocm_npu_staging_linux.yml",
        "build_rocm_npu_staging_windows.yml",
    ),
)
def test_build_jobs_do_not_rerun_ruff(workflow_name):
    workflow = read_workflow(workflow_name)
    assert "ruff" not in workflow


def test_tracked_python_stays_inside_ruff_first_party_trees():
    tracked = subprocess.check_output(
        ["git", "ls-files", "*.py"], cwd=ROOT, text=True
    ).splitlines()
    stray = [
        path.replace("\\", "/")
        for path in tracked
        if path.replace("\\", "/") not in FIRST_PARTY_FILES
        and not path.replace("\\", "/").startswith(FIRST_PARTY_PREFIXES)
    ]
    assert stray == []
