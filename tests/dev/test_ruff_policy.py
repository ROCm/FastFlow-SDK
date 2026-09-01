# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Presubmit Python format and lint policy."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from build_tools.dev import clang_format, cli, format_tools
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
    assert format_tools.PINNED_RUFF_VERSION == config["required-version"].lstrip("=")


def test_presubmit_runs_config_driven_ruff_via_format():
    presubmit = workflow_job(read_workflow("ci.yml"), "presubmit")
    assert "requirements/test.lock" in presubmit
    assert "python dev.py format --check" in presubmit
    assert "python -m ruff" not in presubmit
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


def test_dev_cli_defaults_ruff_and_format_to_rewrite():
    parser = cli.create_parser()
    assert parser.parse_args(["ruff"]).check is False
    assert parser.parse_args(["ruff", "--check"]).check is True
    assert parser.parse_args(["format"]).check is False
    assert parser.parse_args(["format", "--check"]).check is True


def test_missing_ruff_returns_install_hint(capsys):
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(
            format_tools,
            "resolve_pinned_ruff",
            lambda: (None, "missing"),
        )
        assert format_tools.run_ruff(check=True) == 1
    err = capsys.readouterr().err
    assert "missing" in err
    assert format_tools.install_command() in err


def test_format_runs_ruff_then_clang_format(monkeypatch):
    calls = []
    monkeypatch.setattr(
        format_tools,
        "run_ruff",
        lambda *, check: calls.append(("ruff", check)) or 0,
    )
    monkeypatch.setattr(
        clang_format,
        "run_clang_format",
        lambda *, check: calls.append(("cxx", check)) or 2,
    )
    assert format_tools.cmd_format(SimpleNamespace(check=True)) == 2
    assert calls == [("ruff", True), ("cxx", True)]


def test_pinned_ruff_accepts_first_party_python():
    result = subprocess.run(
        [sys.executable, str(ROOT / "dev.py"), "ruff", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_format_check_accepts_first_party_sources():
    result = subprocess.run(
        [sys.executable, str(ROOT / "dev.py"), "format", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stderr + result.stdout
    assert result.returncode == 0, output
    assert "Python (ruff)" in output
    assert "C/C++ (clang-format)" in output
    assert "files already formatted" in output
    assert (
        f"{len(clang_format.first_party_cxx_files())} C/C++ files already formatted"
        in output
    )
