# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Presubmit C/C++ format policy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from build_tools.dev import clang_format, cli
from tests.support.workflows import read_workflow, workflow_job

pytestmark = pytest.mark.presubmit

ROOT = Path(__file__).resolve().parents[2]
VENDOR_PREFIXES = ("projects/", "third_party/")
CXX_GLOBS = ("*.c", "*.cc", "*.cpp", "*.h", "*.hpp")


def test_clang_format_config_uses_google_style():
    config = (ROOT / ".clang-format").read_text(encoding="utf-8")
    assert "BasedOnStyle: Google" in config
    assert "SortIncludes: Never" in config


def test_presubmit_runs_pinned_clang_format_check():
    presubmit = workflow_job(read_workflow("ci.yml"), "presubmit")
    assert "requirements/test.lock" in presubmit
    assert "python dev.py format --check" in presubmit
    assert "python dev.py clang-format --check" not in presubmit
    assert f"clang-format=={clang_format.PINNED_VERSION}" in (
        ROOT / "requirements" / "test.txt"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "workflow_name",
    (
        "build_rocm_npu_staging_linux.yml",
        "build_rocm_npu_staging_windows.yml",
    ),
)
def test_build_jobs_do_not_rerun_clang_format(workflow_name):
    workflow = read_workflow(workflow_name)
    assert "clang-format" not in workflow


def test_tracked_cxx_stays_inside_clang_format_first_party_trees():
    tracked = subprocess.check_output(
        ["git", "ls-files", *CXX_GLOBS], cwd=ROOT, text=True
    ).splitlines()
    stray = [
        path.replace("\\", "/")
        for path in tracked
        if not path.replace("\\", "/").startswith(VENDOR_PREFIXES)
        and not path.replace("\\", "/").startswith(clang_format.FIRST_PARTY_PREFIXES)
    ]
    assert stray == []


def test_dev_cli_defaults_clang_format_to_rewrite():
    parser = cli.create_parser()
    assert parser.parse_args(["clang-format"]).check is False
    assert parser.parse_args(["clang-format", "--check"]).check is True


def test_missing_clang_format_returns_install_hint(capsys):
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(
            clang_format,
            "resolve_pinned_executable",
            lambda: (None, "missing"),
        )
        assert clang_format.run_clang_format(check=True) == 1
    err = capsys.readouterr().err
    assert "missing" in err
    assert clang_format.install_command() in err


def test_pinned_clang_format_accepts_first_party_cxx():
    result = subprocess.run(
        [sys.executable, str(ROOT / "dev.py"), "clang-format", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stderr + result.stdout
    assert result.returncode == 0, output
    assert (
        f"{len(clang_format.first_party_cxx_files())} C/C++ files already formatted"
        in output
    )


def test_contributing_documents_pinned_format_tools():
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "requirements/test.lock" in text
    assert "python3 dev.py format --check" in text
    assert "python3 -m ruff" not in text
    assert "python3 dev.py clang-format --check" not in text
