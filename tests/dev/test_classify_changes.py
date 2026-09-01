# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / ".github" / "scripts" / "classify_changes.py"
SPEC = importlib.util.spec_from_file_location("classify_changes", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
classify_changes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(classify_changes)

pytestmark = pytest.mark.presubmit


@pytest.mark.parametrize(
    ("changed_files", "expected"),
    (
        (["README.md", "docs/release-notes.MD"], False),
        (["README.md", ".github/workflows/ci.yml"], True),
        ([], True),
    ),
)
def test_full_ci_classification(changed_files, expected):
    assert classify_changes.requires_full_ci(changed_files) is expected


def test_non_pull_request_events_run_full_ci_without_diff():
    with mock.patch.object(
        classify_changes, "changed_files_between"
    ) as changed_files_between:
        assert classify_changes.classify_changes("push", "", "")
    changed_files_between.assert_not_called()


def test_pull_request_uses_base_to_head_diff():
    with mock.patch.object(
        classify_changes,
        "changed_files_between",
        return_value=["README.md"],
    ) as changed_files_between:
        assert not classify_changes.classify_changes("pull_request", "base", "head")
    changed_files_between.assert_called_once_with("base", "head")
