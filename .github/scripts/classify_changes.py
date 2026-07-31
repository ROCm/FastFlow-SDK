# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import pathlib
import subprocess
from collections.abc import Iterable


def changed_files_between(base_sha: str, head_sha: str) -> list[str]:
    if not base_sha or not head_sha:
        raise ValueError("pull request base and head SHAs are required")
    return subprocess.check_output(
        ["git", "diff", "--name-only", base_sha, head_sha],
        text=True,
    ).splitlines()


def requires_full_ci(changed_files: Iterable[str]) -> bool:
    paths = list(changed_files)
    return not paths or any(
        pathlib.PurePosixPath(path).suffix.lower() != ".md" for path in paths
    )


def classify_changes(event_name: str, base_sha: str, head_sha: str) -> bool:
    if event_name != "pull_request":
        return True
    return requires_full_ci(changed_files_between(base_sha, head_sha))


def main() -> None:
    run_full_ci = classify_changes(
        os.environ["EVENT_NAME"],
        os.environ.get("BASE_SHA", ""),
        os.environ.get("HEAD_SHA", ""),
    )
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"run_full_ci={str(run_full_ci).lower()}\n")


if __name__ == "__main__":
    main()
