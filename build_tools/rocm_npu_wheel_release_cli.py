# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Resolve release inputs and prepare validated ROCm NPU wheel payloads."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_tools.rocm_npu_wheel_release import (
    prepare_runtime_fastflowlm_release_payload,
    resolve_release_inputs,
)


def _source_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve(args: argparse.Namespace) -> int:
    release = resolve_release_inputs(
        args.ref or os.environ.get("GITHUB_REF_NAME", ""),
        args.package_version,
        args.run_number,
        publish_target=args.publish_target,
    )
    with args.github_output.open("a", encoding="utf-8") as output:
        for name, value in release.items():
            output.write(f"{name}={value}\n")
    for name, value in release.items():
        print(f"{name}={value}")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    release_directory = prepare_runtime_fastflowlm_release_payload(
        args.dir,
        args.output_dir,
        version=args.version,
        release_name=args.release_name,
        tag_name=args.tag_name,
        source_ref=args.source_ref,
        source_sha=args.source_sha or _source_sha(),
        expected_wheel_tags=tuple(args.expected_wheel_tag),
        github_run_id=args.github_run_id,
        github_run_attempt=args.github_run_attempt,
    )
    print(f"prepare ROCm NPU wheel release: ok -> {release_directory}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--ref", default="")
    resolve.add_argument("--package-version", default="")
    resolve.add_argument(
        "--publish-target",
        choices=("none", "github-prerelease"),
        default="none",
    )
    resolve.add_argument(
        "--run-number",
        type=int,
        default=int(os.environ.get("GITHUB_RUN_NUMBER", "0")),
    )
    resolve.add_argument(
        "--github-output",
        type=Path,
        default=Path(os.environ.get("GITHUB_OUTPUT", "release-outputs.txt")),
    )
    resolve.set_defaults(handler=_resolve)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--version", required=True)
    prepare.add_argument("--release-name", required=True)
    prepare.add_argument("--tag-name", required=True)
    prepare.add_argument("--source-ref", required=True)
    prepare.add_argument("--source-sha")
    prepare.add_argument(
        "--expected-wheel-tag",
        action="append",
        default=[],
        help="platform tag requiring one wheel per distribution; may be repeated",
    )
    prepare.add_argument("--github-run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    prepare.add_argument(
        "--github-run-attempt",
        default=os.environ.get("GITHUB_RUN_ATTEMPT", ""),
    )
    prepare.set_defaults(handler=_prepare)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
