# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse qualification commands and translate outcomes to console reports."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from .candidate import CandidateError, write_candidate_manifest
from .contracts import COMMAND_REPORT_SCHEMA_VERSION
from .errors import QualificationError, QualificationTerminated
from .offline import (
    CandidateMismatchError,
    EvidenceError,
    IntakePolicyError,
    intake_evidence,
    load_qualification_policy,
    pack_evidence,
    verify_archive,
)
from .profiles import ProfileError, profile_names
from .power import inhibit_system_sleep
from .runner import finalize_partial_evidence, run_qualification


class QualificationArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, json_errors: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_errors = json_errors

    def error(self, message: str) -> None:
        if self.json_errors:
            print(
                json.dumps(
                    {
                        "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
                        "status": "error",
                        "error": message,
                    },
                    sort_keys=True,
                )
            )
            raise SystemExit(2)
        super().error(message)


def create_parser(*, json_errors: bool = False) -> argparse.ArgumentParser:
    parser = QualificationArgumentParser(
        prog="rocm-npu qualify", json_errors=json_errors
    )
    profile_selection = parser.add_mutually_exclusive_group()
    profile_selection.add_argument(
        "--fast",
        action="store_true",
        help="run the bounded CI parity profile",
    )
    profile_selection.add_argument(
        "--profile",
        choices=profile_names(),
        help="select a packaged profile (default: default)",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=Path("qualification-evidence"),
    )
    parser.add_argument("--bundle-output", type=Path)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path(os.environ.get("FLM_MODEL_PATH", Path.home() / ".cache" / "flm")),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--list-profiles", action="store_true")
    return parser


def _offline_parser(
    command: str, *, json_errors: bool
) -> argparse.ArgumentParser:
    parser = QualificationArgumentParser(
        prog=f"rocm-npu qualify {command}", json_errors=json_errors
    )
    parser.add_argument("--json", action="store_true")
    if command == "candidate":
        parser.add_argument("--wheel-dir", action="append", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--expected-source-revision")
    elif command == "pack":
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
    elif command == "verify":
        parser.add_argument("--input", type=Path, required=True)
        parser.add_argument("--candidate-manifest", type=Path)
    elif command == "intake":
        parser.add_argument("--input-dir", type=Path, required=True)
        parser.add_argument("--candidate-manifest", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--bundle-output", type=Path, required=True)
        parser.add_argument(
            "--policy",
            required=True,
            help="packaged policy name (for example 'release') or JSON path",
        )
        parser.add_argument(
            "--require-status", choices=("pass", "fail", "interrupted")
        )
    return parser


def _print_summary(summary: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(summary, sort_keys=True))
    elif summary["status"] == "pass":
        print(f"qualification passed: {summary.get('output', '')}")
    else:
        print(
            f"qualification {summary['status']}: {summary.get('error', '')}",
            file=sys.stderr,
        )


def _error_summary(error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
        "status": "error",
        "error": str(error),
    }


def _run_offline_command(
    command: str, argv: list[str], *, json_output: bool
) -> int:
    parser = _offline_parser(command, json_errors=json_output)
    if json_output and any(argument in ("--help", "-h") for argument in argv):
        _print_summary(
            {
                "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
                "status": "pass",
                "help": parser.format_help(),
            },
            json_output=True,
        )
        return 0
    args = parser.parse_args(argv)
    try:
        if command == "candidate":
            output = write_candidate_manifest(
                args.wheel_dir,
                args.output,
                expected_source_revision=args.expected_source_revision,
            )
            candidate = json.loads(output.read_text(encoding="utf-8"))
            summary = {
                "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
                "status": "pass",
                "output": str(output.resolve()),
                "version": candidate["version"],
                "source_revision": candidate["source_revision"],
                "packages": len(candidate["packages"]),
            }
        elif command == "pack":
            summary = pack_evidence(args.input, args.output)
        elif command == "verify":
            summary = verify_archive(
                args.input, candidate_manifest=args.candidate_manifest
            )
        else:
            policy = load_qualification_policy(args.policy)
            summary = intake_evidence(
                args.input_dir,
                args.candidate_manifest,
                args.output_dir,
                args.bundle_output,
                require_status=args.require_status,
                policy=policy,
            )
        exit_code = 0
    except IntakePolicyError as error:
        exit_code = 5
        summary = _error_summary(error)
    except CandidateMismatchError as error:
        exit_code = 4
        summary = _error_summary(error)
    except (CandidateError, EvidenceError, OSError) as error:
        exit_code = 3
        summary = _error_summary(error)
    _print_summary(summary, json_output=json_output)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    json_output = bool(
        "--json" in effective_argv
        or os.environ.get("ROCM_NPU_OUTPUT_FORMAT") == "json"
    )
    if effective_argv and effective_argv[0] in {
        "candidate",
        "pack",
        "verify",
        "intake",
    }:
        return _run_offline_command(
            effective_argv[0], effective_argv[1:], json_output=json_output
        )
    if effective_argv and effective_argv[0] == "run":
        effective_argv = effective_argv[1:]
    parser = create_parser(json_errors=json_output)
    if json_output and any(
        argument in ("--help", "-h") for argument in effective_argv
    ):
        print(
            json.dumps(
                {
                    "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
                    "status": "pass",
                    "help": parser.format_help(),
                },
                sort_keys=True,
            )
        )
        return 0
    args = parser.parse_args(effective_argv)
    termination_signals = [signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        termination_signals.append(signal.SIGBREAK)
    previous_handlers = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in termination_signals
    }

    def terminate(signal_number, _frame):
        raise QualificationTerminated(signal_number)

    for signal_number in termination_signals:
        signal.signal(signal_number, terminate)
    try:
        with inhibit_system_sleep():
            exit_code, summary = run_qualification(args)
    except QualificationTerminated as terminated:
        exit_code = 128 + int(terminated.args[0])
        summary = finalize_partial_evidence(
            args.output, status="interrupted", error="qualification terminated"
        ) or {
            "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
            "status": "interrupted",
            "error": "qualification terminated",
        }
    except KeyboardInterrupt:
        exit_code = 130
        summary = finalize_partial_evidence(
            args.output, status="interrupted", error="qualification interrupted"
        ) or {
            "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
            "status": "interrupted",
            "error": "qualification interrupted",
        }
    except (QualificationError, ProfileError, OSError) as error:
        exit_code = 2
        summary = finalize_partial_evidence(
            args.output, status="fail", error=str(error)
        ) or _error_summary(error)
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)
    if args.bundle_output is not None and (args.output / "evidence.json").is_file():
        try:
            bundle = pack_evidence(args.output, args.bundle_output)
            summary["bundle_output"] = bundle["output"]
            summary["bundle_sha256"] = bundle["sha256"]
        except (EvidenceError, OSError) as error:
            exit_code = 2
            summary = {
                **_error_summary(
                    RuntimeError(f"could not seal evidence bundle: {error}")
                ),
                "output": str(args.output.resolve()),
            }
    _print_summary(summary, json_output=json_output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
