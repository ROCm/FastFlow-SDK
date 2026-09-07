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
    stage_github_release,
    verify_archive,
)
from .power import inhibit_system_sleep
from .profiles import ProfileError, profile_names
from .runner import (
    BENCHMARK_MODES,
    finalize_partial_evidence,
    run_qualification,
)


def _positive_int(value: str) -> int:
    """argparse type: a strictly positive integer."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer, got {value!r}"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {parsed}")
    return parsed


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
    profile_selection.add_argument(
        "--all-models",
        action="store_true",
        help=(
            "run a complete sweep over every model in the installed FastFlowLM "
            "registry (dynamic discovery, no per-file pinning): single-prompt "
            "run, multi-turn, and a benchmark (8k by default, see "
            "--bench-context) for chat models; serve-probe for embed/ASR "
            "models. Implies --keep-going."
        ),
    )
    profile_selection.add_argument(
        "--bench",
        nargs="?",
        const="all",
        default=None,
        choices=BENCHMARK_MODES,
        metavar="MODE",
        help=(
            "run a measurement-only benchmark sweep: the on-NPU decode benchmark "
            "for chat models and the serve-probe for embed/ASR models (no "
            "single-prompt or multi-turn stages). MODE selects the coverage; "
            "'all' (the default when the flag is given bare) sweeps every model "
            "in the installed FastFlowLM registry, while 'dev' runs the same "
            "stages over a curated set of chat tags. Tune the run with "
            "--bench-context and --bench-iterations, and narrow the chosen set "
            "with --models. Implies --keep-going."
        ),
    )
    parser.add_argument(
        "--models",
        default=None,
        metavar="TAG[,TAG...]",
        help=(
            "comma-separated FastFlowLM tags to restrict a --bench or "
            "--all-models sweep to (for example --models=llama3.2:1b,qwen3:0.6b). "
            "With --bench all or --all-models, tags absent from the installed "
            "registry are skipped with a warning. With --bench dev, --models "
            "further restricts the curated chat set: tags not in that set are "
            "skipped even if they are installed. Has no effect without --bench "
            "or --all-models."
        ),
    )
    parser.add_argument(
        "--bench-context",
        choices=("1k", "4k", "8k"),
        default=None,
        help=(
            "context length for the benchmark stage of --bench or --all-models "
            "(default: 1k for --bench, 8k for --all-models; ignored otherwise)"
        ),
    )
    parser.add_argument(
        "--bench-iterations",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "number of decode-benchmark iterations per chat model for a --bench "
            "or --all-models sweep (a positive integer; default 5 for --bench, "
            "and the packaged benchmark config's own count for --all-models)"
        ),
    )
    parser.add_argument(
        "--flm",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "HRX FastFlowLM entrypoint (bin/flm) or its FHS install root, as "
            "produced by `dev.py build fastflowlm`. When set, qualify uses this "
            "build instead of the `rocm-npu-fastflowlm` wheel binary (needed "
            "when the venv wheel lags the in-tree 1.0.5 install). Off by "
            "default so packaged-wheel CI keeps using the installed product."
        ),
    )
    parser.add_argument(
        "--xrt-flm",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "additionally exercise each model through an XRT-backed FastFlowLM "
            "build (only with --bench or --all-models; off by default; Linux "
            "and Windows, requires a host XRT install). PATH is the XRT "
            "FastFlowLM entrypoint (bin/flm.exe on Windows, bin/flm on Linux) "
            "or its FHS install root, produced by `dev.py build fastflowlm-xrt`. "
            "Supplying it is what enables the comparison. --bench widens "
            "decode/serve metric tables; --all-models runs the single-prompt "
            "greedy chat on XRT and records an exact numerical match of Model "
            "RAW Output text. The comparison child is pointed at a discovered "
            "host XRT prefix (XILINX_XRT, platform default install, or PATH) "
            "via XILINX_XRT and the loader path (LD_LIBRARY_PATH on Linux, "
            "PATH on Windows); XRT results are best-effort. Status is PASS, "
            "FAIL, or N/A "
            "(unavailable); metric cells are N/A unless the comparison "
            "passed. XRT never changes the overall pass/fail."
        ),
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
    parser.add_argument(
        "--skip-xadx",
        action="store_true",
        help="skip production-kernel XADX cases (keep prior results when reusing evidence)",
    )
    parser.add_argument(
        "--reuse-evidence",
        type=Path,
        help=(
            "merge this run into an existing evidence directory: keep tests "
            "for models that are not selected and rewrite report.md in place"
        ),
    )
    return parser


def _offline_parser(command: str, *, json_errors: bool) -> argparse.ArgumentParser:
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
        parser.add_argument("--require-status", choices=("pass", "fail", "interrupted"))
    elif command == "stage-release":
        parser.add_argument("--payload-dir", type=Path, required=True)
        parser.add_argument("--aggregate", type=Path, required=True)
        parser.add_argument(
            "--index",
            type=Path,
            help=(
                "qualification-index.json from intake (default: next to --aggregate)"
            ),
        )
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument(
            "--candidate-manifest",
            type=Path,
            help=(
                "canonical candidate manifest (default: payload candidate-manifest.json)"
            ),
        )
    return parser


def _print_summary(summary: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(summary, sort_keys=True))
    elif summary["status"] == "pass":
        staged = summary.get("output") or summary.get("bundle_output") or ""
        if summary.get("file_count") is not None:
            print(
                f"github release upload staged: {staged} "
                f"({summary['file_count']} files)"
            )
        else:
            print(f"qualification passed: {staged}")
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


def _run_offline_command(command: str, argv: list[str], *, json_output: bool) -> int:
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
        elif command == "intake":
            policy = load_qualification_policy(args.policy)
            summary = intake_evidence(
                args.input_dir,
                args.candidate_manifest,
                args.output_dir,
                args.bundle_output,
                require_status=args.require_status,
                policy=policy,
            )
        elif command == "stage-release":
            summary = stage_github_release(
                args.payload_dir,
                args.aggregate,
                args.output_dir,
                index=args.index,
                candidate_manifest=args.candidate_manifest,
            )
        else:
            raise EvidenceError(f"unsupported offline command: {command}")
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
        "--json" in effective_argv or os.environ.get("ROCM_NPU_OUTPUT_FORMAT") == "json"
    )
    if effective_argv and effective_argv[0] in {
        "candidate",
        "pack",
        "verify",
        "intake",
        "stage-release",
    }:
        return _run_offline_command(
            effective_argv[0], effective_argv[1:], json_output=json_output
        )
    if effective_argv and effective_argv[0] == "run":
        effective_argv = effective_argv[1:]
    parser = create_parser(json_errors=json_output)
    if json_output and any(argument in ("--help", "-h") for argument in effective_argv):
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
