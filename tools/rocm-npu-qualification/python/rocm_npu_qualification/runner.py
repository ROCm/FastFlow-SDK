# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execute qualification profiles and maintain evidence lifecycle state."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from .benchmark import benchmark_csv_checkpoint as _benchmark_csv_checkpoint
from .commands import run_command, run_interactive_command
from .contracts import (
    COMMAND_REPORT_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    XADX_REPORT_SCHEMA_VERSION,
    xadx_rounds,
)
from .evidence import EvidenceWriter
from .errors import QualificationError
from .fastflowlm import (
    MAX_LENGTH_PATTERN,
    RUNTIME_ERROR_PATTERN,
    extract_raw_outputs,
    fresh_benchmark_csvs,
    require_1k_checkpoint,
    smoke_phrase_matches,
    verify_xclbin_inventory,
)
from .hashing import canonical_json_sha256, sha256_file
from .installation import (
    numeric_runtime_identities as _numeric_runtime_identities,
    normalize_candidate as _normalize_candidate_manifest,
    package_versions as _package_versions,
    validate_protocol as _validate_protocol,
    validate_versions as _validate_versions,
)
from .models import ModelError, fastflow_environment, resolve_model
from .profiles import load_profile, profile_names
from .prompts import materialize_turn
from .xadx import XadxError, run_recipe, validate_xadx_report


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_core_report(
    core: Path,
    command: str,
    *,
    runtime: str,
    writer: EvidenceWriter,
) -> tuple[dict[str, Any], int]:
    argv: list[str | os.PathLike[str]] = [core, "--json", command]
    if runtime:
        argv.extend(("--runtime", runtime))
    result = run_command(
        argv,
        log_path=writer.path(f"logs/tests/core-{command}.log"),
        timeout=300,
        echo=True,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise QualificationError(
            f"rocm-npu {command} did not emit valid JSON"
        ) from error
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != COMMAND_REPORT_SCHEMA_VERSION
        or report.get("command") != command
        or report.get("status") not in {"pass", "fail", "error"}
        or not isinstance(report.get("product"), dict)
        or not isinstance(report["product"].get("version"), str)
    ):
        raise QualificationError(f"rocm-npu {command} emitted an invalid report")
    return report, result.returncode


def _resolve_flm() -> tuple[Path, Path, dict[str, str]]:
    try:
        import rocm_npu_fastflowlm

        path = Path(rocm_npu_fastflowlm.flm_path())
        config = Path(rocm_npu_fastflowlm.config_path())
        xclbins = Path(rocm_npu_fastflowlm.xclbin_path())
        model_info = Path(rocm_npu_fastflowlm.model_info_path())
        package_root = Path(rocm_npu_fastflowlm.package_root())
    except (ImportError, AttributeError, OSError) as error:
        raise QualificationError(
            "rocm-npu-fastflowlm installed-product paths are unavailable"
        ) from error
    if not path.is_file():
        raise QualificationError(f"installed flm command is missing: {path}")
    environment = dict(os.environ)
    environment.setdefault("HRX_GPU_DRIVER", "amdxdna")
    environment["FLM_CONFIG_PATH"] = str(config)
    environment["FLM_XCLBIN_PATH"] = str(xclbins)
    environment["FLM_MODELINFO_PATH"] = str(model_info)
    if os.name == "nt":
        try:
            from rocm_npu_runtime import binary_dir
        except (ImportError, AttributeError) as error:
            raise QualificationError(
                "rocm-npu-runtime binary directory is unavailable"
            ) from error
        runtime_override = environment.get("ROCM_NPU_RUNTIME_PATH")
        override_directory = None
        if runtime_override:
            override_path = Path(runtime_override)
            override_directory = (
                override_path if override_path.is_dir() else override_path.parent
            )
        environment["PATH"] = os.pathsep.join(
            str(path)
            for path in (
                str(package_root / "bin"),
                override_directory,
                str(binary_dir()),
                environment.get("PATH", ""),
            )
            if path
        )
    return path.resolve(), config.resolve(), environment


def _test_model(
    model: dict[str, Any],
    *,
    flm: Path,
    model_cache: Path,
    offline: bool,
    writer: EvidenceWriter,
    environment: dict[str, str],
    prior_tests: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    environment = fastflow_environment(
        model_cache, offline=offline, base_environment=environment
    )
    records: list[dict[str, Any]] = []
    prior_tests = [] if prior_tests is None else prior_tests

    def write_progress() -> None:
        writer.write_json("tests.json", [*prior_tests, *records])

    if model["xclbins"]:
        xclbin_root = Path(environment.get("FLM_XCLBIN_PATH", "")) / "xclbins"
        errors = verify_xclbin_inventory(xclbin_root, model["xclbins"])
        record = {
            "model": model["tag"],
            "name": "production-xclbin-inventory",
            "kind": "xclbin",
            "status": "fail" if errors else "pass",
            "duration_seconds": 0.0,
            "log": "",
            "metrics": {"files_verified": len(model["xclbins"])},
            **({"error": "; ".join(errors)} if errors else {}),
        }
        records.append(record)
        write_progress()
        if errors:
            return records

    for test in model["tests"]:
        if test["kind"] == "bench":
            safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", model["tag"])
            work = writer.path(f"bench/{safe_tag}/{test['name']}")
            work.mkdir(parents=True, exist_ok=True)
            stamp = work / ".started"
            stamp.write_text("", encoding="utf-8")
            started_ns = stamp.stat().st_mtime_ns
            log = writer.path(f"logs/tests/{safe_tag}-{test['name']}.log")
            config_resource = resources.files("rocm_npu_qualification").joinpath(
                test["config"]
            )
            if not config_resource.is_file():
                raise QualificationError(
                    f"benchmark configuration is missing: {test['config']}"
                )
            with resources.as_file(config_resource) as config:
                result = run_command(
                    [flm, "bench", model["tag"], "-i", config],
                    log_path=log,
                    env=environment,
                    timeout=test["timeout_seconds"],
                    cwd=work,
                    echo=True,
                )
            transcript = result.stdout + "\n" + result.stderr
            semantic_error = ""
            checkpoint = None
            csv_path = None
            raw_outputs = extract_raw_outputs(transcript)
            raw = raw_outputs[0] if raw_outputs else ""
            if RUNTIME_ERROR_PATTERN.search(transcript):
                semantic_error = "FastFlowLM emitted an HRX runtime error"
            elif MAX_LENGTH_PATTERN.search(transcript):
                semantic_error = "FastFlowLM reached the generation length limit"
            else:
                try:
                    checkpoint = require_1k_checkpoint(
                        transcript,
                        minimum=test["checkpoint_min"],
                        maximum=test["checkpoint_max"],
                    )
                except ValueError as error:
                    semantic_error = str(error)
            if not semantic_error and len(re.findall(r"[A-Za-z]{2,}", raw)) < 8:
                semantic_error = "FastFlowLM benchmark output is not coherent text"
            if (
                not semantic_error
                and test["expected_regex"]
                and re.search(test["expected_regex"], raw, re.IGNORECASE | re.DOTALL)
                is None
            ):
                semantic_error = "FastFlowLM benchmark output did not match"
            csvs = fresh_benchmark_csvs(work, model["tag"], started_ns=started_ns)
            if not semantic_error and not csvs:
                semantic_error = "FastFlowLM benchmark did not write a fresh CSV"
            csv_row: list[str] = []
            csv_metrics: dict[str, int | float] = {}
            if csvs:
                csv_path = csvs[-1]
                with csv_path.open(
                    newline="", encoding="utf-8", errors="replace"
                ) as stream:
                    csv_rows = list(csv.reader(stream))
                if not semantic_error and checkpoint is not None:
                    try:
                        csv_row, csv_metrics = _benchmark_csv_checkpoint(
                            csv_rows, max(1, round(checkpoint / 1024))
                        )
                    except ValueError as error:
                        semantic_error = str(error)
            passed = result.returncode == 0 and not semantic_error
            records.append(
                {
                    "model": model["tag"],
                    "name": test["name"],
                    "kind": "bench",
                    "status": "pass" if passed else "fail",
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "duration_seconds": result.duration_seconds,
                    "log": writer.relative(log),
                    "turn_count": 1,
                    "process_iteration": 1,
                    "metrics": {
                        "checkpoint_tokens": checkpoint,
                        "csv_checkpoint_row": csv_row,
                        "csv_checkpoint_metrics": csv_metrics,
                    },
                    "artifacts": (
                        [] if csv_path is None else [writer.relative(csv_path)]
                    ),
                    **(
                        {}
                        if passed
                        else {"error": semantic_error or "FastFlowLM benchmark failed"}
                    ),
                }
            )
            write_progress()
            continue

        turns = [materialize_turn(turn) for turn in test["turns"]]
        for iteration in range(1, test["process_iterations"] + 1):
            name = (
                test["name"]
                if test["process_iterations"] == 1
                else f"{test['name']}-{iteration:02d}"
            )
            safe_tag = "".join(
                character if character.isalnum() else "-" for character in model["tag"]
            )
            log = writer.path(f"logs/tests/{safe_tag}-{name}.log")
            input_lines = [
                *test["setup_commands"],
                *(turn["prompt"] for turn in turns),
            ]
            if os.name == "nt":
                result = run_command(
                    [
                        flm,
                        "run",
                        model["tag"],
                        "--ctx-len",
                        str(test["ctx_len"]),
                    ],
                    log_path=log,
                    env=environment,
                    input_text="".join(f"{line}\n" for line in input_lines),
                    timeout=test["timeout_seconds"],
                    echo=True,
                )
            else:
                result = run_interactive_command(
                    [
                        flm,
                        "run",
                        model["tag"],
                        "--ctx-len",
                        str(test["ctx_len"]),
                    ],
                    log_path=log,
                    env=environment,
                    input_lines=input_lines,
                    timeout=test["timeout_seconds"],
                    echo=True,
                )
            transcript = result.stdout + "\n" + result.stderr
            raw_outputs = extract_raw_outputs(transcript)
            semantic_error = ""
            if RUNTIME_ERROR_PATTERN.search(transcript):
                semantic_error = "FastFlowLM emitted an HRX runtime error"
            elif MAX_LENGTH_PATTERN.search(transcript):
                semantic_error = "FastFlowLM reached the generation length limit"
            elif len(raw_outputs) != len(turns):
                semantic_error = (
                    f"FastFlowLM emitted {len(raw_outputs)} responses; "
                    f"expected {len(turns)}"
                )
            else:
                for turn_number, (raw, turn) in enumerate(
                    zip(raw_outputs, turns), start=1
                ):
                    if turn.get("expected") and not smoke_phrase_matches(
                        raw, turn["expected"]
                    ):
                        semantic_error = f"FastFlowLM turn {turn_number} did not match"
                    elif (
                        turn.get("expected_regex")
                        and re.search(turn["expected_regex"], raw, re.IGNORECASE)
                        is None
                    ):
                        semantic_error = (
                            f"FastFlowLM turn {turn_number} did not match regex"
                        )
                    elif not re.search(r"[A-Za-z]{4,}", raw):
                        semantic_error = (
                            f"FastFlowLM turn {turn_number} has no visible text"
                        )
                    if semantic_error:
                        break
            passed = result.returncode == 0 and not semantic_error
            records.append(
                {
                    "model": model["tag"],
                    "name": name,
                    "kind": "run",
                    "status": "pass" if passed else "fail",
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "duration_seconds": result.duration_seconds,
                    "log": writer.relative(log),
                    "turn_count": len(turns),
                    "process_iteration": iteration,
                    "soak_iterations": test["process_iterations"],
                    "metrics": {
                        "responses": len(raw_outputs),
                        "raw_outputs_sha256": canonical_json_sha256(raw_outputs),
                    },
                    "artifacts": [],
                    **(
                        {}
                        if passed
                        else {"error": semantic_error or "FastFlowLM model test failed"}
                    ),
                }
            )
            write_progress()
    return records


def _report_xadx_plan(recipe_path: Path, rounds: dict[str, int]) -> None:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    cases = recipe.get("cases")
    if not isinstance(cases, list):
        return
    print(
        f"qualify: XADX recipe scheduled {len(cases)} cases "
        f"(lifecycle={rounds['lifecycle_rounds']}, "
        f"resident={rounds['resident_rounds']}, "
        f"readback={rounds['readback_rounds']})",
        file=sys.stderr,
        flush=True,
    )
    for case in cases:
        if isinstance(case, dict) and isinstance(case.get("name"), str):
            print(
                f"qualify: test fastflowlm-production-kernels/"
                f"{case['name']}: scheduled",
                file=sys.stderr,
                flush=True,
            )


def _test_xadx(
    writer: EvidenceWriter, *, target_family: str, profile_name: str
) -> list[dict[str, Any]]:
    started = time.monotonic()
    try:
        rounds = xadx_rounds(profile_name)
        asset_resource = resources.files("rocm_npu_qualification").joinpath(
            "xadx_assets"
        )
        if not asset_resource.is_dir():
            raise XadxError("qualification wheel has no embedded XADX assets")
        with resources.as_file(asset_resource) as asset_root:
            recipe_path = asset_root / "recipe.json"
            _report_xadx_plan(recipe_path, rounds)
            report = run_recipe(
                recipe_path,
                asset_root,
                target_family=target_family,
                **rounds,
            )
    except (OSError, ValueError, XadxError) as error:
        failure = {
            "phase": "package-validation",
            "round": None,
            "message": str(error),
        }
        report = {
            "schema_version": XADX_REPORT_SCHEMA_VERSION,
            "report_kind": "package-validation",
            "status": "fail",
            "architecture": "unknown",
            "case_names": ["package-validation"],
            "case_results": [
                {
                    "name": "package-validation",
                    "status": "fail",
                    "dispatches": 0,
                    "readbacks": 0,
                    "validated_bytes": 0,
                    "duration_seconds": max(time.monotonic() - started, 0.0),
                    "mismatch_count": 0,
                    "first_mismatch": None,
                    "failures": [failure],
                }
            ],
            "failures": [{"case": "package-validation", **failure}],
        }
    validate_xadx_report(report)
    writer.write_json("xadx-report.json", report)
    records = []
    case_results = report.get("case_results")
    if not isinstance(case_results, list):
        case_results = [
            {
                "name": name,
                "status": report["status"],
                "dispatches": 0,
                "readbacks": 0,
                "validated_bytes": 0,
                "duration_seconds": 0.0,
                "mismatch_count": 0,
                "first_mismatch": None,
                "failures": list(report.get("failures", [])),
            }
            for name in report.get("case_names", ["package-validation"])
        ]
    for case_result in case_results:
        name = case_result["name"]
        status = case_result["status"]
        failures = case_result["failures"]
        readback_rounds = int(case_result["readbacks"])
        rounds = int(case_result["dispatches"])
        records.append(
            {
                "model": "fastflowlm-production-kernels",
                "name": name,
                "kind": "xadx",
                "status": status,
                "duration_seconds": float(
                    case_result.get("duration_seconds", 0.0)
                ),
                "log": "xadx-report.json",
                "rounds": rounds,
                "metrics": {
                    "architecture": report.get("architecture", "unknown"),
                    "lifecycle_rounds": report.get("lifecycle_rounds", 0),
                    "resident_rounds": report.get("resident_rounds", 0),
                    "readback_rounds": readback_rounds,
                    "validated_bytes": int(
                        case_result.get("validated_bytes", 0)
                    ),
                    "mismatch_count": int(case_result.get("mismatch_count", 0)),
                    "first_mismatch": case_result.get("first_mismatch"),
                },
                "artifacts": ["xadx-report.json"],
                **(
                    {}
                    if status == "pass"
                    else {
                        "error": (
                            report.get("reason")
                            or "; ".join(
                                failure["message"] for failure in failures
                            )
                            or "FastFlowLM XADX qualification failed"
                        )
                    }
                ),
            }
        )
    return records


def _summary(evidence: dict[str, Any], output: Path) -> dict[str, Any]:
    failures = [
        {
            "model": test.get("model"),
            "name": test.get("name"),
            "error": test.get("error"),
            "log": test.get("log"),
        }
        for test in evidence.get("tests", [])
        if test.get("status") != "pass"
    ]
    return {
        "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
        "status": evidence["status"],
        "profile": evidence.get("profile", {}).get("name"),
        "output": str(output.resolve()),
        "models": len(evidence.get("models", [])),
        "tests": len(evidence.get("tests", [])),
        "failed_tests": sum(
            test.get("status") != "pass" for test in evidence.get("tests", [])
        ),
        "failures": failures,
    }


def _report_test_results(tests: list[dict[str, Any]]) -> None:
    for test in tests:
        print(
            f"qualify: test {test.get('model', 'unknown')}/"
            f"{test.get('name', 'unnamed')}: {test.get('status', 'unknown')}",
            file=sys.stderr,
            flush=True,
        )


def _selected_profile_name(args: argparse.Namespace) -> str:
    if getattr(args, "fast", False):
        return "fast"
    return getattr(args, "profile", None) or "default"


def run_qualification(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    core, core_version, _ = _validate_protocol()
    installed = _package_versions()
    _validate_versions(installed, core_version, None)
    if args.list_profiles:
        return 0, {
            "schema_version": COMMAND_REPORT_SCHEMA_VERSION,
            "status": "pass",
            "profiles": profile_names(),
            "packages": installed,
        }

    profile_name = _selected_profile_name(args)
    profile = load_profile(profile_name)
    candidate = None
    if args.candidate_manifest is not None:
        try:
            candidate = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise QualificationError("candidate manifest is invalid") from error
        if not isinstance(candidate, dict):
            raise QualificationError("candidate manifest must be an object")
        candidate = _normalize_candidate_manifest(candidate)

    _validate_versions(installed, core_version, candidate, args.candidate_manifest)
    numeric_runtime = _numeric_runtime_identities()
    writer = EvidenceWriter(args.output)
    writer.write_candidate_manifest(candidate)
    writer.write_json("profile.json", profile)
    started = _now()
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "running",
        "subject": {
            "version": core_version,
            "source_revision": (
                candidate["source_revision"] if candidate is not None else ""
            ),
            "candidate_binding": {
                "present": candidate is not None,
                "manifest_sha256": (
                    sha256_file(writer.path("candidate-manifest.json"))
                    if candidate is not None
                    else None
                ),
            },
            "packages": installed,
            "numeric_runtime": numeric_runtime,
        },
        "observed_platform": {},
        "profile": {
            "name": profile["name"],
            "description": profile["description"],
            "candidate_bound": candidate is not None,
        },
        "models": [],
        "tests": [],
        "started_at": started,
        "finished_at": None,
    }
    writer.write_json("evidence.json", evidence)

    runtime = os.environ.get("ROCM_NPU_RUNTIME_PATH", "")
    core_version_report, version_exit = _run_core_report(
        core, "version", runtime=runtime, writer=writer
    )
    writer.write_json("core-version.json", core_version_report)
    probe, probe_exit = _run_core_report(core, "probe", runtime=runtime, writer=writer)
    writer.write_json("probe.json", probe)
    doctor, doctor_exit = _run_core_report(
        core, "doctor", runtime=runtime, writer=writer
    )
    writer.write_json("doctor.json", doctor)
    self_test, self_test_exit = _run_core_report(
        core, "self-test", runtime=runtime, writer=writer
    )
    writer.write_json("self-test.json", self_test)
    for command, report in (
        ("version", core_version_report),
        ("probe", probe),
        ("doctor", doctor),
        ("self-test", self_test),
    ):
        reported_version = report["product"]["version"]
        if reported_version != core_version:
            raise QualificationError(
                f"rocm-npu {command} reported version {reported_version}, "
                f"expected {core_version}"
            )
    evidence["subject"]["source_revision"] = core_version_report.get("product", {}).get(
        "source_revision", ""
    )
    if candidate is not None:
        candidate_revision = candidate.get("source_revision")
        if (
            not isinstance(candidate_revision, str)
            or candidate_revision != evidence["subject"]["source_revision"]
        ):
            raise QualificationError(
                "candidate manifest source revision does not match the native core"
            )
    evidence["observed_platform"] = probe.get("data", {})
    writer.write_json("evidence.json", evidence)
    native_results = (
        ("version", core_version_report, version_exit),
        ("probe", probe, probe_exit),
        ("doctor", doctor, doctor_exit),
        ("self-test", self_test, self_test_exit),
    )
    for command, report, returncode in native_results:
        passed = returncode == 0 and report["status"] == "pass"
        evidence["tests"].append(
            {
                "model": "native-core",
                "name": command,
                "kind": "native",
                "status": "pass" if passed else "fail",
                "returncode": returncode,
                "duration_seconds": 0,
                "log": f"logs/tests/core-{command}.log",
                "metrics": {"report_status": report["status"]},
                **({} if passed else {"error": f"native core {command} did not pass"}),
            }
        )
    _report_test_results(evidence["tests"][-len(native_results) :])
    if any(
        returncode != 0 or report["status"] != "pass"
        for _, report, returncode in native_results
    ):
        if not args.keep_going:
            evidence["status"] = "fail"
            evidence["finished_at"] = _now()
            writer.write_json("models.json", evidence["models"])
            writer.write_json("tests.json", evidence["tests"])
            writer.write_junit(evidence["tests"])
            writer.write_json("evidence.json", evidence)
            writer.finalize()
            return 1, _summary(evidence, args.output)

    xadx_tests = _test_xadx(
        writer,
        target_family=str(evidence["observed_platform"].get("target_family", "")),
        profile_name=profile["name"],
    )
    evidence["tests"].extend(xadx_tests)
    _report_test_results(xadx_tests)
    writer.write_json("tests.json", evidence["tests"])
    writer.write_json("evidence.json", evidence)
    if any(test["status"] != "pass" for test in xadx_tests) and not args.keep_going:
        evidence["status"] = "fail"
        evidence["finished_at"] = _now()
        writer.write_json("models.json", evidence["models"])
        writer.write_json("tests.json", evidence["tests"])
        writer.write_junit(evidence["tests"])
        writer.write_json("evidence.json", evidence)
        writer.finalize()
        return 1, _summary(evidence, args.output)

    flm, flm_config, flm_environment = _resolve_flm()
    version_result = run_command(
        [flm, "--version"],
        log_path=writer.path("logs/tests/flm-version.log"),
        env=flm_environment,
        timeout=30,
        echo=True,
    )
    version_output = (version_result.stdout or version_result.stderr).strip()
    # FastFlowLM 0.9.45 reports its version correctly but returns 1 on Windows.
    # Validate the installed-product output instead of accepting/rejecting its
    # nonstandard status code in isolation.
    if re.fullmatch(r"FLM v[0-9]+(?:\.[0-9]+)+", version_output) is None:
        raise QualificationError(
            "installed flm --version did not emit a valid version "
            f"(exit {version_result.returncode})"
        )
    evidence["subject"]["flm"] = {
        "package_version": installed["rocm-npu-fastflowlm"],
        "command": str(flm),
        "version_output": version_output,
    }
    writer.write_json("evidence.json", evidence)

    for model in profile["models"]:
        safe_tag = "".join(
            character if character.isalnum() else "-" for character in model["tag"]
        )
        pull_log = writer.path(f"logs/pull/{safe_tag}.log")
        print(
            f"qualify: resolving model {model['tag']} "
            "(missing files will be pulled)",
            file=sys.stderr,
            flush=True,
        )
        try:
            model_record, _ = resolve_model(
                model,
                flm=flm,
                config_path=flm_config,
                model_cache=args.model_cache,
                pull_log=pull_log,
                offline=args.offline,
                no_pull=args.no_pull,
                environment=flm_environment,
                echo=True,
            )
            evidence["models"].append(model_record)
            writer.write_json("models.json", evidence["models"])
            writer.write_json("evidence.json", evidence)
            source = "downloaded" if model_record["pulled"] else "cached"
            print(
                f"qualify: resolved model {model['tag']} ({source})",
                file=sys.stderr,
                flush=True,
            )
        except (ModelError, OSError) as error:
            print(
                f"qualify: model resolution failed for {model['tag']}: {error}",
                file=sys.stderr,
                flush=True,
            )
            if not pull_log.is_file():
                pull_log.write_text(
                    "Model resolution failed before a pull command log was "
                    f"created.\n\n{error}\n",
                    encoding="utf-8",
                )
            evidence["models"].append(
                {
                    "tag": model["tag"],
                    "identity_policy": model["identity_policy"],
                    "source": model["source"],
                    "revision": model["revision"],
                    "status": "fail",
                    "error": str(error),
                    "files": [],
                }
            )
            evidence["tests"].append(
                {
                    "model": model["tag"],
                    "name": "model-resolution",
                    "kind": "model-resolution",
                    "status": "fail",
                    "duration_seconds": 0,
                    "log": f"logs/pull/{safe_tag}.log",
                    "error": str(error),
                }
            )
            _report_test_results([evidence["tests"][-1]])
            failed = True
        else:
            try:
                tests = _test_model(
                    model,
                    flm=flm,
                    model_cache=args.model_cache,
                    offline=args.offline,
                    writer=writer,
                    environment=flm_environment,
                    prior_tests=evidence["tests"],
                )
            except OSError as error:
                tests = [
                    {
                        "model": model["tag"],
                        "name": "model-execution",
                        "kind": "qualification",
                        "status": "fail",
                        "duration_seconds": 0,
                        "log": "",
                        "error": str(error),
                    }
                ]
            evidence["tests"].extend(tests)
            _report_test_results(tests)
            writer.write_json("tests.json", evidence["tests"])
            failed = any(test["status"] != "pass" for test in tests)
        writer.write_json("models.json", evidence["models"])
        writer.write_json("tests.json", evidence["tests"])
        writer.write_json("evidence.json", evidence)
        if failed and not args.keep_going:
            break

    evidence["status"] = (
        "pass"
        if evidence["tests"]
        and all(test["status"] == "pass" for test in evidence["tests"])
        else "fail"
    )
    evidence["finished_at"] = _now()
    writer.write_json("models.json", evidence["models"])
    writer.write_json("tests.json", evidence["tests"])
    writer.write_junit(evidence["tests"])
    writer.write_json("evidence.json", evidence)
    writer.finalize()
    return (0 if evidence["status"] == "pass" else 1), _summary(evidence, args.output)


def finalize_partial_evidence(
    output: Path, *, status: str, error: str
) -> dict[str, Any] | None:
    evidence_path = output / "evidence.json"
    if not evidence_path.is_file():
        return None
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(evidence, dict):
        return None
    for key in ("models", "tests"):
        sidecar = output / f"{key}.json"
        if not sidecar.is_file():
            continue
        try:
            value = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, list):
            evidence[key] = value
    evidence["status"] = status
    evidence["finished_at"] = _now()
    tests = evidence.setdefault("tests", [])
    tests.append(
        {
            "model": "qualification",
            "name": "interruption" if status == "interrupted" else "internal-error",
            "kind": "qualification",
            "status": "fail",
            "duration_seconds": 0,
            "log": "",
            "error": error,
        }
    )
    writer = EvidenceWriter(output, allow_existing=True)
    writer.write_json("models.json", evidence.get("models", []))
    writer.write_json("tests.json", tests)
    writer.write_junit(tests)
    writer.write_json("evidence.json", evidence)
    writer.finalize()
    return _summary(evidence, output)
