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
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from .benchmark import benchmark_csv_checkpoint as _benchmark_csv_checkpoint
from .commands import run_command, run_interactive_command, serve_probe
from .contracts import (
    COMMAND_REPORT_SCHEMA_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    XADX_REPORT_SCHEMA_VERSION,
    xadx_rounds,
)
from .evidence import EvidenceWriter
from .errors import QualificationError
from .fastflowlm import (
    BENCHMARK_PASS_SENTINEL_PATTERN,
    MAX_LENGTH_PATTERN,
    extract_raw_outputs,
    fresh_benchmark_csvs,
    has_runtime_error,
    require_benchmark_schedule,
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
from .models import (
    ModelError,
    enumerate_registry_models,
    fastflow_environment,
    resolve_model,
    resolve_model_dynamic,
)
from .profiles import load_profile, profile_names
from .prompts import materialize_turn
from .report import build_markdown_report
from .xadx import XadxError, run_recipe, validate_xadx_report


# Canonical complete-sweep prompts (mirrors the HRX experiment harness). The
# 2+2 turn is left open-ended: forcing "one short sentence" pushes some small
# models (e.g. the translation-tuned lfm2-trans:2.6b) into a terse, wrong reply
# like "2." at greedy decoding, whereas given room to answer they reliably reach
# "4". A numeric-only answer no longer trips the coherence gate because the
# expected_regex below asserts "4" directly (see the per-turn checks).
_SWEEP_SINGLE_PROMPT = "What is the capital of France? Answer in one short sentence."
_SWEEP_MULTI_TURNS: tuple[dict[str, Any], ...] = (
    {"prompt": "What is the capital of France? Answer in one short sentence."},
    {"prompt": "Name three primary colors."},
    {
        "prompt": "What is 2+2?",
        "expected_regex": r"\b4\b|four",
    },
    {"prompt": "Write one short sentence about the ocean."},
    {"prompt": "What is the largest planet in our solar system?"},
)
# Greedy chat setup shared with the curated profiles: pin top-k and temperature
# to make decoding deterministic. The sweep previously set only topk, which left
# temperature and generation length model-default and made the arithmetic turn
# more flake-prone. The single-prompt/arithmetic turns are terse, so they keep a
# tight generation cap; the open multi-turn prompts (naming three colors,
# describing the ocean, largest planet) need room for a full answer, so they use
# a larger cap -- gen-lim 16 truncated them mid-sentence. This mirrors the
# curated profiles, which likewise raise gen-lim for their less-constrained
# turns.
_SWEEP_GREEDY_SETUP: tuple[str, ...] = (
    "/set topk 1",
    "/set temp 0",
    "/set gen-lim 16",
)
_SWEEP_GREEDY_SETUP_MULTI: tuple[str, ...] = (
    "/set topk 1",
    "/set temp 0",
    "/set gen-lim 64",
)
# Complete-sweep benchmark stages, selectable via ``--bench-context``. Each entry
# fixes the packaged config, the evidence test name, the transcript checkpoint
# band that isolates the target stage (FLM runs 1k, 2k, 4k, ... stages up to
# ``max_length``), and a per-turn timeout multiplier that grows with context.
_SWEEP_BENCH_CONFIGS: dict[str, dict[str, Any]] = {
    "1k": {
        "config": "bench/bench_1k_smoke.json",
        "name": "context-1k",
        "checkpoint_min": 950,
        # Chat-template overhead is additive: gpt-oss's harmony template
        # checkpoints a 1k prompt at ~1035, which is the only overflow observed
        # so far. The same 128-token slack is applied to every band so a later
        # template cannot miss the 4k/8k ceilings the same way.
        "checkpoint_max": 1152,
        "timeout_factor": 4,
    },
    "4k": {
        "config": "bench/bench_4k_smoke.json",
        "name": "context-4k",
        "checkpoint_min": 3800,
        "checkpoint_max": 4224,
        "timeout_factor": 6,
    },
    "8k": {
        "config": "bench/bench_8k_smoke.json",
        "name": "context-8k",
        "checkpoint_min": 7600,
        "checkpoint_max": 8320,
        "timeout_factor": 8,
    },
}
_SWEEP_BENCH_DEFAULT = "8k"

# --bench is a measurement-only sweep: chat models run the on-NPU decode
# benchmark, embed/ASR models run the serve-probe (their encode-tok/s / RTF / WER
# measurements). Unlike --all-models it omits the single-prompt and multi-turn
# correctness stages. MODE selects coverage ('all' sweeps the whole installed
# registry; 'dev' a curated chat subset); --bench-context / --bench-iterations
# tune the run and an optional --models filter narrows the chosen set.
BENCHMARK_MODES: tuple[str, ...] = ("all", "dev")
_BENCHMARK_MODES = BENCHMARK_MODES
_BENCHMARK_DEFAULT_CONTEXT = "1k"
_BENCHMARK_DEFAULT_ITERATIONS = 5

# --bench dev is the fast development variant of --bench all: instead of the
# whole registry it runs these curated chat tags (including the large gpt-oss
# and qwen3.6-moe MoE; never embed/ASR). It shares --bench's test synthesis,
# tuning flags, XRT comparison, and --models filter -- --models further
# restricts this list, and the only speedup is fewer FastFlowLM tags.
_BENCHMARK_MODEL_TAGS: tuple[str, ...] = (
    "llama3.2:1b",
    "qwen3:0.6b",
    "qwen3.5:0.8b",
    "qwen3vl-it:4b",
    "gemma3:1b",
    "gemma4-it:e2b",
    "lfm2:1.2b",
    "phi4-mini-it:4b",
    "nanbeige4.1:3b",
    "qwen2.5-it:3b",
    "qwen2.5vl-it:3b",
    "gpt-oss:20b",
    "qwen3.6-moe:35b-a3b",
)


def _benchmark_tests_for(
    model: dict[str, Any], *, bench_context: str, iterations: int
) -> list[dict[str, Any]]:
    """Synthesize the measurement stage for a --bench model.

    Chat models run only the on-NPU decode benchmark (no single-prompt or
    multi-turn correctness stages): the per-turn timeout scales with model
    footprint, the benchmark context length, and the iteration count so a large
    MoE running many iterations does not spuriously time out. Embed/ASR models
    run the serve-probe instead, since the decode benchmark does not apply to
    them; their backend-comparable measurements (encode tok/s, RTF, WER) are
    recorded from that probe.
    """
    kind = model.get("kind", "chat")
    footprint = float(model.get("footprint", 0) or 0)
    boot, turn = _sweep_timeouts(footprint)
    if kind in ("embed", "asr"):
        return [
            {
                "name": "serve-probe",
                "kind": "serve",
                "probe_kind": kind,
                "timeout_seconds": min(7200, max(300, boot)),
            }
        ]
    bench = _SWEEP_BENCH_CONFIGS.get(
        bench_context, _SWEEP_BENCH_CONFIGS[_SWEEP_BENCH_DEFAULT]
    )
    bench_timeout = min(
        7200, max(300, turn * bench["timeout_factor"] * iterations)
    )
    return [
        {
            "name": bench["name"],
            "kind": "bench",
            "config": bench["config"],
            "iterations": iterations,
            "timeout_seconds": bench_timeout,
            "checkpoint_min": bench["checkpoint_min"],
            "checkpoint_max": bench["checkpoint_max"],
            "expected_regex": "",
        }
    ]


def _filter_models_by_tags(
    discovered: list[dict[str, Any]],
    tags: list[str],
    *,
    pool: str = "installed FLM registry",
) -> list[dict[str, Any]]:
    """Restrict ``discovered`` to an explicit ``--models`` list.

    ``discovered`` is the set ``--models`` filters: the installed registry for
    ``--bench all``, or the curated chat subset for ``--bench dev``. Preserves
    the requested order and warns (rather than fails) about any tag missing
    from that pool, so a stale or non-curated tag still benchmarks the rest.
    Raises when none of the requested tags are present.
    """
    by_tag = {model["tag"]: model for model in discovered}
    selected = [by_tag[tag] for tag in tags if tag in by_tag]
    missing = [tag for tag in tags if tag not in by_tag]
    if missing:
        print(
            f"qualify: --models skipping tags absent from the {pool}: "
            f"{', '.join(missing)}",
            file=sys.stderr,
            flush=True,
        )
    if not selected:
        raise QualificationError(
            f"none of the --models tags are present in the {pool}"
        )
    return selected


def _select_benchmark_models(
    discovered: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pick the curated --bench dev models from the discovered registry set.

    Preserves the curated order and warns (rather than fails) about any curated
    tag missing from the installed registry, so a registry that has dropped or
    renamed a model still benchmarks the rest.
    """
    by_tag = {model["tag"]: model for model in discovered}
    selected = [by_tag[tag] for tag in _BENCHMARK_MODEL_TAGS if tag in by_tag]
    missing = [tag for tag in _BENCHMARK_MODEL_TAGS if tag not in by_tag]
    if missing:
        print(
            "qualify: --bench dev skipping curated models absent from the "
            f"installed FLM registry: {', '.join(missing)}",
            file=sys.stderr,
            flush=True,
        )
    if not selected:
        raise QualificationError(
            "no curated --bench dev models are present in the installed FLM "
            "registry"
        )
    return selected


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_asr_sample(stack: ExitStack) -> Path | None:
    """Materialize the audio fixture used to exercise ASR transcription.

    Honors the ``ROCM_NPU_ASR_SAMPLE`` override (a real filesystem path),
    otherwise resolves the clip bundled in the package under ``assets/`` via
    :func:`importlib.resources.as_file` so it works whether the wheel is
    installed unzipped or imported from a zip. The extracted path stays valid
    for the lifetime of ``stack``. Returns ``None`` when no usable sample is
    present; the caller then fails the ASR probe closed.
    """
    override = os.environ.get("ROCM_NPU_ASR_SAMPLE")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    try:
        resource = resources.files("rocm_npu_qualification").joinpath(
            "assets", "asr_sample.wav"
        )
        if resource.is_file():
            return stack.enter_context(resources.as_file(resource))
    except (TypeError, FileNotFoundError, ModuleNotFoundError, OSError):
        return None
    return None


def _resolve_asr_reference() -> str | None:
    """Return the accepted transcript for the ASR sample, or ``None``.

    Honors the ``ROCM_NPU_ASR_REFERENCE`` override (inline text or a file path),
    otherwise falls back to the reference bundled alongside the audio fixture.
    The reference is the pinned phrase the transcript must contain and the
    ground truth for the word-error-rate metric; when it is absent the ASR probe
    fails closed rather than accepting an unverified transcript.
    """
    override = os.environ.get("ROCM_NPU_ASR_REFERENCE")
    if override:
        candidate = Path(override)
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
        return override.strip() or None
    try:
        resource = resources.files("rocm_npu_qualification").joinpath(
            "assets", "asr_sample.reference.txt"
        )
        if resource.is_file():
            return resource.read_text(encoding="utf-8").strip() or None
    except (TypeError, FileNotFoundError, ModuleNotFoundError, OSError):
        return None
    return None


def _sweep_timeouts(footprint: float) -> tuple[int, int]:
    """Scale (boot, per-turn) seconds by model footprint in GB."""
    fp = max(0.5, footprint)
    boot = int(120 + fp * 40)
    turn = int(180 + fp * 60)
    return boot, turn


def _default_tests_for(
    model: dict[str, Any], *, bench_context: str = _SWEEP_BENCH_DEFAULT
) -> list[dict[str, Any]]:
    """Synthesize the complete-sweep test set for a discovered registry model.

    chat -> single-prompt run + 5-turn multi-turn run + a context benchmark
    (8k by default, selectable via ``bench_context``); embed/asr -> a single
    serve-probe. Timeouts scale with footprint so large MoE models do not
    spuriously time out, and with the benchmark context length.
    """
    kind = model.get("kind", "chat")
    footprint = float(model.get("footprint", 0) or 0)
    boot, turn = _sweep_timeouts(footprint)
    if kind in ("embed", "asr"):
        return [
            {
                "name": "serve-probe",
                "kind": "serve",
                "probe_kind": kind,
                "timeout_seconds": min(7200, max(300, boot)),
            }
        ]
    bench = _SWEEP_BENCH_CONFIGS.get(
        bench_context, _SWEEP_BENCH_CONFIGS[_SWEEP_BENCH_DEFAULT]
    )
    single_timeout = min(7200, max(180, turn))
    multi_timeout = min(7200, max(300, turn * 2))
    bench_timeout = min(7200, max(300, turn * bench["timeout_factor"]))
    return [
        {
            "name": "single-prompt",
            "kind": "run",
            "turns": [{"prompt": _SWEEP_SINGLE_PROMPT}],
            "setup_commands": _SWEEP_GREEDY_SETUP,
            "process_iterations": 1,
            "timeout_seconds": single_timeout,
            "ctx_len": 1024,
        },
        {
            "name": "multi-turn",
            "kind": "run",
            "turns": [dict(turn) for turn in _SWEEP_MULTI_TURNS],
            "setup_commands": _SWEEP_GREEDY_SETUP_MULTI,
            "process_iterations": 1,
            "timeout_seconds": multi_timeout,
            "ctx_len": 2048,
        },
        {
            "name": bench["name"],
            "kind": "bench",
            "config": bench["config"],
            "timeout_seconds": bench_timeout,
            "checkpoint_min": bench["checkpoint_min"],
            "checkpoint_max": bench["checkpoint_max"],
            "expected_regex": "",
        },
    ]


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


def _host_xrt_prefix() -> Path | None:
    """Return the host XRT install prefix, or ``None`` if it is not present."""
    candidates: list[Path] = []
    env_root = os.environ.get("XILINX_XRT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    if os.name != "nt":
        candidates.append(Path("/opt/xilinx/xrt"))
    seen: set[Path] = set()
    for prefix in candidates:
        try:
            resolved = prefix.resolve()
        except OSError:
            resolved = prefix
        if resolved in seen:
            continue
        seen.add(resolved)
        if prefix.is_dir() and (
            (prefix / "lib").is_dir() or (prefix / "include").is_dir()
        ):
            return resolved
    return None


def _host_xrt_library_dirs(prefix: Path) -> list[str]:
    """Loader directories under a host XRT prefix, if they exist."""
    directories = []
    for relative in ("lib", os.path.join("lib", "x86_64-linux-gnu")):
        path = prefix / relative
        if path.is_dir():
            directories.append(str(path.resolve()))
    return directories


def _resolve_xrt_flm(xrt_flm: Path) -> dict[str, Any]:
    """Resolve a user-supplied XRT FastFlowLM build for best-effort comparison.

    ``xrt_flm`` may point at the ``flm`` entrypoint itself or at its FHS
    install root (as produced by ``dev.py build fastflowlm-xrt``). Always
    returns a dict: ``status: "resolved"`` with ``command``, ``environment``,
    and ``xilinx_xrt`` on success, or ``status: "unavailable"`` with ``error``
    when a usable build or host XRT cannot be located. Never raises: XRT is a
    diagnostic add-on, and the caller stamps ``unavailable`` on every model so
    the report still widens to N/A columns.
    """
    requested = str(xrt_flm)
    executable = "flm.exe" if os.name == "nt" else "flm"
    candidate = xrt_flm.expanduser()
    if candidate.is_file():
        entrypoint = candidate
    elif candidate.is_dir():
        entrypoint = next(
            (
                option
                for option in (candidate / "bin" / executable, candidate / executable)
                if option.is_file()
            ),
            None,
        )
    else:
        entrypoint = None
    if entrypoint is None:
        error = (
            f"--xrt-flm path does not contain an '{executable}' "
            f"executable ({xrt_flm})"
        )
        print(f"qualify: {error}; recording XRT as unavailable", file=sys.stderr, flush=True)
        return {"status": "unavailable", "requested": requested, "error": error}
    # FHS trees keep metadata under share/flm; a flat tree keeps it next to the
    # binary. Accept whichever layout the supplied build uses.
    root = entrypoint.parent
    metadata_roots = [root, root / "share" / "flm", root.parent / "share" / "flm"]
    config = model_info = xclbin_root = None
    for base in metadata_roots:
        if config is None and (base / "model_list.json").is_file():
            config = base / "model_list.json"
        if model_info is None and (base / "model_info.json").is_file():
            model_info = base / "model_info.json"
        if xclbin_root is None and (base / "xclbins").is_dir():
            xclbin_root = base
    missing = [
        name
        for name, value in (
            ("model_list.json", config),
            ("model_info.json", model_info),
            ("xclbins/", xclbin_root),
        )
        if value is None
    ]
    if missing:
        error = (
            "XRT FastFlowLM build is missing "
            f"{', '.join(missing)} near {entrypoint}"
        )
        print(f"qualify: {error}; recording XRT as unavailable", file=sys.stderr, flush=True)
        return {"status": "unavailable", "requested": requested, "error": error}
    host_xrt = _host_xrt_prefix()
    if host_xrt is None:
        error = (
            "host XRT runtime is not installed (set XILINX_XRT or install to "
            "/opt/xilinx/xrt); the XRT comparison uses the host install rather "
            "than a bundled prefix"
        )
        print(f"qualify: {error}; recording XRT as unavailable", file=sys.stderr, flush=True)
        return {"status": "unavailable", "requested": requested, "error": error}
    environment = dict(os.environ)
    environment["FLM_CONFIG_PATH"] = str(config.resolve())
    environment["FLM_XCLBIN_PATH"] = str(xclbin_root.resolve())
    environment["FLM_MODELINFO_PATH"] = str(model_info.resolve())
    # Isolate host XRT to this child only; never rewrite the parent process.
    environment["XILINX_XRT"] = str(host_xrt)
    if os.name != "nt":
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            entry
            for entry in (
                *_host_xrt_library_dirs(host_xrt),
                environment.get("LD_LIBRARY_PATH", ""),
            )
            if entry
        )
    return {
        "status": "resolved",
        "requested": requested,
        "command": str(entrypoint.resolve()),
        "xilinx_xrt": str(host_xrt),
        "environment": environment,
    }


def _xrt_unavailable_metrics(error: str) -> dict[str, Any]:
    return {"status": "unavailable", "error": error}


def _xrt_descriptor(resolved: dict[str, Any]) -> dict[str, Any]:
    """Identity projection recorded on a diagnostic-sweep ``profile.json``."""
    descriptor = {
        "requested": resolved.get("requested"),
        "command": resolved.get("command"),
        "xilinx_xrt": resolved.get("xilinx_xrt"),
        "status": resolved["status"],
    }
    if resolved["status"] != "resolved":
        descriptor["error"] = resolved.get("error") or "XRT comparison unavailable"
    return descriptor


def _run_flm_bench(
    *,
    flm: Path,
    model_tag: str,
    test: dict[str, Any],
    environment: dict[str, str],
    writer: EvidenceWriter,
    work: Path,
    log: Path,
    benchmark_iterations: int,
    runtime_label: str = "HRX",
) -> dict[str, Any]:
    """Run a single ``flm bench`` and parse it into a backend-agnostic result.

    Shared by the HRX benchmark and the optional XRT comparison so both backends
    are measured and validated identically. ``runtime_label`` only affects the
    runtime-error wording (``HRX`` vs ``XRT``). Returns the parsed outcome (pass
    state, checkpoint, CSV metrics, artifact-relative paths); it never appends to
    evidence so callers decide how to record it.
    """
    work.mkdir(parents=True, exist_ok=True)
    stamp = work / ".started"
    stamp.write_text("", encoding="utf-8")
    started_ns = stamp.stat().st_mtime_ns
    config_resource = resources.files("rocm_npu_qualification").joinpath(
        test["config"]
    )
    if not config_resource.is_file():
        raise QualificationError(
            f"benchmark configuration is missing: {test['config']}"
        )
    with resources.as_file(config_resource) as packaged_config:
        try:
            benchmark_config = json.loads(packaged_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise QualificationError(
                f"benchmark configuration is invalid: {test['config']}"
            ) from error
        if not isinstance(benchmark_config, dict):
            raise QualificationError(
                f"benchmark configuration is invalid: {test['config']}"
            )
        benchmark_config["iterations"] = benchmark_iterations
        config = work / "benchmark-config.json"
        config.write_text(
            json.dumps(benchmark_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = run_command(
            [flm, "bench", model_tag, "-i", config.resolve()],
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
    if has_runtime_error(transcript):
        semantic_error = f"FastFlowLM emitted an {runtime_label} runtime error"
    elif MAX_LENGTH_PATTERN.search(transcript):
        semantic_error = "FastFlowLM reached the generation length limit"
    else:
        try:
            expected_outputs = require_benchmark_schedule(
                transcript, iterations=benchmark_iterations
            )
            if len(raw_outputs) != expected_outputs:
                raise ValueError(
                    f"FastFlowLM emitted {len(raw_outputs)} benchmark "
                    f"responses; expected {expected_outputs}"
                )
            checkpoint = require_1k_checkpoint(
                transcript,
                minimum=test["checkpoint_min"],
                maximum=test["checkpoint_max"],
                expected_count=benchmark_iterations,
            )
        except ValueError as error:
            semantic_error = str(error)
    if not semantic_error:
        for output_number, raw in enumerate(raw_outputs, start=1):
            if (
                # Minimum multi-letter word count for the output to read as text.
                # Kept low so a short but valid response (e.g. a coherent refusal
                # like "I can't provide a meaningful response to your request.")
                # is not misflagged; empty/garbage output still trips this.
                len(re.findall(r"[A-Za-z]{2,}", raw)) < 4
                and BENCHMARK_PASS_SENTINEL_PATTERN.search(raw) is None
            ):
                semantic_error = (
                    f"FastFlowLM benchmark output {output_number} is not "
                    "coherent text"
                )
            elif test["expected_regex"] and re.search(
                test["expected_regex"], raw, re.IGNORECASE | re.DOTALL
            ) is None:
                semantic_error = (
                    f"FastFlowLM benchmark output {output_number} did not match"
                )
            if semantic_error:
                break
    csvs = fresh_benchmark_csvs(work, model_tag, started_ns=started_ns)
    if not semantic_error and not csvs:
        semantic_error = "FastFlowLM benchmark did not write a fresh CSV"
    csv_row: list[str] = []
    csv_metrics: dict[str, int | float] = {}
    if csvs:
        csv_path = csvs[-1]
        with csv_path.open(newline="", encoding="utf-8", errors="replace") as stream:
            csv_rows = list(csv.reader(stream))
        if not semantic_error and checkpoint is not None:
            try:
                csv_row, csv_metrics = _benchmark_csv_checkpoint(
                    csv_rows, max(1, round(checkpoint / 1024))
                )
            except ValueError as error:
                semantic_error = str(error)
    passed = result.returncode == 0 and not semantic_error
    return {
        "passed": passed,
        "semantic_error": semantic_error,
        "checkpoint": checkpoint,
        "raw_outputs": raw_outputs,
        "csv_row": csv_row,
        "csv_metrics": csv_metrics,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_seconds": result.duration_seconds,
        "log": writer.relative(log),
        "config": writer.relative(config),
        "csv": None if csv_path is None else writer.relative(csv_path),
    }


def _run_xrt_comparison(
    *,
    xrt_flm: Path,
    model_tag: str,
    test: dict[str, Any],
    environment: dict[str, str],
    writer: EvidenceWriter,
    safe_tag: str,
    benchmark_iterations: int,
) -> tuple[dict[str, Any], list[str]]:
    """Benchmark one model through the XRT build, tolerating any failure.

    Returns ``(metrics, artifacts)`` to fold into the HRX bench record's
    ``metrics.xrt``. Any error resolving or running the XRT build yields a
    ``status: "unavailable"`` marker rather than raising, keeping XRT strictly
    informational.
    """
    log = writer.path(f"logs/tests/{safe_tag}-{test['name']}-xrt.log")
    work = writer.path(f"bench/{safe_tag}/{test['name']}-xrt")
    try:
        result = _run_flm_bench(
            flm=xrt_flm,
            model_tag=model_tag,
            test=test,
            environment=environment,
            writer=writer,
            work=work,
            log=log,
            benchmark_iterations=benchmark_iterations,
            runtime_label="XRT",
        )
    except (QualificationError, OSError) as error:
        return {"status": "unavailable", "error": str(error)}, []
    metrics: dict[str, Any] = {
        "status": "pass" if result["passed"] else "fail",
        "checkpoint_tokens": result["checkpoint"],
        "responses": len(result["raw_outputs"]),
        "csv_checkpoint_row": result["csv_row"],
        "csv_checkpoint_metrics": result["csv_metrics"],
        "log": result["log"],
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
        "duration_seconds": result["duration_seconds"],
    }
    if not result["passed"]:
        metrics["error"] = (
            result["semantic_error"] or "FastFlowLM XRT benchmark failed"
        )
    artifacts = [result["config"]] + (
        [] if result["csv"] is None else [result["csv"]]
    )
    return metrics, artifacts


def _run_xrt_serve_comparison(
    *,
    xrt_flm: Path,
    model_tag: str,
    test: dict[str, Any],
    environment: dict[str, str],
    writer: EvidenceWriter,
    safe_tag: str,
    probe_kind: str,
    audio_sample: Path | None,
    reference_text: str | None,
) -> dict[str, Any]:
    """Serve-probe one embed/ASR model through the XRT build, tolerating failure.

    Returns metrics to fold into the HRX serve record's ``metrics.xrt`` (the
    same encode-tok/s / RTF / WER measurements the HRX probe records). Any error
    resolving or running the XRT build yields a ``status: "unavailable"`` marker
    rather than raising, keeping XRT strictly informational.
    """
    log = writer.path(f"logs/tests/{safe_tag}-{test['name']}-xrt.log")
    serve_argv: list[str | os.PathLike[str]] = [xrt_flm, "serve", model_tag]
    if probe_kind == "embed":
        serve_argv += ["-e", "1"]
    elif probe_kind == "asr":
        serve_argv += ["-a", "1"]
    try:
        result, probe = serve_probe(
            serve_argv,
            log_path=log,
            env=environment,
            probe_kind=probe_kind,
            model=model_tag,
            boot_timeout=test["timeout_seconds"],
            audio_sample=audio_sample,
            reference_text=reference_text,
            echo=True,
        )
    except (QualificationError, OSError) as error:
        return {"status": "unavailable", "error": str(error)}
    transcript = result.stdout
    semantic_error = ""
    if has_runtime_error(transcript):
        semantic_error = "FastFlowLM emitted an XRT runtime error"
    elif not probe["ok"]:
        semantic_error = probe["detail"] or "FastFlowLM XRT serve probe failed"
    passed = result.returncode == 0 and not semantic_error
    metrics: dict[str, Any] = {
        "status": "pass" if passed else "fail",
        "port": probe["port"],
        "detail": probe["detail"],
        "log": writer.relative(log),
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_seconds": result.duration_seconds,
        **(probe.get("metrics") or {}),
    }
    if not passed:
        metrics["error"] = semantic_error or "FastFlowLM XRT serve probe failed"
    return metrics


def _test_model(
    model: dict[str, Any],
    *,
    flm: Path,
    model_cache: Path,
    offline: bool,
    writer: EvidenceWriter,
    environment: dict[str, str],
    prior_tests: list[dict[str, Any]] | None = None,
    xrt_flm: Path | None = None,
    xrt_environment: dict[str, str] | None = None,
    xrt_unavailable: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    environment = fastflow_environment(
        model_cache, offline=offline, base_environment=environment
    )
    xrt_bench_environment = (
        fastflow_environment(
            model_cache, offline=offline, base_environment=xrt_environment
        )
        if xrt_flm is not None and xrt_environment is not None
        else None
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
        if test["kind"] == "serve":
            safe_tag = "".join(
                character if character.isalnum() else "-" for character in model["tag"]
            )
            log = writer.path(f"logs/tests/{safe_tag}-{test['name']}.log")
            probe_kind = test.get("probe_kind", "embed")
            serve_argv: list[str | os.PathLike[str]] = [flm, "serve", model["tag"]]
            with ExitStack() as sample_stack:
                audio_sample: Path | None = None
                reference_text: str | None = None
                if probe_kind == "embed":
                    serve_argv += ["-e", "1"]
                elif probe_kind == "asr":
                    serve_argv += ["-a", "1"]
                    audio_sample = _resolve_asr_sample(sample_stack)
                    reference_text = _resolve_asr_reference()
                result, probe = serve_probe(
                    serve_argv,
                    log_path=log,
                    env=environment,
                    probe_kind=probe_kind,
                    model=model["tag"],
                    boot_timeout=test["timeout_seconds"],
                    audio_sample=audio_sample,
                    reference_text=reference_text,
                    echo=True,
                )
                # XRT comparison mirrors the bench path: strictly best-effort and
                # folded into this HRX record's free-form metrics.xrt (never a
                # separate test), so an unavailable or failing XRT build reports
                # as N/A and can never flip the run's pass/fail. Run it inside the
                # ExitStack so the ASR sample stays valid.
                xrt_serve_metrics: dict[str, Any] | None = None
                if xrt_unavailable is not None:
                    xrt_serve_metrics = dict(xrt_unavailable)
                elif xrt_flm is not None and xrt_bench_environment is not None:
                    xrt_serve_metrics = _run_xrt_serve_comparison(
                        xrt_flm=xrt_flm,
                        model_tag=model["tag"],
                        test=test,
                        environment=xrt_bench_environment,
                        writer=writer,
                        safe_tag=safe_tag,
                        probe_kind=probe_kind,
                        audio_sample=audio_sample,
                        reference_text=reference_text,
                    )
            transcript = result.stdout
            semantic_error = ""
            if has_runtime_error(transcript):
                semantic_error = "FastFlowLM emitted an HRX runtime error"
            elif not probe["ok"]:
                semantic_error = probe["detail"] or "FastFlowLM serve probe failed"
            passed = result.returncode == 0 and not semantic_error
            serve_metrics: dict[str, Any] = {
                "probe_kind": probe_kind,
                "port": probe["port"],
                "detail": probe["detail"],
                **(probe.get("metrics") or {}),
            }
            if xrt_serve_metrics is not None:
                serve_metrics["xrt"] = xrt_serve_metrics
            records.append(
                {
                    "model": model["tag"],
                    "name": test["name"],
                    "kind": "serve",
                    "status": "pass" if passed else "fail",
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "duration_seconds": result.duration_seconds,
                    "log": writer.relative(log),
                    "metrics": serve_metrics,
                    "artifacts": [],
                    **(
                        {}
                        if passed
                        else {"error": semantic_error or "FastFlowLM serve probe failed"}
                    ),
                }
            )
            write_progress()
            continue

        if test["kind"] == "bench":
            safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", model["tag"])
            benchmark_iterations = test.get("iterations", 1)
            hrx = _run_flm_bench(
                flm=flm,
                model_tag=model["tag"],
                test=test,
                environment=environment,
                writer=writer,
                work=writer.path(f"bench/{safe_tag}/{test['name']}"),
                log=writer.path(f"logs/tests/{safe_tag}-{test['name']}.log"),
                benchmark_iterations=benchmark_iterations,
            )
            metrics: dict[str, Any] = {
                "checkpoint_tokens": hrx["checkpoint"],
                "benchmark_iterations": benchmark_iterations,
                "iterations": benchmark_iterations,
                "responses": len(hrx["raw_outputs"]),
                "raw_outputs_sha256": canonical_json_sha256(hrx["raw_outputs"]),
                "csv_checkpoint_row": hrx["csv_row"],
                "csv_checkpoint_metrics": hrx["csv_metrics"],
            }
            if xrt_flm is not None or xrt_unavailable is not None:
                metrics["backend"] = "hrx"
            artifacts = [hrx["config"]] + (
                [] if hrx["csv"] is None else [hrx["csv"]]
            )
            # XRT comparison is strictly best-effort: it is folded into this HRX
            # record's free-form metrics (never a separate test), so an
            # unavailable or failing XRT build reports as N/A and can never flip
            # the run's pass/fail or add JUnit noise. Stamp unavailable when the
            # flag was given but the build could not be resolved, so the report
            # still widens to XRT columns.
            if xrt_unavailable is not None:
                metrics["xrt"] = dict(xrt_unavailable)
            elif xrt_flm is not None and xrt_bench_environment is not None:
                xrt_metrics, xrt_artifacts = _run_xrt_comparison(
                    xrt_flm=xrt_flm,
                    model_tag=model["tag"],
                    test=test,
                    environment=xrt_bench_environment,
                    writer=writer,
                    safe_tag=safe_tag,
                    benchmark_iterations=benchmark_iterations,
                )
                metrics["xrt"] = xrt_metrics
                artifacts += xrt_artifacts
            records.append(
                {
                    "model": model["tag"],
                    "name": test["name"],
                    "kind": "bench",
                    "status": "pass" if hrx["passed"] else "fail",
                    "returncode": hrx["returncode"],
                    "timed_out": hrx["timed_out"],
                    "duration_seconds": hrx["duration_seconds"],
                    "log": hrx["log"],
                    "turn_count": benchmark_iterations,
                    "process_iteration": 1,
                    "metrics": metrics,
                    "artifacts": artifacts,
                    **(
                        {}
                        if hrx["passed"]
                        else {
                            "error": hrx["semantic_error"]
                            or "FastFlowLM benchmark failed"
                        }
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
            max_length_reached = bool(MAX_LENGTH_PATTERN.search(transcript))
            semantic_error = ""
            if has_runtime_error(transcript):
                semantic_error = "FastFlowLM emitted an HRX runtime error"
            elif len(raw_outputs) != len(turns):
                semantic_error = (
                    f"FastFlowLM emitted {len(raw_outputs)} responses; "
                    f"expected {len(turns)}"
                )
            else:
                for turn_number, (raw, turn) in enumerate(
                    zip(raw_outputs, turns), start=1
                ):
                    expected = turn.get("expected")
                    expected_regex = turn.get("expected_regex")
                    if expected and not smoke_phrase_matches(raw, expected):
                        semantic_error = f"FastFlowLM turn {turn_number} did not match"
                    elif (
                        expected_regex
                        and re.search(expected_regex, raw, re.IGNORECASE) is None
                    ):
                        semantic_error = (
                            f"FastFlowLM turn {turn_number} did not match regex"
                        )
                    elif not (expected or expected_regex) and not re.search(
                        r"[A-Za-z]{4,}", raw
                    ):
                        # The generic coherence gate only applies to open-ended
                        # turns. Turns that assert a short answer (e.g. "4" for
                        # 2+2) are validated by expected/expected_regex above, so
                        # a numeric-only reply is not "no visible text".
                        semantic_error = (
                            f"FastFlowLM turn {turn_number} has no visible text"
                        )
                    if semantic_error:
                        break
            # A generation-length cap is only fatal when it stopped a turn from
            # satisfying its expectation. Every turn responded (count matches) and
            # each per-turn check above passed, so a thinking model that ran long
            # before answering is not itself a failure.
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
                        "max_length_reached": max_length_reached,
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

    all_models = bool(getattr(args, "all_models", False))
    # --bench MODE selects a measurement-only sweep; the mode chooses coverage
    # ('all' sweeps the whole registry, 'dev' the curated chat subset).
    # Context/iterations come from the dedicated --bench-context /
    # --bench-iterations flags rather than a packed value.
    bench_mode = getattr(args, "bench", None)
    any_benchmark = bench_mode is not None
    benchmark = bench_mode == "all"
    benchmark_dev = bench_mode == "dev"
    if any_benchmark:
        benchmark_context = (
            getattr(args, "bench_context", None) or _BENCHMARK_DEFAULT_CONTEXT
        )
        raw_iterations = getattr(args, "bench_iterations", None)
        if raw_iterations is None:
            benchmark_iterations = _BENCHMARK_DEFAULT_ITERATIONS
        else:
            benchmark_iterations = int(raw_iterations)
            if benchmark_iterations < 1:
                raise QualificationError(
                    "--bench-iterations must be a positive integer, got "
                    f"{benchmark_iterations}"
                )
    # --models narrows a benchmark sweep to an explicit tag subset. It only has
    # meaning for a benchmark mode; anywhere else it is warned about and ignored.
    # An explicit empty list (``--models=`` / commas-only) is an error: the flag
    # is a restriction, not a no-op that silently sweeps the whole registry.
    models_arg = getattr(args, "models", None)
    model_filter: list[str] | None = None
    if models_arg is not None:
        model_filter = [tag.strip() for tag in models_arg.split(",") if tag.strip()]
        if not model_filter:
            raise QualificationError("--models requires at least one tag")
    if model_filter and not any_benchmark:
        print(
            "qualify: --models has no effect without --bench; ignoring it",
            file=sys.stderr,
            flush=True,
        )
        model_filter = None
    if getattr(args, "bench_iterations", None) is not None and not any_benchmark:
        print(
            "qualify: --bench-iterations has no effect without --bench; ignoring it",
            file=sys.stderr,
            flush=True,
        )
    # Supplying --xrt-flm is what opts into the XRT comparison; there is no
    # separate enable flag.
    xrt_flm_arg = getattr(args, "xrt_flm", None)
    with_xrt = xrt_flm_arg is not None
    if with_xrt and not any_benchmark:
        print(
            "qualify: --xrt-flm has no effect without --bench; "
            "ignoring the XRT comparison",
            file=sys.stderr,
            flush=True,
        )
        with_xrt = False
    # A sweep should surface every model's result, so one failing model must not
    # abort the run; keep-going is implied (still overridable upward).
    is_sweep = all_models or any_benchmark
    keep_going = bool(getattr(args, "keep_going", False) or is_sweep)
    if all_models:
        bench_choice = getattr(args, "bench_context", None) or _SWEEP_BENCH_DEFAULT
        # The sweep is not a packaged profile-v1: it discovers whatever the
        # installed FLM registry exposes, so it cannot pin files/XCLBINs and is
        # not a named contract. Describe it honestly as a diagnostic-sweep
        # descriptor. The concrete tags/tests are filled in after discovery.
        profile = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "kind": "diagnostic-sweep",
            "name": "all-models",
            "description": (
                "Dynamic complete sweep of every model in the installed "
                "FastFlowLM registry: single-prompt run, multi-turn, and a "
                f"{bench_choice} benchmark for chat models; serve-probe for "
                "embed/ASR models."
            ),
            "bench_context": bench_choice,
            "models": [],
        }
    elif benchmark:
        # Also a diagnostic-sweep descriptor: a measurement-only sweep over every
        # model in the installed registry (optionally narrowed by --models). The
        # concrete tags/tests are filled in after discovery.
        scope = (
            f"the {len(model_filter)} model(s) selected via --models"
            if model_filter
            else "every model in the installed FastFlowLM registry"
        )
        profile = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "kind": "diagnostic-sweep",
            "name": "bench-all",
            "description": (
                f"Measurement-only sweep over {scope}: a {benchmark_context} "
                f"decode benchmark of {benchmark_iterations} iteration(s) for "
                "chat models; serve-probe for embed/ASR models."
            ),
            "bench_context": benchmark_context,
            "bench_iterations": benchmark_iterations,
            "models": [],
        }
    elif benchmark_dev:
        # The fast development variant: same measurement stages, but over the
        # curated chat subset (optionally narrowed further by --models). The
        # concrete tags/tests are filled in after discovery.
        scope = (
            f"the {len(model_filter)} model(s) selected via --models"
            if model_filter
            else "a curated set of chat tags"
        )
        profile = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "kind": "diagnostic-sweep",
            "name": "bench-dev",
            "description": (
                f"Development benchmark sweep over {scope}: a {benchmark_context} "
                f"decode benchmark of {benchmark_iterations} iteration(s) per "
                "model."
            ),
            "bench_context": benchmark_context,
            "bench_iterations": benchmark_iterations,
            "models": [],
        }
    else:
        profile_name = _selected_profile_name(args)
        profile = load_profile(profile_name)
    xrt_flm: Path | None = None
    xrt_environment: dict[str, str] | None = None
    xrt_unavailable: dict[str, Any] | None = None
    if with_xrt:
        resolved = _resolve_xrt_flm(Path(xrt_flm_arg))
        profile["xrt"] = _xrt_descriptor(resolved)
        if resolved["status"] == "resolved":
            xrt_flm = Path(resolved["command"])
            xrt_environment = resolved["environment"]
            print(
                f"qualify: XRT comparison enabled using {xrt_flm} "
                f"(host XRT {resolved['xilinx_xrt']})",
                file=sys.stderr,
                flush=True,
            )
        else:
            xrt_unavailable = _xrt_unavailable_metrics(
                resolved.get("error") or "XRT comparison unavailable"
            )
            print(
                "qualify: XRT comparison unavailable "
                f"({xrt_unavailable['error']}); recording N/A for every model",
                file=sys.stderr,
                flush=True,
            )
    candidate = None
    if args.candidate_manifest is not None:
        try:
            candidate = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise QualificationError("candidate manifest is invalid") from error
        if not isinstance(candidate, dict):
            raise QualificationError("candidate manifest must be an object")
        candidate = _normalize_candidate_manifest(candidate)

    # Version/wheel validation still runs against a supplied candidate even for
    # the sweep, so a mismatched install is rejected up front.
    _validate_versions(installed, core_version, candidate, args.candidate_manifest)
    # candidate_bound means "execution equals a sealed wheel profile," not merely
    # "a candidate manifest was passed." The diagnostic sweep is never a sealed
    # wheel profile, so it is always ad-hoc even with --candidate-manifest; intake
    # then refuses it. Non-sweep runs bind when a candidate is present.
    candidate_bound = candidate is not None and not is_sweep
    bound_candidate = candidate if candidate_bound else None
    numeric_runtime = _numeric_runtime_identities()
    writer = EvidenceWriter(args.output)
    writer.write_candidate_manifest(bound_candidate)
    writer.write_json("profile.json", profile)
    started = _now()
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "status": "running",
        "subject": {
            "version": core_version,
            "source_revision": (
                bound_candidate["source_revision"]
                if bound_candidate is not None
                else ""
            ),
            "candidate_binding": {
                "present": candidate_bound,
                "manifest_sha256": (
                    sha256_file(writer.path("candidate-manifest.json"))
                    if candidate_bound
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
            "candidate_bound": candidate_bound,
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
        if not keep_going:
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
    if any(test["status"] != "pass" for test in xadx_tests) and not keep_going:
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

    execution_models = profile["models"]
    if is_sweep:
        discovered = enumerate_registry_models(flm_config)
        if any_benchmark:
            # --bench dev starts from the curated subset; --bench all from the
            # whole registry. --models then narrows whichever base set was chosen.
            base_models = (
                _select_benchmark_models(discovered) if benchmark_dev else discovered
            )
            execution_models = (
                _filter_models_by_tags(
                    base_models,
                    model_filter,
                    pool=(
                        "curated --bench dev set"
                        if benchmark_dev
                        else "installed FLM registry"
                    ),
                )
                if model_filter
                else base_models
            )
            for base in execution_models:
                base["tests"] = _benchmark_tests_for(
                    base,
                    bench_context=benchmark_context,
                    iterations=benchmark_iterations,
                )
            mode = "bench-dev" if benchmark_dev else "bench-all"
            selection = "selected" if model_filter else (
                "curated" if benchmark_dev else "discovered"
            )
            summary_note = (
                f"{mode} sweep {selection} {len(execution_models)} models "
                "from the installed FLM registry"
            )
        else:
            bench_context = getattr(args, "bench_context", None) or _SWEEP_BENCH_DEFAULT
            for base in discovered:
                base["tests"] = _default_tests_for(
                    base, bench_context=bench_context
                )
            execution_models = discovered
            summary_note = (
                f"complete sweep discovered {len(discovered)} models from the "
                "installed FLM registry"
            )
        # The descriptor records what actually ran (tag, kind, test names) rather
        # than a profile-v1 model+test config: a sweep pins nothing, so the full
        # test specs live only in the execution list below.
        profile["models"] = [
            {
                "tag": base["tag"],
                "kind": base.get("kind", "chat"),
                "tests": [test["name"] for test in base["tests"]],
            }
            for base in execution_models
        ]
        writer.write_json("profile.json", profile)
        print(f"qualify: {summary_note}", file=sys.stderr, flush=True)

    resolve = resolve_model_dynamic if is_sweep else resolve_model
    for model in execution_models:
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
            model_record, _ = resolve(
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
                    xrt_flm=xrt_flm,
                    xrt_environment=xrt_environment,
                    xrt_unavailable=xrt_unavailable,
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
        if failed and not keep_going:
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
    if is_sweep:
        kinds = {
            model["tag"]: model.get("kind", "chat")
            for model in profile["models"]
        }
        report_path = writer.path("report.md")
        report_path.write_text(
            build_markdown_report(evidence, kinds=kinds), encoding="utf-8"
        )
        print(
            f"qualify: wrote {profile['name']} sweep report {report_path}",
            file=sys.stderr,
            flush=True,
        )
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
