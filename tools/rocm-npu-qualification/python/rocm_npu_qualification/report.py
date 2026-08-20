# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Render a human-readable Markdown report from a qualification evidence bundle.

The complete-sweep (``--all-models``) run emits, in a single command, the same
tables the manual HRX experiment harness produced: an environment header, a
per-model functional matrix, and an averaged on-NPU benchmark table. Only HRX
numbers are available here, so no XRT columns are rendered.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any

_ENV_ORDER = (
    "os",
    "architecture",
    "npu_device",
    "npu_driver",
    "npu_firmware",
    "target_family",
    "flm_version",
    "python",
    "rocm-npu-runtime",
    "rocm-npu-fastflowlm",
    "rocm-npu-qualification",
    "source_revision",
)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _tests_by_model(evidence: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for test in evidence.get("tests", []):
        grouped.setdefault(str(test.get("model", "")), []).append(test)
    return grouped


def _first(tests: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    return next((test for test in tests if predicate(test)), None)


def _status_label(test: dict[str, Any] | None) -> str:
    if test is None:
        return "—"
    return "PASS" if test.get("status") == "pass" else "FAIL"


def _device_facts(platform_facts: dict[str, Any]) -> dict[str, str]:
    devices = platform_facts.get("devices")
    if not isinstance(devices, list) or not devices:
        return {}
    device = next(
        (entry for entry in devices if entry.get("healthy")), devices[0]
    )
    facts: dict[str, str] = {}
    name = device.get("device_name")
    if name:
        state = "OK" if device.get("healthy") else "not ready"
        facts["npu_device"] = f"{name} (status {state})"
    driver = device.get("driver_version")
    if driver:
        facts["npu_driver"] = str(driver)
    releases = sorted(
        {
            str(candidate.get("release"))
            for candidate in device.get("firmware_candidates", [])
            if candidate.get("release")
        }
    )
    if releases:
        facts["npu_firmware"] = (
            "candidates "
            + ", ".join(releases)
            + " (KMD-loaded not exposed by PnP)"
        )
    return facts


def _environment(evidence: dict[str, Any]) -> dict[str, str]:
    subject = evidence.get("subject", {})
    platform_facts = evidence.get("observed_platform", {})
    env: dict[str, str] = {}
    for key in ("os", "architecture", "target_family"):
        value = platform_facts.get(key)
        if value:
            env[key] = str(value)
    env.update(_device_facts(platform_facts))
    flm = subject.get("flm", {})
    if flm.get("version_output"):
        env["flm_version"] = str(flm["version_output"])
    env["python"] = platform.python_version()
    for name, version in subject.get("packages", {}).items():
        env[name] = str(version)
    if subject.get("source_revision"):
        env["source_revision"] = str(subject["source_revision"])
    return env


def _environment_table(evidence: dict[str, Any]) -> list[str]:
    env = _environment(evidence)
    lines = ["## Environment", "", "| Item | Value |", "|------|-------|"]
    for key in _ENV_ORDER:
        if key in env:
            lines.append(f"| {key} | {env[key]} |")
    lines.append("")
    return lines


def _model_kind(
    tag: str, tests: list[dict[str, Any]], kinds: dict[str, str] | None
) -> str:
    if kinds and tag in kinds:
        return kinds[tag]
    serve = _first(tests, lambda test: test.get("kind") == "serve")
    if serve is not None:
        return str(serve.get("metrics", {}).get("probe_kind", "serve"))
    return "chat"


def _results_table(
    evidence: dict[str, Any], kinds: dict[str, str] | None
) -> list[str]:
    grouped = _tests_by_model(evidence)
    lines = [
        "## HRX per-model results (run/verify)",
        "",
        "| Model | Kind | Single-prompt | Multi-turn (5) | 2+2=4 | Detail |",
        "|-------|------|---------------|----------------|-------|--------|",
    ]
    for model in evidence.get("models", []):
        tag = str(model.get("tag", ""))
        tests = grouped.get(tag, [])
        kind = _model_kind(tag, tests, kinds)
        runs = [test for test in tests if test.get("kind") == "run"]
        single = _first(runs, lambda test: "single" in test.get("name", ""))
        multi = _first(runs, lambda test: "multi" in test.get("name", ""))
        serve = _first(tests, lambda test: test.get("kind") == "serve")
        if kind in ("embed", "asr"):
            single_cell = _status_label(serve)
            multi_cell = "N/A"
            math_cell = "—"
        else:
            single_cell = _status_label(single)
            multi_cell = _status_label(multi)
            if multi is not None and multi.get("status") == "pass":
                math_cell = "yes"
            elif multi is not None:
                math_cell = "no"
            else:
                math_cell = "—"
        failing = _first(
            tests, lambda test: test.get("status") not in ("pass", None)
        )
        if model.get("status") != "pass" and model.get("error"):
            detail = f"resolution: {model['error']}"
        elif failing is not None and failing.get("error"):
            detail = str(failing["error"])
        elif serve is not None:
            detail = str(serve.get("metrics", {}).get("detail", "") or "served")
        else:
            detail = "ok"
        lines.append(
            f"| `{tag}` | {kind} | {single_cell} | {multi_cell} | "
            f"{math_cell} | {detail[:80]} |"
        )
    lines.append("")
    return lines


def _benchmark_table(
    evidence: dict[str, Any], kinds: dict[str, str] | None
) -> list[str]:
    grouped = _tests_by_model(evidence)
    bench_tests = [
        test for test in evidence.get("tests", []) if test.get("kind") == "bench"
    ]
    iteration_counts = {
        int(test.get("metrics", {}).get("iterations", 1)) for test in bench_tests
    }
    contexts = {
        test.get("metrics", {}).get("csv_checkpoint_metrics", {}).get(
            "context_length_k"
        )
        for test in bench_tests
    }
    contexts.discard(None)
    facets = ["HRX"]
    if len(contexts) == 1:
        facets.append(f"{contexts.pop()}k")
    if len(iteration_counts) == 1:
        facets.append(f"averaged over {iteration_counts.pop()} iterations")
    else:
        facets.append("averaged over configured iterations")
    lines = [
        f"## Benchmark ({', '.join(facets)})",
        "",
        "| Model | TTFT s | Prefill tok/s | Decode tok/s |",
        "|-------|--------|---------------|--------------|",
    ]
    for model in evidence.get("models", []):
        tag = str(model.get("tag", ""))
        tests = grouped.get(tag, [])
        # Embedding/ASR models run a serve-probe, not the decode benchmark; their
        # measurements are rendered in dedicated modality tables below, so keep
        # them out of the decode table instead of emitting meaningless N/A rows.
        if _model_kind(tag, tests, kinds) in ("embed", "asr"):
            continue
        bench = _first(tests, lambda test: test.get("kind") == "bench")
        metrics = (bench or {}).get("metrics", {})
        csv_metrics = metrics.get("csv_checkpoint_metrics") or {}
        if bench is None or bench.get("status") != "pass" or not csv_metrics:
            lines.append(f"| `{tag}` | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| `{tag}` | "
            f"{_fmt(csv_metrics.get('ttft_avg_s'))} | "
            f"{_fmt(csv_metrics.get('prefill_avg_toks_per_s'))} | "
            f"{_fmt(csv_metrics.get('decoding_avg_toks_per_s'))} |"
        )
    lines.append("")
    return lines


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{value * 100:.1f}"


def _serve_metrics(
    evidence: dict[str, Any], kinds: dict[str, str] | None, wanted: str
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return (tag, test, metrics) triples for models of the wanted serve kind."""
    grouped = _tests_by_model(evidence)
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for model in evidence.get("models", []):
        tag = str(model.get("tag", ""))
        tests = grouped.get(tag, [])
        if _model_kind(tag, tests, kinds) != wanted:
            continue
        serve = _first(tests, lambda test: test.get("kind") == "serve")
        if serve is None:
            continue
        rows.append((tag, serve, serve.get("metrics", {}) or {}))
    return rows


def _embedding_table(
    evidence: dict[str, Any], kinds: dict[str, str] | None
) -> list[str]:
    rows = _serve_metrics(evidence, kinds, "embed")
    if not rows:
        return []
    lines = [
        "## Embedding probe (HRX)",
        "",
        "| Model | Status | Encode tok/s |",
        "|-------|--------|--------------|",
    ]
    for tag, serve, metrics in rows:
        lines.append(
            f"| `{tag}` | {_status_label(serve)} | "
            f"{_fmt(metrics.get('encode_toks_per_s'))} |"
        )
    lines.append("")
    return lines


def _asr_table(
    evidence: dict[str, Any], kinds: dict[str, str] | None
) -> list[str]:
    rows = _serve_metrics(evidence, kinds, "asr")
    if not rows:
        return []
    lines = [
        "## ASR probe (HRX)",
        "",
        "| Model | Status | RTF | WER % |",
        "|-------|--------|-----|-------|",
    ]
    for tag, serve, metrics in rows:
        lines.append(
            f"| `{tag}` | {_status_label(serve)} | "
            f"{_fmt(metrics.get('real_time_factor'))} | "
            f"{_fmt_pct(metrics.get('word_error_rate'))} |"
        )
    lines.append("")
    return lines


def _notes(evidence: dict[str, Any]) -> list[str]:
    failures = [
        f"- **{test.get('kind', 'test')}** `{test.get('model', '')}`/"
        f"{test.get('name', '')}: {test.get('error', '')}"
        for test in evidence.get("tests", [])
        if test.get("status") not in ("pass", None)
    ]
    lines = ["## Issues / notes", ""]
    lines.extend(failures or ["- No failures recorded."])
    lines.append("")
    return lines


def build_markdown_report(
    evidence: dict[str, Any],
    *,
    kinds: dict[str, str] | None = None,
    generated_at: str | None = None,
) -> str:
    """Render the qualification evidence bundle as a Markdown report."""
    profile = evidence.get("profile", {})
    stamp = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    lines = [
        f"# ROCm NPU qualification — {profile.get('name', 'run')} "
        f"({evidence.get('status', 'unknown')})",
        "",
        f"_Generated {stamp}._",
        "",
    ]
    lines += _environment_table(evidence)
    lines += _results_table(evidence, kinds)
    lines += _benchmark_table(evidence, kinds)
    lines += _embedding_table(evidence, kinds)
    lines += _asr_table(evidence, kinds)
    lines += _notes(evidence)
    return "\n".join(lines) + "\n"
