# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Controlled subprocess execution with redacted evidence logs."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization:\s*(?:bearer|basic))\s+\S+"),
    re.compile(r"(?i)(\bbearer)\s+\S+"),
    re.compile(r"(?i)(cookie:\s*)[^\r\n]+"),
    re.compile(r"(?i)(--(?:access-)?token|--password|--secret|--api-key)(\s+)\S+"),
    re.compile(r"(?i)([?&](?:access_)?token=)[^&\s]+"),
    re.compile(
        r"""(?ix)
        (["']?(?:token|password|secret|api[_-]?key)["']?\s*[:=]\s*)
        ["']?[^"',\s]+
        """
    ),
)


def redact(text: str) -> str:
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(r"\1<redacted>", result)
    return result


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool
    log: str


def _managed_process_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_process_tree(
    process: subprocess.Popen[str], *, grace_seconds: float = 1.0
) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_command(
    argv: Sequence[str | os.PathLike[str]],
    *,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    timeout: float | None = None,
    cwd: Path | None = None,
    echo: bool = False,
) -> CommandResult:
    normalized = tuple(os.fspath(item) for item in argv)
    started = time.monotonic()
    timed_out = False
    process = subprocess.Popen(
        normalized,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(env) if env is not None else None,
        cwd=cwd,
        **_managed_process_kwargs(),
    )
    try:
        try:
            stdout, stderr = process.communicate(input=input_text, timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            _terminate_process_tree(process)
            stdout, stderr = process.communicate()
            stdout = stdout or error.stdout or ""
            stderr = stderr or error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            returncode = 124
            stderr += f"\ncommand timed out after {timeout} seconds\n"
    finally:
        if process.poll() is None:
            _terminate_process_tree(process)
    duration = time.monotonic() - started
    stdout = redact(stdout)
    stderr = redact(stderr)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_log = (
        f"$ {redact(' '.join(normalized))}\n"
        f"[exit={returncode} duration={duration:.6f}s]\n"
        f"--- stdout ---\n{stdout}"
        f"\n--- stderr ---\n{stderr}"
    )
    log_path.write_text(rendered_log, encoding="utf-8")
    if echo:
        print(
            f"qualification log: {log_path.as_posix()}\n{rendered_log}",
            file=sys.stderr,
            flush=True,
        )
    return CommandResult(
        argv=normalized,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        timed_out=timed_out,
        log=log_path.as_posix(),
    )


# Require a trailing non-digit so a char-by-char reader cannot latch onto a
# partial port (e.g. matching "5" before the rest of "52625" has streamed in),
# while still tolerating a trailing suffix such as " with 10 I/O threads".
_SERVE_PORT_PATTERN = re.compile(r"WebServer started on port (\d+)(?=\D)")

# A fixed, multi-clause input for the embedding probe. A longer prompt makes the
# encode-throughput measurement (tokens / request latency) less noisy than a
# three-word string while staying deterministic across runs.
_EMBED_PROBE_INPUT = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly "
    "over the distant hills and a gentle breeze moves across the quiet valley."
)

# Tokenize on words/numbers for word error rate; punctuation and case are
# normalized away so a leading "- " or trailing period does not inflate WER.
_WORD_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _wav_duration_seconds(path: Path) -> float | None:
    """Return the duration of a PCM WAV file in seconds, or ``None``.

    Used to normalize ASR latency into a real-time factor. Any malformed or
    non-WAV input degrades to ``None`` so the probe still records latency.
    """
    try:
        with wave.open(os.fspath(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except (wave.Error, OSError, EOFError):
        return None
    if not rate:
        return None
    return frames / float(rate)


def _word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Word error rate of ``hypothesis`` against ``reference`` in [0, inf).

    Case- and punctuation-insensitive Levenshtein distance over word tokens,
    divided by the reference word count. Returns ``None`` when the reference has
    no word tokens (WER would be undefined).
    """
    ref = _WORD_TOKEN_PATTERN.findall(reference.lower())
    hyp = _WORD_TOKEN_PATTERN.findall(hypothesis.lower())
    if not ref:
        return None
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            substitution = previous[j - 1] + (0 if ref_word == hyp_word else 1)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitution))
        previous = current
    return previous[-1] / len(ref)


def _transcript_contains_phrase(transcript: str, phrase: str) -> bool:
    """Whether ``phrase`` appears as a contiguous run of words in ``transcript``.

    Matching is case- and punctuation-insensitive over word tokens. This is the
    ASR pass gate: a real transcription must reproduce the pinned reference
    phrase, not merely emit any two alphabetic words. Returns ``False`` when the
    phrase has no word tokens, so an empty pin can never vacuously pass.
    """
    expected = _WORD_TOKEN_PATTERN.findall(phrase.lower())
    if not expected:
        return False
    observed = _WORD_TOKEN_PATTERN.findall(transcript.lower())
    span = len(expected)
    return any(
        observed[start : start + span] == expected
        for start in range(0, len(observed) - span + 1)
    )


def _post_audio_transcription(
    base_url: str, audio_path: Path, model: str, timeout: float
) -> str:
    """POST an audio file to the OpenAI-compatible transcription endpoint.

    Returns the transcribed ``text`` field. Raises on transport/protocol
    errors or a malformed response so the caller can record a probe error.
    """
    boundary = "----rocmNpuQualification" + uuid.uuid4().hex
    audio_bytes = audio_path.read_bytes()
    preamble = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"{model}\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; '
        f'filename="{audio_path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    closing = f"\r\n--{boundary}--\r\n".encode()
    body = preamble + audio_bytes + closing
    request = urllib.request.Request(
        f"{base_url}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return payload["text"]
    raise ValueError("transcription response did not contain a 'text' field")


def serve_probe(
    argv: Sequence[str | os.PathLike[str]],
    *,
    log_path: Path,
    env: Mapping[str, str],
    probe_kind: str,
    model: str,
    boot_timeout: float,
    http_timeout: float = 120.0,
    audio_sample: Path | None = None,
    reference_text: str | None = None,
    echo: bool = False,
) -> tuple[CommandResult, dict[str, Any]]:
    """Start ``flm serve`` for an embed/ASR model and probe the HTTP endpoint.

    For ``embed`` models a ``/v1/embeddings`` request must return a non-empty
    vector; the probe records the embedding dimension, request latency, and
    (when the server reports token usage) encode throughput. For ``asr`` models
    the audio is transcribed via ``/v1/audio/transcriptions`` and the transcript
    must contain the pinned ``reference_text`` phrase (case/punctuation
    insensitive); the probe records request latency, the real-time factor
    (latency / audio duration), and the word error rate against the reference.
    The ASR probe fails closed: a missing sample, a missing reference, or a
    transcript that omits the pinned phrase is a failure, never a partial pass.
    Returns the captured :class:`CommandResult` plus a probe detail dict whose
    ``metrics`` entry carries the modality-specific measurements above.
    """
    normalized = tuple(os.fspath(item) for item in argv)
    started = time.monotonic()
    process = subprocess.Popen(
        normalized,
        stdin=None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=dict(env),
        **_managed_process_kwargs(),
    )
    assert process.stdout is not None
    chunks: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for chunk in iter(lambda: process.stdout.read(1), ""):
                chunks.put(chunk)
        finally:
            chunks.put(None)

    reader = threading.Thread(
        target=read_output, name="qualification-serve-output", daemon=True
    )
    reader.start()
    output: list[str] = []
    port: int | None = None
    detail = ""
    ok = False
    probe_metrics: dict[str, Any] = {}
    timed_out = False
    deadline = started + boot_timeout
    try:
        while time.monotonic() < deadline and port is None:
            try:
                chunk = chunks.get(timeout=0.1)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if chunk is None:
                break
            output.append(chunk)
            match = _SERVE_PORT_PATTERN.search("".join(output))
            if match:
                port = int(match.group(1))
        if port is None:
            timed_out = process.poll() is None
            detail = "server did not report a listening port"
        else:
            base = f"http://127.0.0.1:{port}"
            time.sleep(2.0)
            try:
                if probe_kind == "embed":
                    request = urllib.request.Request(
                        f"{base}/v1/embeddings",
                        data=json.dumps(
                            {"model": model, "input": _EMBED_PROBE_INPUT}
                        ).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    request_started = time.perf_counter()
                    with urllib.request.urlopen(
                        request, timeout=http_timeout
                    ) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    request_latency = time.perf_counter() - request_started
                    vector = (payload.get("data") or [{}])[0].get("embedding")
                    if isinstance(vector, list) and vector:
                        ok = True
                        dimension = len(vector)
                        # Only latency/throughput move with the execution
                        # backend (HRX vs XRT). The embedding dimension and the
                        # input token count are fixed by the model and probe
                        # input, so they stay in the human-readable detail but
                        # out of the backend-comparison metrics.
                        probe_metrics = {"request_latency_s": request_latency}
                        summary = (
                            f"embedding dim={dimension}, "
                            f"latency={request_latency * 1000:.1f}ms"
                        )
                        usage = payload.get("usage")
                        prompt_tokens = (
                            usage.get("prompt_tokens")
                            if isinstance(usage, dict)
                            else None
                        )
                        # FLM currently returns usage.prompt_tokens=0 for
                        # /v1/embeddings. Fall back to a word-token count of
                        # the probe input so encode throughput is still
                        # measured from request latency.
                        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
                            prompt_tokens = len(
                                _WORD_TOKEN_PATTERN.findall(_EMBED_PROBE_INPUT)
                            )
                        if prompt_tokens > 0 and request_latency > 0:
                            toks_per_s = prompt_tokens / request_latency
                            probe_metrics["input_tokens"] = prompt_tokens
                            probe_metrics["encode_toks_per_s"] = toks_per_s
                            summary += f", {toks_per_s:.1f} tok/s"
                        detail = summary
                    else:
                        detail = "embeddings endpoint returned no vector"
                elif audio_sample is not None and Path(audio_sample).is_file():
                    # asr: transcribe a real audio sample end to end and require
                    # the pinned reference phrase to appear in the transcript.
                    sample = Path(audio_sample)
                    audio_seconds = _wav_duration_seconds(sample)
                    request_started = time.perf_counter()
                    transcript_text = _post_audio_transcription(
                        base, sample, model, http_timeout
                    )
                    request_latency = time.perf_counter() - request_started
                    words = _WORD_TOKEN_PATTERN.findall(transcript_text)
                    phrase_ok = reference_text is not None and (
                        _transcript_contains_phrase(transcript_text, reference_text)
                    )
                    if phrase_ok:
                        ok = True
                        # Latency, real-time factor, and WER can all shift with
                        # the execution backend (HRX vs XRT). Audio duration and
                        # transcript word count are fixed by the fixture, so they
                        # remain in the detail text but not in the metrics.
                        probe_metrics = {"request_latency_s": request_latency}
                        summary = (
                            f"transcribed {len(words)} words from {sample.name}, "
                            f"latency={request_latency * 1000:.1f}ms"
                        )
                        if audio_seconds and audio_seconds > 0:
                            real_time_factor = request_latency / audio_seconds
                            probe_metrics["real_time_factor"] = real_time_factor
                            summary += (
                                f", RTF={real_time_factor:.3f} "
                                f"({audio_seconds:.1f}s audio)"
                            )
                        wer = _word_error_rate(reference_text, transcript_text)
                        if wer is not None:
                            probe_metrics["word_error_rate"] = wer
                            summary += f", WER={wer * 100:.1f}%"
                        detail = f"{summary}: {transcript_text.strip()[:120]!r}"
                    elif reference_text is None:
                        # No pin: we cannot assert the audio was really
                        # transcribed, so fail closed rather than accept any text.
                        detail = (
                            "ASR probe has no reference transcript to verify "
                            "against; failing closed. Transcript was "
                            f"{transcript_text.strip()[:160]!r}"
                        )
                    else:
                        detail = (
                            "transcription did not contain the expected phrase "
                            f"{reference_text.strip()[:80]!r}: "
                            f"{transcript_text.strip()[:160]!r}"
                        )
                else:
                    # asr but the bundled sample could not be resolved. A
                    # listening port is not evidence that transcription works, so
                    # fail closed instead of treating a health check as success.
                    detail = (
                        "ASR sample unavailable; failing closed (bundled "
                        "asr_sample.wav is missing or ROCM_NPU_ASR_SAMPLE points "
                        "at no readable file)"
                    )
            except (urllib.error.URLError, OSError, ValueError) as error:
                detail = f"probe error: {error}"
    finally:
        if process.poll() is None:
            _terminate_process_tree(process)
        process.wait()
        reader.join(timeout=1)
        process.stdout.close()
    duration = time.monotonic() - started
    stdout = redact("".join(output))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_log = (
        f"$ {redact(' '.join(normalized))}\n"
        f"[serve probe ok={ok} port={port} duration={duration:.6f}s]\n"
        f"--- merged stdout/stderr ---\n{stdout}"
        f"\n--- probe detail ---\n{detail}\n"
    )
    log_path.write_text(rendered_log, encoding="utf-8")
    if echo:
        print(
            f"qualification log: {log_path.as_posix()}\n{rendered_log}",
            file=sys.stderr,
            flush=True,
        )
    result = CommandResult(
        argv=normalized,
        returncode=0 if ok else 1,
        stdout=stdout,
        stderr="",
        duration_seconds=duration,
        timed_out=timed_out,
        log=log_path.as_posix(),
    )
    return result, {
        "probe_kind": probe_kind,
        "port": port,
        "detail": detail,
        "ok": ok,
        "metrics": probe_metrics,
    }


def run_interactive_command(
    argv: Sequence[str | os.PathLike[str]],
    *,
    log_path: Path,
    env: Mapping[str, str],
    input_lines: Sequence[str],
    timeout: float,
    prompt_marker: str = ">>> ",
    exit_input: str = "/bye",
    echo: bool = False,
) -> CommandResult:
    normalized = tuple(os.fspath(item) for item in argv)
    started = time.monotonic()
    process = subprocess.Popen(
        normalized,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=dict(env),
        **_managed_process_kwargs(),
    )
    assert process.stdin is not None
    assert process.stdout is not None
    chunks: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for chunk in iter(lambda: process.stdout.read(1), ""):
                chunks.put(chunk)
        finally:
            chunks.put(None)

    reader = threading.Thread(
        target=read_output, name="qualification-flm-output", daemon=True
    )
    reader.start()
    deadline = started + timeout
    output: list[str] = []
    prompts_seen = 0
    lines_sent = 0
    timed_out = False
    try:
        while True:
            if time.monotonic() >= deadline and process.poll() is None:
                timed_out = True
                _terminate_process_tree(process)
            try:
                chunk = chunks.get(timeout=0.1)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    break
                continue
            if chunk is None:
                break
            output.append(chunk)
            prompt_count = "".join(output).count(prompt_marker)
            while prompts_seen < prompt_count and not process.stdin.closed:
                prompts_seen += 1
                try:
                    if lines_sent < len(input_lines):
                        process.stdin.write(input_lines[lines_sent] + "\n")
                        process.stdin.flush()
                        lines_sent += 1
                    else:
                        process.stdin.write(exit_input + "\n")
                        process.stdin.flush()
                        process.stdin.close()
                except (BrokenPipeError, OSError):
                    break
    finally:
        if not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            _terminate_process_tree(process)
        process.wait()
        reader.join(timeout=1)
        process.stdout.close()
    returncode = process.returncode
    duration = time.monotonic() - started
    stdout = redact("".join(output))
    stderr = ""
    if timed_out:
        returncode = 124
        stderr = f"command timed out after {timeout} seconds\n"
    elif lines_sent != len(input_lines):
        returncode = returncode or 2
        stderr = (
            f"command exited after receiving {lines_sent} interactive inputs; "
            f"expected {len(input_lines)}\n"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_log = (
        f"$ {redact(' '.join(normalized))}\n"
        f"[exit={returncode} duration={duration:.6f}s]\n"
        f"--- merged stdout/stderr ---\n{stdout}"
        f"\n--- runner diagnostics ---\n{stderr}"
    )
    log_path.write_text(rendered_log, encoding="utf-8")
    if echo:
        print(
            f"qualification log: {log_path.as_posix()}\n{rendered_log}",
            file=sys.stderr,
            flush=True,
        )
    return CommandResult(
        argv=normalized,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        timed_out=timed_out,
        log=log_path.as_posix(),
    )
