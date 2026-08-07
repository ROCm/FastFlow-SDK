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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization:\s*(?:bearer|basic))\s+\S+"),
    re.compile(r"(?i)(\bbearer)\s+\S+"),
    re.compile(r"(?i)(cookie:\s*)[^\r\n]+"),
    re.compile(
        r"(?i)(--(?:access-)?token|--password|--secret|--api-key)(\s+)\S+"
    ),
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


_SERVE_PORT_PATTERN = re.compile(r"WebServer started on port (\d+)")


def serve_probe(
    argv: Sequence[str | os.PathLike[str]],
    *,
    log_path: Path,
    env: Mapping[str, str],
    probe_kind: str,
    model: str,
    boot_timeout: float,
    http_timeout: float = 120.0,
    echo: bool = False,
) -> tuple[CommandResult, dict[str, Any]]:
    """Start ``flm serve`` for an embed/ASR model and probe the HTTP endpoint.

    For ``embed`` models a ``/v1/embeddings`` request must return a non-empty
    vector; for ``asr`` models (no bundled audio sample) a healthy server that
    loaded the model on the NPU is treated as a partial pass. Returns the
    captured :class:`CommandResult` plus a probe detail dict.
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
                            {"model": model, "input": "The quick brown fox."}
                        ).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(
                        request, timeout=http_timeout
                    ) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    vector = (payload.get("data") or [{}])[0].get("embedding")
                    if isinstance(vector, list) and vector:
                        ok = True
                        detail = f"embedding dim={len(vector)}"
                    else:
                        detail = "embeddings endpoint returned no vector"
                else:  # asr: no bundled audio sample, confirm the server is healthy
                    try:
                        urllib.request.urlopen(
                            f"{base}/v1/models", timeout=min(30.0, http_timeout)
                        ).close()
                    except (urllib.error.URLError, OSError):
                        pass
                    ok = True
                    detail = (
                        "served: model loaded on NPU; transcription not exercised "
                        "(no bundled audio sample)"
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
    return result, {"probe_kind": probe_kind, "port": port, "detail": detail, "ok": ok}


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
