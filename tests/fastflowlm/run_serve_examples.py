#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the examples/serve scripts against `flm serve` on this machine's NPU.

Start each server the way the scripts' docstrings tell a user to, run the
scripts unchanged with the interpreter running this file, and check what they
print. The FastFlowLM NPU jobs run it after qualification, which leaves
qwen3:0.6b and gemma4-it:e2b in the model cache; flm downloads the embedding
model on first use.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "serve"
SERVE_PORT = re.compile(r"WebServer started on port (\d+)\D")
BOOT_SECONDS = 900
SCRIPT_SECONDS = 600


@dataclass(frozen=True)
class Script:
    name: str
    # Must match what the script prints.
    expect: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Server:
    args: tuple[str, ...]
    scripts: tuple[Script, ...]


SERVERS = (
    Server(
        ("qwen3:0.6b", "--embed", "1"),
        (
            Script("chat.py", r"\w"),
            Script("embeddings.py", r"\d+ dimensions, cosine similarity"),
        ),
    ),
    Server(
        ("gemma4-it:e2b",),
        (Script("tool_call.py", r"tool call: get_weather\("),),
    ),
)
# Scripts CI does not run, and why.
NOT_RUN = {"transcribe.py": "Whisper would be a 1 GB download on every run"}


class Serve:
    """One `flm serve` process tree, its merged output, and the port it reports."""

    def __init__(self, flm: str, args: tuple[str, ...], env: dict[str, str]):
        self.command = [flm, "serve", *args]
        self.output: list[str] = []
        self._ready = threading.Event()
        self.port: int | None = None
        group = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            **group,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        # By character: flm may not end the port line until much later.
        assert self.process.stdout is not None
        tail = ""
        for char in iter(lambda: self.process.stdout.read(1), ""):
            self.output.append(char)
            if self.port is None:
                tail = (tail + char)[-200:]
                if match := SERVE_PORT.search(tail):
                    self.port = int(match.group(1))
                    self._ready.set()
        self._ready.set()

    def wait_ready(self, seconds: float) -> int:
        if not self._ready.wait(seconds) or self.port is None:
            raise RuntimeError(f"{' '.join(self.command)} reported no port")
        return self.port

    def _signal_group(self, number: int) -> None:
        # The server can exit between poll() and the signal.
        try:
            os.killpg(self.process.pid, number)
        except ProcessLookupError:
            pass

    def stop(self) -> None:
        if self.process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                )
            else:
                self._signal_group(signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    self._signal_group(signal.SIGKILL)
                self.process.kill()
                self.process.wait()
        self._reader.join(timeout=5)


def run_server(flm: str, server: Server, env: dict[str, str]) -> list[str]:
    failures = []
    serve = Serve(flm, server.args, env)
    try:
        port = serve.wait_ready(BOOT_SECONDS)
        script_env = {
            **env,
            "FLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "PYTHONIOENCODING": "utf-8",
        }
        for script in server.scripts:
            print(f"== {script.name} (flm serve {' '.join(server.args)})", flush=True)
            started = time.monotonic()
            result = subprocess.run(
                [sys.executable, str(EXAMPLES / script.name), *script.args],
                env=script_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=SCRIPT_SECONDS,
            )
            print(result.stdout + result.stderr, end="", flush=True)
            took = time.monotonic() - started
            if result.returncode != 0:
                failures.append(f"{script.name} exited {result.returncode}")
            elif not re.search(script.expect, result.stdout):
                failures.append(f"{script.name} did not print {script.expect!r}")
            else:
                print(f"-- {script.name}: ok ({took:.1f}s)", flush=True)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        failures.append(str(error))
    finally:
        serve.stop()
    if failures:
        print("".join(serve.output)[-20000:], end="", file=sys.stderr)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", type=Path, required=True)
    options = parser.parse_args(argv)
    flm = shutil.which("flm", path=str(Path(sys.executable).parent))
    if flm is None:
        print(f"flm is not installed beside {sys.executable}", file=sys.stderr)
        return 1
    env = {**os.environ, "FLM_MODEL_PATH": str(options.model_cache.resolve())}
    failures = []
    for server in SERVERS:
        failures += run_server(flm, server, env)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    print("serve examples: " + ("FAIL" if failures else "ok"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
