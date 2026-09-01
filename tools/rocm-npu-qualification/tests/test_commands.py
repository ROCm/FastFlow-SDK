# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise managed qualification subprocess cleanup."""

from __future__ import annotations

import os
import queue
import sys
import time
from pathlib import Path

import pytest

from rocm_npu_qualification import commands

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def _tree_command(sentinel: Path, *, interactive: bool) -> list[str]:
    child = (
        "import pathlib,time;"
        f"p=pathlib.Path({str(sentinel)!r});"
        "[(p.open('a',encoding='utf-8').write('x'),time.sleep(.01))"
        " for _ in range(1000)]"
    )
    parent = (
        "import pathlib,subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
        f"p=pathlib.Path({str(sentinel)!r});"
        "deadline=time.monotonic()+2;"
        "\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        + ("print('>>> ',end='',flush=True);" if interactive else "")
        + "time.sleep(10)"
    )
    return [sys.executable, "-c", parent]


def _assert_file_stops_changing(path: Path, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    previous_size = path.stat().st_size
    while time.monotonic() < deadline:
        time.sleep(0.01)
        current_size = path.stat().st_size
        if current_size != previous_size:
            pytest.fail("descendant process continued writing after tree cleanup")


@pytest.mark.parametrize("interactive", (False, True))
def test_timeout_terminates_descendant_process_tree(tmp_path, interactive):
    sentinel = tmp_path / "orphan.txt"
    argv = _tree_command(sentinel, interactive=interactive)
    if interactive:
        result = commands.run_interactive_command(
            argv,
            log_path=tmp_path / "interactive.log",
            env=os.environ,
            input_lines=[],
            timeout=0.2,
        )
    else:
        result = commands.run_command(
            argv,
            log_path=tmp_path / "command.log",
            timeout=0.2,
        )
    assert result.returncode == 124
    assert result.timed_out
    assert sentinel.is_file()
    _assert_file_stops_changing(sentinel)


def test_interruption_always_invokes_tree_cleanup(monkeypatch, tmp_path):
    class InterruptQueue(queue.Queue):
        def get(self, *args, **kwargs):
            raise KeyboardInterrupt

    calls = []
    terminate = commands._terminate_process_tree

    def track(process, **kwargs):
        calls.append(process.pid)
        terminate(process, **kwargs)

    monkeypatch.setattr(commands.queue, "Queue", InterruptQueue)
    monkeypatch.setattr(commands, "_terminate_process_tree", track)
    with pytest.raises(KeyboardInterrupt):
        commands.run_interactive_command(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            log_path=tmp_path / "interrupt.log",
            env=os.environ,
            input_lines=[],
            timeout=30,
        )
    assert len(calls) == 1
