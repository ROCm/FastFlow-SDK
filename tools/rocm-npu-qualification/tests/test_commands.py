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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools" / "rocm-npu-qualification" / "python"))

from rocm_npu_qualification import commands  # noqa: E402


pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def _tree_command(sentinel: Path, *, interactive: bool) -> list[str]:
    child = (
        "import pathlib,time;"
        "time.sleep(3.0);"
        f"pathlib.Path({str(sentinel)!r}).write_text('orphan',encoding='utf-8')"
    )
    parent = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
        + ("print('>>> ',end='',flush=True);" if interactive else "")
        + "time.sleep(10)"
    )
    return [sys.executable, "-c", parent]


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
    time.sleep(3.1)
    assert not sentinel.exists()


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
