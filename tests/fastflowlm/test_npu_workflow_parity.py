# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


def _logical_test_commands(filename):
    text = (WORKFLOWS / filename).read_text(encoding="utf-8")
    commands = set()
    for line in text.splitlines():
        if "dev.py test-" not in line:
            continue
        command = line.split("dev.py ", 1)[1].strip()
        if command.startswith(
            (
                "test-hrx-cts",
                "test-npu-package",
                "test-fastflowlm-kernels",
                "test-fastflowlm",
            )
        ):
            commands.add(command)
    return commands


def _runner_labels(filename):
    text = (WORKFLOWS / filename).read_text(encoding="utf-8")
    return set(
        re.findall(
            r"^\s+- ((?:linux|windows)-gfx\d+-gpu-rocm(?:-npu)?)\s*$",
            text,
            re.MULTILINE,
        )
    )


class TestNpuWorkflowParity:
    def test_compiler_npu_matrices_use_identical_hrx_and_xadx_commands(self):
        expected = {"test-hrx-cts", "test-npu-package"}
        assert _logical_test_commands("build_rocm_npu_staging_linux.yml") == expected
        assert _logical_test_commands("build_rocm_npu_staging_windows.yml") == expected

    def test_fastflowlm_npu_matrices_use_kernel_and_fast_qualification(self):
        expected = {
            (
                "test-fastflowlm-kernels --wheel-package "
                "--dir .kernel-artifact --iterations 20 --readback-iterations 32"
            ),
        }
        for filename in (
            "fastflowlm_hrx_linux.yml",
            "fastflowlm_hrx_windows.yml",
        ):
            text = (WORKFLOWS / filename).read_text(encoding="utf-8")
            assert _logical_test_commands(filename) == expected
            assert re.search(r"rocm-npu(?:\.exe)? qualify", text)
            assert "--fast" in text
            assert "--candidate-manifest" not in text
            assert re.search(r"rocm-npu(?:\.exe)? qualify verify", text)
            assert "qualify candidate" not in text
            assert "qualify intake" not in text
            assert "--bundle-output .qualification-run.evidence.zip" in text
            assert "test-fastflowlm --wheel-package" not in text

    def test_all_declared_runner_labels_are_covered_by_shared_matrix_jobs(self):
        compiler_labels = _runner_labels(
            "build_rocm_npu_staging_linux.yml"
        ) | _runner_labels("build_rocm_npu_staging_windows.yml")
        assert compiler_labels == {
            "linux-gfx1150-gpu-rocm-npu",
            "linux-gfx1151-gpu-rocm",
            "linux-gfx1152-gpu-rocm",
            "linux-gfx1153-gpu-rocm",
            "windows-gfx1151-gpu-rocm",
            "windows-gfx1152-gpu-rocm",
        }
