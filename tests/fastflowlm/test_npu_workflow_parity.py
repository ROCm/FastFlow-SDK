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

    def test_fastflowlm_npu_matrices_use_identical_kernel_and_soak_commands(self):
        expected = {
            (
                "test-fastflowlm-kernels --wheel-package "
                "--dir .kernel-artifact --iterations 20 --readback-iterations 32"
            ),
            "test-fastflowlm --wheel-package --inference-iterations 5",
        }
        assert _logical_test_commands("fastflowlm_hrx_linux.yml") == expected
        assert _logical_test_commands("fastflowlm_hrx_windows.yml") == expected

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
