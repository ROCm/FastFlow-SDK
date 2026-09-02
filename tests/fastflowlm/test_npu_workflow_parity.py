# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import re

import pytest

from tests.support.workflows import read_workflow

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


def _logical_test_commands(filename):
    text = read_workflow(filename)
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
    text = read_workflow(filename)
    return set(
        re.findall(
            r"(?:linux|windows)-gfx\d+-gpu-rocm(?:-npu)?",
            text,
        )
    )


class TestNpuWorkflowParity:
    def test_compiler_npu_matrices_use_identical_hrx_and_xadx_commands(self):
        # Execute-only XADX + HRX CTS. Do not add test-npu-iron-ops: that is
        # local IRON pytest and is not part of NPU CI.
        expected = {"test-hrx-cts", "test-npu-package"}
        assert _logical_test_commands("build_rocm_npu_staging_linux.yml") == expected
        assert _logical_test_commands("build_rocm_npu_staging_windows.yml") == expected

    def test_fastflowlm_npu_matrices_use_kernel_and_fast_qualification(self):
        # Kernel qualification is part of `rocm-npu qualify run`; no separate
        # helper artifact or legacy dev.py test is allowed in the NPU jobs.
        expected = set()
        for filename in (
            "fastflowlm_hrx_linux.yml",
            "fastflowlm_hrx_windows.yml",
        ):
            text = read_workflow(filename)
            assert _logical_test_commands(filename) == expected
            assert "qualify run" in text
            assert "test-fastflowlm-kernels" not in text
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

    @pytest.mark.parametrize(
        ("platform", "workflow_names"),
        (
            (
                "linux",
                ("build_rocm_npu_staging_linux.yml", "fastflowlm_hrx_linux.yml"),
            ),
            (
                "windows",
                (
                    "build_rocm_npu_staging_windows.yml",
                    "fastflowlm_hrx_windows.yml",
                ),
            ),
        ),
    )
    def test_npu_matrices_support_platform_runner_variables(
        self, platform, workflow_names
    ):
        variable = f"vars.ROCM_NPU_{platform.upper()}_RUNNER_LABELS"
        expected_labels = _runner_labels(workflow_names[0])
        for workflow_name in workflow_names:
            workflow = read_workflow(workflow_name)
            assert f"fromJSON({variable} ||" in workflow
            assert _runner_labels(workflow_name) == expected_labels
