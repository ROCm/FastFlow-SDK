# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import re

import pytest

from tests.support.workflows import read_workflow, workflow_job

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


def _logical_test_commands(filename, job=None):
    text = read_workflow(filename)
    if job is not None:
        text = workflow_job(text, job)
    commands = set()
    for line in text.splitlines():
        if "dev.py test-" not in line:
            continue
        command = line.split("dev.py ", 1)[1].strip()
        if command.startswith(
            (
                "test-hrx-cts",
                "test-npu-package",
                "test-npu-jit",
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
        assert (
            _logical_test_commands("build_fastflow_sdk_linux.yml", "npu_e2e")
            == expected
        )
        assert (
            _logical_test_commands("build_fastflow_sdk_windows.yml", "npu_e2e")
            == expected
        )

    def test_compiler_npu_e2e_stays_execute_only(self):
        for filename in (
            "build_fastflow_sdk_linux.yml",
            "build_fastflow_sdk_windows.yml",
        ):
            e2e = workflow_job(read_workflow(filename), "npu_e2e")
            assert "fastflow-sdk-compiler-wheels-" not in e2e
            assert "fastflow-sdk-compiler-bindings-" not in e2e
            assert "dev.py install-compiler-wheels" not in e2e
            assert "test-npu-jit" not in e2e
            assert "--build-only" not in e2e

    def test_compiler_npu_examples_compile_and_run_on_primary_sku(self):
        linux = workflow_job(
            read_workflow("build_fastflow_sdk_linux.yml"), "npu_examples"
        )
        windows = workflow_job(
            read_workflow("build_fastflow_sdk_windows.yml"), "npu_examples"
        )
        assert _logical_test_commands(
            "build_fastflow_sdk_linux.yml", "npu_examples"
        ) == {"test-npu-jit"}
        assert _logical_test_commands(
            "build_fastflow_sdk_windows.yml", "npu_examples"
        ) == {"test-npu-jit"}
        for job, label in (
            (linux, "linux-gfx1150-gpu-rocm-npu"),
            (windows, "windows-gfx1151-gpu-rocm"),
        ):
            assert f"runs-on: {label}" in job
            assert "strategy:" not in job
            assert "fastflow-sdk-compiler-wheels-" in job
            assert "npu-prereqs-" in job
            assert "npu-test-package-" not in job
            assert "dev.py install-compiler-wheels" in job
            assert "dev.py npu-probe" in job
            assert "--build-only" not in job
            assert "timeout-minutes: 30" in job
        assert "linux-gfx1151-gpu-rocm" not in linux
        image = (
            "ghcr.io/rocm/therock_build_manylinux_x86_64@"
            "sha256:702a5133851e6d1daf1207d2c9fbb01c2667914a5b6dc5a01faeb3ce66ea6421"
        )
        assert image in linux
        assert "--device=/dev/accel/accel0" in linux
        assert "python-version: '3.12.10'" in windows

    def test_compiler_workflows_do_not_ship_bindings_only_jit_surrogate(self):
        for filename in (
            "build_fastflow_sdk_linux.yml",
            "build_fastflow_sdk_windows.yml",
        ):
            text = read_workflow(filename)
            assert "--bindings-only" not in text
            assert "--from-test-package" not in text
            assert "jit_smoke_run" not in text
            assert "iron-jit/" not in text

    def test_fastflowlm_npu_matrices_use_kernel_and_fast_qualification(self):
        # Kernel qualification is part of `fastflow qualify run`; no separate
        # helper artifact or legacy dev.py test is allowed in the NPU jobs.
        expected = set()
        for filename in (
            "fastflowlm_hrx_linux.yml",
            "fastflowlm_hrx_windows.yml",
        ):
            text = read_workflow(filename)
            assert _logical_test_commands(filename) == expected
            assert "npu_examples:" not in text
            assert "qualify run" in text
            assert "test-fastflowlm-kernels" not in text
            assert re.search(r"fastflow(?:\.exe)? qualify", text)
            assert "--fast" in text
            assert "--candidate-manifest" not in text
            assert re.search(r"fastflow(?:\.exe)? qualify verify", text)
            assert "qualify candidate" not in text
            assert "qualify intake" not in text
            assert "--bundle-output .qualification-run.evidence.zip" in text
            assert "test-fastflowlm --wheel-package" not in text

    def test_all_declared_runner_labels_are_covered_by_shared_matrix_jobs(self):
        compiler_labels = _runner_labels(
            "build_fastflow_sdk_linux.yml"
        ) | _runner_labels("build_fastflow_sdk_windows.yml")
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
                ("build_fastflow_sdk_linux.yml", "fastflowlm_hrx_linux.yml"),
            ),
            (
                "windows",
                (
                    "build_fastflow_sdk_windows.yml",
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
