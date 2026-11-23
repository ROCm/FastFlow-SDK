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

COLLECTOR = "collect_fastflow_sdk_ci_release.yml"
RELEASE = "release_fastflow_sdk_pypi_wheels.yml"
VERIFY = "verify_fastflow_sdk_pypi_wheels.yml"
ARTIFACT = "fastflow-sdk-pypi-wheel-"
REPACK_STEPS = (
    "for platform in linux windows; do\n"
    '            python dev.py package-pypi-wheel --release-dir release --platform "${platform}" --dir pypi\n',
    "python dev.py audit-pypi-wheel pypi/fastflow_sdk-*-manylinux_*_x86_64.whl",
    "python -m twine check --strict pypi/*.whl",
    "requirements/packaging-linux.lock",
    f"name: {ARTIFACT}linux\n",
    "path: pypi/fastflow_sdk-*-manylinux_*_x86_64.whl*",
    f"name: {ARTIFACT}windows\n",
    "path: pypi/fastflow_sdk-*-win_amd64.whl*",
)


@pytest.mark.parametrize("workflow", [COLLECTOR, RELEASE])
def test_ci_and_release_repack_with_the_same_steps(workflow):
    text = read_workflow(workflow)

    positions = [text.index(step) for step in REPACK_STEPS[:3]]
    assert positions == sorted(positions)
    for step in REPACK_STEPS:
        assert step in text, step


def test_collector_repacks_only_after_uploading_the_unchanged_release():
    collector = workflow_job(read_workflow(COLLECTOR), "collect")
    release_upload = collector.index("path: release/**")

    assert release_upload < collector.index("dev.py package-pypi-wheel")
    assert "--dir pypi" in collector


def test_reusable_verify_workflow_inherits_caller_permissions():
    workflow = read_workflow(VERIFY)

    assert "workflow_call:" in workflow.split("\njobs:\n", 1)[0]
    assert "permissions:" not in workflow


def test_ci_installs_the_wheels_only_after_merge():
    job = workflow_job(read_workflow("ci.yml"), "pypi_wheels")

    assert f"uses: ./.github/workflows/{VERIFY}" in job
    assert "needs: collect_releases" in job
    assert "if: github.event_name != 'pull_request'" in job


def test_release_entry_point_is_manual_attests_and_verifies():
    workflow = read_workflow(RELEASE)
    triggers = workflow.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    build = workflow_job(workflow, "build")
    verify = workflow_job(workflow, "verify")

    assert re.findall(r"(?m)^  (\w+):", triggers) == ["workflow_dispatch"]
    assert "id-token: write" in build
    assert "attestations: write" in build
    assert "subject-path: pypi/*.whl" in build
    assert build.index("twine check") < build.index("actions/attest-build-provenance@")
    assert "needs: build" in verify
    assert f"uses: ./.github/workflows/{VERIFY}" in verify


def test_published_releases_are_attested_before_repacking():
    build = workflow_job(read_workflow(RELEASE), "build")
    download = build.index("gh release download")
    verify = build.index("gh attestation verify")
    package = build.index("dev.py package-pypi-wheel")

    assert download < verify < package
    assert "--pattern '*.whl' --pattern SHA256SUMS" in build


@pytest.mark.parametrize(
    ("job", "artifact", "command"),
    [
        (
            "install",
            "${{ matrix.platform }}",
            "python dev.py verify-pypi-wheel dist\n",
        ),
        (
            "npu_linux",
            "linux",
            "python3 dev.py verify-pypi-wheel dist --require-hardware",
        ),
        (
            "npu_windows",
            "windows",
            "python dev.py verify-pypi-wheel dist --require-hardware",
        ),
    ],
)
def test_every_install_job_verifies_its_platform_wheel(job, artifact, command):
    block = workflow_job(read_workflow(VERIFY), job)

    assert f"name: {ARTIFACT}{artifact}\n" in block
    assert command in block


def test_install_matrix_covers_the_supported_python_range():
    install = workflow_job(read_workflow(VERIFY), "install")

    for runner in ("ubuntu-24.04", "azure-windows-scale-rocm"):
        assert install.count(f"runner: {runner}") == 2
    for platform in ("linux", "windows"):
        assert install.count(f"platform: {platform}\n") == 2
    assert install.count('python: "3.14"') == 2
    assert 'python: "3.12"' in install
    assert 'python: "3.12.10"' in install


def test_linux_npu_job_runs_in_a_manylinux_2_28_container():
    linux = read_workflow("build_fastflow_sdk_linux.yml")
    image = re.search(r"image: (ghcr\.io/rocm/therock_build_manylinux\S+)", linux)
    npu = workflow_job(read_workflow(VERIFY), "npu_linux")

    assert image is not None
    assert f"image: {image.group(1)}" in npu
    assert "--device=/dev/accel/accel0" in npu
