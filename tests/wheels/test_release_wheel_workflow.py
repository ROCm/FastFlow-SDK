# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import re

import pytest

from tests.support.workflows import (
    WORKFLOWS,
    read_workflow,
    workflow_job as _job,
)


pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


class TestRocmNpuReleaseWheelWorkflow:
    def test_linux_vcpkg_identity_does_not_trigger_pipefail_sigpipe(self):
        workflow = read_workflow("fastflowlm_hrx_linux.yml")
        assert "ldd --version | head" not in workflow
        assert 'ldd_version="$(ldd --version 2>&1)"' in workflow
        assert 'ldd_version="${ldd_version%%$\'\\n\'*}"' in workflow

    def test_platform_builds_generate_matching_qualification_wheels(self):
        assert not (WORKFLOWS / "build_qualification_wheel.yml").exists()
        linux = read_workflow("fastflowlm_hrx_linux.yml")
        windows = read_workflow("fastflowlm_hrx_windows.yml")
        for workflow in (linux, windows):
            generate = workflow.index("generate-fastflowlm-xadx --fresh")
            package = workflow.index("package-qualification-wheel")
            assert generate < package
            assert "--xadx-assets build/fastflowlm-xadx-assets" in workflow
            install = workflow.split(
                "- name: Install qualification plugin wheel and numeric runtime", 1
            )[1].split("- name: Probe NPU driver", 1)[0]
            assert "--no-deps" not in install
            assert "--no-index" not in install
            assert "test-fastflowlm-kernels --wheel-package" not in workflow
            assert "fastflowlm-kernel-smoke-" not in workflow
            assert "if: ${{ inputs.qualification_artifact == '' }}" not in workflow
        assert "rocm-npu-qualification-wheel'" in linux
        assert (
            "build/qualification-artifacts/rocm_npu_qualification-*.whl"
            in linux
        )
        assert "rocm-npu-qualification-wheel-windows-local" in windows
        assert "verify_qualification_wheel_parity.py" in windows

    def test_hidden_qualification_evidence_is_uploaded(self):
        for name in ("fastflowlm_hrx_linux.yml", "fastflowlm_hrx_windows.yml"):
            workflow = read_workflow(name)
            assert workflow.count("include-hidden-files: true") == 2
            assert (
                "Run bounded kernel and model qualification through native dispatch"
                in workflow
            )

    def test_release_dependency_closure_uses_tested_helper(self):
        workflow = read_workflow("release_rocm_npu_wheels.yml")
        section = workflow[
            workflow.index("Resolve qualification dependencies") :
            workflow.index("Resolve compiler core")
        ]
        assert (
            "python -m build_tools.verify_qualification_dependencies" in section
        )
        assert "--no-cache-dir" in section
        assert "--no-index" in section
        assert "--no-deps" in section
        assert '"${numeric_requirements[@]}"' in section
        assert "numeric-runtime.json" in section
        assert 'rm -rf "${target}"' in section
        assert "ROCM_NPU_INSTALL_TARGET" not in section

    def test_release_compiler_and_ops_resolution_uses_tested_helper(self):
        workflow = read_workflow("release_rocm_npu_wheels.yml")
        compiler_resolution = workflow.split(
            "- name: Resolve compiler core and bindings", 1
        )[1].split("- name: Resolve ops wheel", 1)[0]
        ops_resolution = workflow.split("- name: Resolve ops wheel", 1)[1].split(
            "- name: Upload release payload", 1
        )[0]
        assert (
            "python -m build_tools.resolve_release_wheel" in compiler_resolution
        )
        assert "--product compiler" in compiler_resolution
        assert '"${compiler_meta}"' in compiler_resolution
        assert "--find-links release" in compiler_resolution
        assert "--no-index" not in compiler_resolution
        assert "COMPILER_BINDINGS_RUNTIME_REQUIREMENTS" not in compiler_resolution
        assert "--abi abi3" not in compiler_resolution
        assert (
            "python -m build_tools.resolve_release_wheel" in ops_resolution
        )
        assert "--product ops" in ops_resolution
        assert '"${ops_wheel}"' in ops_resolution
        assert "rocm-npu-ops==${ROCM_NPU_PACKAGE_VERSION}" not in ops_resolution
        assert "--abi abi3" not in ops_resolution

    def test_release_platform_jobs_validate_their_own_qualification_wheels(self):
        ci = read_workflow("ci.yml")
        assert ci.count(
            "qualification_artifact: rocm-npu-qualification-wheel"
        ) == 1

        release = read_workflow("release_rocm_npu_wheels.yml")
        assert release.count(
            "qualification_artifact: rocm-npu-qualification-wheel"
        ) == 2
        for job in ("windows_wheels", "windows_publish_wheels"):
            section = _job(release, job)
            assert "qualification_artifact:" not in section
            assert "build_artifacts: true" in section
            assert "run_npu_e2e: true" in section
        assert "windows_wheels_npu:" not in release
        assert "windows_publish_wheels_npu:" not in release
        assert "rocm-npu-qualification-wheel-windows-local" in release
        assert "verify_qualification_wheel_parity.py" in release

    def test_ci_runs_presubmit_before_expensive_jobs(self):
        workflow = read_workflow("ci.yml")

        assert "presubmit:" in workflow
        assert "actions/setup-go@924ae3a1cded613372ab5595356fb5720e22ba16" in workflow
        assert "go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12" in workflow
        assert "ruff check --select F821 dev.py build_tools tests" in workflow
        assert "rocm_npu_wheel_release_cli.py resolve" in workflow
        assert '--publish-target none' in workflow
        assert "default_package_version" not in workflow
        for name in ("linux", "windows", "linux_flm", "windows_flm"):
            assert "needs: presubmit" in _job(workflow, name)
            assert "run_npu_e2e: true" in _job(workflow, name)
        assert "linux_npu:" not in workflow
        assert "windows_npu:" not in workflow
        assert "linux_flm_npu:" not in workflow
        assert "windows_flm_npu:" not in workflow
        assert (
            "uses: ./.github/workflows/fastflowlm_hrx_linux.yml"
            in _job(workflow, "linux_flm")
        )
        assert (
            "uses: ./.github/workflows/fastflowlm_hrx_windows.yml"
            in _job(workflow, "windows_flm")
        )

    def test_pr_ci_compares_independently_generated_qualification_wheels(self):
        workflow = read_workflow("ci.yml")
        collect = _job(workflow, "collect_releases")
        collector = read_workflow("collect_rocm_npu_ci_release.yml")
        assert "needs:" in collect
        for producer in ("linux", "windows", "linux_flm", "windows_flm"):
            assert f"- {producer}" in collect
        assert "always() &&" in collect
        assert "!cancelled()" in collect
        assert (
            "uses: ./.github/workflows/collect_rocm_npu_ci_release.yml" in collect
        )
        assert (
            "package_version: ${{ needs.presubmit.outputs.package_version }}" in collect
        )
        assert "name: rocm-npu-qualification-wheel" in collector
        assert "name: rocm-npu-qualification-wheel-windows-local" in collector
        assert "name: rocm-npu-linux-x86_64" in collector
        assert "name: rocm-npu-windows-x86_64" in collector
        assert "verify_qualification_wheel_parity.py" in collector
        assert "rocm_npu_wheel_release_cli.py prepare" in collector
        assert "Create release-wide qualification candidate" in collector
        assert "download_qualification_numeric_wheels.py" not in collector
        assert (
            "Resolve qualification dependencies for every supported target"
            in collector
        )
        assert "--no-cache-dir" in collector
        assert "--no-index" in collector
        assert '"${numeric_requirements[@]}"' in collector
        assert 'rm -rf "${target}"' in collector
        assert "rm -rf wheelhouse archives qualification-parity" in collector
        assert "numeric-runtime-cp312.lock" not in collector

    def test_release_is_dispatch_only_and_reuses_both_npu_validations(self):
        workflow = read_workflow("release_rocm_npu_wheels.yml")

        assert "workflow_dispatch:" in workflow
        assert "pull_request:" not in workflow
        assert "push:" not in workflow
        assert "schedule:" not in workflow
        assert "uses: ./.github/workflows/fastflowlm_hrx_linux.yml" in workflow
        assert "uses: ./.github/workflows/fastflowlm_hrx_windows.yml" in workflow
        reusable_jobs = {
            "linux_wheels": "fastflowlm_hrx_linux.yml",
            "windows_wheels": "fastflowlm_hrx_windows.yml",
            "linux_publish_wheels": "fastflowlm_hrx_linux.yml",
            "windows_publish_wheels": "fastflowlm_hrx_windows.yml",
            "linux_compiler_wheels": "build_rocm_npu_staging_linux.yml",
            "windows_compiler_wheels": "build_rocm_npu_staging_windows.yml",
            "linux_publish_compiler_wheels": "build_rocm_npu_staging_linux.yml",
            "windows_publish_compiler_wheels": "build_rocm_npu_staging_windows.yml",
        }
        for job, reusable in reusable_jobs.items():
            section = _job(workflow, job)
            assert f"uses: ./.github/workflows/{reusable}" in section
            assert "ref: ${{ needs.prepare.outputs.source_sha }}" in section
        assert (
            "package_version: ${{ needs.prepare.outputs.package_version }}" in workflow
        )
        assert "ref: ${{ needs.prepare.outputs.source_sha }}" in _job(
            workflow, "collect"
        )
        assert "source_sha: ${{ steps.source.outputs.sha }}" in workflow
        assert "git merge-base --is-ancestor" in workflow
        assert '"${GITHUB_REF}" != refs/heads/main' in workflow
        assert "if: ${{ inputs.publish_target == 'github-prerelease' }}" in workflow
        assert "fetch-depth: 0" in workflow
        assert "--release-platform" not in workflow
        assert '--publish-target "${INPUT_PUBLISH_TARGET}"' in workflow
        assert "rocm-npu-runtime-fastflowlm-linux-x86_64" in workflow
        assert "rocm-npu-runtime-fastflowlm-windows-x86_64" in workflow
        assert "rocm-npu-linux-x86_64" in workflow
        assert "rocm-npu-windows-x86_64" in workflow
        assert "--archive-dir archives" in workflow
        assert "release/*.zip,release/*.tar.zst" in workflow
        assert "release/*.zip.manifest.json" in workflow
        assert "release/*.tar.zst.manifest.json" in workflow
        assert "rocm-npu-compiler-wheels-linux-x86_64" in workflow
        assert "rocm-npu-compiler-wheels-windows-x86_64" in workflow
        assert "rocm-npu-qualification-wheel" in workflow
        assert "build_qualification_wheel.yml" not in workflow
        assert "path: wheelhouse/qualification" in workflow
        assert '"${SOURCE_SHA}" != "${GITHUB_SHA}"' in workflow
        assert '"${qualification_wheel}"' in workflow
        assert "for python_minor in 3.11 3.12 3.13 3.14" in workflow
        assert "download_qualification_numeric_wheels.py" not in workflow
        assert "numeric-runtime-cp312.lock" not in workflow
        assert "python -m twine check --strict wheelhouse/*/*.whl" in workflow
        assert "python build_tools/rocm_npu_wheel_release_cli.py" in workflow
        assert "--expected-wheel-tag py3-none-manylinux_2_28_x86_64" in workflow
        assert "--expected-wheel-tag py3-none-linux_x86_64" in workflow
        assert "--expected-wheel-tag py3-none-win_amd64" in workflow
        assert "--expected-wheel-tag py3-none-any" in workflow
        assert "--expected-wheel-tag cp312-abi3-manylinux_2_28_x86_64" in workflow
        assert "--expected-wheel-tag cp312-abi3-win_amd64" in workflow
        assert "-cp313-" not in workflow
        assert "-cp314-" not in workflow
        assert "--platform linux_x86_64" in workflow
        assert "--platform manylinux_2_28_x86_64" in workflow
        assert "python -m build_tools.verify_qualification_dependencies" in workflow
        assert "python -m build_tools.resolve_release_wheel" in workflow
        assert "rm -rf wheelhouse archives qualification-parity" in workflow
        assert (
            "ncipollo/release-action@339a81892b84b4eeb0f6e744e4574d79d0d9b8dd"
            in workflow
        )
        assert "commit: ${{ needs.prepare.outputs.source_sha }}" in workflow
        assert "needs.collect.outputs.source_sha" not in workflow
        for name in (
            "linux_wheels",
            "windows_wheels",
            "linux_compiler_wheels",
            "windows_compiler_wheels",
        ):
            section = _job(workflow, name)
            assert "attest_wheels: false" in section
            assert "id-token: write" not in section
            assert "attestations: write" not in section
        for name in (
            "linux_publish_wheels",
            "windows_publish_wheels",
            "linux_publish_compiler_wheels",
            "windows_publish_compiler_wheels",
        ):
            section = _job(workflow, name)
            assert "attest_wheels: true" in section
            assert "id-token: write" in section
            assert "attestations: write" in section
        assert "always() &&" in workflow
        assert "actions/attest-build-provenance@" not in workflow

        linux_validation = workflow.split("\n  linux_wheels:", 1)[1].split(
            "\n  windows_wheels:", 1
        )[0]
        windows_validation = workflow.split("\n  windows_wheels:", 1)[1].split(
            "\n  linux_publish_wheels:", 1
        )[0]
        for validation in (linux_validation, windows_validation):
            assert "if: ${{ inputs.publish_target == 'none' }}" in validation
            assert "id-token: write" not in validation
            assert "attestations: write" not in validation
        publish = workflow.split("\n  publish_github_prerelease:", 1)[1]
        assert "vars.ROCM_NPU_ENABLE_GITHUB_PRERELEASE == 'true'" in publish
        assert "Create draft GitHub prerelease (disabled)" in publish

    def test_qualification_candidate_is_release_wide(self):
        for name in ("fastflowlm_hrx_linux.yml", "fastflowlm_hrx_windows.yml"):
            workflow = read_workflow(name)
            assert "qualify candidate" not in workflow
            assert "qualify verify" in workflow
            assert "qualify intake" not in workflow
        release = read_workflow("release_rocm_npu_wheels.yml")
        assert "Create release-wide qualification candidate" in release
        assert "--wheel-dir release" in release
        assert "--output release/candidate-manifest.json" in release
        assert "release/candidate-manifest.json" in release
        assert "draft: true" in release
        assert "immutableCreate: true" not in release
        publish = release.split("\n  publish_github_prerelease:", 1)[1]
        assert "vars.ROCM_NPU_ENABLE_GITHUB_PRERELEASE == 'true'" in publish

    def test_release_uses_neutral_public_names_and_shared_release_lock(self):
        workflow = read_workflow("release_rocm_npu_wheels.yml")

        assert (
            "rocm-npu-wheels-${{ needs.prepare.outputs.package_version }}" in workflow
        )
        assert "tag: ${{ needs.prepare.outputs.tag_name }}" in workflow
        assert "name: ${{ needs.prepare.outputs.release_name }}" in workflow
        assert "environment: rocm-npu-prerelease" in workflow
        assert "requirements/release-wheels.lock" in workflow
        assert "0.1.0" not in workflow
        assert "rocm-npu-staging" not in workflow
        assert not (WORKFLOWS / "release_rocm_npu_wheels_windows.yml").exists()

    @pytest.mark.parametrize("platform", ("linux", "windows"))
    def test_platform_wheel_ci_accepts_release_ref_and_version_inputs(self, platform):
        self._assert_platform_workflow_contract(platform)

    @pytest.mark.parametrize("platform", ("linux", "windows"))
    def test_compiler_workflows_inherit_caller_permissions(self, platform):
        workflow = read_workflow(f"build_rocm_npu_staging_{platform}.yml")
        workflow_header = workflow.split("\njobs:", 1)[0]
        assert "\npermissions:" not in workflow_header, (
            "reusable workflow permissions would cap the caller's "
            "OIDC and attestation grants"
        )

    def test_pr_compiler_ci_builds_once_and_tests_every_python_minor(self):
        ci = read_workflow("ci.yml")
        for platform in ("linux", "windows"):
            workflow_name = f"build_rocm_npu_staging_{platform}.yml"
            assert ci.count(f"uses: ./.github/workflows/{workflow_name}") == 1
            build_job = _job(ci, platform)
            assert "run_npu_e2e: true" in build_job
            workflow = read_workflow(workflow_name)
            assert workflow.count("--binding build/iron-tools-install/python") == 1
            assert "build-compiler-bindings" not in workflow
            for minor in ("3.12", "3.13", "3.14"):
                compact = minor.replace(".", "")
                assert f"compiler-wheel-verify-{compact[-3:]}" in workflow
            assert "dev.py package-compiler-wheels" in workflow
            assert "--product build/rocm-npu-compiler-product" in workflow
            assert "dev.py verify-compiler-wheels" in workflow
            assert "dev.py package-ops-wheels" in workflow
            assert "dev.py verify-ops-wheels" in workflow
            assert "Package and verify ops wheel" in workflow
            assert "--reuse-venv" in workflow
            assert "ops-wheel-verify-313" not in workflow
            assert "ops-wheel-verify-314" not in workflow
            assert "--skip-compile" not in workflow
            assert "actions/attest-build-provenance@" in workflow
            assert "ROCM_NPU_PACKAGE_VERSION: ${{ inputs.package_version }}" in workflow
            assert workflow.count("--consumer-only") == 2
            assert workflow.count("--consumer-root build/compiler-wheel-consumer") == 3
            assert workflow.count("--consumer-e2e") == 2
            assert "UV_LINK_MODE: hardlink" in workflow
            assert "build/caches/uv" in workflow
            assert '"setuptools>=64" wheel' not in workflow

        linux_workflow = read_workflow("build_rocm_npu_staging_linux.yml")
        assert "actions/setup-python@" not in linux_workflow
        assert "--require-hashes -r requirements/packaging-linux.lock" in linux_workflow
        assert "build/artifacts/rocm_npu_ops*.whl" in linux_workflow
        assert "subject-path:" in linux_workflow
        for tag in ("cp313-cp313", "cp314-cp314"):
            assert f"/opt/python/{tag}/bin/python" in linux_workflow

        windows_workflow = read_workflow("build_rocm_npu_staging_windows.yml")
        for cache in ("Python package caches", "vcpkg binary packages", "ccache"):
            assert f"Restore {cache}" in windows_workflow
            assert f"Save {cache}" in windows_workflow
        assert "build/caches/pip" in windows_workflow
        assert "vcpkg-downloads" not in windows_workflow
        assert (
            "subject-path: build/artifacts/rocm_npu_compiler_*.whl" in windows_workflow
        )
        assert "build/artifacts/rocm_npu_ops*.whl" not in windows_workflow
        assert "{core,bindings}" not in windows_workflow

    def _assert_platform_workflow_contract(self, platform: str):
        workflow = read_workflow(f"fastflowlm_hrx_{platform}.yml")
        workflow_call = re.search(
            r"workflow_call:.*?workflow_dispatch:", workflow, re.DOTALL
        )
        assert workflow_call is not None
        assert "ref:" in workflow_call.group(0)
        assert "package_version:" in workflow_call.group(0)
        assert "attest_wheels:" in workflow_call.group(0)
        assert workflow.count("ref: ${{ inputs.ref || github.sha }}") == 2
        assert "\npermissions:\n" not in workflow
        build_header = workflow.split("\n  build:", 1)[1].split("\n    steps:", 1)[0]
        assert "permissions:" not in build_header
        assert "ROCM_NPU_PACKAGE_VERSION: ${{ inputs.package_version }}" in workflow
        metadata_check = workflow.index("-m twine check --strict")
        transitive_command = (
            "build_tools/rocm_npu_windows_wheel_cli.py verify"
            if platform == "windows"
            else "dev.py verify-fastflowlm-wheels"
        )
        transitive_install = workflow.index(transitive_command)
        attestation = workflow.index("actions/attest-build-provenance@")
        archive_suffix = ".zip" if platform == "windows" else ".tar.zst"
        assert (
            f"build/artifacts/rocm-npu-{platform}-x86_64{archive_suffix}"
            in workflow
        )
        artifact_upload = workflow.index("name: Upload runtime")
        npu_job = workflow.index("\n  npu:")
        assert metadata_check < transitive_install
        assert transitive_install < attestation
        assert attestation < artifact_upload
        assert transitive_install < npu_job
