# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


class RocmNpuReleaseWheelWorkflowTest(unittest.TestCase):
    def test_ci_runs_presubmit_before_expensive_jobs(self):
        workflow = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

        self.assertIn("presubmit:", workflow)
        self.assertIn(
            "actions/setup-go@924ae3a1cded613372ab5595356fb5720e22ba16", workflow
        )
        self.assertIn(
            "go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12",
            workflow,
        )
        self.assertIn(
            "ruff check --select F821 dev.py build_tools tests",
            workflow,
        )
        self.assertEqual(workflow.count("needs: presubmit"), 4)

    def test_release_is_dispatch_only_and_reuses_both_npu_validations(self):
        workflow = (WORKFLOWS / "release_rocm_npu_wheels.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("push:", workflow)
        self.assertNotIn("schedule:", workflow)
        self.assertIn("uses: ./.github/workflows/fastflowlm_hrx_linux.yml", workflow)
        self.assertIn("uses: ./.github/workflows/fastflowlm_hrx_windows.yml", workflow)
        self.assertEqual(
            workflow.count("uses: ./.github/workflows/fastflowlm_hrx_linux.yml"),
            2,
        )
        self.assertEqual(
            workflow.count("uses: ./.github/workflows/fastflowlm_hrx_windows.yml"),
            2,
        )
        self.assertEqual(
            workflow.count(
                "uses: ./.github/workflows/build_rocm_npu_staging_linux.yml"
            ),
            2,
        )
        self.assertEqual(
            workflow.count(
                "uses: ./.github/workflows/build_rocm_npu_staging_windows.yml"
            ),
            2,
        )
        self.assertIn(
            "package_version: ${{ needs.prepare.outputs.package_version }}",
            workflow,
        )
        self.assertEqual(
            workflow.count("ref: ${{ needs.prepare.outputs.source_sha }}"), 9
        )
        self.assertIn("source_sha: ${{ steps.source.outputs.sha }}", workflow)
        self.assertIn("git merge-base --is-ancestor", workflow)
        self.assertIn('"${GITHUB_REF}" != refs/heads/main', workflow)
        self.assertIn(
            "if: ${{ inputs.publish_target == 'github-prerelease' }}",
            workflow,
        )
        self.assertIn("fetch-depth: 0", workflow)
        self.assertNotIn("--release-platform", workflow)
        self.assertIn('--publish-target "${INPUT_PUBLISH_TARGET}"', workflow)
        self.assertIn("rocm-npu-runtime-fastflowlm-linux-x86_64", workflow)
        self.assertIn("rocm-npu-runtime-fastflowlm-windows-x86_64", workflow)
        self.assertIn("rocm-npu-compiler-wheels-linux-x86_64", workflow)
        self.assertIn("rocm-npu-compiler-wheels-windows-x86_64", workflow)
        self.assertIn("python -m twine check --strict wheelhouse/*/*.whl", workflow)
        self.assertIn("python build_tools/rocm_npu_wheel_release_cli.py", workflow)
        self.assertIn("--expected-wheel-tag py3-none-manylinux_2_28_x86_64", workflow)
        self.assertIn("--expected-wheel-tag py3-none-linux_x86_64", workflow)
        self.assertIn("--expected-wheel-tag py3-none-win_amd64", workflow)
        self.assertIn("--expected-wheel-tag py3-none-any", workflow)
        self.assertIn(
            "--expected-wheel-tag cp312-abi3-manylinux_2_28_x86_64",
            workflow,
        )
        self.assertIn("--expected-wheel-tag cp312-abi3-win_amd64", workflow)
        self.assertIn("--abi abi3", workflow)
        self.assertNotIn("-cp313-", workflow)
        self.assertNotIn("-cp314-", workflow)
        self.assertIn("--platform linux_x86_64", workflow)
        self.assertIn("--platform manylinux_2_28_x86_64", workflow)
        self.assertIn('"rocm-npu-fastflowlm==${ROCM_NPU_PACKAGE_VERSION}"', workflow)
        self.assertIn('"rocm-npu-runtime": version', workflow)
        self.assertIn('"rocm-npu-compiler-bindings": version', workflow)
        compiler_resolution = workflow.split(
            "- name: Resolve compiler core and bindings", 1
        )[1].split("- name: Upload release payload", 1)[0]
        self.assertNotIn("--no-index", compiler_resolution)
        self.assertIn('--abi "${python_tag}"', compiler_resolution)
        self.assertIn('"${compiler_meta}"', compiler_resolution)
        self.assertIn(
            "COMPILER_BINDINGS_RUNTIME_REQUIREMENTS", compiler_resolution
        )
        self.assertIn(
            "ncipollo/release-action@339a81892b84b4eeb0f6e744e4574d79d0d9b8dd",
            workflow,
        )
        self.assertIn("commit: ${{ needs.prepare.outputs.source_sha }}", workflow)
        self.assertNotIn("needs.collect.outputs.source_sha", workflow)
        self.assertEqual(workflow.count("attest_wheels: false"), 4)
        self.assertEqual(workflow.count("attest_wheels: true"), 4)
        self.assertEqual(workflow.count("id-token: write"), 4)
        self.assertEqual(workflow.count("attestations: write"), 4)
        self.assertIn("always() &&", workflow)
        self.assertNotIn("actions/attest-build-provenance@", workflow)

        linux_validation = workflow.split("\n  linux_wheels:", 1)[1].split(
            "\n  windows_wheels:", 1
        )[0]
        windows_validation = workflow.split("\n  windows_wheels:", 1)[1].split(
            "\n  linux_publish_wheels:", 1
        )[0]
        for validation in (linux_validation, windows_validation):
            self.assertIn("if: ${{ inputs.publish_target == 'none' }}", validation)
            self.assertNotIn("id-token: write", validation)
            self.assertNotIn("attestations: write", validation)
        publish = workflow.split("\n  publish_github_prerelease:", 1)[1]
        self.assertIn("always() &&", publish)
        self.assertIn("needs.prepare.result == 'success'", publish)
        self.assertIn("needs.collect.result == 'success'", publish)

    def test_release_uses_neutral_public_names_and_shared_release_lock(self):
        workflow = (WORKFLOWS / "release_rocm_npu_wheels.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "rocm-npu-wheels-${{ needs.prepare.outputs.package_version }}",
            workflow,
        )
        self.assertIn("tag: ${{ needs.prepare.outputs.tag_name }}", workflow)
        self.assertIn("name: ${{ needs.prepare.outputs.release_name }}", workflow)
        self.assertIn("environment: rocm-npu-prerelease", workflow)
        self.assertIn("requirements-release-wheels.lock", workflow)
        self.assertNotIn("0.1.0", workflow)
        self.assertNotIn("rocm-npu-staging", workflow)
        self.assertFalse((WORKFLOWS / "release_rocm_npu_wheels_windows.yml").exists())

    def test_platform_wheel_ci_accepts_release_ref_and_version_inputs(self):
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform):
                self._assert_platform_workflow_contract(platform)

    def test_compiler_workflows_inherit_caller_permissions(self):
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform):
                workflow = (
                    WORKFLOWS / f"build_rocm_npu_staging_{platform}.yml"
                ).read_text(encoding="utf-8")
                workflow_header = workflow.split("\njobs:", 1)[0]
                self.assertNotIn(
                    "\npermissions:",
                    workflow_header,
                    "reusable workflow permissions would cap the caller's "
                    "OIDC and attestation grants",
                )

    def test_pr_compiler_ci_builds_once_and_tests_every_python_minor(self):
        ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        for platform in ("linux", "windows"):
            workflow_name = f"build_rocm_npu_staging_{platform}.yml"
            self.assertIn(f"uses: ./.github/workflows/{workflow_name}", ci)
            workflow = (WORKFLOWS / workflow_name).read_text(encoding="utf-8")
            self.assertEqual(
                workflow.count("--binding build/iron-tools-install/python"),
                1,
            )
            self.assertNotIn("build-compiler-bindings", workflow)
            for minor in ("3.12", "3.13", "3.14"):
                compact = minor.replace(".", "")
                self.assertIn(
                    f"compiler-wheel-verify-{compact[-3:]}",
                    workflow,
                )
            self.assertIn("dev.py package-compiler-wheels", workflow)
            self.assertIn(
                "--product build/rocm-npu-compiler-product", workflow
            )
            self.assertIn("dev.py verify-compiler-wheels", workflow)
            self.assertIn("actions/attest-build-provenance@", workflow)
            self.assertIn(
                "ROCM_NPU_PACKAGE_VERSION: ${{ inputs.package_version }}",
                workflow,
            )
            self.assertEqual(workflow.count("--consumer-only"), 2)
            self.assertEqual(
                workflow.count("--consumer-root build/compiler-wheel-consumer"),
                3,
            )
            self.assertEqual(workflow.count("--consumer-e2e"), 2)
            self.assertIn("UV_LINK_MODE: hardlink", workflow)
            self.assertIn("build/caches/uv", workflow)
            self.assertNotIn('"setuptools>=64" wheel', workflow)

        linux_workflow = (
            WORKFLOWS / "build_rocm_npu_staging_linux.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("actions/setup-python@", linux_workflow)
        self.assertIn(
            "--require-hashes -r requirements-packaging-linux.lock",
            linux_workflow,
        )
        for tag in ("cp313-cp313", "cp314-cp314"):
            self.assertIn(f"/opt/python/{tag}/bin/python", linux_workflow)

        windows_workflow = (
            WORKFLOWS / "build_rocm_npu_staging_windows.yml"
        ).read_text(encoding="utf-8")
        for cache in ("Python package caches", "vcpkg binary packages", "ccache"):
            self.assertIn(f"Restore {cache}", windows_workflow)
            self.assertIn(f"Save {cache}", windows_workflow)
        self.assertIn("build/caches/pip", windows_workflow)
        self.assertNotIn("vcpkg-downloads", windows_workflow)

    def _assert_platform_workflow_contract(self, platform: str):
        workflow = (WORKFLOWS / f"fastflowlm_hrx_{platform}.yml").read_text(
            encoding="utf-8"
        )
        workflow_call = re.search(
            r"workflow_call:.*?workflow_dispatch:", workflow, re.DOTALL
        )
        self.assertIsNotNone(workflow_call)
        self.assertIn("ref:", workflow_call.group(0))
        self.assertIn("package_version:", workflow_call.group(0))
        self.assertIn("attest_wheels:", workflow_call.group(0))
        self.assertEqual(workflow.count("ref: ${{ inputs.ref || github.sha }}"), 2)
        self.assertNotIn("\npermissions:\n", workflow)
        build_header = workflow.split("\n  build:", 1)[1].split("\n    steps:", 1)[0]
        self.assertNotIn("permissions:", build_header)
        self.assertIn(
            "ROCM_NPU_PACKAGE_VERSION: ${{ inputs.package_version }}",
            workflow,
        )
        metadata_check = workflow.index("-m twine check --strict")
        transitive_command = (
            "build_tools/rocm_npu_windows_wheel_cli.py verify"
            if platform == "windows"
            else "dev.py verify-fastflowlm-wheels"
        )
        transitive_install = workflow.index(transitive_command)
        attestation = workflow.index("actions/attest-build-provenance@")
        artifact_upload = workflow.index("name: Upload runtime")
        npu_job = workflow.index("\n  npu:")
        self.assertLess(metadata_check, transitive_install)
        self.assertLess(transitive_install, attestation)
        self.assertLess(attestation, artifact_upload)
        self.assertLess(transitive_install, npu_job)


if __name__ == "__main__":
    unittest.main()
