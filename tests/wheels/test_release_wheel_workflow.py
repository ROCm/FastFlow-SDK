# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


class RocmNpuReleaseWheelWorkflowTest(unittest.TestCase):
    def test_ci_lints_workflows_before_expensive_jobs(self):
        workflow = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

        self.assertIn(
            "actions/setup-go@924ae3a1cded613372ab5595356fb5720e22ba16", workflow
        )
        self.assertIn(
            "go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.12",
            workflow,
        )
        self.assertEqual(workflow.count("needs: workflow_lint"), 4)

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
        self.assertIn(
            "package_version: ${{ needs.prepare.outputs.package_version }}",
            workflow,
        )
        self.assertEqual(
            workflow.count("ref: ${{ needs.prepare.outputs.source_sha }}"), 3
        )
        self.assertIn("source_sha: ${{ steps.source.outputs.sha }}", workflow)
        self.assertNotIn("--release-platform", workflow)
        self.assertIn('--publish-target "${INPUT_PUBLISH_TARGET}"', workflow)
        self.assertIn("rocm-npu-runtime-fastflowlm-linux-x86_64", workflow)
        self.assertIn("rocm-npu-runtime-fastflowlm-windows-x86_64", workflow)
        self.assertIn("python -m twine check --strict wheelhouse/*/*.whl", workflow)
        self.assertIn("python build_tools/rocm_npu_wheel_release_cli.py", workflow)
        self.assertIn("--expected-wheel-tag py3-none-manylinux_2_28_x86_64", workflow)
        self.assertIn("--expected-wheel-tag py3-none-win_amd64", workflow)
        self.assertIn('"rocm-npu-fastflowlm==${ROCM_NPU_PACKAGE_VERSION}"', workflow)
        self.assertIn('"rocm-npu-runtime": version', workflow)
        self.assertIn(
            "ncipollo/release-action@339a81892b84b4eeb0f6e744e4574d79d0d9b8dd",
            workflow,
        )
        self.assertIn("commit: ${{ needs.prepare.outputs.source_sha }}", workflow)
        self.assertNotIn("needs.collect.outputs.source_sha", workflow)
        self.assertEqual(
            workflow.count(
                "attest_wheels: ${{ inputs.publish_target == 'github-prerelease' }}"
            ),
            2,
        )
        self.assertEqual(workflow.count("id-token: write"), 2)
        self.assertEqual(workflow.count("attestations: write"), 2)
        self.assertNotIn("actions/attest-build-provenance@", workflow)

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
        metadata_check = workflow.index("python -m twine check --strict")
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
