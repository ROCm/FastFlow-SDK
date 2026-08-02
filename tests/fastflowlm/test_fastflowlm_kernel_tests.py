# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import copy
import json
import tempfile
from pathlib import Path

import pytest

from build_tools import fastflowlm_kernel_tests as kernels

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


class TestFastFlowLMKernels:
    def _report(self):
        group_ids = {
            group: f"{index + 1:016x}"
            for index, group in enumerate(
                sorted(set(kernels.EXPECTED_IMPLEMENTATION_GROUPS.values()))
            )
        }
        return {
            "status": "pass",
            "capability_contract_id": kernels.CAPABILITIES["contract_id"],
            "implementation_id_algorithm": "fnv1a64",
            "iterations": 20,
            "lifecycle_rounds": 5,
            "resident_rounds": 20,
            "preflight_dispatches": kernels.EXPECTED_PREFLIGHT_DISPATCHES,
            "dispatches": (
                kernels.EXPECTED_PREFLIGHT_DISPATCHES + len(kernels.EXPECTED_CASES) * 25
            ),
            "elapsed_seconds": 1.0,
            "device_region_seconds": 0.5,
            "process_wall_seconds": 2.0,
            "failures": 0,
            "cases": [
                {
                    "name": name,
                    "family": name.split("/")[0],
                    "kernel": name.split("/")[1],
                    "vector_id": name.split("/")[2],
                    "dimensions": dimensions,
                    "lifecycle_rounds": 5,
                    "resident_rounds": 20,
                    "validated_bytes": validated_bytes,
                    "mismatches": 0,
                    "maximum_absolute_error": 0.0,
                    "maximum_relative_error": 0.0,
                    "maximum_tolerance_ratio": 0.0,
                    "absolute_tolerance": absolute_tolerance,
                    "relative_tolerance": relative_tolerance,
                    "implementation_id": group_ids[
                        kernels.EXPECTED_IMPLEMENTATION_GROUPS[name]
                    ],
                    "weight_provenance": weight_provenance,
                    "reference_kind": reference_kind,
                    "comparison": comparison,
                    "device_seconds": 0.01,
                    "failures": 0,
                }
                for (
                    name,
                    dimensions,
                    validated_bytes,
                    weight_provenance,
                    reference_kind,
                    comparison,
                    absolute_tolerance,
                    relative_tolerance,
                ) in kernels.EXPECTED_CASES
            ],
        }

    def test_validate_report_requires_all_numeric_cases(self):
        assert kernels.validate_report(self._report())["status"] == "pass"
        report = self._report()
        report["cases"].pop()
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)
        report = self._report()
        report["cases"][0]["validated_bytes"] = 0
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)
        report = self._report()
        report["cases"][0]["mismatches"] = 1
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)
        report = self._report()
        report["cases"][0], report["cases"][1] = (
            report["cases"][1],
            report["cases"][0],
        )
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)
        report = self._report()
        report["cases"][0]["dimensions"] = [64, 512]
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)
        report = self._report()
        report["dispatches"] -= 1
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)
        report = self._report()
        report["cases"][0]["maximum_absolute_error"] = float("nan")
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)
        report = self._report()
        report["cases"][0]["weight_provenance"] = "unknown"
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)
        report = self._report()
        report["cases"][3]["maximum_relative_error"] = 1e9
        assert kernels.validate_report(report)["status"] == "pass"
        report = self._report()
        report["cases"][3]["maximum_tolerance_ratio"] = 1.01
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)
        report = self._report()
        report["cases"][3]["absolute_tolerance"] = 0.1
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)
        report = self._report()
        report["cases"][0]["implementation_id"] = report["cases"][2][
            "implementation_id"
        ]
        with pytest.raises(RuntimeError):
            kernels.validate_report(report)

    def test_new_implementation_is_derived_from_one_family_entry(self):
        capabilities = copy.deepcopy(kernels.CAPABILITIES)
        family = copy.deepcopy(capabilities["families"][0])
        family["name"] = "FutureModel-NPU2"
        family["xclbin_directory"] = "FutureModel-NPU2"
        family["implementation_groups"]["dequant"] = "future-dequant-v1"
        family["implementation_groups"]["mm"] = "future-mm-v1"
        capabilities["families"].append(family)
        cases, groups, preflight = kernels._expected_matrix(capabilities)
        assert len(cases) == len(kernels.EXPECTED_CASES) + 10
        assert preflight == kernels.EXPECTED_PREFLIGHT_DISPATCHES + 3
        assert (
            groups["FutureModel-NPU2/mm/isolated-structured"]
            == "future-mm-v1"
        )

    def _stage(self, root):
        executables = {}
        (root / "bin").mkdir()
        for role in kernels.TARGETS:
            executable = root / "bin" / kernels.executable_name(role)
            executable.write_bytes(f"portable {role} helper".encode())
            executables[role] = executable
        capabilities = root / kernels.CAPABILITIES_RELATIVE
        capabilities.parent.mkdir(parents=True)
        capabilities.write_bytes(kernels.CAPABILITIES_SOURCE.read_bytes())
        kernels._write_manifest(root)
        return executables, capabilities

    def test_manifest_verification_detects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            executables, capabilities = self._stage(root)
            assert kernels.verify_tree(root) == executables
            for role, executable in executables.items():
                original = executable.read_bytes()
                executable.write_bytes(b"tampered")
                with pytest.raises(RuntimeError, match=role):
                    kernels.verify_tree(root)
                executable.write_bytes(original)
            capabilities.write_bytes(b"{}")
            with pytest.raises(RuntimeError):
                kernels.verify_tree(root)

    def test_manifest_covers_every_helper(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._stage(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            assert set(manifest["executables"]) == set(kernels.TARGETS)
            manifest["executables"].pop("readback-smoke")
            manifest_path.write_text(json.dumps(manifest))
            with pytest.raises(RuntimeError):
                kernels.verify_tree(root)

    def test_manifest_has_stable_product_and_platform_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._stage(root)
            manifest = json.loads((root / "manifest.json").read_text())
            assert manifest["schema_version"] == 2
            assert manifest["product"] == kernels.PRODUCT
            assert manifest["source_revision"] == kernels.source_revision()
