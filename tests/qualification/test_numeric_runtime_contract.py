# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from types import SimpleNamespace

import pytest

from build_tools import download_qualification_numeric_wheels as downloader
from build_tools import release_policy
from build_tools import rocm_npu_qualification_wheel as qualification_wheel
from build_tools import verify_qualification_dependencies as dependency_verifier
from tests.support.modules import load_module

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def _shipping_numeric_runtime():
    path = (
        release_policy.ROOT
        / "tools/rocm-npu-qualification/python/rocm_npu_qualification"
        / "numeric_runtime.py"
    )
    return load_module("shipping_numeric_runtime", path)


def test_numeric_runtime_contract_has_no_secondary_version_authority():
    contract = release_policy.numeric_runtime_contract()
    shipping = _shipping_numeric_runtime()
    assert shipping.NUMERIC_RUNTIME == contract["packages"]
    assert shipping.NUMERIC_PYTHON_ABIS == tuple(contract["python_abis"])
    assert shipping.NUMERIC_REQUIRES_PYTHON == contract["requires_python"]
    assert qualification_wheel.NUMERIC_RUNTIME == contract["packages"]
    assert (
        qualification_wheel.QUALIFICATION_REQUIRES_PYTHON
        == contract["requires_python"]
    )
    assert release_policy.numeric_package_versions() == contract["packages"]
    assert release_policy.numeric_python_abis() == tuple(contract["python_abis"])


def test_downloader_stages_every_attested_python_and_platform(monkeypatch, tmp_path):
    commands = []

    def capture(command, *, check):
        assert check is True
        commands.append(command)

    monkeypatch.setattr(downloader.subprocess, "run", capture)
    downloader.download_numeric_wheels(tmp_path)
    contract = release_policy.numeric_runtime_contract()
    assert len(commands) == 2 * len(contract["python_abis"])
    assert {
        (command[command.index("--abi") + 1], command[command.index("--platform") + 1])
        for command in commands
    } == {
        (abi, platform)
        for abi in contract["python_abis"]
        for platform in ("manylinux_2_28_x86_64", "win_amd64")
    }


def test_dependency_verifier_uses_numeric_packages_from_contract(
    monkeypatch, tmp_path
):
    contract = {
        "schema_version": 1,
        "requires_python": ">=3.11,<3.15",
        "python_abis": ["cp311", "cp312", "cp313", "cp314"],
        "packages": {"ml-dtypes": "1.2.3", "numpy": "4.5.6"},
    }
    contract_path = tmp_path / "numeric-runtime.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    expected = {
        "rocm-npu-qualification": "7.8.9",
        "rocm-npu-fastflowlm": "7.8.9",
        "rocm-npu-runtime": "7.8.9",
        **contract["packages"],
    }
    monkeypatch.setattr(
        dependency_verifier.metadata,
        "distributions",
        lambda *, path: [
            SimpleNamespace(
                metadata={"Name": distribution.replace("-", "_")},
                version=version,
            )
            for distribution, version in expected.items()
        ],
    )

    dependency_verifier.verify_installed_distributions(
        tmp_path / "install", "7.8.9", contract_path
    )
