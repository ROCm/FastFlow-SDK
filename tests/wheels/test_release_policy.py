# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json

import pytest

from build_tools.release_policy import (
    compiler_python_stable_abi,
    compiler_python_versions,
    compiler_requires_python,
    numeric_runtime_contract,
    release_base_version,
)

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]


def _write_policy(tmp_path, wheel_release: str, *, schema_version: int = 1):
    path = tmp_path / "release.toml"
    path.write_text(
        f"schema_version = {schema_version}\n[wheel_release]\n{wheel_release}",
        encoding="utf-8",
    )
    return path


def test_release_policy_loads_validated_wheel_release(tmp_path):
    policy = _write_policy(
        tmp_path,
        'base_version = "1.2.3"\n'
        'compiler_python_stable_abi = "3.12"\n'
        'compiler_python_versions = ["3.12", "3.13", "3.14"]\n',
    )
    assert release_base_version(policy) == "1.2.3"
    assert compiler_python_versions(policy) == ("3.12", "3.13", "3.14")
    assert compiler_python_stable_abi(policy) == "3.12"
    assert compiler_requires_python(policy) == ">=3.12,<3.15"


@pytest.mark.parametrize(
    ("wheel_release", "message"),
    [
        ('base_version = "1.2.3rc1"\n', "stable numeric version"),
        (
            'compiler_python_versions = ["3.12", "3.12"]\n',
            "duplicate compiler Python minor",
        ),
        (
            (
                'compiler_python_stable_abi = "3.13"\n'
                'compiler_python_versions = ["3.12", "3.13"]\n'
            ),
            "stable ABI floor",
        ),
    ],
)
def test_release_policy_rejects_invalid_wheel_release(
    tmp_path, wheel_release, message
):
    policy = _write_policy(tmp_path, wheel_release)
    loader = (
        release_base_version
        if "base_version" in wheel_release
        else compiler_python_versions
        if "stable_abi" not in wheel_release
        else compiler_python_stable_abi
    )
    with pytest.raises(RuntimeError, match=message):
        loader(policy)


def test_release_policy_rejects_unknown_schema(tmp_path):
    policy = _write_policy(tmp_path, 'base_version = "1.2.3"\n', schema_version=2)
    with pytest.raises(RuntimeError, match="unsupported release policy schema"):
        release_base_version(policy)


def test_numeric_runtime_contract_validates_complete_schema(tmp_path):
    path = tmp_path / "numeric-runtime.json"
    document = {
        "schema_version": 1,
        "requires_python": ">=3.11,<3.13",
        "python_abis": ["cp311", "cp312"],
        "packages": {"numpy": "2.4.6"},
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    assert numeric_runtime_contract(path) == document

    document["python_abis"] = ["cp311", "py312"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid numeric runtime contract"):
        numeric_runtime_contract(path)
