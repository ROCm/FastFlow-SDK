# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import email.parser
import json
import subprocess
import zipfile

import pytest

from build_tools import fastflow_sdk_meta_wheel, paths
from build_tools.artifact import verify_sha256_sidecar
from build_tools.fastflow_sdk_wheel_common import (
    COMPILER_DISTRIBUTION,
    FASTFLOWLM_DISTRIBUTION,
    OPS_DISTRIBUTION,
    SDK_DISTRIBUTION,
    SDK_PACKAGE,
)
from build_tools.release_policy import compiler_requires_python

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


def test_sdk_wheel_pins_every_product_at_its_version(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "BUILD", tmp_path / "build")
    version = "0.1.0.dev1"

    wheel = fastflow_sdk_meta_wheel.build_sdk_wheel(
        output_directory=tmp_path / "dist", version=version
    )

    assert wheel.name == f"fastflow_sdk-{version}-py3-none-any.whl"
    verify_sha256_sidecar(wheel, kind="wheel")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = email.parser.Parser().parsestr(
            archive.read(f"fastflow_sdk-{version}.dist-info/METADATA").decode()
        )
        tags = email.parser.Parser().parsestr(
            archive.read(f"fastflow_sdk-{version}.dist-info/WHEEL").decode()
        )
        manifest = json.loads(archive.read(f"{SDK_PACKAGE}/manifest.json"))
    assert metadata["Name"] == SDK_DISTRIBUTION
    assert metadata["Version"] == version
    assert metadata["Requires-Python"] == compiler_requires_python()
    assert metadata["License-Expression"] == "MIT"
    assert sorted(metadata.get_all("Requires-Dist")) == sorted(
        f"{distribution} =={version}"
        for distribution in (
            FASTFLOWLM_DISTRIBUTION,
            COMPILER_DISTRIBUTION,
            OPS_DISTRIBUTION,
        )
    )
    assert tags["Tag"] == "py3-none-any"
    assert tags["Root-Is-Purelib"] == "true"
    assert f"{SDK_PACKAGE}/__init__.py" in names
    assert not [
        name
        for name in names
        if not name.startswith(
            (f"{SDK_PACKAGE}/", f"fastflow_sdk-{version}.dist-info/")
        )
    ]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=paths.ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert manifest["product"] == SDK_DISTRIBUTION
    assert manifest["sources"]["fastflow-sdk"]["revision"] == head


def test_rebuild_replaces_the_previous_sdk_wheel(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "BUILD", tmp_path / "build")
    output = tmp_path / "dist"
    fastflow_sdk_meta_wheel.build_sdk_wheel(
        output_directory=output, version="0.1.0.dev1"
    )
    fastflow_sdk_meta_wheel.build_sdk_wheel(
        output_directory=output, version="0.1.0.dev2"
    )

    assert sorted(path.name for path in output.iterdir()) == [
        "fastflow_sdk-0.1.0.dev2-py3-none-any.whl",
        "fastflow_sdk-0.1.0.dev2-py3-none-any.whl.sha256",
    ]
