# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import json
import subprocess
import sys

import pytest

from build_tools import fastflowlm_qualification_common as qualification


pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def test_build_helper_import_does_not_require_numpy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['numpy'] = None; "
                "import build_tools.fastflowlm_qualification_common"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_file_record_is_relative_and_content_bound(tmp_path):
    payload = tmp_path / "nested" / "payload.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"qualification")
    assert qualification.file_record(tmp_path, payload) == {
        "path": "nested/payload.bin",
        "size": 13,
        "sha256": hashlib.sha256(b"qualification").hexdigest(),
    }


def test_capability_loader_rejects_unsafe_family_paths(tmp_path):
    document = json.loads(
        qualification.CAPABILITIES_SOURCE.read_text(encoding="utf-8")
    )
    document["families"][0]["xclbin_directory"] = "../outside"
    path = tmp_path / "capabilities.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="family capability"):
        qualification.load_capabilities(path)


def test_shipping_recipe_generators_are_directly_loadable():
    generators = qualification.recipe_generators()
    assert generators.predicted_size("zero", {"size": 17}) == 17
