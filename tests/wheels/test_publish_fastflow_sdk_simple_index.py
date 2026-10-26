# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest

from build_tools.publish_fastflow_sdk_simple_index import (
    collect_wheels,
    main,
    wheel_distribution,
    write_simple_index,
)

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]

BASE = "https://github.com/ROCm/FastFlow-SDK/releases/download"
NEW = "fastflow-sdk-v0.1.0.dev202610230003"
OLD = "fastflow-sdk-v0.1.0.dev202610050001"
NEW_RUNTIME = "fastflow_sdk_runtime-0.1.0.dev202610230003-py3-none-win_amd64.whl"
OLD_RUNTIME = "fastflow_sdk_runtime-0.1.0.dev202610050001-py3-none-win_amd64.whl"
META = "fastflow_sdk-0.1.0.dev202610230003-py3-none-any.whl"


def _asset(tag: str, name: str, digest: str | None = None) -> dict[str, object]:
    asset: dict[str, object] = {
        "name": name,
        "browser_download_url": f"{BASE}/{tag}/{name}",
    }
    if digest is not None:
        asset["digest"] = f"sha256:{digest}"
    return asset


def test_wheel_distribution_normalizes_the_project_name() -> None:
    assert wheel_distribution(NEW_RUNTIME) == "fastflow-sdk-runtime"
    assert wheel_distribution(META) == "fastflow-sdk"


def test_index_lists_every_published_release_and_skips_drafts(tmp_path: Path) -> None:
    releases = [
        {
            "tag_name": NEW,
            "draft": False,
            "assets": [
                _asset(NEW, NEW_RUNTIME, "a" * 64),
                _asset(NEW, META, "b" * 64),
                _asset(NEW, "SHA256SUMS"),
                _asset(NEW, "index.html"),
            ],
        },
        {
            "tag_name": OLD,
            "draft": False,
            "assets": [_asset(OLD, OLD_RUNTIME), _asset(OLD, "SHA256SUMS")],
        },
        {
            "tag_name": "rocm-npu-v0.1.0.dev202609130034",
            "draft": False,
            "assets": [
                _asset(
                    "rocm-npu-v0.1.0.dev202609130034",
                    "rocm_npu_runtime-0.1.0.dev202609130034-py3-none-win_amd64.whl",
                    "d" * 64,
                )
            ],
        },
        {
            "tag_name": "fastflow-sdk-v0.1.0.dev202610300001",
            "draft": True,
            "assets": [_asset("draft", "fastflow_sdk_runtime-9-py3-none-any.whl")],
        },
    ]
    sums = {f"{BASE}/{OLD}/SHA256SUMS": f"{'c' * 64}  {OLD_RUNTIME}\n"}

    output = write_simple_index(collect_wheels(releases, sums.__getitem__), tmp_path)

    assert (output / ".nojekyll").is_file()
    assert not (output / "index.html").exists()
    root = (output / "simple" / "index.html").read_text(encoding="utf-8")
    assert 'href="fastflow-sdk/"' in root
    assert 'href="fastflow-sdk-runtime/"' in root
    assert "rocm-npu" not in root
    assert not (output / "simple" / "rocm-npu-runtime").exists()
    runtime = (output / "simple" / "fastflow-sdk-runtime" / "index.html").read_text(
        encoding="utf-8"
    )
    assert f"{BASE}/{NEW}/{NEW_RUNTIME}#sha256={'a' * 64}" in runtime
    assert f"{BASE}/{OLD}/{OLD_RUNTIME}#sha256={'c' * 64}" in runtime
    assert "202610300001" not in runtime


def test_cli_reads_slurped_release_pages(tmp_path: Path) -> None:
    releases = tmp_path / "releases.json"
    releases.write_text(
        json.dumps(
            [
                [
                    {
                        "tag_name": NEW,
                        "draft": False,
                        "assets": [_asset(NEW, META, "b" * 64)],
                    }
                ],
                [
                    {
                        "tag_name": OLD,
                        "draft": False,
                        "assets": [_asset(OLD, OLD_RUNTIME, "c" * 64)],
                    }
                ],
            ]
        ),
        encoding="utf-8",
    )

    assert main(["--releases", str(releases), "--output", str(tmp_path / "pages")]) == 0

    simple = tmp_path / "pages" / "simple"
    assert (simple / "fastflow-sdk" / "index.html").is_file()
    assert (simple / "fastflow-sdk-runtime" / "index.html").is_file()


def test_missing_digest_fails(tmp_path: Path) -> None:
    releases = [{"tag_name": NEW, "draft": False, "assets": [_asset(NEW, NEW_RUNTIME)]}]
    with pytest.raises(RuntimeError, match="no sha256 digest"):
        collect_wheels(releases, lambda url: "")
