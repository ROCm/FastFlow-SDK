# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the PEP 503 simple index for FastFlow SDK GitHub Releases.

The index lists every wheel attached to a published (non-draft) release whose
tag starts with ``fastflow-sdk-v``, and links each one at its release asset
URL. It is rebuilt from the releases on every run, so the deployed site always
matches what GitHub serves.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

_WHEEL = re.compile(r"^(?P<name>[^-]+)-[^-]+(?:-[^-]+)?-[^-]+-[^-]+-[^-]+\.whl$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RELEASE_TAG_PREFIX = "fastflow-sdk-v"


def canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_distribution(filename: str) -> str:
    match = _WHEEL.match(filename)
    if match is None:
        raise RuntimeError(f"not a wheel filename: {filename}")
    return canonical_distribution(match.group("name"))


def _parse_sha256sums(text: str) -> dict[str, str]:
    digests = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        digests[name.strip().lstrip("*")] = digest.lower()
    return digests


def _release_checksums(
    assets: list[dict[str, object]], fetch_text: Callable[[str], str]
) -> dict[str, str]:
    for asset in assets:
        if asset.get("name") == "SHA256SUMS":
            return _parse_sha256sums(fetch_text(str(asset["browser_download_url"])))
    return {}


def collect_wheels(
    releases: list[dict[str, object]],
    fetch_text: Callable[[str], str],
) -> dict[str, dict[str, tuple[str, str]]]:
    """Map each distribution to ``filename -> (url, sha256)``."""
    projects: dict[str, dict[str, tuple[str, str]]] = {}
    for release in releases:
        tag = str(release.get("tag_name") or "")
        if release.get("draft") or not tag.startswith(RELEASE_TAG_PREFIX):
            continue
        assets = list(release.get("assets") or [])
        checksums: dict[str, str] | None = None
        for asset in assets:
            filename = str(asset.get("name") or "")
            if not filename.endswith(".whl"):
                continue
            files = projects.setdefault(wheel_distribution(filename), {})
            if filename in files:
                continue
            digest = str(asset.get("digest") or "")
            if digest.startswith("sha256:"):
                digest = digest.removeprefix("sha256:")
            else:
                if checksums is None:
                    checksums = _release_checksums(assets, fetch_text)
                digest = checksums.get(filename, "")
            if not _SHA256.match(digest):
                raise RuntimeError(f"no sha256 digest for release asset {filename}")
            files[filename] = (str(asset["browser_download_url"]), digest)
    return projects


def _render(links: list[tuple[str, str]]) -> str:
    body = "<br/>\n".join(
        f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'
        for href, label in links
    )
    return f"<!doctype html>\n<html><body>\n{body}\n</body></html>\n"


def write_simple_index(
    projects: dict[str, dict[str, tuple[str, str]]], output_directory: Path
) -> Path:
    """Write ``simple/`` and ``.nojekyll`` under ``output_directory``."""
    if not projects:
        raise RuntimeError("no wheels are attached to a published release")
    output_directory = Path(output_directory)
    simple = output_directory / "simple"
    simple.mkdir(parents=True, exist_ok=True)
    (output_directory / ".nojekyll").write_bytes(b"")
    (simple / "index.html").write_text(
        _render([(f"{name}/", name) for name in sorted(projects)]),
        encoding="utf-8",
    )
    for name, files in sorted(projects.items()):
        directory = simple / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.html").write_text(
            _render(
                [
                    (f"{url}#sha256={digest}", filename)
                    for filename, (url, digest) in sorted(files.items())
                ]
            ),
            encoding="utf-8",
        )
    return output_directory


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "fastflow-sdk-simple-index"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--releases",
        type=Path,
        required=True,
        help="JSON from `gh api --paginate --slurp repos/<owner>/<name>/releases`",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    pages = json.loads(args.releases.read_text(encoding="utf-8"))
    releases = [release for page in pages for release in page]
    write_simple_index(collect_wheels(releases, _fetch_text), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
