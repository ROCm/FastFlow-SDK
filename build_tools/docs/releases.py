# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The release notes and hardware pages, from the published GitHub releases.

Input is the releases API JSON the Pages workflow already fetches for the
package index (``gh api --paginate --slurp``). The hardware table is the
documented release's own "Qualification evidence" section, so it lists exactly
the systems and drivers that release was validated on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from build_tools.publish_fastflow_sdk_simple_index import RELEASE_TAG_PREFIX

INDEX_URL = "https://rocm.github.io/FastFlow-SDK/simple"
RELEASES_URL = "https://github.com/ROCm/FastFlow-SDK/releases"
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def load_releases(path: Path, tag: str | None = None) -> list[dict]:
    """Published FastFlow SDK releases, newest first, from ``tag`` back if given.

    Raises:
        ValueError: ``tag`` is not a published release.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    flat = (
        [r for page in data for r in page]
        if data and isinstance(data[0], list)
        else data
    )
    releases = sorted(
        (
            r
            for r in flat
            if not r.get("draft")
            and str(r.get("tag_name", "")).startswith(RELEASE_TAG_PREFIX)
        ),
        key=lambda r: r.get("published_at") or "",
        reverse=True,
    )
    if tag is None:
        return releases
    tags = [r["tag_name"] for r in releases]
    if tag not in tags:
        raise ValueError(f"{tag} is not a published FastFlow SDK release")
    return releases[tags.index(tag) :]


def latest_tag(path: Path) -> str:
    """The tag of the newest stable release, else of the newest prerelease.

    Stable releases are the ones on PyPI, which the documented plain
    ``pip install fastflow-sdk`` installs; a newer development build would not
    match it.

    Raises:
        ValueError: Nothing is published.
    """
    releases = load_releases(path)
    if not releases:
        raise ValueError("no FastFlow SDK release is published")
    stable = [r for r in releases if not r.get("prerelease")]
    return str((stable or releases)[0]["tag_name"])


def version(release: dict) -> str:
    return str(release["tag_name"]).removeprefix(RELEASE_TAG_PREFIX)


def _lines(body: str) -> list[str]:
    return body.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def section(body: str, title: str) -> str | None:
    """The text under the level-2 heading ``title``, without the heading."""
    lines = _lines(body)
    out: list[str] | None = None
    fenced = False
    for line in lines:
        heading = None if fenced else _HEADING.match(line)
        if _FENCE.match(line):
            fenced = not fenced
        if heading and len(heading.group(1)) <= 2:
            if out is not None:
                break
            if len(heading.group(1)) == 2 and heading.group(2).lower() == title.lower():
                out = []
            continue
        if out is not None:
            out.append(line)
    return "\n".join(out).strip() if out is not None else None


def demote(markdown: str, levels: int) -> str:
    """Push every heading outside code fences down by ``levels``."""
    out = []
    fenced = False
    for line in _lines(markdown):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and _HEADING.match(line):
            line = "#" * levels + line
        out.append(line)
    return "\n".join(out)


def _date(release: dict) -> str:
    return str(release.get("published_at") or "")[:10]


_UNAVAILABLE = (
    "\n!!! note\n    This build of the documentation had no release data. "
    f"See [GitHub Releases]({RELEASES_URL}).\n"
)


def releases_page(releases: list[dict] | None) -> str:
    header = f"""# Release notes

Every FastFlow SDK release is on [GitHub Releases]({RELEASES_URL}) with its
wheels, standalone archives, checksums, and qualification evidence. Install a
specific release by pinning
its version:

```bash
python -m pip install --only-binary=:all: --extra-index-url {INDEX_URL} "fastflow-sdk==<version>"
```
"""
    if releases is None:
        return header + _UNAVAILABLE
    sections = []
    for release in releases:
        name = version(release)
        body = str(release.get("body") or "")
        notes = section(body, "What's new")
        if notes is None:
            notes = body.split("\n## ", 1)[0].strip()
        sections.append(
            f"## {name}\n\n"
            f"Published {_date(release)}. "
            f"[Release page and assets]({release.get('html_url', RELEASES_URL)}).\n\n"
            f"{demote(notes, 1)}\n"
        )
    return header + "\n" + "\n".join(sections)


def hardware_page(releases: list[dict] | None) -> str:
    header = """# Hardware and drivers

FastFlow SDK targets x86_64 Linux and Windows machines with an AMD Ryzen AI
NPU that implements XDNA2 (NPU2): Strix Point, Strix Halo, Krackan Point, and
Gorgon Point. Windows means native PowerShell; WSL is not supported. NPU1
(Phoenix, Hawk Point) devices are reported by `fastflow probe` and
`fastflow doctor`, but releases are not qualified on them.

Check a machine with:

```bash
fastflow probe --json   # OS, driver, firmware, devices, power mode
fastflow self-test      # a real dispatch on the NPU
```

After installing or changing a driver, run `fastflow self-test`. If it fails,
reboot and retry before debugging further.
"""
    if not releases:
        return header + _UNAVAILABLE
    release = releases[0]
    url = release.get("html_url", RELEASES_URL)
    evidence = section(str(release.get("body") or ""), "Qualification evidence")
    if not evidence:
        return header + (
            f"\n!!! warning\n    Release {version(release)} published no "
            f"qualification summary. See [its release page]({url}).\n"
        )
    return (
        header
        + f"\n## Qualified systems for {version(release)}\n\n"
        + f"This is the qualification summary published with [the release]({url}). "
        "Other driver versions may work but were not validated.\n\n"
        + demote(evidence, 1)
        + "\n"
    )
