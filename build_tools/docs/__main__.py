# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Generate the documentation pages, then build or serve the site.

python -m build_tools.docs generate   # write the generated pages into docs/
python -m build_tools.docs build      # generate, then zensical build --strict
python -m build_tools.docs serve      # generate, then zensical serve
python -m build_tools.docs latest-release --releases releases.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from build_tools.docs import pages, releases

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m build_tools.docs", description=__doc__.splitlines()[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "build", "serve"):
        commands.add_parser(name)
    latest = commands.add_parser(
        "latest-release",
        help="print the newest stable release's tag, else the newest prerelease's",
    )
    latest.add_argument("--releases", type=Path, required=True)
    options = parser.parse_args(argv)
    if options.command == "latest-release":
        print(releases.latest_tag(options.releases))
        return 0
    written = pages.write(pages.inputs_from_environment(ROOT, os.environ))
    print(f"docs: generated {len(written)} files under {pages.DOCS.as_posix()}/")
    if options.command == "generate":
        return 0
    zensical = [sys.executable, "-m", "zensical", options.command]
    if options.command == "build":
        zensical += ["--clean", "--strict"]
    return subprocess.run(zensical, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
