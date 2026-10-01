# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Source checkout operations for developer commands."""

import sys

from build_tools import environment, paths

# Submodules with nested submodules of their own, and those a patch touches.
RECURSIVE_SUBMODULES = ("aiebu", "iron-tools", "FastFlowLM")


def cmd_deps(args):
    """Print the compiler-stack dependency manifest."""
    print((paths.ROOT / "requirements" / "compiler-stack.txt").read_text())


def cmd_submodules(args):
    """Initialize, reset, or clean the requested source submodules."""
    force = ["--force"] if args.reset else []
    shallow = ["--depth", "1", "--jobs", "4"] if args.shallow else []
    # sync before update: update fetches from the URL recorded in .git/config, not from
    # .gitmodules, so repointing a submodule's remote is silently ignored on every clone
    # that already initialized it -- it keeps building from the old one. sync is
    # idempotent and skips submodules that have no config entry yet (fresh checkouts).
    sync = ["git", "submodule", "sync"]
    update = ["git", "submodule", "update", "--init", *force]
    flat = [name for name in args.names if name not in RECURSIVE_SUBMODULES]
    if flat:
        targets = [f"projects/{name}" for name in flat]
        environment.sh([*sync, *targets], cwd=paths.ROOT)
        environment.sh([*update, *shallow, *targets], cwd=paths.ROOT)
    for name in args.names:
        if name in RECURSIVE_SUBMODULES:
            environment.sh([*sync, "--recursive", f"projects/{name}"], cwd=paths.ROOT)
            environment.sh(
                [*update, "--recursive", *shallow, f"projects/{name}"],
                cwd=paths.ROOT,
            )
    for name in args.clean:
        if name not in args.names:
            sys.exit(f"--clean {name}: not among the initialized submodules")
        environment.sh(["git", "-C", str(paths.PROJ / name), "clean", "-qfdx"])
    print(f"submodules: ok ({', '.join(args.names)})")
