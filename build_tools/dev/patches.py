# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Source patch and offline-fetch helpers for the developer command router."""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

from build_tools import paths
from build_tools.artifact import git_source_state, sha256_file
from build_tools.dev.profiles import PIN_SECTION_TO_SUBMODULE
from build_tools.environment import sh, venv_py


def manifest():
    with open(paths.ROOT / "manifests" / "versions.toml", "rb") as stream:
        return tomllib.load(stream)


def eudsl_sdist():
    """(cache_path, pin) for the pinned eudsl sdist. cache_path may not exist yet."""
    pin = manifest().get("eudsl", {})
    if not pin:
        return None, {}
    fname = f"{pin['package'].replace('-', '_')}-{pin['version']}.tar.gz"
    return paths.FETCH_CACHE / fname, pin


def eudsl_cache_ok():
    """True iff the pinned eudsl sdist is in the cache and its sha256 matches."""
    path, pin = eudsl_sdist()
    return bool(pin) and path.exists() and sha256_file(path) == pin.get("sha256")


def _patch_state(patch, sub):
    """Return (state, apply options) for a patch against submodule dir `sub`.

    Patches are normalized to LF. Some upstream files are tracked as CRLF, so if
    strict matching fails, retry context matching without treating line-ending
    whitespace as semantic. Non-whitespace source drift still fails.
    """
    for options in ((), ("--ignore-space-change",)):
        rev = subprocess.run(
            ["git", "-C", str(sub), "apply", "--reverse", "--check", *options, str(patch)],
            capture_output=True,
        )
        if rev.returncode == 0:
            return "applied", options
        fwd = subprocess.run(
            ["git", "-C", str(sub), "apply", "--check", *options, str(patch)],
            capture_output=True,
        )
        if fwd.returncode == 0:
            return "appliable", options
    return "broken", ()


def apply_patches(submodule_names=None):
    """Apply the temporary submodule patches (idempotent). These live only in
    rocm-npu-staging and are upstreamable as-is; see patches/README.md."""
    if not paths.PATCHES.exists():
        return
    selected = set(submodule_names) if submodule_names is not None else None
    for sub_dir in sorted(p for p in paths.PATCHES.iterdir() if p.is_dir()):
        if selected is not None and sub_dir.name not in selected:
            continue
        sub = paths.PROJ / sub_dir.name
        if not sub.exists():
            sys.exit(f"submodule missing for patches/{sub_dir.name} (run: git submodule update --init projects/{sub_dir.name})")
        patches = sorted(sub_dir.glob("*.patch"))
        states = [_patch_state(patch, sub) for patch in patches]
        if any(state == "broken" for state, _ in states):
            try:
                exact = git_source_state(sub) == declared_patch_expected_state(
                    sub_dir.name, sub)
            except RuntimeError:
                exact = False
            if exact:
                for patch in patches:
                    print(f"patch: {sub_dir.name}/{patch.name} already applied")
                continue
        for patch in patches:
            st, options = _patch_state(patch, sub)
            if st == "applied":
                print(f"patch: {sub_dir.name}/{patch.name} already applied")
            elif st == "appliable":
                sh(["git", "-C", str(sub), "apply", *options, str(patch)])
                print(f"patch: applied {sub_dir.name}/{patch.name}")
            else:
                sys.exit(f"patch does not apply cleanly: {patch} (submodule moved? regenerate it)")


def declared_patch_expected_state(component, source):
    """Builds the expected source state without materializing another checkout."""
    temporary_root = Path(
        os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
    ).resolve()
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix="rns-patch-", dir=temporary_root))
    index = temporary_parent / "index"
    env = dict(os.environ, GIT_INDEX_FILE=str(index))
    try:
        subprocess.run(
            ["git", "-C", str(source), "read-tree", "HEAD"],
            check=True,
            env=env,
            capture_output=True,
        )
        for patch in sorted((paths.PATCHES / component).glob("*.patch")):
            resolved = False
            for options in ((), ("--ignore-space-change",)):
                reverse = subprocess.run(
                    ["git", "-C", str(source), "apply", "--cached", "--reverse",
                     "--check", *options, str(patch)],
                    env=env,
                    capture_output=True,
                )
                if reverse.returncode == 0:
                    resolved = True
                    break
                result = subprocess.run(
                    ["git", "-C", str(source), "apply", "--cached", "--check",
                     *options, str(patch)],
                    env=env,
                    capture_output=True,
                )
                if result.returncode:
                    continue
                subprocess.run(
                    ["git", "-C", str(source), "apply", "--cached",
                     *options, str(patch)],
                    check=True,
                    env=env,
                    capture_output=True,
                )
                resolved = True
                break
            if not resolved:
                raise RuntimeError(f"declared patch does not apply: {patch}")

        changed_output = subprocess.run(
            ["git", "-C", str(source), "diff", "--cached", "--name-only",
             "--diff-filter=ACDMRTUXB", "-z", "HEAD"],
            check=True,
            env=env,
            capture_output=True,
        ).stdout
        changed_paths = sorted(
            item.decode("utf-8", errors="surrogateescape")
            for item in changed_output.split(b"\0") if item
        )
        tracked_paths = []
        changed_files = []
        for relative in changed_paths:
            existed = subprocess.run(
                ["git", "-C", str(source), "cat-file", "-e", f"HEAD:{relative}"],
                capture_output=True,
            ).returncode == 0
            if existed:
                tracked_paths.append(relative)
            else:
                ignored = subprocess.run(
                    ["git", "-C", str(source), "check-ignore", "--no-index",
                     "--quiet", "--", relative],
                    capture_output=True,
                ).returncode == 0
                if ignored:
                    continue

            staged = subprocess.run(
                ["git", "-C", str(source), "ls-files", "--stage", "-z",
                 "--", relative],
                check=True,
                env=env,
                capture_output=True,
            ).stdout
            record = {"path": relative.replace("\\", "/")}
            if not staged:
                record["deleted"] = True
            else:
                header = staged.split(b"\t", 1)[0].decode("ascii")
                blob = header.split()[1]
                data = subprocess.run(
                    ["git", "-C", str(source), "cat-file", "blob", blob],
                    check=True,
                    capture_output=True,
                ).stdout
                record.update({
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                })
            changed_files.append(record)

        diff_command = [
            "git", "-C", str(source), "diff", "--cached", "--binary",
            "--no-ext-diff", "HEAD",
        ]
        if tracked_paths:
            diff_command.extend(["--", *tracked_paths])
            diff = subprocess.run(
                diff_command, check=True, env=env, capture_output=True
            ).stdout
        else:
            diff = b""
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "revision": revision,
            "dirty": bool(changed_files),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "changed_files": changed_files,
        }
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def validate_declared_patch_state(component, source):
    """Proves the source tree equals its pin plus the ordered patch sequence."""
    expected = declared_patch_expected_state(component, source)
    actual = git_source_state(source)
    if actual != expected:
        expected_files = {item["path"]: item for item in expected["changed_files"]}
        actual_files = {item["path"]: item for item in actual["changed_files"]}
        differing = sorted(
            path for path in set(expected_files) | set(actual_files)
            if expected_files.get(path) != actual_files.get(path))
        raise RuntimeError(
            f"{component} source differs from its pin plus declared patches: "
            f"{', '.join(differing) or 'tracked diff metadata'}")


def cmd_fetch(args, submodule_names=None):
    """Prepare sources for an offline build (the single network step): apply temporary
    patches + prefetch the build-time-fetched eudsl sdist into a gitignored,
    hash-verified cache. Idempotent."""
    apply_patches(submodule_names)
    if submodule_names is not None and "iron-tools" not in submodule_names:
        print("fetch: selected source patches are ready (no compiler-stack prefetch needed)")
        return
    path, pin = eudsl_sdist()
    if not pin:
        sys.exit("manifests/versions.toml has no [eudsl] section")
    if eudsl_cache_ok():
        print(f"fetch: eudsl already cached + verified ({path.name})")
        return
    paths.FETCH_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"fetch: downloading {pin['package']}=={pin['version']}")
    sh([str(venv_py()), "-m", "pip", "download", f"{pin['package']}=={pin['version']}",
        "-f", pin["find_links"], "--no-deps", "--no-binary", ":all:", "-d", str(paths.FETCH_CACHE)])
    got = sha256_file(path) if path.exists() else "<missing>"
    if got != pin["sha256"]:
        path.unlink(missing_ok=True)
        sys.exit(f"fetch: sha256 mismatch for {path.name}\n  expected {pin['sha256']}\n  got      {got}")
    print(f"fetch: ok -> {path} (sha256 verified)")


def check_pins(submodules):
    """Verify selected submodule checkouts against manifests/versions.toml."""
    man = manifest()
    problems = []
    for sec, sub in PIN_SECTION_TO_SUBMODULE.items():
        if sub not in submodules:
            continue
        pinned = man.get(sec, {}).get("commit")
        if not pinned:
            continue
        sub_dir = paths.PROJ / sub
        if not (sub_dir / ".git").exists():   # not checked out — rev-parse here would walk up to the superproject
            print(f"  [ABSENT] {sub}: required by this doctor profile")
            problems.append(sub)
            continue
        r = subprocess.run(["git", "-C", str(sub_dir), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        head = r.stdout.strip()
        match = head == pinned
        print(f"  [{'ok  ' if match else 'DRIFT'}] {sub}: manifest {pinned[:12]} vs HEAD {(head[:12] or '?')}")
        if not match:
            problems.append(sub)
    return problems


def doctor_patch_states(component, source, patches, *, deep=False):
    """Checks a patch series, replaying it only when requested or ambiguous."""
    states = [_patch_state(patch, source)[0] for patch in patches]
    if not deep and "broken" not in states:
        return states
    try:
        exact = git_source_state(source) == declared_patch_expected_state(
            component, source)
        series_valid = True
    except RuntimeError:
        exact = False
        series_valid = False
    if exact:
        return ["applied"] * len(patches)
    if series_valid:
        return ["pending" if state == "broken" else state for state in states]
    return states
