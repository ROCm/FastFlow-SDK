# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Platform-neutral ROCm NPU wheel release payload preparation."""

from __future__ import annotations

import email.parser
import html
import json
import re
import shutil
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from build_tools.fastflowlm_artifact import reset_directory
from build_tools.rocm_npu_compiler_wheels import (
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
)
from build_tools.rocm_npu_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    COMPILER_STABLE_ABI_MINOR,
    FASTFLOWLM_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
    sha256_file,
    validate_version,
    verify_wheel_checksums,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE_POLICY_PATH = ROOT / "release.toml"
RELEASE_DISTRIBUTIONS = (
    RUNTIME_DISTRIBUTION,
    FASTFLOWLM_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_BINDINGS_DISTRIBUTION,
)


def release_base_version(policy_path: Path = RELEASE_POLICY_PATH) -> str:
    """Loads the stable version targeted by the current development train."""
    with Path(policy_path).open("rb") as policy_file:
        policy = tomllib.load(policy_file)
    if policy.get("schema_version") != 1:
        raise RuntimeError(f"unsupported release policy schema: {policy_path}")
    release = policy.get("wheel_release")
    base_version = release.get("base_version") if isinstance(release, dict) else None
    if not isinstance(base_version, str) or not re.fullmatch(
        r"[0-9]+(?:\.[0-9]+)*", base_version
    ):
        raise RuntimeError(
            "wheel_release.base_version must be a stable numeric version: "
            f"{policy_path}"
        )
    return validate_version(base_version)


def compiler_python_versions(
    policy_path: Path = RELEASE_POLICY_PATH,
) -> tuple[str, ...]:
    with Path(policy_path).open("rb") as policy_file:
        policy = tomllib.load(policy_file)
    release = policy.get("wheel_release")
    versions = release.get("compiler_python_versions") if isinstance(release, dict) else None
    if not isinstance(versions, list) or not versions:
        raise RuntimeError(
            "wheel_release.compiler_python_versions must be a non-empty list: "
            f"{policy_path}"
        )
    result = tuple(str(version) for version in versions)
    if (
        len(set(result)) != len(result)
        or any(not re.fullmatch(r"3\.[0-9]+", version) for version in result)
    ):
        raise RuntimeError(
            f"wheel_release.compiler_python_versions is invalid: {policy_path}"
        )
    return result


def compiler_python_stable_abi(
    policy_path: Path = RELEASE_POLICY_PATH,
) -> str:
    with Path(policy_path).open("rb") as policy_file:
        policy = tomllib.load(policy_file)
    release = policy.get("wheel_release")
    stable_abi = (
        release.get("compiler_python_stable_abi")
        if isinstance(release, dict)
        else None
    )
    if (
        not isinstance(stable_abi, str)
        or not re.fullmatch(r"3\.[0-9]+", stable_abi)
        or stable_abi != COMPILER_STABLE_ABI_MINOR
        or stable_abi not in compiler_python_versions(policy_path)
    ):
        raise RuntimeError(
            "wheel_release.compiler_python_stable_abi must match the pinned "
            f"MLIR stable ABI floor {COMPILER_STABLE_ABI_MINOR}: {policy_path}"
        )
    return stable_abi


def validate_release_version(
    package_version: str, policy_path: Path = RELEASE_POLICY_PATH
) -> str:
    """Requires a development, RC, or stable version in the configured train."""
    base_version = release_base_version(policy_path)
    package_version = validate_version(package_version)
    if not re.fullmatch(
        rf"{re.escape(base_version)}(?:\.dev[0-9]+|rc[0-9]+)?", package_version
    ):
        raise RuntimeError(
            f"release version {package_version} does not belong to the "
            f"configured {base_version} release train"
        )
    return package_version


def resolve_release_inputs(
    source_ref: str,
    package_version: str,
    run_number: int,
    *,
    publish_target: str = "none",
    now: datetime | None = None,
    policy_path: Path = RELEASE_POLICY_PATH,
) -> dict[str, str]:
    """Resolves and validates the inputs shared by platform release workflows."""
    if not source_ref or any(character in source_ref for character in "\r\n"):
        raise RuntimeError("release source ref must be a non-empty single line")
    if run_number < 1:
        raise RuntimeError("GitHub run number must be positive")
    if publish_target not in ("none", "github-prerelease"):
        raise RuntimeError(f"unsupported wheel publish target: {publish_target}")
    base_version = release_base_version(policy_path)
    if not package_version:
        timestamp = now or datetime.now(timezone.utc)
        package_version = (
            f"{base_version}.dev"
            f"{timestamp.astimezone(timezone.utc):%Y%m%d}{run_number:04d}"
        )
    package_version = validate_release_version(package_version, policy_path)
    is_prerelease = package_version != base_version
    if publish_target == "github-prerelease" and not is_prerelease:
        raise RuntimeError(
            f"stable release {package_version} cannot be published as a GitHub "
            "prerelease; use publish_target=none for validation"
        )
    return {
        "ref": source_ref,
        "package_version": package_version,
        "is_prerelease": str(is_prerelease).lower(),
        "tag_name": f"rocm-npu-v{package_version}",
        "release_name": f"rocm-npu {package_version}",
    }


def _wheel_metadata(wheel: Path):
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"expected one METADATA file in wheel: {wheel}")
        return email.parser.Parser().parsestr(
            archive.read(metadata_names[0]).decode("utf-8")
        )


def _wheel_tags(wheel: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(wheel_names) != 1:
            raise RuntimeError(f"expected one WHEEL file in wheel: {wheel}")
        metadata = email.parser.Parser().parsestr(
            archive.read(wheel_names[0]).decode("utf-8")
        )
    return tuple(metadata.get_all("Tag") or ())


def _copy_release_wheel(wheel: Path, output_directory: Path) -> dict[str, object]:
    checksum = wheel.with_name(wheel.name + ".sha256")
    digest = sha256_file(wheel)
    fields = checksum.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != wheel.name:
        raise RuntimeError(f"malformed wheel checksum: {checksum}")
    shutil.copy2(wheel, output_directory / wheel.name)
    shutil.copy2(checksum, output_directory / checksum.name)
    return {
        "distribution": _wheel_metadata(wheel)["Name"],
        "filename": wheel.name,
        "sha256": digest,
        "size": wheel.stat().st_size,
        "wheel_tags": list(_wheel_tags(wheel)),
    }


def _find_release_wheels(
    wheel_directory: Path, expected_wheel_tags: tuple[str, ...]
) -> list[Path]:
    if not expected_wheel_tags:
        tags = {
            "py3-none-any",
            "py3-none-manylinux_2_28_x86_64",
            "py3-none-linux_x86_64",
            "py3-none-win_amd64",
        }
        abi = "cp" + compiler_python_stable_abi().replace(".", "")
        tags.update(
            {
                f"{abi}-abi3-manylinux_2_28_x86_64",
                f"{abi}-abi3-win_amd64",
            }
        )
        expected_wheel_tags = tuple(sorted(tags))
    expected_tags = set(expected_wheel_tags)
    wheels: dict[tuple[str, str], Path] = {}
    unexpected = []
    for wheel in sorted(wheel_directory.rglob("*.whl")):
        distribution = _wheel_metadata(wheel)["Name"]
        if distribution not in RELEASE_DISTRIBUTIONS:
            unexpected.append(wheel.name)
            continue
        matched_tags = expected_tags.intersection(_wheel_tags(wheel))
        if expected_tags and len(matched_tags) != 1:
            raise RuntimeError(
                f"{distribution} wheel tags {_wheel_tags(wheel)} do not include exactly "
                f"one expected tag: {expected_wheel_tags}"
            )
        wheel_tag = next(iter(matched_tags), "")
        key = (distribution, wheel_tag)
        if key in wheels:
            raise RuntimeError(
                f"release wheel set contains duplicate {distribution} wheels "
                f"for tag {wheel_tag or '<unspecified>'}"
            )
        wheels[key] = wheel
    if unexpected:
        raise RuntimeError(
            "release wheel directory contains unexpected wheels:\n  "
            + "\n  ".join(unexpected)
        )

    manylinux = next(
        (
            tag for tag in expected_tags
            if tag.startswith("py3-none-manylinux_") and tag.endswith("_x86_64")
        ),
        None,
    )
    linux = "py3-none-linux_x86_64"
    windows = "py3-none-win_amd64"
    required_base_tags = {"py3-none-any", linux, windows}
    if manylinux:
        required_base_tags.add(manylinux)
    missing_tags = sorted(required_base_tags - expected_tags)
    if missing_tags:
        raise RuntimeError(
            "release wheel expected tags omit required compiler/runtime tags: "
            + ", ".join(missing_tags)
        )
    if not manylinux:
        raise RuntimeError("release wheel expected tags omit a manylinux x86_64 tag")
    expected_keys = {
        (RUNTIME_DISTRIBUTION, manylinux),
        (RUNTIME_DISTRIBUTION, windows),
        (FASTFLOWLM_DISTRIBUTION, linux),
        (FASTFLOWLM_DISTRIBUTION, windows),
        (COMPILER_DISTRIBUTION, "py3-none-any"),
        (COMPILER_CORE_DISTRIBUTION, manylinux),
        (COMPILER_CORE_DISTRIBUTION, windows),
    }
    abi = "cp" + compiler_python_stable_abi().replace(".", "")
    expected_keys.add(
        (COMPILER_BINDINGS_DISTRIBUTION, f"{abi}-abi3-{manylinux.split('-', 2)[2]}")
    )
    expected_keys.add(
        (COMPILER_BINDINGS_DISTRIBUTION, f"{abi}-abi3-win_amd64")
    )
    missing = sorted(expected_keys.difference(wheels))
    if missing:
        details = ", ".join(
            f"{distribution}[{tag or 'unspecified'}]"
            for distribution, tag in missing
        )
        raise RuntimeError(f"release wheel set is incomplete; missing {details}")
    return [wheels[key] for key in sorted(expected_keys)]


def _write_simple_index(
    release_directory: Path, artifacts: list[dict[str, object]]
) -> None:
    links = []
    for artifact in sorted(artifacts, key=lambda item: str(item["filename"])):
        filename = str(artifact["filename"])
        digest = str(artifact["sha256"])
        links.append(
            f'<a href="{html.escape(filename)}#sha256={digest}">'
            f"{html.escape(filename)}</a>"
        )
    (release_directory / "index.html").write_text(
        "<!doctype html>\n<html><body>\n"
        + "<br/>\n".join(links)
        + "\n</body></html>\n",
        encoding="utf-8",
    )

    simple = release_directory / "simple"
    distributions = sorted({str(artifact["distribution"]) for artifact in artifacts})
    simple.mkdir(parents=True)
    (simple / "index.html").write_text(
        "<!doctype html>\n<html><body>\n"
        + "<br/>\n".join(
            f'<a href="{html.escape(distribution)}/">{html.escape(distribution)}</a>'
            for distribution in distributions
        )
        + "\n</body></html>\n",
        encoding="utf-8",
    )
    for distribution in distributions:
        package_directory = simple / distribution
        package_directory.mkdir(parents=True)
        package_links = []
        for artifact in sorted(
            (item for item in artifacts if str(item["distribution"]) == distribution),
            key=lambda item: str(item["filename"]),
        ):
            filename = str(artifact["filename"])
            digest = str(artifact["sha256"])
            package_links.append(
                f'<a href="../../{html.escape(filename)}#sha256={digest}">'
                f"{html.escape(filename)}</a>"
            )
        (package_directory / "index.html").write_text(
            "<!doctype html>\n<html><body>\n" + "<br/>\n".join(package_links) + "\n"
            "</body></html>\n",
            encoding="utf-8",
        )


def _write_release_checksums(release_directory: Path) -> None:
    checksum_lines = []
    # GitHub Release assets are flat. Keep this aggregate limited to the
    # top-level files that the publication job actually uploads.
    files = (item for item in release_directory.iterdir() if item.is_file())
    for path in sorted(files):
        if path.name == "SHA256SUMS":
            continue
        checksum_lines.append(
            f"{sha256_file(path)}  {path.relative_to(release_directory).as_posix()}"
        )
    (release_directory / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )


def prepare_rocm_npu_release_payload(
    wheel_directory: Path,
    release_directory: Path,
    *,
    version: str,
    release_name: str,
    tag_name: str,
    source_ref: str,
    source_sha: str,
    expected_wheel_tags: tuple[str, ...] = (),
    github_run_id: str = "",
    github_run_attempt: str = "",
) -> Path:
    """Validates tested wheels and stages the complete release payload."""
    version = validate_release_version(version)
    wheel_directory = Path(wheel_directory)
    release_directory = Path(release_directory)
    wheels = _find_release_wheels(wheel_directory, expected_wheel_tags)
    verify_wheel_checksums({wheel.name: wheel for wheel in wheels})
    reset_directory(release_directory, release_directory.parent)

    artifacts = []
    for wheel in wheels:
        metadata = _wheel_metadata(wheel)
        distribution = metadata["Name"]
        expected_prefix = distribution.replace("-", "_") + "-"
        if "staging" in wheel.name.casefold():
            raise RuntimeError(f"release wheel name contains staging: {wheel.name}")
        if not wheel.name.startswith(expected_prefix):
            raise RuntimeError(
                f"{distribution} wheel filename has wrong normalized prefix: "
                f"{wheel.name}"
            )
        if metadata["Version"] != version:
            raise RuntimeError(
                f"{distribution} version mismatch: {metadata['Version']} != {version}"
            )
        expected_requirements = {
            FASTFLOWLM_DISTRIBUTION: {
                f"{RUNTIME_DISTRIBUTION} =={version}",
            },
            COMPILER_DISTRIBUTION: {
                f"{COMPILER_CORE_DISTRIBUTION} =={version}",
                f"{COMPILER_BINDINGS_DISTRIBUTION} =={version}",
            },
            COMPILER_BINDINGS_DISTRIBUTION: {
                f"{COMPILER_CORE_DISTRIBUTION} =={version}",
                *COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
            },
            RUNTIME_DISTRIBUTION: set(),
            COMPILER_CORE_DISTRIBUTION: set(),
        }[distribution]
        actual_requirements = set(metadata.get_all("Requires-Dist") or [])
        if actual_requirements != expected_requirements:
            raise RuntimeError(
                f"{distribution} requirements mismatch: "
                f"{sorted(actual_requirements)} != "
                f"{sorted(expected_requirements)}"
            )
        if distribution == COMPILER_BINDINGS_DISTRIBUTION:
            tags = _wheel_tags(wheel)
            stable_abi = "cp" + compiler_python_stable_abi().replace(".", "")
            if not any(
                re.fullmatch(rf"{stable_abi}-abi3-.+", tag) for tag in tags
            ):
                raise RuntimeError(
                    f"{COMPILER_BINDINGS_DISTRIBUTION} must use the "
                    f"{stable_abi}-abi3 stable ABI tag: "
                    f"{tags}"
                )
        artifacts.append(_copy_release_wheel(wheel, release_directory))

    artifacts.sort(key=lambda item: str(item["filename"]))
    manifest = {
        "schema_version": 1,
        "product": "rocm-npu",
        "version": version,
        "release_name": release_name,
        "tag_name": tag_name,
        "source_ref": source_ref,
        "source_sha": source_sha,
        "github_run_id": github_run_id,
        "github_run_attempt": github_run_attempt,
        "artifacts": artifacts,
    }
    (release_directory / "rocm-npu-release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_simple_index(release_directory, artifacts)
    _write_release_checksums(release_directory)
    return release_directory


# Backward-compatible import for callers migrating with the expanded release.
prepare_runtime_fastflowlm_release_payload = prepare_rocm_npu_release_payload
