# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Platform-neutral ROCm NPU wheel release payload preparation."""

from __future__ import annotations

import email.parser
import hashlib
import html
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from build_tools import release_policy
from build_tools.archive import read_member
from build_tools.artifact import (
    reset_directory,
    sha256_file,
    sha256_sidecar_path,
    verify_sha256_sidecar,
)
from build_tools.fastflowlm_artifact import artifact_name as standalone_artifact_name
from build_tools.rocm_npu_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    FASTFLOWLM_DISTRIBUTION,
    OPS_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
    validate_version,
    verify_wheel_checksums,
)

ROOT = Path(__file__).resolve().parents[1]
NUMERIC_RUNTIME = release_policy.numeric_package_versions()


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


RELEASE_DISTRIBUTIONS = (
    RUNTIME_DISTRIBUTION,
    FASTFLOWLM_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_BINDINGS_DISTRIBUTION,
    OPS_DISTRIBUTION,
)


def validate_release_version(
    package_version: str,
    policy_path: Path = release_policy.RELEASE_POLICY_PATH,
) -> str:
    """Requires a development, RC, or stable version in the configured train."""
    base_version = release_policy.release_base_version(policy_path)
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
    policy_path: Path = release_policy.RELEASE_POLICY_PATH,
) -> dict[str, str]:
    """Resolves and validates the inputs shared by platform release workflows."""
    if not source_ref or any(character in source_ref for character in "\r\n"):
        raise RuntimeError("release source ref must be a non-empty single line")
    if run_number < 1:
        raise RuntimeError("GitHub run number must be positive")
    if publish_target not in ("none", "github-prerelease"):
        raise RuntimeError(f"unsupported wheel publish target: {publish_target}")
    base_version = release_policy.release_base_version(policy_path)
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


def _copy_release_wheel(
    wheel: Path, output_directory: Path, *, expected_source_sha: str
) -> dict[str, object]:
    checksum = sha256_sidecar_path(wheel)
    digest = verify_sha256_sidecar(wheel, kind="wheel")
    shutil.copy2(wheel, output_directory / wheel.name)
    shutil.copy2(checksum, output_directory / checksum.name)
    with zipfile.ZipFile(wheel) as archive:
        manifests = [
            name
            for name in archive.namelist()
            if name.count("/") == 1 and name.endswith("/manifest.json")
        ]
        if len(manifests) != 1:
            raise RuntimeError(
                f"release wheel must contain one package manifest: {wheel}"
            )
        package_manifest_bytes = archive.read(manifests[0])
        package_manifest_sha256 = hashlib.sha256(package_manifest_bytes).hexdigest()
    try:
        package_manifest = json.loads(package_manifest_bytes)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid wheel package manifest: {wheel}") from error
    embedded_revision = (
        package_manifest.get("sources", {}).get("rocm-npu-staging", {}).get("revision")
    )
    if embedded_revision != expected_source_sha:
        raise RuntimeError(
            "wheel source revision does not match release source: "
            f"{wheel}: embedded={embedded_revision!r}, "
            f"expected={expected_source_sha!r}"
        )
    return {
        "distribution": _wheel_metadata(wheel)["Name"],
        "filename": wheel.name,
        "sha256": digest,
        "package_manifest_sha256": package_manifest_sha256,
        "size": wheel.stat().st_size,
        "wheel_tags": list(_wheel_tags(wheel)),
    }


def _copy_release_archives(
    archive_directory: Path,
    output_directory: Path,
    *,
    expected_version: str,
    expected_source_sha: str,
) -> list[dict[str, object]]:
    linux_artifact = standalone_artifact_name("linux", "x86_64")
    windows_artifact = standalone_artifact_name("windows", "x86_64")
    expected = {
        f"{linux_artifact}.tar.zst": ("linux-x86_64", "Linux", linux_artifact),
        f"{windows_artifact}.zip": ("windows-x86_64", "Windows", windows_artifact),
    }
    archives = {
        path.name: path
        for path in Path(archive_directory).rglob("*")
        if path.is_file() and path.name in expected
    }
    if set(archives) != set(expected):
        missing = sorted(set(expected) - set(archives))
        raise RuntimeError(
            "release standalone archive set is incomplete; missing "
            + ", ".join(missing)
        )
    result = []
    for name, (platform_name, system_name, artifact_name) in expected.items():
        archive = archives[name]
        checksum = sha256_sidecar_path(archive)
        provenance = archive.with_name(archive.name + ".manifest.json")
        digest = verify_sha256_sidecar(archive, kind="standalone archive")
        try:
            provenance_data = json.loads(provenance.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"standalone archive provenance is invalid: {provenance}"
            ) from error
        bundle = provenance_data.get("bundle")
        archive_manifest_name = f"{artifact_name}/manifest.json"
        try:
            embedded_bundles = [
                json.loads(
                    read_member(
                        archive,
                        archive_manifest_name,
                        size_limit=10_000_000,
                    ).decode("utf-8")
                )
            ]
        except (
            OSError,
            RuntimeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise RuntimeError(
                f"standalone archive manifest is invalid: {archive}"
            ) from error
        staging_source = (
            bundle.get("sources", {}).get("rocm-npu-staging", {})
            if isinstance(bundle, dict)
            else {}
        )
        if (
            provenance_data.get("schema_version") != 1
            or provenance_data.get("archive") != name
            or provenance_data.get("archive_sha256") != digest
            or not isinstance(bundle, dict)
            or embedded_bundles != [bundle]
            or bundle.get("schema_version") != 1
            or bundle.get("version") != expected_version
            or bundle.get("artifact") != artifact_name
            or bundle.get("platform", {}).get("system") != system_name
            or staging_source.get("revision") != expected_source_sha
            or staging_source.get("dirty") is not False
            or not isinstance(bundle.get("files"), list)
            or not bundle["files"]
        ):
            raise RuntimeError(
                f"standalone archive provenance does not match release: {archive}"
            )
        shutil.copy2(archive, output_directory / archive.name)
        shutil.copy2(checksum, output_directory / checksum.name)
        shutil.copy2(provenance, output_directory / provenance.name)
        result.append(
            {
                "kind": "standalone-runtime",
                "platform": platform_name,
                "filename": archive.name,
                "sha256": digest,
                "size": archive.stat().st_size,
                "provenance": provenance.name,
            }
        )
    return result


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
        abi = "cp" + release_policy.compiler_python_stable_abi().replace(".", "")
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
        distribution = _canonical_distribution(_wheel_metadata(wheel)["Name"])
        if distribution in NUMERIC_RUNTIME:
            continue
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
            tag
            for tag in expected_tags
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
        (QUALIFICATION_DISTRIBUTION, "py3-none-any"),
        (OPS_DISTRIBUTION, "py3-none-any"),
        (COMPILER_CORE_DISTRIBUTION, manylinux),
        (COMPILER_CORE_DISTRIBUTION, windows),
    }
    abi = "cp" + release_policy.compiler_python_stable_abi().replace(".", "")
    expected_keys.add(
        (COMPILER_BINDINGS_DISTRIBUTION, f"{abi}-abi3-{manylinux.split('-', 2)[2]}")
    )
    expected_keys.add((COMPILER_BINDINGS_DISTRIBUTION, f"{abi}-abi3-win_amd64"))
    missing = sorted(expected_keys.difference(wheels))
    if missing:
        details = ", ".join(
            f"{distribution}[{tag or 'unspecified'}]" for distribution, tag in missing
        )
        raise RuntimeError(f"release wheel set is incomplete; missing {details}")
    extra = sorted(set(wheels).difference(expected_keys))
    if extra:
        details = ", ".join(
            f"{distribution}[{tag or 'unspecified'}]" for distribution, tag in extra
        )
        raise RuntimeError(f"release wheel set contains unexpected variants: {details}")
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
    archive_directory: Path | None = None,
) -> Path:
    """Validates tested wheels and stages the complete release payload."""
    version = validate_release_version(version)
    wheel_directory = Path(wheel_directory)
    release_directory = Path(release_directory)
    wheels = _find_release_wheels(wheel_directory, expected_wheel_tags)
    verify_wheel_checksums({wheel.name: wheel for wheel in wheels})
    reset_directory(release_directory, release_directory.parent)

    wheel_artifacts = []
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
            QUALIFICATION_DISTRIBUTION: {
                f"{FASTFLOWLM_DISTRIBUTION} =={version}",
                *(
                    f"{distribution} =={numeric_version}"
                    for distribution, numeric_version in NUMERIC_RUNTIME.items()
                ),
            },
            COMPILER_DISTRIBUTION: {
                f"{COMPILER_CORE_DISTRIBUTION} =={version}",
                f"{COMPILER_BINDINGS_DISTRIBUTION} =={version}",
            },
            OPS_DISTRIBUTION: {
                f"{COMPILER_DISTRIBUTION} =={version}",
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
            stable_abi = "cp" + release_policy.compiler_python_stable_abi().replace(
                ".", ""
            )
            if not any(re.fullmatch(rf"{stable_abi}-abi3-.+", tag) for tag in tags):
                raise RuntimeError(
                    f"{COMPILER_BINDINGS_DISTRIBUTION} must use the "
                    f"{stable_abi}-abi3 stable ABI tag: "
                    f"{tags}"
                )
        wheel_artifacts.append(
            _copy_release_wheel(
                wheel,
                release_directory,
                expected_source_sha=source_sha,
            )
        )

    artifacts = list(wheel_artifacts)
    if archive_directory is not None:
        artifacts.extend(
            _copy_release_archives(
                Path(archive_directory),
                release_directory,
                expected_version=version,
                expected_source_sha=source_sha,
            )
        )
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
    _write_simple_index(release_directory, wheel_artifacts)
    _write_release_checksums(release_directory)
    return release_directory
