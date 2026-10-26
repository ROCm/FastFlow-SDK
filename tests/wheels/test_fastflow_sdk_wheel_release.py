# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import zstandard

from build_tools.fastflow_sdk_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_PACKAGE,
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    COMPILER_PACKAGE,
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    OPS_DISTRIBUTION,
    OPS_PACKAGE,
    QUALIFICATION_DISTRIBUTION,
    QUALIFICATION_PACKAGE,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    SDK_DISTRIBUTION,
    SDK_PACKAGE,
    build_wheel,
)
from build_tools.fastflow_sdk_wheel_release import (
    _copy_release_archives,
    prepare_fastflow_sdk_release_payload,
    resolve_release_inputs,
)
from build_tools.release_policy import (
    numeric_package_versions,
    numeric_python_abis,
)
from build_tools.wheel_payload import write_package_manifest as _write_package_manifest
from tests.support.wheels import make_package_tree

NUMERIC_RUNTIME = numeric_package_versions()
NUMERIC_PYTHON_ABIS = numeric_python_abis()


pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


def _write_release_archive(directory, name, *, system, artifact, version, source_sha):
    bundle = {
        "schema_version": 1,
        "version": version,
        "artifact": artifact,
        "platform": {"system": system},
        "sources": {
            "fastflow-sdk": {
                "revision": source_sha,
                "dirty": False,
            }
        },
        "files": [{"path": "bin/fastflow"}],
    }
    payload = (json.dumps(bundle) + "\n").encode()
    archive = directory / name
    manifest_name = f"{artifact}/manifest.json"
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr(manifest_name, payload)
    else:
        uncompressed = io.BytesIO()
        with tarfile.open(fileobj=uncompressed, mode="w") as tar:
            info = tarfile.TarInfo(manifest_name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        archive.write_bytes(
            zstandard.ZstdCompressor().compress(uncompressed.getvalue())
        )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(name + ".sha256").write_text(
        f"{digest}  {name}\n", encoding="ascii"
    )
    archive.with_name(name + ".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "archive": name,
                "archive_sha256": digest,
                "bundle": bundle,
            }
        ),
        encoding="utf-8",
    )
    return archive


class TestFastflowSdkWheelRelease:
    def test_resolve_cli_does_not_require_release_validation_packages(self, tmp_path):
        output = tmp_path / "github-output"
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "build_tools/fastflow_sdk_wheel_release_cli.py",
                "resolve",
                "--ref",
                "release-branch",
                "--package-version",
                "0.1.0.dev1",
                "--run-number",
                "1",
                "--github-output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "package_version=0.1.0.dev1" in output.read_text(encoding="utf-8")

    def test_copy_release_archives_requires_and_records_both_platforms(self, tmp_path):
        archives = tmp_path / "archives"
        output = tmp_path / "release"
        archives.mkdir()
        output.mkdir()
        archive_metadata = {
            "fastflow-sdk-linux-x86_64.tar.zst": ("Linux", "fastflow-sdk-linux-x86_64"),
            "fastflow-sdk-windows-x86_64.zip": (
                "Windows",
                "fastflow-sdk-windows-x86_64",
            ),
        }
        for name, (system, artifact) in archive_metadata.items():
            _write_release_archive(
                archives,
                name,
                system=system,
                artifact=artifact,
                version="1.2.3",
                source_sha="abc123",
            )
        records = _copy_release_archives(
            archives,
            output,
            expected_version="1.2.3",
            expected_source_sha="abc123",
        )
        assert {record["platform"] for record in records} == {
            "linux-x86_64",
            "windows-x86_64",
        }
        assert (output / "fastflow-sdk-windows-x86_64.zip").is_file()
        stale_archive = _write_release_archive(
            archives,
            "fastflow-sdk-windows-x86_64.zip",
            system="Windows",
            artifact="fastflow-sdk-windows-x86_64",
            version="1.2.2",
            source_sha="abc123",
        )
        stale = stale_archive.with_name(stale_archive.name + ".manifest.json")
        stale_data = json.loads(stale.read_text(encoding="utf-8"))
        stale_data["bundle"]["version"] = "1.2.3"
        stale.write_text(json.dumps(stale_data), encoding="utf-8")
        with pytest.raises(RuntimeError, match="manifest is invalid|does not match"):
            _copy_release_archives(
                archives,
                output,
                expected_version="1.2.3",
                expected_source_sha="abc123",
            )

    def test_resolve_release_inputs_uses_explicit_version(self, tmp_path):
        policy = self._write_policy(tmp_path, "1.2.3")
        release = resolve_release_inputs(
            "release-branch",
            "1.2.3rc1",
            7,
            publish_target="github-prerelease",
            policy_path=policy,
        )

        assert release["ref"] == "release-branch"
        assert release["package_version"] == "1.2.3rc1"
        assert release["is_prerelease"] == "true"
        assert release["tag_name"] == "fastflow-sdk-v1.2.3rc1"
        assert release["release_name"] == "FastFlow SDK 1.2.3rc1"

    def test_resolve_release_inputs_allows_stable_validation(self, tmp_path):
        policy = self._write_policy(tmp_path, "1.2.3")
        release = resolve_release_inputs(
            "main",
            "1.2.3",
            7,
            publish_target="none",
            policy_path=policy,
        )

        assert release["package_version"] == "1.2.3"
        assert release["is_prerelease"] == "false"

    def test_resolve_release_inputs_rejects_stable_prerelease_publication(
        self, tmp_path
    ):
        policy = self._write_policy(tmp_path, "1.2.3")
        with pytest.raises(
            RuntimeError, match="cannot be published as a GitHub prerelease"
        ):
            resolve_release_inputs(
                "main",
                "1.2.3",
                7,
                publish_target="github-prerelease",
                policy_path=policy,
            )

    def test_resolve_release_inputs_rejects_version_from_another_train(self, tmp_path):
        policy = self._write_policy(tmp_path, "1.2.3")
        with pytest.raises(
            RuntimeError, match="does not belong to the configured 1.2.3 release train"
        ):
            resolve_release_inputs(
                "release-branch",
                "9.0.0rc1",
                7,
                policy_path=policy,
            )

    @pytest.mark.parametrize("version", ("1.2.3a1", "1.2.3.post1", "1.2.3+local"))
    def test_resolve_release_inputs_rejects_non_train_release_forms(
        self, tmp_path, version
    ):
        policy = self._write_policy(tmp_path, "1.2.3")
        with pytest.raises(RuntimeError, match="does not belong"):
            resolve_release_inputs("main", version, 1, policy_path=policy)

    def test_resolve_release_inputs_generates_utc_development_version(self, tmp_path):
        policy = self._write_policy(tmp_path, "2.3.0")
        release = resolve_release_inputs(
            "main",
            "",
            7,
            now=datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
            policy_path=policy,
        )

        assert release["package_version"] == "2.3.0.dev202607240007"

    def test_prepare_release_payload_validates_and_indexes_tested_wheels(
        self, tmp_path
    ):
        root = tmp_path
        version = "0.1.0.dev1"
        self._build_release_wheels(
            root,
            version,
            output_directory=root / "dist" / "windows",
        )
        self._build_release_wheels(
            root,
            version,
            wheel_tag="py3-none-manylinux_2_28_x86_64",
            fastflowlm_wheel_tag="py3-none-linux_x86_64",
            output_directory=root / "dist" / "linux",
            include_meta=False,
        )
        archives = root / "archives"
        archives.mkdir()
        for name, system, artifact in (
            (
                "fastflow-sdk-linux-x86_64.tar.zst",
                "Linux",
                "fastflow-sdk-linux-x86_64",
            ),
            (
                "fastflow-sdk-windows-x86_64.zip",
                "Windows",
                "fastflow-sdk-windows-x86_64",
            ),
        ):
            _write_release_archive(
                archives,
                name,
                system=system,
                artifact=artifact,
                version=version,
                source_sha="abc123",
            )

        release = prepare_fastflow_sdk_release_payload(
            root / "dist",
            root / "release",
            version=version,
            release_name=f"fastflow {version}",
            tag_name=f"fastflow-sdk-v{version}",
            source_ref="main",
            source_sha="abc123",
            archive_directory=archives,
            expected_wheel_tags=self._expected_tags(),
            github_run_id="42",
            github_run_attempt="1",
        )

        manifest = json.loads(
            (release / "fastflow-sdk-release-manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["product"] == "fastflow-sdk"
        assert manifest["version"] == version
        assert manifest["source_ref"] == "main"
        assert manifest["source_sha"] == "abc123"
        wheel_artifacts = [
            artifact for artifact in manifest["artifacts"] if "distribution" in artifact
        ]
        archive_artifacts = [
            artifact
            for artifact in manifest["artifacts"]
            if artifact.get("kind") == "standalone-runtime"
        ]
        assert {artifact["platform"] for artifact in archive_artifacts} == {
            "linux-x86_64",
            "windows-x86_64",
        }
        for artifact in archive_artifacts:
            assert (release / artifact["filename"]).is_file()
            assert (release / artifact["provenance"]).is_file()
        distribution_counts = {}
        for artifact in wheel_artifacts:
            distribution_counts[artifact["distribution"]] = (
                distribution_counts.get(artifact["distribution"], 0) + 1
            )
        assert distribution_counts == {
            FASTFLOWLM_DISTRIBUTION: 2,
            RUNTIME_DISTRIBUTION: 2,
            COMPILER_DISTRIBUTION: 1,
            QUALIFICATION_DISTRIBUTION: 1,
            OPS_DISTRIBUTION: 1,
            SDK_DISTRIBUTION: 1,
            COMPILER_CORE_DISTRIBUTION: 2,
            COMPILER_BINDINGS_DISTRIBUTION: 2,
        }
        artifact_tags = set()
        for artifact in wheel_artifacts:
            filename = artifact["filename"]
            assert "staging" not in filename
            assert artifact.get("kind") != "qualification-numeric-runtime"
            assert len(artifact["package_manifest_sha256"]) == 64
            artifact_tags.update(artifact["wheel_tags"])
            assert (release / filename).is_file()
            assert (release / f"{filename}.sha256").is_file()
        assert artifact_tags == {
            "py3-none-manylinux_2_28_x86_64",
            "py3-none-linux_x86_64",
            "py3-none-win_amd64",
            "py3-none-any",
            "cp312-abi3-manylinux_2_28_x86_64",
            "cp312-abi3-win_amd64",
        }
        assert not list(release.glob("numpy-*.whl"))
        assert not list(release.glob("ml_dtypes-*.whl"))

        simple = release / "simple"
        assert (simple / RUNTIME_DISTRIBUTION / "index.html").is_file()
        assert (simple / FASTFLOWLM_DISTRIBUTION / "index.html").is_file()
        for distribution in (
            RUNTIME_DISTRIBUTION,
            FASTFLOWLM_DISTRIBUTION,
            COMPILER_DISTRIBUTION,
            QUALIFICATION_DISTRIBUTION,
            OPS_DISTRIBUTION,
            SDK_DISTRIBUTION,
            COMPILER_CORE_DISTRIBUTION,
            COMPILER_BINDINGS_DISTRIBUTION,
        ):
            index = (simple / distribution / "index.html").read_text(encoding="utf-8")
            if distribution == RUNTIME_DISTRIBUTION:
                assert "manylinux_2_28_x86_64" in index
            elif distribution == FASTFLOWLM_DISTRIBUTION:
                assert "py3-none-linux_x86_64" in index
            if distribution not in (
                COMPILER_DISTRIBUTION,
                QUALIFICATION_DISTRIBUTION,
                OPS_DISTRIBUTION,
                SDK_DISTRIBUTION,
            ):
                assert "win_amd64" in index
        checksum_names = {
            line.split("  ", 1)[1]
            for line in (release / "SHA256SUMS")
            .read_text(encoding="ascii")
            .splitlines()
        }
        published_names = {
            path.name
            for path in release.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        }
        assert checksum_names == published_names
        assert "fastflow-sdk-release-manifest.json" in checksum_names
        assert not any(name.startswith("simple/") for name in checksum_names)

    def test_prepare_release_payload_rejects_version_outside_policy(self, tmp_path):
        root = tmp_path
        with pytest.raises(RuntimeError, match="does not belong"):
            prepare_fastflow_sdk_release_payload(
                root / "dist",
                root / "release",
                version="9.0.0rc1",
                release_name="fastflow 9.0.0rc1",
                tag_name="fastflow-sdk-v9.0.0rc1",
                source_ref="main",
                source_sha="abc123",
            )

    def test_prepare_release_payload_requires_matching_runtime_dependency(
        self, tmp_path
    ):
        root = tmp_path
        version = "0.1.0.dev1"
        self._build_release_wheels(
            root,
            version,
            runtime_requirement="0.1.0.dev0",
            output_directory=root / "dist" / "windows",
        )
        self._build_release_wheels(
            root,
            version,
            runtime_requirement="0.1.0.dev0",
            wheel_tag="py3-none-manylinux_2_28_x86_64",
            fastflowlm_wheel_tag="py3-none-linux_x86_64",
            output_directory=root / "dist" / "linux",
            include_meta=False,
        )

        with pytest.raises(
            RuntimeError, match=f"{FASTFLOWLM_DISTRIBUTION} requirements mismatch"
        ):
            prepare_fastflow_sdk_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"fastflow {version}",
                tag_name=f"fastflow-sdk-v{version}",
                source_ref="main",
                source_sha="abc123",
            )

    def test_prepare_release_payload_requires_binding_runtime_dependencies(
        self, tmp_path
    ):
        root = tmp_path
        version = "0.1.0.dev1"
        self._build_release_wheels(
            root,
            version,
            binding_runtime_requirements=(),
            output_directory=root / "dist" / "windows",
        )
        self._build_release_wheels(
            root,
            version,
            binding_runtime_requirements=(),
            wheel_tag="py3-none-manylinux_2_28_x86_64",
            fastflowlm_wheel_tag="py3-none-linux_x86_64",
            output_directory=root / "dist" / "linux",
            include_meta=False,
        )

        with pytest.raises(
            RuntimeError,
            match=f"{COMPILER_BINDINGS_DISTRIBUTION} requirements mismatch",
        ):
            prepare_fastflow_sdk_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"fastflow {version}",
                tag_name=f"fastflow-sdk-v{version}",
                source_ref="main",
                source_sha="abc123",
            )

    def test_prepare_release_payload_requires_sdk_to_pin_every_product(self, tmp_path):
        root = tmp_path
        version = "0.1.0.dev1"
        self._build_release_wheels(
            root,
            version,
            sdk_requirements=(
                f"{FASTFLOWLM_DISTRIBUTION} =={version}",
                f"{COMPILER_DISTRIBUTION} =={version}",
            ),
            output_directory=root / "dist" / "windows",
        )
        self._build_release_wheels(
            root,
            version,
            wheel_tag="py3-none-manylinux_2_28_x86_64",
            fastflowlm_wheel_tag="py3-none-linux_x86_64",
            output_directory=root / "dist" / "linux",
            include_meta=False,
        )

        with pytest.raises(
            RuntimeError, match=f"{SDK_DISTRIBUTION} requirements mismatch"
        ):
            prepare_fastflow_sdk_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"fastflow {version}",
                tag_name=f"fastflow-sdk-v{version}",
                source_ref="main",
                source_sha="abc123",
            )

    def test_prepare_release_payload_rejects_extra_wheels(self, tmp_path):
        root = tmp_path
        version = "0.1.0.dev1"
        self._build_release_wheels(root, version)
        build_wheel(
            self._package_tree(root, "fastflow_sdk_extra"),
            "fastflow-sdk-extra",
            version,
            root / "dist",
            summary="extra",
            tag="py3-none-win_amd64",
            license_expression="MIT",
        )

        with pytest.raises(
            RuntimeError, match="release wheel directory contains unexpected wheels"
        ):
            prepare_fastflow_sdk_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"fastflow {version}",
                tag_name=f"fastflow-sdk-v{version}",
                source_ref="main",
                source_sha="abc123",
            )

    def test_prepare_release_payload_rejects_extra_allowed_variant(self, tmp_path):
        root = tmp_path
        version = "0.1.0.dev1"
        self._build_release_wheels(
            root, version, output_directory=root / "dist/windows"
        )
        self._build_release_wheels(
            root,
            version,
            wheel_tag="py3-none-manylinux_2_28_x86_64",
            fastflowlm_wheel_tag="py3-none-linux_x86_64",
            output_directory=root / "dist/linux",
            include_meta=False,
        )
        build_wheel(
            self._package_tree(root, "qualification_windows_variant"),
            QUALIFICATION_DISTRIBUTION,
            version,
            root / "dist/extra",
            summary="unexpected qualification variant",
            tag="py3-none-win_amd64",
            requires=(f"{FASTFLOWLM_DISTRIBUTION} =={version}",),
        )

        with pytest.raises(RuntimeError, match="unexpected variants"):
            prepare_fastflow_sdk_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"fastflow {version}",
                tag_name=f"fastflow-sdk-v{version}",
                source_ref="main",
                source_sha="abc123",
            )

    def test_prepare_release_payload_rejects_wrong_platform_tag(self, tmp_path):
        root = tmp_path
        version = "0.1.0.dev1"
        self._build_release_wheels(
            root, version, wheel_tag="py3-none-manylinux_2_28_x86_64"
        )

        with pytest.raises(RuntimeError, match="release wheel set is incomplete"):
            prepare_fastflow_sdk_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"fastflow {version}",
                tag_name=f"fastflow-sdk-v{version}",
                source_ref="main",
                source_sha="abc123",
                expected_wheel_tags=self._expected_tags(),
            )

    def test_prepare_release_payload_requires_every_platform(self, tmp_path):
        root = tmp_path
        version = "0.1.0.dev1"
        self._build_release_wheels(root, version)

        with pytest.raises(RuntimeError, match="release wheel set is incomplete"):
            prepare_fastflow_sdk_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"fastflow {version}",
                tag_name=f"fastflow-sdk-v{version}",
                source_ref="main",
                source_sha="abc123",
                expected_wheel_tags=self._expected_tags(),
            )

    def _build_release_wheels(
        self,
        root: Path,
        version: str,
        *,
        runtime_requirement: str | None = None,
        binding_runtime_requirements: tuple[
            str, ...
        ] = COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
        wheel_tag: str = "py3-none-win_amd64",
        fastflowlm_wheel_tag: str | None = None,
        output_directory: Path | None = None,
        include_meta: bool = True,
        sdk_requirements: tuple[str, ...] | None = None,
    ) -> None:
        output_directory = output_directory or root / "dist"
        build_wheel(
            self._package_tree(root, RUNTIME_PACKAGE),
            RUNTIME_DISTRIBUTION,
            version,
            output_directory,
            summary="runtime",
            tag=wheel_tag,
        )
        build_wheel(
            self._package_tree(root, FASTFLOWLM_PACKAGE),
            FASTFLOWLM_DISTRIBUTION,
            version,
            output_directory,
            summary="fastflowlm",
            tag=fastflowlm_wheel_tag or wheel_tag,
            requires=(f"{RUNTIME_DISTRIBUTION} =={runtime_requirement or version}",),
            console_scripts={"flm": f"{FASTFLOWLM_PACKAGE}:main"},
        )
        if include_meta:
            build_wheel(
                self._package_tree(root, "fastflow_sdk_compiler_meta"),
                COMPILER_DISTRIBUTION,
                version,
                output_directory,
                summary="compiler",
                tag="py3-none-any",
                requires=(
                    f"{COMPILER_CORE_DISTRIBUTION} =={version}",
                    f"{COMPILER_BINDINGS_DISTRIBUTION} =={version}",
                ),
                root_is_purelib=True,
            )
            build_wheel(
                self._package_tree(root, QUALIFICATION_PACKAGE),
                QUALIFICATION_DISTRIBUTION,
                version,
                output_directory,
                summary="qualification",
                tag="py3-none-any",
                requires=(
                    f"{FASTFLOWLM_DISTRIBUTION} =={version}",
                    *(
                        f"{distribution} =={numeric_version}"
                        for distribution, numeric_version in NUMERIC_RUNTIME.items()
                    ),
                ),
                console_scripts={
                    "fastflow-qualify": f"{QUALIFICATION_PACKAGE}.cli:main"
                },
                root_is_purelib=True,
            )
            build_wheel(
                self._package_tree(root, OPS_PACKAGE),
                OPS_DISTRIBUTION,
                version,
                output_directory,
                summary="ops",
                tag="py3-none-any",
                requires=(f"{COMPILER_DISTRIBUTION} =={version}",),
                root_is_purelib=True,
            )
            build_wheel(
                self._package_tree(root, SDK_PACKAGE),
                SDK_DISTRIBUTION,
                version,
                output_directory,
                summary="sdk",
                tag="py3-none-any",
                requires=sdk_requirements
                or (
                    f"{FASTFLOWLM_DISTRIBUTION} =={version}",
                    f"{COMPILER_DISTRIBUTION} =={version}",
                    f"{OPS_DISTRIBUTION} =={version}",
                ),
                root_is_purelib=True,
            )
        build_wheel(
            self._package_tree(root, COMPILER_PACKAGE),
            COMPILER_CORE_DISTRIBUTION,
            version,
            output_directory,
            summary="compiler core",
            tag=wheel_tag,
        )
        platform_tag = wheel_tag.split("-", 2)[2]
        build_wheel(
            self._package_tree(
                root, f"{COMPILER_BINDINGS_PACKAGE}_abi3_{platform_tag}"
            ),
            COMPILER_BINDINGS_DISTRIBUTION,
            version,
            output_directory,
            summary="compiler bindings",
            tag=f"cp312-abi3-{platform_tag}",
            requires=(
                f"{COMPILER_CORE_DISTRIBUTION} =={version}",
                *binding_runtime_requirements,
            ),
            requires_python=">=3.12,<3.15",
        )
        numeric_platform = wheel_tag.split("-", 2)[2]
        for abi in NUMERIC_PYTHON_ABIS:
            for distribution, numeric_version in NUMERIC_RUNTIME.items():
                build_wheel(
                    self._package_tree(
                        root,
                        f"{distribution.replace('-', '_')}_{abi}_{numeric_platform}",
                    ),
                    distribution,
                    numeric_version,
                    output_directory.parent / "numeric",
                    summary="qualification numeric runtime fixture",
                    tag=f"{abi}-{abi}-{numeric_platform}",
                    license_expression="BSD-3-Clause",
                    root_is_purelib=False,
                )

    def _package_tree(self, root: Path, package_name: str) -> Path:
        tree = make_package_tree(root, package_name, exist_ok=True)
        package = tree / package_name
        _write_package_manifest(
            package,
            {
                "schema_version": 1,
                "product": package_name,
                "sources": {
                    "fastflow-sdk": {"revision": "abc123"},
                },
            },
        )
        return tree

    def _write_policy(self, root: Path, base_version: str) -> Path:
        policy = root / "release.toml"
        policy.write_text(
            f'schema_version = 1\n[wheel_release]\nbase_version = "{base_version}"\n'
            'compiler_python_stable_abi = "3.12"\n'
            'compiler_python_versions = ["3.12", "3.13", "3.14"]\n',
            encoding="utf-8",
        )
        return policy

    def _expected_tags(self) -> tuple[str, ...]:
        tags = {
            "py3-none-any",
            "py3-none-manylinux_2_28_x86_64",
            "py3-none-linux_x86_64",
            "py3-none-win_amd64",
        }
        tags.update(
            {
                "cp312-abi3-manylinux_2_28_x86_64",
                "cp312-abi3-win_amd64",
            }
        )
        return tuple(sorted(tags))
