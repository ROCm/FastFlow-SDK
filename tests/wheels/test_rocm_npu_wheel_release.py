# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from build_tools.rocm_npu_compiler_wheels import (
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
)
from build_tools.rocm_npu_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_PACKAGE,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    COMPILER_PACKAGE,
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    build_wheel,
)
from build_tools.rocm_npu_wheel_release import (
    prepare_rocm_npu_release_payload,
    resolve_release_inputs,
)


pytestmark = [pytest.mark.compiler_stack, pytest.mark.fastflowlm]


class TestRocmNpuWheelRelease:
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
        assert release["tag_name"] == "rocm-npu-v1.2.3rc1"
        assert release["release_name"] == "rocm-npu 1.2.3rc1"

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
            now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
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

        release = prepare_rocm_npu_release_payload(
            root / "dist",
            root / "release",
            version=version,
            release_name=f"rocm-npu {version}",
            tag_name=f"rocm-npu-v{version}",
            source_ref="main",
            source_sha="abc123",
            expected_wheel_tags=self._expected_tags(),
            github_run_id="42",
            github_run_attempt="1",
        )

        manifest = json.loads(
            (release / "rocm-npu-release-manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["product"] == "rocm-npu"
        assert manifest["version"] == version
        assert manifest["source_ref"] == "main"
        assert manifest["source_sha"] == "abc123"
        distribution_counts = {}
        for artifact in manifest["artifacts"]:
            distribution_counts[artifact["distribution"]] = (
                distribution_counts.get(artifact["distribution"], 0) + 1
            )
        assert distribution_counts == {
            FASTFLOWLM_DISTRIBUTION: 2,
            RUNTIME_DISTRIBUTION: 2,
            COMPILER_DISTRIBUTION: 1,
            COMPILER_CORE_DISTRIBUTION: 2,
            COMPILER_BINDINGS_DISTRIBUTION: 2,
        }
        artifact_tags = set()
        for artifact in manifest["artifacts"]:
            filename = artifact["filename"]
            assert "staging" not in filename
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

        simple = release / "simple"
        assert (simple / RUNTIME_DISTRIBUTION / "index.html").is_file()
        assert (simple / FASTFLOWLM_DISTRIBUTION / "index.html").is_file()
        for distribution in (
            RUNTIME_DISTRIBUTION,
            FASTFLOWLM_DISTRIBUTION,
            COMPILER_DISTRIBUTION,
            COMPILER_CORE_DISTRIBUTION,
            COMPILER_BINDINGS_DISTRIBUTION,
        ):
            index = (simple / distribution / "index.html").read_text(encoding="utf-8")
            if distribution == RUNTIME_DISTRIBUTION:
                assert "manylinux_2_28_x86_64" in index
            elif distribution == FASTFLOWLM_DISTRIBUTION:
                assert "py3-none-linux_x86_64" in index
            if distribution != COMPILER_DISTRIBUTION:
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
        assert "rocm-npu-release-manifest.json" in checksum_names
        assert not any((name.startswith("simple/") for name in checksum_names))

    def test_prepare_release_payload_rejects_version_outside_policy(self, tmp_path):
        root = tmp_path
        with pytest.raises(RuntimeError, match="does not belong"):
            prepare_rocm_npu_release_payload(
                root / "dist",
                root / "release",
                version="9.0.0rc1",
                release_name="rocm-npu 9.0.0rc1",
                tag_name="rocm-npu-v9.0.0rc1",
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
            prepare_rocm_npu_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"rocm-npu {version}",
                tag_name=f"rocm-npu-v{version}",
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
            prepare_rocm_npu_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"rocm-npu {version}",
                tag_name=f"rocm-npu-v{version}",
                source_ref="main",
                source_sha="abc123",
            )

    def test_prepare_release_payload_rejects_extra_wheels(self, tmp_path):
        root = tmp_path
        version = "0.1.0.dev1"
        self._build_release_wheels(root, version)
        build_wheel(
            self._package_tree(root, "rocm_npu_extra"),
            "rocm-npu-extra",
            version,
            root / "dist",
            summary="extra",
            tag="py3-none-win_amd64",
            license_expression="MIT",
        )

        with pytest.raises(
            RuntimeError, match="release wheel directory contains unexpected wheels"
        ):
            prepare_rocm_npu_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"rocm-npu {version}",
                tag_name=f"rocm-npu-v{version}",
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
            prepare_rocm_npu_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"rocm-npu {version}",
                tag_name=f"rocm-npu-v{version}",
                source_ref="main",
                source_sha="abc123",
                expected_wheel_tags=self._expected_tags(),
            )

    def test_prepare_release_payload_requires_every_platform(self, tmp_path):
        root = tmp_path
        version = "0.1.0.dev1"
        self._build_release_wheels(root, version)

        with pytest.raises(RuntimeError, match="release wheel set is incomplete"):
            prepare_rocm_npu_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"rocm-npu {version}",
                tag_name=f"rocm-npu-v{version}",
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
                self._package_tree(root, "rocm_npu_compiler_meta"),
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

    def _package_tree(self, root: Path, package_name: str) -> Path:
        tree = root / f"tree-{package_name}"
        package = tree / package_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
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
