# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from build_tools.rocm_npu_wheel_common import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    build_wheel,
)
from build_tools.rocm_npu_wheel_release import (
    prepare_runtime_fastflowlm_release_payload,
    resolve_release_inputs,
)


class RocmNpuWheelReleaseTest(unittest.TestCase):
    def test_resolve_release_inputs_uses_explicit_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = self._write_policy(Path(temporary), "1.2.3")
            release = resolve_release_inputs(
                "release-branch",
                "1.2.3rc1",
                7,
                publish_target="github-prerelease",
                policy_path=policy,
            )

        self.assertEqual(release["ref"], "release-branch")
        self.assertEqual(release["package_version"], "1.2.3rc1")
        self.assertEqual(release["is_prerelease"], "true")
        self.assertEqual(release["tag_name"], "rocm-npu-v1.2.3rc1")
        self.assertEqual(release["release_name"], "rocm-npu 1.2.3rc1")

    def test_resolve_release_inputs_allows_stable_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = self._write_policy(Path(temporary), "1.2.3")
            release = resolve_release_inputs(
                "main",
                "1.2.3",
                7,
                publish_target="none",
                policy_path=policy,
            )

        self.assertEqual(release["package_version"], "1.2.3")
        self.assertEqual(release["is_prerelease"], "false")

    def test_resolve_release_inputs_rejects_stable_prerelease_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = self._write_policy(Path(temporary), "1.2.3")
            with self.assertRaisesRegex(
                RuntimeError, "cannot be published as a GitHub prerelease"
            ):
                resolve_release_inputs(
                    "main",
                    "1.2.3",
                    7,
                    publish_target="github-prerelease",
                    policy_path=policy,
                )

    def test_resolve_release_inputs_rejects_version_from_another_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = self._write_policy(Path(temporary), "1.2.3")
            with self.assertRaisesRegex(
                RuntimeError, "does not belong to the configured 1.2.3 release train"
            ):
                resolve_release_inputs(
                    "release-branch",
                    "9.0.0rc1",
                    7,
                    policy_path=policy,
                )

    def test_resolve_release_inputs_rejects_non_train_release_forms(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = self._write_policy(Path(temporary), "1.2.3")
            for version in ("1.2.3a1", "1.2.3.post1", "1.2.3+local"):
                with (
                    self.subTest(version=version),
                    self.assertRaisesRegex(RuntimeError, "does not belong"),
                ):
                    resolve_release_inputs("main", version, 1, policy_path=policy)

    def test_resolve_release_inputs_generates_utc_development_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = self._write_policy(Path(temporary), "2.3.0")
            release = resolve_release_inputs(
                "main",
                "",
                7,
                now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
                policy_path=policy,
            )

        self.assertEqual(release["package_version"], "2.3.0.dev202607240007")

    def test_prepare_release_payload_validates_and_indexes_tested_wheels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            )

            release = prepare_runtime_fastflowlm_release_payload(
                root / "dist",
                root / "release",
                version=version,
                release_name=f"rocm-npu {version}",
                tag_name=f"rocm-npu-v{version}",
                source_ref="main",
                source_sha="abc123",
                expected_wheel_tags=(
                    "py3-none-manylinux_2_28_x86_64",
                    "py3-none-linux_x86_64",
                    "py3-none-win_amd64",
                ),
                github_run_id="42",
                github_run_attempt="1",
            )

            manifest = json.loads(
                (release / "rocm-npu-release-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["product"], "rocm-npu")
            self.assertEqual(manifest["version"], version)
            self.assertEqual(manifest["source_ref"], "main")
            self.assertEqual(manifest["source_sha"], "abc123")
            self.assertEqual(
                [artifact["distribution"] for artifact in manifest["artifacts"]],
                [
                    FASTFLOWLM_DISTRIBUTION,
                    FASTFLOWLM_DISTRIBUTION,
                    RUNTIME_DISTRIBUTION,
                    RUNTIME_DISTRIBUTION,
                ],
            )
            artifact_tags = set()
            for artifact in manifest["artifacts"]:
                filename = artifact["filename"]
                self.assertNotIn("staging", filename)
                artifact_tags.update(artifact["wheel_tags"])
                self.assertTrue((release / filename).is_file())
                self.assertTrue((release / f"{filename}.sha256").is_file())
            self.assertEqual(
                artifact_tags,
                {
                    "py3-none-manylinux_2_28_x86_64",
                    "py3-none-linux_x86_64",
                    "py3-none-win_amd64",
                },
            )

            simple = release / "simple"
            self.assertTrue((simple / RUNTIME_DISTRIBUTION / "index.html").is_file())
            self.assertTrue((simple / FASTFLOWLM_DISTRIBUTION / "index.html").is_file())
            for distribution in (RUNTIME_DISTRIBUTION, FASTFLOWLM_DISTRIBUTION):
                index = (simple / distribution / "index.html").read_text(
                    encoding="utf-8"
                )
                if distribution == RUNTIME_DISTRIBUTION:
                    self.assertIn("manylinux_2_28_x86_64", index)
                else:
                    self.assertIn("py3-none-linux_x86_64", index)
                self.assertIn("win_amd64", index)
                self.assertEqual(index.count("<a href="), 2)
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
            self.assertEqual(checksum_names, published_names)
            self.assertIn("rocm-npu-release-manifest.json", checksum_names)
            self.assertFalse(any(name.startswith("simple/") for name in checksum_names))

    def test_prepare_release_payload_rejects_version_outside_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "does not belong"):
                prepare_runtime_fastflowlm_release_payload(
                    root / "dist",
                    root / "release",
                    version="9.0.0rc1",
                    release_name="rocm-npu 9.0.0rc1",
                    tag_name="rocm-npu-v9.0.0rc1",
                    source_ref="main",
                    source_sha="abc123",
                )

    def test_prepare_release_payload_requires_matching_runtime_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = "0.1.0.dev1"
            self._build_release_wheels(root, version, runtime_requirement="0.1.0.dev0")

            with self.assertRaisesRegex(
                RuntimeError,
                f"{FASTFLOWLM_DISTRIBUTION} must require "
                f"{RUNTIME_DISTRIBUTION} =={version}",
            ):
                prepare_runtime_fastflowlm_release_payload(
                    root / "dist",
                    root / "release",
                    version=version,
                    release_name=f"rocm-npu {version}",
                    tag_name=f"rocm-npu-v{version}",
                    source_ref="main",
                    source_sha="abc123",
                )

    def test_prepare_release_payload_rejects_extra_wheels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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

            with self.assertRaisesRegex(
                RuntimeError, "release wheel directory contains unexpected wheels"
            ):
                prepare_runtime_fastflowlm_release_payload(
                    root / "dist",
                    root / "release",
                    version=version,
                    release_name=f"rocm-npu {version}",
                    tag_name=f"rocm-npu-v{version}",
                    source_ref="main",
                    source_sha="abc123",
                )

    def test_prepare_release_payload_rejects_wrong_platform_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = "0.1.0.dev1"
            self._build_release_wheels(
                root, version, wheel_tag="py3-none-manylinux_2_28_x86_64"
            )

            with self.assertRaisesRegex(
                RuntimeError, "do not include exactly one expected tag"
            ):
                prepare_runtime_fastflowlm_release_payload(
                    root / "dist",
                    root / "release",
                    version=version,
                    release_name=f"rocm-npu {version}",
                    tag_name=f"rocm-npu-v{version}",
                    source_ref="main",
                    source_sha="abc123",
                    expected_wheel_tags=("py3-none-win_amd64",),
                )

    def test_prepare_release_payload_requires_every_platform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = "0.1.0.dev1"
            self._build_release_wheels(root, version)

            with self.assertRaisesRegex(
                RuntimeError, "release wheel set is incomplete"
            ):
                prepare_runtime_fastflowlm_release_payload(
                    root / "dist",
                    root / "release",
                    version=version,
                    release_name=f"rocm-npu {version}",
                    tag_name=f"rocm-npu-v{version}",
                    source_ref="main",
                    source_sha="abc123",
                    expected_wheel_tags=(
                        "py3-none-manylinux_2_28_x86_64",
                        "py3-none-win_amd64",
                    ),
                )

    def _build_release_wheels(
        self,
        root: Path,
        version: str,
        *,
        runtime_requirement: str | None = None,
        wheel_tag: str = "py3-none-win_amd64",
        fastflowlm_wheel_tag: str | None = None,
        output_directory: Path | None = None,
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

    def _package_tree(self, root: Path, package_name: str) -> Path:
        tree = root / f"tree-{package_name}"
        package = tree / package_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        return tree

    def _write_policy(self, root: Path, base_version: str) -> Path:
        policy = root / "release.toml"
        policy.write_text(
            f'schema_version = 1\n[wheel_release]\nbase_version = "{base_version}"\n',
            encoding="utf-8",
        )
        return policy


if __name__ == "__main__":
    unittest.main()
