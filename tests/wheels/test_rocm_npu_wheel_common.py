# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import csv
import hashlib
import io
import json
import zipfile

try:
    from cyclonedx.schema import SchemaVersion
    from cyclonedx.validation.json import JsonStrictValidator
    from packageurl import PackageURL
except ImportError:
    SchemaVersion = None
    JsonStrictValidator = None
    PackageURL = None

import pytest

from build_tools import rocm_npu_wheel_common
from build_tools.rocm_npu_wheel_common import (
    build_wheel,
    find_distribution_wheels,
    find_wheel_path_references,
    remove_distribution_wheels,
    sha256_file,
    validate_version,
    verify_wheel_checksums,
)


pytestmark = [pytest.mark.compiler_stack, pytest.mark.fastflowlm]


class TestRocmNpuWheelCommon:
    def test_package_manifest_records_payload_checksums(self, tmp_path):
        package = tmp_path
        payload = package / "bin" / "runtime.bin"
        payload.parent.mkdir()
        payload.write_bytes(b"runtime")

        manifest = rocm_npu_wheel_common._write_package_manifest(
            package, {"schema_version": 1}
        )

        assert manifest["files"] == [
            {"path": "bin/runtime.bin", "size": 7, "sha256": sha256_file(payload)}
        ]
        assert (package / "manifest.json").is_file()

    def test_build_and_find_wheel_for_explicit_platform_tag(self, tmp_path):
        root = tmp_path
        tree = root / "tree"
        package = tree / "example_package"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        wheel = build_wheel(
            tree,
            "example-package",
            "1.2.3",
            root / "dist",
            summary="example",
            tag="py3-none-test_platform",
            license_expression="MIT",
        )
        assert wheel.name == "example_package-1.2.3-py3-none-test_platform.whl"
        assert find_distribution_wheels(root / "dist", ("example-package",)) == {
            "example-package": wheel
        }
        verify_wheel_checksums({"example-package": wheel})

    @pytest.mark.skipif(
        JsonStrictValidator is None,
        reason="CycloneDX validation dependencies are not installed",
    )
    def test_wheel_metadata_record_modes_and_console_scripts(self, tmp_path):
        root = tmp_path
        tree = root / "tree"
        package = tree / "example_package"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        executable = package / "bin" / "tool"
        executable.parent.mkdir()
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        (package / "licenses").mkdir()
        (package / "licenses" / "LICENSE").write_text(
            "example license\n", encoding="utf-8"
        )
        executable_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        (package / "manifest.json").write_text(
            json.dumps(
                {
                    "sources": {
                        "example": {
                            "revision": "abc123",
                            "dirty": False,
                            "tracked_diff_sha256": "0" * 64,
                        }
                    },
                    "toolchain": {"python": "3.12"},
                    "dependencies": {
                        "provider": "vcpkg",
                        "resolved_packages": [{"package": "zlib", "version": "1.3.1"}],
                        "runtime_file_packages": {"tool": "zlib"},
                    },
                    "files": [
                        {
                            "path": "bin/tool",
                            "size": executable.stat().st_size,
                            "sha256": executable_digest,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        wheel = build_wheel(
            tree,
            "example-package",
            "1.2.3",
            root / "dist",
            summary="example",
            tag="py3-none-test_platform",
            requires=("rocm-npu-runtime ==1.2.3",),
            console_scripts={"example": "example_package:main"},
            license_expression="MIT",
        )

        with zipfile.ZipFile(wheel) as archive:
            dist_info = "example_package-1.2.3.dist-info"
            names = set(archive.namelist())
            assert "example_package/__init__.py" in names
            assert f"{dist_info}/METADATA" in names
            assert f"{dist_info}/WHEEL" in names
            assert f"{dist_info}/RECORD" in names
            assert f"{dist_info}/licenses/LICENSE" in names
            assert f"{dist_info}/sboms/cyclonedx.json" in names
            assert (
                archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
                == "[console_scripts]\nexample = example_package:main\n"
            )
            metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
            assert "Metadata-Version: 2.4\n" in metadata
            assert "Name: example-package\n" in metadata
            assert "License-Expression: MIT\n" in metadata
            assert "License-File: LICENSE\n" in metadata
            assert "Description-Content-Type: text/markdown\n" in metadata
            assert metadata.endswith("\n\nexample\n")
            sbom = json.loads(archive.read(f"{dist_info}/sboms/cyclonedx.json"))
            assert sbom["bomFormat"] == "CycloneDX"
            assert sbom["specVersion"] == "1.6"
            validation_errors = JsonStrictValidator(SchemaVersion.V1_6).validate_str(
                json.dumps(sbom)
            )
            assert not validation_errors, validation_errors
            components = {
                component["name"]: component for component in sbom["components"]
            }
            assert "rocm-npu-runtime" in components
            assert (
                components["rocm-npu-runtime"]["purl"]
                == "pkg:pypi/rocm-npu-runtime@1.2.3"
            )
            assert components["zlib"]["version"] == "1.3.1"
            assert components["bin/tool"]["hashes"] == [
                {"alg": "SHA-256", "content": executable_digest}
            ]
            for component in sbom["components"]:
                if component.get("purl"):
                    assert (
                        PackageURL.from_string(component["purl"]).to_string()
                        == component["purl"]
                    )
            root_component = sbom["metadata"]["component"]
            assert root_component["licenses"] == [{"expression": "MIT"}]
            assert root_component["pedigree"]["ancestors"][0]["version"] == "abc123"
            assert sbom["metadata"]["tools"] == [{"name": "python", "version": "3.12"}]
            wheel_metadata = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
            assert "Tag: py3-none-test_platform\n" in wheel_metadata
            record = archive.read(f"{dist_info}/RECORD").decode("utf-8")
            assert f"{dist_info}/entry_points.txt,sha256=" in record
            assert f"{dist_info}/RECORD,," in record
            record_hashes = {row[0]: row[1] for row in csv.reader(io.StringIO(record))}
            manifest = json.loads(archive.read("example_package/manifest.json"))
            manifest_files = {
                item["path"]: item["sha256"] for item in manifest["files"]
            }
            sbom_files = {
                component["name"]: component["hashes"][0]["content"]
                for component in sbom["components"]
                if component["type"] == "file"
            }
            assert sbom_files == manifest_files
            for path, digest in sbom_files.items():
                wheel_path = f"example_package/{path}"
                payload = archive.read(wheel_path)
                assert hashlib.sha256(payload).hexdigest() == digest
                assert record_hashes[wheel_path] == rocm_npu_wheel_common._wheel_hash(
                    payload
                )
            package_mode = (
                archive.getinfo("example_package/__init__.py").external_attr >> 16
            )
            executable_mode = (
                archive.getinfo("example_package/bin/tool").external_attr >> 16
            )
            assert package_mode & 511 == 420
            assert executable_mode & 511 == 493

    def test_wheel_build_is_deterministic(self, tmp_path):
        root = tmp_path
        tree = root / "tree"
        package = tree / "example_package"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        arguments = (
            tree,
            "example-package",
            "1.2.3",
        )
        first = build_wheel(
            *arguments,
            root / "dist-a",
            summary="example",
            tag="py3-none-test_platform",
            license_expression="MIT",
            compression_level=6,
        )
        second = build_wheel(
            *arguments,
            root / "dist-b",
            summary="example",
            tag="py3-none-test_platform",
            license_expression="MIT",
            compression_level=6,
        )
        assert first.read_bytes() == second.read_bytes()

    def test_remove_distribution_wheels_removes_only_requested_products(self, tmp_path):
        root = tmp_path
        stale = root / "example_package-1-py3-none-any.whl"
        stale.write_bytes(b"wheel")
        stale.with_name(stale.name + ".sha256").write_text(
            "checksum\n", encoding="ascii"
        )
        unrelated = root / "other_package-1-py3-none-any.whl"
        unrelated.write_bytes(b"wheel")
        remove_distribution_wheels(root, ("example-package",))
        assert not stale.exists()
        assert not stale.with_name(stale.name + ".sha256").exists()
        assert unrelated.exists()

    def test_path_scan_detects_slash_case_and_utf16_variants(self, tmp_path):
        root = tmp_path
        tree = root / "tree"
        package = tree / "example_package"
        package.mkdir(parents=True)
        forbidden = root / "Checkout"
        spelling = str(forbidden.resolve()).replace("\\", "/").upper()
        (package / "payload.bin").write_bytes(spelling.encode("utf-16le"))
        wheel = build_wheel(
            tree,
            "example-package",
            "1",
            root / "dist",
            summary="example",
            tag="py3-none-test_platform",
            license_expression="MIT",
        )
        references = find_wheel_path_references(
            {"example-package": wheel}, (forbidden,)
        )
        assert len(references) == 1
        assert references[0].member == "example_package/payload.bin"

    def test_path_scan_detects_reference_across_stream_chunk_boundary(self, tmp_path):
        root = tmp_path
        tree = root / "tree"
        package = tree / "example_package"
        package.mkdir(parents=True)
        forbidden = root / "checkout"
        needle = str(forbidden.resolve()).encode("utf-8")
        (package / "payload.bin").write_bytes(
            b"x" * (1024 * 1024 - len(needle) // 2) + needle
        )
        wheel = build_wheel(
            tree,
            "example-package",
            "1",
            root / "dist",
            summary="example",
            tag="py3-none-test_platform",
            license_expression="MIT",
        )
        references = find_wheel_path_references(
            {"example-package": wheel}, (forbidden,)
        )
        assert len(references) == 1

    def test_version_validation_rejects_non_normalized_input(self):
        assert validate_version("1.2.3.dev4+local.1") == "1.2.3.dev4+local.1"
        with pytest.raises(RuntimeError, match="invalid PEP 440"):
            validate_version("1.2.3 DEV")
