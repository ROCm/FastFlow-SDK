# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    from cyclonedx.schema import SchemaVersion
    from cyclonedx.validation.json import JsonStrictValidator
    from packageurl import PackageURL
except ImportError:
    SchemaVersion = None
    JsonStrictValidator = None
    PackageURL = None

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


class RocmNpuWheelCommonTest(unittest.TestCase):
    def test_package_manifest_records_payload_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            payload = package / "bin" / "runtime.bin"
            payload.parent.mkdir()
            payload.write_bytes(b"runtime")

            manifest = rocm_npu_wheel_common._write_package_manifest(
                package, {"schema_version": 1}
            )

            self.assertEqual(
                manifest["files"],
                [
                    {
                        "path": "bin/runtime.bin",
                        "size": 7,
                        "sha256": sha256_file(payload),
                    }
                ],
            )
            self.assertTrue((package / "manifest.json").is_file())

    def test_build_and_find_wheel_for_explicit_platform_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            self.assertEqual(
                wheel.name,
                "example_package-1.2.3-py3-none-test_platform.whl",
            )
            self.assertEqual(
                find_distribution_wheels(root / "dist", ("example-package",)),
                {"example-package": wheel},
            )
            verify_wheel_checksums({"example-package": wheel})

    @unittest.skipUnless(
        JsonStrictValidator is not None,
        "CycloneDX validation dependencies are not installed",
    )
    def test_wheel_metadata_record_modes_and_console_scripts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
                json.dumps({
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
                        "resolved_packages": [
                            {"package": "zlib", "version": "1.3.1"}
                        ],
                        "runtime_file_packages": {"tool": "zlib"},
                    },
                    "files": [{
                        "path": "bin/tool",
                        "size": executable.stat().st_size,
                        "sha256": executable_digest,
                    }],
                }),
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
                self.assertIn("example_package/__init__.py", names)
                self.assertIn(f"{dist_info}/METADATA", names)
                self.assertIn(f"{dist_info}/WHEEL", names)
                self.assertIn(f"{dist_info}/RECORD", names)
                self.assertIn(f"{dist_info}/licenses/LICENSE", names)
                self.assertIn(f"{dist_info}/sboms/cyclonedx.json", names)
                self.assertEqual(
                    archive.read(f"{dist_info}/entry_points.txt").decode("utf-8"),
                    "[console_scripts]\nexample = example_package:main\n",
                )
                metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
                self.assertIn("Metadata-Version: 2.4\n", metadata)
                self.assertIn("Name: example-package\n", metadata)
                self.assertIn("License-Expression: MIT\n", metadata)
                self.assertIn("License-File: LICENSE\n", metadata)
                self.assertIn("Description-Content-Type: text/markdown\n", metadata)
                self.assertTrue(metadata.endswith("\n\nexample\n"))
                sbom = json.loads(
                    archive.read(f"{dist_info}/sboms/cyclonedx.json")
                )
                self.assertEqual(sbom["bomFormat"], "CycloneDX")
                self.assertEqual(sbom["specVersion"], "1.6")
                validation_errors = JsonStrictValidator(
                    SchemaVersion.V1_6
                ).validate_str(json.dumps(sbom))
                self.assertFalse(validation_errors, validation_errors)
                components = {
                    component["name"]: component
                    for component in sbom["components"]
                }
                self.assertIn("rocm-npu-runtime", components)
                self.assertEqual(
                    components["rocm-npu-runtime"]["purl"],
                    "pkg:pypi/rocm-npu-runtime@1.2.3",
                )
                self.assertEqual(components["zlib"]["version"], "1.3.1")
                self.assertEqual(
                    components["bin/tool"]["hashes"],
                    [{"alg": "SHA-256", "content": executable_digest}],
                )
                for component in sbom["components"]:
                    if component.get("purl"):
                        self.assertEqual(
                            PackageURL.from_string(component["purl"]).to_string(),
                            component["purl"],
                        )
                root_component = sbom["metadata"]["component"]
                self.assertEqual(
                    root_component["licenses"], [{"expression": "MIT"}]
                )
                self.assertEqual(
                    root_component["pedigree"]["ancestors"][0]["version"],
                    "abc123",
                )
                self.assertEqual(
                    sbom["metadata"]["tools"],
                    [{"name": "python", "version": "3.12"}],
                )
                wheel_metadata = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
                self.assertIn("Tag: py3-none-test_platform\n", wheel_metadata)
                record = archive.read(f"{dist_info}/RECORD").decode("utf-8")
                self.assertIn(f"{dist_info}/entry_points.txt,sha256=", record)
                self.assertIn(f"{dist_info}/RECORD,,", record)
                record_hashes = {
                    row[0]: row[1] for row in csv.reader(io.StringIO(record))
                }
                manifest = json.loads(
                    archive.read("example_package/manifest.json")
                )
                manifest_files = {
                    item["path"]: item["sha256"] for item in manifest["files"]
                }
                sbom_files = {
                    component["name"]: component["hashes"][0]["content"]
                    for component in sbom["components"]
                    if component["type"] == "file"
                }
                self.assertEqual(sbom_files, manifest_files)
                for path, digest in sbom_files.items():
                    wheel_path = f"example_package/{path}"
                    payload = archive.read(wheel_path)
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
                    self.assertEqual(
                        record_hashes[wheel_path],
                        rocm_npu_wheel_common._wheel_hash(payload),
                    )
                package_mode = (
                    archive.getinfo("example_package/__init__.py").external_attr >> 16
                )
                executable_mode = (
                    archive.getinfo("example_package/bin/tool").external_attr >> 16
                )
                self.assertEqual(package_mode & 0o777, 0o644)
                self.assertEqual(executable_mode & 0o777, 0o755)

    def test_wheel_build_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_remove_distribution_wheels_removes_only_requested_products(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stale = root / "example_package-1-py3-none-any.whl"
            stale.write_bytes(b"wheel")
            stale.with_name(stale.name + ".sha256").write_text(
                "checksum\n", encoding="ascii"
            )
            unrelated = root / "other_package-1-py3-none-any.whl"
            unrelated.write_bytes(b"wheel")
            remove_distribution_wheels(root, ("example-package",))
            self.assertFalse(stale.exists())
            self.assertFalse(stale.with_name(stale.name + ".sha256").exists())
            self.assertTrue(unrelated.exists())

    def test_path_scan_detects_slash_case_and_utf16_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            self.assertEqual(len(references), 1)
            self.assertEqual(references[0].member, "example_package/payload.bin")

    def test_path_scan_detects_reference_across_stream_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            self.assertEqual(len(references), 1)

    def test_version_validation_rejects_non_normalized_input(self):
        self.assertEqual(validate_version("1.2.3.dev4+local.1"), "1.2.3.dev4+local.1")
        with self.assertRaisesRegex(RuntimeError, "invalid PEP 440"):
            validate_version("1.2.3 DEV")


if __name__ == "__main__":
    unittest.main()
