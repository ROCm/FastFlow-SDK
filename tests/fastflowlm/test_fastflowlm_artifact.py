# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

from build_tools.fastflowlm_artifact import (
    _linux_dependencies_and_licenses,
    _is_linux_system_runtime,
    _pe_imports,
    create_archive,
    extract_archive,
    git_source_state,
    sha256_file,
    validate_source_policy,
    verify_bundle,
)


class FastFlowLMArtifactTest(unittest.TestCase):
    def make_bundle(self, root):
        bundle = root / "fastflowlm-hrx-test-x86_64"
        payload = bundle / "bin" / ("flm.exe" if os.name == "nt" else "flm")
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"runtime")
        manifest = {
            "schema_version": 1,
            "files": [{"path": payload.relative_to(bundle).as_posix(),
                       "size": payload.stat().st_size,
                       "sha256": sha256_file(payload)}],
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return bundle

    def test_verify_bundle_rejects_modified_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            payload = bundle / "bin" / "flm"
            payload.parent.mkdir()
            payload.write_bytes(b"good")
            manifest = {
                "schema_version": 1,
                "files": [{
                    "path": "bin/flm",
                    "size": 4,
                    "sha256": "770e607624d689265ca6c44884d0807d9b054d23c473c106c72be9de08b7376c",
                }],
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            verify_bundle(bundle)
            payload.write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                verify_bundle(bundle)

    def test_git_source_state_records_modified_and_untracked_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"],
                           cwd=repository, check=True)
            subprocess.run(["git", "config", "user.name", "Artifact Test"],
                           cwd=repository, check=True)
            tracked = repository / "tracked.txt"
            tracked.write_text("original\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "--quiet", "-m", "initial"],
                           cwd=repository, check=True)
            self.assertFalse(git_source_state(repository)["dirty"])

            tracked.write_text("modified\n", encoding="utf-8")
            (repository / "new.txt").write_text("new\n", encoding="utf-8")
            state = git_source_state(repository)
            self.assertTrue(state["dirty"])
            self.assertEqual([entry["path"] for entry in state["changed_files"]],
                             ["new.txt", "tracked.txt"])

    def test_linux_system_runtime_classification(self):
        for name in ("libc.so.6", "libm.so.6", "libpthread.so.0",
                     "ld-linux-x86-64.so.2", "linux-vdso.so.1"):
            self.assertTrue(_is_linux_system_runtime(name), name)
        self.assertFalse(_is_linux_system_runtime("libcurl.so.4"))

    def test_linux_system_dependency_uses_source_rpm_license_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "lib").mkdir()
            (bundle / "lib" / "libstdc++.so.6.0.25").write_bytes(b"")
            license_source = bundle / "rpm-root" / "usr" / "share" / "doc" / "gcc" / "COPYING.RUNTIME"
            license_source.parent.mkdir(parents=True)
            license_source.write_text("runtime exception\n", encoding="utf-8")

            def fake_which(name):
                return {
                    "ldconfig": "/sbin/ldconfig",
                    "patchelf": "/bin/patchelf",
                    "rpm": "/bin/rpm",
                }.get(name)

            def fake_run(command, **kwargs):
                normalized = [Path(command[0]).name, *command[1:]]
                if normalized == ["ldconfig", "-p"]:
                    return subprocess.CompletedProcess(
                        command, 0,
                        stdout="  libstdc++.so.6 (libc6,x86-64) => /lib64/libstdc++.so.6\n",
                    )
                if normalized[:2] == ["patchelf", "--print-soname"]:
                    return subprocess.CompletedProcess(command, 0, stdout="libstdc++.so.6\n")
                if normalized[:4] == ["rpm", "-qf", "--qf", "%{NAME}\n"]:
                    return subprocess.CompletedProcess(command, 0, stdout="libstdc++\n")
                if normalized == ["rpm", "-q", "libstdc++"]:
                    return subprocess.CompletedProcess(command, 0, stdout="libstdc++-8.5.0\n")
                if normalized == ["rpm", "-ql", "libstdc++"]:
                    return subprocess.CompletedProcess(command, 0, stdout="/lib64/libstdc++.so.6\n")
                if normalized == ["rpm", "-q", "--qf", "%{SOURCERPM}", "libstdc++"]:
                    return subprocess.CompletedProcess(command, 0, stdout="gcc-8.5.0.src.rpm")
                if normalized == ["rpm", "-qa", "--qf", "%{NAME}\t%{SOURCERPM}\n"]:
                    return subprocess.CompletedProcess(
                        command, 0,
                        stdout="libstdc++\tgcc-8.5.0.src.rpm\ngcc\tgcc-8.5.0.src.rpm\n",
                    )
                if normalized == ["rpm", "-ql", "gcc"]:
                    return subprocess.CompletedProcess(command, 0, stdout=f"{license_source}\n")
                raise AssertionError(f"unexpected command: {command!r}")

            with mock.patch("build_tools.fastflowlm_artifact.shutil.which", fake_which), \
                 mock.patch("build_tools.fastflowlm_artifact.subprocess.run", fake_run):
                dependencies = _linux_dependencies_and_licenses(bundle, set())

            copied = bundle / "licenses" / "system" / "libstdc++" / "gcc" / "COPYING.RUNTIME"
            self.assertEqual(copied.read_text(encoding="utf-8"), "runtime exception\n")
            self.assertEqual(dependencies["license_source_packages"], {"libstdc++": ["gcc"]})
            self.assertEqual(dependencies["generated_license_records"], {})
            self.assertEqual(dependencies["packages_without_license_files"], [])

    def test_linux_system_dependency_skips_internal_runtime_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "lib").mkdir()
            (bundle / "lib" / "libq4_npu_eXpress.so").write_bytes(b"")
            with mock.patch("build_tools.fastflowlm_artifact.shutil.which",
                            lambda name: None):
                dependencies = _linux_dependencies_and_licenses(
                    bundle, set(), {"libq4_npu_eXpress.so"})
            self.assertEqual(dependencies["packages"], [])
            self.assertEqual(dependencies["unmapped_libraries"], [])

    @unittest.skipUnless(os.name == "nt", "PE fixture is built on Windows")
    def test_parse_pe_imports(self):
        executable = Path(os.environ["SystemRoot"]) / "System32" / "where.exe"
        imports = _pe_imports(executable)
        self.assertIn("kernel32.dll", imports)

    def test_source_policy_enforces_component_pins_and_clean_staging(self):
        root = Path(__file__).resolve().parents[2]
        versions = tomllib.loads(
            (root / "manifests" / "versions.toml").read_text(encoding="utf-8"))
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip()
        manifest = {"sources": {
            "rocm-npu-staging": {"revision": revision, "dirty": False},
            "hrx": {"revision": versions["hrx"]["commit"]},
            "FastFlowLM": {"revision": versions["fastflowlm"]["commit"]},
        }}
        validate_source_policy(manifest, root)
        manifest["sources"]["hrx"]["revision"] = "0" * 40
        with self.assertRaisesRegex(RuntimeError, "hrx revision"):
            validate_source_policy(manifest, root)
        manifest["sources"]["hrx"]["revision"] = versions["hrx"]["commit"]
        manifest["sources"]["rocm-npu-staging"]["dirty"] = True
        with self.assertRaisesRegex(RuntimeError, "dirty staging"):
            validate_source_policy(manifest, root)

    def test_verify_bundle_rejects_unexpected_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "manifest.json").write_text(
                json.dumps({"schema_version": 1, "files": []}), encoding="utf-8")
            (bundle / "extra").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
                verify_bundle(bundle)

    def test_verify_bundle_rejects_duplicate_manifest_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            record = {"path": "bin/flm", "size": 0, "sha256": "0" * 64}
            (bundle / "manifest.json").write_text(json.dumps({
                "schema_version": 1, "files": [record, record],
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate path"):
                verify_bundle(bundle)

    def test_verify_bundle_rejects_unsafe_manifest_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "files": [{"path": "../escape", "size": 0,
                           "sha256": "0" * 64}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unsafe path"):
                verify_bundle(bundle)

    def test_archive_round_trip(self):
        if os.name != "nt" and not shutil.which("zstd"):
            self.skipTest("zstd is required for the Linux archive format")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self.make_bundle(root / "stage")
            archive = create_archive(bundle, root / "artifacts")
            self.assertTrue(archive.with_name(archive.name + ".sha256").is_file())
            extracted = extract_archive(archive, root / "extracted", allowed_root=root)
            executable = "flm.exe" if os.name == "nt" else "flm"
            self.assertEqual((extracted / "bin" / executable).read_bytes(), b"runtime")

    def test_archive_is_reproducible(self):
        if os.name != "nt" and not shutil.which("zstd"):
            self.skipTest("zstd is required for the Linux archive format")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self.make_bundle(root / "stage")
            first = create_archive(bundle, root / "first")
            second = create_archive(bundle, root / "second")
            self.assertEqual(sha256_file(first), sha256_file(second))

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape", b"bad")
            with self.assertRaisesRegex(RuntimeError, "escapes extraction root"):
                extract_archive(archive, root / "extracted", allowed_root=root)

    def test_failed_extraction_preserves_previous_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "extracted"
            destination.mkdir()
            sentinel = destination / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            archive = root / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape", b"bad")
            with self.assertRaisesRegex(RuntimeError, "escapes extraction root"):
                extract_archive(archive, destination, allowed_root=root)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_zip_duplicate_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr("bundle/file", b"first")
                    output.writestr("bundle/file", b"second")
            with self.assertRaisesRegex(RuntimeError, "duplicate paths"):
                extract_archive(archive, root / "extracted", allowed_root=root)

    def test_zip_normalized_duplicate_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "duplicate.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("bundle/a/../file", b"first")
                output.writestr("bundle/file", b"second")
            with self.assertRaisesRegex(RuntimeError, "duplicate normalized path"):
                extract_archive(archive, root / "extracted", allowed_root=root)

    def test_extract_refuses_to_remove_allowed_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "empty.zip"
            with zipfile.ZipFile(archive, "w"):
                pass
            with self.assertRaisesRegex(RuntimeError, "refusing to remove"):
                extract_archive(archive, root, allowed_root=root)

    @unittest.skipUnless(os.name != "nt" and shutil.which("zstd"),
                         "tar.zst security test requires zstd")
    def test_tar_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tar_path = root / "bad.tar"
            with tarfile.open(tar_path, "w") as output:
                info = tarfile.TarInfo("bundle/link")
                info.type = tarfile.SYMTYPE
                info.linkname = "../outside"
                output.addfile(info)
            archive = root / "bad.tar.zst"
            subprocess.run(["zstd", "-q", "-f", str(tar_path), "-o", str(archive)],
                           check=True)
            with self.assertRaisesRegex(RuntimeError, "special files"):
                extract_archive(archive, root / "extracted", allowed_root=root)


if __name__ == "__main__":
    unittest.main()
