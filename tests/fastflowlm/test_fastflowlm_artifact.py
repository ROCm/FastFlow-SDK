# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import tomllib
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from build_tools.artifact import (
    directory_state,
    git_source_state,
    sha256_file,
)
from build_tools.fastflowlm_artifact import (
    _copy_windows_redistributable_closure,
    _is_linux_system_runtime,
    _linux_dependencies_and_licenses,
    _linux_package_owner_candidates,
    _strip_windows_system_dlls,
    _system_license_destination,
    artifact_name,
    create_archive,
    extract_archive,
    validate_source_policy,
    verify_bundle,
    verify_cli_fresh_install,
    verify_fastflowlm_fresh_installs,
)
from build_tools.native_dependencies import pe_imports

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


def _init_git_repository(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Artifact Test"],
        cwd=root,
        check=True,
    )
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "initial"],
        cwd=root,
        check=True,
    )
    return root


class TestFastFlowLMArtifact:
    def test_artifact_name_uses_rocm_npu_product_stem(self):
        assert artifact_name("Linux", "x86_64") == "rocm-npu-linux-x86_64"
        assert artifact_name("Windows", "AMD64") == "rocm-npu-windows-x86_64"

    def make_bundle(self, root):
        bundle = root / "rocm-npu-test-x86_64"
        payload = bundle / "bin" / ("flm.exe" if os.name == "nt" else "flm")
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"runtime")
        if os.name != "nt":
            payload.chmod(0o755)
        manifest = {
            "schema_version": 1,
            "files": [
                {
                    "path": payload.relative_to(bundle).as_posix(),
                    "size": payload.stat().st_size,
                    "sha256": sha256_file(payload),
                }
            ],
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return bundle

    def test_verify_bundle_rejects_modified_file(self, tmp_path):
        bundle = tmp_path
        payload = bundle / "bin" / "flm"
        payload.parent.mkdir()
        payload.write_bytes(b"good")
        manifest = {
            "schema_version": 1,
            "files": [
                {
                    "path": "bin/flm",
                    "size": 4,
                    "sha256": "770e607624d689265ca6c44884d0807d9b054d23c473c106c72be9de08b7376c",
                }
            ],
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        verify_bundle(bundle)
        payload.write_bytes(b"bad")
        with pytest.raises(RuntimeError, match="verification failed"):
            verify_bundle(bundle)

    def test_git_source_state_records_modified_and_untracked_files(self, tmp_path):
        repository = _init_git_repository(tmp_path, {"tracked.txt": "original\n"})
        tracked = repository / "tracked.txt"
        assert not git_source_state(repository)["dirty"]

        tracked.write_text("modified\n", encoding="utf-8")
        (repository / "new.txt").write_text("new\n", encoding="utf-8")
        state = git_source_state(repository)
        assert state["dirty"]
        assert [entry["path"] for entry in state["changed_files"]] == [
            "new.txt",
            "tracked.txt",
        ]

    def test_git_source_state_records_staged_only_changes_as_dirty(self, tmp_path):
        repository = _init_git_repository(tmp_path, {"tracked.txt": "original\n"})
        tracked = repository / "tracked.txt"
        tracked.write_text("staged\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
        state = git_source_state(repository)
        assert state["dirty"]
        assert [entry["path"] for entry in state["changed_files"]] == ["tracked.txt"]

    def test_fresh_install_verification_checks_sources_installs_and_dependency(
        self, tmp_path
    ):
        root = tmp_path
        marker_name = ".fresh.json"
        hrx_source = _init_git_repository(
            root / "hrx-source", {"source.txt": "source\n"}
        )
        fastflowlm_source = _init_git_repository(
            root / "fastflowlm-source", {"source.txt": "source\n"}
        )
        hrx_install = root / "hrx-install"
        fastflowlm_install = root / "fastflowlm-install"
        flm_name = "flm.exe" if os.name == "nt" else "flm"
        (hrx_install / "bin").mkdir(parents=True)
        (hrx_install / "bin" / "hrx.dll").write_bytes(b"hrx")
        (fastflowlm_install / "bin").mkdir(parents=True)
        (fastflowlm_install / "bin" / flm_name).write_bytes(b"flm")

        hrx_marker = {
            "source": git_source_state(hrx_source),
            "install": directory_state(hrx_install, exclude_names=(marker_name,)),
        }
        (hrx_install / marker_name).write_text(json.dumps(hrx_marker), encoding="utf-8")
        fastflowlm_marker = {
            "source": git_source_state(fastflowlm_source),
            "hrx": hrx_marker,
            "install": directory_state(
                fastflowlm_install, exclude_names=(marker_name,)
            ),
        }
        (fastflowlm_install / marker_name).write_text(
            json.dumps(fastflowlm_marker), encoding="utf-8"
        )

        markers = verify_fastflowlm_fresh_installs(
            hrx_install=hrx_install,
            hrx_source=hrx_source,
            fastflowlm_install=fastflowlm_install,
            fastflowlm_source=fastflowlm_source,
            marker_name=marker_name,
        )
        assert markers["HRX"] == hrx_marker
        (fastflowlm_install / "bin" / flm_name).write_bytes(b"changed")
        with pytest.raises(RuntimeError, match="install changed"):
            verify_fastflowlm_fresh_installs(
                hrx_install=hrx_install,
                hrx_source=hrx_source,
                fastflowlm_install=fastflowlm_install,
                fastflowlm_source=fastflowlm_source,
                marker_name=marker_name,
            )

    def test_cli_fresh_install_checks_source_version_and_install(self, tmp_path):
        root = _init_git_repository(tmp_path / "source", {"source.txt": "source\n"})
        hrx_source = _init_git_repository(
            tmp_path / "hrx-source", {"header.h": "header\n"}
        )
        hrx_file = hrx_source / "header.h"
        hrx_install = tmp_path / "hrx-install"
        hrx_install.mkdir()
        hrx_binary = hrx_install / "hrx.dll"
        hrx_binary.write_bytes(b"hrx")
        install = tmp_path / "install"
        cli_name = "rocm-npu.exe" if os.name == "nt" else "rocm-npu"
        (install / "bin").mkdir(parents=True)
        (install / "bin" / cli_name).write_bytes(b"cli")
        marker_name = ".fresh.json"
        marker = {
            "source": git_source_state(root),
            "hrx_headers": git_source_state(hrx_source),
            "version": "1.2.3",
            "install": directory_state(install, exclude_names=(marker_name,)),
        }
        (install / marker_name).write_text(json.dumps(marker), encoding="utf-8")
        (hrx_install / marker_name).write_text(
            json.dumps(
                {
                    "source": git_source_state(hrx_source),
                    "install": directory_state(
                        hrx_install, exclude_names=(marker_name,)
                    ),
                }
            ),
            encoding="utf-8",
        )
        with mock.patch(
            "build_tools.fastflowlm_artifact.default_package_version",
            return_value="1.2.3",
        ):
            assert (
                verify_cli_fresh_install(
                    root=root,
                    cli_install=install,
                    hrx_install=hrx_install,
                    hrx_source=hrx_source,
                    marker_name=marker_name,
                )
                == marker
            )
            hrx_file.write_text("changed\n", encoding="utf-8")
            with pytest.raises(RuntimeError, match="different HRX headers"):
                verify_cli_fresh_install(
                    root=root,
                    cli_install=install,
                    hrx_install=hrx_install,
                    hrx_source=hrx_source,
                    marker_name=marker_name,
                )
            hrx_file.write_text("header\n", encoding="utf-8")
            hrx_binary.write_bytes(b"stale")
            with pytest.raises(RuntimeError, match="HRX install changed"):
                verify_cli_fresh_install(
                    root=root,
                    cli_install=install,
                    hrx_install=hrx_install,
                    hrx_source=hrx_source,
                    marker_name=marker_name,
                )
            hrx_binary.write_bytes(b"hrx")
            (install / "bin" / cli_name).write_bytes(b"stale")
            with pytest.raises(RuntimeError, match="install changed"):
                verify_cli_fresh_install(
                    root=root,
                    cli_install=install,
                    hrx_install=hrx_install,
                    hrx_source=hrx_source,
                    marker_name=marker_name,
                )

    def test_linux_system_runtime_classification(self):
        for name in (
            "libc.so.6",
            "libm.so.6",
            "libpthread.so.0",
            "ld-linux-x86-64.so.2",
            "linux-vdso.so.1",
        ):
            assert _is_linux_system_runtime(name), name
        assert not _is_linux_system_runtime("libcurl.so.4")

    def test_linux_system_license_destination_for_own_license_directory(self):
        bundle = Path("/bundle")
        source = Path("/usr/share/licenses/libstdc++/COPYING")
        assert (
            _system_license_destination(bundle, "libstdc++", source, "libstdc++")
            == bundle / "licenses" / "system" / "libstdc++" / "COPYING"
        )

    def test_linux_system_dependency_uses_source_rpm_license_files(self, tmp_path):
        bundle = tmp_path
        (bundle / "lib").mkdir()
        library = bytearray(120)
        library[:4] = b"\x7fELF"
        library[4] = 2
        library[5] = 1
        struct.pack_into("<H", library, 16, 3)
        struct.pack_into("<Q", library, 32, 64)
        struct.pack_into("<H", library, 52, 64)
        struct.pack_into("<H", library, 54, 56)
        struct.pack_into("<H", library, 56, 1)
        struct.pack_into("<I", library, 64, 2)
        (bundle / "lib" / "libstdc++.so.6.0.25").write_bytes(library)
        license_source = (
            bundle
            / "rpm-root"
            / "usr"
            / "share"
            / "licenses"
            / "gcc"
            / "COPYING.RUNTIME"
        )
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
                    command,
                    0,
                    stdout="  libstdc++.so.6 (libc6,x86-64) => /lib64/libstdc++.so.6\n",
                )
            if normalized[:2] == ["patchelf", "--print-soname"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout="libstdc++.so.6\n"
                )
            if normalized[:4] == ["rpm", "-qf", "--qf", "%{NAME}\n"]:
                return subprocess.CompletedProcess(command, 0, stdout="libstdc++\n")
            if normalized == ["rpm", "-q", "libstdc++"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout="libstdc++-8.5.0\n"
                )
            if normalized == ["rpm", "-ql", "libstdc++"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout="/lib64/libstdc++.so.6\n"
                )
            if normalized == ["rpm", "-q", "--qf", "%{SOURCERPM}", "libstdc++"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout="gcc-8.5.0.src.rpm"
                )
            if normalized == ["rpm", "-qa", "--qf", "%{NAME}\t%{SOURCERPM}\n"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="libstdc++\tgcc-8.5.0.src.rpm\ngcc\tgcc-8.5.0.src.rpm\n",
                )
            if normalized == ["rpm", "-ql", "gcc"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout=f"{license_source}\n"
                )
            raise AssertionError(f"unexpected command: {command!r}")

        with (
            mock.patch("build_tools.fastflowlm_artifact.shutil.which", fake_which),
            mock.patch("build_tools.fastflowlm_artifact.subprocess.run", fake_run),
        ):
            dependencies = _linux_dependencies_and_licenses(bundle, set())

        copied = (
            bundle / "licenses" / "system" / "libstdc++" / "gcc" / "COPYING.RUNTIME"
        )
        assert copied.read_text(encoding="utf-8") == "runtime exception\n"
        assert dependencies["license_source_packages"] == {"libstdc++": ["gcc"]}
        assert dependencies["generated_license_records"] == {}
        assert dependencies["packages_without_license_files"] == []

    def test_linux_system_dependency_skips_internal_runtime_files(self, tmp_path):
        bundle = tmp_path
        (bundle / "lib").mkdir()
        (bundle / "lib" / "libq4_npu_eXpress.so").write_bytes(b"")
        with mock.patch(
            "build_tools.fastflowlm_artifact.shutil.which", lambda name: None
        ):
            dependencies = _linux_dependencies_and_licenses(
                bundle, set(), {"libq4_npu_eXpress.so"}
            )
        assert dependencies["packages"] == []
        assert dependencies["unmapped_libraries"] == []

    def test_linux_package_owner_candidates_keep_regular_path(self, tmp_path):
        source = tmp_path / "libexample.so.1"
        source.write_bytes(b"")
        assert _linux_package_owner_candidates(source) == [source]

    @pytest.mark.linux
    def test_linux_package_owner_candidates_resolve_usrmerge_symlinks(self, tmp_path):
        if not sys.platform.startswith("linux"):
            pytest.skip("requires Linux symlink semantics")
        root = tmp_path
        target = root / "usr" / "lib" / "libexample.so.1"
        source = root / "lib" / "libexample.so.1"
        target.parent.mkdir(parents=True)
        source.parent.mkdir(parents=True)
        target.write_bytes(b"")
        source.symlink_to(target)
        assert _linux_package_owner_candidates(source) == [source, target]

    @pytest.mark.windows
    def test_parse_pe_imports(self):
        if os.name != "nt":
            pytest.skip("PE fixture is built on Windows")
        executable = Path(os.environ["SystemRoot"]) / "System32" / "where.exe"
        imports = pe_imports(executable)
        assert "kernel32.dll" in imports

    def test_windows_redistributable_closure_copies_openmp_runtime(self, tmp_path):
        root = tmp_path
        bundle = root / "bundle"
        executable = bundle / "bin" / "flm.exe"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"MZfake")
        redist = root / "redist"
        vcomp = redist / "x64" / "Microsoft.VC143.OpenMP" / "vcomp140.dll"
        vcomp.parent.mkdir(parents=True)
        vcomp.write_bytes(b"MZruntime")

        with (
            mock.patch.dict(os.environ, {"VCToolsRedistDir": str(redist)}),
            mock.patch(
                "build_tools.fastflowlm_artifact.pe_imports",
                side_effect=lambda path: (
                    ["vcomp140.dll"] if Path(path).name == "flm.exe" else []
                ),
            ),
        ):
            _copy_windows_redistributable_closure(bundle)

        assert (bundle / "bin" / "vcomp140.dll").read_bytes() == b"MZruntime"

    def test_strip_windows_system_dlls_keeps_private_runtime(self, tmp_path):
        bundle = tmp_path / "bundle"
        bin_dir = bundle / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "qwen3_npu.dll").write_bytes(b"private")
        (bin_dir / "shell32.dll").write_bytes(b"os-copy")
        system32 = tmp_path / "System32"
        system32.mkdir()
        (system32 / "shell32.dll").write_bytes(b"os")

        with mock.patch.dict(os.environ, {"SystemRoot": str(tmp_path)}):
            _strip_windows_system_dlls(bundle)

        assert (bin_dir / "qwen3_npu.dll").read_bytes() == b"private"
        assert not (bin_dir / "shell32.dll").exists()

    def test_source_policy_enforces_component_pins_and_clean_staging(self):
        root = Path(__file__).resolve().parents[2]
        versions = tomllib.loads(
            (root / "manifests" / "versions.toml").read_text(encoding="utf-8")
        )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest = {
            "sources": {
                "rocm-npu-staging": {"revision": revision, "dirty": False},
                "hrx": {"revision": versions["hrx"]["commit"]},
                "FastFlowLM": {"revision": versions["fastflowlm"]["commit"]},
            }
        }
        validate_source_policy(manifest, root)
        manifest["sources"]["hrx"]["revision"] = "0" * 40
        with pytest.raises(RuntimeError, match="hrx revision"):
            validate_source_policy(manifest, root)
        manifest["sources"]["hrx"]["revision"] = versions["hrx"]["commit"]
        manifest["sources"]["rocm-npu-staging"]["dirty"] = True
        with pytest.raises(RuntimeError, match="dirty staging"):
            validate_source_policy(manifest, root)

    def test_verify_bundle_rejects_unexpected_file(self, tmp_path):
        bundle = tmp_path
        (bundle / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": []}), encoding="utf-8"
        )
        (bundle / "extra").write_text("unexpected", encoding="utf-8")
        with pytest.raises(RuntimeError, match="inventory mismatch"):
            verify_bundle(bundle)

    def test_verify_bundle_rejects_duplicate_manifest_path(self, tmp_path):
        bundle = tmp_path
        record = {"path": "bin/flm", "size": 0, "sha256": "0" * 64}
        (bundle / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "files": [record, record],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="duplicate path"):
            verify_bundle(bundle)

    def test_verify_bundle_rejects_unsafe_manifest_path(self, tmp_path):
        bundle = tmp_path
        (bundle / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "files": [{"path": "../escape", "size": 0, "sha256": "0" * 64}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="unsafe path"):
            verify_bundle(bundle)

    def test_archive_round_trip(self, tmp_path):
        if os.name != "nt" and not shutil.which("zstd"):
            pytest.skip("zstd is required for the Linux archive format")
        root = tmp_path
        bundle = self.make_bundle(root / "stage")
        archive = create_archive(bundle, root / "artifacts")
        assert archive.with_name(archive.name + ".sha256").is_file()
        provenance = archive.with_name(archive.name + ".manifest.json")
        provenance_data = json.loads(provenance.read_text(encoding="utf-8"))
        assert provenance_data["archive"] == archive.name
        assert provenance_data["archive_sha256"] == sha256_file(archive)
        assert provenance_data["bundle"]["schema_version"] == 1
        extracted = extract_archive(archive, root / "extracted", allowed_root=root)
        executable = "flm.exe" if os.name == "nt" else "flm"
        assert (extracted / "bin" / executable).read_bytes() == b"runtime"
        if os.name != "nt":
            assert os.access(extracted / "bin" / executable, os.X_OK)

    def test_archive_is_reproducible(self, tmp_path):
        if os.name != "nt" and not shutil.which("zstd"):
            pytest.skip("zstd is required for the Linux archive format")
        root = tmp_path
        bundle = self.make_bundle(root / "stage")
        first = create_archive(bundle, root / "first")
        second = create_archive(bundle, root / "second")
        assert sha256_file(first) == sha256_file(second)

    def test_zip_path_traversal_is_rejected(self, tmp_path):
        root = tmp_path
        archive = root / "malicious.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../escape", b"bad")
        with pytest.raises(RuntimeError, match="escapes extraction root"):
            extract_archive(archive, root / "extracted", allowed_root=root)

    def test_failed_extraction_preserves_previous_destination(self, tmp_path):
        root = tmp_path
        destination = root / "extracted"
        destination.mkdir()
        sentinel = destination / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        archive = root / "malicious.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../escape", b"bad")
        with pytest.raises(RuntimeError, match="escapes extraction root"):
            extract_archive(archive, destination, allowed_root=root)
        assert sentinel.read_text(encoding="utf-8") == "keep"

    def test_zip_duplicate_paths_are_rejected(self, tmp_path):
        root = tmp_path
        archive = root / "duplicate.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("bundle/file", b"first")
                output.writestr("bundle/file", b"second")
        with pytest.raises(RuntimeError, match="duplicate normalized path"):
            extract_archive(archive, root / "extracted", allowed_root=root)

    def test_zip_normalized_duplicate_paths_are_rejected(self, tmp_path):
        root = tmp_path
        archive = root / "duplicate.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("bundle/a/../file", b"first")
            output.writestr("bundle/file", b"second")
        with pytest.raises(RuntimeError, match="duplicate normalized path"):
            extract_archive(archive, root / "extracted", allowed_root=root)

    def test_extract_refuses_to_remove_allowed_root(self, tmp_path):
        root = tmp_path
        archive = root / "empty.zip"
        with zipfile.ZipFile(archive, "w"):
            pass
        with pytest.raises(RuntimeError, match="refusing to remove"):
            extract_archive(archive, root, allowed_root=root)

    def test_tar_link_is_rejected(self, tmp_path):
        if os.name == "nt" or not shutil.which("zstd"):
            pytest.skip("tar.zst security test requires zstd")
        root = tmp_path
        tar_path = root / "bad.tar"
        with tarfile.open(tar_path, "w") as output:
            info = tarfile.TarInfo("bundle/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../outside"
            output.addfile(info)
        archive = root / "bad.tar.zst"
        subprocess.run(
            ["zstd", "-q", "-f", str(tar_path), "-o", str(archive)], check=True
        )
        with pytest.raises(RuntimeError, match="links and special files"):
            extract_archive(archive, root / "extracted", allowed_root=root)
