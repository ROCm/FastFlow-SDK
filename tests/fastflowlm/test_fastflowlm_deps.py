# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from unittest import mock

import pytest

from build_tools.dev import environment
from build_tools.fastflowlm_deps import (
    dependency_errors,
    dependency_policy,
    dependency_paths,
    vcpkg_commit,
    vcpkg_triplet,
)


ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


def populate_policy_files(root: Path) -> None:
    for relative in (
        ".github/vcpkg/fastflowlm/vcpkg.json",
        ".github/vcpkg/triplets/x64-linux-rel.cmake",
        ".github/vcpkg/triplets/x64-windows-rel.cmake",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())


def populate_dependency_tree(root: Path, system: str) -> dict[str, Path | str]:
    populate_policy_files(root)
    paths = dependency_paths(root, system)
    for key in ("executable", "toolchain"):
        path = Path(paths[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    Path(paths["triplet_root"]).mkdir(parents=True)
    status = Path(paths["install_root"]) / "vcpkg/status"
    status.parent.mkdir(parents=True)
    status.write_text("Status: install ok installed\n", encoding="utf-8")
    Path(paths["marker"]).write_text(
        json.dumps(paths["policy"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


class TestFastFlowLMDependencies:
    def test_platform_triplets_are_release_only_overlays(self):
        assert vcpkg_triplet("Windows") == "x64-windows-rel"
        assert vcpkg_triplet("Linux") == "x64-linux-rel"
        for triplet in ("x64-windows-rel", "x64-linux-rel"):
            contents = (ROOT / ".github/vcpkg/triplets" / f"{triplet}.cmake").read_text()
            assert "VCPKG_BUILD_TYPE release" in contents
            assert "VCPKG_LIBRARY_LINKAGE dynamic" in contents
        linux_contents = (
            ROOT / ".github/vcpkg/triplets/x64-linux-rel.cmake"
        ).read_text()
        assert "VCPKG_TARGET_ARCHITECTURE x64" in linux_contents
        assert "VCPKG_CRT_LINKAGE dynamic" in linux_contents
        assert "VCPKG_CMAKE_SYSTEM_NAME Linux" in linux_contents

    def test_windows_cache_identity_does_not_execute_compiler(self):
        workflow = (
            ROOT / ".github/workflows/fastflowlm_hrx_windows.yml"
        ).read_text()
        assert "(Get-Command cl.exe -ErrorAction Stop).Source" in workflow
        assert "VersionInfo.FileVersion" in workflow
        assert "(& cl 2>&1" not in workflow

    def test_linux_ci_separates_build_and_packaging_prerequisites(self):
        workflow = (
            ROOT / ".github/workflows/fastflowlm_hrx_linux.yml"
        ).read_text()
        for package in (
            "bash", "kernel-headers", "nasm", "patchelf", "perl",
            "perl-IPC-Cmd", "zstd"
        ):
            assert package in workflow
        assert "dev.py doctor fastflowlm-packaging" in workflow

    def test_fastflowlm_build_gate_requires_nasm_but_not_packaging_tools(self):
        available = {
            "bash", "curl", "gcc", "g++", "git", "make", "pkg-config", "tar",
            "nasm", "unzip",
        }

        def which(name):
            return f"/usr/bin/{name}" if name in available else None

        with (
            mock.patch.object(environment.shutil, "which", side_effect=which),
            mock.patch.object(
                environment, "compilers", return_value=("gcc", "g++")
            ),
            mock.patch.object(environment, "have_header", return_value=True),
            mock.patch.object(
                environment, "have_perl_module", return_value=True
            ),
        ):
            ok, missing = environment.check_fastflowlm_linux_host(
                verbose=False
            )

        assert ok
        assert missing == []

        with (
            mock.patch.object(
                environment.shutil,
                "which",
                side_effect=lambda name: which(name) if name != "nasm" else None,
            ),
            mock.patch.object(
                environment, "compilers", return_value=("gcc", "g++")
            ),
            mock.patch.object(environment, "have_header", return_value=True),
            mock.patch.object(
                environment, "have_perl_module", return_value=True
            ),
        ):
            ok, missing = environment.check_fastflowlm_linux_host(
                verbose=False
            )

        assert not ok
        assert missing == ["nasm"]

    def test_fastflowlm_packaging_gate_reports_packaging_tools_only(self):
        with mock.patch.object(environment.shutil, "which", return_value=None):
            ok, missing = environment.check_fastflowlm_linux_packaging_tools(
                archive=True, verbose=False)

        assert not ok
        assert missing == ["patchelf", "zstd"]
        assert "nasm" not in missing

    def test_vcpkg_caches_are_saved_only_after_complete_setup(self):
        for name in ("fastflowlm_hrx_linux.yml", "fastflowlm_hrx_windows.yml"):
            workflow = (ROOT / ".github/workflows" / name).read_text()
            assert "id: fastflowlm-deps" in workflow
            assert "fastflowlm-vcpkg-v2-downloads-" in workflow
            assert "fastflowlm-vcpkg-v2-binaries-" in workflow
            assert (
                "if: success() && steps.fastflowlm-deps.outcome == 'success'"
                in workflow
            )
            assert "if: always() && steps.vcpkg" not in workflow

    def test_linux_runtime_uses_cmake_dependency_sets(self):
        cmake = ROOT / "projects/FastFlowLM/src/CMakeLists.txt"
        if not cmake.exists():
            pytest.skip("FastFlowLM submodule is not initialized")
        contents = cmake.read_text()
        # Guard against the misspelled command name.
        assert "install(RUNTIME_DEPENDENCIES" not in contents
        # Windows bundles its runtime dependencies via an install-time
        # RUNTIME_DEPENDENCY_SET.
        assert contents.count("install(RUNTIME_DEPENDENCY_SET") >= 1
        # Linux resolves the closure with CMake's dependency machinery too, but
        # scans flm together with the prebuilt engine .so as ROOTS via
        # file(GET_RUNTIME_DEPENDENCIES). Roots are traversed for their deps but
        # never emitted, so the engine libs' private deps (libgomp/libmvec) are
        # bundled without dropping unpatched engine copies into the lib dir.
        assert "file(GET_RUNTIME_DEPENDENCIES" in contents

    def test_windows_and_linux_use_repository_managed_trees(self, tmp_path):
        populate_policy_files(tmp_path)
        windows = dependency_paths(tmp_path, "Windows")
        linux = dependency_paths(tmp_path, "Linux")
        assert windows["install_root"] != linux["install_root"]
        assert (
            Path(windows["install_root"]).parent
            == Path(linux["install_root"]).parent
        )
        assert Path(windows["triplet_root"]).name == "x64-windows-rel"
        assert Path(linux["triplet_root"]).name == "x64-linux-rel"

    def test_complete_windows_dependency_tree_passes(self, tmp_path):
        paths = populate_dependency_tree(tmp_path, "Windows")
        assert dependency_errors(
            paths, cargo_available=True, rustc_version=(1, 80, 0)
        ) == []

    def test_policy_change_selects_a_new_install_tree(self, tmp_path):
        populate_policy_files(tmp_path)
        before = dependency_paths(tmp_path, "Linux")
        triplet = tmp_path / ".github/vcpkg/triplets/x64-linux-rel.cmake"
        triplet.write_text(triplet.read_text() + "# policy change\n")
        after = dependency_paths(tmp_path, "Linux")
        assert before["install_root"] != after["install_root"]
        assert before["policy"] != after["policy"]

    def test_policy_marker_mismatch_is_rejected(self, tmp_path):
        paths = populate_dependency_tree(tmp_path, "Linux")
        marker = json.loads(Path(paths["marker"]).read_text())
        marker["triplet_sha256"] = "0" * 64
        Path(paths["marker"]).write_text(json.dumps(marker))
        errors = dependency_errors(
            paths, cargo_available=True, rustc_version=(1, 80, 0)
        )
        assert any("policy marker mismatch" in error for error in errors)

    def test_manifest_owns_cross_platform_dependency_set(self):
        manifest = json.loads(
            (ROOT / ".github/vcpkg/fastflowlm/vcpkg.json").read_text())
        dependencies = {
            item if isinstance(item, str) else item["name"]: item
            for item in manifest["dependencies"]
        }
        for name in ("boost-beast", "boost-program-options", "curl", "ffmpeg",
                     "fftw3", "readline"):
            assert name in dependencies
        assert dependencies["readline"]["platform"] == "!windows"
        assert vcpkg_commit(ROOT) == manifest["builtin-baseline"]
        assert (
            dependency_policy(ROOT, "Linux")["vcpkg_commit"]
            == manifest["builtin-baseline"]
        )
