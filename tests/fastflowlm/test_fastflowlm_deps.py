# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import tempfile
import unittest
from pathlib import Path

from build_tools.fastflowlm_deps import (
    dependency_errors,
    dependency_policy,
    dependency_paths,
    vcpkg_commit,
    vcpkg_triplet,
)


ROOT = Path(__file__).resolve().parents[2]


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


class FastFlowLMDependenciesTest(unittest.TestCase):
    def test_platform_triplets_are_release_only_overlays(self):
        self.assertEqual(vcpkg_triplet("Windows"), "x64-windows-rel")
        self.assertEqual(vcpkg_triplet("Linux"), "x64-linux-rel")
        for triplet in ("x64-windows-rel", "x64-linux-rel"):
            contents = (ROOT / ".github/vcpkg/triplets" / f"{triplet}.cmake").read_text()
            self.assertIn("VCPKG_BUILD_TYPE release", contents)
            self.assertIn("VCPKG_LIBRARY_LINKAGE dynamic", contents)
        linux_contents = (
            ROOT / ".github/vcpkg/triplets/x64-linux-rel.cmake"
        ).read_text()
        self.assertIn("VCPKG_TARGET_ARCHITECTURE x64", linux_contents)
        self.assertIn("VCPKG_CRT_LINKAGE dynamic", linux_contents)
        self.assertIn("VCPKG_CMAKE_SYSTEM_NAME Linux", linux_contents)

    def test_windows_cache_identity_does_not_execute_compiler(self):
        workflow = (
            ROOT / ".github/workflows/fastflowlm_hrx_windows.yml"
        ).read_text()
        self.assertIn("(Get-Command cl.exe -ErrorAction Stop).Source", workflow)
        self.assertIn("VersionInfo.FileVersion", workflow)
        self.assertNotIn("(& cl 2>&1", workflow)

    def test_linux_ci_installs_vcpkg_host_prerequisites(self):
        workflow = (
            ROOT / ".github/workflows/fastflowlm_hrx_linux.yml"
        ).read_text()
        for package in (
            "bash", "kernel-headers", "nasm", "patchelf", "perl", "perl-IPC-Cmd"
        ):
            self.assertIn(package, workflow)

    def test_vcpkg_caches_are_saved_only_after_complete_setup(self):
        for name in ("fastflowlm_hrx_linux.yml", "fastflowlm_hrx_windows.yml"):
            workflow = (ROOT / ".github/workflows" / name).read_text()
            self.assertIn("id: fastflowlm-deps", workflow)
            self.assertIn("fastflowlm-vcpkg-v2-downloads-", workflow)
            self.assertIn("fastflowlm-vcpkg-v2-binaries-", workflow)
            self.assertIn(
                "if: success() && steps.fastflowlm-deps.outcome == 'success'",
                workflow,
            )
            self.assertNotIn("if: always() && steps.vcpkg", workflow)

    def test_linux_runtime_uses_cmake_dependency_sets(self):
        cmake = ROOT / "projects/FastFlowLM/src/CMakeLists.txt"
        if not cmake.exists():
            self.skipTest("FastFlowLM submodule is not initialized")
        contents = cmake.read_text()
        # Guard against the misspelled command name.
        self.assertNotIn("install(RUNTIME_DEPENDENCIES", contents)
        # Windows bundles its runtime dependencies via an install-time
        # RUNTIME_DEPENDENCY_SET.
        self.assertGreaterEqual(
            contents.count("install(RUNTIME_DEPENDENCY_SET"), 1
        )
        # Linux resolves the closure with CMake's dependency machinery too, but
        # scans flm together with the prebuilt engine .so as ROOTS via
        # file(GET_RUNTIME_DEPENDENCIES). Roots are traversed for their deps but
        # never emitted, so the engine libs' private deps (libgomp/libmvec) are
        # bundled without dropping unpatched engine copies into the lib dir.
        self.assertIn("file(GET_RUNTIME_DEPENDENCIES", contents)

    def test_windows_and_linux_use_repository_managed_trees(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate_policy_files(root)
            windows = dependency_paths(root, "Windows")
            linux = dependency_paths(root, "Linux")
        self.assertNotEqual(windows["install_root"], linux["install_root"])
        self.assertEqual(
            Path(windows["install_root"]).parent,
            Path(linux["install_root"]).parent,
        )
        self.assertEqual(Path(windows["triplet_root"]).name, "x64-windows-rel")
        self.assertEqual(Path(linux["triplet_root"]).name, "x64-linux-rel")

    def test_complete_windows_dependency_tree_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = populate_dependency_tree(Path(temporary), "Windows")
            self.assertEqual(dependency_errors(
                paths, cargo_available=True, rustc_version=(1, 80, 0)), [])

    def test_policy_change_selects_a_new_install_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate_policy_files(root)
            before = dependency_paths(root, "Linux")
            triplet = root / ".github/vcpkg/triplets/x64-linux-rel.cmake"
            triplet.write_text(triplet.read_text() + "# policy change\n")
            after = dependency_paths(root, "Linux")
        self.assertNotEqual(before["install_root"], after["install_root"])
        self.assertNotEqual(before["policy"], after["policy"])

    def test_policy_marker_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = populate_dependency_tree(Path(temporary), "Linux")
            marker = json.loads(Path(paths["marker"]).read_text())
            marker["triplet_sha256"] = "0" * 64
            Path(paths["marker"]).write_text(json.dumps(marker))
            errors = dependency_errors(
                paths, cargo_available=True, rustc_version=(1, 80, 0))
        self.assertTrue(any("policy marker mismatch" in error for error in errors))

    def test_manifest_owns_cross_platform_dependency_set(self):
        manifest = json.loads(
            (ROOT / ".github/vcpkg/fastflowlm/vcpkg.json").read_text())
        dependencies = {
            item if isinstance(item, str) else item["name"]: item
            for item in manifest["dependencies"]
        }
        for name in ("boost-beast", "boost-program-options", "curl", "ffmpeg",
                     "fftw3", "readline"):
            self.assertIn(name, dependencies)
        self.assertEqual(dependencies["readline"]["platform"], "!windows")
        self.assertEqual(vcpkg_commit(ROOT), manifest["builtin-baseline"])
        self.assertEqual(
            dependency_policy(ROOT, "Linux")["vcpkg_commit"],
            manifest["builtin-baseline"],
        )


if __name__ == "__main__":
    unittest.main()
