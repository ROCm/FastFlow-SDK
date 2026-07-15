# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import tempfile
import unittest
from pathlib import Path

from build_tools.fastflowlm_deps import (
    resolve_windows_roots,
    windows_dependency_errors,
)

ROOT = Path(__file__).resolve().parents[2]


class FastFlowLMWindowsDependenciesTest(unittest.TestCase):
    def test_requires_beast_header_in_addition_to_program_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            installed = Path(temp_dir)
            include = installed / "include"
            lib = installed / "lib"
            (include / "boost").mkdir(parents=True)
            lib.mkdir()
            (include / "boost" / "program_options.hpp").touch()

            for name in (
                "libboost_program_options.lib",
                "libcurl.lib",
                "avformat.lib",
                "avcodec.lib",
                "avutil.lib",
                "swscale.lib",
                "swresample.lib",
                "fftw3.lib",
                "fftw3f.lib",
                "fftw3l.lib",
            ):
                (lib / name).touch()

            roots = {
                "boost_root": include,
                "boost_lib": lib,
                "vcpkg_root": installed,
                "vcpkg_bin": installed / "bin",
            }
            errors = windows_dependency_errors(
                roots, cargo_available=True, rustc_version=(1, 80, 0)
            )
            self.assertTrue(any("Boost.Beast headers" in error for error in errors))

            (include / "boost" / "beast").mkdir()
            (include / "boost" / "beast" / "core.hpp").touch()
            self.assertEqual(
                windows_dependency_errors(
                    roots, cargo_available=True, rustc_version=(1, 80, 0)
                ),
                [],
            )

    def test_managed_vcpkg_headers_are_selected_as_one_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            installed = root / ".cache" / "vcpkg-installed" / "x64-windows-rel"
            (installed / "include" / "boost").mkdir(parents=True)
            (installed / "lib").mkdir()
            (installed / "lib" / "boost_program_options.lib").touch()

            roots = resolve_windows_roots(root, {})

        self.assertEqual(roots["vcpkg_root"], installed)
        self.assertEqual(roots["boost_root"], installed / "include")
        self.assertEqual(roots["boost_lib"], installed / "lib")

    def test_never_mixes_boost_headers_and_libraries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vcpkg = root / "vcpkg"
            standalone = root / "boost"
            (vcpkg / "include" / "boost").mkdir(parents=True)
            (vcpkg / "lib").mkdir()
            (standalone / "boost").mkdir(parents=True)
            (standalone / "stage" / "lib").mkdir(parents=True)
            (standalone / "stage" / "lib" / "boost_program_options.lib").touch()

            roots = resolve_windows_roots(
                root,
                {
                    "FASTFLOWLM_VCPKG_INSTALLED_DIR": str(vcpkg),
                    "FASTFLOWLM_BOOST_ROOT": str(standalone),
                },
            )

        self.assertEqual(roots["boost_root"], standalone)
        self.assertEqual(roots["boost_lib"], standalone / "stage" / "lib")

    def test_invalid_explicit_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            managed = root / ".cache" / "vcpkg-installed" / "x64-windows-rel"
            (managed / "include" / "boost").mkdir(parents=True)
            (managed / "lib").mkdir()
            explicit = root / "does-not-exist"

            roots = resolve_windows_roots(
                root, {"FASTFLOWLM_VCPKG_INSTALLED_DIR": str(explicit)}
            )

        self.assertEqual(roots["vcpkg_root"], explicit)
        self.assertIsNone(roots["boost_lib"])

    def test_manifest_declares_the_beast_header_provider(self):
        manifest = json.loads(
            (ROOT / ".github" / "vcpkg" / "fastflowlm" / "vcpkg.json").read_text()
        )
        names = {
            dependency if isinstance(dependency, str) else dependency["name"]
            for dependency in manifest["dependencies"]
        }
        self.assertIn("boost-beast", names)


if __name__ == "__main__":
    unittest.main()
