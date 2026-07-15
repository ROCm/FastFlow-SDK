# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest import mock

import dev


class DevWindowsToolchainTest(unittest.TestCase):
    def setUp(self):
        self.old_toolchain = dev._TOOLCHAIN

    def tearDown(self):
        dev._TOOLCHAIN = self.old_toolchain

    def test_toolchain_flag_pins_msvc_compilers(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            dev.apply_toolchain_arg(SimpleNamespace(toolchain="msvc"))

            self.assertEqual(dev._compilers(), ("cl", "cl"))
            self.assertEqual(os.environ["CC"], "cl")
            self.assertEqual(os.environ["CXX"], "cl")

    def test_toolchain_flag_pins_clang_cl_compilers(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            dev.apply_toolchain_arg(SimpleNamespace(toolchain="clang-cl"))

            self.assertEqual(dev._compilers(), ("clang-cl", "clang-cl"))
            self.assertEqual(os.environ["CC"], "clang-cl")
            self.assertEqual(os.environ["CXX"], "clang-cl")

    def test_toolchain_flag_pins_gcc_compilers(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            dev.apply_toolchain_arg(SimpleNamespace(toolchain="gcc"))

            self.assertEqual(dev._compilers(), ("gcc", "g++"))
            self.assertEqual(os.environ["CC"], "gcc")
            self.assertEqual(os.environ["CXX"], "g++")

    def test_toolchain_flag_pins_clang_compilers(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            dev.apply_toolchain_arg(SimpleNamespace(toolchain="clang"))

            self.assertEqual(dev._compilers(), ("clang", "clang++"))
            self.assertEqual(os.environ["CC"], "clang")
            self.assertEqual(os.environ["CXX"], "clang++")

    def test_hrx_xclbinutil_uses_default_build_dir_for_auto_toolchain(self):
        dev._TOOLCHAIN = "auto"

        self.assertEqual(dev._hrx_xclbinutil_build_dir().name, "bld")

    def test_hrx_xclbinutil_uses_toolchain_specific_build_dir(self):
        dev._TOOLCHAIN = "msvc"

        self.assertEqual(dev._hrx_xclbinutil_build_dir().name, "bld-msvc")

    @unittest.skipUnless(dev.IS_WINDOWS, "Windows-specific Git Bash path policy")
    def test_git_bash_path_converts_windows_drive_path_without_cygpath(self):
        with mock.patch("dev.shutil.which", return_value=None):
            self.assertEqual(
                dev._git_bash_path(Path("C:/work/repo/test/run_tests.sh")),
                "/c/work/repo/test/run_tests.sh",
            )

    @unittest.skipUnless(dev.IS_WINDOWS, "Windows-specific ccache policy")
    def test_msvc_cl_skips_ccache_launcher(self):
        dev._TOOLCHAIN = "msvc"

        with mock.patch("dev.shutil.which", return_value="ccache"):
            self.assertEqual(dev._ccache_flags(), [])

    @unittest.skipUnless(dev.IS_WINDOWS, "Windows-specific ccache policy")
    def test_auto_windows_toolchain_skips_ccache_launcher(self):
        dev._TOOLCHAIN = "auto"

        with mock.patch("dev.shutil.which", return_value="ccache"):
            self.assertEqual(dev._ccache_flags(), [])

    @unittest.skipUnless(dev.IS_WINDOWS, "Windows-specific ccache policy")
    def test_clang_cl_keeps_ccache_launcher(self):
        dev._TOOLCHAIN = "clang-cl"

        with mock.patch("dev.shutil.which", return_value="ccache"):
            self.assertIn(
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                dev._ccache_flags(),
            )


if __name__ == "__main__":
    unittest.main()
