# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from build_tools.dev import doctor, paths, profiles


class SystemCheckTest(unittest.TestCase):
    def test_profile_routing_and_all_result(self):
        with (
            mock.patch.object(
                doctor, "check_compiler_stack_system", return_value=(False, ["z", "a"])
            ) as compiler,
            mock.patch.object(
                doctor, "check_fastflowlm_system", return_value=(True, ["a"])
            ) as fastflowlm,
        ):
            self.assertEqual(
                doctor.check_system("compiler-stack", verbose=False),
                (False, ["z", "a"]),
            )
            self.assertEqual(
                doctor.check_system("fastflowlm", verbose=False),
                (True, ["a"]),
            )
            self.assertEqual(
                doctor.check_system("all", verbose=False),
                (False, ["a", "z"]),
            )

        self.assertEqual(compiler.call_args_list, [mock.call(False), mock.call(False)])
        self.assertEqual(
            fastflowlm.call_args_list, [mock.call(False), mock.call(False)]
        )

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown doctor profile: other"):
            doctor.check_system("other", verbose=False)

    def test_packaging_profile_routes_to_packaging_check(self):
        with mock.patch.object(
            doctor,
            "check_fastflowlm_packaging_system",
            return_value=(False, ["zstd"]),
        ) as packaging:
            self.assertEqual(
                doctor.check_system(
                    "fastflowlm-packaging", verbose=False
                ),
                (False, ["zstd"]),
            )
        packaging.assert_called_once_with(False)

    def test_windows_branch_uses_patchable_platform_flag(self):
        env = {"Path": "toolchain"}

        def which(executable, selected_env=None, *, windows=None):
            self.assertIs(selected_env, env)
            self.assertTrue(windows)
            return None

        with (
            mock.patch.object(
                doctor.environment, "compilers", return_value=(None, None)
            ),
            mock.patch.object(
                doctor.environment, "windows_msvc_env", return_value=env
            ),
            mock.patch.object(doctor.environment, "which", side_effect=which),
        ):
            self.assertEqual(
                doctor.check_compiler_stack_system(
                    verbose=False, windows=True
                ),
                (False, []),
            )


class DoctorCommandTest(unittest.TestCase):
    def _repository_layout(self, root):
        patch_dir = root / "patches" / "hrx"
        patch_dir.mkdir(parents=True)
        patch_file = patch_dir / "0001-test.patch"
        patch_file.touch()
        (root / "projects" / "hrx" / ".git").mkdir(parents=True)
        return patch_file

    def test_quick_and_deep_patch_wiring(self):
        for deep in (False, True):
            with self.subTest(deep=deep), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                patch_file = self._repository_layout(root)
                args = SimpleNamespace(profile="fastflowlm", deep=deep)
                with (
                    mock.patch.object(paths, "PATCHES", root / "patches"),
                    mock.patch.object(paths, "PROJ", root / "projects"),
                    mock.patch.object(doctor, "check_system", return_value=(True, [])),
                    mock.patch.object(
                        doctor.patches, "check_pins", return_value=[]
                    ) as check_pins,
                    mock.patch.object(
                        doctor.patches,
                        "doctor_patch_states",
                        return_value=["applied"],
                    ) as patch_states,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(doctor.cmd_doctor(args), 0)

                check_pins.assert_called_once_with(
                    set(profiles.profile("fastflowlm").submodules)
                )
                patch_states.assert_called_once_with(
                    "hrx",
                    root / "projects" / "hrx",
                    [patch_file],
                    deep=deep,
                )

    def test_failures_produce_nonzero_return(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._repository_layout(root)
            output = io.StringIO()
            with (
                mock.patch.object(paths, "PATCHES", root / "patches"),
                mock.patch.object(paths, "PROJ", root / "projects"),
                mock.patch.object(
                    doctor, "check_system", return_value=(False, ["missing-system"])
                ),
                mock.patch.object(
                    doctor.patches, "check_pins", return_value=["hrx"]
                ),
                mock.patch.object(
                    doctor.patches,
                    "doctor_patch_states",
                    return_value=["broken"],
                ),
                contextlib.redirect_stdout(output),
            ):
                result = doctor.cmd_doctor(
                    SimpleNamespace(profile="fastflowlm", deep=False)
                )

        self.assertEqual(result, 1)
        text = output.getvalue()
        self.assertIn("PATCH does not apply: 0001-test.patch", text)
        self.assertIn("MISSING system deps", text)
        self.assertIn("SUBMODULE PIN problems in: hrx", text)

    def test_compiler_stack_checks_prefetch_and_returns_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patch_file = (
                root
                / "patches"
                / "iron-tools"
                / "0001-eudsl-for-amd-llvm-23.patch"
            )
            patch_file.parent.mkdir(parents=True)
            patch_file.write_text("+eudsl-python-extras==1.2.3\n")
            with (
                mock.patch.object(paths, "PATCHES", root / "patches"),
                mock.patch.object(paths, "PROJ", root / "projects"),
                mock.patch.object(doctor, "check_system", return_value=(True, [])),
                mock.patch.object(doctor.patches, "check_pins", return_value=[]),
                mock.patch.object(
                    doctor.patches, "eudsl_cache_ok", return_value=True
                ) as cache_ok,
                mock.patch.object(
                    doctor.patches,
                    "manifest",
                    return_value={"eudsl": {"version": "1.2.3"}},
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = doctor.cmd_doctor(
                    SimpleNamespace(profile="compiler-stack", deep=False)
                )

        self.assertEqual(result, 0)
        cache_ok.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
