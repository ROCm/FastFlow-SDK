# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
from types import SimpleNamespace
from unittest import mock

import pytest

from build_tools import paths
from build_tools.dev import doctor, profiles

pytestmark = pytest.mark.presubmit


class TestSystemCheck:
    def test_profile_routing_and_all_result(self):
        with (
            mock.patch.object(
                doctor, "check_compiler_stack_system", return_value=(False, ["z", "a"])
            ) as compiler,
            mock.patch.object(
                doctor, "check_fastflowlm_system", return_value=(True, ["a"])
            ) as fastflowlm,
        ):
            assert doctor.check_system("compiler-stack", verbose=False) == (
                False,
                ["z", "a"],
            )
            assert doctor.check_system("fastflowlm", verbose=False) == (
                True,
                ["a"],
            )
            assert doctor.check_system("all", verbose=False) == (
                False,
                ["a", "z"],
            )

        assert compiler.call_args_list == [mock.call(False), mock.call(False)]
        assert fastflowlm.call_args_list == [
            mock.call(False),
            mock.call(False),
        ]

    def test_unknown_profile_is_rejected(self):
        with pytest.raises(ValueError, match="unknown doctor profile: other"):
            doctor.check_system("other", verbose=False)

    def test_packaging_profile_routes_to_packaging_check(self):
        with mock.patch.object(
            doctor,
            "check_fastflowlm_packaging_system",
            return_value=(False, ["zstd"]),
        ) as packaging:
            assert doctor.check_system("fastflowlm-packaging", verbose=False) == (
                False,
                ["zstd"],
            )
        packaging.assert_called_once_with(False)

    def test_windows_branch_uses_patchable_platform_flag(self):
        env = {"Path": "toolchain"}

        def which(executable, selected_env=None, *, windows=None):
            assert selected_env is env
            assert windows
            return None

        with (
            mock.patch.object(
                doctor.environment, "compilers", return_value=(None, None)
            ),
            mock.patch.object(doctor.environment, "windows_msvc_env", return_value=env),
            mock.patch.object(doctor.environment, "which", side_effect=which),
        ):
            assert doctor.check_compiler_stack_system(verbose=False, windows=True) == (
                False,
                [],
            )


class TestDoctorCommand:
    def _repository_layout(self, root):
        patch_dir = root / "patches" / "hrx"
        patch_dir.mkdir(parents=True)
        patch_file = patch_dir / "0001-test.patch"
        patch_file.touch()
        (root / "projects" / "hrx" / ".git").mkdir(parents=True)
        return patch_file

    @pytest.mark.parametrize("deep", [False, True])
    def test_quick_and_deep_patch_wiring(self, deep, tmp_path):
        patch_file = self._repository_layout(tmp_path)
        args = SimpleNamespace(profile="fastflowlm", deep=deep)
        with (
            mock.patch.object(paths, "PATCHES", tmp_path / "patches"),
            mock.patch.object(paths, "PROJ", tmp_path / "projects"),
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
            assert doctor.cmd_doctor(args) == 0

        check_pins.assert_called_once_with(
            set(profiles.profile("fastflowlm").submodules)
        )
        patch_states.assert_called_once_with(
            "hrx",
            tmp_path / "projects" / "hrx",
            [patch_file],
            deep=deep,
        )

    def test_failures_produce_nonzero_return(self, tmp_path, capsys):
        self._repository_layout(tmp_path)
        with (
            mock.patch.object(paths, "PATCHES", tmp_path / "patches"),
            mock.patch.object(paths, "PROJ", tmp_path / "projects"),
            mock.patch.object(
                doctor, "check_system", return_value=(False, ["missing-system"])
            ),
            mock.patch.object(doctor.patches, "check_pins", return_value=["hrx"]),
            mock.patch.object(
                doctor.patches,
                "doctor_patch_states",
                return_value=["broken"],
            ),
        ):
            result = doctor.cmd_doctor(
                SimpleNamespace(profile="fastflowlm", deep=False)
            )

        assert result == 1
        text = capsys.readouterr().out
        assert "PATCH does not apply: 0001-test.patch" in text
        assert "MISSING system deps" in text
        assert "SUBMODULE PIN problems in: hrx" in text

    def test_compiler_stack_checks_prefetch_and_returns_success(self, tmp_path):
        patch_file = (
            tmp_path / "patches" / "iron-tools" / "0001-eudsl-for-amd-llvm-23.patch"
        )
        patch_file.parent.mkdir(parents=True)
        patch_file.write_text("+eudsl-python-extras==1.2.3\n")
        with (
            mock.patch.object(paths, "PATCHES", tmp_path / "patches"),
            mock.patch.object(paths, "PROJ", tmp_path / "projects"),
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

        assert result == 0
        cache_ok.assert_called_once_with()
