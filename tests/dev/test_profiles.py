# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from build_tools.dev import profiles

pytestmark = pytest.mark.presubmit
ROOT = Path(__file__).resolve().parents[2]


def test_requirement_profiles_are_grouped_and_locked():
    expected = {
        "compiler-stack",
        "fastflowlm",
        "iron-ops-test",
        "npu-test-runtime",
        "packaging-linux",
        "release-wheels",
        "test",
    }
    requirements = ROOT / "requirements"
    assert {path.stem for path in requirements.glob("*.txt")} == expected
    assert {path.stem for path in requirements.glob("*.lock")} == expected
    assert not list(ROOT.glob("requirements*.txt"))
    assert not list(ROOT.glob("requirements*.lock"))


class TestProfiles:
    def test_profile_specs_are_frozen(self):
        spec = profiles.PROFILE_SPECS["compiler-stack"]
        with pytest.raises(FrozenInstanceError):
            spec.setup = False
        with pytest.raises(AttributeError):
            spec.submodules.add("other")

    def test_doctor_and_setup_memberships_are_intentional(self):
        assert set(profiles.doctor_profile_names()) == {
            "compiler-stack",
            "fastflowlm",
            "fastflowlm-packaging",
            "all",
        }
        assert set(profiles.setup_profile_names()) == {
            "compiler-stack",
            "fastflowlm",
            "npu-e2e",
            "npu-test-runtime",
        }
        assert "all" not in profiles.setup_profile_names()
        assert "npu-e2e" not in profiles.doctor_profile_names()
        assert "npu-test-runtime" not in profiles.doctor_profile_names()

    def test_profile_submodule_requirements(self):
        assert profiles.PROFILE_SPECS["compiler-stack"].submodules == {
            "llvm-project",
            "aiebu",
            "iron-tools",
            "hrx",
            "iron-ops",
        }
        assert profiles.PROFILE_SPECS["fastflowlm"].submodules == {
            "hrx",
            "FastFlowLM",
        }
        assert profiles.PROFILE_SPECS["npu-e2e"].submodules == {
            "hrx",
            "aiebu",
            "iron-tools",
            "iron-ops",
        }
        assert not profiles.PROFILE_SPECS["npu-test-runtime"].submodules
        assert profiles.PROFILE_SPECS["all"].submodules == set(
            profiles.PIN_SECTION_TO_SUBMODULE.values()
        )

    def test_profile_python_requirements(self):
        assert (
            profiles.profile("compiler-stack").requirements.default
            == "requirements/compiler-stack.lock"
        )
        assert (
            profiles.profile("fastflowlm").requirements.default
            == "requirements/fastflowlm.lock"
        )
        assert profiles.profile("fastflowlm").requirements.require_hashes
        assert (
            profiles.profile("npu-test-runtime").requirements.default
            == "requirements/npu-test-runtime.lock"
        )
        assert profiles.profile("npu-test-runtime").requirements.require_hashes
        assert profiles.PROFILE_SPECS["all"].requirements is None

    @pytest.mark.parametrize(
        ("profile_name", "actions"),
        [
            ("compiler-stack", (True, False)),
            ("fastflowlm", (False, True)),
            ("npu-e2e", (True, False)),
            ("npu-test-runtime", (False, False)),
        ],
    )
    def test_setup_actions_are_profile_policy(self, profile_name, actions):
        spec = profiles.profile(profile_name)
        assert (spec.install_iron_ops, spec.install_fastflowlm_native) == actions

    def test_eudsl_prefetch_is_profile_policy(self):
        assert profiles.profile("compiler-stack").needs_eudsl
        assert profiles.profile("all").needs_eudsl
        assert not profiles.profile("fastflowlm").needs_eudsl

    def test_doctor_system_checks_are_profile_policy(self):
        assert profiles.profile("compiler-stack").system_checks == ("compiler-stack",)
        assert profiles.profile("fastflowlm").system_checks == ("fastflowlm",)
        assert profiles.profile("fastflowlm-packaging").system_checks == (
            "fastflowlm-packaging",
        )
        assert profiles.profile("all").system_checks == (
            "compiler-stack",
            "fastflowlm",
        )

    def test_profile_lookup_rejects_unknown_names(self):
        with pytest.raises(KeyError):
            profiles.profile("unknown")
