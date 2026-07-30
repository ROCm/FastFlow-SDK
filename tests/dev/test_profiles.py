# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import unittest
from dataclasses import FrozenInstanceError

from build_tools.dev import profiles


class ProfilesTest(unittest.TestCase):
    def test_profile_specs_are_frozen(self):
        spec = profiles.PROFILE_SPECS["compiler-stack"]
        with self.assertRaises(FrozenInstanceError):
            spec.setup = False
        with self.assertRaises(AttributeError):
            spec.submodules.add("other")

    def test_doctor_and_setup_memberships_are_intentional(self):
        self.assertEqual(
            set(profiles.doctor_profile_names()),
            {
                "compiler-stack",
                "fastflowlm",
                "fastflowlm-packaging",
                "all",
            },
        )
        self.assertEqual(
            set(profiles.setup_profile_names()),
            {"compiler-stack", "fastflowlm", "npu-e2e", "npu-runtime"},
        )
        self.assertNotIn("all", profiles.setup_profile_names())
        self.assertNotIn("npu-e2e", profiles.doctor_profile_names())
        self.assertNotIn("npu-runtime", profiles.doctor_profile_names())

    def test_profile_submodule_requirements(self):
        self.assertEqual(
            profiles.PROFILE_SPECS["compiler-stack"].submodules,
            {
                "llvm-project",
                "iron-tools",
                "hrx",
                "iron-ops",
            },
        )
        self.assertEqual(
            profiles.PROFILE_SPECS["fastflowlm"].submodules,
            {"hrx", "FastFlowLM"},
        )
        self.assertEqual(
            profiles.PROFILE_SPECS["npu-e2e"].submodules,
            {"hrx", "iron-tools", "iron-ops"},
        )
        self.assertFalse(profiles.PROFILE_SPECS["npu-runtime"].submodules)
        self.assertEqual(
            profiles.PROFILE_SPECS["all"].submodules,
            set(profiles.PIN_SECTION_TO_SUBMODULE.values()),
        )

    def test_profile_python_requirements(self):
        self.assertEqual(
            profiles.profile("compiler-stack").requirements.default,
            "requirements.lock",
        )
        self.assertEqual(
            profiles.profile("fastflowlm").requirements.default,
            "requirements-fastflowlm.lock",
        )
        self.assertTrue(
            profiles.profile("fastflowlm").requirements.require_hashes
        )
        self.assertEqual(
            profiles.profile("npu-runtime").requirements.default,
            "requirements-npu-runtime.txt",
        )
        self.assertIsNone(profiles.PROFILE_SPECS["all"].requirements)

    def test_setup_actions_are_profile_policy(self):
        expected = {
            "compiler-stack": (True, False),
            "fastflowlm": (False, True),
            "npu-e2e": (True, False),
            "npu-runtime": (False, False),
        }
        for profile_name, actions in expected.items():
            with self.subTest(profile=profile_name):
                spec = profiles.profile(profile_name)
                self.assertEqual(
                    (spec.install_iron_ops, spec.install_fastflowlm_native),
                    actions,
                )

    def test_eudsl_prefetch_is_profile_policy(self):
        self.assertTrue(profiles.profile("compiler-stack").needs_eudsl)
        self.assertTrue(profiles.profile("all").needs_eudsl)
        self.assertFalse(profiles.profile("fastflowlm").needs_eudsl)

    def test_doctor_system_checks_are_profile_policy(self):
        self.assertEqual(
            profiles.profile("compiler-stack").system_checks,
            ("compiler-stack",),
        )
        self.assertEqual(
            profiles.profile("fastflowlm").system_checks,
            ("fastflowlm",),
        )
        self.assertEqual(
            profiles.profile("fastflowlm-packaging").system_checks,
            ("fastflowlm-packaging",),
        )
        self.assertEqual(
            profiles.profile("all").system_checks,
            ("compiler-stack", "fastflowlm"),
        )

    def test_profile_lookup_rejects_unknown_names(self):
        with self.assertRaises(KeyError):
            profiles.profile("unknown")


if __name__ == "__main__":
    unittest.main()
