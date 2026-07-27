# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Declarative dependency profiles used by developer commands."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RequirementsSpec:
    """Platform-specific requirements input for one setup profile."""

    default: str
    windows: str | None = None
    fallback_if_missing: str | None = None
    require_hashes: bool = False

    def filename(self, *, windows: bool) -> str:
        return self.windows if windows and self.windows else self.default


@dataclass(frozen=True)
class ProfileSpec:
    """Doctor and setup policy for one dependency closure."""

    submodules: frozenset[str]
    doctor: bool = False
    system_checks: tuple[str, ...] = ()
    setup: bool = False
    requirements: RequirementsSpec | None = None
    needs_eudsl: bool = False
    check_system_before_setup: bool = False
    install_iron_ops: bool = False
    install_fastflowlm_native: bool = False


_PINNED_SUBMODULES = (
    ("llvm", "llvm-project"),
    ("iron_tools", "iron-tools"),
    ("hrx", "hrx"),
    ("fastflowlm", "FastFlowLM"),
    ("iron_ops", "iron-ops"),
)

PIN_SECTION_TO_SUBMODULE = dict(_PINNED_SUBMODULES)

_COMPILER_STACK = frozenset(
    {"llvm-project", "iron-tools", "hrx", "iron-ops"}
)
_FASTFLOWLM = frozenset({"hrx", "FastFlowLM"})

PROFILE_SPECS = {
    "compiler-stack": ProfileSpec(
        submodules=_COMPILER_STACK,
        doctor=True,
        system_checks=("compiler-stack",),
        setup=True,
        requirements=RequirementsSpec(
            default="requirements.lock",
            windows="requirements.txt",
            fallback_if_missing="requirements.txt",
        ),
        needs_eudsl=True,
        check_system_before_setup=True,
        install_iron_ops=True,
    ),
    "fastflowlm": ProfileSpec(
        submodules=_FASTFLOWLM,
        doctor=True,
        system_checks=("fastflowlm",),
        setup=True,
        requirements=RequirementsSpec(
            default="requirements-fastflowlm.lock",
            require_hashes=True,
        ),
        install_fastflowlm_native=True,
    ),
    "fastflowlm-packaging": ProfileSpec(
        submodules=_FASTFLOWLM,
        doctor=True,
        system_checks=("fastflowlm-packaging",),
    ),
    # `all` validates every pin but is intentionally not a setup profile.
    "all": ProfileSpec(
        submodules=frozenset(PIN_SECTION_TO_SUBMODULE.values()),
        doctor=True,
        system_checks=("compiler-stack", "fastflowlm"),
        needs_eudsl=True,
    ),
    # Device test preparation does not need the full compiler-stack closure.
    "npu-e2e": ProfileSpec(
        submodules=frozenset({"hrx", "iron-tools", "iron-ops"}),
        setup=True,
        requirements=RequirementsSpec(
            default="requirements.lock",
            windows="requirements.txt",
            fallback_if_missing="requirements.txt",
        ),
        install_iron_ops=True,
    ),
    # Runtime-only setup has no source submodule requirement.
    "npu-runtime": ProfileSpec(
        submodules=frozenset(),
        setup=True,
        requirements=RequirementsSpec(default="requirements-npu-runtime.txt"),
    ),
}

def profile(name: str) -> ProfileSpec:
    return PROFILE_SPECS[name]


def doctor_profile_names() -> tuple[str, ...]:
    return tuple(sorted(name for name, spec in PROFILE_SPECS.items() if spec.doctor))


def setup_profile_names() -> tuple[str, ...]:
    return tuple(sorted(name for name, spec in PROFILE_SPECS.items() if spec.setup))
