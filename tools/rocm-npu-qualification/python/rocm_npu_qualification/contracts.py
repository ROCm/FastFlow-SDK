# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Version identifiers for independently evolving qualification contracts."""

CANDIDATE_SCHEMA_VERSION = 1
COMMAND_REPORT_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 1
KERNEL_RECIPE_SCHEMA_VERSION = 1
POLICY_SCHEMA_VERSION = 1
PROFILE_SCHEMA_VERSION = 1
QUALIFICATION_CONTRACT_SCHEMA_VERSION = 1
XADX_REPORT_SCHEMA_VERSION = 1

# How many times each qualification profile repeats the XADX case set. Rounds
# buy repetition, not coverage: every profile runs every case, and the case set
# alone already dispatches each implementation against each value population.
# What more rounds add is the chance to observe state surviving where it should
# not -- resources leaking across a runtime lifecycle, a resident buffer
# accumulating between dispatches, a rewritten input the device kept a stale view
# of. Two lifecycle rounds is the least that can observe cross-lifecycle leakage
# at all, so even the bounded profile pays for two.
#
# Every coherence case receives the full readback depth. The runner validates
# this depth against each case's actual variant count, so recipe evolution cannot
# silently make the rotation uneven.
XADX_ROUNDS = {
    name: {
        "lifecycle_rounds": 5,
        "resident_rounds": 20,
        "readback_rounds": 32,
    }
    # "all-models", "bench-all", and "bench-dev" are the diagnostic sweeps.
    # They share the curated soak depth (5/20/32) but are named explicitly rather
    # than falling through to the default policy, so a sweep's XADX execution can
    # never drift silently.
    for name in (
        "fast", "default", "extended", "all-models", "bench-all", "bench-dev"
    )
}


def xadx_rounds(profile_name: str) -> dict[str, int]:
    """How many rounds the named profile runs.

    An unknown profile takes the default policy rather than none: a profile that
    forgets to say should still be qualified, just not at soak depth.
    """
    return dict(XADX_ROUNDS.get(profile_name, XADX_ROUNDS["default"]))
