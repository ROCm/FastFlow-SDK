# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic prompt materialization for bounded context validation."""

from __future__ import annotations

from typing import Any


_FILLER_WORDS = (
    "amber",
    "bridge",
    "cobalt",
    "delta",
    "engine",
    "fabric",
    "granite",
    "harbor",
    "island",
    "junction",
    "kernel",
    "lantern",
    "matrix",
    "network",
    "orbit",
    "pipeline",
    "quartz",
    "runtime",
    "signal",
    "tensor",
)


def context_tail_marker_prompt(specification: dict[str, Any]) -> str:
    """Builds a long prompt with a marker followed by enough text to expose truncation."""
    filler_word_count = int(specification["filler_word_count"])
    marker_word_offset = int(specification["marker_word_offset"])
    marker = specification["marker"]
    words = [
        _FILLER_WORDS[index % len(_FILLER_WORDS)] for index in range(filler_word_count)
    ]
    words.insert(
        marker_word_offset,
        f"The required validation marker is {marker}.",
    )
    document = " ".join(words)
    return (
        "This is an exact-copy test, not a summarization task. "
        "Read the complete deterministic validation document. "
        f"{document} "
        "Do not summarize. Find the sentence beginning with "
        "'The required validation marker is' near the end of the document. "
        "Reply with only its hyphenated marker value, exactly as written."
    )


def materialize_turn(turn: dict[str, Any]) -> dict[str, Any]:
    if "prompt" in turn:
        return dict(turn)
    specification = turn["prompt_generator"]
    if specification["kind"] != "context-tail-marker-v1":
        raise ValueError(
            f"unknown qualification prompt generator: {specification['kind']}"
        )
    materialized = dict(turn)
    materialized["prompt"] = context_tail_marker_prompt(specification)
    del materialized["prompt_generator"]
    return materialized
