# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared normalization for qualification policy contracts."""

from __future__ import annotations

from typing import Any

from .contracts import POLICY_SCHEMA_VERSION


class PolicyValidationError(ValueError):
    pass


def normalize_policy(
    value: Any, *, expected_name: str | None = None
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "name", "requirements"}
        or value.get("schema_version") != POLICY_SCHEMA_VERSION
        or not isinstance(value.get("name"), str)
        or (expected_name is not None and value["name"] != expected_name)
        or not isinstance(value.get("requirements"), list)
        or not value["requirements"]
    ):
        raise PolicyValidationError("qualification policy is invalid")
    requirements = []
    for requirement in value["requirements"]:
        if (
            not isinstance(requirement, dict)
            or set(requirement)
            != {"platform_os", "profile", "target_family", "minimum_runs"}
            or requirement.get("platform_os") not in {"linux", "windows"}
            or not isinstance(requirement.get("profile"), str)
            or not requirement["profile"]
            or not isinstance(requirement.get("target_family"), str)
            or not requirement["target_family"]
            or not isinstance(requirement.get("minimum_runs"), int)
            or requirement["minimum_runs"] < 1
        ):
            raise PolicyValidationError(
                "qualification policy requirement is invalid"
            )
        requirements.append(dict(requirement))
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "name": value["name"],
        "requirements": requirements,
    }
