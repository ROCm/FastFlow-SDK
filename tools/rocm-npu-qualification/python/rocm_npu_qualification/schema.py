# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small standard-library JSON Schema subset used by offline qualification."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from importlib import resources
from typing import Any


class SchemaValidationError(ValueError):
    pass


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        ),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(expected, False)


def _resolve_reference(
    root: dict[str, Any], reference: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_root = root
    fragment = reference
    if not reference.startswith("#"):
        document, separator, fragment_value = reference.partition("#")
        name = document.rsplit("/", 1)[-1]
        try:
            target_root = load_packaged_schema(name)
        except (OSError, json.JSONDecodeError) as error:
            raise SchemaValidationError(
                f"unresolved schema reference {reference!r}"
            ) from error
        fragment = f"#{fragment_value}" if separator else "#"
    if fragment == "#":
        return target_root, target_root
    if not fragment.startswith("#/"):
        raise SchemaValidationError(f"unsupported schema reference {reference!r}")
    value: Any = target_root
    for part in fragment[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or part not in value:
            raise SchemaValidationError(f"unresolved schema reference {reference!r}")
        value = value[part]
    if not isinstance(value, dict):
        raise SchemaValidationError(f"invalid schema reference {reference!r}")
    return value, target_root


def _validate(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        resolved, resolved_root = _resolve_reference(root, schema["$ref"])
        _validate(value, resolved, resolved_root, path)
    if "not" in schema:
        try:
            _validate(value, schema["not"], root, path)
        except SchemaValidationError:
            pass
        else:
            raise SchemaValidationError(f"{path} matches a forbidden schema")
    for keyword, expected_matches in (("oneOf", 1), ("anyOf", None)):
        choices = schema.get(keyword)
        if isinstance(choices, list):
            matches = 0
            for choice in choices:
                try:
                    _validate(value, choice, root, path)
                except SchemaValidationError:
                    continue
                matches += 1
            if (expected_matches is None and matches == 0) or (
                expected_matches is not None and matches != expected_matches
            ):
                raise SchemaValidationError(f"{path} does not satisfy {keyword}")
    expected = schema.get("type")
    if expected is not None:
        choices = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, choice) for choice in choices):
            raise SchemaValidationError(f"{path} has invalid type")
    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path} does not equal the required constant")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path} is not an allowed value")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise SchemaValidationError(f"{path} is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise SchemaValidationError(f"{path} is too long")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            raise SchemaValidationError(f"{path} does not match its pattern")
        if schema.get("format") == "date-time":
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise SchemaValidationError(
                    f"{path} is not an ISO date-time"
                ) from error
            if parsed.tzinfo is None:
                raise SchemaValidationError(f"{path} date-time has no timezone")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path} is below its minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path} exceeds its maximum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise SchemaValidationError(f"{path} is below its exclusive minimum")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise SchemaValidationError(f"{path} exceeds its exclusive maximum")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise SchemaValidationError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise SchemaValidationError(f"{path} has too many items")
        if schema.get("uniqueItems"):
            for index, item in enumerate(value):
                if item in value[:index]:
                    raise SchemaValidationError(f"{path} has duplicate items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate(item, item_schema, root, f"{path}[{index}]")
    if isinstance(value, dict):
        if len(value) < schema.get("minProperties", 0):
            raise SchemaValidationError(f"{path} has too few properties")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            raise SchemaValidationError(f"{path} has too many properties")
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise SchemaValidationError(f"{path} is missing {missing}")
        properties = schema.get("properties", {})
        for key, child in properties.items():
            if key in value:
                _validate(value[key], child, root, f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise SchemaValidationError(
                    f"{path} has unexpected properties {extra}"
                )
        elif isinstance(schema.get("additionalProperties"), dict):
            for key in set(value) - set(properties):
                _validate(
                    value[key],
                    schema["additionalProperties"],
                    root,
                    f"{path}.{key}",
                )
    for child in schema.get("allOf", []):
        _validate(value, child, root, path)
    condition = schema.get("if")
    if isinstance(condition, dict):
        try:
            _validate(value, condition, root, path)
            matches = True
        except SchemaValidationError:
            matches = False
        branch = schema.get("then" if matches else "else")
        if isinstance(branch, dict):
            _validate(value, branch, root, path)


def validate_schema(value: Any, schema: dict[str, Any]) -> None:
    _validate(value, schema, schema, "$")


def load_packaged_schema(name: str) -> dict[str, Any]:
    resource = resources.files("rocm_npu_qualification").joinpath(
        "schemas", name
    )
    return json.loads(resource.read_text(encoding="utf-8"))
