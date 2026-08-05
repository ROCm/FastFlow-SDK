# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Small standard-library JSON Schema subset used by offline qualification."""

from __future__ import annotations

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
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(expected, False)


def _resolve_reference(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise SchemaValidationError(f"unsupported schema reference {reference!r}")
    value: Any = root
    for part in reference[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or part not in value:
            raise SchemaValidationError(f"unresolved schema reference {reference!r}")
        value = value[part]
    if not isinstance(value, dict):
        raise SchemaValidationError(f"invalid schema reference {reference!r}")
    return value


def _validate(
    value: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        _validate(value, _resolve_reference(root, schema["$ref"]), root, path)
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
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise SchemaValidationError(f"{path} has too few items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate(item, item_schema, root, f"{path}[{index}]")
    if isinstance(value, dict):
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
    import json

    resource = resources.files("rocm_npu_qualification").joinpath(
        "schemas", name
    )
    return json.loads(resource.read_text(encoding="utf-8"))
