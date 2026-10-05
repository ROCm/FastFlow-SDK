# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Versioned qualification profile loading and validation."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from importlib import resources
from pathlib import PurePosixPath
from typing import Any

from .contracts import PROFILE_SCHEMA_VERSION


class ProfileError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileError(message)


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def _valid_prompt_generator(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "kind",
            "filler_word_count",
            "marker_word_offset",
            "marker",
        }
        and value.get("schema_version") == 1
        and value.get("kind") == "context-tail-marker-v1"
        and type(value.get("filler_word_count")) is int
        and 1000 <= value["filler_word_count"] <= 10000
        and type(value.get("marker_word_offset")) is int
        and 0 <= value["marker_word_offset"] < value["filler_word_count"]
        and isinstance(value.get("marker"), str)
        and re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)+", value["marker"]) is not None
    )


def _validate_digest_records(tag: str, label: str, value: Any) -> None:
    _require(isinstance(value, list), f"{tag}: {label} must be an array")
    seen: set[str] = set()
    for record in value:
        _require(isinstance(record, dict), f"{tag}: {label} record is invalid")
        _require(
            set(record) == {"path", "sha256"},
            f"{tag}: {label} properties are invalid",
        )
        path = record.get("path")
        _require(_safe_relative_path(path), f"{tag}: {label} path is invalid")
        _require(path not in seen, f"{tag}: duplicate {label} path")
        seen.add(path)
        _require(
            re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) is not None,
            f"{tag}: {label} SHA256 is invalid",
        )


def validate_profile(profile: Any) -> dict[str, Any]:
    _require(isinstance(profile, dict), "profile must be an object")
    _require(
        set(profile) == {"schema_version", "name", "description", "models"},
        "profile properties are invalid",
    )
    _require(
        profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "unsupported profile schema",
    )
    _require(
        isinstance(profile.get("name"), str)
        and re.fullmatch(r"[a-z0-9-]+", profile["name"]) is not None,
        "profile name is invalid",
    )
    _require(isinstance(profile.get("description"), str), "description is required")
    models = profile.get("models")
    _require(isinstance(models, list) and models, "profile models must be non-empty")
    seen_models: set[str] = set()
    for model in models:
        _require(isinstance(model, dict), "model entry must be an object")
        _require(
            set(model)
            == {
                "tag",
                "source",
                "revision",
                "identity_policy",
                "registry_sha256",
                "files",
                "xclbins",
                "tests",
            },
            "model properties are invalid",
        )
        tag = model.get("tag")
        _require(isinstance(tag, str) and tag, "model tag is required")
        _require(tag not in seen_models, f"duplicate model tag: {tag}")
        seen_models.add(tag)
        _require(
            isinstance(model.get("source"), str) and model["source"],
            f"{tag}: source is required",
        )
        policy = model.get("identity_policy")
        _require(
            policy in {"revision-sha256", "content-sha256"},
            f"{tag}: identity policy is invalid",
        )
        revision = model.get("revision")
        _require(
            (policy == "revision-sha256" and isinstance(revision, str) and revision)
            or (policy == "content-sha256" and revision is None),
            f"{tag}: revision does not match identity policy",
        )
        _require(
            re.fullmatch(r"[0-9a-f]{64}", str(model.get("registry_sha256", "")))
            is not None,
            f"{tag}: registry SHA256 is invalid",
        )
        _validate_digest_records(tag, "files", model.get("files"))
        _require(model["files"], f"{tag}: files must be non-empty")
        _validate_digest_records(tag, "xclbins", model.get("xclbins"))
        tests = model.get("tests")
        _require(isinstance(tests, list) and tests, f"{tag}: tests are required")
        seen_tests: set[str] = set()
        for test in tests:
            _require(isinstance(test, dict), f"{tag}: test entry is invalid")
            name = test.get("name")
            _require(isinstance(name, str) and name, f"{tag}: test name is required")
            _require(name not in seen_tests, f"{tag}: duplicate test: {name}")
            seen_tests.add(name)
            kind = test.get("kind")
            _require(kind in {"run", "bench"}, f"{tag}/{name}: kind is invalid")
            if kind == "run":
                _require(
                    set(test)
                    == {
                        "name",
                        "kind",
                        "turns",
                        "setup_commands",
                        "process_iterations",
                        "timeout_seconds",
                        "ctx_len",
                    },
                    f"{tag}/{name}: run properties are invalid",
                )
                turns = test.get("turns")
                _require(
                    isinstance(turns, list) and 1 <= len(turns) <= 32,
                    f"{tag}/{name}: turns are required",
                )
                for turn in turns:
                    prompt_keys = (
                        {"prompt", "prompt_generator"} & set(turn)
                        if isinstance(turn, dict)
                        else set()
                    )
                    _require(
                        isinstance(turn, dict)
                        and set(turn)
                        <= {
                            "prompt",
                            "prompt_generator",
                            "expected",
                            "expected_regex",
                        }
                        and len(prompt_keys) == 1
                        and (
                            (
                                isinstance(turn.get("prompt"), str)
                                and bool(turn["prompt"])
                            )
                            or _valid_prompt_generator(turn.get("prompt_generator"))
                        )
                        and not ("expected" in turn and "expected_regex" in turn),
                        f"{tag}/{name}: turn is invalid",
                    )
                    for field in ("expected", "expected_regex"):
                        if field in turn:
                            _require(
                                isinstance(turn[field], str) and turn[field],
                                f"{tag}/{name}: {field} is invalid",
                            )
                    if "expected_regex" in turn:
                        try:
                            re.compile(turn["expected_regex"])
                        except re.error as error:
                            raise ProfileError(
                                f"{tag}/{name}: expected regex is invalid"
                            ) from error
                _require(
                    isinstance(test["setup_commands"], list)
                    and all(
                        isinstance(command, str) and command
                        for command in test["setup_commands"]
                    ),
                    f"{tag}/{name}: setup commands are invalid",
                )
                _require(
                    isinstance(test.get("process_iterations"), int)
                    and 1 <= test["process_iterations"] <= 100,
                    f"{tag}/{name}: process iterations are invalid",
                )
                _require(
                    isinstance(test.get("ctx_len"), int)
                    and 1 <= test["ctx_len"] <= 131072,
                    f"{tag}/{name}: context length is invalid",
                )
            else:
                allowed_properties = {
                    "name",
                    "kind",
                    "config",
                    "timeout_seconds",
                    "checkpoint_min",
                    "checkpoint_max",
                    "expected_regex",
                }
                _require(
                    set(test)
                    in (allowed_properties, allowed_properties | {"iterations"}),
                    f"{tag}/{name}: bench properties are invalid",
                )
                _require(
                    _safe_relative_path(test.get("config")),
                    f"{tag}/{name}: bench config is invalid",
                )
                _require(
                    isinstance(test.get("checkpoint_min"), int)
                    and isinstance(test.get("checkpoint_max"), int)
                    and 1 <= test["checkpoint_min"] <= test["checkpoint_max"],
                    f"{tag}/{name}: checkpoint range is invalid",
                )
                _require(
                    isinstance(test.get("expected_regex"), str),
                    f"{tag}/{name}: expected regex is invalid",
                )
                if test["expected_regex"]:
                    try:
                        re.compile(test["expected_regex"])
                    except re.error as error:
                        raise ProfileError(
                            f"{tag}/{name}: expected regex is invalid"
                        ) from error
                _require(
                    type(test.get("iterations", 1)) is int
                    and 1 <= test.get("iterations", 1) <= 100,
                    f"{tag}/{name}: benchmark iterations are invalid",
                )
            _require(
                isinstance(test.get("timeout_seconds"), int)
                and 1 <= test["timeout_seconds"] <= 7200,
                f"{tag}/{name}: timeout is invalid",
            )
    return profile


def profile_names() -> list[str]:
    root = resources.files("fastflow_sdk_qualification").joinpath("profiles")
    return sorted(
        path.name[:-5] for path in root.iterdir() if path.name.endswith(".json")
    )


def merge_inherited_profile(
    value: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    """Materialize an `extends` profile, including `test_extensions`.

    Candidate contract generation must use this same merge so sealed evidence
    profiles match the embedded qualification_contract.
    """
    allowed_properties = {
        "schema_version",
        "name",
        "description",
        "extends",
        "models",
    }
    _require(
        set(value) in (allowed_properties, allowed_properties | {"test_extensions"}),
        "inherited profile properties are invalid",
    )
    _require(
        value["schema_version"] == PROFILE_SCHEMA_VERSION,
        "unsupported profile schema",
    )
    merged = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "name": value["name"],
        "description": value["description"],
        "models": [*deepcopy(parent["models"]), *deepcopy(value.get("models", []))],
    }
    models_by_tag = {model["tag"]: model for model in merged["models"]}
    test_extensions = value.get("test_extensions", [])
    _require(
        isinstance(test_extensions, list),
        "inherited profile test extensions must be an array",
    )
    extended_models: set[str] = set()
    for extension in test_extensions:
        _require(
            isinstance(extension, dict) and set(extension) == {"model", "tests"},
            "inherited profile test extension is invalid",
        )
        model_tag = extension.get("model")
        _require(
            isinstance(model_tag, str) and model_tag in models_by_tag,
            "inherited profile test extension references an unknown model",
        )
        _require(
            model_tag not in extended_models,
            "inherited profile contains duplicate model test extensions",
        )
        extended_models.add(model_tag)
        tests = extension.get("tests")
        _require(
            isinstance(tests, list) and tests,
            "inherited profile test extension must contain tests",
        )
        models_by_tag[model_tag]["tests"].extend(deepcopy(tests))
    return validate_profile(merged)


def materialize_profiles(raw_profiles: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve a profile name→raw-JSON map the same way installed profiles load."""
    loading: set[str] = set()

    def resolve(name: str) -> dict[str, Any]:
        if name in loading or name not in raw_profiles:
            raise ProfileError(f"qualification profile inheritance is invalid: {name}")
        value = raw_profiles[name]
        if isinstance(value, dict) and "extends" in value:
            parent_name = value["extends"]
            _require(
                isinstance(parent_name, str) and parent_name != name,
                "profile parent is invalid",
            )
            _require(value.get("name") == name, "profile name does not match filename")
            loading.add(name)
            parent = resolve(parent_name)
            loading.remove(name)
            return merge_inherited_profile(value, parent)
        profile = validate_profile(value)
        _require(profile["name"] == name, "profile name does not match filename")
        return profile

    return {name: resolve(name) for name in sorted(raw_profiles)}


def _load_profile(name: str, loading: set[str]) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ProfileError("profile name is invalid")
    if name in loading:
        raise ProfileError(f"cyclic qualification profile inheritance: {name}")
    root = resources.files("fastflow_sdk_qualification").joinpath("profiles")
    path = root.joinpath(f"{name}.json")
    if not path.is_file():
        raise ProfileError(f"unknown qualification profile: {name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "extends" in value:
        parent_name = value["extends"]
        _require(
            isinstance(parent_name, str) and parent_name != name,
            "profile parent is invalid",
        )
        _require(value.get("name") == name, "profile name does not match filename")
        parent = _load_profile(parent_name, loading | {name})
        return merge_inherited_profile(value, parent)
    profile = validate_profile(value)
    _require(profile["name"] == name, "profile name does not match filename")
    return profile


def load_profile(name: str) -> dict[str, Any]:
    return _load_profile(name, set())
