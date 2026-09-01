# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FastFlowLM model cache resolution and immutable inventory."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .commands import CommandResult, run_command
from .hashing import canonical_json_sha256, file_inventory


class ModelError(RuntimeError):
    pass


def _derive_identity(url: str) -> tuple[str, str, str | None]:
    """Derive (identity_policy, source, revision) from a registry URL.

    A Hugging Face ``/resolve/<revision>`` URL pins an immutable revision
    (``revision-sha256``); any other URL is a mutable source (``content-sha256``).
    """
    if "/resolve/" in url:
        source, revision = url.split("/resolve/", 1)
        return "revision-sha256", source, revision
    return "content-sha256", url, None


def _classify_model_kind(family: str, labels: list[str], details_family: str) -> str:
    """Classify a registry model as chat, embed, or asr (mirrors the sweep harness)."""
    if (
        "embeddings" in labels
        or family.startswith("embed")
        or details_family.startswith("embed")
    ):
        return "embed"
    if family.startswith("whisper") or details_family.startswith("whisper"):
        return "asr"
    return "chat"


def fastflow_environment(
    model_cache: Path,
    *,
    offline: bool,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(base_environment or os.environ)
    environment["FLM_MODEL_PATH"] = str(model_cache.resolve())
    if offline:
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "ROCM_NPU_QUALIFICATION_OFFLINE": "1",
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "http_proxy": "http://127.0.0.1:9",
                "https_proxy": "http://127.0.0.1:9",
                "all_proxy": "http://127.0.0.1:9",
                "NO_PROXY": "",
                "no_proxy": "",
            }
        )
    return environment


def _registry_model(
    model: dict[str, Any], config_path: Path
) -> tuple[dict[str, Any], str, str | None, str, str]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_type, size = model["tag"].split(":", 1)
        entry = config["models"][model_type][size]
        model_path = config.get("model_path", "")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ModelError(
            f"{model['tag']}: model is absent from the installed FLM registry"
        ) from error
    registry_digest = canonical_json_sha256(config)
    expected_registry = model.get("registry_sha256")
    if expected_registry is not None and registry_digest != expected_registry:
        raise ModelError(
            f"{model['tag']}: installed FLM registry does not match profile"
        )
    if not isinstance(entry, dict):
        raise ModelError(f"{model['tag']}: FLM registry entry is invalid")
    if not isinstance(model_path, str) or (
        model_path and PurePosixPath(model_path).name != model_path
    ):
        raise ModelError(f"{model['tag']}: FLM registry model path is invalid")
    url = entry.get("url")
    if not isinstance(url, str) or not url:
        raise ModelError(f"{model['tag']}: FLM registry URL is invalid")
    identity_policy, source, revision = _derive_identity(url)
    if model["identity_policy"] == "revision-sha256":
        if identity_policy != "revision-sha256":
            raise ModelError(f"{model['tag']}: FLM registry identity is mutable")
        if model["source"] != source or model["revision"] != revision:
            raise ModelError(
                f"{model['tag']}: profile identity does not match installed FLM registry"
            )
    else:
        source, revision = url, None
        if model["source"] != source or identity_policy != "content-sha256":
            raise ModelError(
                f"{model['tag']}: mutable registry source does not match profile"
            )
    return entry, source, revision, model_path, registry_digest


def _verify_expected_files(
    model: dict[str, Any],
    model_directory: Path,
    required_files: list[str],
) -> list[dict[str, object]]:
    expected = {entry["path"]: entry["sha256"] for entry in model["files"]}
    required = set(required_files)
    if set(expected) != required:
        raise ModelError(
            f"{model['tag']}: profile must pin the complete registry file set; "
            f"missing={sorted(required - set(expected))}, "
            f"extra={sorted(set(expected) - required)}"
        )
    inventory = file_inventory(model_directory)
    actual = {entry["path"]: entry["sha256"] for entry in inventory}
    missing = sorted(set(expected).union(required_files) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        path for path, digest in expected.items() if actual.get(path) != digest
    )
    if missing or unexpected or mismatched:
        raise ModelError(
            f"{model['tag']}: model content verification failed; "
            f"missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}"
        )
    if not inventory:
        raise ModelError(f"{model['tag']}: model cache contains no files")
    return inventory


def resolve_model(
    model: dict[str, Any],
    *,
    flm: Path,
    config_path: Path,
    model_cache: Path,
    pull_log: Path,
    offline: bool,
    no_pull: bool,
    environment: Mapping[str, str],
    echo: bool = False,
) -> tuple[dict[str, Any], list[CommandResult]]:
    model_cache.mkdir(parents=True, exist_ok=True)
    environment = fastflow_environment(
        model_cache, offline=offline, base_environment=environment
    )
    registry, source, revision, model_path, registry_digest = _registry_model(
        model, config_path
    )
    model_name = registry.get("name")
    required_files = registry.get("files")
    if (
        not isinstance(model_name, str)
        or not model_name
        or Path(model_name).name != model_name
        or not isinstance(required_files, list)
        or not required_files
        or not all(
            isinstance(path, str)
            and path
            and "\\" not in path
            and not PurePosixPath(path).is_absolute()
            and ".." not in PurePosixPath(path).parts
            for path in required_files
        )
    ):
        raise ModelError(f"{model['tag']}: FLM registry file contract is invalid")
    model_directory = model_cache / model_path / model_name
    commands: list[CommandResult] = []
    cached = all((model_directory / path).is_file() for path in required_files)
    pulled = False
    if not cached:
        if offline:
            raise ModelError(
                f"{model['tag']}: model is absent and --offline forbids pulling"
            )
        if no_pull:
            raise ModelError(
                f"{model['tag']}: model is absent and --no-pull was requested"
            )
    if not cached and not offline and not no_pull:
        pull = run_command(
            [flm, "pull", model["tag"]],
            log_path=pull_log,
            env=environment,
            timeout=1800,
            echo=echo,
        )
        commands.append(pull)
        if pull.returncode != 0:
            raise ModelError(
                f"{model['tag']}: flm pull failed with exit {pull.returncode}"
            )
        pulled = True
        check = run_command(
            [flm, "check", model["tag"]],
            log_path=pull_log.with_name(pull_log.stem + "-check-after.log"),
            env=environment,
            timeout=120,
            echo=echo,
        )
        commands.append(check)
        if check.returncode != 0:
            raise ModelError(f"{model['tag']}: downloaded model failed flm check")
    inventory = _verify_expected_files(model, model_directory, required_files)
    return (
        {
            "tag": model["tag"],
            "identity_policy": model["identity_policy"],
            "source": source,
            "revision": revision,
            "cache_directory": (PurePosixPath(model_path) / model_name).as_posix(),
            "registry_sha256": registry_digest,
            "pulled": pulled,
            "files": inventory,
            "status": "pass",
        },
        commands,
    )


def enumerate_registry_models(config_path: Path) -> list[dict[str, Any]]:
    """Enumerate every model in the installed FLM registry as sweep candidates.

    Unlike curated profiles, these synthetic model records are derived directly
    from ``model_list.json`` at runtime and carry no pre-pinned per-file/xclbin
    digests -- identity still comes from the registry URL. Each record carries a
    ``kind`` (chat/embed/asr), ``footprint`` and ``think`` hint so the runner can
    synthesize footprint-scaled tests. Entries without a usable URL are skipped.
    """
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        families = config["models"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ModelError("installed FLM registry is unavailable or invalid") from error
    if not isinstance(families, dict):
        raise ModelError("installed FLM registry model table is invalid")
    registry_digest = canonical_json_sha256(config)
    models: list[dict[str, Any]] = []
    for family, tags in families.items():
        if not isinstance(tags, dict):
            continue
        for size, spec in tags.items():
            if not isinstance(spec, dict):
                continue
            url = spec.get("url")
            if not isinstance(url, str) or not url:
                continue
            identity_policy, source, revision = _derive_identity(url)
            details = spec.get("details")
            if not isinstance(details, dict):
                details = {}
            labels = spec.get("label")
            if not isinstance(labels, list):
                labels = []
            details_family = str(details.get("family", family) or family)
            kind = _classify_model_kind(str(family), labels, details_family)
            try:
                footprint = float(spec.get("footprint", 0) or 0)
            except (TypeError, ValueError):
                footprint = 0.0
            models.append(
                {
                    "tag": f"{family}:{size}",
                    "source": source,
                    "revision": revision,
                    "identity_policy": identity_policy,
                    "registry_sha256": registry_digest,
                    "xclbins": [],
                    "kind": kind,
                    "footprint": footprint,
                    "think": bool(details.get("think", False)),
                }
            )
    if not models:
        raise ModelError("installed FLM registry contains no usable models")
    return models


def resolve_model_dynamic(
    model: dict[str, Any],
    *,
    flm: Path,
    config_path: Path,
    model_cache: Path,
    pull_log: Path,
    offline: bool,
    no_pull: bool,
    environment: Mapping[str, str],
    echo: bool = False,
) -> tuple[dict[str, Any], list[CommandResult]]:
    """Resolve a model without pre-pinned digests (complete-sweep discovery).

    Mirrors :func:`resolve_model` but records the on-disk inventory that ``flm
    pull`` produced instead of comparing against profile-pinned file hashes.
    Identity still comes from the installed FLM registry, and the returned record
    satisfies the evidence ``model`` schema.
    """
    model_cache.mkdir(parents=True, exist_ok=True)
    environment = fastflow_environment(
        model_cache, offline=offline, base_environment=environment
    )
    registry, source, revision, model_path, registry_digest = _registry_model(
        model, config_path
    )
    model_name = registry.get("name")
    required_files = registry.get("files")
    if (
        not isinstance(model_name, str)
        or not model_name
        or Path(model_name).name != model_name
        or not isinstance(required_files, list)
        or not required_files
        or not all(
            isinstance(path, str)
            and path
            and "\\" not in path
            and not PurePosixPath(path).is_absolute()
            and ".." not in PurePosixPath(path).parts
            for path in required_files
        )
    ):
        raise ModelError(f"{model['tag']}: FLM registry file contract is invalid")
    model_directory = model_cache / model_path / model_name
    commands: list[CommandResult] = []
    cached = all((model_directory / path).is_file() for path in required_files)
    pulled = False
    if not cached:
        if offline:
            raise ModelError(
                f"{model['tag']}: model is absent and --offline forbids pulling"
            )
        if no_pull:
            raise ModelError(
                f"{model['tag']}: model is absent and --no-pull was requested"
            )
    if not cached and not offline and not no_pull:
        pull = run_command(
            [flm, "pull", model["tag"]],
            log_path=pull_log,
            env=environment,
            timeout=1800,
            echo=echo,
        )
        commands.append(pull)
        if pull.returncode != 0:
            raise ModelError(
                f"{model['tag']}: flm pull failed with exit {pull.returncode}"
            )
        pulled = True
        check = run_command(
            [flm, "check", model["tag"]],
            log_path=pull_log.with_name(pull_log.stem + "-check-after.log"),
            env=environment,
            timeout=120,
            echo=echo,
        )
        commands.append(check)
        if check.returncode != 0:
            raise ModelError(f"{model['tag']}: downloaded model failed flm check")
    inventory = file_inventory(model_directory)
    if not inventory:
        raise ModelError(f"{model['tag']}: model cache contains no files")
    present = {entry["path"] for entry in inventory}
    required = set(required_files)
    missing = sorted(required - present)
    # Discovery has no per-file digests to pin, but the registry still declares
    # the complete file set for the model. Mirror resolve_model and require the
    # cache to match it exactly: extra files mean the cache is not the model FLM
    # published, so record neither them nor a "pass".
    unexpected = sorted(present - required)
    if missing or unexpected:
        raise ModelError(
            f"{model['tag']}: model cache does not match the registry file set; "
            f"missing required files {missing}, unexpected files {unexpected}"
        )
    return (
        {
            "tag": model["tag"],
            "identity_policy": model["identity_policy"],
            "source": source,
            "revision": revision,
            "cache_directory": (PurePosixPath(model_path) / model_name).as_posix(),
            "registry_sha256": registry_digest,
            "pulled": pulled,
            "files": inventory,
            "status": "pass",
        },
        commands,
    )
