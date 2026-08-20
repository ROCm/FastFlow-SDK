# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test qualification profiles, execution, and evidence generation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[3]
QUALIFICATION_PYTHON = ROOT / "tools" / "rocm-npu-qualification" / "python"

from rocm_npu_qualification import (  # noqa: E402
    benchmark,
    cli,
    installation,
    models,
    power,
    prompts,
    report,
    runner,
)
from rocm_npu_qualification.commands import (  # noqa: E402
    CommandResult,
    _transcript_contains_phrase,
    _wav_duration_seconds,
    _word_error_rate,
    redact,
    run_command,
    serve_probe,
)
from rocm_npu_qualification.hashing import canonical_json_sha256  # noqa: E402
from rocm_npu_qualification import offline  # noqa: E402
from rocm_npu_qualification.offline import EvidenceError, pack_evidence  # noqa: E402
from rocm_npu_qualification.evidence import EvidenceWriter  # noqa: E402
from rocm_npu_qualification.fastflowlm import (  # noqa: E402
    require_1k_checkpoint,
    require_benchmark_schedule,
    verify_xclbin_inventory,
)
from rocm_npu_qualification.hashing import file_inventory  # noqa: E402
from rocm_npu_qualification.numeric_runtime import NUMERIC_RUNTIME  # noqa: E402
from rocm_npu_qualification.profiles import (  # noqa: E402
    ProfileError,
    load_profile,
    validate_profile,
)

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def test_inhibit_system_sleep_is_noop_off_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(power.os, "name", "posix")
    monkeypatch.setattr(
        power, "_set_thread_execution_state", lambda flags: calls.append(flags)
    )

    with power.inhibit_system_sleep():
        pass

    assert calls == []


def test_inhibit_system_sleep_restores_windows_execution_state(monkeypatch):
    calls = []
    monkeypatch.setattr(power.os, "name", "nt")
    monkeypatch.setattr(
        power,
        "_set_thread_execution_state",
        lambda flags: calls.append(flags) or 1,
    )

    with power.inhibit_system_sleep():
        assert calls == [power._ES_CONTINUOUS | power._ES_SYSTEM_REQUIRED]

    assert calls == [
        power._ES_CONTINUOUS | power._ES_SYSTEM_REQUIRED,
        power._ES_CONTINUOUS,
    ]


def test_inhibit_system_sleep_restores_after_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(power.os, "name", "nt")
    monkeypatch.setattr(
        power,
        "_set_thread_execution_state",
        lambda flags: calls.append(flags) or 1,
    )

    with pytest.raises(RuntimeError, match="qualification failed"):
        with power.inhibit_system_sleep():
            raise RuntimeError("qualification failed")

    assert calls[-1] == power._ES_CONTINUOUS


def _result(returncode=0, stdout="ok\n"):
    return CommandResult(
        argv=("flm",),
        returncode=returncode,
        stdout=stdout,
        stderr="",
        duration_seconds=0.1,
        timed_out=False,
        log="log",
    )


def _model():
    return {
        "tag": "qwen3:0.6b",
        "source": "https://test",
        "revision": "immutable",
        "identity_policy": "revision-sha256",
        "files": [
            {
                "path": "weights.bin",
                "sha256": hashlib.sha256(b"model").hexdigest(),
            }
        ],
        "xclbins": [],
        "tests": [
            {
                "name": "basic",
                "kind": "run",
                "turns": [{"prompt": "hello"}],
                "setup_commands": [],
                "process_iterations": 1,
                "timeout_seconds": 10,
                "ctx_len": 1024,
            }
        ],
    }


def _config(root: Path) -> Path:
    path = root / "model_list.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "qwen3": {
                        "0.6b": {
                            "name": "model",
                            "url": "https://test/resolve/immutable",
                            "files": ["weights.bin"],
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_plugin_api_mismatch(monkeypatch, tmp_path):
    monkeypatch.setenv("ROCM_NPU_PLUGIN_API", "99")
    monkeypatch.setenv("ROCM_NPU_CORE_EXECUTABLE", str(tmp_path / "core"))
    with pytest.raises(cli.QualificationError, match="unsupported"):
        installation.validate_protocol()


def test_package_version_mismatch():
    with pytest.raises(cli.QualificationError, match="do not match"):
        installation.validate_versions(
            {
                "rocm-npu-runtime": "1.0",
                "rocm-npu-fastflowlm": "2.0",
                "rocm-npu-qualification": "1.0",
            },
            "1.0",
            None,
        )


def test_profile_schema_rejection_and_packaged_profiles():
    invalid = _model()
    with pytest.raises(ProfileError):
        validate_profile({"schema_version": 1, "name": "../bad", "models": [invalid]})
    assert load_profile("fast")["models"][0]["tag"] == "qwen3:0.6b"
    assert [model["tag"] for model in load_profile("default")["models"]] == [
        "qwen3:0.6b",
        "qwen3.5:0.8b",
        "lfm2:1.2b",
    ]
    extended = load_profile("extended")
    assert extended["models"][-1]["tag"] == "llama3.1:8b"
    lfm_tests = {
        test["name"]: test
        for model in extended["models"]
        if model["tag"] == "lfm2:1.2b"
        for test in model["tests"]
    }
    assert lfm_tests["long-lived-context-1k"]["iterations"] == 100


def test_profile_selection_and_fast_parity_contract():
    parser = cli.create_parser()
    assert runner._selected_profile_name(parser.parse_args([])) == "default"
    assert runner._selected_profile_name(parser.parse_args(["--fast"])) == "fast"
    assert (
        runner._selected_profile_name(parser.parse_args(["--profile", "extended"]))
        == "extended"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--fast", "--profile", "default"])

    fast = load_profile("fast")
    qwen = fast["models"][0]
    run = next(test for test in qwen["tests"] if test["kind"] == "run")
    bench = next(test for test in qwen["tests"] if test["kind"] == "bench")
    assert qwen["tag"] == "qwen3:0.6b"
    assert run["process_iterations"] == 5
    assert len(run["turns"]) == 2
    assert run["setup_commands"] == [
        "/set topk 1",
        "/set temp 0",
        "/set gen-lim 16",
    ]
    assert (bench["checkpoint_min"], bench["checkpoint_max"]) == (950, 1024)
    assert {entry["path"] for entry in qwen["xclbins"]} == {
        f"Qwen3-0.6B-NPU2/{name}.xclbin" for name in ("attn", "dequant", "layer", "mm")
    }


def test_profile_registry_digest_matches_fastflowlm_source():
    registry_path = ROOT / "projects" / "FastFlowLM" / "src" / "model_list.json"
    if not registry_path.is_file():
        pytest.skip("FastFlowLM sources are not initialized")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    profile_digest = load_profile("fast")["models"][0]["registry_sha256"]
    assert profile_digest == canonical_json_sha256(registry)


@pytest.mark.skipif(os.name != "nt", reason="Windows DLL search-path contract")
def test_resolve_flm_prefers_authoritative_runtime_override(monkeypatch, tmp_path):
    package_root = tmp_path / "fastflowlm"
    flm = package_root / "bin" / "flm.exe"
    config = package_root / "model_list.json"
    xclbins = package_root / "xclbins"
    model_info = package_root / "model_info.json"
    runtime_bin = tmp_path / "installed-runtime" / "bin"
    candidate = tmp_path / "candidate" / "hrx.dll"
    for path in (flm, config, model_info, candidate):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    xclbins.mkdir()
    runtime_bin.mkdir(parents=True)

    monkeypatch.setitem(
        sys.modules,
        "rocm_npu_fastflowlm",
        SimpleNamespace(
            flm_path=lambda: flm,
            config_path=lambda: config,
            xclbin_path=lambda: xclbins,
            model_info_path=lambda: model_info,
            package_root=lambda: package_root,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "rocm_npu_runtime",
        SimpleNamespace(binary_dir=lambda: runtime_bin),
    )
    monkeypatch.setenv("ROCM_NPU_RUNTIME_PATH", str(candidate))
    monkeypatch.setenv("PATH", "existing-path")

    resolved_flm, resolved_config, environment = runner._resolve_flm()

    assert resolved_flm == flm.resolve()
    assert resolved_config == config.resolve()
    assert environment["PATH"].split(os.pathsep) == [
        str(package_root / "bin"),
        str(candidate.parent),
        str(runtime_bin),
        "existing-path",
    ]


def test_4k_profiles_separate_benchmark_from_tail_retrieval():
    models = {
        model["tag"]: model
        for profile in (load_profile("default"), load_profile("extended"))
        for model in profile["models"]
    }
    for tag in ("lfm2:1.2b", "llama3.1:8b"):
        tests = {test["name"]: test for test in models[tag]["tests"]}
        assert tests["context-4k"]["expected_regex"] == ""
        retrieval = tests["context-4k-retrieval"]
        turn = prompts.materialize_turn(retrieval["turns"][0])
        assert retrieval["ctx_len"] == 4096
        assert turn["prompt"].startswith(
            "This is an exact-copy test, not a summarization task."
        )
        assert "HRX-NPU-CONTEXT-4K-PASS" in turn["prompt"]
        assert turn["prompt"].endswith(
            "Reply with only its hyphenated marker value, exactly as written."
        )
        assert len(turn["prompt"].split()) > 3200

    bench = json.loads(
        (
            QUALIFICATION_PYTHON
            / "rocm_npu_qualification"
            / "bench"
            / "bench_4k_smoke.json"
        ).read_text(encoding="utf-8")
    )
    assert bench["max_length"] == 4096
    assert "Begin your response with exactly HRX-NPU-4K-PASS." in bench["input_text"]
    assert len(set(bench["input_text"].split())) > 100
    assert "alpha alpha alpha" not in bench["input_text"]


def test_profile_rejects_unbounded_soak_and_bad_regex():
    profile = copy.deepcopy(load_profile("fast"))
    run = next(test for test in profile["models"][0]["tests"] if test["kind"] == "run")
    run["process_iterations"] = 101
    with pytest.raises(ProfileError, match="process iterations"):
        validate_profile(profile)
    run["process_iterations"] = 1
    run["turns"][0] = {"prompt": "hello", "expected_regex": "("}
    with pytest.raises(ProfileError, match="expected regex"):
        validate_profile(profile)
    run["turns"][0] = {
        "prompt_generator": {
            "schema_version": 1,
            "kind": "context-tail-marker-v1",
            "filler_word_count": 999,
            "marker_word_offset": 900,
            "marker": "INVALID",
        }
    }
    with pytest.raises(ProfileError, match="turn is invalid"):
        validate_profile(profile)

    bench = next(
        test
        for test in load_profile("fast")["models"][0]["tests"]
        if test["kind"] == "bench"
    )
    profile = copy.deepcopy(load_profile("fast"))
    profile_bench = next(
        test for test in profile["models"][0]["tests"] if test["kind"] == "bench"
    )
    profile_bench["iterations"] = 101
    with pytest.raises(ProfileError, match="benchmark iterations"):
        validate_profile(profile)
    assert "iterations" not in bench


def test_xclbin_inventory_uses_installed_package_layout(tmp_path):
    package_xclbins = tmp_path / "share" / "flm" / "xclbins"
    xclbin = package_xclbins / "family" / "kernel.xclbin"
    xclbin.parent.mkdir(parents=True)
    xclbin.write_bytes(b"production kernel")
    model = _model()
    model["xclbins"] = [
        {
            "path": "family/kernel.xclbin",
            "sha256": hashlib.sha256(b"production kernel").hexdigest(),
        }
    ]
    model["tests"] = []

    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=True,
        writer=EvidenceWriter(tmp_path / "evidence"),
        environment={"FLM_XCLBIN_PATH": str(package_xclbins.parent)},
    )

    assert records[0]["name"] == "production-xclbin-inventory"
    assert records[0]["status"] == "pass"


def test_content_pinned_mutable_registry(monkeypatch, tmp_path):
    model = _model()
    model.update(
        {
            "source": "https://mutable.test/model",
            "revision": None,
            "identity_policy": "content-sha256",
        }
    )
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "weights.bin").write_bytes(b"model")
    config = _config(tmp_path)
    value = json.loads(config.read_text(encoding="utf-8"))
    value["models"]["qwen3"]["0.6b"]["url"] = model["source"]
    config.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(models, "run_command", lambda *args, **kwargs: _result())
    record, commands = models.resolve_model(
        model,
        flm=Path("flm"),
        config_path=config,
        model_cache=tmp_path,
        pull_log=tmp_path / "pull.log",
        offline=True,
        no_pull=True,
        environment={},
    )
    assert commands == []
    assert record["identity_policy"] == "content-sha256"
    assert record["revision"] is None


def test_model_inventory_must_completely_pin_registry(tmp_path):
    model = _model()
    model["files"] = []
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "weights.bin").write_bytes(b"model")
    with pytest.raises(models.ModelError, match="complete registry file set"):
        models.resolve_model(
            model,
            flm=Path("flm"),
            config_path=_config(tmp_path),
            model_cache=tmp_path,
            pull_log=tmp_path / "pull.log",
            offline=True,
            no_pull=True,
            environment={},
        )


def test_model_inventory_rejects_unexpected_cached_file(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "weights.bin").write_bytes(b"model")
    (tmp_path / "model" / "unapproved.json").write_text("{}")
    with pytest.raises(models.ModelError, match="unexpected=.*unapproved.json"):
        models.resolve_model(
            _model(),
            flm=Path("flm"),
            config_path=_config(tmp_path),
            model_cache=tmp_path,
            pull_log=tmp_path / "pull.log",
            offline=True,
            no_pull=True,
            environment={},
        )


def test_automatic_pull_of_missing_model(monkeypatch, tmp_path):
    results = iter((_result(0, "pulled"), _result(0)))
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(map(str, argv)))
        if len(calls) == 1:
            (tmp_path / "model").mkdir()
            (tmp_path / "model" / "weights.bin").write_bytes(b"model")
        return next(results)

    monkeypatch.setattr(models, "run_command", fake_run)
    record, _ = models.resolve_model(
        _model(),
        flm=Path("flm"),
        config_path=_config(tmp_path),
        model_cache=tmp_path,
        pull_log=tmp_path / "pull.log",
        offline=False,
        no_pull=False,
        environment={},
    )
    assert [call[1] for call in calls] == ["pull", "check"]
    assert record["pulled"] is True
    assert record["files"][0]["path"] == "weights.bin"


def test_cached_model_is_verified_without_network_pull(monkeypatch, tmp_path):
    (tmp_path / "unrelated-model.bin").write_bytes(b"unrelated")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "weights.bin").write_bytes(b"model")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(map(str, argv)))
        return _result()

    monkeypatch.setattr(models, "run_command", fake_run)
    record, _ = models.resolve_model(
        _model(),
        flm=Path("flm"),
        config_path=_config(tmp_path),
        model_cache=tmp_path,
        pull_log=tmp_path / "pull.log",
        offline=False,
        no_pull=False,
        environment={},
    )
    assert calls == []
    assert record["pulled"] is False
    assert [file["path"] for file in record["files"]] == ["weights.bin"]


def test_registry_digest_is_approved_by_profile(tmp_path):
    model = _model()
    model["registry_sha256"] = "0" * 64
    with pytest.raises(models.ModelError, match="registry does not match profile"):
        models.resolve_model(
            model,
            flm=Path("flm"),
            config_path=_config(tmp_path),
            model_cache=tmp_path,
            pull_log=tmp_path / "pull.log",
            offline=True,
            no_pull=True,
            environment={},
        )


def test_registry_digest_canonicalizes_json_formatting(tmp_path):
    model = _model()
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "weights.bin").write_bytes(b"model")
    config = _config(tmp_path)
    value = json.loads(config.read_text(encoding="utf-8"))
    model["registry_sha256"] = canonical_json_sha256(value)
    config.write_bytes(
        json.dumps(value, indent=2).replace("\n", "\r\n").encode("utf-8")
    )

    record, commands = models.resolve_model(
        model,
        flm=Path("flm"),
        config_path=config,
        model_cache=tmp_path,
        pull_log=tmp_path / "pull.log",
        offline=True,
        no_pull=True,
        environment={},
    )

    assert commands == []
    assert record["registry_sha256"] == model["registry_sha256"]


@pytest.mark.parametrize(
    ("offline", "no_pull", "message"),
    ((True, False, "--offline"), (False, True, "--no-pull")),
)
def test_missing_model_pull_prohibitions(
    monkeypatch, tmp_path, offline, no_pull, message
):
    monkeypatch.setattr(models, "run_command", lambda *args, **kwargs: _result(1))
    with pytest.raises(models.ModelError, match=message):
        models.resolve_model(
            _model(),
            flm=Path("flm"),
            config_path=_config(tmp_path),
            model_cache=tmp_path,
            pull_log=tmp_path / "pull.log",
            offline=offline,
            no_pull=no_pull,
            environment={},
        )


def test_offline_overrides_all_proxy_spellings(monkeypatch, tmp_path):
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.setenv(name, "https://network.example")
    environment = models.fastflow_environment(tmp_path, offline=True)
    assert not any(
        value == "https://network.example"
        for name, value in environment.items()
        if "proxy" in name.lower()
    )


def test_model_cache_environment_is_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    environment = models.fastflow_environment(Path("relative-cache"), offline=False)
    assert environment["FLM_MODEL_PATH"] == str((tmp_path / "relative-cache").resolve())


def test_pull_failure(monkeypatch, tmp_path):
    results = iter((_result(7),))
    monkeypatch.setattr(models, "run_command", lambda *args, **kwargs: next(results))
    with pytest.raises(models.ModelError, match="pull failed"):
        models.resolve_model(
            _model(),
            flm=Path("flm"),
            config_path=_config(tmp_path),
            model_cache=tmp_path,
            pull_log=tmp_path / "pull.log",
            offline=False,
            no_pull=False,
            environment={},
        )


def test_model_hash_mismatch(monkeypatch, tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "weights.bin").write_bytes(b"wrong")
    model = _model()
    model["files"] = [{"path": "weights.bin", "sha256": "0" * 64}]
    monkeypatch.setattr(models, "run_command", lambda *args, **kwargs: _result())
    with pytest.raises(models.ModelError, match="verification failed"):
        models.resolve_model(
            model,
            flm=Path("flm"),
            config_path=_config(tmp_path),
            model_cache=tmp_path,
            pull_log=tmp_path / "pull.log",
            offline=False,
            no_pull=False,
            environment={},
        )


def _qualification_args(tmp_path, *, keep_going=False):
    return SimpleNamespace(
        profile="fast",
        fast=False,
        output=tmp_path / "evidence",
        model_cache=tmp_path / "models",
        offline=False,
        no_pull=False,
        json=True,
        candidate_manifest=None,
        keep_going=keep_going,
        list_profiles=False,
    )


def _patch_qualification(monkeypatch, tmp_path, *, doctor_exit):
    core = tmp_path / "rocm-npu"
    core.write_bytes(b"core")
    monkeypatch.setattr(runner, "_validate_protocol", lambda: (core, "1.2.3", True))
    monkeypatch.setattr(
        runner,
        "_package_versions",
        lambda: {name: "1.2.3" for name in installation.PACKAGE_DISTRIBUTIONS},
    )
    monkeypatch.setattr(
        runner,
        "_numeric_runtime_identities",
        lambda: {
            distribution: {
                "version": version,
                "compatibility_tags": ["cp312-cp312-win_amd64"],
                "record_sha256": "d" * 64,
                "payload_sha256": "e" * 64,
            }
            for distribution, version in NUMERIC_RUNTIME.items()
        },
    )
    monkeypatch.setattr(
        runner, "_test_xadx", lambda writer, *, target_family, profile_name: []
    )

    def core_report(core, command, **kwargs):
        kwargs["writer"].path(f"logs/tests/core-{command}.log").write_text(
            f"{command}\n", encoding="utf-8"
        )
        report = {
            "schema_version": 1,
            "command": command,
            "status": "pass" if command != "doctor" or doctor_exit == 0 else "fail",
            "product": {"version": "1.2.3", "source_revision": "abc"},
            "data": {
                "os": "Windows 11",
                "architecture": "x86_64",
                "target_family": "npu2",
            },
        }
        return report, doctor_exit if command == "doctor" else 0

    monkeypatch.setattr(runner, "_run_core_report", core_report)


def test_native_doctor_failure_preserves_partial_evidence(monkeypatch, tmp_path):
    _patch_qualification(monkeypatch, tmp_path, doctor_exit=1)
    exit_code, summary = runner.run_qualification(_qualification_args(tmp_path))
    assert exit_code == 1
    assert summary["status"] == "fail"
    evidence = json.loads(
        (tmp_path / "evidence" / "evidence.json").read_text(encoding="utf-8")
    )
    native = {test["name"]: test for test in evidence["tests"]}
    assert native["version"]["status"] == "pass"
    assert native["probe"]["status"] == "pass"
    assert native["doctor"]["status"] == "fail"
    assert native["self-test"]["status"] == "pass"
    assert (tmp_path / "evidence" / "junit.xml").is_file()
    assert (tmp_path / "evidence" / "SHA256SUMS").is_file()


def test_keep_going_after_doctor_and_model_failure(monkeypatch, tmp_path, capsys):
    _patch_qualification(monkeypatch, tmp_path, doctor_exit=1)
    flm = tmp_path / "flm"
    flm.write_bytes(b"flm")
    config = _config(tmp_path)
    monkeypatch.setattr(
        runner, "_resolve_flm", lambda: (flm, config, {"FLM_ENV": "installed"})
    )
    version_environments = []

    def fake_run(*args, **kwargs):
        version_environments.append(kwargs.get("env"))
        return _result(returncode=1, stdout="FLM v1.2.3\n")

    monkeypatch.setattr(runner, "run_command", fake_run)
    monkeypatch.setattr(
        runner,
        "resolve_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(models.ModelError("bad model")),
    )
    exit_code, summary = runner.run_qualification(
        _qualification_args(tmp_path, keep_going=True)
    )
    assert exit_code == 1
    assert summary["failures"][-1] == {
        "model": "qwen3:0.6b",
        "name": "model-resolution",
        "error": "bad model",
        "log": "logs/pull/qwen3-0-6b.log",
    }
    stderr = capsys.readouterr().err
    assert "qualify: resolving model qwen3:0.6b" in stderr
    assert "qualify: model resolution failed for qwen3:0.6b: bad model" in stderr
    assert "qualify: test qwen3:0.6b/model-resolution: fail" in stderr
    evidence = json.loads(
        (tmp_path / "evidence" / "evidence.json").read_text(encoding="utf-8")
    )
    assert {test["name"] for test in evidence["tests"]} == {
        "version",
        "probe",
        "doctor",
        "self-test",
        "model-resolution",
    }
    assert (
        json.loads((tmp_path / "evidence" / "tests.json").read_text(encoding="utf-8"))
        == evidence["tests"]
    )
    pull_log = tmp_path / "evidence" / "logs" / "pull" / "qwen3-0-6b.log"
    assert pull_log.is_file()
    assert "bad model" in pull_log.read_text(encoding="utf-8")
    assert pack_evidence(tmp_path / "evidence", tmp_path / "failure.zip")[
        "status"
    ] == "pass"
    assert version_environments[0]["FLM_ENV"] == "installed"


def test_model_test_uses_existing_fastflowlm_transcript_semantics(
    monkeypatch, tmp_path
):
    model = _model()
    model["tests"][0]["turns"][0]["expected"] = "expected phrase"
    monkeypatch.setattr(
        runner,
        "run_command",
        lambda *args, **kwargs: _result(stdout="Model RAW Output:\nwrong phrase\n>>>"),
    )
    monkeypatch.setattr(
        runner,
        "run_interactive_command",
        lambda *args, **kwargs: _result(stdout="Model RAW Output:\nwrong phrase\n>>>"),
    )
    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=False,
        writer=EvidenceWriter(tmp_path / "evidence"),
        environment={},
    )
    assert records[0]["status"] == "fail"
    assert "did not match" in records[0]["error"]


def test_model_test_validates_every_turn(monkeypatch, tmp_path):
    model = _model()
    model["tests"][0]["turns"] = [
        {"prompt": "alpha", "expected": "alpha"},
        {"prompt": "bravo", "expected": "bravo"},
    ]
    transcript = "Model RAW Output:\nalpha\n>>>\nModel RAW Output:\nbravo\n>>>"
    monkeypatch.setattr(
        runner, "run_command", lambda *args, **kwargs: _result(stdout=transcript)
    )
    monkeypatch.setattr(
        runner,
        "run_interactive_command",
        lambda *args, **kwargs: _result(stdout=transcript),
    )
    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=False,
        writer=EvidenceWriter(tmp_path / "evidence"),
        environment={},
    )
    assert records[0]["status"] == "pass"
    assert records[0]["turn_count"] == 2
    assert records[0]["metrics"]["responses"] == 2
    assert records[0]["metrics"]["raw_outputs_sha256"] == canonical_json_sha256(
        ["alpha", "bravo"]
    )


def test_model_test_reports_expected_regex_before_generic_text_quality(
    monkeypatch, tmp_path
):
    model = _model()
    model["tests"][0]["turns"] = [
        {"prompt": "beta", "expected_regex": "beta"}
    ]
    transcript = "Model RAW Output:\n\u03b2\n>>>"
    monkeypatch.setattr(
        runner, "run_command", lambda *args, **kwargs: _result(stdout=transcript)
    )
    monkeypatch.setattr(
        runner,
        "run_interactive_command",
        lambda *args, **kwargs: _result(stdout=transcript),
    )

    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=False,
        writer=EvidenceWriter(tmp_path / "evidence"),
        environment={},
    )

    assert records[0]["status"] == "fail"
    assert records[0]["error"] == "FastFlowLM turn 1 did not match regex"


def test_model_test_materializes_generated_context_prompt(monkeypatch, tmp_path):
    model = _model()
    marker = "HRX-NPU-CONTEXT-4K-PASS"
    model["tests"][0].update(
        {
            "turns": [
                {
                    "prompt_generator": {
                        "schema_version": 1,
                        "kind": "context-tail-marker-v1",
                        "filler_word_count": 1000,
                        "marker_word_offset": 900,
                        "marker": marker,
                    },
                    "expected_regex": marker,
                }
            ],
            "ctx_len": 4096,
        }
    )
    captured = {}

    def fake_command(*args, **kwargs):
        captured.update(kwargs)
        return _result(stdout=f"Model RAW Output:\n{marker}\n>>>")

    monkeypatch.setattr(runner, "run_command", fake_command)
    monkeypatch.setattr(runner, "run_interactive_command", fake_command)
    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=False,
        writer=EvidenceWriter(tmp_path / "evidence"),
        environment={},
    )
    submitted = captured.get("input_text") or "\n".join(captured["input_lines"])
    assert records[0]["status"] == "pass"
    assert marker in submitted
    assert len(submitted.split()) > 1000


def test_benchmark_records_checkpoint_and_fresh_csv(monkeypatch, tmp_path):
    model = _model()
    model["tests"] = [
        {
            "name": "context-1k",
            "kind": "bench",
            "config": "bench/bench_1k_smoke.json",
            "timeout_seconds": 10,
            "checkpoint_min": 950,
            "checkpoint_max": 1024,
            "expected_regex": "HRX[- ]NPU[- ]4K[- ]PASS",
        }
    ]

    def fake_bench(*args, **kwargs):
        csv_path = kwargs["cwd"] / "bench_qwen3_0.6b_result.csv"
        csv_path.write_text(
            ",".join(benchmark.BENCHMARK_CSV_HEADER)
            + "\n1,2.5,0,2.5,2.5,391,0,391,391,35,0,35,35\n",
            encoding="utf-8",
        )
        return _result(
            stdout=(
                "Creating checkpoint at context length 1001\n"
                "Starting benchmark for 1k and iteration 1\n"
                "Model RAW Output:\nHRX-NPU-4K-PASS. This response contains enough "
                "coherent words to validate model output safely.\n>>>"
            )
        )

    monkeypatch.setattr(runner, "run_command", fake_bench)
    prior_tests = [{"name": "completed-native-check"}]
    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=False,
        writer=EvidenceWriter(tmp_path / "evidence"),
        environment={},
        prior_tests=prior_tests,
    )
    assert records[0]["status"] == "pass"
    assert records[0]["kind"] == "bench"
    assert records[0]["metrics"]["checkpoint_tokens"] == 1001
    assert records[0]["metrics"]["csv_checkpoint_row"][0] == "1"
    assert (
        records[0]["metrics"]["csv_checkpoint_metrics"]["prefill_avg_toks_per_s"] == 391
    )
    sidecar = json.loads(
        (tmp_path / "evidence" / "tests.json").read_text(encoding="utf-8")
    )
    assert [record["name"] for record in sidecar] == [
        "completed-native-check",
        "context-1k",
    ]


def test_benchmark_soak_validates_every_output_and_generated_config(
    monkeypatch, tmp_path
):
    model = _model()
    model["tests"] = [
        {
            "name": "long-lived-context-1k",
            "kind": "bench",
            "config": "bench/bench_1k_smoke.json",
            "iterations": 2,
            "timeout_seconds": 10,
            "checkpoint_min": 950,
            "checkpoint_max": 1024,
            "expected_regex": "validated",
        }
    ]
    captured_config = {}

    def fake_bench(*args, **kwargs):
        config = Path(args[0][-1])
        assert config.is_absolute()
        captured_config.update(json.loads(config.read_text(encoding="utf-8")))
        csv_path = kwargs["cwd"] / "bench_qwen3_0.6b_result.csv"
        csv_path.write_text(
            ",".join(benchmark.BENCHMARK_CSV_HEADER)
            + "\n1,2.5,0,2.5,2.5,391,0,391,391,35,0,35,35\n",
            encoding="utf-8",
        )
        return _result(
            stdout=(
                "Creating checkpoint at context length 1001\n"
                "Starting benchmark for 1k and iteration 1\n"
                "Model RAW Output:\nThe first validated response contains enough "
                "coherent words for the qualification oracle.\n>>>\n"
                "Creating checkpoint at context length 1002\n"
                "Starting benchmark for 1k and iteration 2\n"
                "Model RAW Output:\nThe second validated response contains enough "
                "coherent words for the qualification oracle.\n>>>"
            )
        )

    monkeypatch.setattr(runner, "run_command", fake_bench)
    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=False,
        writer=EvidenceWriter(tmp_path / "evidence"),
        environment={},
    )

    assert records[0]["status"] == "pass"
    assert records[0]["turn_count"] == 2
    assert records[0]["metrics"]["benchmark_iterations"] == 2
    assert records[0]["metrics"]["responses"] == 2
    assert captured_config["iterations"] == 2
    assert records[0]["artifacts"][0].endswith("benchmark-config.json")
    assert (tmp_path / "evidence" / records[0]["artifacts"][0]).is_file()

    bad_output = (
        "Creating checkpoint at context length 1001\n"
        "Starting benchmark for 1k and iteration 1\n"
        "Model RAW Output:\nThe first validated response contains enough coherent "
        "words for the qualification oracle.\n>>>\n"
        "Creating checkpoint at context length 1002\n"
        "Starting benchmark for 1k and iteration 2\n"
        "Model RAW Output:\nbad\n>>>"
    )
    monkeypatch.setattr(
        runner, "run_command", lambda *args, **kwargs: _result(stdout=bad_output)
    )
    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=False,
        writer=EvidenceWriter(tmp_path / "bad-evidence"),
        environment={},
    )
    assert records[0]["status"] == "fail"
    assert "output 2" in records[0]["error"]


def test_benchmark_accepts_terse_sentinel_only_output(monkeypatch, tmp_path):
    # Small/omni models may obey "begin with exactly HRX-NPU-4K-PASS" and emit
    # only the terse sentinel for a stage. That is fewer than eight words, but the
    # sentinel exception must keep it from being flagged as incoherent.
    model = _model()
    model["tests"] = [
        {
            "name": "context-1k",
            "kind": "bench",
            "config": "bench/bench_1k_smoke.json",
            "timeout_seconds": 10,
            "checkpoint_min": 950,
            "checkpoint_max": 1024,
            "expected_regex": "HRX[- ]NPU[- ]4K[- ]PASS",
        }
    ]

    def fake_bench(*args, **kwargs):
        csv_path = kwargs["cwd"] / "bench_qwen3_0.6b_result.csv"
        csv_path.write_text(
            ",".join(benchmark.BENCHMARK_CSV_HEADER)
            + "\n1,2.5,0,2.5,2.5,391,0,391,391,35,0,35,35\n",
            encoding="utf-8",
        )
        return _result(
            stdout=(
                "Creating checkpoint at context length 1001\n"
                "Starting benchmark for 1k and iteration 1\n"
                "Model RAW Output:\nHRX-NPU-4K-PASS\n>>>"
            )
        )

    monkeypatch.setattr(runner, "run_command", fake_bench)
    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=False,
        writer=EvidenceWriter(tmp_path / "evidence"),
        environment={},
    )
    assert records[0]["status"] == "pass"


def test_benchmark_csv_rejects_malformed_or_nonfinite_metrics():
    header = list(benchmark.BENCHMARK_CSV_HEADER)
    valid = [
        "1",
        "2.5",
        "0",
        "2.5",
        "2.5",
        "391",
        "0",
        "391",
        "391",
        "35",
        "0",
        "35",
        "35",
    ]
    invalid_csvs = [
        [["wrong", *header[1:]], valid],
        [header, ["1"]],
        [header, [*valid[:5], "nan", *valid[6:]]],
        [header, [*valid[:5], "0", *valid[6:]]],
        [header, valid, valid],
    ]
    for rows in invalid_csvs:
        with pytest.raises(ValueError):
            benchmark.benchmark_csv_checkpoint(rows, 1)


def test_benchmark_accepts_explicit_1k_stage_without_checkpoint_log():
    assert require_1k_checkpoint("Starting benchmark for 1k and iteration 1") == 1024


def test_benchmark_requires_every_soak_checkpoint():
    transcript = "\n".join(
        f"Starting benchmark for 1k and iteration {iteration}"
        for iteration in range(1, 4)
    )
    assert require_1k_checkpoint(transcript, expected_count=3) == 1024
    with pytest.raises(ValueError, match="at least 4 checkpoint"):
        require_1k_checkpoint(transcript, expected_count=4)


def test_checkpoint_returns_largest_in_window_for_single_iteration():
    # Gemma-family omni models emit several in-window checkpoints for a single
    # stage (sliding-window/local attention). Any value in the target band is a
    # valid checkpoint for the stage, so the largest is returned rather than
    # failing on the multiplicity.
    transcript = "\n".join(
        f"Creating checkpoint at context length {value}"
        for value in (977, 1009, 1945, 1977, 3881, 3912)
    )
    assert require_1k_checkpoint(transcript, minimum=3800, maximum=4096) == 3912


def test_benchmark_schedule_covers_each_stage_and_iteration_once():
    transcript = "\n".join(
        f"Starting benchmark for {context}k and iteration {iteration}"
        for iteration in (1, 2)
        for context in (4, 2, 1)
    )
    assert require_benchmark_schedule(transcript, iterations=2) == 6
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        require_benchmark_schedule(transcript.rsplit("\n", 1)[0], iterations=2)


def test_benchmark_selects_target_from_multi_stage_run():
    transcript = "\n".join(
        f"Starting benchmark for {size}k and iteration 1" for size in (4, 2, 1)
    )
    assert require_1k_checkpoint(transcript, minimum=3800, maximum=4096) == 4096


def test_xclbin_inventory_reports_missing_and_mismatched_files(tmp_path):
    (tmp_path / "mismatch.xclbin").write_bytes(b"wrong")
    errors = verify_xclbin_inventory(
        tmp_path,
        [
            {"path": "missing.xclbin", "sha256": "0" * 64},
            {"path": "mismatch.xclbin", "sha256": "0" * 64},
        ],
    )
    assert errors == [
        "missing missing.xclbin",
        "SHA256 mismatch for mismatch.xclbin",
    ]


def test_deterministic_inventory_and_secret_redaction(tmp_path):
    (tmp_path / "b").write_bytes(b"b")
    (tmp_path / "a").write_bytes(b"a")
    assert [entry["path"] for entry in file_inventory(tmp_path)] == ["a", "b"]
    text = redact("Authorization: Bearer secret-value token=abc password=hunter2")
    assert "secret-value" not in text
    assert "hunter2" not in text


def test_command_log_can_be_mirrored_to_stderr(tmp_path, capsys):
    result = run_command(
        [sys.executable, "-c", "print('qualification-output')"],
        log_path=tmp_path / "command.log",
        echo=True,
    )

    assert result.returncode == 0
    mirrored = capsys.readouterr().err
    assert "qualification log:" in mirrored
    assert "qualification-output" in mirrored


def test_xadx_plan_is_visible_before_execution(tmp_path, capsys):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps({"cases": [{"name": "dequant/zero"}, {"name": "mm/full"}]}),
        encoding="utf-8",
    )

    runner._report_xadx_plan(
        recipe,
        {
            "lifecycle_rounds": 5,
            "resident_rounds": 20,
            "readback_rounds": 32,
        },
    )

    output = capsys.readouterr()
    assert output.out == ""
    assert "XADX recipe scheduled 2 cases" in output.err
    assert "fastflowlm-production-kernels/dequant/zero: scheduled" in output.err
    assert "fastflowlm-production-kernels/mm/full: scheduled" in output.err


def test_installed_tree_rejects_file_absent_from_record(tmp_path):
    package_root = tmp_path / "site-packages" / "example"
    metadata_root = tmp_path / "site-packages" / "example.dist-info"
    package_root.mkdir(parents=True)
    metadata_root.mkdir()
    source = package_root / "module.py"
    source.write_text("value = 1\n")
    extra = package_root / "injected.py"
    extra.write_text("value = 2\n")

    with pytest.raises(cli.QualificationError, match="unrecorded file.*injected.py"):
        installation.verify_recorded_install_tree(
            "example", package_root, metadata_root, {source}
        )

    extra.unlink()
    pycache = package_root / "__pycache__"
    pycache.mkdir()
    (pycache / "module.cpython-312.pyc").write_bytes(b"generated")
    (metadata_root / "INSTALLER").write_text("pip\n")
    installation.verify_recorded_install_tree(
        "example", package_root, metadata_root, {source}
    )


def test_json_stdout_is_one_document(monkeypatch, tmp_path, capsys):
    core = tmp_path / "rocm-npu"
    core.write_bytes(b"core")
    monkeypatch.setenv("ROCM_NPU_PLUGIN_API", "1")
    monkeypatch.setenv("ROCM_NPU_CORE_EXECUTABLE", str(core.resolve()))
    monkeypatch.setenv("ROCM_NPU_CORE_VERSION", "1.2.3")
    monkeypatch.setenv("ROCM_NPU_OUTPUT_FORMAT", "json")
    monkeypatch.setattr(
        runner,
        "_package_versions",
        lambda: {name: "1.2.3" for name in installation.PACKAGE_DISTRIBUTIONS},
    )
    assert cli.main(["--list-profiles"]) == 0
    output = capsys.readouterr()
    parsed = json.loads(output.out)
    assert parsed["profiles"] == ["default", "extended", "fast"]
    assert output.err == ""


def test_evidence_file_inventory_is_deterministic(tmp_path):
    writer = EvidenceWriter(tmp_path)
    writer.write_json("evidence.json", {"status": "partial"})
    writer.write_json("profile.json", {"name": "test"})
    writer.write_junit([])
    writer.finalize()
    first = (tmp_path / "SHA256SUMS").read_text(encoding="ascii")
    writer.finalize()
    assert (tmp_path / "SHA256SUMS").read_text(encoding="ascii") == first


def test_junit_captures_metrics_and_unsupported_cases(tmp_path):
    writer = EvidenceWriter(tmp_path)
    writer.write_junit(
        [
            {
                "model": "qwen3:0.6b",
                "name": "context-1k",
                "kind": "bench",
                "status": "pass",
                "duration_seconds": 1.0,
                "turn_count": 1,
                "metrics": {"checkpoint_tokens": 1001},
                "log": "logs/test.log",
            },
            {
                "model": "xadx",
                "name": "npu1",
                "kind": "xadx",
                "status": "unsupported",
                "duration_seconds": 0.0,
                "error": "NPU2 required",
                "log": "",
            },
        ]
    )
    text = (tmp_path / "junit.xml").read_text(encoding="utf-8")
    assert 'skipped="1"' in text
    assert "checkpoint_tokens" in text
    assert "<skipped>NPU2 required</skipped>" in text


def test_existing_evidence_output_is_rejected(tmp_path):
    (tmp_path / "stale-secret.log").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        EvidenceWriter(tmp_path)


def test_json_argument_errors_are_one_document(monkeypatch, capsys):
    monkeypatch.delenv("ROCM_NPU_OUTPUT_FORMAT", raising=False)
    with pytest.raises(SystemExit) as exited:
        cli.main(["--json", "--unsupported"])
    assert exited.value.code == 2
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "error"
    assert output.err == ""


def test_interruption_finalizes_partial_evidence(tmp_path):
    writer = EvidenceWriter(tmp_path)
    writer.write_json(
        "evidence.json",
        {
            "schema_version": 1,
            "status": "running",
            "profile": {"name": "models"},
            "models": [],
            "tests": [],
            "finished_at": None,
        },
    )
    summary = runner.finalize_partial_evidence(
        tmp_path, status="interrupted", error="stopped"
    )
    assert summary["status"] == "interrupted"
    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["tests"][0]["name"] == "interruption"
    assert (tmp_path / "SHA256SUMS").is_file()


def test_interruption_preserves_completed_core_and_model_identity(
    monkeypatch, tmp_path
):
    _patch_qualification(monkeypatch, tmp_path, doctor_exit=0)
    flm = tmp_path / "flm"
    config = _config(tmp_path)
    flm.write_bytes(b"flm")
    monkeypatch.setattr(runner, "_resolve_flm", lambda: (flm, config, {}))
    monkeypatch.setattr(
        runner,
        "run_command",
        lambda *args, **kwargs: _result(stdout="FLM v1.2.3\n"),
    )
    monkeypatch.setattr(
        runner,
        "resolve_model",
        lambda *args, **kwargs: (
            {
                "tag": "qwen3:0.6b",
                "identity_policy": "revision-sha256",
                "source": "https://test",
                "revision": "immutable",
                "cache_directory": "model",
                "registry_sha256": "a" * 64,
                "pulled": False,
                "files": [{"path": "weights.bin", "size": 5, "sha256": "b" * 64}],
                "status": "pass",
            },
            [],
        ),
    )
    monkeypatch.setattr(
        runner,
        "_test_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    output = tmp_path / "interrupted"
    assert cli.main(["--output", str(output), "--json"]) == 130
    evidence = json.loads((output / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "interrupted"
    assert evidence["subject"]["source_revision"] == "abc"
    assert evidence["models"][0]["files"][0]["sha256"] == "b" * 64


def test_termination_signal_finalizes_partial_evidence(monkeypatch, tmp_path):
    output = tmp_path / "terminated"
    termination_signal = signal.SIGBREAK if os.name == "nt" else signal.SIGTERM

    def terminate(args):
        writer = EvidenceWriter(args.output)
        writer.write_json(
            "evidence.json",
            {
                "schema_version": 1,
                "status": "running",
                "profile": {"name": "models"},
                "models": [],
                "tests": [],
                "finished_at": None,
            },
        )
        signal.raise_signal(termination_signal)
        raise AssertionError("termination handler did not interrupt qualification")

    monkeypatch.setattr(cli, "run_qualification", terminate)
    assert (
        cli.main(["--output", str(output), "--json"]) == 128 + termination_signal
    )
    evidence = json.loads((output / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "interrupted"
    assert (output / "SHA256SUMS").is_file()


def test_evidence_schema_rejects_fabricated_empty_pass():
    schema = json.loads(
        (
            ROOT
            / "tools/rocm-npu-qualification/python/rocm_npu_qualification"
            / "schemas/evidence-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    fabricated = {
        "schema_version": 1,
        "status": "pass",
        "subject": {
            "version": "1.2.3",
            "source_revision": "a" * 40,
            "candidate_binding": {
                "present": False,
                "manifest_sha256": None,
            },
            "packages": {
                distribution: "1.2.3"
                for distribution in installation.PACKAGE_DISTRIBUTIONS
            },
            "numeric_runtime": {
                distribution: {
                    "version": version,
                    "compatibility_tags": ["cp312-cp312-win_amd64"],
                    "record_sha256": "d" * 64,
                    "payload_sha256": "e" * 64,
                }
                for distribution, version in NUMERIC_RUNTIME.items()
            },
            "flm": {
                "package_version": "1.2.3",
                "command": "flm",
                "version_output": "1.2.3",
            },
        },
        "observed_platform": {},
        "profile": {
            "name": "fast",
            "description": "test",
            "candidate_bound": False,
        },
        "models": [],
        "tests": [],
        "started_at": "2026-08-03T00:00:00Z",
        "finished_at": "2026-08-03T00:01:00Z",
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(fabricated)


def test_evidence_schema_accepts_portable_ad_hoc_pass():
    schema = json.loads(
        (
            ROOT
            / "tools/rocm-npu-qualification/python/rocm_npu_qualification"
            / "schemas/evidence-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    evidence = {
        "schema_version": 1,
        "status": "pass",
        "subject": {
            "version": "1.2.3",
            "source_revision": "a" * 40,
            "candidate_binding": {
                "present": False,
                "manifest_sha256": None,
            },
            "packages": {
                distribution: "1.2.3"
                for distribution in installation.PACKAGE_DISTRIBUTIONS
            },
            "numeric_runtime": {
                distribution: {
                    "version": version,
                    "compatibility_tags": ["cp312-cp312-win_amd64"],
                    "record_sha256": "d" * 64,
                    "payload_sha256": "e" * 64,
                }
                for distribution, version in NUMERIC_RUNTIME.items()
            },
            "flm": {
                "package_version": "1.2.3",
                "command": "flm",
                "version_output": "1.2.3",
            },
        },
        "observed_platform": {"target_family": "npu2"},
        "profile": {
            "name": "fast",
            "description": "test",
            "candidate_bound": False,
        },
        "models": [
            {
                "tag": "qwen3:0.6b",
                "identity_policy": "revision-sha256",
                "source": "https://test",
                "revision": "immutable",
                "status": "pass",
                "cache_directory": "models/qwen",
                "registry_sha256": "b" * 64,
                "pulled": False,
                "files": [
                    {
                        "path": "model.bin",
                        "size": 1,
                        "sha256": "c" * 64,
                    }
                ],
            }
        ],
        "tests": [
            {
                "model": "qwen3:0.6b",
                "name": "multi-turn-01",
                "kind": "run",
                "status": "pass",
                "duration_seconds": 1.0,
                "log": "logs/tests/run.log",
                "turn_count": 2,
                "process_iteration": 1,
                "soak_iterations": 5,
                "metrics": {"responses": 2},
                "artifacts": [],
            },
            {
                "model": "fastflowlm-production-kernels",
                "name": "host-readback/coherence",
                "kind": "xadx",
                "status": "pass",
                "duration_seconds": 1.0,
                "log": "xadx-report.json",
                "rounds": 47,
                "metrics": {"readback_rounds": 32},
                "artifacts": ["xadx-report.json"],
            },
        ],
        "started_at": "2026-08-03T00:00:00Z",
        "finished_at": "2026-08-03T00:01:00Z",
    }
    Draft202012Validator(schema).validate(evidence)


def test_shipped_runner_loads_and_executes_the_packaged_recipe(monkeypatch, tmp_path):
    assets = tmp_path / "xadx_assets"
    assets.mkdir()
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    schemas.joinpath("xadx-report-v1.schema.json").write_text(
        (
            ROOT
            / "tools/rocm-npu-qualification/python/rocm_npu_qualification"
            / "schemas/xadx-report-v1.schema.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    recipe = {"schema_version": 1, "sentinel": "packaged-recipe"}
    (assets / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    observed = {}

    def execute(body, root, **kwargs):
        observed.update({"recipe": body, "root": root, "kwargs": kwargs})
        return {
            "schema_version": 1,
            "report_kind": "execution",
            "status": "pass",
            "recipe_sha256": "a" * 64,
            "generator_version": "kernel-recipe-generator-v1",
            "executables": [
                {
                    "id": "dequant-" + "a" * 16,
                    "xclbin_path": "xclbins/test.xclbin",
                    "xclbin_sha256": "a" * 64,
                    "transaction": "transactions/test.bin",
                    "transaction_sha256": "b" * 64,
                    "xadx": "xadx/test.xadx",
                    "xadx_sha256": "c" * 64,
                }
            ],
            "architecture": "amdxdna",
            "case_names": ["recipe-case"],
            "failures": [],
            "lifecycle_rounds": 5,
            "resident_rounds": 20,
            "readback_rounds": 32,
            "readback_by_case": {"recipe-case": 32},
            "numeric_runtime": dict(NUMERIC_RUNTIME),
            "cases": 1,
            "materialized_buffers": [],
            "case_results": [
                {
                    "name": "recipe-case",
                    "status": "pass",
                    "dispatches": 57,
                    "readbacks": 32,
                    "validated_bytes": 114,
                    "duration_seconds": 1.25,
                    "mismatch_count": 0,
                    "first_mismatch": None,
                    "failures": [],
                }
            ],
        }

    monkeypatch.setattr(runner.resources, "files", lambda _: tmp_path)
    monkeypatch.setattr(runner, "run_recipe", execute)
    records = runner._test_xadx(
        EvidenceWriter(tmp_path / "evidence"),
        target_family="npu2",
        profile_name="fast",
    )
    assert observed["recipe"] == assets / "recipe.json"
    assert observed["root"] == assets
    assert observed["kwargs"] == {
        "target_family": "npu2",
        "lifecycle_rounds": 5,
        "resident_rounds": 20,
        "readback_rounds": 32,
    }
    assert records[0]["rounds"] == 57
    assert records[0]["metrics"]["validated_bytes"] == 114
    assert records[0]["metrics"]["mismatch_count"] == 0


def test_runner_writes_schema_valid_package_validation_report(
    monkeypatch, tmp_path
):
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    schemas.joinpath("xadx-report-v1.schema.json").write_text(
        (
            ROOT
            / "tools/rocm-npu-qualification/python/rocm_npu_qualification"
            / "schemas/xadx-report-v1.schema.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner.resources, "files", lambda _: tmp_path)
    writer = EvidenceWriter(tmp_path / "evidence")
    records = runner._test_xadx(
        writer, target_family="npu2", profile_name="fast"
    )
    report = json.loads((writer.root / "xadx-report.json").read_text())
    assert report["report_kind"] == "package-validation"
    assert report["status"] == "fail"
    assert report["case_results"][0]["failures"][0]["phase"] == (
        "package-validation"
    )
    assert records[0]["status"] == "fail"


def _sweep_config(root: Path) -> Path:
    path = root / "model_list.json"
    path.write_text(
        json.dumps(
            {
                "model_path": "models",
                "models": {
                    "qwen3": {
                        "0.6b": {
                            "name": "Qwen3-0.6B-NPU2",
                            "url": "https://hf.example/Qwen3-0.6B-NPU2/resolve/rev1",
                            "files": ["model.q4nx"],
                            "footprint": 0.6,
                            "details": {"family": "qwen3", "think": True},
                        }
                    },
                    "embed-gemma": {
                        "300m": {
                            "name": "Embedding-Gemma-300M-NPU2",
                            "url": "https://hf.example/Embedding-Gemma-300M-NPU2",
                            "files": ["model.q4nx"],
                            "footprint": 0.62,
                            "details": {"family": "embed-gemma"},
                            "label": ["embeddings"],
                        }
                    },
                    "whisper-v3": {
                        "turbo": {
                            "name": "Whisper-V3-Turbo-NPU2",
                            "url": "https://hf.example/Whisper-V3-Turbo-NPU2",
                            "files": ["model.q4nx"],
                            "footprint": 1.0,
                            "details": {"family": "whisper-v3"},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_enumerate_registry_models_classifies_and_pins_identity(tmp_path):
    config = _sweep_config(tmp_path)
    discovered = {model["tag"]: model for model in models.enumerate_registry_models(config)}
    assert set(discovered) == {"qwen3:0.6b", "embed-gemma:300m", "whisper-v3:turbo"}
    assert discovered["qwen3:0.6b"]["kind"] == "chat"
    assert discovered["qwen3:0.6b"]["identity_policy"] == "revision-sha256"
    assert discovered["qwen3:0.6b"]["source"] == "https://hf.example/Qwen3-0.6B-NPU2"
    assert discovered["qwen3:0.6b"]["revision"] == "rev1"
    assert discovered["qwen3:0.6b"]["think"] is True
    assert discovered["embed-gemma:300m"]["kind"] == "embed"
    assert discovered["embed-gemma:300m"]["identity_policy"] == "content-sha256"
    assert discovered["embed-gemma:300m"]["revision"] is None
    assert discovered["whisper-v3:turbo"]["kind"] == "asr"
    digest = canonical_json_sha256(json.loads(config.read_text(encoding="utf-8")))
    assert all(
        model["registry_sha256"] == digest and model["xclbins"] == []
        for model in discovered.values()
    )


def test_enumerate_registry_models_skips_entries_without_url(tmp_path):
    path = tmp_path / "model_list.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    "good": {"1b": {"name": "Good", "url": "https://hf.example/Good"}},
                    "bad": {"1b": {"name": "Bad"}},
                }
            }
        ),
        encoding="utf-8",
    )
    discovered = models.enumerate_registry_models(path)
    assert [model["tag"] for model in discovered] == ["good:1b"]


def test_resolve_model_dynamic_builds_inventory_without_pins(tmp_path):
    config = _sweep_config(tmp_path)
    cache = tmp_path / "cache"
    model_directory = cache / "models" / "Qwen3-0.6B-NPU2"
    model_directory.mkdir(parents=True)
    (model_directory / "model.q4nx").write_bytes(b"weights")
    model = next(
        entry
        for entry in models.enumerate_registry_models(config)
        if entry["tag"] == "qwen3:0.6b"
    )
    record, commands = models.resolve_model_dynamic(
        model,
        flm=tmp_path / "flm",
        config_path=config,
        model_cache=cache,
        pull_log=tmp_path / "pull.log",
        offline=False,
        no_pull=False,
        environment={},
    )
    assert commands == []  # already cached: no pull/check
    assert record["status"] == "pass"
    assert record["pulled"] is False
    assert record["cache_directory"] == "models/Qwen3-0.6B-NPU2"
    assert [entry["path"] for entry in record["files"]] == ["model.q4nx"]
    assert record["files"][0]["sha256"] == hashlib.sha256(b"weights").hexdigest()


def test_resolve_model_dynamic_reports_missing_required_file(monkeypatch, tmp_path):
    config = _sweep_config(tmp_path)
    cache = tmp_path / "cache"
    model_directory = cache / "models" / "Qwen3-0.6B-NPU2"
    model = next(
        entry
        for entry in models.enumerate_registry_models(config)
        if entry["tag"] == "qwen3:0.6b"
    )

    def fake_run(argv, **kwargs):
        # simulate a pull that produces the wrong file set
        model_directory.mkdir(parents=True, exist_ok=True)
        (model_directory / "unexpected.txt").write_bytes(b"noise")
        return _result(returncode=0)

    monkeypatch.setattr(models, "run_command", fake_run)
    with pytest.raises(models.ModelError, match="missing required files"):
        models.resolve_model_dynamic(
            model,
            flm=tmp_path / "flm",
            config_path=config,
            model_cache=cache,
            pull_log=tmp_path / "pull.log",
            offline=False,
            no_pull=False,
            environment={},
        )


def test_default_tests_for_scopes_by_kind():
    chat = runner._default_tests_for({"kind": "chat", "footprint": 0.6})
    assert [test["kind"] for test in chat] == ["run", "run", "bench"]
    single, multi, bench = chat
    assert [turn["prompt"] for turn in single["turns"]] == [runner._SWEEP_SINGLE_PROMPT]
    assert len(multi["turns"]) == 5
    assert multi["turns"][2]["expected_regex"] == r"\b4\b|four"
    # 8k is the default complete-sweep benchmark stage.
    assert runner._SWEEP_BENCH_DEFAULT == "8k"
    assert bench["name"] == "context-8k"
    assert bench["config"] == "bench/bench_8k_smoke.json"
    assert bench["checkpoint_min"] == 7600 and bench["checkpoint_max"] == 8192
    for kind in ("embed", "asr"):
        serve = runner._default_tests_for({"kind": kind, "footprint": 6.0})
        assert len(serve) == 1
        assert serve[0]["kind"] == "serve"
        assert serve[0]["probe_kind"] == kind
    # timeouts scale with footprint
    small = runner._default_tests_for({"kind": "chat", "footprint": 0.6})[0]
    large = runner._default_tests_for({"kind": "chat", "footprint": 22.0})[0]
    assert large["timeout_seconds"] > small["timeout_seconds"]


def test_default_tests_for_selects_benchmark_context():
    expected = {
        "1k": ("context-1k", "bench/bench_1k_smoke.json", 950, 1024),
        "4k": ("context-4k", "bench/bench_4k_smoke.json", 3800, 4096),
        "8k": ("context-8k", "bench/bench_8k_smoke.json", 7600, 8192),
    }
    for context, (name, config, cmin, cmax) in expected.items():
        tests = runner._default_tests_for(
            {"kind": "chat", "footprint": 1.0}, bench_context=context
        )
        bench = next(test for test in tests if test["kind"] == "bench")
        assert (bench["name"], bench["config"]) == (name, config)
        assert (bench["checkpoint_min"], bench["checkpoint_max"]) == (cmin, cmax)
    # larger context stages get proportionally larger benchmark timeouts
    model = {"kind": "chat", "footprint": 1.0}
    timeouts = {
        context: next(
            test
            for test in runner._default_tests_for(model, bench_context=context)
            if test["kind"] == "bench"
        )["timeout_seconds"]
        for context in ("1k", "4k", "8k")
    }
    assert timeouts["1k"] < timeouts["4k"] < timeouts["8k"]


def test_eight_k_benchmark_config_runs_three_iterations():
    config = json.loads(
        (
            QUALIFICATION_PYTHON
            / "rocm_npu_qualification"
            / "bench"
            / "bench_8k_smoke.json"
        ).read_text(encoding="utf-8")
    )
    assert config["max_length"] == 8192
    assert config["iterations"] == 3
    assert len(set(config["input_text"].split())) > 100


def test_build_markdown_report_renders_env_and_hrx_benchmark():
    evidence = {
        "status": "pass",
        "profile": {"name": "all-models"},
        "subject": {
            "source_revision": "a" * 40,
            "packages": {
                "rocm-npu-runtime": "1.2.3",
                "rocm-npu-fastflowlm": "1.2.3",
                "rocm-npu-qualification": "1.2.3",
            },
            "flm": {"version_output": "FLM v0.9.45"},
        },
        "observed_platform": {
            "os": "Windows 11",
            "architecture": "x86_64",
            "target_family": "npu2",
            "devices": [
                {
                    "device_name": "NPU Compute Accelerator Device",
                    "driver_version": "32.0.203.314",
                    "healthy": True,
                    "firmware_candidates": [
                        {"release": "1.0.21.43"},
                        {"release": "1.5.5.391"},
                    ],
                }
            ],
        },
        "models": [
            {"tag": "qwen3:0.6b", "status": "pass"},
            {"tag": "embed-gemma:300m", "status": "pass"},
        ],
        "tests": [
            {"model": "qwen3:0.6b", "name": "single-prompt", "kind": "run", "status": "pass"},
            {"model": "qwen3:0.6b", "name": "multi-turn", "kind": "run", "status": "pass"},
            {
                "model": "qwen3:0.6b",
                "name": "context-8k",
                "kind": "bench",
                "status": "pass",
                "metrics": {
                    "iterations": 3,
                    "csv_checkpoint_metrics": {
                        "context_length_k": 8,
                        "ttft_avg_s": 0.922,
                        "prefill_avg_toks_per_s": 1075.37,
                        "decoding_avg_toks_per_s": 52.15,
                    },
                },
            },
            {
                "model": "embed-gemma:300m",
                "name": "serve-probe",
                "kind": "serve",
                "status": "pass",
                "metrics": {
                    "probe_kind": "embed",
                    "detail": "embedding dim=768",
                    "embedding_dim": 768,
                    "request_latency_s": 0.05,
                    "input_tokens": 30,
                    "encode_toks_per_s": 600.0,
                },
            },
        ],
    }
    text = report.build_markdown_report(
        evidence,
        kinds={"qwen3:0.6b": "chat", "embed-gemma:300m": "embed"},
        generated_at="2026-08-07T00:00:00Z",
    )
    assert "## Environment" in text
    assert "| os | Windows 11 |" in text
    assert "| npu_driver | 32.0.203.314 |" in text
    assert "candidates 1.0.21.43, 1.5.5.391 (KMD-loaded not exposed by PnP)" in text
    assert "| flm_version | FLM v0.9.45 |" in text
    # context and iteration count live in the heading, not a column
    assert "## Benchmark (HRX, 8k, averaged over 3 iterations)" in text
    # no XRT columns
    assert "XRT" not in text
    # chat functional row and averaged HRX benchmark row
    assert "| `qwen3:0.6b` | chat | PASS | PASS | yes |" in text
    assert "| `qwen3:0.6b` | 0.922 | 1075.370 | 52.150 |" in text
    # embed model: serve-backed single, multi N/A in the functional matrix
    assert "| `embed-gemma:300m` | embed | PASS | N/A | — |" in text
    # embed models are not in the decode benchmark table; they get their own
    # modality table with dim / latency / encode throughput.
    assert "| `embed-gemma:300m` | N/A | N/A | N/A |" not in text
    assert "## Embedding probe (HRX)" in text
    # collapsed to the single backend-sensitive throughput metric
    assert "| Model | Status | Encode tok/s |" in text
    assert "| `embed-gemma:300m` | PASS | 600.000 |" in text


def test_word_error_rate_is_case_and_punctuation_insensitive():
    # Identical up to punctuation/case -> zero WER.
    assert _word_error_rate("This is something.", "- This is something.") == 0.0
    # One substitution out of three reference words.
    assert _word_error_rate("this is something", "this is nothing") == pytest.approx(
        1 / 3
    )
    # One deletion out of three reference words.
    assert _word_error_rate("this is something", "this is") == pytest.approx(1 / 3)
    # Empty reference is undefined.
    assert _word_error_rate("", "anything") is None


def test_wav_duration_seconds_reads_pcm_header(tmp_path):
    import wave

    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)  # exactly 1.0s of silence
    assert _wav_duration_seconds(path) == pytest.approx(1.0)
    # A non-WAV file degrades to None instead of raising.
    bogus = tmp_path / "not.wav"
    bogus.write_text("nope", encoding="utf-8")
    assert _wav_duration_seconds(bogus) is None


def test_transcript_contains_phrase_is_content_specific():
    # Case- and punctuation-insensitive contiguous phrase match.
    assert _transcript_contains_phrase("- This is something.", "this is something")
    assert _transcript_contains_phrase("Well, this is something!", "this is something")
    # A different transcript with the same word count must not pass.
    assert not _transcript_contains_phrase("that was nothing", "this is something")
    # "Any two alphabetic words" must not vacuously satisfy the pin.
    assert not _transcript_contains_phrase("hello world", "this is something")
    # An empty pin can never pass, so a missing reference fails closed.
    assert not _transcript_contains_phrase("this is something", "")


# A minimal OpenAI-compatible stand-in for `flm serve`: it prints the port line
# serve_probe scans for, then answers the embeddings / transcription endpoints.
_FAKE_SERVE_SERVER = r'''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path == "/v1/embeddings":
            payload = {
                "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}],
                "usage": {"prompt_tokens": 12},
            }
        else:
            content_type = self.headers.get("Content-Type", "")
            multipart_ok = (
                "multipart/form-data" in content_type
                and b'name="model"' in body
                and b'filename=' in body
            )
            text = "This is something." if multipart_ok else "malformed request"
            payload = {"text": text}
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw)

server = HTTPServer(("127.0.0.1", 0), Handler)
print("WebServer started on port %d ready" % server.server_address[1], flush=True)
server.serve_forever()
'''


def _fake_serve_argv(tmp_path):
    script = tmp_path / "fake_serve.py"
    script.write_text(_FAKE_SERVE_SERVER, encoding="utf-8")
    return [sys.executable, str(script)]


def test_serve_probe_embed_parses_port_and_reports_throughput(tmp_path):
    result, probe = serve_probe(
        _fake_serve_argv(tmp_path),
        log_path=tmp_path / "probe.log",
        env=dict(os.environ),
        probe_kind="embed",
        model="embed-gemma:300m",
        boot_timeout=30.0,
        http_timeout=15.0,
    )
    assert probe["ok"] is True
    assert isinstance(probe["port"], int) and probe["port"] > 0
    assert result.returncode == 0
    assert probe["metrics"]["encode_toks_per_s"] > 0
    assert "embedding dim=4" in probe["detail"]


def test_serve_probe_asr_transcribes_multipart_and_matches_phrase(tmp_path):
    import wave

    sample = tmp_path / "clip.wav"
    with wave.open(str(sample), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)  # 1.0s
    result, probe = serve_probe(
        _fake_serve_argv(tmp_path),
        log_path=tmp_path / "probe.log",
        env=dict(os.environ),
        probe_kind="asr",
        model="whisper-v3:turbo",
        boot_timeout=30.0,
        http_timeout=15.0,
        audio_sample=sample,
        reference_text="This is something.",
    )
    assert probe["ok"] is True
    assert result.returncode == 0
    assert probe["metrics"]["word_error_rate"] == 0.0
    assert probe["metrics"]["real_time_factor"] > 0


def test_serve_probe_asr_fails_closed_without_a_sample(tmp_path):
    result, probe = serve_probe(
        _fake_serve_argv(tmp_path),
        log_path=tmp_path / "probe.log",
        env=dict(os.environ),
        probe_kind="asr",
        model="whisper-v3:turbo",
        boot_timeout=30.0,
        http_timeout=15.0,
        audio_sample=None,
        reference_text="This is something.",
    )
    # A listening port is not evidence that transcription works.
    assert probe["ok"] is False
    assert result.returncode == 1
    assert "failing closed" in probe["detail"]


def test_serve_probe_asr_fails_closed_when_phrase_missing(tmp_path):
    import wave

    sample = tmp_path / "clip.wav"
    with wave.open(str(sample), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    result, probe = serve_probe(
        _fake_serve_argv(tmp_path),
        log_path=tmp_path / "probe.log",
        env=dict(os.environ),
        probe_kind="asr",
        model="whisper-v3:turbo",
        boot_timeout=30.0,
        http_timeout=15.0,
        audio_sample=sample,
        reference_text="a completely different sentence",
    )
    assert probe["ok"] is False
    assert result.returncode == 1
    assert "expected phrase" in probe["detail"]


def test_build_markdown_report_renders_asr_probe_metrics():
    evidence = {
        "status": "pass",
        "profile": {"name": "all-models"},
        "subject": {"packages": {}},
        "observed_platform": {},
        "models": [{"tag": "whisper-v3:turbo", "status": "pass"}],
        "tests": [
            {
                "model": "whisper-v3:turbo",
                "name": "serve-probe",
                "kind": "serve",
                "status": "pass",
                "metrics": {
                    "probe_kind": "asr",
                    "detail": "transcribed 3 words",
                    "request_latency_s": 1.5,
                    "transcript_words": 3,
                    "audio_seconds": 3.0,
                    "real_time_factor": 0.5,
                    "word_error_rate": 0.0,
                },
            }
        ],
    }
    text = report.build_markdown_report(
        evidence,
        kinds={"whisper-v3:turbo": "asr"},
        generated_at="2026-08-07T00:00:00Z",
    )
    assert "## ASR probe (HRX)" in text
    # collapsed to speed (RTF) plus the independent quality axis (WER)
    assert "| Model | Status | RTF | WER % |" in text
    assert "| `whisper-v3:turbo` | PASS | 0.500 | 0.0 |" in text
    # ASR models stay out of the decode benchmark table.
    assert "| `whisper-v3:turbo` | N/A | N/A | N/A |" not in text


def test_multi_iteration_checkpoint_collapses_repeated_stages():
    # FLM prints one checkpoint line per stage per iteration; a 3-iteration 8k
    # run repeats the 8k stage three times and must still resolve one target.
    transcript = "\n".join(
        f"Creating checkpoint at context length {value}"
        for _ in range(3)
        for value in (968, 1936, 3872, 7744)
    )
    assert require_1k_checkpoint(transcript, minimum=7600, maximum=8192) == 7744


def test_serve_branch_records_embed_probe(monkeypatch, tmp_path):
    model = {
        "tag": "embed-gemma:300m",
        "xclbins": [],
        "tests": [
            {
                "name": "serve-probe",
                "kind": "serve",
                "probe_kind": "embed",
                "timeout_seconds": 300,
            }
        ],
    }

    def fake_serve_probe(argv, **kwargs):
        assert "serve" in [str(item) for item in argv]
        assert "-e" in [str(item) for item in argv]
        Path(kwargs["log_path"]).write_text("served\n", encoding="utf-8")
        return (
            _result(returncode=0, stdout="WebServer started on port 11434\n"),
            {"probe_kind": "embed", "port": 11434, "detail": "embedding dim=768", "ok": True},
        )

    monkeypatch.setattr(runner, "serve_probe", fake_serve_probe)
    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=False,
        writer=EvidenceWriter(tmp_path / "evidence"),
        environment={},
    )
    assert len(records) == 1
    assert records[0]["kind"] == "serve"
    assert records[0]["status"] == "pass"
    assert records[0]["metrics"]["port"] == 11434


def test_serve_branch_fails_on_probe_error(monkeypatch, tmp_path):
    model = {
        "tag": "whisper-v3:turbo",
        "xclbins": [],
        "tests": [
            {
                "name": "serve-probe",
                "kind": "serve",
                "probe_kind": "asr",
                "timeout_seconds": 300,
            }
        ],
    }

    def fake_serve_probe(argv, **kwargs):
        Path(kwargs["log_path"]).write_text("no port\n", encoding="utf-8")
        return (
            _result(returncode=1, stdout=""),
            {"probe_kind": "asr", "port": None, "detail": "server did not report a listening port", "ok": False},
        )

    monkeypatch.setattr(runner, "serve_probe", fake_serve_probe)
    records = runner._test_model(
        model,
        flm=tmp_path / "flm",
        model_cache=tmp_path / "models",
        offline=False,
        writer=EvidenceWriter(tmp_path / "evidence"),
        environment={},
    )
    assert records[0]["status"] == "fail"
    assert "did not report a listening port" in records[0]["error"]


def _all_models_args(tmp_path):
    return SimpleNamespace(
        profile=None,
        fast=False,
        all_models=True,
        output=tmp_path / "evidence",
        model_cache=tmp_path / "models",
        offline=False,
        no_pull=False,
        json=True,
        candidate_manifest=None,
        keep_going=False,
        list_profiles=False,
        bench_context="8k",
    )


def _run_all_models_sweep(monkeypatch, tmp_path, *, candidate_manifest=None):
    _patch_qualification(monkeypatch, tmp_path, doctor_exit=0)

    def core_report(core, command, **kwargs):
        kwargs["writer"].path(f"logs/tests/core-{command}.log").write_text(
            f"{command}\n", encoding="utf-8"
        )
        report = {
            "schema_version": 1,
            "command": command,
            "status": "pass",
            # passing evidence requires a hex source_revision (40-64 chars)
            "product": {"version": "1.2.3", "source_revision": "a" * 40},
            "data": {
                "os": "Windows 11",
                "architecture": "x86_64",
                "target_family": "npu2",
            },
        }
        return report, 0

    monkeypatch.setattr(runner, "_run_core_report", core_report)
    flm = tmp_path / "flm"
    flm.write_bytes(b"flm")
    config = _sweep_config(tmp_path)
    monkeypatch.setattr(runner, "_resolve_flm", lambda: (flm, config, {"FLM_ENV": "x"}))
    monkeypatch.setattr(
        runner, "run_command", lambda *a, **k: _result(stdout="FLM v1.2.3\n")
    )

    def fake_resolve(model, **kwargs):
        return (
            {
                "tag": model["tag"],
                "identity_policy": model["identity_policy"],
                "source": model["source"],
                "revision": model["revision"],
                "status": "pass",
                "cache_directory": "models/x",
                "registry_sha256": "b" * 64,
                "pulled": False,
                "files": [{"path": "model.q4nx", "size": 1, "sha256": "c" * 64}],
            },
            [],
        )

    monkeypatch.setattr(runner, "resolve_model_dynamic", fake_resolve)

    def fake_test_model(model, **kwargs):
        kwargs["writer"].path("logs/tests/x.log").write_text(
            "ok\n", encoding="utf-8"
        )
        records = []
        for test in model["tests"]:
            record = {
                "model": model["tag"],
                "name": test["name"],
                "kind": test["kind"],
                "status": "pass",
                "duration_seconds": 0.1,
                "log": "logs/tests/x.log",
            }
            if test["kind"] == "run":
                record.update(
                    turn_count=len(test["turns"]),
                    process_iteration=1,
                    metrics={"responses": len(test["turns"])},
                    artifacts=[],
                )
            elif test["kind"] == "bench":
                record.update(
                    metrics={
                        "checkpoint_tokens": 7744,
                        "iterations": 3,
                        "csv_checkpoint_metrics": {
                            "context_length_k": 8,
                            "ttft_avg_s": 0.922,
                            "prefill_avg_toks_per_s": 1075.37,
                            "decoding_avg_toks_per_s": 52.15,
                        },
                    },
                    artifacts=[],
                )
            elif test["kind"] == "serve":
                record.update(
                    metrics={"probe_kind": test["probe_kind"], "port": 11434},
                    artifacts=[],
                )
            records.append(record)
        return records

    monkeypatch.setattr(runner, "_test_model", fake_test_model)

    args = _all_models_args(tmp_path)
    if candidate_manifest is not None:
        args.candidate_manifest = candidate_manifest
    exit_code, summary = runner.run_qualification(args)
    evidence = json.loads(
        (tmp_path / "evidence" / "evidence.json").read_text(encoding="utf-8")
    )
    return exit_code, summary, evidence


def test_all_models_sweep_generates_schema_valid_evidence(monkeypatch, tmp_path):
    exit_code, summary, evidence = _run_all_models_sweep(monkeypatch, tmp_path)
    assert exit_code == 0
    assert summary["status"] == "pass"
    assert summary["profile"] == "all-models"
    assert summary["models"] == 3
    assert evidence["profile"]["name"] == "all-models"
    # The sweep is a diagnostic-sweep descriptor, not a packaged profile-v1: it
    # records what ran (tag/kind/test names) and pins nothing.
    profile = json.loads(
        (tmp_path / "evidence" / "profile.json").read_text(encoding="utf-8")
    )
    assert profile["kind"] == "diagnostic-sweep"
    assert profile["name"] == "all-models"
    assert profile["bench_context"] == "8k"
    assert all(
        set(model) == {"tag", "kind", "tests"} for model in profile["models"]
    )
    assert all(
        all(isinstance(name, str) for name in model["tests"])
        for model in profile["models"]
    )
    kinds = {test["kind"] for test in evidence["tests"]}
    assert {"run", "bench", "serve", "native"} <= kinds
    # the default complete sweep benchmarks at 8k context
    bench_names = {
        test["name"] for test in evidence["tests"] if test["kind"] == "bench"
    }
    assert bench_names == {"context-8k"}
    # a single command also emits the human-readable report with the HRX tables
    report_text = (tmp_path / "evidence" / "report.md").read_text(encoding="utf-8")
    assert "## Environment" in report_text
    assert "## Benchmark (HRX, 8k, averaged over 3 iterations)" in report_text
    assert "| 0.922 | 1075.370 | 52.150 |" in report_text
    assert "XRT" not in report_text
    schema = json.loads(
        (
            ROOT
            / "tools/rocm-npu-qualification/python/rocm_npu_qualification"
            / "schemas/evidence-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(evidence)


def test_all_models_with_candidate_manifest_stays_ad_hoc(monkeypatch, tmp_path):
    # Passing --candidate-manifest to the sweep must not bind it: candidate_bound
    # tracks "execution equals a sealed wheel profile," which the sweep never is.
    # Version validation still runs; the embedded manifest stays the ad-hoc
    # placeholder so intake refuses the sweep.
    validated: list[dict | None] = []
    monkeypatch.setattr(
        runner,
        "_validate_versions",
        lambda installed, core_version, candidate, *a, **k: validated.append(
            candidate
        ),
    )
    monkeypatch.setattr(
        runner, "_normalize_candidate_manifest", lambda candidate: candidate
    )
    # The sweep's native core reports source_revision "a" * 40 (see the helper),
    # and the candidate is still validated against the core, so it must agree.
    manifest_path = tmp_path / "candidate.json"
    manifest_path.write_text(
        json.dumps({"source_revision": "a" * 40, "packages": {}}),
        encoding="utf-8",
    )

    exit_code, summary, evidence = _run_all_models_sweep(
        monkeypatch, tmp_path, candidate_manifest=manifest_path
    )

    assert exit_code == 0
    # The candidate was still handed to version validation.
    assert {"source_revision": "a" * 40, "packages": {}} in validated
    # ...but the run is ad-hoc, not candidate-bound.
    assert evidence["profile"]["candidate_bound"] is False
    assert evidence["subject"]["candidate_binding"]["present"] is False
    assert evidence["subject"]["candidate_binding"]["manifest_sha256"] is None
    embedded = json.loads(
        (tmp_path / "evidence" / "candidate-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    # The embedded manifest is the ad-hoc placeholder, so candidate-bound intake
    # refuses this bundle (it never claims to equal a sealed wheel profile).
    assert embedded == {
        "schema_version": embedded["schema_version"],
        "kind": "ad-hoc",
        "candidate_bound": False,
    }


def test_validate_layout_allows_optional_report_overlay():
    required = {
        "SHA256SUMS": b"",
        "candidate-manifest.json": b"{}",
        "evidence.json": b"{}",
        "junit.xml": b"<x/>",
        "models.json": b"[]",
        "profile.json": b"{}",
        "tests.json": b"[]",
    }
    # report.md is an accepted optional overlay: no unexpected-nesting error.
    offline._validate_layout({**required, "report.md": b"# report"})
    # An unknown top-level file is still rejected.
    with pytest.raises(EvidenceError, match="unexpected nesting"):
        offline._validate_layout({**required, "surprise.md": b"nope"})


def test_all_models_is_mutually_exclusive_with_profile():
    parser = cli.create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--all-models", "--profile", "fast"])
    args = parser.parse_args(["--all-models"])
    assert args.all_models is True
    # 8k is the default benchmark stage; 1k/4k/8k are selectable
    assert args.bench_context == "8k"
    assert parser.parse_args(["--all-models", "--bench-context", "4k"]).bench_context == "4k"
    with pytest.raises(SystemExit):
        parser.parse_args(["--all-models", "--bench-context", "16k"])
