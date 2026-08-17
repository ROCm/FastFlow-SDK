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
    prompts,
    runner,
)
from rocm_npu_qualification.commands import (  # noqa: E402
    CommandResult,
    redact,
    run_command,
)
from rocm_npu_qualification.hashing import canonical_json_sha256  # noqa: E402
from rocm_npu_qualification.offline import pack_evidence  # noqa: E402
from rocm_npu_qualification.evidence import EvidenceWriter  # noqa: E402
from rocm_npu_qualification.fastflowlm import (  # noqa: E402
    require_1k_checkpoint,
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
    assert load_profile("extended")["models"][-1]["tag"] == "llama3.1:8b"


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
