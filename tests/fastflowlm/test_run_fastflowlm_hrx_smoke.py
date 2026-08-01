# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


SCRIPT = Path(__file__).with_name("run_fastflowlm_hrx_smoke.py")
SPEC = importlib.util.spec_from_file_location("fastflowlm_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]


class TestFastFlowLMSmoke:
    def test_extract_raw_output(self):
        transcript = "before\nModel RAW Output:\ncoherent text\n>>> prompt\nafter\n"
        assert smoke.extract_raw_output(transcript) == "coherent text"

    def test_extract_multiple_raw_outputs(self):
        transcript = (
            "Model RAW Output:\nfirst response\n"
            "[FLM]  Prefill chunk 1/1\n"
            "Model RAW Output:\nsecond response\n"
        )
        assert smoke.extract_raw_outputs(transcript) == [
            "first response",
            "second response",
        ]

    def test_extract_benchmark_raw_output_stops_before_metrics(self):
        transcript = "Model RAW Output:\ncoherent text\n[FLM]  TTFT: 1.0s\n"
        assert smoke.extract_raw_output(transcript) == "coherent text"

    def test_smoke_phrase_match_allows_markdown_emphasis(self):
        assert smoke.smoke_phrase_matches(
            "HRX runtime smoke test **two**.",
            "HRX runtime smoke test two.",
        )

    def test_smoke_phrase_match_rejects_wrong_text(self):
        assert not smoke.smoke_phrase_matches(
            "HRX runtime smoke test too.",
            "HRX runtime smoke test two.",
        )

    def test_smoke_phrase_match_rejects_missing_punctuation(self):
        assert not smoke.smoke_phrase_matches(
            "HRX runtime smoke test two",
            "HRX runtime smoke test two.",
        )

    def test_smoke_setup_commands_default_to_greedy_short_generation(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert smoke.smoke_setup_commands() == [
                "/set topk 1",
                "/set gen-lim 16",
            ]

    def test_smoke_setup_commands_can_be_overridden_or_disabled(self):
        with mock.patch.dict(
            os.environ,
            {"FLM_SMOKE_SETUP_COMMANDS": "/set gen-lim 8\n\n/set topk 2\n"},
        ):
            assert smoke.smoke_setup_commands() == [
                "/set gen-lim 8",
                "/set topk 2",
            ]
        with mock.patch.dict(os.environ, {"FLM_SMOKE_SETUP_COMMANDS": ""}):
            assert smoke.smoke_setup_commands() == []

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("250ms", 0.25), ("2m", 120.0), ("bad", 9)],
    )
    def test_parse_duration(self, value, expected):
        assert smoke.parse_duration_seconds(value, 9) == expected

    def test_positive_inference_iterations(self):
        assert smoke.positive_int("5") == 5

    @pytest.mark.parametrize("value", ["0", "-1", "invalid"])
    def test_nonpositive_inference_iterations(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            smoke.positive_int(value)

    def test_parse_inference_iterations(self):
        assert smoke.parse_args([]).inference_iterations == 1
        assert (
            smoke.parse_args(["--inference-iterations", "5"]).inference_iterations
            == 5
        )

    def test_csv_has_1k_row(self, tmp_path):
        path = tmp_path / "bench.csv"
        path.write_text("Prompt Length,TTFT\n1,2.5\n", encoding="utf-8")
        assert smoke.csv_has_1k_row(path)

    def test_require_1k_checkpoint_accepts_actual_near_boundary(self):
        text = "\x1b[31m[FLM] Creating checkpoint at context length 981\x1b[0m"
        assert smoke.require_1k_checkpoint(text) == 981

    @pytest.mark.parametrize(
        "text",
        [
            "benchmark completed without a checkpoint",
            (
                "Creating checkpoint at context length 981\n"
                "Creating checkpoint at context length 982"
            ),
            "Creating checkpoint at context length unknown",
        ],
    )
    def test_require_1k_checkpoint_rejects_missing_or_duplicate_report(self, text):
        with pytest.raises(smoke.SmokeFailure):
            smoke.require_1k_checkpoint(text)

    @pytest.mark.parametrize("value", [949, 1025])
    def test_require_1k_checkpoint_rejects_out_of_range_report(self, value):
        with pytest.raises(smoke.SmokeFailure):
            smoke.require_1k_checkpoint(
                f"Creating checkpoint at context length {value}"
            )

    def test_installed_package_paths_use_wheel_modules(self, tmp_path):
        flm = tmp_path / "flm"
        hrx = tmp_path / "libhrx.so"
        config = tmp_path / "model_list.json"
        xclbins = tmp_path / "share" / "flm"
        for path in (flm, hrx, config):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        (xclbins / "xclbins").mkdir(parents=True)
        fastflowlm = types.SimpleNamespace(
            flm_path=lambda: flm,
            config_path=lambda: config,
            xclbin_path=lambda: xclbins,
        )
        runtime = types.SimpleNamespace(libhrx_path=lambda: hrx)
        with mock.patch.dict(sys.modules, {
                "rocm_npu_fastflowlm": fastflowlm,
                "rocm_npu_runtime": runtime,
        }):
            paths = smoke.installed_package_paths()
        assert paths["flm_bin"] == flm
        assert paths["hrx_runtime"] == hrx
        assert paths["config_path"] == config
        assert paths["xclbin_prefix"] == xclbins

    def test_installed_package_paths_prefers_console_script(self, tmp_path):
        scripts = tmp_path / "venv" / ("Scripts" if smoke.IS_WINDOWS else "bin")
        script = scripts / ("flm.exe" if smoke.IS_WINDOWS else "flm")
        script.parent.mkdir(parents=True)
        script.write_text("", encoding="utf-8")
        package_flm = tmp_path / "package" / "bin" / script.name
        hrx = tmp_path / "libhrx.so"
        config = tmp_path / "model_list.json"
        xclbins = tmp_path / "share" / "flm"
        for path in (package_flm, hrx, config):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        (xclbins / "xclbins").mkdir(parents=True)
        fastflowlm = types.SimpleNamespace(
            flm_path=lambda: package_flm,
            config_path=lambda: config,
            xclbin_path=lambda: xclbins,
        )
        runtime = types.SimpleNamespace(libhrx_path=lambda: hrx)
        with (
            mock.patch.dict(sys.modules, {
                    "rocm_npu_fastflowlm": fastflowlm,
                    "rocm_npu_runtime": runtime,
            }),
            mock.patch.object(smoke.sysconfig, "get_path", return_value=str(scripts)),
        ):
            paths = smoke.installed_package_paths()
        assert paths["flm_bin"] == script

    def test_run_tee_redirects_input_and_writes_log(self, tmp_path, capsys):
        child = (
            "import sys; "
            "print('ready', flush=True); "
            "print(sys.stdin.readline().strip(), flush=True)"
        )
        log = tmp_path / "child.log"
        output = smoke.run_tee(
            [sys.executable, "-u", "-c", child],
            env=dict(os.environ),
            input_text="hello\n",
            timeout=5,
            output_path=log,
        )
        assert output == "ready\nhello\n"
        assert log.read_text(encoding="utf-8") == output
        assert output in capsys.readouterr().out

    def test_run_tee_can_drive_prompt_paced_input(self, tmp_path):
        child = (
            "import sys\n"
            "for _ in range(2):\n"
            "    sys.stdout.write('>>> ')\n"
            "    sys.stdout.flush()\n"
            "    line = sys.stdin.readline().strip()\n"
            "    print(line, flush=True)\n"
            "    print('Model RAW Output:', flush=True)\n"
            "    print('response ' + line, flush=True)\n"
            "sys.stdout.write('>>> ')\n"
            "sys.stdout.flush()\n"
            "print(sys.stdin.readline().strip(), flush=True)\n"
        )
        log = tmp_path / "child.log"
        output = smoke.run_tee(
            [sys.executable, "-u", "-c", child],
            env=dict(os.environ),
            input_lines=["one", "two"],
            timeout=5,
            output_path=log,
        )
        assert smoke.extract_raw_outputs(output) == [
            "response one",
            "response two",
        ]
        assert ">>> /bye\n" in output
        assert log.read_text(encoding="utf-8") == output
