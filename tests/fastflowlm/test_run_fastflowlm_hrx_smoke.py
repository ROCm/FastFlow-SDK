# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("run_fastflowlm_hrx_smoke.py")
SPEC = importlib.util.spec_from_file_location("fastflowlm_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class FastFlowLMSmokeTest(unittest.TestCase):
    def test_extract_raw_output(self):
        transcript = "before\nModel RAW Output:\ncoherent text\n>>> prompt\nafter\n"
        self.assertEqual(smoke.extract_raw_output(transcript), "coherent text")

    def test_extract_multiple_raw_outputs(self):
        transcript = (
            "Model RAW Output:\nfirst response\n"
            "[FLM]  Prefill chunk 1/1\n"
            "Model RAW Output:\nsecond response\n"
        )
        self.assertEqual(
            smoke.extract_raw_outputs(transcript),
            ["first response", "second response"],
        )

    def test_extract_benchmark_raw_output_stops_before_metrics(self):
        transcript = "Model RAW Output:\ncoherent text\n[FLM]  TTFT: 1.0s\n"
        self.assertEqual(smoke.extract_raw_output(transcript), "coherent text")

    def test_smoke_phrase_match_allows_markdown_emphasis(self):
        self.assertTrue(
            smoke.smoke_phrase_matches(
                "HRX runtime smoke test **two**.",
                "HRX runtime smoke test two.",
            )
        )

    def test_smoke_phrase_match_rejects_wrong_text(self):
        self.assertFalse(
            smoke.smoke_phrase_matches(
                "HRX runtime smoke test too.",
                "HRX runtime smoke test two.",
            )
        )

    def test_smoke_phrase_match_rejects_missing_punctuation(self):
        self.assertFalse(
            smoke.smoke_phrase_matches(
                "HRX runtime smoke test two",
                "HRX runtime smoke test two.",
            )
        )

    def test_smoke_setup_commands_default_to_greedy_short_generation(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                smoke.smoke_setup_commands(),
                ["/set topk 1", "/set gen-lim 16"],
            )

    def test_smoke_setup_commands_can_be_overridden_or_disabled(self):
        with mock.patch.dict(
            os.environ,
            {"FLM_SMOKE_SETUP_COMMANDS": "/set gen-lim 8\n\n/set topk 2\n"},
        ):
            self.assertEqual(
                smoke.smoke_setup_commands(),
                ["/set gen-lim 8", "/set topk 2"],
            )
        with mock.patch.dict(os.environ, {"FLM_SMOKE_SETUP_COMMANDS": ""}):
            self.assertEqual(smoke.smoke_setup_commands(), [])

    def test_parse_duration(self):
        self.assertEqual(smoke.parse_duration_seconds("250ms", 9), 0.25)
        self.assertEqual(smoke.parse_duration_seconds("2m", 9), 120.0)
        self.assertEqual(smoke.parse_duration_seconds("bad", 9), 9)

    def test_positive_inference_iterations(self):
        self.assertEqual(smoke.positive_int("5"), 5)
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                smoke.positive_int(value)

    def test_parse_inference_iterations(self):
        self.assertEqual(smoke.parse_args([]).inference_iterations, 1)
        self.assertEqual(
            smoke.parse_args(["--inference-iterations", "5"]).inference_iterations,
            5,
        )

    def test_csv_has_1k_row(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bench.csv"
            path.write_text("Prompt Length,TTFT\n1,2.5\n", encoding="utf-8")
            self.assertTrue(smoke.csv_has_1k_row(path))

    def test_installed_package_paths_use_wheel_modules(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            flm = root / "flm"
            hrx = root / "libhrx.so"
            config = root / "model_list.json"
            xclbins = root / "share" / "flm"
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
            self.assertEqual(paths["flm_bin"], flm)
            self.assertEqual(paths["hrx_runtime"], hrx)
            self.assertEqual(paths["config_path"], config)
            self.assertEqual(paths["xclbin_prefix"], xclbins)

    def test_installed_package_paths_prefers_console_script(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scripts = root / "venv" / ("Scripts" if smoke.IS_WINDOWS else "bin")
            script = scripts / ("flm.exe" if smoke.IS_WINDOWS else "flm")
            script.parent.mkdir(parents=True)
            script.write_text("", encoding="utf-8")
            package_flm = root / "package" / "bin" / script.name
            hrx = root / "libhrx.so"
            config = root / "model_list.json"
            xclbins = root / "share" / "flm"
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
            self.assertEqual(paths["flm_bin"], script)

    def test_run_tee_redirects_input_and_writes_log(self):
        child = (
            "import sys; "
            "print('ready', flush=True); "
            "print(sys.stdin.readline().strip(), flush=True)"
        )
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "child.log"
            output = smoke.run_tee(
                [sys.executable, "-u", "-c", child],
                env=dict(os.environ),
                input_text="hello\n",
                timeout=5,
                output_path=log,
            )
            self.assertEqual(output, "ready\nhello\n")
            self.assertEqual(log.read_text(encoding="utf-8"), output)

    def test_run_tee_can_drive_prompt_paced_input(self):
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
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "child.log"
            output = smoke.run_tee(
                [sys.executable, "-u", "-c", child],
                env=dict(os.environ),
                input_lines=["one", "two"],
                timeout=5,
                output_path=log,
            )
            self.assertEqual(
                smoke.extract_raw_outputs(output),
                ["response one", "response two"],
            )
            self.assertIn(">>> /bye\n", output)
            self.assertEqual(log.read_text(encoding="utf-8"), output)


if __name__ == "__main__":
    unittest.main()
