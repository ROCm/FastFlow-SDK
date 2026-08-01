#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Cross-platform FastFlowLM-on-HRX multi-turn inference and benchmark smoke test."""

import argparse
import csv
import hashlib
import os
import queue
import re
import subprocess
import sys
import sysconfig
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IS_WINDOWS = os.name == "nt"
FASTFLOWLM_XCLBIN_RELEASE = "v0.9.45"
FASTFLOWLM_QWEN3_06B_XCLBINS = {
    "attn.xclbin": "9acc30b9412d01b57c0c8c909851eaaaa3607b38d4212b7028c6bf87cdac1d2c",
    "dequant.xclbin": "575d45e47993315f95f601e07f9f21164f30515f80e232e16255d7c5f2a1d4bf",
    "layer.xclbin": "9b26f1089ace21f84e26d126957175a6e34eed2a099ca30656fb2a03ef7eea6b",
    "mm.xclbin": "4afc2a6f2459c0ddcf7de61193a6e83e53a951a0f6c33521456d569bab2c7142",
}


class SmokeFailure(RuntimeError):
    pass


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-iterations",
        type=positive_int,
        default=1,
        help="number of independent two-turn FLM inference processes to validate",
    )
    return parser.parse_args(argv)


def parse_duration_seconds(value, default):
    if not value:
        return default
    text = str(value).strip().lower()
    try:
        if text.endswith("ms"):
            return max(0.001, float(text[:-2]) / 1000.0)
        if text.endswith("s"):
            return float(text[:-1])
        if text.endswith("m"):
            return float(text[:-1]) * 60.0
        return float(text)
    except ValueError:
        return default


def prepend_search_path(env, key, entries):
    parts = [str(entry) for entry in entries if Path(entry).is_dir()]
    current = env.get(key, "")
    if current:
        parts.append(current)
    env[key] = os.pathsep.join(parts)


def run_tee(
    cmd,
    *,
    env,
    cwd=None,
    input_text=None,
    input_lines=None,
    timeout=None,
    output_path=None,
    prompt_marker=">>> ",
    exit_input="/bye",
):
    """Runs a command while streaming merged output to the console and a log."""
    if input_text is not None and input_lines is not None:
        raise ValueError("input_text and input_lines are mutually exclusive")
    command = [str(part) for part in cmd]
    print(f"+ {' '.join(command)}", flush=True)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    input_lines = list(input_lines) if input_lines is not None else None

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=(
            subprocess.PIPE
            if input_text is not None or input_lines is not None
            else None
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if input_text is not None:
        try:
            process.stdin.write(input_text)
            process.stdin.flush()
        except BrokenPipeError:
            pass
        finally:
            process.stdin.close()

    chunks = queue.Queue()

    def read_output():
        try:
            for chunk in iter(lambda: process.stdout.read(1), ""):
                chunks.put(chunk)
        finally:
            chunks.put(None)

    reader = threading.Thread(target=read_output, name="fastflowlm-output", daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout if timeout is not None else None
    output = []
    timed_out = False
    prompts_seen = 0
    lines_sent = 0
    exit_sent = False

    log = output_path.open("w", encoding="utf-8", errors="replace") if output_path else None
    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline and process.poll() is None:
                timed_out = True
                process.kill()
            try:
                chunk = chunks.get(timeout=0.1)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    break
                continue
            if chunk is None:
                break
            output.append(chunk)
            print(chunk, end="", flush=True)
            if log:
                log.write(chunk)
                log.flush()
            if input_lines is not None and process.stdin and not process.stdin.closed:
                text = "".join(output)
                current_prompts = text.count(prompt_marker)
                while prompts_seen < current_prompts:
                    prompts_seen += 1
                    try:
                        if lines_sent < len(input_lines):
                            process.stdin.write(input_lines[lines_sent] + "\n")
                            process.stdin.flush()
                            lines_sent += 1
                        elif not exit_sent:
                            if exit_input is not None:
                                process.stdin.write(exit_input + "\n")
                                process.stdin.flush()
                            process.stdin.close()
                            exit_sent = True
                        else:
                            break
                    except BrokenPipeError:
                        break
    finally:
        if log:
            log.close()

    returncode = process.wait()
    reader.join(timeout=1)
    process.stdout.close()
    text = "".join(output)
    if timed_out:
        raise SmokeFailure(f"command timed out after {timeout:g}s: {' '.join(command)}")
    if input_lines is not None and lines_sent != len(input_lines):
        raise SmokeFailure(
            f"command exited after receiving {lines_sent} interactive inputs; "
            f"expected {len(input_lines)}: {' '.join(command)}"
        )
    if returncode != 0:
        raise SmokeFailure(f"command exited with status {returncode}: {' '.join(command)}")
    return text


def fail_if_runtime_errors(text, label, path):
    if re.search(r"\[hrx\]\[ERROR\]|INTERNAL;|ERT_CMD_CHAIN did not complete|ert state [0-9]+", text):
        raise SmokeFailure(f"{label} emitted HRX runtime errors: {path}")
    if re.search(r"\[WARNING\].*Max length reached", text):
        raise SmokeFailure(f"{label} hit the generation length limit: {path}")


def extract_raw_outputs(text):
    outputs = []
    lines = None
    for line in text.splitlines():
        if "Model RAW Output:" in line:
            if lines is not None:
                outputs.append("\n".join(lines).strip())
            lines = []
            continue
        if lines is not None and (line.startswith(">>>") or line.startswith("[FLM]")):
            outputs.append("\n".join(lines).strip())
            lines = None
            continue
        if lines is not None:
            lines.append(line)
    if lines is not None:
        outputs.append("\n".join(lines).strip())
    return outputs


def extract_raw_output(text):
    outputs = extract_raw_outputs(text)
    return outputs[0] if outputs else ""


def normalize_smoke_phrase(text):
    text = text.strip()
    previous = None
    while text != previous:
        previous = text
        text = re.sub(r"`([^`\n]+)`", r"\1", text)
        text = re.sub(r"(\*\*|__|\*|_)([^*_`\n]+?)\1", r"\2", text)
    return text


def smoke_phrase_matches(raw_text, expected):
    return normalize_smoke_phrase(raw_text) == expected.strip()


def smoke_setup_commands():
    value = os.environ.get("FLM_SMOKE_SETUP_COMMANDS")
    if value is None:
        return ["/set topk 1", "/set gen-lim 16"]
    return [line.strip() for line in value.splitlines() if line.strip()]


def verify_qwen3_xclbins(xclbin_prefix):
    xclbin_dir = xclbin_prefix / "xclbins" / "Qwen3-0.6B-NPU2"
    errors = []
    for name, expected in FASTFLOWLM_QWEN3_06B_XCLBINS.items():
        path = xclbin_dir / name
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"{path}: expected {expected}, got {actual}")
    if errors:
        raise SmokeFailure(
            f"FastFlowLM {FASTFLOWLM_XCLBIN_RELEASE} Qwen3 xclbin check failed:\n  - "
            + "\n  - ".join(errors)
        )


def csv_has_1k_row(path):
    with path.open(newline="", encoding="utf-8", errors="replace") as stream:
        return any(row and row[0].strip() == "1" for row in csv.reader(stream))

def require_1k_checkpoint(text, minimum=950, maximum=1024):
    values = [
        int(value)
        for value in re.findall(
            r"Creating checkpoint at context length\s+(\d+)", text
        )
    ]
    if len(values) != 1:
        raise SmokeFailure(
            "FastFlowLM 1k benchmark must report exactly one actual "
            f"checkpoint length; got {values or 'none'}"
        )
    actual = values[0]
    if not minimum <= actual <= maximum:
        raise SmokeFailure(
            "FastFlowLM 1k benchmark did not exercise a near-1k prompt: "
            f"got {actual} tokens, expected {minimum}..{maximum}"
        )
    return actual


def require_file(path, label):
    if not path.is_file():
        raise SmokeFailure(f"missing {label}: {path}")


def installed_package_paths():
    try:
        import rocm_npu_fastflowlm
        import rocm_npu_runtime
    except ImportError as error:
        raise SmokeFailure(f"installed rocm-npu wheel package is unavailable: {error}") from error
    script = Path(sysconfig.get_path("scripts")) / ("flm.exe" if IS_WINDOWS else "flm")
    return {
        "flm_bin": script if script.is_file() else Path(rocm_npu_fastflowlm.flm_path()),
        "hrx_runtime": Path(rocm_npu_runtime.libhrx_path()),
        "config_path": Path(rocm_npu_fastflowlm.config_path()),
        "xclbin_prefix": Path(rocm_npu_fastflowlm.xclbin_path()),
    }


def main():
    args = parse_args()
    bundle = Path(os.environ["FASTFLOWLM_BUNDLE"]) if os.environ.get("FASTFLOWLM_BUNDLE") else None
    installed_package = os.environ.get("FASTFLOWLM_INSTALLED_PACKAGE") == "1"
    fastflowlm_dir = Path(os.environ.get("FASTFLOWLM_DIR") or (ROOT / "projects" / "FastFlowLM"))
    flm_src = fastflowlm_dir / "src"
    library_name = "hrx.dll" if IS_WINDOWS else "libhrx.so"
    default_install = Path(os.environ.get("HRX_INSTALL") or (ROOT / "build" / "hrx-install"))
    runtime_dir = Path(os.environ.get("LIBHRX_DIR") or
                       (default_install / ("bin" if IS_WINDOWS else "lib")))
    package_paths = installed_package_paths() if installed_package else {}
    flm_bin = (package_paths["flm_bin"] if installed_package else
               bundle / "bin" / ("flm.exe" if IS_WINDOWS else "flm") if bundle else
               Path(os.environ.get("FLM_BIN") or (flm_src / "build" / ("flm.exe" if IS_WINDOWS else "flm"))))
    hrx_runtime = (package_paths["hrx_runtime"] if installed_package else
                   bundle / ("bin/hrx.dll" if IS_WINDOWS else "lib/libhrx.so") if bundle else
                   Path(os.environ.get("HRX_LIBHRX") or (runtime_dir / library_name)))
    model = os.environ.get("FLM_MODEL") or "qwen3:0.6b"
    model_cache = Path(os.environ.get("FLM_MODEL_PATH") or (ROOT / "build" / "FastFlowLM-models"))
    config_path = (package_paths["config_path"] if installed_package else
                   bundle / "share" / "flm" / "model_list.json" if bundle else
                   Path(os.environ.get("FLM_CONFIG_PATH") or (flm_src / "model_list.json")))
    xclbin_prefix = (package_paths["xclbin_prefix"] if installed_package else
                     bundle / "share" / "flm" if bundle else
                     Path(os.environ.get("FLM_XCLBIN_PATH") or flm_src))
    bench_config = Path(os.environ.get("FLM_BENCH_CONFIG") or (ROOT / "tests" / "fastflowlm" / "bench_1k_smoke.json"))
    work_dir = Path(os.environ.get("FLM_SMOKE_WORK_DIR") or (ROOT / "build" / "FastFlowLM-smoke"))
    smoke_turns = [
        (
            os.environ.get("FLM_SMOKE_PROMPT")
            or "Repeat exactly: HRX runtime smoke test one.",
            os.environ.get("FLM_SMOKE_EXPECTED")
            or "HRX runtime smoke test one.",
            os.environ.get("FLM_SMOKE_EXPECTED_REGEX"),
        ),
        (
            os.environ.get("FLM_SMOKE_PROMPT_2")
            or "Repeat exactly: HRX runtime smoke test two.",
            os.environ.get("FLM_SMOKE_EXPECTED_2")
            or "HRX runtime smoke test two.",
            os.environ.get("FLM_SMOKE_EXPECTED_REGEX_2"),
        ),
    ]
    setup_commands = smoke_setup_commands()
    bench_regex = os.environ.get("FLM_BENCH_EXPECTED_REGEX")

    require_file(flm_bin, "flm binary")
    require_file(config_path, "FastFlowLM model list")
    require_file(hrx_runtime, "HRX runtime library")
    require_file(bench_config, "benchmark config")
    if not (xclbin_prefix / "xclbins").is_dir():
        raise SmokeFailure(f"missing FastFlowLM xclbins: {xclbin_prefix / 'xclbins'}")
    if model == "qwen3:0.6b":
        verify_qwen3_xclbins(xclbin_prefix)

    model_cache.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if installed_package:
        env.setdefault("HRX_GPU_DRIVER", "amdxdna")
        env["FLM_CONFIG_PATH"] = str(config_path)
        env["FLM_XCLBIN_PATH"] = str(xclbin_prefix)
    elif not bundle:
        env.setdefault("HRX_GPU_DRIVER", "amdxdna")
        env["FLM_CONFIG_PATH"] = str(config_path)
        env["FLM_XCLBIN_PATH"] = str(xclbin_prefix)
    env["FLM_MODEL_PATH"] = str(model_cache)
    if not bundle and not installed_package:
        if IS_WINDOWS:
            prepend_search_path(env, "PATH", [flm_bin.parent, hrx_runtime.parent, flm_src / "lib"])
        else:
            prepend_search_path(env, "LD_LIBRARY_PATH", [hrx_runtime.parent, flm_src / "lib"])

    print("FastFlowLM HRX smoke")
    print(f"  model:       {model}")
    print(f"  flm:         {flm_bin}")
    print(f"  hrx runtime: {hrx_runtime}")
    print(f"  model cache: {model_cache}")
    print(f"  inference processes: {args.inference_iterations}")
    print(f"  turns per process: {len(smoke_turns)}")
    print(f"  setup commands: {len(setup_commands)}")

    run_tee([flm_bin, "pull", model], env=env)
    run_tee([flm_bin, "check", model], env=env)

    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    bench_output = work_dir / f"bench_{safe_model}.txt"

    for iteration in range(1, args.inference_iterations + 1):
        if args.inference_iterations == 1:
            run_output = work_dir / f"run_{safe_model}.txt"
            raw_output = work_dir / f"run_{safe_model}_raw.txt"
        else:
            suffix = f"{iteration:02d}-of-{args.inference_iterations:02d}"
            run_output = work_dir / f"run_{safe_model}_{suffix}.txt"
            raw_output = work_dir / f"run_{safe_model}_{suffix}_raw.txt"

        print(
            "Running FastFlowLM inference smoke "
            f"{iteration}/{args.inference_iterations}..."
        )
        print("== FastFlowLM inference transcript begin ==")
        input_lines = setup_commands + [prompt for prompt, _, _ in smoke_turns]
        if IS_WINDOWS:
            input_kwargs = {
                "input_text": "".join(f"{line}\n" for line in input_lines)
            }
        else:
            input_kwargs = {"input_lines": input_lines}
        run_text = run_tee(
            [flm_bin, "run", model, "--ctx-len", "1024"],
            env=env,
            timeout=parse_duration_seconds(os.environ.get("FLM_RUN_TIMEOUT"), 180.0),
            output_path=run_output,
            **input_kwargs,
        )
        print("== FastFlowLM inference transcript end ==")
        fail_if_runtime_errors(run_text, "FastFlowLM run", run_output)
        raw_texts = extract_raw_outputs(run_text)
        raw_output.write_text(
            "\n\n".join(raw_texts) + "\n", encoding="utf-8", errors="replace"
        )
        if len(raw_texts) != len(smoke_turns):
            raise SmokeFailure(
                f"FastFlowLM run produced {len(raw_texts)} model responses; "
                f"expected {len(smoke_turns)}: {run_output}"
            )
        for turn, (raw_text, (_, expected, expected_regex)) in enumerate(
            zip(raw_texts, smoke_turns), start=1
        ):
            if not re.search(r"[A-Za-z]{4,}", raw_text):
                raise SmokeFailure(
                    f"FastFlowLM turn {turn} did not produce visible model output: "
                    f"{run_output}"
                )
            if expected_regex and not re.search(
                expected_regex, raw_text, re.IGNORECASE
            ):
                raise SmokeFailure(
                    f"FastFlowLM turn {turn} did not match the expected phrase: "
                    f"{raw_output}\nExpected regex: {expected_regex}"
                )
            if not expected_regex and not smoke_phrase_matches(raw_text, expected):
                raise SmokeFailure(
                    f"FastFlowLM turn {turn} did not match the smoke phrase: "
                    f"{raw_output}\nExpected: {expected!r}; got: {raw_text!r}"
                )

    stamp_file = work_dir / ".bench-start"
    stamp_file.write_text("", encoding="utf-8")
    stamp_mtime = stamp_file.stat().st_mtime_ns
    print("Running FastFlowLM 1k benchmark smoke...")
    bench_text = run_tee(
        [flm_bin, "bench", model, "-i", bench_config],
        env=env,
        cwd=work_dir,
        timeout=parse_duration_seconds(os.environ.get("FLM_BENCH_TIMEOUT"), 90.0),
        output_path=bench_output,
    )
    fail_if_runtime_errors(bench_text, "FastFlowLM bench", bench_output)
    actual_checkpoint = require_1k_checkpoint(bench_text)
    bench_raw_text = extract_raw_output(bench_text)
    bench_raw_output = work_dir / f"bench_{safe_model}_raw.txt"
    bench_raw_output.write_text(bench_raw_text + "\n", encoding="utf-8", errors="replace")
    if len(re.findall(r"[A-Za-z]{2,}", bench_raw_text)) < 8:
        raise SmokeFailure(f"FastFlowLM benchmark did not produce coherent text: {bench_raw_output}")
    if bench_regex and not re.search(
        bench_regex, bench_raw_text, re.IGNORECASE | re.DOTALL
    ):
        raise SmokeFailure(
            f"FastFlowLM benchmark output did not summarize the input: {bench_raw_output}\n"
            f"Expected regex: {bench_regex}"
        )

    csvs = [
        path
        for path in work_dir.glob(f"bench_{safe_model}_*.csv")
        if path.stat().st_mtime_ns >= stamp_mtime
    ]
    if not csvs:
        raise SmokeFailure("FastFlowLM benchmark completed but did not write a fresh CSV")
    result_csv = max(csvs, key=lambda path: path.stat().st_mtime_ns)
    if not csv_has_1k_row(result_csv):
        raise SmokeFailure(f"FastFlowLM benchmark CSV does not contain a 1k result row: {result_csv}")
    print(
        "FastFlowLM HRX smoke: PASS "
        f"({actual_checkpoint} actual prompt tokens) -> {result_csv}"
    )


if __name__ == "__main__":
    try:
        main()
    except SmokeFailure as error:
        print(f"FastFlowLM HRX smoke: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
