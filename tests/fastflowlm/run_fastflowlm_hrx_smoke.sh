#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FASTFLOWLM_DIR="${FASTFLOWLM_DIR:-$ROOT/projects/FastFlowLM}"
HRX_BUILD="${HRX_BUILD:-$ROOT/projects/hrx/build/cmake}"
FLM_BIN="${FLM_BIN:-$FASTFLOWLM_DIR/src/build/flm}"
FLM_MODEL="${FLM_MODEL:-qwen3:0.6b}"
FLM_MODEL_PATH="${FLM_MODEL_PATH:-$ROOT/build/FastFlowLM-models}"
FLM_BENCH_CONFIG="${FLM_BENCH_CONFIG:-$ROOT/tests/fastflowlm/bench_1k_smoke.json}"
FLM_SMOKE_WORK_DIR="${FLM_SMOKE_WORK_DIR:-$ROOT/build/FastFlowLM-smoke}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing $label: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "missing $label: $path" >&2
    exit 1
  fi
}

require_file "$FLM_BIN" "flm binary"
require_file "$FASTFLOWLM_DIR/src/model_list.json" "FastFlowLM model list"
require_dir "$FASTFLOWLM_DIR/src/xclbins" "FastFlowLM xclbins"
require_file "$HRX_BUILD/libhrx/src/libhrx/libhrx.so" "libhrx.so"
require_file "$FLM_BENCH_CONFIG" "benchmark config"

mkdir -p "$FLM_MODEL_PATH" "$FLM_SMOKE_WORK_DIR"

export HRX_GPU_DRIVER="${HRX_GPU_DRIVER:-amdxdna}"
export FLM_CONFIG_PATH="$FASTFLOWLM_DIR/src/model_list.json"
export FLM_XCLBIN_PATH="$FASTFLOWLM_DIR/src"
export FLM_MODEL_PATH
export LD_LIBRARY_PATH="$HRX_BUILD/libhrx/src/libhrx:$FASTFLOWLM_DIR/src/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "FastFlowLM HRX smoke"
echo "  model:       $FLM_MODEL"
echo "  flm:         $FLM_BIN"
echo "  hrx build:   $HRX_BUILD"
echo "  model cache: $FLM_MODEL_PATH"

"$FLM_BIN" pull "$FLM_MODEL"
"$FLM_BIN" check "$FLM_MODEL"

safe_model="${FLM_MODEL//:/_}"
run_output="$FLM_SMOKE_WORK_DIR/run_${safe_model}.txt"

echo "Running FastFlowLM inference smoke..."
run_cmd=("$FLM_BIN" run "$FLM_MODEL" --ctx-len 1024)
if command -v timeout >/dev/null 2>&1; then
  run_cmd=(timeout "${FLM_RUN_TIMEOUT:-180s}" "${run_cmd[@]}")
fi

echo "== FastFlowLM inference transcript begin =="
printf 'Repeat exactly: HRX runtime smoke test.\n' | "${run_cmd[@]}" 2>&1 | tee "$run_output"
echo "== FastFlowLM inference transcript end =="

if ! awk '/Model RAW Output:/ {seen=1; next} seen && /[[:alpha:]]{4,}/ {found=1} END {exit found ? 0 : 1}' "$run_output"; then
  echo "FastFlowLM run completed but did not produce visible model output: $run_output" >&2
  exit 1
fi

stamp_file="$FLM_SMOKE_WORK_DIR/.bench-start"
: > "$stamp_file"

(
  cd "$FLM_SMOKE_WORK_DIR"
  "$FLM_BIN" bench "$FLM_MODEL" -i "$FLM_BENCH_CONFIG"
)

csv="$(find "$FLM_SMOKE_WORK_DIR" -maxdepth 1 -type f -name "bench_${safe_model}_*.csv" -newer "$stamp_file" -print | sort | tail -n 1)"
if [[ -z "$csv" ]]; then
  echo "FastFlowLM benchmark completed but did not write a fresh CSV" >&2
  exit 1
fi

if ! grep -q '^1,' "$csv"; then
  echo "FastFlowLM benchmark CSV does not contain a 1k result row: $csv" >&2
  exit 1
fi

echo "== FastFlowLM inference transcript replay =="
cat "$run_output"
echo "== End FastFlowLM inference transcript replay =="
echo "FastFlowLM HRX smoke: PASS -> $csv"
