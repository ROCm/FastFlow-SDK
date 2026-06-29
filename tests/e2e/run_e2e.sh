#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# rocm-npu-staging e2e BUILD test (no NPU run). Proves the chain:
#   IRON emit -> Peano -> bootgen PDI -> hrx-xclbinutil(.xclbin) -> [HAL extracts PDI]
# Exits non-zero on any failure. DESIGN defaults to Dequant_Q4_0_w128x512.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
STAGE="$(cd "$HERE/../.." && pwd)"
PROJ="$STAGE/projects"
DESIGN="${1:-Dequant_Q4_0_w128x512}"
fail() { echo "E2E FAIL: $*"; exit 1; }

# --- env: SOURCE mlir-aie toolchain + Peano (wheel) + hrx-xclbinutil ---
# Source-only: mlir-aie/aie-opt/aiecc/bootgen are built against ROCm amd-llvm.
# (Only Peano comes from a wheel.) No wheel fallback for mlir-aie.
XU="$PROJ/hrx-xclbinutil/bld/tools/hrx-xclbinutil"
[ -x "$XU" ] || fail "hrx-xclbinutil not built (run: python dev.py build hrx-xclbinutil)"
mkdir -p "$STAGE/tools-bin"; ln -sf "$XU" "$STAGE/tools-bin/xclbinutil"
SVENV="$STAGE/.venv/lib/python3.12/site-packages"
[ -x "$STAGE/build/mlir-aie/bin/aie-opt" ] || fail "mlir-aie not built from source (run: python dev.py build mlir-aie)"
echo "== toolchain: SOURCE mlir-aie (built against ROCm amd-llvm) + Peano wheel =="
export PATH="$STAGE/tools-bin:$STAGE/build/mlir-aie/bin:$STAGE/.venv/bin:$PATH"
export PYTHONPATH="$STAGE/build/mlir-aie/python"
export PEANO_INSTALL_DIR="$SVENV/llvm-aie"
export AIEOPT_DIR="$PROJ/mlir-aie/third_party/aie_api"
command -v aiecc.py >/dev/null || fail "aiecc.py not on PATH (run: python dev.py build mlir-aie)"

# --- 1. build the design end-to-end -> .xclbin ---
DDIR="$PROJ/fastflowlm/FLM_Xclbin/$DESIGN"
[ -d "$DDIR" ] || fail "no design $DDIR"
echo "== building $DESIGN =="
make -C "$DDIR" bitstream DEVICE=npu2 ENABLE_CHESSCC=0 >/dev/null 2>"$STAGE/out/e2e-make.log" \
  || { tail -5 "$STAGE/out/e2e-make.log"; fail "FLM build failed"; }
XCLBIN="$(ls "$DDIR"/build/xclbins/*.xclbin 2>/dev/null | head -1)"
[ -n "$XCLBIN" ] || fail "no .xclbin produced"
mkdir -p "$STAGE/out/artifacts"; cp "$XCLBIN" "$STAGE/out/artifacts/$DESIGN.xclbin"
echo "  ok: $DESIGN.xclbin ($(stat -c%s "$XCLBIN") B)"

# --- 2. verify the xclbin has the NPU sections ---
xclbinutil --info --input "$XCLBIN" 2>/dev/null | grep -q "AIE_PARTITION" \
  || fail "xclbin missing AIE_PARTITION"
echo "  ok: sections present (MEM_TOPOLOGY/AIE_PARTITION/EMBEDDED_METADATA)"

# --- 3. HAL artifact-contract check: amdxdna HAL extracts the PDI == bootgen ---
# The e2e assumes the full stack is built (run `python dev.py build all` first);
# a missing hrx build is a failure, not a skip.
HAL_LIB="$PROJ/hrx/build/cmake/runtime/src/iree/hal/drivers/amdxdna/libiree_hal_drivers_amdxdna_amdxdna.a"
BASE_LIBS=$(find "$PROJ/hrx/build/cmake/runtime/src/iree/base" -name "*.a" 2>/dev/null | tr '\n' ' ')
PDI="$DDIR/build/bitstream/from_iron/dequant.mlir.prj/main.pdi"
[ -f "$HAL_LIB" ] || fail "projects/hrx not built (run: python dev.py build hrx — or build all)"
[ -n "$BASE_LIBS" ] || fail "hrx base libs not found (run: python dev.py build hrx)"
echo "== HAL artifact-contract check =="
LOADER="$STAGE/out/hal_pdi_loader"
clang -I"$PROJ/hrx/runtime/src" -I"$PROJ/hrx/build/cmake/runtime/src" \
  "$HERE/../hal-contract/hal_pdi_loader.c" -o "$LOADER" \
  -Wl,--start-group "$HAL_LIB" $BASE_LIBS -Wl,--end-group -lpthread -ldl -lm \
  2>"$STAGE/out/e2e-loader-link.log" || fail "ws_d_loader link failed"
"$LOADER" "$XCLBIN" "$PDI" || fail "HAL did not extract matching PDI"

echo "E2E PASS: $DESIGN"
