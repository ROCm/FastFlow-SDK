#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# rocm-npu-staging e2e BUILD test (no NPU run). Proves the chain:
#   IRON emit -> Peano -> bootgen PDI -> hrx-xclbinutil(.xclbin) -> [HAL extracts PDI]
# Exits non-zero on any failure. DESIGN is an iron-tools programming_examples/basic
# design and defaults to matrix_scalar_add.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
STAGE="$(cd "$HERE/../.." && pwd)"
PROJ="$STAGE/projects"
DESIGN="${1:-matrix_scalar_add}"
mkdir -p "$STAGE/out"
fail() { echo "E2E FAIL: $*"; exit 1; }

# --- env: SOURCE iron-tools toolchain + Peano (wheel) + hrx-xclbinutil ---
# Source-only: iron-tools/aie-opt/aiecc/bootgen are built against ROCm amd-llvm.
# (Only Peano comes from a wheel.) No wheel fallback for iron-tools.
XU="$PROJ/hrx-xclbinutil/bld/tools/hrx-xclbinutil"
[ -x "$XU" ] || fail "hrx-xclbinutil not built (run: python dev.py build hrx-xclbinutil)"
mkdir -p "$STAGE/tools-bin"; ln -sf "$XU" "$STAGE/tools-bin/xclbinutil"
SVENV="$STAGE/.venv/lib/python3.12/site-packages"
[ -x "$STAGE/build/iron-tools/bin/aie-opt" ] || fail "iron-tools not built from source (run: python dev.py build iron-tools)"
echo "== toolchain: SOURCE iron-tools (built against ROCm amd-llvm) + Peano wheel =="
export PATH="$STAGE/tools-bin:$STAGE/build/iron-tools/bin:$STAGE/.venv/bin:$PATH"
export PYTHONPATH="$STAGE/build/iron-tools/python"
export PEANO_INSTALL_DIR="$SVENV/llvm-aie"
export AIEOPT_DIR="$PROJ/iron-tools/third_party/aie_api"
command -v aiecc.py >/dev/null || fail "aiecc.py not on PATH (run: python dev.py build iron-tools)"

# --- 1. build the design end-to-end -> .xclbin ---
# NPU2=1 selects devicename=npu2 (see programming_examples/makefile-common). The
# design's @iron.jit compile() writes build/final.xclbin plus the aiecc project dir
# build/final.prj/, which holds bootgen's main.pdi (used by step 3).
DDIR="$PROJ/iron-tools/programming_examples/basic/$DESIGN"
[ -d "$DDIR" ] || fail "no design $DDIR"
echo "== building $DESIGN =="
make -C "$DDIR" all NPU2=1 >/dev/null 2>"$STAGE/out/e2e-make.log" \
  || { tail -5 "$STAGE/out/e2e-make.log"; fail "design build failed"; }
XCLBIN="$DDIR/build/final.xclbin"
[ -f "$XCLBIN" ] || fail "no .xclbin produced"
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
PDI="$DDIR/build/final.prj/main.pdi"
[ -f "$HAL_LIB" ] || fail "projects/hrx not built (run: python dev.py build hrx — or build all)"
[ -n "$BASE_LIBS" ] || fail "hrx base libs not found (run: python dev.py build hrx)"
echo "== HAL artifact-contract check =="
LOADER="$STAGE/out/hal_pdi_loader"
# Use the same compiler dev.py builds the HAL libs with: canonical gcc, $CC override,
# clang only as a last resort (the manylinux CI image ships gcc, not clang).
CC="${CC:-$(command -v gcc || command -v cc || command -v clang)}"
[ -n "$CC" ] || fail "no C compiler found (need gcc/cc/clang)"
"$CC" -I"$PROJ/hrx/runtime/src" -I"$PROJ/hrx/build/cmake/runtime/src" \
  "$HERE/../hal-contract/hal_pdi_loader.c" -o "$LOADER" \
  -Wl,--start-group "$HAL_LIB" $BASE_LIBS -Wl,--end-group -lpthread -ldl -lm \
  2>"$STAGE/out/e2e-loader-link.log" \
  || { tail -10 "$STAGE/out/e2e-loader-link.log"; fail "HAL PDI loader build failed"; }
"$LOADER" "$XCLBIN" "$PDI" || fail "HAL did not extract matching PDI"

echo "E2E PASS: $DESIGN"
