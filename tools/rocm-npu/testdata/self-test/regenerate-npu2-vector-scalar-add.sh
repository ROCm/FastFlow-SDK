#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
FIXTURE="${SCRIPT_DIR}/npu2-vector-scalar-add"
PACKAGE_CASE="${ROOT}/build/npu-test-package/cases/vector_scalar_add"
DESIGN_BUILD="${ROOT}/projects/iron-tools/programming_examples/basic/vector_scalar_add/build"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

cd "${ROOT}"
"${PYTHON}" dev.py package-npu-test-package

if [[ ! -f "${PACKAGE_CASE}/kernel.xadx" ]]; then
  echo "missing regenerated XADX: ${PACKAGE_CASE}/kernel.xadx" >&2
  exit 1
fi

cp "${PACKAGE_CASE}/kernel.xadx" "${FIXTURE}/kernel.xadx"

FIXTURE="${FIXTURE}" DESIGN_BUILD="${DESIGN_BUILD}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
import struct
from pathlib import Path

fixture = Path(os.environ["FIXTURE"])
manifest_path = fixture / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

input_bytes = b"".join(struct.pack("<i", value) for value in range(1, 1025))
expected_bytes = b"".join(struct.pack("<i", value) for value in range(2, 1026))

artifacts = manifest["producer"]["artifacts"]
artifacts["xadx_sha256"] = hashlib.sha256(
    (fixture / "kernel.xadx").read_bytes()
).hexdigest()
design_build = Path(os.environ["DESIGN_BUILD"])
artifacts["xclbin_sha256"] = hashlib.sha256(
    (design_build / "final.xclbin").read_bytes()
).hexdigest()
artifacts["transaction_sha256"] = hashlib.sha256(
    (design_build / "insts.bin").read_bytes()
).hexdigest()
artifacts["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
artifacts["expected_sha256"] = hashlib.sha256(expected_bytes).hexdigest()

manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

echo "updated ${FIXTURE}/kernel.xadx and manifest hashes"
echo "review producer revisions in ${FIXTURE}/manifest.json before committing"
