# rocm-npu-staging

A minimal ROCm NPU stack, built **together from source**. Self-contained: pinned revisions in
[manifests/versions.toml](manifests/versions.toml), Python deps in
[requirements.txt](requirements.txt), orchestrated by [dev.py](dev.py).

The stack (everything built from source against one ROCm LLVM; only Peano is a wheel):

```
LLVM+MLIR (ROCm amd-llvm pin, LLVM 23) ─┐
                                         ├─ mlir-aie/IRON ─┐
Peano (llvm-aie wheel, AIE codegen) ─────┘                 ├─ FastFlowLM FLM_Xclbin ─ .xclbin
hrx-xclbinutil (xclbin packaging) ──────────────────────┘   ─ HRX amdxdna HAL extracts PDI
```

`projects/` are git submodules: `llvm-project` (jtuyls/llvm-project @ the ROCm
amd-llvm pin), `mlir-aie`, `fastflowlm` (ROCm/FastFlowLM), `hrx`, and
`hrx-xclbinutil` (transitional).

## Quickstart (`dev.py` — setup + build + test)

System prerequisites (apt — the only manual step; cmake/ninja come from pip):
`sudo apt-get install -y clang lld python3-dev git build-essential`.
Run `python dev.py doctor` to check and get the exact install command for anything missing.

```bash
python dev.py doctor             # check system (apt) prereqs
python dev.py setup              # .venv + Python deps (hash-pinned requirements.lock); cmake/ninja/Peano via pip
python dev.py build hrx-xclbinutil
python dev.py build llvm         # LLVM+MLIR from the amd-llvm pin (long; uses all cores)
python dev.py build mlir-aie     # aie-opt/aie-translate/bootgen against build/llvm
python dev.py build hrx          # the amdxdna HAL (IREE runtime)
python dev.py build flm          # an FLM design -> .xclbin (full source toolchain)
python dev.py build all          # all of the above, in order
python dev.py test               # e2e BUILD test (no NPU run)
python dev.py deps               # print requirements.txt
```

Build-only on a host without an NPU. The **e2e test** builds an FLM design
end-to-end to a `.xclbin` with the source toolchain, then verifies the amdxdna HAL
extracts the PDI byte-for-byte == bootgen's — closing the artifact contract. It
requires `projects/hrx` to be built (`dev.py build all` does this). Running on
hardware needs the amdxdna driver + an NPU.
