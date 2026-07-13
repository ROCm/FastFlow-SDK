# rocm-npu-staging

A staging repo for validating the ROCm NPU software stack from pinned source
revisions. The quickest path is FastFlowLM running directly on HRX's `amdxdna`
runtime; the full compiler-stack path additionally builds LLVM, IRON tools, and
xclbin packaging.

Pinned revisions live in [manifests/versions.toml](manifests/versions.toml),
Python deps in [requirements.txt](requirements.txt), and common commands are
orchestrated by [dev.py](dev.py).

`projects/` are git submodules: `hrx`, `FastFlowLM`, `llvm-project`
(jtuyls/llvm-project @ the ROCm amd-llvm pin), `iron-tools`
(Xilinx/mlir-aie), `iron-ops` (amd/IRON — the operator library, built and run
against the source toolchain), and `hrx-xclbinutil` (transitional).

## FastFlowLM HRX Quickstart

This is the entry point for NPU runtime validation. It builds only minimal HRX
with the `amdxdna` HAL and the pinned FastFlowLM app, then pulls `qwen3:0.6b`,
runs an inference smoke test, and runs a 1k benchmark through HRX.

Ubuntu prerequisites:

```bash
sudo apt-get install -y \
  build-essential cmake curl git libavcodec-dev libavformat-dev libavutil-dev \
  libboost-program-options-dev libcurl4-openssl-dev libdrm-dev libfftw3-dev \
  libreadline-dev libswresample-dev libswscale-dev ninja-build pkg-config python3
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- -y --profile minimal --default-toolchain stable
export PATH="$HOME/.cargo/bin:$PATH"
```

FastFlowLM's tokenizer build uses Cargo and currently needs `rustc >= 1.80`.
No special runtime environment needs to be sourced; `dev.py test-fastflowlm`
sets the HRX library path and runtime variables for the smoke run.

```bash
git submodule update --init projects/hrx
git submodule update --init --recursive projects/FastFlowLM

RNS_BUILD_JOBS=8 python3 dev.py build hrx
RNS_BUILD_JOBS=8 python3 dev.py build fastflowlm
python3 dev.py test-fastflowlm
```

Use `RNS_BUILD_JOBS=8` on shared local machines if full parallelism is
disruptive. CI uses the runner's default parallelism.

The smoke test requires the amdxdna driver and an NPU. It prints the FastFlowLM
transcript, rejects HRX runtime errors, validates that the raw model output
contains the smoke phrase, and writes the benchmark CSV under
`build/FastFlowLM-smoke/`.

## Full Compiler-Stack Quickstart

Use this path when you also need the source-built LLVM/IRON toolchain and the
IRON end-to-end artifact checks.

The stack, built from source against one ROCm LLVM except for the Peano wheel:

```
LLVM+MLIR (ROCm amd-llvm pin, LLVM 23) ─┐
                                         ├─ iron-tools/IRON design ─ .xclbin
Peano (llvm-aie wheel, AIE codegen) ─────┘                         │
hrx-xclbinutil (xclbin packaging) ─────────────────────────────────┘  ─ HRX amdxdna HAL extracts PDI
```

System prerequisites for the compiler-stack flow (apt; cmake/ninja come from
pip after setup): `sudo apt-get install -y gcc g++ python3-dev git make
libssl-dev`. The canonical build toolchain is **gcc** (matching TheRock's
pinned manylinux gcc-toolset). To build with clang locally instead, install
`clang lld` and set `CC=clang CXX=clang++`.

```bash
python3 dev.py doctor             # check system prereqs
python3 dev.py setup              # .venv + Python deps; cmake/ninja/Peano via pip
python3 dev.py build hrx-xclbinutil
python3 dev.py build llvm         # LLVM+MLIR from the amd-llvm pin
python3 dev.py build iron-tools   # aie-opt/aie-translate/bootgen against build/llvm
python3 dev.py build hrx          # the amdxdna HAL (IREE runtime)
python3 dev.py build all          # all of the above, in order
python3 dev.py test [DESIGN]      # e2e build test, no NPU run
python3 dev.py test-iron-ops [OP] # build the iron-ops operators, no NPU run
python3 dev.py deps               # print requirements.txt
```

Build-only works on a host without an NPU. The **e2e test** builds an
iron-tools IRON design (`programming_examples/basic/matrix_scalar_add` by
default) end-to-end to a `.xclbin` with the source toolchain, then verifies the
amdxdna HAL extracts the PDI byte-for-byte == bootgen's. Running on hardware
needs the amdxdna driver and an NPU.

Another design can be passed as `DESIGN`, but it must be Peano-capable and use
`makefile-common`'s `jit_xclbin` rule; the `jit_xclbin_elf` designs
additionally emit `insts.elf`, which needs `aiebu-asm` (not part of this stack).

**`test-iron-ops`** builds the same chain from the other direction: the `iron-ops`
operator library (`ElementwiseAdd`, …) rather than an iron-tools example, compiling the
operator's own `aie_kernels/*.cc` through Peano. iron-ops is developed against the
`mlir_aie` *wheel*, so this is what catches its drift from our source pin — on a CPU
runner rather than on the NPU. Which operators it covers is
[tests/iron-ops/suite.py](tests/iron-ops/suite.py); adding a row extends it.

On a machine **with** an NPU, `dev.py test-npu` dispatches an iron-tools design through
the HAL, and `dev.py test-npu-iron-ops` runs the iron-ops operators' own tests on the
device (their references, their tolerances) with `IRON_RUNTIME=hrx`. The latter needs
their test deps: `dev.py setup --iron-ops-tests` (torch, pytest — see
[requirements-iron-ops.txt](requirements-iron-ops.txt); the build path does not need them).
