# Build FastFlow SDK from source

Use this guide to build FastFlow SDK from this checkout instead of installing a
release. The steps below build the runtime (HRX and FastFlowLM): a minimal HRX
with the `amdxdna` HAL and the pinned FastFlowLM app, which can pull
`qwen3:0.6b` and run a smoke test. The compiler-stack instructions cover the
source-built LLVM/IRON toolchain and the IRON end-to-end artifact checks.

Ubuntu prerequisites:

```bash
sudo apt-get install -y \
  bash build-essential ca-certificates curl git libdrm-dev \
  linux-libc-dev make nasm perl pkg-config python3 python3-venv unzip zip
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- -y --profile minimal --default-toolchain 1.97.0
export PATH="$HOME/.cargo/bin:$PATH"
```

Extra Linux packaging prerequisites:

```bash
sudo apt-get install -y patchelf zstd
```

Windows prerequisites (PowerShell, no WSL):

```powershell
winget install --exact --id Microsoft.VisualStudio.2022.BuildTools `
  --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install --exact --id Python.Python.3.12
winget install --exact --id Git.Git
winget install --exact --id Rustlang.Rustup
rustup toolchain install 1.97.0 --profile minimal
rustup default 1.97.0
git config --global core.longpaths true
```

On Linux and Windows, `dev.py setup --profile fastflowlm` installs the pinned
Python and native dependencies into gitignored repository caches. The host
package manager supplies only build tools and platform interfaces. CMake and
Ninja come from `requirements/fastflowlm.lock`.

Python environment manifests live under `requirements/`. Each editable `.txt`
source has a generated, hash-pinned `.lock` companion; setup profiles select
the smallest environment that supports their build or test responsibility.

This checkout pins Rust 1.97.0 for the FastFlowLM tokenizer build. HRX is
installed to a clean prefix and FastFlowLM consumes its exported `hrx::hrx`
CMake target. No HRX source or build-tree paths are part of the FLM build
contract.

## Build the runtime and FastFlowLM

Run the runtime and FastFlowLM workflow on either platform. Use the matching
native command shell:

Windows (PowerShell):

```powershell
git clone https://github.com/ROCm/FastFlow-SDK.git
cd fastflow

git submodule update --init projects/hrx
git submodule update --init --recursive projects/FastFlowLM

python dev.py setup --profile fastflowlm
python dev.py doctor fastflowlm
python dev.py build hrx --fresh
python dev.py build fastflowlm --fresh
python dev.py test-fastflowlm
```

Linux (Bash):

```bash
git clone https://github.com/ROCm/FastFlow-SDK.git
cd fastflow

git submodule update --init projects/hrx
git submodule update --init --recursive projects/FastFlowLM

python3 dev.py setup --profile fastflowlm
python3 dev.py doctor fastflowlm
python3 dev.py build hrx --fresh
python3 dev.py build fastflowlm --fresh
python3 dev.py test-fastflowlm
```

`build fastflowlm-xrt` builds a diagnostic, XRT-backed FastFlowLM into a
separate `build/FastFlowLM-xrt-install/` tree (never shipped in the HRX wheels).
It links the host XRT install discovered from `$XILINX_XRT`, the platform
default (`/opt/xilinx/xrt` on Linux; Program Files or `C:\Xilinx\XRT` on
Windows), or an XRT runtime already on `PATH`. It installs an FHS `bin/flm`
tree; it does not bundle a relocatable XRT runtime. Point
`fastflow qualify --bench --xrt-flm <path>` or
`fastflow qualify --all-models --xrt-flm <path>` at this tree for the optional
HRX-vs-XRT comparison (decode metrics on `--bench`; greedy numerical match of
the single-prompt run on `--all-models`). Qualification sets `XILINX_XRT` and
the loader path (`LD_LIBRARY_PATH` on Linux, `PATH` on Windows) only on that
comparison child.

```bash
python3 dev.py build fastflowlm-xrt --fresh
```

`package-fastflowlm` creates a relocatable Windows `.zip` or Linux `.tar.zst`,
verifies it, and re-extracts it under `build/FastFlowLM-extracted/`. To recheck
the packaging prerequisites and create the archive, run:

```bash
python3 dev.py doctor fastflowlm-packaging
python3 dev.py package-fastflowlm
```

To recheck the extracted archive later, run:

```bash
python3 dev.py verify-fastflowlm
```

The smoke test requires the amdxdna driver and an NPU. It prints the FastFlowLM
transcript, rejects HRX runtime errors, requires the exact raw smoke phrase and
a coherent benchmark summary, and validates the benchmark CSV under
`build/FastFlowLM-smoke/`.

## Build the compiler stack

Use this path when you need the source-built LLVM/IRON toolchain and the IRON
end-to-end artifact checks. Use `python3` on Linux and `python` on Windows.

`python dev.py --help` (Windows) or `python3 dev.py --help` (Linux) is the
canonical, generated inventory of developer commands; use the same interpreter
prefix with `dev.py COMMAND --help` for command flags.

The stack, built from source against one ROCm LLVM except for the Peano wheel:

```
LLVM+MLIR (ROCm amd-llvm pin, LLVM 24) ──┐
                                         ├─ iron-tools/IRON design ─ .xclbin
Peano (llvm-aie wheel, AIE codegen) ─────┘                         │
hrx-xclbinutil (xclbin packaging; vendored in mlir-aie) ───────────┘  ─ HRX amdxdna HAL extracts PDI
```

`hrx-xclbinutil` lives inside iron-tools at
`projects/iron-tools/third_party/hrx-xclbinutil` (vendored in-tree).
`dev.py build hrx-xclbinutil` builds that tree.

Additional Linux prerequisites for the compiler-stack flow (CMake and Ninja
come from the uv-managed environment after setup):

```bash
sudo apt-get install -y gcc g++ python3-dev git make libssl-dev
```

Linux uses **gcc** by default. To build with clang locally, install `clang lld`
and set `CC=clang CXX=clang++`.

On Windows, open Git Bash from a Visual Studio developer environment and use
the MSVC toolchain with `clang-cl`. Install the OpenSSL development files and
set `OPENSSL_ROOT_DIR` to their installation. CMake and Ninja come from the
Python environment on both platforms.

On Windows, configure Git before initializing the submodules:

```bash
git config --global core.autocrlf false
git config --global core.eol lf
git config --global core.longpaths true
```

These settings keep the source and patch files in LF format and allow long
Windows paths. Set `CC` and `CXX` to `clang-cl` in the Visual Studio developer
environment before running the Windows commands below.

Linux:

```bash
git submodule update --init --recursive projects/iron-tools
python3 dev.py doctor compiler-stack # quick post-setup check: prereqs, pins, patches, prefetch
python3 dev.py doctor compiler-stack --deep # also replay every patch series in a temporary Git index
python3 dev.py setup              # .venv + cached uv install; cmake/ninja/Peano from wheels
python3 dev.py build hrx-xclbinutil
python3 dev.py build llvm         # LLVM+MLIR from the amd-llvm pin
python3 dev.py build iron-tools   # aie-opt/aie-translate/bootgen against build/llvm
python3 dev.py build hrx          # the amdxdna HAL (IREE runtime)
python3 dev.py build all          # native CLI + compiler stack + HRX, in order
python3 dev.py test [DESIGN]      # e2e build test, no NPU run
python3 dev.py test-npu-jit --build-only # compiler + IRON JIT smoke, no NPU run
python3 dev.py test-iron-ops [OP] # build the iron-ops operators, no NPU run
python3 dev.py deps               # print requirements/compiler-stack.txt
```

Windows (Git Bash):

```bash
git submodule update --init --recursive projects/iron-tools
python dev.py doctor compiler-stack # quick post-setup check: prereqs, pins, patches, prefetch
python dev.py doctor compiler-stack --deep # also replay every patch series in a temporary Git index
python dev.py setup              # .venv + cached uv install; cmake/ninja/Peano from wheels
python dev.py build hrx-xclbinutil
python dev.py build llvm         # LLVM+MLIR from the amd-llvm pin
python dev.py build iron-tools   # aie-opt/aie-translate/bootgen against build/llvm
python dev.py build hrx          # the amdxdna HAL (IREE runtime)
python dev.py build all          # native CLI + compiler stack + HRX, in order
python dev.py test [DESIGN]      # e2e build test, no NPU run
python dev.py test-npu-jit --build-only # compiler + IRON JIT smoke, no NPU run
python dev.py test-iron-ops [OP] # build the iron-ops operators, no NPU run
python dev.py deps               # print requirements/compiler-stack.txt
```

The Windows CI uses the quick compiler-stack doctor check above; the
`--deep` patch replay is currently part of the Linux validation path. Use
`dev.py COMMAND --help` for the available options.

Build-only works on a host without an NPU. The **e2e test** builds an
iron-tools IRON design (`programming_examples/basic/matrix_scalar_add` by
default) end-to-end to a `.xclbin` with the source toolchain, then verifies
that the amdxdna HAL extracts the same PDI bytes as bootgen. Running on
hardware needs the amdxdna driver and an NPU.

Another design can be passed as `DESIGN`, but it must be Peano-capable and use
`makefile-common`'s `jit_xclbin` rule. The `jit_xclbin_elf` designs additionally
emit `insts.elf` using the pinned `aiebu-asm` built by `dev.py build all` and
shipped in the compiler product.

**`test-iron-ops`** builds the same chain from the other direction: the
`iron-ops` operator library (`ElementwiseAdd`, ...) rather than an iron-tools
example, compiling mlir-aie's `aie_kernels/*.cc` through Peano.
iron-ops is developed against the `mlir_aie` *wheel*, so this is what catches
its drift from our source pin, without an NPU. Which operators it covers is
[`build_tools/iron.py`](../build_tools/iron.py) `OPERATORS`; adding a row
extends it.

On a machine **with** an NPU, `dev.py test-npu` dispatches an iron-tools design
through the HAL, `dev.py test-npu-jit` compiles and dispatches the `@iron.jit`
smoke design, and `dev.py test-npu-iron-ops` runs the iron-ops operators' own
tests on the device (their references, their tolerances) with
`NPU_RUNTIME=hrx`. The latter needs their test deps:
`dev.py setup --iron-ops-tests` (torch, pytest -- see
[`requirements/iron-ops-test.txt`](../requirements/iron-ops-test.txt); the
build path does not need them).
