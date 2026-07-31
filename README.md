# ROCm-NPU[^1]


## A Complete Software Stack for AMD Neural Processing Engines

ROCm-NPU is a bundle of software components that allow you to run and develop 
applications on AMD XDNA Neural Processing Engines (NPUs). It consists of:

- **FastFlowLM**[^2]: A set of optimized machine learning models
- **HRX**[^3]: A runtime that executes NPU applications
- **iron-tools**: A programming framework and compiler (formerly MLIR-AIE[^4])
- **iron-ops**: A set of common machine learning operator implementation 
   examples built using *iron-tools*

Note that these top-level components will recursively pull in additional 
requirements; each sub-project has more information on these.

[^1]: ROCm-NPU: ROCm for Neural Processing Units
[^2]: FastFlowLM: Fast Flow Language Models
[^3]: HRX: HIP Runtime Extended, HIP: Heterogeneous-compute Interface for Portability
[^4]: MLIR-AIE: Multi-Level Intermediate Representation for AI Engines

## Getting Started

What is your goal? Click to jump to the appropriate section.

- [I want to **run fast models** on my NPU-enabled device.](#run-fastflowlm-models-using-hrx)
- [I want to **develop my own models** and applications.](#develop-your-own-models-and-applications-using-iron-tools)


## Run *FastFlowLM* models using *HRX*

*FastFlowLM* provides a set of hand-optimized, blazingly fast large language 
models that run natively on your AMD NPU. Follow below instructions to run 
these.

> [!WARNING]
> **Current MoE xclbins can hang or produce invalid output**
>
> Current ROCm-NPU packages contain prebuilt AIE xclbins with incorrect address
> carry handling when an MoE weight buffer straddles a 4 GiB-aligned GPU virtual
> address boundary. The affected NPU2 models are:
>
> - `qwen3.6-moe:35b-a3b` (`Qwen3.6-35B-A3B-NPU2`)
> - `gpt-oss:20b` (`GPT-OSS-20B-NPU2`)
> - `gpt-oss-sg:20b` (`GPT-OSS-Safeguard-20b-NPU2`)
>
> This was observed on Windows, but the faulty address arithmetic is in the
> xclbin and can be triggered on any platform whose allocator places a weight
> buffer across such a boundary. The source fix has landed, but existing
> packages remain affected until these xclbins are regenerated and released.
> Retrying may change buffer placement, but it is not a reliable workaround.

This is the entry point for NPU runtime validation. It builds only minimal HRX
with the `amdxdna` HAL and the pinned FastFlowLM app, then pulls `qwen3:0.6b`,
runs an inference smoke test, and runs a 1k benchmark through HRX.

Ubuntu prerequisites:

```bash
sudo apt-get install -y \
  bash build-essential ca-certificates curl git libdrm-dev libipc-cmd-perl \
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

FastFlowLM's tokenizer build uses Cargo and currently needs `rustc >= 1.80`.
HRX is installed to a clean prefix and FastFlowLM consumes its exported
`hrx::hrx` CMake target. No HRX source or build-tree paths are part of the FLM
build contract.

Run the same workflow on both platforms. Use `python3` on Ubuntu and `python`
on Windows:

```bash
git submodule update --init projects/hrx
git submodule update --init --recursive projects/FastFlowLM

python3 dev.py setup --profile fastflowlm
python3 dev.py doctor fastflowlm
python3 dev.py build hrx --fresh
python3 dev.py build fastflowlm --fresh
python3 dev.py test-fastflowlm
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

CI uses the same archive to hand off from CPU build runners to separate NPU test
runners.

The smoke test requires the amdxdna driver and an NPU. It prints the FastFlowLM
transcript, rejects HRX runtime errors, requires the exact raw smoke phrase and
a coherent benchmark summary, and validates the benchmark CSV under
`build/FastFlowLM-smoke/`.

### Windows driver compatibility

The following Windows MCDM packages have been validated on Strix
(`PCI\VEN_1022&DEV_17F0`, Windows build `26200`) and Krackan Point
(BirmanPlus-KRK, Ryzen AI 7 350, `PCI\VEN_1022&DEV_17F0`, Windows build
`26100`). A passing row means Qwen3 0.6B FastFlowLM-to-HRX execution passed
(Strix: full smoke/CTS; Krackan: 20-iteration soak + 1k benchmark).

| AMD XDNA driver | Firmware evidence | FLM Qwen3 0.6B |
| --- | --- | --- |
| `32.0.203.314` | loaded `1.0.21.43` (`xrt-smi`) | Pass (Strix) |
| `32.0.203.329` | loaded `1.0.21.43` (`xrt-smi`) | Pass (Strix + Krackan) |
| `32.0.20101.3760` | loaded `1.1.2.64` (`xrt-smi`) | Pass (Strix + Krackan) |
| `32.0.20102.3930` | package candidates `1.1.2.64`, `1.1.2.65`, `1.1.2.73`, `1.5.5.391` (`npu-probe`); loaded `1.1.2.64` on Krackan (`xrt-smi`) | Pass (Strix + Krackan) |

Reboot after changing the driver before validating the NPU. A hot package
switch can leave device execution unhealthy even when Windows reports the new
driver and a healthy PnP status. Run `python dev.py npu-probe` after reboot to
record the active driver and the packaged firmware hashes before issuing NPU
work. This probe deliberately has no XRT dependency, so on Windows it reports
the firmware candidates found beside the active KMD. When available, the
package's `xrt-smi examine` performs a live XRT device query and reports the
loaded firmware version; that value comes from the XRT/KMD device interface,
not from Windows PnP metadata.

## Develop your own Models and Applications Using *iron-tools*

*iron-tools* provides programming abstractions and a compiler toolchain that 
target the AI Engine (AIE) array in the AMD NPU. A guide and programming 
examples document how to build your own applications. 
*FastFlowLM* was built using *iron-tools*. 

- The **[MLIR-AIE documentation](https://xilinx.github.io/mlir-aie/dev/)**
  contains the latest instructions for getting started with iron-tools,
  formerly known as MLIR-AIE.

> Note: *iron-tools* is currently undergoing renaming from its old name, 
> MLIR-AIE, so you might see references to both of these terms. They mean the
> same thing!

<details>
<summary>Click to expand instructions on <b>building iron-tools from source</b></summary>

Use this path when you also need the source-built LLVM/IRON toolchain and the
IRON end-to-end artifact checks.

The stack, built from source against one ROCm LLVM except for the Peano wheel:

```
LLVM+MLIR (ROCm amd-llvm pin, LLVM 23) ──┐
                                         ├─ iron-tools/IRON design ─ .xclbin
Peano (llvm-aie wheel, AIE codegen) ─────┘                         │
hrx-xclbinutil (xclbin packaging; vendored in mlir-aie) ───────────┘  ─ HRX amdxdna HAL extracts PDI
```

`hrx-xclbinutil` is no longer a standalone submodule: it lives inside mlir-aie at
`projects/iron-tools/third_party/hrx-xclbinutil` and is pulled in by the recursive
`iron-tools` submodule init. `dev.py build hrx-xclbinutil` builds that nested checkout.

System prerequisites for the compiler-stack flow (apt; cmake/ninja come from
the uv-managed environment after setup): `sudo apt-get install -y gcc g++ python3-dev git make
libssl-dev`. The canonical build toolchain is **gcc** (matching TheRock's
pinned manylinux gcc-toolset). To build with clang locally instead, install
`clang lld` and set `CC=clang CXX=clang++`.

```bash
python3 dev.py doctor compiler-stack # quick post-setup check: prereqs, pins, patches, prefetch
python3 dev.py doctor compiler-stack --deep # also replay every patch series in a temporary Git index
python3 dev.py setup              # .venv + cached uv install; cmake/ninja/Peano from wheels
python3 dev.py build hrx-xclbinutil
python3 dev.py build llvm         # LLVM+MLIR from the amd-llvm pin
python3 dev.py build iron-tools   # aie-opt/aie-translate/bootgen against build/llvm
python3 dev.py build hrx          # the amdxdna HAL (IREE runtime)
python3 dev.py build all          # all of the above, in order
python3 dev.py test [DESIGN]      # e2e build test, no NPU run
python3 dev.py test-iron-ops [OP] # build the iron-ops operators, no NPU run
python3 dev.py deps               # print requirements/compiler-stack.txt
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
[`requirements/iron-ops-test.txt`](requirements/iron-ops-test.txt); the build path does not need them).

</details>

## Additional documentation

- [ROCm NPU wheel releases](docs/wheel-releases.md): wheel versioning,
  validation, and publication policy.
