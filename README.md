# ROCm-NPU

ROCm-NPU is a software stack for running and developing applications on
AMD XDNA Neural Processing Engines (NPUs).

> [!IMPORTANT]
> **This is an early release.** Interfaces, packages, models, and supported
> hardware or driver combinations can change between snapshots. Use a
> [GitHub Release](https://github.com/ROCm/rocm-npu/releases) for
> installable artifacts and for the driver versions validated with that
> build. These packages are not published to PyPI.

It consists of:

- **FastFlowLM** (Fast Flow Language Models): NPU-optimized language-model
  inference
- **HRX** (HIP Runtime Extended): a runtime that executes NPU applications
- **iron-tools**: a programming framework and compiler for the AI Engine
  (AIE) array, formerly MLIR-AIE
- **iron-ops**: example machine-learning operators built with *iron-tools*

The native `rocm-npu` CLI ships with the runtime and reports device, driver,
and runtime health. Building from this repository also pulls additional
subproject requirements; each subproject has more detail. See
[CONTRIBUTING.md](CONTRIBUTING.md) if you want to change the stack.

## Supported platforms

This early snapshot targets **Linux and Windows** on **x86_64**, with
**AMD Ryzen AI NPUs that implement XDNA2 (NPU2)**. Windows instructions are
native PowerShell; WSL is not the supported Windows path.

Validated NPU driver versions are listed in each GitHub release. After
installing or changing a driver, run `rocm-npu self-test`. If that fails,
reboot and retry; a hot package switch can occasionally leave the device
unhealthy even when the OS reports the new driver as installed.
`rocm-npu probe` and `rocm-npu doctor` report the active driver, firmware,
and runtime health.

Runtime and FastFlowLM Python wheels require Python 3.11 or newer. Compiler
wheels require Python 3.12, 3.13, or 3.14. The standalone FastFlowLM/HRX
archives do not require Python.

## Getting started

- [I want to **install a release** and run models.](#install-a-release-and-run-models)
- [I want to **develop my own models** and applications.](#develop-your-own-models-and-applications-using-iron-tools)
- [I want to **build ROCm-NPU from source**.](#build-rocm-npu-from-source)

## Install a release and run models

Download a [GitHub Release](https://github.com/ROCm/rocm-npu/releases).
The release description lists the driver versions validated for that build.

### Standalone archive

Each release includes a relocatable FastFlowLM + HRX archive:

- Windows: `rocm-npu-windows-x86_64.zip`
- Linux: `rocm-npu-linux-x86_64.tar.zst`

Extract it and run the binaries from the archive root (`bin/flm` and
`bin/rocm-npu`):

```powershell
Expand-Archive .\rocm-npu-windows-x86_64.zip -DestinationPath .
cd .\rocm-npu-windows-x86_64
.\bin\rocm-npu.exe doctor
.\bin\flm.exe list
.\bin\flm.exe run qwen3:0.6b
```

```bash
tar --zstd -xf rocm-npu-linux-x86_64.tar.zst
cd rocm-npu-linux-x86_64
./bin/rocm-npu doctor
./bin/flm list
./bin/flm run qwen3:0.6b
```

`rocm-npu self-test` also dispatches a packaged NPU2 fixture. The first
`flm run` downloads model assets and needs Hugging Face access. `flm list`
shows the models included in that build.

### Python wheels

Wheels ship on the same GitHub release, not on PyPI. Installing
`rocm-npu-fastflowlm` pulls the matching `rocm-npu-runtime` wheel. Replace
`<tag>` and `<version>` with values from the release (tags look like
`rocm-npu-v<version>`):

```powershell
py -m pip install --only-binary=:all: `
  --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/<tag>" `
  "rocm-npu-fastflowlm==<version>"
rocm-npu doctor
flm run qwen3:0.6b
```

```bash
python3 -m pip install --only-binary=:all: \
  --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/<tag>" \
  "rocm-npu-fastflowlm==<version>"
rocm-npu doctor
flm run qwen3:0.6b
```

To install the compiler from the same release (use `py -m pip` on Windows):

```bash
python3 -m pip install --only-binary=:all: \
  --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/<tag>" \
  "rocm-npu-compiler==<version>"
```

See [native runtime CLI](docs/runtime-cli.md) for `rocm-npu` commands.

## Develop your own Models and Applications Using *iron-tools*

*iron-tools* provides programming abstractions and a compiler toolchain that
target the AI Engine (AIE) array in the AMD NPU. A guide and programming
examples document how to build your own applications.
*FastFlowLM* was built using *iron-tools*.

If you installed a release, install `rocm-npu-compiler` from the same tag
(see above) and follow the iron-tools documentation:

- The **[MLIR-AIE documentation](https://xilinx.github.io/mlir-aie/dev/)**
  contains the latest instructions for getting started with iron-tools,
  formerly known as MLIR-AIE.

> [!NOTE]
> *iron-tools* is currently undergoing renaming from its old name, MLIR-AIE,
> so you might see references to both of these terms. They mean the same
> thing.

To build the compiler toolchain from this checkout instead of the release
wheel, see [Build ROCm-NPU from source](#build-rocm-npu-from-source).

## Build ROCm-NPU from source

Use this path to build ROCm-NPU from this checkout instead of installing a
release. The steps below build the runtime (HRX and FastFlowLM): a minimal HRX
with the `amdxdna` HAL and the pinned FastFlowLM app, which can pull
`qwen3:0.6b` and run a smoke test. Building the compiler toolchain from source
is in the details at the end of this section.

```bash
git clone https://github.com/ROCm/rocm-npu.git
cd rocm-npu
```

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

This checkout pins Rust 1.97.0 for the FastFlowLM tokenizer build. HRX is
installed to a clean prefix and FastFlowLM consumes its exported `hrx::hrx`
CMake target. No HRX source or build-tree paths are part of the FLM build
contract.

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

`build fastflowlm-xrt` builds a diagnostic, XRT-backed FastFlowLM into a
separate `build/FastFlowLM-xrt-install/` tree (never shipped in the HRX wheels).
It links the host XRT install (`$XILINX_XRT` or `/opt/xilinx/xrt`) and installs
an FHS `bin/flm` tree; it does not bundle a relocatable XRT runtime. Point
`rocm-npu qualify --bench --xrt-flm <path>` at this tree for the optional
HRX-vs-XRT comparison (Linux; qualification sets `XILINX_XRT` /
`LD_LIBRARY_PATH` only on that comparison child).

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

<details>
<summary>Click to expand instructions on <b>building the compiler from source</b></summary>

Use this path when you also need the source-built LLVM/IRON toolchain and the
IRON end-to-end artifact checks.

`python3 dev.py --help` is the canonical, generated inventory of developer
commands; use `python3 dev.py COMMAND --help` for command flags.

The stack, built from source against one ROCm LLVM except for the Peano wheel:

```
LLVM+MLIR (ROCm amd-llvm pin, LLVM 23) ──┐
                                         ├─ iron-tools/IRON design ─ .xclbin
Peano (llvm-aie wheel, AIE codegen) ─────┘                         │
hrx-xclbinutil (xclbin packaging; vendored in mlir-aie) ───────────┘  ─ HRX amdxdna HAL extracts PDI
```

`hrx-xclbinutil` lives inside iron-tools at
`projects/iron-tools/third_party/hrx-xclbinutil` and is pulled in by the
recursive `iron-tools` submodule init. `dev.py build hrx-xclbinutil` builds
that nested checkout.

System prerequisites for the compiler-stack flow (apt; cmake/ninja come from
the uv-managed environment after setup): `sudo apt-get install -y gcc g++ python3-dev git make
libssl-dev`. The canonical Linux compiler is **gcc**. To build with clang
locally instead, install `clang lld` and set `CC=clang CXX=clang++`.

```bash
python3 dev.py doctor compiler-stack # quick post-setup check: prereqs, pins, patches, prefetch
python3 dev.py doctor compiler-stack --deep # also replay every patch series in a temporary Git index
python3 dev.py setup              # .venv + cached uv install; cmake/ninja/Peano from wheels
python3 dev.py build hrx-xclbinutil
python3 dev.py build llvm         # LLVM+MLIR from the amd-llvm pin
python3 dev.py build iron-tools   # aie-opt/aie-translate/bootgen against build/llvm
python3 dev.py build hrx          # the amdxdna HAL (IREE runtime)
python3 dev.py build all          # native CLI + compiler stack + HRX, in order
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

- [Native runtime CLI](docs/runtime-cli.md): `rocm-npu version`, `probe`,
  `doctor`, and `self-test`
- [Model qualification](docs/qualification.md): optional qualification
  profiles and evidence
- [Third-party notices](docs/third-party-notices.md)
- [Contributing](CONTRIBUTING.md)

## License

This repository is licensed under the [MIT License](LICENSE).

FastFlowLM NPU kernel binaries shipped in releases are not open source and
are subject to additional terms. Those terms are included with the packages
as `licenses/FastFlowLM-TERMS.md`.

Please file issues in this repository. See [SECURITY.md](SECURITY.md) to
report a vulnerability. This is an early snapshot, so expect breaking
changes between releases.
