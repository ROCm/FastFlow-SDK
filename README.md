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

### Key terms

- **XDNA2 / NPU2**: the AMD Ryzen™ AI NPU generation targeted by this release.
- **HRX**: the runtime layer that loads and dispatches NPU applications.
- **AIE**: AMD's AI Engine array, the programmable compute fabric targeted by
  *iron-tools*.
- **PDI**: the processor device image extracted from a compiled design and
  loaded by the NPU runtime.
- **Peano**: the LLVM-based AIE compiler toolchain used to generate device code.

## Intended audience

This early-access release enables developers to run FastFlowLM models on AMD Ryzen AI XDNA2 NPUs, validate the NPU runtime with `rocm-npu`, and experiment with the `iron-tools` compiler stack.

## Supported models

Each release includes a registry of the FastFlowLM models validated for that build. Run `flm list` after installation to see the models available in your release.

The examples use `qwen3:0.6b` because it is the smallest recommended model for verifying NPU inference.

For the complete model catalog, see the [FastFlowLM model documentation](https://fastflowlm.com/docs/models/).

## Supported platforms

This early snapshot targets **Linux and Windows** on **x86_64**, with
**AMD Ryzen AI NPUs that implement XDNA2 (NPU2)**. Windows instructions are
native PowerShell; WSL is not the supported Windows path.

Validated NPU driver versions are listed in each GitHub release. After
installing or changing a driver, run `rocm-npu self-test`. If that fails,
reboot and retry; a hot package switch can occasionally leave the device
unhealthy even when the OS reports the new driver as installed.
`rocm-npu probe` and `rocm-npu doctor` report the active driver, firmware, NPU
power mode where supported, and runtime health. `rocm-npu powermode` reports the
current mode. Use `rocm-npu powermode <mode>` to change it. Power-mode changes
may require elevated privileges depending on the installed driver.

Runtime and FastFlowLM Python wheels require Python 3.11 or newer. Compiler
wheels require Python 3.12, 3.13, or 3.14. The standalone FastFlowLM/HRX
archives do not require Python.

## Getting started

- [I want to **install a release** and run models.](#install-a-release-and-run-models)
- [I want to **develop my own models** and applications.](#develop-your-own-models-and-applications-using-iron-tools)
- [I want to **build ROCm-NPU from source**.](docs/build-from-source.md)

## Install a release and run models

You can either download and unpack binaries from a
[GitHub Release](https://github.com/ROCm/rocm-npu/releases) directly,
or install everything into a Python virtual environment. The latter path is
required if you plan to develop your own applications.

### Option A: Standalone archive

Each release includes a relocatable FastFlowLM + HRX archive:

- Windows: `rocm-npu-windows-x86_64.zip`
- Linux: `rocm-npu-linux-x86_64.tar.zst`

Extract it and run the binaries from the archive root (`bin/flm` and
`bin/rocm-npu`):

```powershell
# Windows
Expand-Archive .\rocm-npu-windows-x86_64.zip -DestinationPath .
cd .\rocm-npu-windows-x86_64
.\bin\rocm-npu.exe version
.\bin\rocm-npu.exe probe --json
.\bin\rocm-npu.exe doctor
.\bin\rocm-npu.exe self-test
.\bin\flm.exe list
.\bin\flm.exe run qwen3:0.6b
```

```bash
# Linux
tar --zstd -xf rocm-npu-linux-x86_64.tar.zst
cd rocm-npu-linux-x86_64
./bin/rocm-npu version
./bin/rocm-npu probe --json
./bin/rocm-npu doctor
./bin/rocm-npu self-test
./bin/flm list
./bin/flm run qwen3:0.6b
```

`rocm-npu self-test` also dispatches a packaged NPU2 fixture. The first
`flm run` downloads model assets and needs Hugging Face access.
Inside the interactive `flm run` prompt, use `/bye` to exit. `flm list` shows
the models included in that build.

The archive cannot compile NPU programs; use the wheels in Option B for that.

### Option B: Python wheels

Wheels ship on the same GitHub release, not on PyPI. Installing
`rocm-npu-fastflowlm` pulls the matching `rocm-npu-runtime` wheel.

Install Python and create a virtual environment. Activate the environment
before every command below:

```powershell
# Windows
winget install --exact --id Python.Python.3.12
py -3.12 -m venv $HOME\rocm-npu
& $HOME\rocm-npu\Scripts\Activate.ps1
```

```bash
# Linux
sudo apt-get install -y python3.12 python3.12-venv
python3.12 -m venv ~/rocm-npu
source ~/rocm-npu/bin/activate
```

Replace `<tag>` and `<version>` with the values from the release. For example,
if the release tag is `rocm-npu-v0.1.0.dev202609050033`, then use `<tag> = rocm-npu-v0.1.0.dev202609050033` and `<version> = 0.1.0.dev202609050033`

```bash
python -m pip install --only-binary=:all: --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/<tag>" "rocm-npu-fastflowlm==<version>"
```

To validate the runtime, use the following commands

```bash
rocm-npu version
rocm-npu probe --json
rocm-npu doctor
rocm-npu self-test
flm list
flm run qwen3:0.6b
```

To also compile your own NPU programs, see
[Develop your own models and applications](#develop-your-own-models-and-applications-using-iron-tools).

See [native runtime CLI](docs/runtime-cli.md) for `rocm-npu` commands.

## Develop your own models and applications using *iron-tools*

*iron-tools* provides programming abstractions and a compiler toolchain that
targets the AI Engine (AIE) array in the AMD NPU. *FastFlowLM* was built using
*iron-tools*.

> [!NOTE]
> *iron-tools* is currently undergoing renaming from its old name, MLIR-AIE,
> so you might see references to both of these terms. They mean the same
> thing.

`iron-tools` ships as a vendored part of ROCm-NPU in the `rocm-npu-compiler`
package. It needs a Python 3.12, 3.13, or 3.14 virtual environment. Set up
and activate the environment before running the commands below; instructions
are the same as in the previous section.

Install `rocm-npu-compiler` and `rocm-npu-runtime` from a
[GitHub Release](https://github.com/ROCm/rocm-npu/releases): the compiler builds
the design, the runtime dispatches it on the NPU. Replace `<tag>` and
`<version>` with values from that release (tags look like
`rocm-npu-v<version>`):

```bash
python -m pip install --only-binary=:all: --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/<tag>" "rocm-npu-compiler==<version>" "rocm-npu-runtime==<version>"
```

For example:
```bash
python -m pip install --only-binary=:all: --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/rocm-npu-v0.1.0.dev202608250029" "rocm-npu-compiler==0.1.0.dev202608250029" "rocm-npu-runtime==0.1.0.dev202608250029"
```

### Install the IRON operator library

`rocm-npu-ops` is the IRON operator library
(`from iron.operators import ElementwiseAdd`).
It depends on `rocm-npu-compiler` at the same version, so a compiler install
above is enough, or this command will pull the compiler if it is missing:

```bash
python -m pip install --only-binary=:all: --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/<tag>" "rocm-npu-ops==<version>"
```

If you need bleeding-edge `iron-tools` features, you may also choose to install
directly from the [upstream repository](https://github.com/Xilinx/mlir-aie)
instead of the vendored `rocm-npu-compiler` package; see instructions there.

### Hello world example

[tests/npu/jit_smoke.py](tests/npu/jit_smoke.py) contains a small IRON JIT example
program that we can compile for the NPU using `iron-tools`. It copies a buffer
DDR -> MemTile -> DDR: pure DMA, no compute kernel.

After installing `rocm-npu-compiler` and `rocm-npu-runtime`, clone or download
this repository and run:

```bash
python tests/npu/jit_smoke.py
# PASS: 4096 int32 elements copied through the NPU
```

### Learn the programming model

- **[MLIR-AIE programming guide](https://xilinx.github.io/mlir-aie/dev/)** --
  the iron-tools tutorial series.
- **[MLIR-AIE programming examples](https://github.com/Xilinx/mlir-aie/tree/main/programming_examples)** --
  designs from `passthrough_kernel` up to matrix multiplication.

Both assume an MLIR-AIE source checkout with Makefiles and XRT. The
`rocm-npu-compiler` wheel already gives you the toolchain, and the runtime here
is HRX, so read the IRON design code and skip their setup steps.

To build ROCm-NPU from this checkout instead of installing a release, follow
the [build from source guide](docs/build-from-source.md). It covers the runtime,
FastFlowLM, compiler stack, platform prerequisites, packaging, and tests.

## Additional documentation

- [Native runtime CLI](docs/runtime-cli.md): `rocm-npu version`, `probe`,
  `doctor`, `self-test`, and `powermode`
- [tests/npu/jit_smoke.py](tests/npu/jit_smoke.py): compiler/JIT smoke test
  and the smallest complete iron-tools design
- iron-tools programming:
  [guide](https://xilinx.github.io/mlir-aie/dev/) and
  [examples](https://github.com/Xilinx/mlir-aie/tree/main/programming_examples)
- [Model qualification](docs/qualification.md): optional `rocm-npu qualify`
  profiles and evidence bundles for validating releases
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
