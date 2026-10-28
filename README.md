# FastFlow SDK

FastFlow SDK is a new software stack that brings together emerging capabilities,
tools, and runtimes for developing custom GenAI applications for AMD XDNA Neural
Processing Engines (NPUs).

> [!IMPORTANT]
> **This is an early-access release** that enables developers to experiment with
> the iron-tools compiler stack for custom model development and run
> out-of-the-box FastFlowLM models on AMD Ryzen AI XDNA2 NPUs using the new HRX
> runtime. Interfaces, packages, models, and supported
> hardware or driver combinations can change between snapshots. Use a
> [GitHub Release](https://github.com/ROCm/FastFlow-SDK/releases) for
> installable artifacts and for the driver versions validated with that
> build. These packages are not published to PyPI.

> [!TIP]
> For a higher-level application-serving experience, use [Lemonade Server](https://lemonade-server.ai/),
> which provides a validated FastFlowLM integration.

It consists of:

- **FastFlowLM** (Fast Flow Language Models): NPU-optimized language-model
  inference
- **HRX** (HIP Runtime Extended): a runtime that executes NPU applications
- **iron-tools**: a programming framework and compiler for the AI Engine
  (AIE) array, formerly MLIR-AIE
- **iron-ops**: example machine-learning operators built with *iron-tools*

The native `fastflow` CLI ships with the runtime and reports device, driver,
and runtime health. Building from this repository also pulls additional
subproject requirements; each subproject has more details. See
[CONTRIBUTING.md](CONTRIBUTING.md) if you want to change the stack.

### Key terms

- **XDNA2 / NPU2**: the AMD Ryzen™ AI NPU generation targeted by this release.
- **HRX**: the runtime layer that loads and dispatches NPU applications.
- **AIE**: AMD's AI Engine array, the programmable compute fabric targeted by
  *iron-tools*.
- **PDI**: the programmable device image extracted from a compiled design and
  loaded by the NPU runtime.
- **Peano**: the LLVM-based AIE compiler toolchain used to generate device code.

## Intended audience

This early-access release enables developers to use hands-on utilities for
custom application development with the `iron-tools` compiler stack, run FastFlowLM
models on AMD Ryzen AI XDNA2 NPUs using the HRX runtime, and validate the NPU
runtime with `fastflow`.

## Supported models

Each release includes a registry of FastFlowLM models validated for that build.
Run `flm list` after installation to see the models available in your release.

The examples use `qwen3:0.6b` because it is the smallest recommended model for
verifying NPU inference.

For the complete model catalog, see the
[FastFlowLM model documentation](https://fastflowlm.com/docs/models/).

## Supported platforms

This early-access release targets **Linux and Windows** on **x86_64**, with
**AMD Ryzen AI NPUs that implement XDNA2 (NPU2)**. Windows instructions are
native PowerShell; WSL is not the supported Windows path.

Validated NPU driver versions are listed in each GitHub release. After
installing or changing a driver, run `fastflow self-test`. If that fails,
reboot and retry; a hot package switch can occasionally leave the device
unhealthy even when the OS reports the new driver as installed.
`fastflow probe` and `fastflow doctor` report the active driver, firmware, NPU
power mode where supported, and runtime health. `fastflow powermode` reports the
current mode. Use `fastflow powermode <mode>` to change it. Power-mode changes
may require elevated privileges depending on the installed driver.

Runtime and FastFlowLM Python wheels require Python 3.11 or newer. Compiler
wheels require Python 3.12, 3.13, or 3.14. The standalone FastFlowLM/HRX
archives do not require Python.

## Getting started

- [I want to **install a release** and run models.](#install-a-release-and-run-models)
- [I want to **replace Gemma 4 GEMM and dequant with IRON**.](docs/examples/iron-gemma4-gemm-dequant.md)
- [I want to **compile and serve the IRON Gemma 4 example**.](docs/examples/iron-gemma4.md)
- [I want to **develop my own models** and applications.](#develop-your-own-models-and-applications-using-iron-tools)
- [I want to **build FastFlow SDK from source**.](docs/build-from-source.md)

## Install a release and run models

You can either download and unpack binaries from a
[GitHub Release](https://github.com/ROCm/FastFlow-SDK/releases) directly,
or install everything into a Python virtual environment. The latter path is
required if you plan to develop your own applications.

### Option A: Standalone archive

Each release includes a relocatable FastFlowLM + HRX archive:

- Windows: `fastflow-sdk-windows-x86_64.zip`
- Linux: `fastflow-sdk-linux-x86_64.tar.zst`

Extract it and run the binaries from the archive root (`bin/flm` and
`bin/fastflow`):

```powershell
# Windows
Expand-Archive .\fastflow-sdk-windows-x86_64.zip -DestinationPath .
cd .\fastflow-sdk-windows-x86_64
.\bin\fastflow.exe version
.\bin\fastflow.exe probe --json
.\bin\fastflow.exe doctor
.\bin\fastflow.exe self-test
.\bin\flm.exe list
.\bin\flm.exe run qwen3:0.6b
```

```bash
# Linux
tar --zstd -xf fastflow-sdk-linux-x86_64.tar.zst
cd fastflow-sdk-linux-x86_64
./bin/fastflow version
./bin/fastflow probe --json
./bin/fastflow doctor
./bin/fastflow self-test
./bin/flm list
./bin/flm run qwen3:0.6b
```

`fastflow self-test` also dispatches a packaged NPU2 fixture. The first
`flm run` downloads model assets and needs Hugging Face access.
Inside the interactive `flm run` prompt, use `/bye` to exit. `flm list` shows
the models included in that build.

The archive cannot compile NPU programs; use the wheels in Option B for that.

### Option B: Python wheels

Wheels ship on GitHub releases, not on PyPI, and install from the FastFlow SDK
package index at `https://rocm.github.io/FastFlow-SDK/simple`. The
`fastflow-sdk` package installs everything: FastFlowLM, the HRX runtime, the
*iron-tools* compiler, and the IRON operator library.

Install Python and create a virtual environment. Activate the environment
before every command below:

```powershell
# Windows
winget install --exact --id Python.Python.3.12
py -3.12 -m venv $HOME\fastflow
& $HOME\fastflow\Scripts\Activate.ps1
```

```bash
# Linux
sudo apt-get install -y python3.12 python3.12-venv
python3.12 -m venv ~/fastflow
source ~/fastflow/bin/activate
```

Install the latest release:

```bash
python -m pip install --pre --only-binary=:all: --extra-index-url https://rocm.github.io/FastFlow-SDK/simple fastflow-sdk
```

Releases are currently development builds, so always pass `--pre`. Without it,
`pip install fastflow-sdk` installs the `fastflow-sdk 0.0.1` placeholder from
PyPI instead of the SDK. To install a specific release, pin its version, for
example `"fastflow-sdk==0.1.0.dev202610230003"`; a pinned pre-release version
does not need `--pre`. The version is the release tag without the
`fastflow-sdk-v` prefix.

To validate the runtime, use the following commands:

```bash
fastflow version
fastflow probe --json
fastflow doctor
fastflow self-test
flm list
flm run qwen3:0.6b
```

To compile your own NPU programs with the installed compiler, see
[Develop your own models and applications](#develop-your-own-models-and-applications-using-iron-tools).

See [native runtime CLI](docs/runtime-cli.md) for `fastflow` commands.

## Develop your own models and applications using *iron-tools*

*iron-tools* provides programming abstractions and a compiler toolchain that
targets the AI Engine (AIE) array in the AMD NPU. *FastFlowLM* was built using
*iron-tools*.

> [!NOTE]
> *iron-tools* is currently undergoing renaming from its old name, MLIR-AIE,
> so you might see references to both of these terms. They mean the same
> thing.

`iron-tools` ships as a vendored part of FastFlow SDK in the `fastflow-sdk-compiler`
package and needs Python 3.12, 3.13, or 3.14. The `fastflow-sdk` install in
[Option B](#option-b-python-wheels) already includes it, together with the
runtime that dispatches designs on the NPU and the IRON operator library
(`from iron.operators import ElementwiseAdd`).

If you need bleeding-edge `iron-tools` features, you may also choose to install
directly from the [upstream repository](https://github.com/Xilinx/mlir-aie)
instead of the vendored `fastflow-sdk-compiler` package; see instructions there.

### Hello world example

[tests/npu/jit_smoke.py](tests/npu/jit_smoke.py) contains a small IRON JIT example
program that we can compile for the NPU using `iron-tools`. It copies a buffer
DDR -> MemTile -> DDR: pure DMA, no compute kernel.

After installing `fastflow-sdk`, clone or download this repository and run:

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
`fastflow-sdk-compiler` wheel already gives you the toolchain, and the runtime here
is HRX, so read the IRON design code and skip their setup steps.

To build FastFlow SDK from this checkout instead of installing a release, follow
the [build from source guide](docs/build-from-source.md). It covers the runtime,
FastFlowLM, compiler stack, platform prerequisites, packaging, and tests.

## Known limitations

- **Preemption:** FastFlowLM contains opt-in support for inserting NPU
  preemption points, but it is disabled by default. The native HRX/XADX flow
  does not yet support or validate the required preemption save/restore
  payloads end-to-end, so FastFlowLM preemption must remain disabled when using
  this flow.
- **`load_pdi` and control scratchpad:** FastFlow SDK packaging does not yet carry
  standalone PDIs, `load_pdi` relocations, or scratchpad parameter metadata.
  IRON sequences using `load_pdi` or `ScratchpadParameter` are therefore not
  currently supported.

## Additional documentation

- [Native runtime CLI](docs/runtime-cli.md): `fastflow version`, `probe`,
  `doctor`, `self-test`, and `powermode`
- [IRON Gemma 4 GEMM/dequant](docs/examples/iron-gemma4-gemm-dequant.md): replace
  FastFlowLM projection GEMMs and runtime dequantization with IRON
- [IRON Gemma 4 E2E](docs/examples/iron-gemma4.md): compile packaged or local IRON
  operators and serve the replacement FastFlowLM engine
- [tests/npu/jit_smoke.py](tests/npu/jit_smoke.py): compiler/JIT smoke test
  and the smallest complete iron-tools design
- iron-tools programming:
  [guide](https://xilinx.github.io/mlir-aie/dev/) and
  [examples](https://github.com/Xilinx/mlir-aie/tree/main/programming_examples)
- [Model qualification](docs/qualification.md): optional `fastflow qualify`
  profiles and evidence bundles for validating releases
- [Third-party notices](docs/third-party-notices.md)
- [Contributing](CONTRIBUTING.md)

## License

This repository is licensed under the [MIT License](LICENSE).

Please file issues in this repository. See [SECURITY.md](SECURITY.md) to
report a vulnerability. This is an early snapshot, so expect breaking
changes between releases.
