# FastFlow SDK

FastFlow SDK is AMD's software stack for generative-AI applications on the
XDNA2 neural processing unit (NPU) in Ryzen AI processors, on Linux and
Windows. This wheel contains the whole SDK:

- **FastFlowLM** (`flm`): runs language, vision, speech, and embedding models
  on the NPU and serves them through an OpenAI-compatible API.
- **iron-tools** (MLIR-AIE): the compiler and the `aie.iron` Python API for
  writing your own NPU programs.
- **IRON**: a library of NPU operators built with iron-tools.
- **HRX**: the runtime that loads and dispatches NPU programs, with the
  `fastflow` CLI that checks a machine's driver, device, and runtime.
- **Qualification**: `fastflow qualify` runs the release test profiles on your
  machine.

```bash
python -m pip install fastflow-sdk
fastflow self-test       # dispatches a packaged program on the NPU
flm serve qwen3:0.6b     # OpenAI-compatible server on http://127.0.0.1:52625/v1
```

Requires Python 3.12 to 3.14 on x86-64 Linux (glibc 2.28 or newer) or
Windows, and an AMD XDNA2 NPU driver. See the
[documentation](https://rocm.github.io/FastFlow-SDK/) for supported drivers,
guides, and the release notes.

## Program the NPU with IRON

The wheel includes everything needed to write your own NPU programs in Python:
the `aie.iron` API, the compiler, and the runtime that dispatches the result.
The smallest complete program copies a buffer through the NPU and back:

```bash
curl -fsSLO https://rocm.github.io/FastFlow-SDK/examples/hello_npu.py
python hello_npu.py      # PASS: 4096 int32 elements copied through the NPU
```

In Windows PowerShell, type `curl.exe` instead of `curl`, which there is an
alias for `Invoke-WebRequest`.

From there:

- [Hello NPU](https://rocm.github.io/FastFlow-SDK/hello-npu/) walks through
  that program.
- [IRON operators](https://rocm.github.io/FastFlow-SDK/reference/operators/)
  lists the library operators, their parameters, and operands.
- [Replace Gemma 4 GEMM and dequant with IRON](https://rocm.github.io/FastFlow-SDK/examples/iron-gemma4-gemm-dequant/)
  swaps the projection GEMMs of Gemma 4 E2B for IRON operators and serves the
  model through FastFlowLM.
- [Compile and serve the IRON Gemma 4 example](https://rocm.github.io/FastFlow-SDK/examples/iron-gemma4/)
  replaces every operator of Gemma 4 E2B with IRON.

The Gemma 4 examples also need CMake, a C++ compiler, and Rust.

This wheel bundles the binaries of one qualified
[GitHub release](https://github.com/ROCm/FastFlow-SDK/releases) unchanged.
Development builds are published as separate component wheels on the
project's own package index; do not mix the two in one environment.
