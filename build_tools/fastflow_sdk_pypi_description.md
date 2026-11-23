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

This wheel bundles the binaries of one qualified
[GitHub release](https://github.com/ROCm/FastFlow-SDK/releases) unchanged.
Development builds are published as separate component wheels on the
project's own package index; do not mix the two in one environment.
