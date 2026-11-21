# FastFlow SDK

FastFlow SDK is AMD's software stack for generative-AI applications on the
XDNA2 neural processing unit (NPU) in Ryzen AI processors, on Linux and
Windows. One `pip install` gives you:

- **FastFlowLM** (`flm`): runs language, vision, speech, and embedding models
  on the NPU and serves them through an OpenAI-compatible API.
- **iron-tools** (MLIR-AIE): the compiler and the `aie.iron` Python API for
  writing your own NPU programs.
- **IRON**: a library of NPU operators, such as GEMM, attention,
  normalization, and RoPE, built with iron-tools.
- **HRX**: the runtime that loads and dispatches NPU programs, with the
  `fastflow` CLI that checks a machine's driver, device, and runtime.

--8<-- "README.md:pip-install"

```bash
fastflow self-test       # dispatches a packaged program on the NPU
flm serve qwen3:0.6b     # OpenAI-compatible server on http://127.0.0.1:52625/v1
```

!!! warning "Early access"
    Releases are development snapshots. Interfaces, packages, models, and
    supported drivers can change between releases. Each
    [release](releases.md) lists the drivers it was qualified on.

## Choose a path

<div class="grid cards" markdown>

- **Run and serve models**

    Chat with a model in the terminal, or call it from any OpenAI client.

    [Run and serve models](serve.md)

- **Program the NPU**

    Compile and run your first IRON design, then use the operator library.

    [Hello NPU](hello-npu.md)

- **Check a machine**

    Find out why the NPU, driver, or runtime is not working.

    [Troubleshooting](reference/troubleshooting.md)

- **Work with an AI agent**

    Machine-readable docs, JSON CLI output, and rules for coding agents.

    [For AI agents](agents.md)

</div>

## How the pieces fit

| You want to | Use | Package |
| --- | --- | --- |
| Run or serve a supported model | `flm` | `fastflow-sdk-fastflowlm` |
| Write an NPU kernel or design in Python | `aie.iron` | `fastflow-sdk-compiler` |
| Reuse tuned NPU operators | `iron.operators` | `fastflow-sdk-ops` |
| Check the device, driver, and runtime | `fastflow` | `fastflow-sdk-runtime` |
| All of the above | | `fastflow-sdk` |

Programs compiled with iron-tools run on the HRX runtime, which ships in the
same install; you do not need XRT. For a validated, higher-level serving
application, see [Lemonade Server](https://lemonade-server.ai/), which
integrates FastFlowLM.

## Supported platforms

x86_64 Linux and Windows (native PowerShell, not WSL) with an AMD Ryzen AI NPU
that implements XDNA2 (NPU2): Strix Point, Strix Halo, Krackan Point, and
Gorgon Point. See [Hardware and drivers](reference/hardware.md) for the systems
and drivers the latest release was qualified on.
