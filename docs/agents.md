# For AI agents

This page is for coding agents, and for people setting one up, that build NPU
applications with FastFlow SDK.

## Machine-readable documentation

| URL | Contents |
| --- | --- |
| [`/llms.txt`](https://rocm.github.io/FastFlow-SDK/llms.txt) | Index of this site with a one-line summary per page |
| [`/llms-full.txt`](https://rocm.github.io/FastFlow-SDK/llms-full.txt) | The whole site as one Markdown file |
| Any page URL with `index.md` appended | That page as Markdown |
| `/examples/<path>` | The scripts the pages show, for example [`/examples/hello_npu.py`](https://rocm.github.io/FastFlow-SDK/examples/hello_npu.py) and [`/examples/serve/chat.py`](https://rocm.github.io/FastFlow-SDK/examples/serve/chat.py) |

The site documents one release, normally the latest, and
[Component versions](reference/versions.md) names it: commands, pins, and
examples match what installing that release from the index gives you.

The reference pages ([CLI](reference/cli.md), [Troubleshooting](reference/troubleshooting.md),
[Models](reference/models.md), [IRON operators](reference/operators.md),
[Hardware and drivers](reference/hardware.md), [Component versions](reference/versions.md),
[Release notes](releases.md)) are generated from the SDK's sources and
releases each time the site is published.

## Rules

1. **Install from the FastFlow SDK index.** Use
   `python -m pip install --pre --only-binary=:all: --extra-index-url https://rocm.github.io/FastFlow-SDK/simple fastflow-sdk`.
   Plain `pip install fastflow-sdk` installs the 0.0.1 placeholder from PyPI
   instead of the SDK. Use Python 3.12 to 3.14 in a virtual environment.
2. **Do not install XRT or upstream `mlir-aie` wheels into the same
   environment.** The SDK ships its own compiler and the HRX runtime, and
   importing `aie` selects HRX.
3. **Check the machine with JSON, not text.** Run `fastflow doctor --json`
   (or `fastflow self-test --json` for a real dispatch). Read `status` and each
   `checks[].id`; look up every check that is not `pass` in
   [Troubleshooting](reference/troubleshooting.md). Exit status is `0` for pass,
   `1` for a failed check, `2` for invalid input. Human output is not stable.
4. **To run a supported model, serve it; do not write kernels.** Start
   `flm serve <tag>` and use the OpenAI API at `http://127.0.0.1:52625/v1`
   with any API key. Tags are on [Models](reference/models.md). See
   [Run and serve models](serve.md).
5. **For custom NPU compute, start from a working program.** Prefer an
   existing [IRON operator](reference/operators.md#run-an-operator) over
   writing a kernel. For a new design, begin with [Hello NPU](hello-npu.md)
   and target `NPU2()`.
6. **Match upstream examples to the shipped pins.** MLIR-AIE and IRON
   documentation describes their default branches. The SDK pins specific
   revisions; use the links on [Component versions](reference/versions.md)
   when an example does not match the installed API.
7. **Know the limits.** NPU2 (XDNA2) only. Linux or native Windows, not WSL.
   No `load_pdi` or `ScratchpadParameter` in `@iron.jit` sequences; an
   `iron.Graph` compiled to a full ELF supports them, on the drivers listed
   under [Hello NPU limitations](hello-npu.md#limitations). Setting the NPU
   power mode needs root on Linux or an elevated shell on Windows.
8. **After a driver change, reboot before debugging.**

## Task to command

| Task | Command |
| --- | --- |
| Is the NPU usable? | `fastflow doctor --json` |
| Does a real dispatch work? | `fastflow self-test --json` |
| Which driver and firmware are active? | `fastflow probe --json` |
| Which models are available? | `flm list` |
| Serve a model to an application | `flm serve qwen3:0.6b` |
| Check the compiler works without an NPU | `python hello_npu.py --build-only` ([Hello NPU](hello-npu.md)) |
| Run a library operator on the NPU | The AXPY example on [IRON operators](reference/operators.md#run-an-operator) |
| Set NPU power mode | `fastflow powermode performance` |
| Which versions are installed? | `fastflow version --json`, `flm version` |

## Report shape

Every built-in `fastflow --json` command writes one report. `data` is left
out when a command has none, `--help` stays plain text, and external plugins
write their own output.

```json
{
  "schema_version": 1,
  "command": "doctor",
  "status": "fail",
  "product": {"version": "<version>", "source_revision": "<commit>", "build_platform": "Linux-x86_64"},
  "checks": [
    {"id": "device.access", "status": "fail", "message": "no accelerator device can be opened read/write", "details": {"usable_count": 0}}
  ],
  "data": {"probe": {}}
}
```

The [JSON contract](contracts/runtime-cli-json.md) defines the fields and
what may change between versions.
