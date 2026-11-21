# Install

## Requirements

| | |
| --- | --- |
| Operating system | Linux or Windows, x86_64. On Windows use native PowerShell; WSL is not supported. |
| NPU | AMD Ryzen AI with an XDNA2 (NPU2) NPU. See [Hardware and drivers](reference/hardware.md). |
| Driver | The `amdxdna` driver on Linux, the AMD NPU driver on Windows. Each release lists the versions it was qualified on. |
| Python | 3.12, 3.13, or 3.14 for the full SDK. 3.11 or newer for the runtime and FastFlowLM alone. The standalone archive needs no Python. |

## Python wheels

The wheels are attached to [GitHub Releases](https://github.com/ROCm/FastFlow-SDK/releases)
and installed through the FastFlow SDK package index,
`https://rocm.github.io/FastFlow-SDK/simple`. Create a virtual environment and
activate it before every command below:

--8<-- "README.md:venv"

Install the latest release:

--8<-- "README.md:pip-install"

Keep `--pre` while releases are development builds. Without it, pip installs
the `fastflow-sdk 0.0.1` placeholder from PyPI instead of the SDK. To install
one release, pin its version, `"fastflow-sdk==<version>"`, with a version from
the [Release notes](releases.md); an exact pin does not need `--pre`.

`fastflow-sdk` installs these packages, which you can also install alone:

| Package | Provides |
| --- | --- |
| `fastflow-sdk-fastflowlm` | `flm`, the model registry, and its matching runtime |
| `fastflow-sdk-runtime` | `fastflow`, the HRX runtime library, and the `fastflow_sdk_runtime` Python helpers |
| `fastflow-sdk-compiler` | iron-tools: `aie.iron`, `aiecc`, `aie-opt`, Peano |
| `fastflow-sdk-ops` | the IRON operator library, `iron.operators` |

`fastflow-sdk-qualification` is optional: it adds `fastflow qualify` for
validating a release on a machine. See [Model qualification](qualification.md).

## Verify the install

--8<-- "README.md:validate"

`fastflow doctor` and `fastflow self-test` exit 0 when everything passes.
`flm run` downloads the model on first use and needs access to Hugging Face;
type `/bye` to leave the chat. If a check fails, look it up in
[Troubleshooting](reference/troubleshooting.md). After installing or changing a
driver, reboot before you debug further.

## Standalone archive

Each release also has a relocatable FastFlowLM and HRX archive,
`fastflow-sdk-windows-x86_64.zip` or `fastflow-sdk-linux-x86_64.tar.zst`. It
runs models without Python but cannot compile NPU programs.

--8<-- "README.md:archive"

## Upgrade

Add `--upgrade` to the install command. `fastflow-sdk` pins each of its
packages to its own version, so they move to the new release together:

```bash
python -m pip install --upgrade --pre --only-binary=:all: --extra-index-url https://rocm.github.io/FastFlow-SDK/simple fastflow-sdk
```

To build the SDK yourself instead, see [Build from source](build-from-source.md).
