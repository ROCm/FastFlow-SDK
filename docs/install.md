# Install

## Requirements

| | |
| --- | --- |
| Operating system | Linux or Windows, x86_64. On Windows use native PowerShell; WSL is not supported. |
| NPU | AMD Ryzen AI with an XDNA2 (NPU2) NPU. See [Hardware and drivers](reference/hardware.md). |
| Driver | The `amdxdna` driver on Linux, the AMD NPU driver on Windows. Each release lists the versions it was qualified on. |
| Python | 3.12, 3.13, or 3.14. The standalone archive needs no Python. |

## Python wheels

Releases are on [PyPI](https://pypi.org/project/fastflow-sdk/) as one
`fastflow-sdk` wheel per platform. Create a virtual environment and activate it
before every command below:

--8<-- "README.md:venv"

Install the latest release:

--8<-- "README.md:pip-install"

To install one release, pin its version, `"fastflow-sdk==<version>"`, with a
version from the [Release notes](releases.md).

The wheel contains these parts of the SDK:

| Part | Provides |
| --- | --- |
| FastFlowLM | `flm` and the model registry |
| Runtime | `fastflow`, the HRX runtime library, and the `fastflow_sdk_runtime` Python helpers |
| Compiler | iron-tools: `aie.iron`, `aiecc`, `aie-opt`, Peano |
| Operators | the IRON operator library, `iron.operators` |
| Qualification | `fastflow qualify`, for validating a release on a machine. See [Model qualification](qualification.md). |

### Development builds

--8<-- "README.md:dev-builds"

A development build's packages are `fastflow-sdk-fastflowlm`,
`fastflow-sdk-runtime`, `fastflow-sdk-compiler`, `fastflow-sdk-ops`, and the
optional `fastflow-sdk-qualification`. You can also install one alone; the
runtime and FastFlowLM alone need Python 3.11 or newer.

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

Add `--upgrade` to the install command:

```bash
python -m pip install --upgrade fastflow-sdk
```

For a development build, add `--upgrade` to its install command instead.
`fastflow-sdk` pins each of its packages to its own version, so they move to
the new build together.

To build the SDK yourself instead, see [Build from source](build-from-source.md).
