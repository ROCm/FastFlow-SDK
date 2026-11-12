# Compile and serve the IRON Gemma 4 example

This guide compiles the IRON operator overrides for Gemma 4 E2B, builds them
into the `gemma4-open` model extension shipped by the SDK, and serves that
plugin through FastFlowLM. The compiler, operator sources, runtime, and
extension sources all come from one FastFlow SDK wheel release. Product `flm`
keeps its prebuilt Gemma 4 engine.

For a smaller example that replaces only projection GEMMs and runtime
dequantization, see
[Replace Gemma 4 GEMM and dequant with IRON](iron-gemma4-gemm-dequant.md).

The example targets Linux or Windows, Python 3.12, and an XDNA2 (NPU2) device.
Compilation does not use the NPU, but serving and inference do. On Ubuntu,
install the host build dependencies:

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ libdrm-dev ninja-build
```

On Windows, install CMake, Ninja, and Visual Studio 2022, then use an x64
Developer PowerShell so CMake can find MSVC.

## Install the SDK

Install the latest [FastFlow SDK release](https://github.com/ROCm/FastFlow-SDK/releases).
The `fastflow-sdk` meta-package installs matching compiler, operator, runtime,
and FastFlowLM wheels. Current releases use prerelease versions, so pass
`--pre`.

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --pre --only-binary=:all: \
  --extra-index-url https://rocm.github.io/FastFlow-SDK/simple \
  fastflow-sdk
```

Windows:

```powershell
py -3.12 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --pre --only-binary=:all: `
  --extra-index-url https://rocm.github.io/FastFlow-SDK/simple `
  fastflow-sdk
```

Keep this environment active for every command in the guide. Check the runtime
and device before starting the larger compilation:

```console
fastflow version
fastflow doctor
```

## Download Gemma 4 E2B

`FLM_MODEL_PATH` is the directory above the `models` directory that FastFlowLM
creates. The open catalog entry is pulled after the extension exists, because
`gemma4-open-it:e2b` is not in the built-in model list. Pulling it requires
network access to its model host. If the model is already cached elsewhere,
point `FLM_MODEL_PATH` at that cache instead.

Linux:

```bash
export FLM_MODEL_PATH=$HOME/.cache/fastflowlm
mkdir -p "$FLM_MODEL_PATH"
```

Windows:

```powershell
$env:FLM_MODEL_PATH = Join-Path $HOME ".cache\fastflowlm"
New-Item -ItemType Directory -Force $env:FLM_MODEL_PATH | Out-Null
```

## Select the IRON application source

The operator wheel contains the release-matched Gemma application source. Use
it by default. The
[upstream application source](https://github.com/amd/IRON/tree/devel/iron/applications/gemma4_flm)
is also available for browsing and development.

Linux:

```bash
export IRON_GEMMA4_APP=$(python -c \
  'from pathlib import Path; import iron; print(Path(iron.__file__).resolve().parent / "applications/gemma4_flm")')
```

Windows:

```powershell
$IronGemma4App = python -c @'
from pathlib import Path
import iron
print(Path(iron.__file__).resolve().parent / "applications/gemma4_flm")
'@
```

To modify or replace the example, point the application path at your own source
directory. It should contain `build.py`, `overrides.hpp`, and any local helper
files. The SDK installation supplies the IRON Python APIs and compiler, so a
separate IRON checkout is not required.

Linux:

```bash
export IRON_GEMMA4_APP=/path/to/application
```

Windows:

```powershell
$IronGemma4App = "C:\path\to\application"
```

The packaged source is tested with its SDK release. User-provided sources must
remain compatible with that release's compiler and runtime interfaces.

## Set the build paths

The wheels expose the packaged FastFlowLM source and runtime paths through
Python. The generated IRON sequence headers also need the compiler wheel's C++
headers. These paths do not change when using your own application sources;
only the application path selected above changes.

Linux:

```bash
export FASTFLOW_SDK_ROOT=$(python -c \
  'import fastflow_sdk_runtime as p; print(p.package_root())')
export FLM_GEMMA4_SOURCE=$(python -c \
  'import fastflow_sdk_fastflowlm as p; print(p.gemma4e_root())')
export FLM_ENGINE_LIB_DIR=$(python -c \
  'import fastflow_sdk_fastflowlm as p; print(p.engine_lib_dir())')
export FLM_ASSETS=$(python -c \
  'import fastflow_sdk_fastflowlm as p; print(p.xclbin_path())')
export IRON_INCLUDE_DIR=$(python -c \
  'from pathlib import Path; import fastflow_sdk_compiler as p; print(Path(p.__file__).resolve().parent / "include")')

export GEMMA4_BUILD=$PWD/build/gemma4-iron
mkdir -p "$GEMMA4_BUILD"
```

Windows:

```powershell
$FastFlowSdkRoot = python -c "import fastflow_sdk_runtime as p; print(p.package_root())"
$FlmGemma4Source = python -c "import fastflow_sdk_fastflowlm as p; print(p.gemma4e_root())"
$FlmEngineLibDir = python -c "import fastflow_sdk_fastflowlm as p; print(p.engine_lib_dir())"
$FlmAssets = python -c "import fastflow_sdk_fastflowlm as p; print(p.xclbin_path())"
$IronIncludeDir = python -c @'
from pathlib import Path
import fastflow_sdk_compiler as p
print(Path(p.__file__).resolve().parent / "include")
'@

# Keep this path short: generated operator names can exceed Windows path limits.
$Gemma4Build = Join-Path $HOME "g4sdk"
New-Item -ItemType Directory -Force $Gemma4Build | Out-Null
```

## Compile the IRON operators

`build.py` compiles 59 operator configurations for the Gemma 4 E2B text path.
It stages replacement xclbins, instruction sequences, repacked weights, and
generated C++ sequence builders under the build directory. The command can take
several minutes.

Linux:

```bash
(
  cd "$IRON_GEMMA4_APP"
  python build.py "$FLM_ASSETS/xclbins" "$GEMMA4_BUILD"
)
```

Windows:

```powershell
Push-Location $IronGemma4App
try {
  python build.py (Join-Path $FlmAssets "xclbins") $Gemma4Build
} finally {
  Pop-Location
}
```

A successful build ends with a message like:

```text
staged .../xclbins/Gemma4-E2B-IT-NPU2
```

## Build the model extension

Configure the Gemma extension shipped by the FastFlowLM wheel against the
installed runtime, the application override header, the generated sequence
headers, and the xclbin tree `build.py` just staged. CMake copies that tree,
including `iron/`, into the bundle. `flm` loads it from there.

This step also requires Rust and Cargo on both platforms. Install them with
[rustup](https://rustup.rs/), using the `x86_64-pc-windows-msvc` toolchain on
Windows. Ensure `rustc` and `cargo` are on `PATH` in the shell running CMake
(the x64 Developer PowerShell on Windows). Cargo needs network access to
download dependencies that are not already cached.

Linux:

```bash
cmake -S "$FLM_GEMMA4_SOURCE" -B "$GEMMA4_BUILD/engine" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  "-DFASTFLOW_SDK_ROOT=$FASTFLOW_SDK_ROOT" \
  "-DFLM_ENGINE_LIB_DIR=$FLM_ENGINE_LIB_DIR" \
  "-DFLM_IRON_INCLUDE_DIRS=$IRON_GEMMA4_APP;$GEMMA4_BUILD/gen;$IRON_INCLUDE_DIR" \
  -DFLM_EXTENSION_ENGINE_HEADER=overrides.hpp \
  "-DFLM_GEMMA4_XCLBIN_DIR=$GEMMA4_BUILD/xclbins/Gemma4-E2B-IT-NPU2"

cmake --build "$GEMMA4_BUILD/engine" --parallel 2
```

Windows:

```powershell
cmake -S $FlmGemma4Source -B "$Gemma4Build\engine" -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  "-DFASTFLOW_SDK_ROOT=$FastFlowSdkRoot" `
  "-DFLM_ENGINE_LIB_DIR=$FlmEngineLibDir" `
  "-DFLM_IRON_INCLUDE_DIRS=$IronGemma4App;$Gemma4Build\gen;$IronIncludeDir" `
  -DFLM_EXTENSION_ENGINE_HEADER=overrides.hpp `
  "-DFLM_GEMMA4_XCLBIN_DIR=$Gemma4Build\xclbins\Gemma4-E2B-IT-NPU2"

cmake --build "$Gemma4Build\engine" --parallel 2
```

The extension bundle is:

- Linux: `$GEMMA4_BUILD/engine/bundle/extension.json`
- Windows: `$Gemma4Build\engine\bundle\extension.json`

## Serve and verify the result

Pull and serve `gemma4-open-it:e2b` through that bundle. The plugin carries its
own engine and the staged xclbins, so the loader path and `FLM_XCLBIN_PATH`
stay unchanged.

Linux:

```bash
flm --model-extension "$GEMMA4_BUILD/engine/bundle/extension.json" \
  pull gemma4-open-it:e2b
flm --model-extension "$GEMMA4_BUILD/engine/bundle/extension.json" \
  serve gemma4-open-it:e2b --prefill-chunk-len 512 --port 11434
```

Windows:

```powershell
$Extension = Join-Path $Gemma4Build "engine\bundle\extension.json"
flm --model-extension $Extension pull gemma4-open-it:e2b
flm --model-extension $Extension serve gemma4-open-it:e2b `
  --prefill-chunk-len 512 --port 11434
```

The server log should identify the staged IRON instruction sequences:

```text
[info]  IRON operators from .../xclbins/Gemma4-E2B-IT-NPU2/iron
```

From another terminal, send a deterministic chat request.

Linux:

```bash
curl -fsS http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4-open-it:e2b",
    "stream": false,
    "temperature": 0,
    "top_k": 1,
    "messages": [
      {"role": "user", "content": "Reply with only the word hello."}
    ],
    "options": {"num_predict": 8}
  }'
```

Windows:

```powershell
$Body = @{
  model = "gemma4-open-it:e2b"
  stream = $false
  temperature = 0
  top_k = 1
  messages = @(
    @{ role = "user"; content = "Reply with only the word hello." }
  )
  options = @{ num_predict = 8 }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/chat" `
  -Method Post -ContentType "application/json" -Body $Body
```

The response should contain an assistant message with `hello`. Stop the server
with `Ctrl-C`.
