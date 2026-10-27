# Compile and serve the IRON Gemma 4 example

This guide compiles the IRON operator overrides for Gemma 4 E2B, rebuilds the
FastFlowLM Gemma engine with those overrides, and serves the result through the
FastFlowLM API. The compiler, operator sources, runtime, and FastFlowLM sources
all come from one FastFlow SDK wheel release.

The example targets Linux or Windows, Python 3.12, and an XDNA2 (NPU2) device.
Compilation does not use the NPU, but serving and inference do. Install CMake,
Ninja, and a host C++ compiler before starting. On Windows, use an x64 Developer
PowerShell for Visual Studio 2022 so CMake can find MSVC.

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
creates. Pulling the model requires network access to its model host. If the
model is already cached elsewhere, point `FLM_MODEL_PATH` at that cache instead.

Linux:

```bash
export FLM_MODEL_PATH=$HOME/.cache/fastflowlm
mkdir -p "$FLM_MODEL_PATH"
flm pull gemma4-it:e2b
```

Windows:

```powershell
$env:FLM_MODEL_PATH = Join-Path $HOME ".cache\fastflowlm"
New-Item -ItemType Directory -Force $env:FLM_MODEL_PATH | Out-Null
flm pull gemma4-it:e2b
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

## Build the override engine

Configure the Gemma engine source shipped by the FastFlowLM wheel against the
installed runtime, the application override header, and the generated sequence
headers.

Linux:

```bash
cmake -S "$FLM_GEMMA4_SOURCE" -B "$GEMMA4_BUILD/engine" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  "-DFASTFLOW_SDK_ROOT=$FASTFLOW_SDK_ROOT" \
  "-DFLM_ENGINE_LIB_DIR=$FLM_ENGINE_LIB_DIR" \
  "-DFLM_IRON_INCLUDE_DIRS=$IRON_GEMMA4_APP;$GEMMA4_BUILD/gen;$IRON_INCLUDE_DIR" \
  -DFLM_OVERRIDES_HEADER=overrides.hpp

cmake --build "$GEMMA4_BUILD/engine" \
  --target gemma4e_npu --parallel 2
```

Windows:

```powershell
cmake -S $FlmGemma4Source -B "$Gemma4Build\engine" -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  "-DFASTFLOW_SDK_ROOT=$FastFlowSdkRoot" `
  "-DFLM_ENGINE_LIB_DIR=$FlmEngineLibDir" `
  "-DFLM_IRON_INCLUDE_DIRS=$IronGemma4App;$Gemma4Build\gen;$IronIncludeDir" `
  -DFLM_OVERRIDES_HEADER=overrides.hpp

cmake --build "$Gemma4Build\engine" `
  --target gemma4e_npu --parallel 2
```

The replacement engine is:

- Linux: `$GEMMA4_BUILD/engine/engines/libgemma4e_npu.so`
- Windows: `$Gemma4Build\engine\engines\gemma4e_npu.dll`

## Serve and verify the result

Put the replacement engine first in the dynamic loader search path and point
FastFlowLM at the staged xclbins.

Linux:

```bash
(
  export LD_LIBRARY_PATH=$GEMMA4_BUILD/engine/engines${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export FLM_XCLBIN_PATH=$GEMMA4_BUILD

  flm serve gemma4-it:e2b --prefill-chunk-len 512 --port 11434
)
```

Windows:

```powershell
$OldPath = $env:PATH
$OldXclbinPath = $env:FLM_XCLBIN_PATH
try {
  $env:PATH = "$Gemma4Build\engine\engines;$env:PATH"
  $env:FLM_XCLBIN_PATH = $Gemma4Build

  flm serve gemma4-it:e2b --prefill-chunk-len 512 --port 11434
} finally {
  $env:PATH = $OldPath
  $env:FLM_XCLBIN_PATH = $OldXclbinPath
}
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
    "model": "gemma4-it:e2b",
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
  model = "gemma4-it:e2b"
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
with `Ctrl-C`; the Linux subshell or Windows `finally` block then restores the
loader and xclbin environment used by the calling shell.
