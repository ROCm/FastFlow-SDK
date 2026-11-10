# Replace Gemma 4 GEMM and dequant with IRON

This example replaces only the projection GEMMs and their runtime
dequantization in Gemma 4 E2B. FastFlowLM continues to run every other
operator. The application source is available under
[`examples/iron/gemma4_gemm_dequant`](../../examples/iron/gemma4_gemm_dequant).

The example is adapted from IRON's
[GEMM and dequant exploration](https://github.com/amd/IRON/blob/devel/iron/applications/gemma4_flm/README_gemm_dequant.md).
That tutorial also compares offline BF16 and BFP16 weights. This guide keeps
the smaller runtime-dequantization path.

## Prerequisites

Complete [Install the SDK](iron-gemma4.md#install-the-sdk) and
[Download Gemma 4 E2B](iron-gemma4.md#download-gemma-4-e2b) from the full
Gemma 4 example. Keep the virtual environment active and run the commands
below from the FastFlow SDK repository root. If you downloaded the application
source separately, set its path below to that directory instead.

## Set the build paths

Linux:

```bash
export IRON_GEMM_DEQUANT_APP=$PWD/examples/iron/gemma4_gemm_dequant
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

export GEMM_DEQUANT_BUILD=$PWD/build/gemma4-gemm-dequant
mkdir -p "$GEMM_DEQUANT_BUILD"
```

Windows:

```powershell
$IronGemmDequantApp = Join-Path $PWD "examples\iron\gemma4_gemm_dequant"
$FastFlowSdkRoot = python -c "import fastflow_sdk_runtime as p; print(p.package_root())"
$FlmGemma4Source = python -c "import fastflow_sdk_fastflowlm as p; print(p.gemma4e_root())"
$FlmEngineLibDir = python -c "import fastflow_sdk_fastflowlm as p; print(p.engine_lib_dir())"
$FlmAssets = python -c "import fastflow_sdk_fastflowlm as p; print(p.xclbin_path())"
$IronIncludeDir = python -c @'
from pathlib import Path
import fastflow_sdk_compiler as p
print(Path(p.__file__).resolve().parent / "include")
'@

# Generated operator names require a short path on Windows.
$GemmDequantBuild = Join-Path $HOME "g4gd"
New-Item -ItemType Directory -Force $GemmDequantBuild | Out-Null
```

## Compile the operators

The build compiles 46 Gemma 4 GEMM and dequant configurations. It stages the
replacement xclbins and instruction sequences under the build directory.

Linux:

```bash
python "$IRON_GEMM_DEQUANT_APP/build.py" \
  "$FLM_ASSETS/xclbins" "$GEMM_DEQUANT_BUILD"
```

Windows:

```powershell
python "$IronGemmDequantApp\build.py" `
  (Join-Path $FlmAssets "xclbins") $GemmDequantBuild
```

## Build the model extension

Configure the Gemma extension shipped by the FastFlowLM wheel. CMake copies
the staged `Gemma4-E2B-IT-NPU2` tree, including `iron/`, into the bundle.
This build also requires Rust and Cargo, as described in
[Build the model extension](iron-gemma4.md#build-the-model-extension).

Linux:

```bash
cmake -S "$FLM_GEMMA4_SOURCE" -B "$GEMM_DEQUANT_BUILD/engine" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  "-DFASTFLOW_SDK_ROOT=$FASTFLOW_SDK_ROOT" \
  "-DFLM_ENGINE_LIB_DIR=$FLM_ENGINE_LIB_DIR" \
  "-DFLM_IRON_INCLUDE_DIRS=$IRON_GEMM_DEQUANT_APP;$IRON_INCLUDE_DIR" \
  -DFLM_EXTENSION_ENGINE_HEADER=overrides.hpp \
  "-DFLM_GEMMA4_XCLBIN_DIR=$GEMM_DEQUANT_BUILD/xclbins/Gemma4-E2B-IT-NPU2"

cmake --build "$GEMM_DEQUANT_BUILD/engine" --parallel 2
```

Windows:

```powershell
cmake -S $FlmGemma4Source -B "$GemmDequantBuild\engine" -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  "-DFASTFLOW_SDK_ROOT=$FastFlowSdkRoot" `
  "-DFLM_ENGINE_LIB_DIR=$FlmEngineLibDir" `
  "-DFLM_IRON_INCLUDE_DIRS=$IronGemmDequantApp;$IronIncludeDir" `
  -DFLM_EXTENSION_ENGINE_HEADER=overrides.hpp `
  "-DFLM_GEMMA4_XCLBIN_DIR=$GemmDequantBuild\xclbins\Gemma4-E2B-IT-NPU2"

cmake --build "$GemmDequantBuild\engine" --parallel 2
```

## Serve and verify

Linux:

```bash
flm --model-extension "$GEMM_DEQUANT_BUILD/engine/bundle/extension.json" \
  pull gemma4-open-it:e2b
flm --model-extension "$GEMM_DEQUANT_BUILD/engine/bundle/extension.json" \
  serve gemma4-open-it:e2b --prefill-chunk-len 512 --port 11434
```

Windows:

```powershell
$Extension = Join-Path $GemmDequantBuild "engine\bundle\extension.json"
flm --model-extension $Extension pull gemma4-open-it:e2b
flm --model-extension $Extension serve gemma4-open-it:e2b `
  --prefill-chunk-len 512 --port 11434
```

The server should report:

```text
[info]  IRON GEMM and dequant from .../xclbins/Gemma4-E2B-IT-NPU2/iron
```

Send the deterministic request from
[the full Gemma 4 example](iron-gemma4.md#serve-and-verify-the-result). The
assistant response should contain `hello`.
