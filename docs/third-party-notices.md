# Third-party notices

## JSON for Modern C++

The native `fastflow` executable incorporates JSON for Modern C++,
version 3.12.0, from https://github.com/nlohmann/json.

Copyright © 2013-2025 Niels Lohmann.

It is distributed under the MIT License. Products containing `fastflow`
include the complete license as `licenses/nlohmann-json-LICENSE.MIT`.
Vendored-file provenance and update instructions are recorded in
[`third_party/nlohmann_json/README.md`](https://github.com/ROCm/FastFlow-SDK/blob/main/third_party/nlohmann_json/README.md).

## LLVM OpenMP runtime

The Linux FastFlowLM wheel and the Linux standalone archive include `libomp.so`,
the LLVM OpenMP runtime, built from the `llvm-project` commit pinned in
`manifests/versions.toml` (https://github.com/ROCm/llvm-project). FastFlowLM's
engine libraries use it in place of the GCC OpenMP runtime.

It is distributed under the Apache License v2.0 with LLVM Exceptions. Those
products include the complete license as `licenses/llvm-openmp-LICENSE.TXT`.
