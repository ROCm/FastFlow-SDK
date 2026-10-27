# Gemma 4 GEMM and dequant overrides

These sources replace FastFlowLM's Gemma 4 projection GEMMs and runtime
dequantization with IRON operators. Follow the
[SDK example guide](../../../docs/examples/iron-gemma4-gemm-dequant.md) to
compile and serve them from an installed FastFlow SDK.

The example is adapted from IRON's
[Gemma 4 GEMM/dequant application](https://github.com/amd/IRON/tree/devel/iron/applications/gemma4_flm).
