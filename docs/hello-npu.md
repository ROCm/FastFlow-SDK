# Hello NPU

This is the smallest complete NPU program: a buffer copied from host memory
through a memory tile and back, with no compute kernel. Compiling and running
it exercises the whole chain, from IRON Python through the MLIR-AIE compiler
to an NPU binary dispatched by HRX. CI compiles and runs this exact file on
an NPU.

## Run it

Install the SDK ([Install](install.md)), then download and run the program:

```bash
curl -fsSLO https://rocm.github.io/FastFlow-SDK/examples/hello_npu.py
python hello_npu.py
# PASS: 4096 int32 elements copied through the NPU
```

`python hello_npu.py --build-only` compiles without an NPU and prints the
paths of the generated `xclbin` and instruction stream.

Importing `aie` with the SDK installed selects the HRX runtime, so you do not
need XRT. Values you set yourself for `NPU_RUNTIME` or `HRX_LIBHRX` take
precedence.

## The program

```python
--8<-- "examples/hello_npu.py"
```

- `@iron.jit` compiles the function the first time it is called with a new
  compile-time value (`n`) and caches the result.
- An `ObjectFifo` moves data between tiles; `.forward()` routes it through a
  memory tile.
- The `Runtime` sequence fills the input from host memory and drains the
  output back.
- `Program(NPU2(), rt)` targets the XDNA2 NPU and lets the placer choose
  shim tiles (`AnyShimTile`).
- `iron.arange(..., device="npu")` and `iron.zeros_like` allocate buffers the
  NPU can read and write.

## Next steps

- Add compute: the [MLIR-AIE programming guide](https://xilinx.github.io/mlir-aie/dev/)
  introduces workers, kernels, and tiling.
  The [programming examples](https://github.com/Xilinx/mlir-aie/tree/main/programming_examples)
  go from `passthrough_kernel` to matrix multiplication. They assume an
  MLIR-AIE checkout with Makefiles and XRT; with the SDK you already have the
  toolchain and runtime, so read the IRON design code and skip their setup.
- Use tuned operators instead of writing kernels: the
  [IRON operators](reference/operators.md#run-an-operator) cover GEMM,
  attention, normalization, activations, and more, and one runs on the NPU in
  a dozen lines.
- Put your kernels in a real model:
  [Replace Gemma 4 GEMM and dequant with IRON](examples/iron-gemma4-gemm-dequant.md).

## Limitations

- The `@iron.jit` path this example uses runs an `xclbin` and an instruction
  stream, so its sequences cannot use `load_pdi` or `ScratchpadParameter`:
  that path carries no standalone PDIs, their relocations, or scratchpad
  metadata. An `iron.Graph` compiled to a [full ELF](glossary.md) carries
  them. Running one needs NPU firmware 1.0.20.31 or later on Linux. On
  Windows it needs NPU driver 32.0.20101.3760 or later, or 32.0.203.376 or
  later on the 32.0.203 line; 32.0.203.329 and older refuse it.
  `fastflow probe` reports both versions.
- NPU preemption is not supported on the HRX path.
