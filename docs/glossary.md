# Glossary

The same parts have several names across AMD, upstream projects, and older
documentation. This page maps them.

| Term | Meaning | Also called |
| --- | --- | --- |
| NPU | Neural processing unit: the AI accelerator in Ryzen AI processors. | IPU (older name) |
| XDNA2 | The NPU architecture this SDK targets, in Strix Point, Strix Halo, Krackan Point, and Gorgon Point. | NPU2, AIE2P, `aie2p`, `npu2` |
| XDNA | The previous NPU architecture, in Phoenix and Hawk Point. Reported by `fastflow probe`, not qualified. | NPU1, AIE2 |
| AIE | AI Engine: the array of compute, memory, and shim (I/O) tiles inside the NPU. | AI Engine array |
| `amdxdna` | The Linux kernel driver for AMD NPUs. | XDNA driver, KMD |
| FastFlow SDK | This stack: FastFlowLM, HRX, iron-tools, and IRON in one install. | `fastflow-sdk` |
| FastFlowLM | The NPU inference engine for language, vision, speech, and embedding models. | FLM, `flm` |
| HRX | HIP Runtime Extended: the runtime that loads and dispatches NPU programs. Replaces XRT for this stack. | `libhrx`, `hrx.dll` |
| iron-tools | The AIE compiler toolchain and its Python API. | MLIR-AIE, `mlir-aie`, `aie`, `aie.iron` |
| IRON (API) | The Python API in iron-tools for writing NPU designs: `ObjectFifo`, `Worker`, `Runtime`, `Program`, `@iron.jit`. | `aie.iron` |
| IRON (library) | The operator library built with iron-tools, in the `iron` package. | iron-ops, `amd/IRON`, `fastflow-sdk-ops` |
| Peano | The LLVM-based compiler that generates AIE core code. | llvm-aie |
| `aiecc` | The iron-tools compiler driver that turns a design into an NPU binary. | |
| XRT | Xilinx Runtime, the runtime MLIR-AIE and IRON use upstream. Not needed with FastFlow SDK. | |
| xclbin | A container with the NPU configuration and kernels for a design. | |
| PDI | Programmable device image: the configuration loaded into the NPU, extracted from a compiled design. | |
| XADX | The HRX executable format that `fastflow self-test` dispatches. | |
| Full ELF | A self-contained NPU executable that can hold several designs and their PDIs. | |
| ObjectFifo | IRON's abstraction for moving data between tiles with synchronized buffers. | |
| Shim tile | An AIE tile that connects the array to host memory. | `ShimTile` |
| Mem tile | An AIE tile with shared memory between shim and compute tiles. | `MemTile`, L2 |
| Power mode | NPU performance setting: `default`, `powersaver`, `balanced`, `performance`, or `turbo`. | `pmode` |
| Qualification | Running a release's test profiles on real hardware and publishing the evidence with the release. | `fastflow qualify` |
| Model extension | An engine built outside `flm` that it loads to run or modify a model. | `--model-extension` |
