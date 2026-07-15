# rocm-npu-staging

A staging repo for validating the ROCm NPU software stack from pinned source
revisions. The quickest path is FastFlowLM running directly on HRX's `amdxdna`
runtime; the full compiler-stack path additionally builds LLVM, IRON tools, and
xclbin packaging.

Pinned revisions live in [manifests/versions.toml](manifests/versions.toml),
Python deps in [requirements.txt](requirements.txt), and common commands are
orchestrated by [dev.py](dev.py).

`projects/` are git submodules: `hrx`, `FastFlowLM`, `llvm-project`
(jtuyls/llvm-project @ the ROCm amd-llvm pin), `iron-tools`
(Xilinx/mlir-aie), `iron-ops` (amd/IRON — the operator library, built and run
against the source toolchain), and `hrx-xclbinutil` (transitional).

## FastFlowLM HRX Quickstart

This is the entry point for NPU runtime validation. It builds only minimal HRX
with the `amdxdna` HAL and the pinned FastFlowLM app, then pulls `qwen3:0.6b`,
runs an inference smoke test, and runs a 1k benchmark through HRX.

Ubuntu prerequisites:

```bash
sudo apt-get install -y \
  bash build-essential ca-certificates curl git libdrm-dev libipc-cmd-perl \
  linux-libc-dev make nasm patchelf perl pkg-config python3 python3-venv unzip \
  zip zstd
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- -y --profile minimal --default-toolchain 1.97.0
export PATH="$HOME/.cargo/bin:$PATH"
```

Windows prerequisites (PowerShell, no WSL):

```powershell
winget install --exact --id Microsoft.VisualStudio.2022.BuildTools `
  --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install --exact --id Python.Python.3.12
winget install --exact --id Git.Git
winget install --exact --id Rustlang.Rustup
rustup toolchain install 1.97.0 --profile minimal
rustup default 1.97.0
git config --global core.longpaths true
```

On Linux and Windows, `dev.py setup --profile fastflowlm` installs the same
native dependency graph from `.github/vcpkg/fastflowlm/vcpkg.json` into the
repository's gitignored cache. Each installation is isolated by a hash of the
manifest and platform triplet, so an older local dependency tree cannot affect
a later build. The pinned vcpkg checkout and release-only, dynamic platform
triplets own Boost, CURL, FFmpeg, FFTW, and Linux readline;
the host package manager supplies only build tools and platform interfaces such
as Linux libdrm, Linux UAPI headers, NASM, and Perl modules required by native
build systems. CMake and Ninja come from `requirements-fastflowlm.lock`.

FastFlowLM's tokenizer build uses Cargo and currently needs `rustc >= 1.80`.
HRX is installed to a clean prefix and FastFlowLM consumes its exported
`hrx::hrx` CMake target. No HRX source or build-tree paths are part of the FLM
build contract.

Run the same workflow on both platforms. Use `python3` on Ubuntu and `python`
on Windows:

```bash
git submodule update --init projects/hrx
git submodule update --init --recursive projects/FastFlowLM

python3 dev.py setup --profile fastflowlm
python3 dev.py doctor fastflowlm
python3 dev.py build hrx --fresh
python3 dev.py build fastflowlm --fresh
python3 dev.py package-fastflowlm
python3 dev.py test-fastflowlm
```

`package-fastflowlm` creates a Windows `.zip` or Linux `.tar.zst`, verifies its
hash manifest, and re-extracts it under `build/FastFlowLM-extracted/`. The
manifest records source pins and actual source deltas, patch hashes, complete
build configuration, dependency/package inventories, toolchain versions,
licenses, and every payload file. Linux packaging normalizes relative ELF
RPATHs and rejects an incomplete runtime dependency closure. A relocated bundle
can be tested directly with:

```bash
python3 dev.py verify-fastflowlm
python3 dev.py npu-probe --bundle build/FastFlowLM-extracted/fastflowlm-hrx-linux-x86_64
python3 dev.py test-fastflowlm --bundle build/FastFlowLM-extracted/fastflowlm-hrx-linux-x86_64
```

Use the corresponding `fastflowlm-hrx-windows-x86_64` directory on Windows.
Bundle execution discovers `model_list.json`, xclbins, HRX, and model-engine
libraries relative to the executable; no build-tree path variables are needed.
The Linux and Windows CI workflows build this archive on CPU runners, then
download it into a separate gfx1151 job with no initialized submodules, build
trees, install prefixes, or development dependency setup.

CI persists vcpkg's verified source/tool downloads separately from its compiled
binary-package archives. Binary cache keys include the OS, architecture,
compiler ABI fingerprint, vcpkg policy, manifest, and platform triplet. The
installed prefix is always reconstructed and is never cached directly.
On managed Windows runners, vcpkg asset downloads use a SHA-512-verifying
transport adapter that tries both Windows HTTP and system curl to accommodate
runner certificate-revocation policy. It does not change package selection or
accept unverified content.

The smoke test requires the amdxdna driver and an NPU. It prints the FastFlowLM
transcript, rejects HRX runtime errors, requires the exact raw smoke phrase and
a coherent benchmark summary, and validates the benchmark CSV under
`build/FastFlowLM-smoke/`.

### Windows driver compatibility

The following Windows MCDM packages have been validated on a Strix
`PCI\VEN_1022&DEV_17F0` NPU with Windows build `26200`. A passing row means
HRX initialization, checked single dispatches, checked native command chains,
and coherent Qwen3 0.6B FastFlowLM-to-HRX execution all passed.

| AMD XDNA driver | Firmware evidence | HRX single | HRX chain | FLM Qwen3 0.6B |
| --- | --- | --- | --- | --- |
| `32.0.203.314` | loaded `1.0.21.43` (`xrt-smi`) | Pass | Pass | Pass |
| `32.0.203.329` | loaded `1.0.21.43` (`xrt-smi`) | Pass | Pass | Pass |
| `32.0.20101.3760` | loaded `1.1.2.64` (`xrt-smi`) | Pass | Pass | Pass |
| `32.0.20102.3930` | package candidates `1.1.2.64`, `1.1.2.65`, `1.1.2.73`, `1.5.5.391` (`npu-probe`) | Pass | Pass | Pass |

Reboot after changing the driver before validating the NPU. A hot package
switch can leave device execution unhealthy even when Windows reports the new
driver and a healthy PnP status. Run `python dev.py npu-probe` after reboot to
record the active driver and the packaged firmware hashes before issuing NPU
work. This probe deliberately has no XRT dependency, so on Windows it reports
the firmware candidates found beside the active KMD. When available, the
package's `xrt-smi examine` performs a live XRT device query and reports the
loaded firmware version; that value comes from the XRT/KMD device interface,
not from Windows PnP metadata.

## Full Compiler-Stack Quickstart

Use this path when you also need the source-built LLVM/IRON toolchain and the
IRON end-to-end artifact checks.

The stack, built from source against one ROCm LLVM except for the Peano wheel:

```
LLVM+MLIR (ROCm amd-llvm pin, LLVM 23) ─┐
                                         ├─ iron-tools/IRON design ─ .xclbin
Peano (llvm-aie wheel, AIE codegen) ─────┘                         │
hrx-xclbinutil (xclbin packaging) ─────────────────────────────────┘  ─ HRX amdxdna HAL extracts PDI
```

System prerequisites for the compiler-stack flow (apt; cmake/ninja come from
pip after setup): `sudo apt-get install -y gcc g++ python3-dev git make
libssl-dev`. The canonical build toolchain is **gcc** (matching TheRock's
pinned manylinux gcc-toolset). To build with clang locally instead, install
`clang lld` and set `CC=clang CXX=clang++`.

```bash
python3 dev.py doctor compiler-stack # check system prereqs, pins, patches, and prefetch
python3 dev.py setup              # .venv + Python deps; cmake/ninja/Peano via pip
python3 dev.py build hrx-xclbinutil
python3 dev.py build llvm         # LLVM+MLIR from the amd-llvm pin
python3 dev.py build iron-tools   # aie-opt/aie-translate/bootgen against build/llvm
python3 dev.py build hrx          # the amdxdna HAL (IREE runtime)
python3 dev.py build all          # all of the above, in order
python3 dev.py test [DESIGN]      # e2e build test, no NPU run
python3 dev.py test-iron-ops [OP] # build the iron-ops operators, no NPU run
python3 dev.py deps               # print requirements.txt
```

Build-only works on a host without an NPU. The **e2e test** builds an
iron-tools IRON design (`programming_examples/basic/matrix_scalar_add` by
default) end-to-end to a `.xclbin` with the source toolchain, then verifies the
amdxdna HAL extracts the PDI byte-for-byte == bootgen's. Running on hardware
needs the amdxdna driver and an NPU.

Another design can be passed as `DESIGN`, but it must be Peano-capable and use
`makefile-common`'s `jit_xclbin` rule; the `jit_xclbin_elf` designs
additionally emit `insts.elf`, which needs `aiebu-asm` (not part of this stack).

**`test-iron-ops`** builds the same chain from the other direction: the `iron-ops`
operator library (`ElementwiseAdd`, …) rather than an iron-tools example, compiling the
operator's own `aie_kernels/*.cc` through Peano. iron-ops is developed against the
`mlir_aie` *wheel*, so this is what catches its drift from our source pin — on a CPU
runner rather than on the NPU. Which operators it covers is
[tests/iron-ops/suite.py](tests/iron-ops/suite.py); adding a row extends it.

On a machine **with** an NPU, `dev.py test-npu` dispatches an iron-tools design through
the HAL, and `dev.py test-npu-iron-ops` runs the iron-ops operators' own tests on the
device (their references, their tolerances) with `IRON_RUNTIME=hrx`. The latter needs
their test deps: `dev.py setup --iron-ops-tests` (torch, pytest — see
[requirements-iron-ops.txt](requirements-iron-ops.txt); the build path does not need them).
