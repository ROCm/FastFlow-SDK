# Temporary submodule patches

Transitional patches `dev.py` applies to submodules at `fetch`/`setup` time.
They live **only here** in rocm-npu-staging and are kept as plain `git apply`-able
diffs so each is **directly upstreamable** (`git apply` → commit → PR) once the
underlying issue is fixed upstream. Layout: `patches/<submodule>/NNNN-*.patch`.

Application is idempotent (already-applied patches are skipped). The patched
submodules carry `ignore = dirty` in `.gitmodules`, so applying a patch does not
show up as submodule drift. Patch files are normalized to LF. `dev.py` first
checks exact context, then tolerates whitespace-only context differences so a
plain LF patch can also target an upstream file tracked as CRLF.

Because they are upstreamable, a patch carries only changes that belong upstream.
CI-only scaffolding belongs in `tests/` here, not inside a patch.

## Editing a patch

Never hand-edit hunk bodies or `@@` headers: the line counts must agree with the
body, and a malformed hunk only fails later, at `git apply` time. Regenerate the
patch instead, from the submodule's pinned commit. To edit `0003`, commit the
patches that precede it so `git diff --cached` yields `0003` alone:

```bash
cd projects/iron-tools
git checkout -f <pin> && git clean -fdx     # -x: patch-created files are gitignored
git apply ../../patches/iron-tools/0001-*.patch
git add -A -f && git commit -qm base        # everything 0003 applies on top of
git apply ../../patches/iron-tools/000[12]-*.patch
# ...edit files...
git add -A -f
git diff --cached > ../../patches/iron-tools/0003-add-hrx-iron-path.patch
```

`git add -f` is required: several files these patches create are gitignored
upstream. Then verify against the pin with `git apply --check` and
`git apply --reverse --check` — `dev.py` uses the reverse check to decide a patch
is already applied, so a patch that does not reverse cleanly breaks `setup`.

## patches/hrx/

### 0001-surface-semaphore-failure-status.patch
A failed device dispatch reached callers as a bare `INTERNAL` with no detail.
`iree_async_semaphore_multi_wait` reports failures as `iree_status_from_code(code)`,
dropping the annotation, and leaves the original status owned by the semaphore.
`hrx_semaphore_wait` now queries it back, so the amdxdna HAL's `did not complete: ert
state N (error_index M, submit_index K)` reaches `dev.py test-npu` and the CTS.

**Remove when:** fixed upstream in hrx (`libhrx/src/libhrx/semaphore.c`).

### 0002-add-buffer-map-persistent-compat.patch
Restores the removed `hrx_buffer_map_persistent` export as a compatibility wrapper
around `hrx_buffer_map_with_mode(..., HRX_MAPPING_MODE_PERSISTENT, ...)`.

**Why:** the pinned FastFlowLM PR's bundled engine libraries were built against the
older libhrx ABI and still import `hrx_buffer_map_persistent`. The current pinned
HRX has the generalized mapping API, so this keeps the staging stack link/runtime
compatible without rebuilding all engine libraries.

**Remove when:** FastFlowLM's bundled engine libraries are rebuilt against the
current HRX mapping API, or HRX carries this compatibility wrapper upstream.

### 0003-disable-command-chain-on-old-amdxdna-stacks.patch
Extends the amdxdna native DDI caps with driver-stack identity and command-chain
policy status, then disables parent `ERT_CMD_CHAIN` on known-problem Linux KMQ
stacks. The exact stock Ubuntu 6.17 amdxdna `srcversion`
`2DBDA75956CAFA9D029EA89` is denied regardless of firmware, and otherwise XDNA2
stacks keep command chains disabled until the reported firmware reaches `1.1.0.0`.
The HAL still records the host-patched runlist path and falls back to direct
child submissions, so FastFlowLM can run on those stacks while newer validated
firmware keeps command chains enabled by default.

**Remove when:** Linux KMQ command chains are validated on stock Ubuntu firmware,
or upstream hrx carries equivalent driver/firmware capability gating.

## patches/FastFlowLM/

### 0001-use-hrx-map-with-mode.patch
Updates the FastFlowLM C++ HRX wrapper to call `hrx_buffer_map_with_mode` with
`HRX_MAPPING_MODE_PERSISTENT` instead of the removed `hrx_buffer_map_persistent`
helper.

**Why:** the pinned HRX public header no longer declares `hrx_buffer_map_persistent`
for new code.

**Remove when:** FastFlowLM PR #602 is updated for the current HRX API.

### 0002-support-redirected-windows-cli-input.patch
Keeps FastFlowLM's Unicode console editor for an interactive Windows console,
but reads from `std::cin` when stdin is redirected. It also changes console modes
only when `GetConsoleMode` confirms that the corresponding standard handle is a
console.

**Why:** the native NPU smoke supplies a prompt through stdin. The Windows CLI
previously called `ReadConsoleInput` on that pipe, which fails immediately and
causes a busy loop at the prompt. The POSIX CLI already treats EOF as `/bye`.

**Remove when:** equivalent redirected-input handling lands in FastFlowLM.

## patches/iron-ops/

### 0001-xrt-free-host-stack.patch
Keeps the operator test path importable without XRT by making `pyxrt` optional,
using the active `IRON_RUNTIME` tensor class in `run_test`, and failing only the
XRT-specific sequence dispatch policies when they are constructed.

**Remove when:** iron-ops supports non-XRT host runtimes upstream.

### 0002-lazy-operator-imports.patch
Makes `iron.operators` re-export operator classes lazily. This prevents importing
one operator, such as `ElementwiseAdd`, from importing sequence-only operators
that require XRT.

**Remove when:** iron-ops avoids eager cross-operator imports upstream.

## patches/iron-tools/

### 0003-add-hrx-iron-path.patch
Adds an HRX host-runtime to mlir-aie's IRON path so a design can run on the NPU
through the hrx amdxdna HAL (`libhrx.so`) instead of XRT — `cmake/modules/FindHRX.cmake`,
`python/utils/hostruntime/hrxruntime/` (the `IRON_RUNTIME=hrx` runtime + xadx builder),
`runtime_lib/test_lib/hrx_test_wrapper.h`, and a `vector_scalar_add` HRX testbench
(`test_chain_hrx.py` / `test_runlist_hrx.cpp`). Drives `dev.py test-npu`.

Ported from Abhishek Varma's `avarma_add_hrx` branch of mlir-aie (the upstream-able
reference integration), scoped to our pin (`git diff <our-pin>..avarma_add_hrx` over the
16 touched files). **Remove when** this lands upstream in mlir-aie.

The hrxruntime binds libhrx by symbol, so it tracks the hrx pin: `hrx_buffer_map_persistent`
became `hrx_buffer_map_with_mode(buffer, mode, flags, offset, size, ptr)`. A removed symbol
surfaces as `undefined symbol` from ctypes at import, which `aie/utils/__init__.py` swallows
into "libhrx.so could not be located" -- `dev.py verify-npu-prereqs` reports the real one.

### 0001-eudsl-for-amd-llvm-23.patch
Edits two files in the submodule (upstream `Xilinx/mlir-aie`):
1. `python/requirements.txt`: bumps the pinned `eudsl-python-extras` to the
   `manifests/versions.toml [eudsl]` version, and repoints its `-f` find-links from
   `https://llvm.github.io/eudsl` to the local prefetch cache (`../../../.cache/find-links`)
   so `vendor_eudsl.py` never lists the remote.
2. `utils/mlir_aie_wheels/vendor_eudsl.py`: adds `--no-build-isolation` to its
   `pip download`, so the sdist's metadata build uses the venv's build deps instead
   of fetching them.

Combined with `dev.py` setting `PIP_FIND_LINKS` (absolute cache) + `PIP_NO_INDEX=1`
for the iron-tools build, eudsl vendoring is **network-free** (resolved from the
sha256-verified prefetched sdist).

**Why:** mlir-aie pins eudsl `0.1.0.20260418` — correct for mlir-aie's *intended*
LLVM. We build mlir-aie's Python bindings against **ROCm amd-llvm 23**, whose
nanobind rejects eudsl 04-18's `ScalarValue` re-init path (`Value(OpResult)`),
breaking IRON design generation. A newer eudsl (`0.1.0.20260629`) fixes that and
— verified — emits **byte-size-identical** lowered IR/xclbin, so it changes nothing
else for our designs.

**Remove when:** mlir-aie's Python/IRON path is built against an aligned LLVM, at
which point mlir-aie's own eudsl pin works and this override is unnecessary. Until
then, also worth upstreaming to mlir-aie as a build-against-newer-LLVM compatibility note.

### 0002-windows-clang-cl.patch
Lets the submodule build under **clang-cl** (the Windows toolchain; see the
`[hrx_xclbinutil]` note in `manifests/versions.toml` for why not `cl`). Four files,
two root causes.

**A. MSVC-style drivers default C++ exceptions OFF, and clang-cl makes `throw` a hard
error** (`cl` only warns, C4530). GCC/Clang enable exceptions by default, so every
`throw` in a TU compiled without `/EHsc` breaks on Windows and only under clang-cl.
Three targets need `/EHsc`:

1. `tools/bootgen/CMakeLists.txt`: its two bootgen blocks tested
   `if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")`. CMake sets `MSVC` true for *"another
   compiler simulating the Visual C++ `cl` command-line syntax"*, but reports
   `CMAKE_CXX_COMPILER_ID` as `Clang` for clang-cl, so the block was skipped and
   bootgen never got `/EHsc` → `stringutils.h:55: cannot use 'throw' with exceptions
   disabled`. Key them on `MSVC`.
2. `test/CppTests/CMakeLists.txt`: *all four* tests here use
   `throw std::runtime_error(...)` as their assertion mechanism, and
   `bfloat16_fallback_test` additionally pulls in `cxxopts.hpp` via `test_utils.h`.
   `add_compile_options(/EHsc)` at **directory** scope, so tests added later are
   covered too rather than failing one CI round at a time.
3. `runtime_lib/test_lib/CMakeLists.txt`: `test_utils` compiles `test_utils.cpp`,
   which throws directly and via `cxxopts.hpp`.

   Note we do **not** consume `test_utils`: it is the host-side helper library for the
   `programming_examples/*/test.cpp` binaries (cxxopts option parsing, instruction-
   sequence loading, `init_xrt_load_kernel(xrt::device&, ...)`), and this stack neither
   builds those hosts nor uses XRT. It is built anyway because upstream's
   `enableTestLibProject` (in `runtime_lib/CMakeLists.txt`) carves out
   *"on Windows, only stage the test_utils interface payload"* — `LibXAIE_FOUND` is true
   on both platforms and neither `VITIS_ROOT` nor `LibXAIE_x86_64_DIR` is defined, so the
   literal `WIN32` in that `elseif` is the sole reason Windows builds it and Linux does
   not. We fix it rather than disable it: `/EHsc` is a real clang-cl bug fix that is
   upstreamable as-is, whereas suppressing upstream's carve-out would be a permanent
   local divergence for the sake of a few seconds of build time.

   It also must *stay*: running the programming_examples through HRX is built on top of
   test_utils (`hrx_test_wrapper.h` is a drop-in for `xrt_test_wrapper.h`, in that same
   directory), and `common.cmake`'s `target_link_test_utils()` prefers the prebuilt
   `runtime_lib/x86_64/test_lib/{lib,include}` this ExternalProject stages.

**B. MS-ABI enums are signed.** `CMakeLists.txt`: skip `-Werror=sign-compare` under
`MSVC`. Enums whose enumerators are all non-negative get an *unsigned* underlying type
under the Itanium ABI but a *signed* one under the MS ABI, so aie-rt's
`XAie_Events`-vs-`u32` comparisons (`xaie_rsc.c`, `xaie_timer.c`, `xaie_rsc_events.c`,
`xaie_perfcnt.c`, `xaie_trace.c`) warn on Windows only. The comparisons are correct;
only the enum's signedness differs.

Deliberately **not** touched: the `cdo_driver_mlir_aie` blocks lower in
`tools/bootgen/CMakeLists.txt` still test `CMAKE_CXX_COMPILER_ID`. There the
`NOT ... MATCHES "MSVC"` arm is *correct* for clang-cl — it supplies
`-Wno-cast-qual -Wno-sign-compare`, which clang-cl accepts.

**Linux is unaffected**, and this is checked rather than assumed: regenerating
`build.ninja` with and without the patch (same cmake) produces a **byte-identical**
file. `MSVC` is empty under gcc and clang, so `-Werror=sign-compare` is still added
and no `/EHsc` appears anywhere. Under real `cl`, hunk A only *adds* correct unwind
semantics where C4530 previously warned. clang-cl is the only configuration whose
build actually changes.

**Remove when:** upstreamed to mlir-aie. All hunks are upstreamable as-is; none is
specific to this stack.
