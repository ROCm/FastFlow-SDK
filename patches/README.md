# Temporary submodule patches

Transitional patches `dev.py` applies to submodules at `fetch`/`setup` time.
They live **only here** in rocm-npu-staging and are kept as plain `git apply`-able
diffs so each is **directly upstreamable** (`git apply` → commit → PR) once the
underlying issue is fixed upstream. Layout: `patches/<submodule>/NNNN-*.patch`.

Application is idempotent (already-applied patches are skipped). The patched
submodules carry `ignore = dirty` in `.gitmodules`, so applying a patch does not
show up as submodule drift.

## patches/iron-tools/

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
