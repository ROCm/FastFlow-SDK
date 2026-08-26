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
patch instead, from the submodule's pinned commit. To edit a patch that applies on
top of earlier ones (e.g. `0005`), apply the preceding patches first so
`git diff` yields the one you are editing alone:

```bash
cd projects/iron-tools
git checkout -f <pin> && git clean -fdx     # -x also clears gitignored build output
git apply ../../patches/iron-tools/000[12]-*.patch
# ...edit the files 0005 touches...
git diff -- <files 0005 touches> > ../../patches/iron-tools/0005-eudsl-download-no-build-isolation.patch
```

Use `git add -f` if a patch you are (re)creating adds a file that is gitignored
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

### 0001-support-redirected-windows-cli-input.patch
Keeps FastFlowLM's Unicode console editor for an interactive Windows console,
but reads from `std::cin` when stdin is redirected. It also changes console modes
only when `GetConsoleMode` confirms that the corresponding standard handle is a
console.

**Why:** the native NPU smoke supplies a prompt through stdin. The Windows CLI
previously called `ReadConsoleInput` on that pipe, which fails immediately and
causes a busy loop at the prompt. The POSIX CLI already treats EOF as `/bye`.

**Remove when:** equivalent redirected-input handling lands in FastFlowLM.

### 0002-remove-configured-runtime-paths.patch
Removes compile-time install and xclbin prefixes from `flm.exe`, together with
their obsolete runtime fallback branches. FastFlowLM already checks explicit
environment variables and paths relative to its executable.

**Why:** configured prefixes embed the build checkout in the Windows executable,
which makes wheel provenance host-specific and defeats strict relocation checks.

**Remove when:** FastFlowLM no longer compiles configured source/install paths
into release executables upstream.

### 0003-prefer-explicit-xrt-dirs.patch
Skips `pkg_check_modules(XRT)` when the caller already set `XRT_INCLUDE_DIR`
and `XRT_LIB_DIR`. Distro `xrt.pc` is often broken (`prefix=/usr//usr`) and
would otherwise override those explicit host paths. pkg-config remains the
discovery fallback when the cache variables are unset.

**Why:** `dev.py build fastflowlm-xrt` passes the discovered host include/lib
dirs. Inventing a stub `xrt.pc` with `Version: 2.0` is not a pin, and leaving
pkg-config first would still pick the broken distro file.

**Remove when:** FastFlowLM prefers caller-provided XRT dirs over pkg-config
upstream.

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

### 0003-adapt-aiecc-native-cli.patch
Rewrites the `aiecc` command lines in `iron/common/compilation/base.py` to the
native `aiecc` "declarative driver" CLI shipped by the current iron-tools pin
(mlir-aie replaced the Python `aiecc.py` entry point with a C++ `tools/aiecc`
binary). The old flags become: `-v` → `--verbose`, `-j1` → `-j 1`, drop
`--no-compile-host` (host compilation is now opt-in via `-o`),
`--aie-generate-xclbin` → `--get-xclbin`, `--aie-generate-npu-insts` →
`--get-npu-insts`, `--generate-full-elf` → `--get-full-elf`, and dropping
`--no-compile` (an insts-only build simply requests only the `--get-npu-insts`
edge). Without this the off-device `dev.py test-iron-ops` gate fails immediately
with `aiecc: Unknown command line argument '--aie-generate-xclbin'`.

**Why here:** iron-ops develops against the mlir_aie wheel, whose `aiecc` still
takes the old CLI; this stack builds iron-tools from a newer source pin.

**Remove when:** iron-ops targets the native `aiecc` CLI upstream (or this stack
pins an iron-tools revision whose `aiecc` still accepts the legacy flags).

## patches/llvm-project/

### 0001-link-python-support-to-stable-abi.patch
Makes MLIR's private Python support DLL link `Python3::SABIModule` when
`MLIR_ENABLE_PYTHON_STABLE_ABI` is enabled. Upstream otherwise links
`Python3_LIBRARY_RELEASE` unconditionally on Windows, causing
`MLIRPythonSupport-mlir.dll` to import `python312.dll` inside an abi3 wheel.
Non-stable builds retain the existing version-specific linkage.

**Remove when:** upstream MLIR selects the stable-ABI import library for private
Python support DLLs.

## patches/iron-tools/

### 0001-eudsl-for-amd-llvm-23.patch
Edits one file in the submodule (upstream `Xilinx/mlir-aie`):
1. `python/requirements.txt`: bumps the pinned `eudsl-python-extras` to the
   `manifests/versions.toml [eudsl]` version, and repoints its `-f` find-links from
   `https://llvm.github.io/eudsl` to the local prefetch cache (`../../../.cache/find-links`)
   so `vendor_eudsl.py` never lists the remote.

(The former second hunk — `--no-build-isolation` in `vendor_eudsl.py` — is now upstream
in mlir-aie's `install_eudsl`, so this patch is `requirements.txt` only.)

Combined with `dev.py` setting `PIP_FIND_LINKS` (absolute cache) + `PIP_NO_INDEX=1`
for the iron-tools build, eudsl vendoring is **network-free** (resolved from the
sha256-verified prefetched sdist).

**Why:** mlir-aie pins eudsl `0.1.0.20260801` for its intended LLVM. This patch
still overrides to the cached `0.1.0.20260629` sdist (verified to emit
size-identical IR on the previous pin) and repoints find-links at
`.cache/find-links` so vendoring stays network-free under `PIP_NO_INDEX`. LLVM
is now aligned with mlir-aie's clone-llvm.sh pin; keep the version override
until 20260801 is confirmed on this stack.

**Remove when:** mlir-aie's Python/IRON path is built against an aligned LLVM, at
which point mlir-aie's own eudsl pin works and this override is unnecessary. Until
then, also worth upstreaming to mlir-aie as a build-against-newer-LLVM compatibility note.

### 0002-windows-clang-cl.patch
Lets the submodule build under **clang-cl** (the Windows toolchain; see the
`[iron_tools]` note in `manifests/versions.toml`, which also covers why the vendored
hrx-xclbinutil needs clang-cl and not `cl`). Four files, two root causes.

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

### 0005-eudsl-download-no-build-isolation.patch
Adds `--no-build-isolation` to the `pip download` in `vendor_eudsl.py`'s `install_eudsl`,
mirroring the flag already on its `pip install` step. eudsl is vendored from an sdist
(`--no-binary`), so pip must build its metadata; by default that happens in an isolated
env that tries to provision `setuptools>=64` from an index. The iron-tools build runs
with `PIP_NO_INDEX=1` (see `dev.py`), so offline that env cannot find setuptools and the
download fails. Reusing the active venv's build backend (which `requirements/compiler-stack.txt` already
installs `setuptools`/`wheel` into, precisely because "venvs omit setuptools") keeps the
step network-free.

**Why:** upstream added `--no-build-isolation` to the *install* step but not the *download*
step, so the download regressed to isolated builds that break under `PIP_NO_INDEX`.

**Remove when:** mlir-aie's `vendor_eudsl.py` passes `--no-build-isolation` on the download
command upstream too.

### 0006-mlir-core-type-stubs-from-build-tree.patch
Makes the "Staging MLIR core type stubs for pyright" step in `python/CMakeLists.txt` pick
the stub directory that actually exists: the install layout
(`${MLIR_INSTALL_PREFIX}/src/python/MLIRPythonExtension.Core.type_stub_gen/`) when present,
else the LLVM **build tree** (`${LLVM_BINARY_DIR}/tools/mlir/python/type_stubs/`).

**Why:** upstream (mlir-aie #3171) copies the `.pyi` stubs only from the MLIR *install*
prefix. This stack builds iron-tools against the LLVM **build tree** (no MLIR install), so
`MLIR_INSTALL_PREFIX` is empty and the copy resolved to a nonexistent `/src/python/...`,
failing the build. The stubs are generated in the build tree, just at a different path.

**Remove when:** mlir-aie sources these stubs from the build tree (or this stack builds
against an installed MLIR).

### 0007-aiecc-capture-stdio-maybe-unused.patch
Marks the `CaptureStdio` members `out`, `savedOut`, and `savedErr` in
`tools/aiecc/Utils.h` as `[[maybe_unused]]`. These are only read inside the
class's `#ifndef _WIN32` stdout/stderr redirect implementation; on Windows the
constructor and destructor bodies compile out, leaving these trivial members
unread. clang-cl then emits `-Wunused-private-field`, which the native `aiecc`
target promotes to an error via `-Werror=unused`, breaking the Windows build.
(`tmpPath` needs no annotation — its `std::string` destructor already counts as
a use.) gcc does not implement this clang-only diagnostic, so the change is a
no-op on Linux.

**Remove when:** upstream mlir-aie compiles `tools/aiecc/Utils.h` cleanly under
clang-cl (annotates the fields, or `#if`-guards them out on Windows).
