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

### 0004-sync-submits-for-multi-submission-command-buffers.patch
Stops the amdxdna direct command buffer from deferring completion when a single
command buffer issues more than one native submission, which deadlocked
`queue_execute` on Windows.

**Why:** deferred completion hands every issued submission to the completion
batch, and `iree_hal_amdxdna_queue_execute_apply` submits that batch only after
the *whole* command buffer has been applied. Windows MCDM keeps one native
submission in flight (`begin_pathb_submission` waits for
`pathb_active_submission_count == 0`, because command objects and the shared
command aperture are mutable staging resources). A command buffer that flushed
two submissions therefore blocked forever on the second: the only thing that
could retire the first was the batch this thread had not submitted yet. The
patch counts the submissions a flush will issue and falls back to submit-and-wait
when there is more than one, so each retires before the next begins. Single
submission command buffers — every one on the Linux path, and the overwhelming
majority on Windows — keep deferred completion and are byte-for-byte unaffected.

Symptom this fixes: FastFlowLM's hybrid-attention (GDN) models — qwen3.5
(0.8b/2b/4b/9b) and qwen3.6-moe — hung on Windows immediately after prefill.
Their decode step records an RTP-only run plus the per-layer dispatches into one
stream flush, and that command buffer is the one that flushes two groups. Prefill
submits one dispatch per flush, so it never hit this.

**Remove when:** upstream hrx either lets a command buffer's deferred batch drain
mid-apply, or allows more than one native submission in flight on Windows MCDM.

### 0005-invalidate-stale-chain-cache-descriptors.patch
Stops the amdxdna device chain-command cache from resubmitting a cached chain
through its descriptor once an in-place rewrite has made that descriptor stale.

**Why:** a cached `iree_hal_amdxdna_chain_cmd_t` holds both the control code it has
realized (`ctrl_words` plus the control-code BO behind them) and a cloned descriptor
of the dispatch that produced it (`src_executable_identity`, `src_entry_point`,
`src_constants`, `src_asm_inst`, ...). The cache reuses an entry through four
matchers, two keyed on the realized contents (device signature, shape) and two on
that descriptor. The shape matcher only requires the control word count and binding
shape to agree, so `iree_hal_amdxdna_update_cached_chain_cmd` legitimately rewrites
an entry's realized words from a *different* control program — but it left the
descriptor describing the previous one. A later flush whose own descriptor equalled
that stale snapshot then took the exact-descriptor fast path, which resubmits the
cached chain **unchanged**: the NPU ran the other program's control code and the
command never retired (`ert state 8`, `ERT_CMD_STATE_TIMEOUT`).

The rewrite now refreshes the constants block in place when it fits, and clears a new
`descriptor_describes_contents` flag whenever the words it realized came from another
control program; both descriptor matchers refuse a retired descriptor. The
content-keyed matchers are unaffected, so the hot rewrite path (209 of 210 flushes on
the repro) still hits.

Symptom this fixes: FastFlowLM's `qwen3.5:{0.8b,2b,4b,9b}` and `qwen3.6-moe:35b-a3b`
decode timed out the first time a bench grew the context back to a size it had already
run (2k → 1k → **2k**), because that regenerates a byte-identical control program and
so a matching descriptor. FLM does not treat a failed wait as fatal, so it then decoded
garbage against a poisoned KV cache until an allocation failed. The client-side
workaround was to defeat the executable cache entirely (`HRX_NOCACHE=1`, ~3% decode).

**Remove when:** upstream hrx keeps chain-cache descriptors in sync with the control
code their commands have realized, or drops the descriptor-keyed fast paths.

### 0006-drop-redundant-aperture-wide-invalidate-on-remap.patch
Stops every Windows path-B dispatch from re-invalidating the whole 64 MiB command
aperture.

**Why:** on the legacy MCDM contract each single dispatch stages its control code
into the command aperture and then crosses the "remap boundary"
(`RefreshCommandApertureGpuMapping`): an `Unlock2`/`Lock2` lock transition so the
miniport observes the freshly staged bytes. It re-acquired the mapping through
`LockCommandApertureGpuAfterBootstrap`, which *also* issues a
`D3DKMTInvalidateCache` over the entire aperture allocation. So every dispatch paid
an allocation-wide cache invalidate on top of the range-scoped one it had already
done — and the CPU-cache eviction that caused then slowed each following opcode-9
publication and hardware-queue submit.

The allocation-wide invalidate is bootstrap-only by construction, as the comment at
the `SubmitAndWaitCommandAperture` call site already says: XRT locks and invalidates
the 64 MiB aperture once, after the opcode-2 bootstrap submit. Afterwards every CPU
write into the aperture is published over exactly the range it touched
(`CopyAndCommitPathBCodeWrites` / `CommitPathBCodeWrites`, which is a KMT invalidate
on this contract), so repeating it allocation-wide adds nothing. The patch splits the
plain mapping lock out of the bootstrap helper: bootstrap keeps lock + allocation-wide
invalidate, the remap boundary keeps the lock transition and drops the invalidate.
Instrumentation showed the re-lock returns the same CPU pointer on all 13626 remaps of
a MoE prefill, so the transition is mapping-neutral and stays.

Symptom this fixes: HRX prefill trailed XRT on every model in the comparison sweep
(geo 1.12–1.17×), and `qwen3.6-moe:35b-a3b` trailed by ~2.3× because it issues ~13.7k
single dispatches per 1k-token prefill. Measured 1k prefill, interleaved A/B in one
binary on driver 32.0.203.314: `qwen3.6-moe:35b-a3b` 24.9 → 34.6 tok/s (TTFT 39.2 →
28.2 s), `qwen3:0.6b` 897 → 1014 tok/s, `gemma4-it:e2b` 460 → 515 tok/s.

**Remove when:** upstream hrx scopes the command-aperture invalidate to the range
being published, or drops the legacy remap boundary altogether.

### 0007-reuse-shape-compatible-partial-elf-single-commands.patch
Lets the single-dispatch command cache reuse an idle PARTIAL_ELF entry of the same
shape when run-template identity misses, instead of allocating a fresh control-code
BO per dispatch.

**Why:** the cache keys on the identity of the executable-owned run template
(`src_asm_inst`/`src_patches` pointers). A model that fans one HAL dispatch out over
many distinct templates never hits — `qwen3.6-moe:35b-a3b` issues ~13.7k single
dispatches per 1k-token prefill and missed 13591 of them. The allocation is not the
real cost: every `MakeResident` records a paging fence, and
`SubmitCommandToHwQueueAfterPaging` drains all pending fences before *each*
hardware-queue submit, of which this path issues ~3.3 per dispatch. 13.6k fresh
allocations therefore taxed 44.8k submits, and that wait dominated the prefill.

The rewrite never reads the cached descriptor as a source of truth —
`iree_hal_amdxdna_rewrite_cached_single_partial_elf_cmd` re-patches every control word
from the *fresh* descriptor and rebinds every buffer — so it only needs a compatible
shape: same queue, same CU index, equal control-word count (the BO was sized for it)
and equal binding count. The patch adds that lookup and then updates the entry's
recorded identity, since the rewrite realizes a different descriptor into it; leaving
the old identity would reintroduce exactly the staleness 0005 fixed for the chain
cache. 56b572e8c already did this for the chain cache and left the single-dispatch
path out.

Measured 1k prefill on driver 32.0.203.314, on top of 0006: `qwen3.6-moe:35b-a3b`
38.65 → 44.84 tok/s (TTFT 25.27 → 21.78 s), pending-paging wait 5.1 s → under 0.2 s,
control-code BO allocations 13591 → ~8. Submit count is unchanged.

**Remove when:** upstream hrx reuses shape-compatible PARTIAL_ELF entries in the
single-dispatch cache as it already does in the chain cache.

### 0008-sync-command-aperture-code-with-cache-lines.patch
Publishes staged command-aperture code with CPU cache-line invalidation for sub-LLC
ranges, keeping the allocation-level KMT transition only for larger ranges.

**Why:** `SyncCommandApertureCode` always used `D3DKMTInvalidateCache`, which costs
~303 µs for the ~7.7 KiB a dispatch stages — over 25× a hardware-queue submit (~11 µs)
— because the miniport services it over the whole 64 MiB allocation rather than the
requested range. 56b572e8c established the XRT-matching policy (cache lines below the
LLC, KMT above) for buffers and explicitly deferred the aperture; this applies it
there too. Neither direction of the publication weakens: the staged bytes already
reach memory via the non-temporal stores and `sfence` in
`CopyAndCommitPathBCodeWrites`, and the NPU instruction cache is still invalidated by
the opcode-9 aperture sync that follows every stage.

This only pays off with 0007 applied. Without it the outstanding paging fences make
every submit drain them first, so removing the KMT transition just relocates the wait
(measured 38.65 → 39.58 tok/s, i.e. noise). Measured interleaved on top of 0007:
`qwen3.6-moe:35b-a3b` 44.84 → 54.13 tok/s (TTFT 21.78 → 18.05 s). With 0007+0008 the
MoE prefill gap to XRT (58.36 tok/s, 16.8 s) is 1.08×, in line with the 1.174× geomean
across 28 dense chat models; `qwen3:0.6b` reaches 1145 tok/s (XRT 1094) and
`gemma4-it:e2b` 560 tok/s. MoE ranges 49.7–55.9 tok/s depending on how long the device
has been under load; `qwen3.5:0.8b` at 4k drifts by the same proportion over a soak, so
that spread is device clock state, not backend behaviour.

Caveat: this is the one patch in the series that crosses a boundary upstream drew
deliberately ("Command-aperture invalidation … remain unchanged"). The reasoning is
derived from the code and from measurement, not from firmware documentation. If
NPU-side coherence for aperture code turns out to depend on the KMT transition rather
than on the opcode-9 publication, drop this patch — 0007 stands alone and keeps most
of the MoE win.

**Remove when:** upstream hrx applies its own sub-LLC synchronization policy to the
command aperture, or confirms the KMT transition is required there.

### 0009-remap-stale-ctrl-code-after-deferred-materialize.patch
Always re-maps a cached control-code BO before an in-place rewrite, and treats a
change in the TXN bytes themselves as `code_changed` on the PARTIAL_ELF chain path.

**Why:** Windows instruction BOs start as deferred host storage. The first submit
materializes them into a KMT mapping and frees that temporary storage, so a
`ctrl_code_mapped_ptr` captured at `make_npu_cmd` time then dangles. Later rewrites
memcpy into freed memory while Path-B publication copies the live BO — still the
first-token image. Even after remapping, Path-B skipped `SyncBuffer` because
`native_owns_control_code_publication` is true; Lock2 writes are not visible to the
aperture copy without that publish. Same-shape PARTIAL_ELF reuse whose TXN bytes
change (baked immediates, not the WRITE32 A1EC sentinel table) needs this remap
and Lock2 publish. It is **not** the qwen3.5 GDN decode mismatch: that RTP
program never entered this rewrite path (see 0010). `FLM_CTX_RTP_FREEZE=1`
matching XRT was a red herring for 0009; the hash stayed `4c83b2954062`.

**Remove when:** upstream hrx re-maps control-code BOs after deferred materialize
(or stops caching the pre-materialize host pointer) and compares TXN bytes on
PARTIAL_ELF chain reuse.

### 0010-partial-elf-empty-patch-table-rtp-chain.patch
Treat an empty host patch table as PARTIAL_ELF-eligible so WRITE32-only control
programs stay in the same chain group as the layer kernels.

**Why:** qwen3.5 `gen_set_context` is four `rtp_write`s of the context length `L`
and therefore has **no DDR patch triples**. The dispatcher required a non-empty
patch table for PARTIAL_ELF, so that run used START_NPU and opened a second
chain group. Windows Path-B submits the two groups as separate native commands;
the first attention child of the layer chain then reads the previous token's
RTP. `FLM_CTX_RTP_FREEZE=1` is the positive control (the missed WRITE32 is a
no-op). Prefill and the first decode still match XRT because `L` has not had to
change yet, or the first WRITE32 lands before any attention that depends on a
new value. GDN layers 0-2 do not read that register; layer 3 is the first
attention. Putting the empty-table program on PARTIAL_ELF makes it child 0 of
the same 24-slot runlist.

**Remove when:** upstream hrx allows empty host patch tables on the PARTIAL_ELF
path, or otherwise keeps WRITE32-only dispatches in the same native chain as
the kernels that consume those registers.

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

The former 0003 (adapt aiecc native CLI) was dropped at the iron-ops `aab8083`
pin: IRON #145/#150 now compile through mlir-aie's `compile_mlir_module()`,
which already emits native `aiecc --get-*` flags.

### 0004-windows-link-fallback.patch
Copies aiecc work-dir inputs when `pathlib.Path.symlink_to` is denied. Windows
without Developer Mode cannot create symlinks; CI runners typically can.

**Remove when:** iron-ops copies (or hard-links) those work-dir inputs upstream.

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

### 0001-eudsl-offline-find-links.patch
Edits one file in the submodule (upstream `Xilinx/mlir-aie`):
1. `python/requirements.txt`: leaves mlir-aie's `eudsl-python-extras` version
   line alone, and repoints its `-f` find-links from `https://llvm.github.io/eudsl`
   to the local prefetch cache (`../../../.cache/find-links`) so `vendor_eudsl.py`
   never lists the remote.

(The former second hunk — `--no-build-isolation` in `vendor_eudsl.py` — is now upstream
in mlir-aie's `install_eudsl`, so this patch is `requirements.txt` only. The former
version override to `0.1.0.20260629` was dropped when this stack's LLVM pin aligned
with mlir-aie's `clone-llvm.sh` commit.)

Combined with `dev.py` setting `PIP_FIND_LINKS` (absolute cache) + `PIP_NO_INDEX=1`
for the iron-tools build, eudsl vendoring is **network-free** (resolved from the
sha256-verified prefetched sdist).

**Why:** mlir-aie pins eudsl from `https://llvm.github.io/eudsl`, which `PIP_NO_INDEX`
cannot see. The patch only rewrites that `-f` onto `.cache/find-links`.

**Remove when:** mlir-aie's `vendor_eudsl.py` / `requirements.txt` can resolve eudsl
from a local find-links cache without a remote `-f`.

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
