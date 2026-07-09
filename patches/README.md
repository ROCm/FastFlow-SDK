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
