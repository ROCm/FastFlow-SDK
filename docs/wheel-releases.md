# FastFlow SDK wheel releases

> [!NOTE]
> This document is the **release-operator** policy for versioning, validating,
> and publishing wheels. To *install* a published build, use the
> [FastFlow SDK package index](#package-index) and the
> [README install section](../README.md#install-a-release-and-run-models).

FastFlow SDK compiler, ops, runtime, and FastFlowLM wheels use
[PEP 440](https://packaging.python.org/en/latest/specifications/version-specifiers/)
versions. The release train targets one stable version while development and
release-candidate builds approach that version:

```text
<base>.dev<UTC date><GitHub run number>
<base>rc1
<base>rc2
<base>
```

Development releases may continue for as long as needed. Do not advance the
patch component merely to publish another development build. Advance it only
after a stable release when beginning a new patch train.

## Version authority

`release.toml` is the single source of truth for the current wheel release
train. To begin a new train, change only:

```toml
[wheel_release]
base_version = "<new stable target>"
compiler_python_stable_abi = "3.12"
compiler_python_versions = ["3.12", "3.13", "3.14"]
```

The release tooling reads this value when it generates an automatic development
version. Workflows and documentation must not duplicate the current numeric
base version.

The top-level `schema_version` identifies the policy format, not the wheel
version. `compiler_python_stable_abi` sets the bindings ABI floor, and
`compiler_python_versions` lists the supported interpreters.

An explicit `package_version` supplied to a manual workflow dispatch overrides
automatic development-version generation. Use that override for release
candidates and stable-release validation. Explicit versions must be the
configured base version, `<base>rcN`, or `<base>.devN`; versions from another
train and alpha, beta, post, or local versions are rejected. Automated GitHub
publication is currently disabled. Development and RC versions may use
`publish_target=github-prerelease` to build the attested release payload, while
the stable base version is accepted only with `publish_target=none`; neither
mode creates a tag or release.

Local non-release wheel builds use a commit-derived development version. That
fallback identifies an individual checkout and is intentionally separate from
the published release train.

## Cross-platform releases

Windows and Linux wheels participating in one coordinated release use the same
version. FastFlowLM requires the runtime wheel at exactly that version. Never
replace or reuse a published version; produce a new development or release
candidate version instead.

`fastflow-sdk-qualification` is `py3-none-any` and requires
`fastflow-sdk-fastflowlm` at exactly the same version plus NumPy 2.4.6 and
ml-dtypes 0.6.0. Pip resolves the numeric wheels from the package index for
CPython 3.11 through 3.14 on Linux and Windows. Candidate v1 binds the required
versions, and qualification evidence records the installed wheel tags, RECORD
digest, and verified payload digest.
Linux and Windows generate the qualification wheel independently from the same
source pins. Linux publishes the canonical
wheel. Windows NPU jobs currently install the Windows-local wheel to avoid
depending on Linux artifact scheduling; those runs remain platform validation,
not canonical candidate evidence. When the canonical artifact is supplied,
Windows must match its qualification contract and XADX provenance before using
it. The platform-native builds therefore remain parallel. The qualification
wheel is an optional GitHub Release asset;
installing the runtime or FastFlowLM products does not install qualification.

The qualification wheel owns compiler-free XADX executables. FastFlowLM wheels
continue to own the raw XCLBINs used for inference and do not carry a second
XADX representation. A release consequently contains the raw XCLBIN and its
embedded XADX copy because the two consumers require different formats; moving
XADX into the FastFlowLM wheel would not remove that duplication. XADX
generation needs initialized FastFlowLM headers, engine libraries, and
production XCLBINs plus an installed HRX serializer, but not a built `flm`
executable.

The Linux runtime wheel is tagged `manylinux_2_28_x86_64` because its complete
native closure satisfies that policy. The FastFlowLM wheel is conservatively
tagged `linux_x86_64`: it resolves `libhrx.so.0` from the separately installed
runtime wheel, while manylinux policy evaluates each wheel independently.

The compiler is a coordinated split distribution:

- `fastflow-sdk-compiler` is the platform-neutral user-facing meta-package.
- `fastflow-sdk-compiler-core` contains the LLVM/MLIR/IRON tools, headers, Peano
  payload, and Python-independent native libraries once per platform.
- `fastflow-sdk-compiler-bindings` contains the `aie` Python package and native
  extensions built once against CPython 3.12's stable ABI. Its wheel tag is
  `cp312-abi3`, which supports the validated Python 3.12 through 3.14 range.

Both the meta-package and bindings require the matching core at exactly the
same version. Moving the 127 MB `AIEAggregateCAPI` library into the core avoids
duplicating it in every CPython bindings wheel. Importing `aie` activates the
core wheel's native-library directory before loading its extension modules.
On Windows, packaging replaces byte-identical Peano multicall aliases with
small native launchers that preserve each Clang, LLD, or readelf driver flavor.
It keeps one full `clang++.exe`, `lld.exe`, and `llvm-readobj.exe`; a changed
upstream alias fails packaging instead of being silently replaced. Compiler
wheels use maximum DEFLATE compression to reduce release and PyPI storage. The
Windows wheel audit compiles, preprocesses, and links through the forwarded
Clang, GNU LLD, COFF LLD, and readelf modes, and checks the Darwin and
WebAssembly LLD flavor selection.
Linux packaging similarly replaces the large LLD and readelf multicall copies
with flavor-preserving shell launchers. It removes the byte-identical
versioned `libLLVM-<major>.so` copy only after comparing it with the retained
`libLLVM.so`, and strips unneeded symbols from staged ELF executables and
shared libraries.
The installed-wheel audit exercises each forwarded alias in its intended
driver flavor and validates the ELF dependency closure after these
transformations; the manylinux policy audit then checks the finished wheels.
When `fastflow-sdk-runtime` is also installed, the same import selects HRX for
in-process IRON (`NPU_RUNTIME`, `IRON_RUNTIME`, and `HRX_LIBHRX` from
`fastflow_sdk_runtime.libhrx_path()`) if those variables are unset. Compiler
wheels do not depend on the runtime package; a compiler-only install skips
the hook. Importing `fastflow_sdk_runtime` performs the same HRX selection.

`fastflow-sdk-ops` is `py3-none-any` and requires `fastflow-sdk-compiler` at exactly
the same version. It ships the IRON `iron` package plus the `aie_kernels`
sources that `AIEContext.base_dir` resolves next to it, so
`from iron.operators import ElementwiseAdd` works after a wheel install.
The wheel is compile-oriented: it does not depend on torch. CPU reference
helpers and stream-dse still import torch at module scope; MHA and the
hand-written SwiGLU sequences do not. `verify-ops-wheels` imports every
`iron.operators.*.op` module (and every lazy `iron.operators` export), and fails
if a torch skip is unused or a skip-list package is missing. CI then installs
that wheel into the Python 3.12 compiler verify venv and compiles every
torch-free operator in `build_tools.iron.host_compile_suite()` in parallel
(the same small shapes as `test-iron-ops`, so `generic/` and `aie2p/` kernels
both go through Peano from site-packages). MHA is in that compile catalog;
IRON devel binds `get_current_device()` itself. The compiler wheel includes `aiebu-asm`;
the dynamic strided-copy row compile-checks runtime parameters in a fused ELF,
and the fused add-to-ReLU row compile-checks multi-PDI reconfiguration and
intermediate scratch routing. SwiGLU stays in `OPERATORS` but is not
host-compiled because it has no packaged numeric case.
NPU CI does not run iron-ops pytest: it dispatches the prebuilt
`npu-test-package` for operators that have `CASE_GENERATORS`.
`dev.py test-npu-iron-ops` is local and still compiles inside `run_test`; CPU
CI is the off-device drift gate.
3.13/3.14 already proved the abi3 compiler; ops is `py3-none-any`. Linux
publishes the canonical ops wheel with the compiler wheel family (same artifact
name; ops files are listed explicitly); Windows still packages and
compile-checks it so a packaging bug cannot hide on one host.

Release collection still resolves the compiler meta-wheel and that ops wheel
with `python -m build_tools.resolve_release_wheel` for every
`compiler_python_versions` interpreter on Linux and Windows (pip `--platform` /
`--abi`, not an import).

`fastflow-sdk` is the `py3-none-any` one-step install. It ships no code (its
only package is the empty `fastflow_sdk_meta`) and requires
`fastflow-sdk-fastflowlm`, `fastflow-sdk-compiler`, and `fastflow-sdk-ops` at
exactly the same version, so it inherits the compiler's Python range. The Linux
compiler job packages it with `dev.py package-sdk-wheel`, and both release
collection workflows resolve it against the collected payload with
`resolve_release_wheel --product sdk`. Releases before
`0.1.0.dev202610230003` do not include it.

The unified workflow builds and validates Linux and Windows wheels from one
source commit. An operator may publish the validated payload manually under
`fastflow-sdk-v<version>` after evidence intake succeeds.

## Manual workflow

The `Release FastFlow SDK Wheels` workflow defaults to validation only. With
`publish_target=none`, it builds and validates both platform wheel sets and
uploads a temporary Actions artifact. `publish_target=github-prerelease` also
builds and attests the release payload, but its GitHub publication job is
disabled; it does not create or push a tag.
The same payload includes the tested standalone FastFlowLM + HRX runtime
archives (`fastflow-sdk-linux-x86_64.tar.zst` and `fastflow-sdk-windows-x86_64.zip`),
checksum sidecars, and detached bundle provenance manifests. Compiler CI also
builds relocatable product archives (`fastflow-sdk-compiler-linux-x86_64.tar.gz` /
`fastflow-sdk-compiler-windows-x86_64.tar.zst`) for packaging and verification, but
only compiler, ops, and `fastflow-sdk` wheels are collected into this release
payload. The ops
wheel is the canonical Linux `py3-none-any` file from the Linux compiler-wheels
artifact. Release assembly
rejects runtime archives whose embedded version, source revision, platform, or
archive digest does not match the resolved release.

Pull-request CI builds one stable-ABI bindings wheel per platform and validates
that same wheel on Python 3.12, 3.13, and 3.14.

The workflow resolves the requested ref once and passes its immutable commit
SHA through build, NPU validation, manifest generation, and release assembly.
The human-readable source ref remains in the release metadata. Validation-only
runs may be dispatched from a pull-request branch and have read-only
permissions; they do not request an OIDC token or create attestations.
Publication runs must be dispatched from `main`, and the resolved source commit
must be an ancestor of `origin/main`. Only those publication builds receive
attestation permissions.

Qualification candidate and evidence publication is deliberately manual.
Release collection emits one self-contained candidate containing every Linux
and Windows runtime/FastFlowLM wheel, the shared qualification wheel,
`candidate-manifest.json`, and `SHA256SUMS`. Every operator machine runs against
that same manifest. The operator collects deterministic `.evidence.zip`
bundles, transfers them to an intake machine, runs `fastflow qualify verify`
and `fastflow qualify intake --policy release --require-status pass`, and
reviews the recorded policy satisfaction in `qualification-index.json`.

The GitHub prerelease publication job is currently disabled. Download the
complete collected Actions artifact, create the draft release and tag manually,
then run `fastflow qualify stage-release` to assemble a single flat upload
directory from that payload plus the intake aggregate
(`qualification-evidence-<version>.zip`), sidecar, and
`qualification-index.json` into an empty output directory. Upload every
top-level file from that directory (wheels including ops and `fastflow-sdk`,
native archives, metadata, checksums, candidate manifest, and the three
evidence attachments).
Do not upload nested `simple/` trees. Before publishing, recheck that every
staged wheel still has the hash recorded in the candidate manifest. Any
rebuilt or replaced wheel invalidates the evidence and requires a new candidate
plus new runs.
Verify the aggregate ZIP with `fastflow qualify verify` before staging or
manually publishing. No workflow creates or publishes the release, transfers
manual evidence, or signs evidence.

`SHA256SUMS` covers the prepared top-level product assets.
`candidate-SHA256SUMS` separately covers `candidate-manifest.json` and its
qualification wheels without replacing the release checksum. The nested PEP
503 `simple/` index in the complete Actions artifact covers only that payload;
the public index is published separately (see [Package index](#package-index)).

## Package index

`https://rocm.github.io/FastFlow-SDK/simple/` is a PEP 503 simple index built
from the wheels attached to every published, non-draft `fastflow-sdk-v*`
release. Its links point at the release download URLs and carry the asset's
GitHub SHA-256 digest, falling back to the release `SHA256SUMS`. Wheels are
never copied to Pages. Each project page lists every release that ships that
project, so `fastflow-sdk` starts at `0.1.0.dev202610230003`.

The index does not update itself. After publishing a release, run the
`Publish FastFlow SDK simple index` workflow from `main` (Actions → Run
workflow); it rebuilds the whole index from the releases API
(`build_tools/publish_fastflow_sdk_simple_index.py`) and deploys it. The
repository's Pages source must be set to GitHub Actions.

All published versions are currently development builds. Install commands use
`--pre` because PyPI hosts an unrelated `fastflow-sdk 0.0.1` final release that
pip otherwise prefers; an exact `==<version>` pin works without it.

Each wheel uses Core Metadata 2.4 with an SPDX license expression and copies
all applicable license and notice files under `.dist-info/licenses/`. It also
contains a CycloneDX 1.6 SBOM under `.dist-info/sboms/`; the SBOM and the
package-local manifest retain source, dependency, toolchain, binary-closure,
and per-file provenance for the shipped product. Tests validate each SBOM
against the official CycloneDX 1.6 schema and require its file hashes to match
the package manifest and wheel `RECORD`. The FastFlowLM SBOM also records its
exact runtime-wheel dependency.

After publication and the index workflow, validate the package index end to
end from clean Linux and Windows virtual environments. Installing
`fastflow-sdk` pulls FastFlowLM, the runtime, the compiler, and ops:

```powershell
py -m pip install --no-cache-dir --only-binary=:all: `
  --extra-index-url https://rocm.github.io/FastFlow-SDK/simple `
  "fastflow-sdk==<version>"
py -m pip check
py -c "import fastflow_sdk_runtime, aie; from iron.operators import ElementwiseAdd; print(ElementwiseAdd.__name__)"
flm version
aie-opt --version
aiecc --help
```

```bash
python3 -m pip install --no-cache-dir --only-binary=:all: \
  --extra-index-url https://rocm.github.io/FastFlow-SDK/simple \
  "fastflow-sdk==<version>"
python3 -m pip check
python3 -c "import fastflow_sdk_runtime, aie; from iron.operators import ElementwiseAdd; print(ElementwiseAdd.__name__)"
flm version
aie-opt --version
aiecc --help
```

Also check that an unpinned `--pre` install of `fastflow-sdk` resolves to the
new version.

The release workflow creates GitHub build-provenance attestations for published
wheels and standalone runtime archives in each native platform build job,
before artifact upload. Publication still waits for the Linux and Windows NPU
jobs to validate the wheel-installed runtime and CLI on hardware.
