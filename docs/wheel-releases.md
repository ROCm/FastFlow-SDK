# ROCm NPU wheel releases

> [!NOTE]
> This document is the **release-operator** policy for versioning, validating,
> and publishing wheels. To *install* a published build, use
> [GitHub Releases](https://github.com/ROCm/rocm-npu/releases) and the
> [README install section](../README.md#install-a-release-and-run-models).

ROCm NPU compiler, runtime, and FastFlowLM wheels use
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

`rocm-npu-qualification` is `py3-none-any` and requires
`rocm-npu-fastflowlm` at exactly the same version plus NumPy 2.4.6 and
ml-dtypes 0.5.4. Pip resolves the numeric wheels from the package index for
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

- `rocm-npu-compiler` is the platform-neutral user-facing meta-package.
- `rocm-npu-compiler-core` contains the LLVM/MLIR/IRON tools, headers, Peano
  payload, and Python-independent native libraries once per platform.
- `rocm-npu-compiler-bindings` contains the `aie` Python package and native
  extensions built once against CPython 3.12's stable ABI. Its wheel tag is
  `cp312-abi3`, which supports the validated Python 3.12 through 3.14 range.

Both the meta-package and bindings require the matching core at exactly the
same version. Moving the 127 MB `AIEAggregateCAPI` library into the core avoids
duplicating it in every CPython bindings wheel. Importing `aie` activates the
core wheel's native-library directory before loading its extension modules.

The unified workflow builds and validates Linux and Windows wheels from one
source commit. An operator may publish the validated payload manually under
`rocm-npu-v<version>` after evidence intake succeeds.

## Manual workflow

The `Release ROCm NPU Wheels` workflow defaults to validation only. With
`publish_target=none`, it builds and validates both platform wheel sets and
uploads a temporary Actions artifact. `publish_target=github-prerelease` also
builds and attests the release payload, but its GitHub publication job is
disabled; it does not create or push a tag.
The same payload includes the tested standalone FastFlowLM + HRX runtime
archives (`rocm-npu-linux-x86_64.tar.zst` and `rocm-npu-windows-x86_64.zip`),
checksum sidecars, and detached bundle provenance manifests. Compiler CI also
builds relocatable product archives (`rocm-npu-compiler-linux-x86_64.tar.gz` /
`rocm-npu-compiler-windows-x86_64.tar.zst`) for packaging and verification, but
only compiler wheels are collected into this release payload. Release assembly
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
bundles, transfers them to an intake machine, runs `rocm-npu qualify verify`
and `rocm-npu qualify intake --policy release --require-status pass`, and
reviews the recorded policy satisfaction in `qualification-index.json`.

The GitHub prerelease publication job is currently disabled. Download the
complete collected Actions artifact, create the draft release and tag manually,
then upload the complete prepared release payload (including runtime,
FastFlowLM, qualification, and compiler wheels, native archives, metadata, and
indexes), `candidate-manifest.json`, `SHA256SUMS`,
`candidate-SHA256SUMS`, the aggregate evidence ZIP, its external
`.zip.sha256`, and `qualification-index.json`. Before publishing, recheck that
every staged wheel still has the hash recorded in the candidate manifest. Any
rebuilt or replaced wheel invalidates the evidence and requires a new candidate
plus new runs.
Verify the aggregate ZIP with `rocm-npu qualify verify` before manually
publishing. No workflow creates or publishes the release, transfers manual
evidence, or signs evidence.

`SHA256SUMS` covers the prepared top-level product assets.
`candidate-SHA256SUMS` separately covers `candidate-manifest.json` and its
qualification wheels without replacing the release checksum. The nested PEP
503 `simple/` index remains available in the complete Actions artifact, and its
wheel links carry their own SHA-256 fragments.

Each wheel uses Core Metadata 2.4 with an SPDX license expression and copies
all applicable license and notice files under `.dist-info/licenses/`. It also
contains a CycloneDX 1.6 SBOM under `.dist-info/sboms/`; the SBOM and the
package-local manifest retain source, dependency, toolchain, binary-closure,
and per-file provenance for the shipped product. Tests validate each SBOM
against the official CycloneDX 1.6 schema and require its file hashes to match
the package manifest and wheel `RECORD`. The FastFlowLM SBOM also records its
exact runtime-wheel dependency.

After publication, validate GitHub's expanded-assets page end to end
from clean Linux and Windows environments:

```powershell
py -m pip install --no-cache-dir --only-binary=:all: `
  --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/<tag>" `
  "rocm-npu-fastflowlm==<version>"
py -m pip check
py -m pip install --no-cache-dir --only-binary=:all: `
  --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/<tag>" `
  "rocm-npu-compiler==<version>"
py -m pip check
aie-opt --version
aiecc --help
```

```bash
python3 -m pip install --no-cache-dir --only-binary=:all: \
  --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/<tag>" \
  "rocm-npu-fastflowlm==<version>"
python3 -m pip check
python3 -m pip install --no-cache-dir --only-binary=:all: \
  --find-links "https://github.com/ROCm/rocm-npu/releases/expanded_assets/<tag>" \
  "rocm-npu-compiler==<version>"
python3 -m pip check
aie-opt --version
aiecc --help
```

The release workflow creates GitHub build-provenance attestations for published
wheels and standalone runtime archives in each native platform build job,
before artifact upload. Publication still waits for the Linux and Windows NPU
jobs to validate the wheel-installed runtime and CLI on hardware.
