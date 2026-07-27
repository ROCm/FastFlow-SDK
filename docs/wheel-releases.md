# ROCm NPU wheel releases

ROCm NPU runtime and FastFlowLM wheels use
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
```

The release tooling reads this value when it generates an automatic development
version. Workflows and documentation must not duplicate the current numeric
base version.

The top-level `schema_version` identifies the format of `release.toml`; it is
independent of the wheel version. Leave it at `1` when changing
`wheel_release.base_version`. Increment it only for an incompatible change to
the policy file format, together with corresponding release-tooling support.

An explicit `package_version` supplied to a manual workflow dispatch overrides
automatic development-version generation. Use that override for release
candidates and stable-release validation. Explicit versions must be the
configured base version, `<base>rcN`, or `<base>.devN`; versions from another
train and alpha, beta, post, or local versions are rejected. At this stage,
GitHub publication is prerelease-only: development and RC versions may use
`publish_target=github-prerelease`, while the stable base version is accepted
only with `publish_target=none`.

Local non-release wheel builds use a commit-derived development version. That
fallback identifies an individual checkout and is intentionally separate from
the published release train.

## Cross-platform releases

Windows and Linux wheels participating in one coordinated release use the same
version. FastFlowLM requires the runtime wheel at exactly that version. Never
replace or reuse a published version; produce a new development or release
candidate version instead.

The Linux runtime wheel is tagged `manylinux_2_28_x86_64` because its complete
native closure satisfies that policy. The FastFlowLM wheel is conservatively
tagged `linux_x86_64`: it resolves `libhrx.so.0` from the separately installed
runtime wheel, while manylinux policy evaluates each wheel independently.

The unified workflow builds Linux and Windows wheels in parallel from one
immutable source commit, completes the platform-specific NPU validation, and
publishes all four wheels under the platform-neutral `rocm-npu-v<version>` tag.
Publication is atomic: a failed platform prevents creation of the immutable
GitHub release.

## Manual workflow

The `Release ROCm NPU Wheels` workflow defaults to validation only. With
`publish_target=none`, it builds and validates both platform wheel sets and
uploads a temporary Actions artifact. With
`publish_target=github-prerelease`, the final job publishes one GitHub
prerelease after both builds, metadata checks, dependency resolution, and all
NPU validations succeed.

The workflow resolves the requested ref once and passes its immutable commit
SHA through build, NPU validation, manifest generation, and release assembly.
The human-readable source ref remains in the release metadata. Validation-only
runs may be dispatched from a pull-request branch and have read-only
permissions; they do not request an OIDC token or create attestations.
Publication runs must be dispatched from `main`, and the resolved source commit
must be an ancestor of `origin/main`. Only those publication builds receive
attestation permissions.
`SHA256SUMS` covers only the flat, top-level files uploaded as GitHub Release
assets. The nested PEP 503 `simple/` index remains available in the complete
Actions artifact, and its wheel links carry their own SHA-256 fragments.

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
  --find-links "https://github.com/<owner>/<repo>/releases/expanded_assets/<tag>" `
  "rocm-npu-fastflowlm==<version>"
py -m pip check
```

```bash
python3 -m pip install --no-cache-dir --only-binary=:all: \
  --find-links "https://github.com/<owner>/<repo>/releases/expanded_assets/<tag>" \
  "rocm-npu-fastflowlm==<version>"
python3 -m pip check
```

The release workflow creates GitHub build-provenance attestations for published
wheels in each native platform build job, before artifact upload. Publication
still waits for every NPU validation to succeed.
