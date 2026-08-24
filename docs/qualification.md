# ROCm NPU model qualification

Install the optional qualification product from the same release assets:

```bash
python -m pip install --find-links <release-assets> \
  "rocm-npu-qualification==<version>"
```

The package depends on the exact same-version `rocm-npu-fastflowlm`, which in
turn depends on `rocm-npu-runtime`, and pins NumPy 2.4.6 and ml-dtypes 0.5.4.
Pip resolves those binary dependencies from the package index. The candidate
manifest binds their required versions, while each evidence bundle records the
installed wheel tags, RECORD digest, and verified payload digest. Only the
runtime wheel owns the `rocm-npu` entry point. The qualification wheel installs
`rocm-npu-qualify`, discovered through the external plugin protocol.
CI resolves Linux and Windows dependencies for every supported CPython ABI from
the single numeric-runtime contract. Release assembly validates that complete
dependency matrix without publishing the third-party wheels.

List and run profiles through the native core:

```bash
rocm-npu plugin list --json
rocm-npu qualify --list-profiles --json
rocm-npu qualify --output qualification-evidence
```

Profiles are package JSON data conforming to `profile-v1.schema.json`.
With no profile option, qualification runs the comprehensive `default` suite:
Qwen3 0.6B, content-pinned Qwen3.5 0.8B, and LFM2 1.2B multi-turn,
bounded-soak, and 1k/4k benchmark cases. Its measured warm-cache NPU2 runtime
is kept below 15 minutes. `--fast` runs the CI gate: five fresh
Qwen3 processes with two deterministic turns each, the production XCLBIN
inventory, the 1k benchmark/CSV checkpoint, native self-test, and the embedded
FastFlowLM XADX stress set. `--profile extended` is an opt-in default-suite
superset adding revision-pinned Llama 3.1 8B multi-turn, soak, and 1k/4k
coverage. CI selects only `--fast`; default and extended are explicit
qualification profiles for broader local or release validation.

`--all-models` runs a complete sweep over *every* model in the installed
FastFlowLM registry instead of a curated profile. Models are discovered
dynamically from `model_list.json` at runtime, so their per-file and XCLBIN
digests are not pre-pinned (identity still comes from the registry URL:
`/resolve/<revision>` pins an immutable revision, other URLs are
content-sourced). Each chat model runs a single-prompt turn, a five-turn
multi-turn conversation, and a benchmark/CSV checkpoint; embed and ASR
models are exercised through a `flm serve` probe (`serve`-kind evidence). The
benchmark stage defaults to 8k context (a three-iteration `bench_8k_smoke.json`
sweep) and can be set to `1k`, `4k`, or `8k` with `--bench-context`.
Per-model timeouts scale with the model footprint and with the benchmark
context length, and the sweep implies `--keep-going` so one failing model does
not abort the run. Because it is a dynamic diagnostic sweep rather than a pinned
profile, `--all-models` is meant for broad local coverage and is not a
candidate-bound release path.

`--bench` runs a measurement-only variant of the sweep. `MODE` selects the
coverage; `--bench all` (the mode used when the flag is given bare) covers
*every* model in the installed registry, but drops the correctness stages: chat
models run *only* the on-NPU decode benchmark (no single-prompt or multi-turn
turns), and embed/ASR models run *only* their serve-probe (the same
encode-tok/s, RTF, and WER measurements the complete sweep records). It defaults
to a `1k` context benchmark of five iterations; tune the context with
`--bench-context <1k|4k|8k>` and the iteration count with `--bench-iterations
<N>` (a positive integer). Pass `--models=<tag>[,<tag>...]` to narrow the
sweep to an explicit subset of registry tags (tags absent from the installed
registry are skipped with a warning); it has no effect without `--bench`.
Like `--all-models`, `--bench` is a dynamic diagnostic sweep (not pinned,
not candidate-bound) and implies `--keep-going`.

`--xrt-flm <path>` (only meaningful with `--bench`) additionally exercises
each model through an XRT-backed FastFlowLM build and widens the report to a
side-by-side HRX-vs-XRT comparison. For chat models the benchmark table gains
XRT columns (`Model | XRT TTFT s | HRX TTFT s | XRT prefill tok/s | HRX prefill
tok/s | XRT decode tok/s | HRX decode tok/s`); for embed/ASR models the
Embedding and ASR probe tables gain XRT columns for the same measurements the
HRX probe records (encode tok/s; RTF and WER). It is off by default -- simply
supplying the path is what enables the comparison (there is no separate enable
flag). Because the shipped wheel only bundles the HRX runtime, point `--xrt-flm`
at a separately built XRT entrypoint (the portable `flm` wrapper, or its install
root) produced by `python dev.py build fastflowlm-xrt`, which stages a
self-contained tree (bundled XRT libraries, the vcpkg runtime closure, and --
when discoverable -- the XRT XDNA driver plugin). XRT is strictly best-effort: a
missing/malformed XRT build or a failing XRT run is recorded as `N/A` and folded
into the HRX record's metrics -- it never emits a separate test and never
changes the run's pass/fail (so a fully passing HRX sweep still passes even when
XRT is unavailable).

In addition to the machine-readable bundle, the `--all-models` and `--bench`
sweeps write a human-readable `report.md` into the output directory. It contains
an environment header (OS, NPU device/driver/firmware candidates, target family,
FLM and package versions), a per-model functional matrix (single-prompt,
five-turn multi-turn, and whether the multi-turn run passed) where those turns
ran, and an on-NPU benchmark table with the FLM-averaged TTFT, prefill, and
decode throughput for each model. By default only HRX numbers are reported. The
context stage and the iteration count the averages span (for example
`HRX, 8k, averaged over 3 iterations`) are stated in the benchmark section
heading rather than repeated per row. Profile runs (`--fast`,
`--profile`) do not emit `report.md`; they produce only the evidence bundle
(`evidence.json`, `tests.json`, `junit.xml`, and the summary).

Qualification first validates plugin protocol and exact package versions, then
runs native `version`, `probe`, `doctor`, and `self-test` through the absolute executable
provided by the core. It inspects the configured model cache and runs
`flm pull <tag>` when a model is missing. Pull and test output is captured in
the evidence logs. `--no-pull` fails instead of downloading. `--offline` never
invokes the network-backed `flm check` or `flm pull`. It resolves the
model-specific directory and required files from the installed registry,
applies offline and proxy-denial controls to inference, and fails if that local
contract is incomplete.

The output directory is a complete report bundle. `evidence.json` records the
overall result, package and candidate identity, source revision, observed
driver/firmware/platform provenance, content-pinned model inventories, and every
native, XADX, run, soak, and benchmark result. Successful native checks are
recorded as tests, not merely omitted. `junit.xml` mirrors test outcomes;
`core-version.json`, `probe.json`, `doctor.json`, `self-test.json`, benchmark
CSVs, per-command logs, and `SHA256SUMS` retain the underlying details.

Useful options include:

- `--model-cache <path>`
- `--candidate-manifest <path>`
- `--keep-going`
- `--bench-context <1k|4k|8k>` (with `--bench` or `--all-models`; default `1k`
  for `--bench`, `8k` for `--all-models`)
- `--bench [all]` (measurement-only sweep; `all` covers the whole registry)
- `--bench-iterations <N>` (decode-benchmark iterations per chat model for
  `--bench`; a positive integer, default `5`)
- `--models <tag>[,<tag>...]` (restrict a `--bench` sweep to explicit
  registry tags; ignored without `--bench`)
- `--xrt-flm <path>` (with `--bench`; add a best-effort HRX-vs-XRT
  comparison using a separately built XRT FastFlowLM, off by default)
- `--json`

Profiles support ad-hoc diagnostics and candidate-bound release runs. Packaged
profiles pin a canonical-JSON model registry digest and model file digests;
the qualification wheel also contains a 53-case NPU2 kernel recipe covering
seven distinct production implementations plus explicit execution variants.
Those XADX executables embed the production XCLBIN and serialized control
transactions. Inputs and expected outputs are generated deterministically at
runtime from the qualification-owned capability contract.
Reports bind the exact per-case dispatch/readback schedule, validated byte
count, and deduplicated materialization inventory. Default 32x256 Q4NX parity
is converter-backed; 32x128 and 16x128 are measured-fact C++/Python interpreter
agreement, with the 53-case device run serving as the independent authority.
See the [qualification evidence v1 contract](contracts/qualification-evidence-v1.md)
for the exact identity, archive, and verification rules.

## Detached evidence collection and intake

Linux publishes the canonical `py3-none-any` qualification wheel. Windows
independently generates the same wheel. While Linux artifact scheduling is
constrained, Windows NPU jobs install that Windows-local wheel; those runs
validate the platform build but do not replace canonical candidate evidence.
When a canonical artifact is explicitly supplied, the Windows workflow first
matches normalized case, source-XCLBIN, transaction, serializer, and XADX
hashes. Release assembly creates one candidate from all Linux and Windows
runtime/FastFlowLM wheels and the canonical wheel:

```bash
rocm-npu qualify candidate \
  --wheel-dir candidate/wheels \
  --output candidate/candidate-manifest.json \
  --expected-source-revision <full-commit> --json
```

The command infers the source revision from every wheel and rejects disagreement.
It writes the strict candidate v1 `candidate-manifest.json` plus
`candidate-SHA256SUMS`; the manifest and all ROCm NPU wheels must share one
artifact root. Transfer that complete
release-wide candidate artifact to every manual hardware machine without
rebuilding or selecting only one platform's identities.
Treat `candidate-SHA256SUMS` as the publication commit marker: consumers must
require it and verify every listed path before using `candidate-manifest.json`.

Collect and seal one run in a single command:

```bash
rocm-npu qualify run \
  --candidate-manifest candidate/candidate-manifest.json \
  --model-cache <cache> \
  --output-dir machine-run \
  --bundle-output machine-run.evidence.zip --json
```

The directory remains available for diagnosis. Finalized pass, fail, and
interrupted directories can also be sealed later with:

```bash
rocm-npu qualify pack \
  --input machine-run --output machine-run.evidence.zip --json
```

ZIP output is deterministic and contains the finalized evidence files exactly
as covered by the internal `SHA256SUMS`. Verify transferred evidence without
NPU hardware, the core plugin environment, network access, or repository build
tools:

```bash
rocm-npu qualify verify \
  --input machine-run.evidence.zip \
  --candidate-manifest candidate/candidate-manifest.json --json
```

Verification applies the archive, candidate, profile, model, platform-probe,
execution-inventory, and checksum rules defined by the evidence contract.

After transferring independently collected machine bundles into one directory,
assemble release intake with:

```bash
rocm-npu qualify intake \
  --input-dir incoming \
  --candidate-manifest candidate/candidate-manifest.json \
  --output-dir assembled \
  --bundle-output qualification-evidence-<version>.zip \
  --policy release --require-status pass --json
```

Intake re-verifies every original bundle, preserves it unchanged under its
SHA256 name, rejects duplicates and mixed/ad-hoc candidates, writes the
versioned `qualification-index.json` and aggregate `SHA256SUMS`, then emits a
deterministic aggregate ZIP and an external
`qualification-evidence-<version>.zip.sha256`. The checked-in `release` policy
requires passing NPU2 `fast` evidence from both Windows and Linux and records
the matched run digests in the index. Verify the final asset itself with:

```bash
rocm-npu qualify verify \
  --input qualification-evidence-<version>.zip \
  --candidate-manifest candidate/candidate-manifest.json --json
```

Artifact transfer and publication remain manual. See
[wheel releases](wheel-releases.md) for the operator flow; rebuilding any
candidate wheel invalidates all collected evidence.

For a shared cache across worktrees:

```powershell
$cache = 'C:\path\to\rocm-npu-staging\.cache\FastFlowLM-models'
rocm-npu qualify --model-cache $cache --output qualification-evidence
rocm-npu qualify --profile extended --model-cache $cache --output extended-evidence
rocm-npu qualify --all-models --model-cache $cache --output all-models-evidence
rocm-npu qualify --all-models --bench-context 1k --model-cache $cache --output all-models-1k-evidence
rocm-npu qualify --bench all --model-cache $cache --output bench-all-evidence
rocm-npu qualify --bench all --bench-context 4k --bench-iterations 20 --model-cache $cache --output bench-all-4k-evidence
rocm-npu qualify --bench all --bench-context 4k --bench-iterations 20 --xrt-flm C:\path\to\FastFlowLM-xrt-install --model-cache $cache --output bench-all-xrt-evidence
rocm-npu qualify --offline --no-pull --model-cache $cache --output offline-evidence
```

On Linux, use the equivalent absolute path. `FLM_MODEL_PATH` may be set instead
of passing `--model-cache`.

Diagnostics answer whether the package and bounded native dispatch work.
Qualification executes profile-defined installed-product model cases and
produces structured evidence. Release acceptance is a later policy and signing
decision; a passing evidence bundle does not publish or accept a release by
itself.
