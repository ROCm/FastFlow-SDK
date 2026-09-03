# Qualification evidence contract v1

A qualification run writes a portable directory containing:

- `evidence.json`: overall subject, platform, model, and test results
- `candidate-manifest.json`: a sanitized identity projection of the supplied
  release candidate, or an explicit ad-hoc placeholder
- `core-version.json`, `probe.json`, `doctor.json`, and `self-test.json`:
  native CLI reports
- `artifacts/native/powermode-before.json`: raw CLI report proving passive
  powermode reporting worked
- `artifacts/native/powermode-set-balanced.json`,
  `artifacts/native/powermode-after-balanced.json`,
  `artifacts/native/powermode-set-performance.json`,
  `artifacts/native/powermode-after-performance.json`,
  `artifacts/native/powermode-set-turbo.json`,
  `artifacts/native/powermode-after-turbo.json`,
  `artifacts/native/powermode-restore.json`, and
  `artifacts/native/powermode-final.json`: raw CLI reports backing the native
  mutating powermode roundtrip test when the host supports unprivileged
  power-mode changes
- `xadx-report.json`: schema-v2 XADX identity, per-case metrics, and structured
  phase/round failures
- `profile.json`: either the exact versioned `profile-v1` executed (curated
  `fast` / `default` / `extended` runs) or a `diagnostic-sweep` descriptor
  (`kind: "diagnostic-sweep"`) recording what a sweep actually ran (name, bench
  context, and the selected tags with their test names). The `--all-models`
  sweep records `name: "all-models"`; the measurement-only `--bench all` sweep
  records `name: "bench-all"` and its `bench_iterations`; `--bench dev` records
  `name: "bench-dev"` and its `bench_iterations`. A diagnostic-sweep
  descriptor is never a packaged contract: it pins no files or XCLBINs and is
  always ad-hoc (`candidate_bound: false`), so intake refuses it.
- `models.json`: model source, revision, pull state, and content hashes
- `junit.xml`: CI-compatible passed and failed test cases
- `report.md`: optional human-readable HRX overlay written for sweeps. It is
  covered by `SHA256SUMS` but is not schema-validated and carries no intake
  semantics; profile runs omit it. When `--bench --xrt-flm <path>` runs, the
  decode benchmark table and the Embedding/ASR probe tables widen to a
  side-by-side HRX-vs-XRT comparison.
- `logs/pull/` and `logs/tests/`: redacted subprocess transcripts
- `SHA256SUMS`: every other evidence file in deterministic path order

`evidence.json` conforms to
`tools/rocm-npu-qualification/python/rocm_npu_qualification/schemas/evidence-v1.schema.json`.
Status is
written as `running` before device work begins and updated as results arrive,
so interrupted and failed runs retain useful partial evidence. Both successes
and failures remain in the final bundle.

An optional `--bench --xrt-flm <path>` comparison never adds test entries: each
model's XRT result is folded into the matching HRX record under the free-form
`metrics.xrt` object (`status`, measurements, log/artifact references, or
`status: "unavailable"` with an error). Chat models fold onto the `bench`
record; embed/ASR models fold onto the `serve` record. This keeps XRT strictly
informational -- it never changes the run's `pass`/`fail` and needs no schema
change. When the flag is given, the diagnostic-sweep `profile.json` records the
requested path, the resolved XRT entrypoint, and the host `XILINX_XRT` prefix
(or `status: "unavailable"` with an error if the build or host XRT could not
be resolved). An unresolved `--xrt-flm` still stamps `metrics.xrt` unavailable
on every model so the report keeps XRT columns.

The subject records exact versions for `rocm-npu-runtime`,
`rocm-npu-fastflowlm`, and `rocm-npu-qualification`, the native core source
revision, and the candidate manifest hash. Model identities always include a
source, revision label, approved registry digest, and SHA-256 inventory of
observed model-cache files. Candidate-backed runs verify installed package
manifests and their complete declared file inventories before device work.
The candidate-v1 qualification contract binds the kernel-recipe-v1 digest,
generator and serializer revisions, transaction hashes, source-XCLBIN hashes,
serialized XADX hashes, and lifecycle, resident, and readback round counts.
Candidate v1 also binds the required NumPy/ml-dtypes versions. Each run records
the installed wheel tags, RECORD SHA-256, and verified payload SHA-256 in its
evidence subject.
Linux supplies the canonical qualification wheel;
Windows-local generation may be exercised for platform validation, but it must
match the canonical wheel's member names, uncompressed bytes, and normalized
modes before its evidence can stand in for a run against the canonical wheel.

Evidence intentionally excludes complete environment dumps. Known bearer
tokens, token query parameters, passwords, secrets, and API keys are redacted
from captured logs.

## Transport archive

`rocm-npu qualify run --bundle-output`, or the recoverability command
`rocm-npu qualify pack`, seals a finalized directory as a deterministic
`.evidence.zip`. The archive has sorted normalized POSIX member names, fixed
timestamps and permissions, no explicit directory entries, and contains
exactly the files covered by the internal `SHA256SUMS`. It does not wrap them
in an additional top-level directory.

`rocm-npu qualify verify` accepts either the directory or archive form. It
checks complete and exclusive checksums, safe paths and regular-file types,
bounded expansion, canonical JUnit projection, evidence/sidecar consistency, artifact
references, pass/fail semantics, and candidate identity. Candidate-bound
verification compares the sanitized embedded candidate with the supplied
canonical manifest and rechecks candidate wheel hashes when those wheels remain
under the canonical manifest root. A passing run must use the exact resolved
profile and policy recorded from the candidate qualification wheel, bind
platform facts exactly to `probe.json`, match every model source, revision,
registry digest, and file hash, and contain every expected test iteration,
native check, XCLBIN inventory, and packaged XADX case.
Passing evidence must contain every raw native and XADX report; report status,
identity, case order, metrics, and round counts must agree with `tests.json`
and the candidate qualification contract. The native powermode roundtrip is
verified from its raw artifacts: baseline query, set/query confirmation for
`balanced`, `performance`, and `turbo`, restore, and final restore confirmation
must agree with the single `powermode-roundtrip` native test record. Linux runs
may mark only this native record `unsupported`, because current AMDXDNA drivers
require root privileges for power-mode sets; such evidence must still contain a
passing `powermode-before.json` baseline report, and every other expected test
must pass.
The XADX report records exact recipe and executable provenance and one result
per case: dispatch/readback counts, validated bytes, duration, mismatch count,
first mismatch, and structured failures. Runtime-generated buffer contracts,
sizes, and digests keep deterministic materialization auditable without
packaging the generated input files.
Verification does not execute device code.

## Intake index

`rocm-npu qualify intake` preserves verified input archives unchanged under
`runs/<sha256>.evidence.zip` and writes `qualification-index.json` conforming to
`qualification-index-v1.schema.json`. Each sorted run entry records the source
bundle name and digest, status, profile, timestamps, normalized platform facts,
source revision, and candidate digest. The index also records the selected
qualification policy, each matched run digest, and satisfaction. Aggregate
`SHA256SUMS` covers the index and every preserved bundle; the aggregate ZIP is
deterministic and has a separate `<aggregate>.sha256` release-asset checksum.
`rocm-npu qualify verify` validates that external checksum, the aggregate
layout and index, every nested evidence bundle, and their shared candidate.

Candidate release intake rejects ad-hoc or mixed-candidate evidence and fails
when any qualification-wheel policy requirement is absent. Aggregate
verification independently reevaluates that packaged policy against every
nested run. Signing, uploading,
publication, and long-term storage remain manual responsibilities outside this
contract.
