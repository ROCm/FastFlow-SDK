# Qualification evidence contract v1

A qualification run writes a portable directory containing:

- `evidence.json`: overall subject, platform, model, and test results
- `candidate-manifest.json`: a sanitized identity projection of the supplied
  release candidate, or an explicit ad-hoc placeholder
- `core-version.json`, `probe.json`, `doctor.json`, and `self-test.json`:
  native CLI reports
- `xadx-report.json`: raw XADX architecture, case, round, and result report
- `profile.json`: the exact versioned profile
- `models.json`: model source, revision, pull state, and content hashes
- `junit.xml`: CI-compatible passed and failed test cases
- `logs/pull/` and `logs/tests/`: redacted subprocess transcripts
- `SHA256SUMS`: every other evidence file in deterministic path order

`evidence.json` conforms to
`tools/rocm-npu-qualification/python/rocm_npu_qualification/schemas/evidence-v1.schema.json`.
Status is
written as `running` before device work begins and updated as results arrive,
so interrupted and failed runs retain useful partial evidence. Both successes
and failures remain in the final bundle.

The subject records exact versions for `rocm-npu-runtime`,
`rocm-npu-fastflowlm`, and `rocm-npu-qualification`, the native core source
revision, and the candidate manifest hash. Model identities always include a
source, revision label, approved registry digest, and SHA-256 inventory of
observed model-cache files. Candidate-backed runs verify installed package
manifests and their complete declared file inventories before device work.
The candidate qualification contract also binds the declarative XADX case
contract, serializer revision, transaction hashes, source-XCLBIN hashes, and
serialized XADX hashes, plus lifecycle, resident, and readback round counts.
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
and the candidate qualification contract.
The XADX report records each runtime-generated buffer contract, size, and
digest so deterministic materialization remains auditable without packaging
the generated input files.
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
