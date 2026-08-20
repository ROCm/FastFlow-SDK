# ASR fixture provenance

## `asr_sample.wav` (not vendored -- copied at wheel-build time)

- **Source:** FastFlowLM (https://github.com/FastFlowLM/FastFlowLM), test
  fixture `src/tb_files/Recording.wav`. It lives in the FastFlowLM *source*
  repository (a testbench input), but is *not* included in any installed
  FastFlowLM package and is *not* fetched by `flm pull` (which only downloads
  model weights). FLM's ASR interface is "bring your own audio file."
- **How it enters the wheel:** it is **not** committed to this repository.
  `build_tools/rocm_npu_qualification_wheel.py` (`_stage_asr_sample`) copies it
  from the pinned FastFlowLM submodule (`manifests/versions.toml`) into
  `rocm_npu_qualification/assets/asr_sample.wav` during wheel staging, verifying
  the byte hash below. The FastFlowLM HRX build jobs always have the submodule
  checked out, so released wheels always carry the clip and stay self-contained
  for offline runs. Builds without the submodule (e.g. presubmit) simply omit
  the clip, and the ASR probe then fails closed.
- **Byte identity (pinned):** SHA-256
  `c0f16ef6b0a9053d9abccb603d1aa1445c62b2b35bbbb4eaa3996422c68df97a`. This is
  asserted at build time (`ASR_SAMPLE_SHA256`); an upstream change fails the
  build so the paired reference transcript cannot drift silently.
- **Format:** WAV / PCM signed 16-bit, 2 channels, 48 kHz, ~3.24 s.
  If this clip is ever regenerated upstream, prefer 16 kHz mono to shrink the
  wheel and match Whisper's native input rate (update the pin + reference).
- **License:** MIT, inherited from the FastFlowLM project (see
  `tools/rocm-npu-qualification/licenses/` and FastFlowLM's `LICENSE`).

## `asr_sample.reference.txt`

The accepted transcript for `asr_sample.wav` (`"This is something."`). It is the
pinned reference phrase the ASR serve-probe requires the transcript to contain
and the ground truth for the word-error-rate metric. Override at runtime with
`ROCM_NPU_ASR_SAMPLE` / `ROCM_NPU_ASR_REFERENCE` when qualifying a different
clip (override both together, since the pin is clip-specific).
