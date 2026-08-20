# ASR fixture provenance

## `asr_sample.wav`

- **Source:** FastFlowLM (https://github.com/FastFlowLM/FastFlowLM), test
  fixture `src/tb_files/Recording.wav`. That file is an untracked developer
  fixture in the upstream tree and is *not* shipped by `flm pull` or in any
  FastFlowLM wheel, so it is vendored here to make offline ASR qualification
  self-contained.
- **Byte identity:** SHA-256
  `c0f16ef6b0a9053d9abccb603d1aa1445c62b2b35bbbb4eaa3996422c68df97a`
  (byte-identical to the upstream `Recording.wav` at time of vendoring).
- **Format:** WAV / PCM signed 16-bit, 2 channels, 48 kHz, ~3.24 s.
  If this clip is ever regenerated, prefer 16 kHz mono to shrink the wheel and
  match Whisper's native input rate.
- **License:** MIT, inherited from the FastFlowLM project (see
  `tools/rocm-npu-qualification/licenses/` and FastFlowLM's `LICENSE`).

## `asr_sample.reference.txt`

The accepted transcript for `asr_sample.wav` (`"This is something."`). It is the
pinned reference phrase the ASR serve-probe requires the transcript to contain
and the ground truth for the word-error-rate metric. Override at runtime with
`ROCM_NPU_ASR_SAMPLE` / `ROCM_NPU_ASR_REFERENCE` when qualifying a different
clip (override both together, since the pin is clip-specific).
