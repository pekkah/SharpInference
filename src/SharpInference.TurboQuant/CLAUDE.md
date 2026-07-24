# SharpInference.TurboQuant — KV cache compression

Two codecs behind `TurboQuantKvCache`:

- **KVarN** — Hadamard + dual-axis Sinkhorn variance normalization + asymmetric
  RTN, 4-bit K / 2-bit V, 128-token tiles (issue #180). Runs on CPU (AVX2 fused
  read kernels) and the CUDA decode path (CUDA-graph decode + chunked prefill).
- **Lloyd-Max codebooks** — 3-4 bit; severely degrades quality on QK-norm models
  such as Qwen3 (issue #432). Remains the fallback for Vulkan / partial-offload /
  MoE-on-GPU / SnapKV. Codebook data lives in `codebooks/`.

`--tq-mode` defaults to `auto`: KVarN where supported, else Lloyd-Max fallback
with a quality warning (#436). The support matrix is centralized (#437) — extend
it rather than scattering capability checks.

Accuracy changes here must go through the perplexity gate
(`perplexity -m model.gguf -f corpus.txt -c 2048 --tq`, corpus helpers in
`scripts/kvarn-gate/`) — see the `parity-check` skill. Encode/decode parity tests
live in `tests/SharpInference.Tests.TurboQuant`.
