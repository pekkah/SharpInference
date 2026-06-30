<#
.SYNOPSIS
    Downloads GGUF models for SharpInference development.
.DESCRIPTION
    Downloads from HuggingFace to the models/ directory. Skips if already present.
    Supports: smollm2, vibethinker, qwen3-8b, olmoe-1b-7b, llama31-70b, qwen3-coder-30b-a3b, qwen36-35b-a3b,
              qwen36-27b-mtp, qwen36-27b-mtp-q5, qwen36-35b-a3b-mtp, carnice-35b-a3b-mtp,
              ornith-9b, ornith-35b,
              gemma4-12b-qat, gemma4-12b-q4km, gemma4-e4b-qat, gemma4-12b-agentic,
              llama4-scout, z-image-turbo, z-image-turbo-q8, realesrgan-x4
.PARAMETER Model
    Which model to download. Default: downloads all text models (skips large image models).
.EXAMPLE
    .\download-model.ps1                                # All text models
    .\download-model.ps1 -Model smollm2                 # SmolLM2 1.7B (1.1 GB)
    .\download-model.ps1 -Model vibethinker             # VibeThinker-1.5B Q8_0 (1.76 GB) — Qwen2-based math/reasoning (default), issue #282
    .\download-model.ps1 -Model vibethinker-q4          # VibeThinker-1.5B Q4_K_M (1.1 GB) — smaller/faster, slightly lossy
    .\download-model.ps1 -Model qwen3-8b                # Qwen3 8B (4.9 GB)
    .\download-model.ps1 -Model olmoe-1b-7b             # OLMoE 1B-7B Instruct Q4_K_M (~4.4 GB) — small MoE for kernel validation
    .\download-model.ps1 -Model llama31-70b             # Llama 3.1 70B (40.8 GB)
    .\download-model.ps1 -Model qwen3-coder-30b-a3b     # Qwen3-Coder 30B-A3B Q4_K_M (18.6 GB)
    .\download-model.ps1 -Model qwen36-35b-a3b          # Qwen3.6 35B-A3B UD-Q4_K_M (22.1 GB) — recommended general MoE for 12 GB hybrid
    .\download-model.ps1 -Model qwen36-27b-mtp          # Qwen3.6 27B-MTP Q4_K_M (15.9 GB) — dense MTP parity oracle for issue #25
    .\download-model.ps1 -Model qwen36-27b-mtp-q5       # Qwen3.6 27B-MTP Q5_K_M (18.5 GB) — higher-quality variant for the MTP bench row
    .\download-model.ps1 -Model qwen36-35b-a3b-mtp -DestDir E:\models  # Qwen3.6 35B-A3B-MTP UD-Q4_K_M (22.7 GB) — MoE MTP perf target for issue #25
    .\download-model.ps1 -Model carnice-35b-a3b-mtp -DestDir E:\models  # Carnice (Qwen3.6-35B-A3B-MTP, agentic/tool-calling) APEX-MTP I-Compact (17.3 GB)
    .\download-model.ps1 -Model ornith-9b              # Ornith-1.0-9B Q4_K_M (~5.6 GB) — DeepReinforce agentic-coding finetune of Qwen3.5 (dense qwen35 arch)
    .\download-model.ps1 -Model ornith-35b -DestDir E:\models  # Ornith-1.0-35B Q4_K_M (~21 GB) — agentic-coding MoE on the existing qwen35moe path
    .\download-model.ps1 -Model gemma4-12b-qat -DestDir E:\models  # Gemma 4 12B-it QAT q4_0 + vision/audio mmproj (~7.2 GB) — issue #124 PRIMARY (official quantization-aware-trained)
    .\download-model.ps1 -Model gemma4-12b-q4km -DestDir E:\models # Gemma 4 12B-it Q4_K_M (~7.3 GB) — issue #124 fallback / K-quant cross-check
    .\download-model.ps1 -Model gemma4-e4b-qat -DestDir E:\models  # Gemma 4 E4B-it QAT q4_0 (~5.15 GB) — fast small Gemma (~1.6× decode vs Q8_0)
    .\download-model.ps1 -Model gemma4-12b-agentic -DestDir E:\models  # Gemma 4 12B agentic/tool-use finetune (yuxinlu1) Q4_K_M (~7.4 GB) — dense gemma4 arch (reasoning; pass --thinking for the thought chain)
    .\download-model.ps1 -Model llama4-scout            # Llama 4 Scout Q4_K_M (60.9 GB, 2 shards)
    .\download-model.ps1 -Model z-image-turbo           # Z-Image-Turbo Q5_K_M + abliterated encoder (~8.5 GB)
    .\download-model.ps1 -Model z-image-turbo-q8        # Z-Image-Turbo Q8_0 + abliterated encoder Q8_0 (~12 GB)
    .\download-model.ps1 -Model realesrgan-x4           # Real-ESRGAN x4plus upscaler (67 MB)
#>
param(
    [ValidateSet("smollm2", "vibethinker", "vibethinker-q4", "qwen3-8b", "qwen3-0.6b", "olmoe-1b-7b", "llama31-70b", "qwen3-coder-30b-a3b", "qwen36-35b-a3b",
                 "qwen36-27b-mtp", "qwen36-27b-mtp-q5", "qwen36-35b-a3b-mtp", "carnice-35b-a3b-mtp",
                 "ornith-9b", "ornith-35b",
                 "gemma4-12b-qat", "gemma4-12b-q4km", "gemma4-e4b-qat", "gemma4-12b-agentic",
                 "llama4-scout", "z-image-turbo", "z-image-turbo-q8", "realesrgan-x4")]
    [string]$Model,
    [string]$DestDir
)

$Models = @{
    "smollm2" = @{
        Files = @("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
        Size  = "1.1 GB"
        Phase = "1-2"
    }
    # VibeThinker-1.5B (WeiboAI) — a fine-tune of Qwen2.5-Math-1.5B, so it loads as a
    # standard `qwen2` GGUF: 28 layers, hidden 1536, 12 heads / 2 KV heads (GQA), head_dim
    # 128, vocab 151936, tied embeddings, NEOX RoPE. The Qwen2 quirk is bias terms on the
    # Q/K/V projections (auto-probed → HasAttnBias) but NOT on the output projection
    # (HasAttnOutputBias probed separately). ChatML prompt format; emits a long <think>
    # chain-of-thought then a \boxed{} answer (handled by the generic think machinery).
    # Recommended sampling: temp 0.6, top_p 0.95, top_k 0; the chat template supplies the
    # math system prompt so none is needed. Verification target for issue #282.
    #
    # Default is Q8_0 (~1.76 GB) — near-lossless, the right call for a 1.5B math model and
    # an easy full-offload fit on a 12 GB card. The smaller Q4_K_M is available as
    # `vibethinker-q4` (slightly lossy, ~0.7 GB less VRAM, faster decode).
    "vibethinker" = @{
        Files = @("VibeThinker-1.5B.Q8_0.gguf")
        Urls  = @("https://huggingface.co/mradermacher/VibeThinker-1.5B-GGUF/resolve/main/VibeThinker-1.5B.Q8_0.gguf")
        Size  = "1.76 GB"
        SizeGB = 1.76
        Phase = "issue #282 (Qwen2 math/reasoning — near-lossless Q8_0, default)"
    }
    "vibethinker-q4" = @{
        Files = @("VibeThinker-1.5B.Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/mradermacher/VibeThinker-1.5B-GGUF/resolve/main/VibeThinker-1.5B.Q4_K_M.gguf")
        Size  = "1.1 GB"
        SizeGB = 1.1
        Phase = "issue #282 (Qwen2 math/reasoning — smaller Q4_K_M)"
    }
    "qwen3-8b" = @{
        Files = @("Qwen3-8B-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf")
        Size  = "4.9 GB"
        Phase = "2b-3"
    }
    # Qwen3-0.6B Q8_0 — speculative-decoding draft for Qwen3-8B (issue #207). Same
    # tokenizer/vocab (151936) as Qwen3-8B; Q8_0 keeps draft quality high so the
    # acceptance rate (and thus the spec-decode speedup) stays in the alpha 0.7-0.8 band.
    "qwen3-0.6b" = @{
        Files = @("Qwen3-0.6B-Q8_0.gguf")
        Urls  = @("https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf")
        Size  = "~0.6 GB"
        Phase = "spec-decode draft (issue #207)"
    }
    # Smallest MoE model that fits in 12 GB VRAM for full-offload kernel validation.
    # OLMoE arch (allenai) — 7B total params, 1B active, 64 experts × 8 active, softmax routing.
    # ModelGraph maps "olmoe" → NEOX RoPE, GQA, no shared expert. Used to validate
    # CudaForwardPass MoE kernels end-to-end on cards that can't fit Qwen3-Coder 30B (17 GB).
    "olmoe-1b-7b" = @{
        Files = @("OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/bartowski/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf")
        Size  = "~4.4 GB"
        Phase = "5a"
    }
    "llama31-70b" = @{
        Files = @("Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf")
        Size  = "40.8 GB"
        Phase = "4"
    }
    "qwen3-coder-30b-a3b" = @{
        Files = @("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
        Size  = "18.6 GB"
        Phase = "5a"
    }
    # Qwen3.6-35B-A3B (non-MTP) — Gated-DeltaNet + sparse-attention MoE (arch="qwen35moe"):
    # 1-in-4 layers full attention, the rest GDN (delta-rule linear attention with a
    # per-head 128×128 matrix state). Kept as the non-MTP control next to qwen36-35b-a3b-mtp
    # for parity / speedup comparisons. Upstream file is UD-prefixed (unsloth dynamic quant).
    "qwen36-35b-a3b" = @{
        Files = @("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
        Size  = "22.1 GB"
        Phase = "qwen35moe non-MTP baseline (parity control vs qwen36-35b-a3b-mtp)"
    }
    # Qwen3.6-27B-MTP — dense 27B with native Multi-Token Prediction heads (issue #25).
    # Local filename is renamed (MTP- prefix) so it doesn't collide with a future
    # non-MTP 27B download from a different repo. Source repo is unsloth's repack
    # rather than ggml-org's because ggml-org only ships BF16/Q8_0; tensor layout
    # is identical (same llama.cpp converter from the same upstream weights).
    "qwen36-27b-mtp" = @{
        Files = @("Qwen3.6-27B-MTP-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf")
        Size  = "15.9 GB"
        Phase = "MTP / issue #25 (dense parity oracle for qwen3_next_mtp tensor layout)"
    }
    # Same model, Q5_K_M quant — issue #28 wants both Q4_K_M and Q5_K_M bench rows so
    # the README shows the quality/throughput trade-off on the MTP path.
    "qwen36-27b-mtp-q5" = @{
        Files = @("Qwen3.6-27B-MTP-Q5_K_M.gguf")
        Urls  = @("https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-Q5_K_M.gguf")
        Size  = "18.5 GB"
        Phase = "MTP / issue #25 (Q5_K_M variant for the MTP bench row)"
    }
    # Qwen3.6-35B-A3B-MTP — same qwen35moe architecture as qwen36-35b-a3b but with
    # MTP heads bolted on. Runs ~23 t/s on the CUDA hybrid path; recommended target
    # for issue #25's ≥1.3× decode speedup criterion. Local filename keeps the MTP-
    # prefix; the unsloth MTP repo only ships UD-prefixed Q4 (no plain Q4_K_M exists).
    "qwen36-35b-a3b-mtp" = @{
        Files = @("Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
        Size  = "22.7 GB"
        Phase = "MTP / issue #25 (MoE decode-throughput target)"
    }
    # Carnice — Hermes-style agentic/tool-calling finetune of Qwen3.6-35B-A3B-MTP
    # (qwen35moe arch, MTP heads preserved). mudler's APEX quant tier; "I-Compact" is the
    # imatrix-calibrated Q4-equivalent (17.3 GB) — smaller than UD-Q4_K_M with comparable
    # quality. Personal-assistant / orchestrator role: routes work to Claude Code via the
    # OpenAI tool adapter (6fa096d) and handles research with local tool loops.
    "carnice-35b-a3b-mtp" = @{
        Files = @("Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf")
        Urls  = @("https://huggingface.co/mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-GGUF/resolve/main/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf")
        Size  = "17.3 GB"
        Phase = "assistant (agentic/tool-calling orchestrator on the qwen35moe MTP path)"
    }
    # ── Ornith-1.0 (DeepReinforce, MIT) — agentic-coding finetunes ────────────
    # Ornith-1.0 is NOT a new architecture: it's DeepReinforce's "self-scaffolding"
    # RL post-train of existing Qwen3.5 / Gemma 4 bases. Self-scaffolding (the model
    # learns to emit its own task harness alongside solution rollouts) is a TRAINING
    # technique — at inference these are ordinary autoregressive transformers and need
    # no special runtime support. HF arches: 9B = `qwen3_5` (dense), 35B/397B =
    # `qwen3_5_moe`. After llama.cpp conversion the GGUF arch strings are `qwen35`
    # (dense) and `qwen35moe` (MoE), both already dispatched by ModelGraph — so the
    # MoE variants ride the existing Gated-DeltaNet + sparse-attention MoE path and
    # tool-calling via the qwen35moe QwenToolCallAdapter. (The models are tagged
    # image-text-to-text; the Qwen3.5 vision projector is not yet implemented, so the
    # text GGUF path here is text-only — fine for the agentic-coding use case.)
    #
    # Sources are bartowski's GGUF republishes (deterministic `<ns>_<model>-<Quant>.gguf`
    # naming). Ornith-1.0-9B is the edge target (43.1 Terminal-Bench 2.1, 69.4 SWE-Bench
    # Verified). If the 9B GGUF carries GDN tensors the hybrid-SSM probe activates
    # automatically; otherwise it loads as a plain dense qwen35 transformer.
    "ornith-9b" = @{
        Files = @("deepreinforce-ai_Ornith-1.0-9B-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/bartowski/deepreinforce-ai_Ornith-1.0-9B-GGUF/resolve/main/deepreinforce-ai_Ornith-1.0-9B-Q4_K_M.gguf")
        Size  = "~5.6 GB"
        Phase = "agentic coding (Qwen3.5-based dense, qwen35 arch)"
    }
    # Ornith-1.0-35B — qwen35moe (Qwen3.5 35B-A3B base): runs on the existing hybrid
    # Gated-DeltaNet + sparse-attention MoE path (same arch as qwen36-35b-a3b), incl.
    # --cpu-moe expert offload. MTP-augmented community quants exist separately and
    # would ride the MtpDecoder path.
    "ornith-35b" = @{
        Files = @("deepreinforce-ai_Ornith-1.0-35B-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/bartowski/deepreinforce-ai_Ornith-1.0-35B-GGUF/resolve/main/deepreinforce-ai_Ornith-1.0-35B-Q4_K_M.gguf")
        Size  = "~21 GB"
        Phase = "agentic coding (qwen35moe MoE path)"
    }
    # ── Gemma 4 12B (dense gemma4_unified) — issue #124 ───────────────────────
    # Google's official quantization-aware-trained (QAT) 4-bit weights. Stored as
    # q4_0 (NOT a K-quant), so this exercises the q4_0 dequant path (gap G0). Best
    # quality-per-byte at 4-bit; fits full-GPU offload on 12 GB VRAM. This is the
    # PRIMARY iteration-1 target.
    #
    # Gemma 4 12B is an encoder-free *unified* multimodal model (text + image +
    # audio + video): it projects raw image patches / audio waveforms straight into
    # the LLM embedding space via lightweight linear layers. In the GGUF ecosystem
    # those projection tensors still ship as a small companion mmproj (~175 MB), so
    # we pull it alongside the text GGUF. The mmproj is only consumed once sharpi
    # grows a multimodal input path (vision plan / issue #250) — the text GGUF runs
    # fine on its own without it.
    "gemma4-12b-qat" = @{
        Files = @("gemma-4-12b-it-qat-q4_0.gguf", "mmproj-gemma-4-12b-it-qat-q4_0.gguf")
        Urls  = @(
            "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/gemma-4-12b-it-qat-q4_0.gguf",
            "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/main/mmproj-gemma-4-12b-it-qat-q4_0.gguf"
        )
        Size  = "~7.2 GB (text 7.0 GB + vision/audio mmproj 0.17 GB)"
        SizeGB = 7.2
        Phase = "issue #124 (PRIMARY — QAT q4_0, dense no-PLE path) + mmproj (multimodal projector)"
    }
    # Gemma 4 E4B-it — Google's official QAT q4_0 weights for the small (effective-4B)
    # Gemma 4. ~5.15 GB vs the 8.19 GB Q8_0: ~1.6× fewer bytes/token → ~1.6× faster
    # decode at near-identical quality (QAT). Same dense gemma4 arch (PLE + 5:1 SWA).
    # Filename is upstream's `gemma-4-E4B_q4_0-it.gguf` (note the underscore — not the
    # `-it-qat-q4_0` pattern of the 12B).
    #
    # The companion mmproj is pulled alongside, but NOTE the E4B vision path is NOT the
    # 12B's. The E4B mmproj is ~992 MB (vs the 12B's tiny ~175 MB linear projector) —
    # it carries a full ENCODER. Confirmed from the header: vision projector_type=`gemma4v`
    # (a 16-block transformer ViT: 768-dim, 12 heads, QK-norm, GeGLU, conv patch-embed) plus
    # an audio encoder projector_type=`gemma4a`. The 12B is encoder-free `gemma4uv` instead.
    # So #250's image path does NOT cover E4B; supporting it needs the ViT (and audio)
    # encoder forward pass (tracked by the E4B vision plan, issue #126). The text GGUF runs
    # fine on its own without the mmproj.
    "gemma4-e4b-qat" = @{
        Files = @("gemma-4-E4B_q4_0-it.gguf", "gemma-4-E4B-it-mmproj.gguf")
        Urls  = @(
            "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf",
            "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B-it-mmproj.gguf"
        )
        Size  = "~6.15 GB (text 5.15 GB + vision mmproj 0.99 GB)"
        SizeGB = 6.15
        Phase = "Gemma 4 E4B QAT q4_0 (fast small Gemma — ~1.6× decode vs Q8_0) + mmproj (conv vision encoder; NOT the 12B gemma4uv path — issue #126)"
    }
    # Community K-quant of the (non-QAT) instruct weights — the Q4_K_M fallback /
    # cross-check next to the QAT q4_0 primary. Exercises the K-quant path so both
    # quant formats are covered for the dense 12B bring-up.
    "gemma4-12b-q4km" = @{
        Files = @("gemma-4-12b-it-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q4_K_M.gguf")
        Size  = "~7.3 GB"
        SizeGB = 7.3
        Phase = "issue #124 (fallback — Q4_K_M K-quant cross-check)"
    }
    # ── Gemma 4 12B agentic finetune (community) ───────────────────────────────
    # yuxinlu1's agentic/tool-use + coding finetune of google/gemma-4-12B-it,
    # distributed as a standard dense `gemma4` (gemma4_unified) GGUF — same arch
    # as gemma4-12b-q4km, so it runs on the existing dense Gemma 4 text path.
    # Tuned on multi-step tool-use trajectories (tau2) and verified coding CoT.
    # Text-only repo (no companion mmproj). Filenames verified against the repo:
    # the GGUFs are `gemma4-v2-<quant>.gguf` (NOT the author's coder-v1
    # `gemma4-<variant>-Q4_K_M.gguf` convention). Q4_K_M (~7.4 GB) is the 12 GB
    # full-offload pick; the repo also has Q3_K_M (~6.1 GB), Q6_K (~9.8 GB) and
    # Q8_0 (~12.7 GB).
    #   https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF/tree/main
    #
    # Unlike stock Gemma 4 instruct (not reasoning-trained), THIS reasoning finetune
    # thinks in Gemma's native thought channel. It answers fine in the default mode;
    # to get the reasoning chain pass `--thinking` (Gemma 4 defaults thinking off for
    # the stock models). Recommended sampling: --temp 1.0 --top-k 64 --top-p 0.95.
    "gemma4-12b-agentic" = @{
        Files = @("gemma4-v2-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF/resolve/main/gemma4-v2-Q4_K_M.gguf")
        Size  = "~7.4 GB"
        SizeGB = 7.4
        Phase = "Gemma 4 12B agentic/tool-use finetune (yuxinlu1) — dense gemma4 text path"
    }
    "llama4-scout" = @{
        Files = @(
            "Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf",
            "Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf"
        )
        Urls  = @(
            "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf",
            "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf"
        )
        Size  = "60.9 GB (2 shards: 46.4 GB + 14.5 GB)"
        Phase = "5b"
    }
    # ── Image generation ──────────────────────────────────────────────────────
    # Z-Image-Turbo Q5_K_M (recommended balance of quality and size)
    #   DiT:      jayn7/Z-Image-Turbo-GGUF        (5.52 GB)
    #   Encoder:  BennyDaBall abliterated Qwen3-4B (2.89 GB, uncensored)
    #   VAE:      Tongyi-MAI/Z-Image-Turbo vae/    (0.33 GB)
    #   Tokenizer: Tongyi-MAI/Z-Image-Turbo tokenizer/ (11 MB)
    "z-image-turbo" = @{
        Files = @(
            "z_image_turbo-Q5_K_M.gguf",
            "Z-Image-AbliteratedV1.Q5_K_M.gguf",
            "z-image-turbo\vae\diffusion_pytorch_model.safetensors",
            "z-image-turbo\tokenizer\tokenizer.json"
        )
        Urls  = @(
            "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q5_K_M.gguf",
            "https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1/resolve/main/Z-Image-AbliteratedV1.Q5_K_M.gguf",
            "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/vae/diffusion_pytorch_model.safetensors",
            "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/tokenizer/tokenizer.json"
        )
        Size  = "~8.5 GB (DiT 5.52 GB + encoder 2.89 GB + VAE 0.33 GB + tokenizer)"
        Phase = "image"
        IsImage = $true
    }
    # Z-Image-Turbo Q8_0 (maximum quality, needs ~16 GB VRAM)
    #   DiT:      jayn7/Z-Image-Turbo-GGUF Q8_0   (7.22 GB)
    #   Encoder:  BennyDaBall abliterated Q8_0     (4.28 GB)
    #   VAE + Tokenizer: same as above
    "z-image-turbo-q8" = @{
        Files = @(
            "z_image_turbo-Q8_0.gguf",
            "Z-Image-AbliteratedV1.Q8_0.gguf",
            "z-image-turbo\vae\diffusion_pytorch_model.safetensors",
            "z-image-turbo\tokenizer\tokenizer.json"
        )
        Urls  = @(
            "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q8_0.gguf",
            "https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1/resolve/main/Z-Image-AbliteratedV1.Q8_0.gguf",
            "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/vae/diffusion_pytorch_model.safetensors",
            "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/tokenizer/tokenizer.json"
        )
        Size  = "~12 GB (DiT 7.22 GB + encoder 4.28 GB + VAE 0.33 GB + tokenizer)"
        Phase = "image"
        IsImage = $true
    }
    # ── Image upscaler ────────────────────────────────────────────────────────
    # Real-ESRGAN x4plus — RRDBNet ×4 upscaler (23 RRDB blocks, 64 feat channels)
    #   Trained by xinntao on synthetic degradations; works well on photos and
    #   generated images. Source: Comfy-Org/Real-ESRGAN_repackaged (BSD-3-Clause)
    #   Use with: sharpi image ... --upscaler models/RealESRGAN_x4plus.safetensors
    "realesrgan-x4" = @{
        Files = @("RealESRGAN_x4plus.safetensors")
        Urls  = @("https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged/resolve/main/RealESRGAN_x4plus.safetensors")
        Size  = "67 MB"
        Phase = "upscaler"
        IsUpscaler = $true
    }
}

$ModelDir = if ($DestDir) { $DestDir } else { Join-Path $PSScriptRoot "..\models" }
if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
}

function Download-File {
    param([string]$url, [string]$path)
    # Ensure parent directory exists (needed for z-image-turbo\vae\ etc.)
    $parent = Split-Path $path -Parent
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L -o $path -C - --progress-bar $url
        if ($LASTEXITCODE -ne 0) { throw "curl exited with code $LASTEXITCODE" }
    }
    else {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $url -OutFile $path -UseBasicParsing
        $ProgressPreference = 'Continue'
    }
}

function Download-Model {
    param([string]$key)
    $info = $Models[$key]

    # Check free disk space. Split-Path -Qualifier returns empty on non-Windows
    # or UNC paths, so fall back and tolerate Get-PSDrive failing.
    $drive     = Split-Path -Qualifier (Resolve-Path $ModelDir)
    $driveName = if ($drive) { $drive.TrimEnd(':') } else { '/' }
    $freeGB    = $null
    try {
        $freeGB = [math]::Round((Get-PSDrive $driveName).Free / 1GB, 1)
        Write-Host "[$key] Free disk on $driveName : $freeGB GB"
    }
    catch {
        Write-Host "[$key] Free disk space could not be determined for $driveName"
    }

    $allPresent = $true
    foreach ($file in $info.Files) {
        if (-not (Test-Path (Join-Path $ModelDir $file))) { $allPresent = $false; break }
    }

    # Guard: refuse to start a download that won't fit (10% headroom for temp/partial files).
    # Only enforced when the model declares a numeric SizeGB. Skipped if files already present.
    # Subtract bytes already on disk so a small top-up (e.g. adding the mmproj next to an
    # existing main GGUF) isn't blocked by the full bundle's SizeGB.
    if (-not $allPresent -and $info.ContainsKey('SizeGB') -and $null -ne $freeGB) {
        $presentGB = (($info.Files |
            Where-Object { Test-Path (Join-Path $ModelDir $_) } |
            ForEach-Object { (Get-Item (Join-Path $ModelDir $_)).Length } |
            Measure-Object -Sum).Sum) / 1GB
        $neededGB = [math]::Round([math]::Max($info.SizeGB - $presentGB, 0) * 1.1, 1)
        if ($neededGB -gt 0 -and $freeGB -lt $neededGB) {
            Write-Error "[$key] Not enough disk space on $driveName : need ~$neededGB GB (incl. headroom), have $freeGB GB. Use -DestDir to pick a drive with more room (e.g. -DestDir E:\models)."
            return
        }
    }

    if ($allPresent) {
        $totalMB = ($info.Files | ForEach-Object { (Get-Item (Join-Path $ModelDir $_)).Length } | Measure-Object -Sum).Sum / 1MB
        Write-Host "[$key] Already complete: $($info.Files -join ', ') ($([math]::Round($totalMB, 1)) MB total) - Phase $($info.Phase)"
        return
    }

    Write-Host "[$key] Downloading $($info.Size) - Phase $($info.Phase)"
    Write-Host "  Files: $($info.Files -join ', ')"
    Write-Host ""

    # Component labels for multi-file bundles
    $labels = @{
        "mmproj-gemma-4-12b-it-qat-q4_0.gguf"               = "Multimodal projector (encoder-free vision/audio)"
        "gemma-4-E4B-it-mmproj.gguf"                         = "Vision (gemma4v ViT) + audio (gemma4a) encoders; NOT the 12B gemma4uv"
        "z_image_turbo-Q5_K_M.gguf"                         = "DiT (image model)"
        "z_image_turbo-Q8_0.gguf"                           = "DiT (image model)"
        "Z-Image-AbliteratedV1.Q5_K_M.gguf"                 = "Text encoder (abliterated Qwen3-4B)"
        "Z-Image-AbliteratedV1.Q8_0.gguf"                   = "Text encoder (abliterated Qwen3-4B)"
        "z-image-turbo\vae\diffusion_pytorch_model.safetensors" = "VAE decoder"
        "z-image-turbo\tokenizer\tokenizer.json"             = "Tokenizer"
    }

    for ($i = 0; $i -lt $info.Files.Count; $i++) {
        $file  = $info.Files[$i]
        $url   = $info.Urls[$i]
        $path  = Join-Path $ModelDir $file
        $label = if ($labels.ContainsKey($file)) { " ($($labels[$file]))" } else { "" }

        if (Test-Path $path) {
            $sizeMB = [math]::Round((Get-Item $path).Length / 1MB, 1)
            Write-Host "  File $($i+1)/$($info.Files.Count)$label`: already present ($sizeMB MB), skipping"
            continue
        }

        Write-Host "  File $($i+1)/$($info.Files.Count)$label`: $file"
        Write-Host "  From: $url"
        Write-Host ""

        try {
            Download-File -url $url -path $path
            $sizeMB = [math]::Round((Get-Item $path).Length / 1MB, 1)
            Write-Host "  File $($i+1) complete: $sizeMB MB"
        }
        catch {
            Write-Error "[$key] Download failed for file $($i+1): $_"
            if (Test-Path $path) { Remove-Item $path }
            return
        }
    }

    $totalMB = ($info.Files | ForEach-Object { (Get-Item (Join-Path $ModelDir $_)).Length } | Measure-Object -Sum).Sum / 1MB
    Write-Host "[$key] All files complete: $([math]::Round($totalMB, 1)) MB total"

    # Print ready-to-use command for image models
    if ($info.IsImage) {
        Write-Host ""
        Write-Host "[$key] Ready to generate images:"
        if ($key -like "z-image-turbo*") {
            $ditFile = $info.Files[0]
            $encFile = $info.Files[1]
            Write-Host "  dotnet run --project src/SharpInference.Cli -c Release -- image \"
            Write-Host "    -m models/$ditFile \"
            Write-Host "    --vae models/z-image-turbo/vae \"
            Write-Host "    --qwen-encoder models/$encFile \"
            Write-Host "    --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \"
            Write-Host "    -p `"your prompt here`" -W 1024 -H 1024 --steps 9 -o output.png"
        }
    }

    if ($info.IsUpscaler) {
        Write-Host ""
        Write-Host "[$key] Ready to use as upscaler (append --upscaler to any image command):"
        Write-Host "  dotnet run --project src/SharpInference.Cli -c Release -- image \"
        Write-Host "    -m models/z_image_turbo-Q5_K_M.gguf \"
        Write-Host "    --vae models/z-image-turbo/vae \"
        Write-Host "    --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \"
        Write-Host "    --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \"
        Write-Host "    --upscaler models/RealESRGAN_x4plus.safetensors \"
        Write-Host "    -p `"your prompt here`" -W 512 -H 512 -o output_4x.png"
        Write-Host "  (output will be 2048x2048 for -W 512 -H 512 with the x4 upscaler)"
    }
}

if ($Model) {
    Download-Model -key $Model
}
else {
    # Default: download text models only (image and upscaler models are large/optional)
    $textModels = $Models.Keys | Where-Object { -not $Models[$_].IsImage -and -not $Models[$_].IsUpscaler } | Sort-Object
    foreach ($key in $textModels) {
        Download-Model -key $key
    }
    Write-Host ""
    Write-Host "Image models not downloaded by default (large). Run explicitly:"
    Write-Host "  .\download-model.ps1 -Model z-image-turbo      # Q5_K_M (~8.5 GB)"
    Write-Host "  .\download-model.ps1 -Model z-image-turbo-q8   # Q8_0   (~12 GB)"
    Write-Host ""
    Write-Host "Upscaler model (optional, enhances generated images 4×):"
    Write-Host "  .\download-model.ps1 -Model realesrgan-x4      # Real-ESRGAN x4plus (67 MB)"
}
