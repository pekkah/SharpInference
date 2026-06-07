using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #156: the all-GPU batched-trunk prefill (originally Gemma-4-only, #136) was
/// opened to any dense model the batched kernels cover. This oracle verifies that on a
/// dense <b>Qwen3-8B Q4_K</b> — which has none of Gemma's PLE / shared-KV / SWA /
/// sandwich-norm machinery, uses SwiGLU (Silu) instead of GEGLU, and the standard
/// 1/sqrt(head_dim) attention scale — the batched prefill engages and agrees with the
/// per-token <see cref="CudaForwardPass.Forward"/> loop.
///
/// Q4_K trunk matmuls have a bit-exact matvec GEMM-N batched path plus two compute-bound
/// prefill variants (#156 Item C: C1 dequant→fp16→cuBLAS GEMM, C2 int8 MMQ); the FlashOff
/// matvec case is the bit-exact oracle for the batched matvec/norm/rope/SwiGLU primitives,
/// while the default case adds flash attention and is argmax-stable (online softmax
/// reassociates the score sum). The C1/C2 paths get their own argmax-stable oracles below.
///
/// Single 5 GB model instance toggled via <see cref="CudaForwardPass.BatchedPrefillEnabled"/>
/// with a <c>ResetCache</c> between runs. Silent-skips when CUDA or the GGUF is absent.
/// </summary>
public sealed class Qwen3CudaBatchedPrefillTests
{
    private const string ModelFile = "Qwen3-8B-Q4_K_M.gguf";

    // Mixed-vocab prompt; >1 token so the batched path (N≥2) engages. Token 151643 is
    // Qwen's BOS-ish/special region; the rest are ordinary ids spread across the vocab.
    private static readonly int[] Tokens =
        { 9707, 11, 1879, 0, 358, 1079, 264, 4108, 1614, 13, 220, 17, 18, 19 };

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        string[] absolute = { $@"E:\models\{ModelFile}", $@"C:\p\sharpi\models\{ModelFile}" };
        foreach (var p in absolute)
            if (File.Exists(p)) return p;
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static HashSet<int> TopKSet(ReadOnlySpan<float> logits, int k)
    {
        var idx = new int[logits.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        var arr = logits.ToArray();
        Array.Sort(idx, (a, b) => arr[b].CompareTo(arr[a]));
        var set = new HashSet<int>();
        for (int i = 0; i < k && i < idx.Length; i++) set.Add(idx[i]);
        return set;
    }

    /// <summary>
    /// The headline #156 check: a dense, non-Gemma Q4_K model must actually take the
    /// batched-trunk prefill (gate opened) and produce argmax-stable logits vs the
    /// per-token loop under the shipped default config (flash TC on).
    /// </summary>
    [Fact]
    public void Qwen3_8B_BatchedPrefill_DefaultMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        // Sanity: this really is a dense, non-Gemma SwiGLU model.
        Assert.Null(hp.LayerHeadDim);
        Assert.False(hp.HasPerLayerTokenEmbd);
        Assert.False(hp.IsMoE);
        Assert.Equal(FfnActivation.Silu, hp.FfnActivation);

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        // Shipped defaults (flash TC + Q4_K matvec GEMM-N).
        fwd.BatchedPrefillEnabled = true;
        var batched = fwd.Prefill(Tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched,
            "Batched-trunk prefill did not engage for dense Q4_K — check IsBatchedPrefillSupported (#156).");

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(Tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);
        Assert.Equal(Argmax(sequential), Argmax(batched));

        float maxAbs = 0f;
        for (int i = 0; i < sequential.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        Assert.True(maxAbs < 1.0f,
            $"Default batched vs sequential logits diverged beyond fp tolerance: maxAbs={maxAbs}.");

        var seqTop = TopKSet(sequential, 5);
        var batTop = TopKSet(batched, 5);
        int overlap = 0;
        foreach (var t in batTop) if (seqTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"Default batched top-5 overlaps the per-token reference in only {overlap}/5 slots.");
    }

    /// <summary>
    /// Issue #157 regression guard: the CUDA dense per-token <see cref="CudaForwardPass.Forward"/>
    /// must apply per-head QK-norm <b>before</b> RoPE for weighted-QK-norm models (Qwen3), matching
    /// the HF Qwen3 reference, llama.cpp <c>build_qwen3</c>, and the trusted CPU
    /// <see cref="Engine.ForwardPass"/>. RoPE does not commute with per-channel-weighted RMSNorm (NEOX RoPE
    /// mixes channels i and i+d/2, which carry different learned q_norm/k_norm weights), so a flipped
    /// order silently degrades output.
    /// <para>
    /// This is a <b>cross-backend</b> oracle on purpose: the CUDA-vs-CUDA batched-prefill oracles
    /// can't catch the bug because the batched path was deliberately built to match the per-token
    /// loop — both were equally wrong. Only the CPU path (always norm→RoPE) is an independent
    /// reference. Pre-fix the CUDA dense path was RoPE→norm and diverged ~9 logits from CPU
    /// (#156); post-fix the two agree at argmax + top-5 (cross-backend Q4_K precision still differs).
    /// </para>
    /// </summary>
    [Fact]
    public void Qwen3_8B_CudaForward_MatchesCpu_QkNormBeforeRope()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        // Precondition: this is exactly the buggy branch — weighted (non-L2) QK-norm + NEOX RoPE.
        Assert.True(hp.HasQkNorm);
        Assert.False(hp.UseL2QkNorm);
        Assert.True(hp.IsNeoxRope);

        // CPU reference (norm→RoPE, matches llama.cpp build_qwen3).
        using var cpu = new CpuBackend();
        using var cpuFwd = new Engine.ForwardPass(model, cpu, hp, maxContextLength: 512);
        var cpuLogits = cpuFwd.Prefill(Tokens).ToArray();

        // CUDA dense per-token path (Forward loop; batched prefill off so this is the
        // RunDeviceRegion ordering under test, not the batched trunk).
        using var cudaFwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512)
        {
            BatchedPrefillEnabled = false,
        };
        var cudaLogits = cudaFwd.Prefill(Tokens).ToArray();
        Assert.False(cudaFwd.LastPrefillWasBatched);

        Assert.Equal(cpuLogits.Length, cudaLogits.Length);

        float maxAbs = 0f;
        for (int i = 0; i < cpuLogits.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(cpuLogits[i] - cudaLogits[i]));

        // Argmax must agree across backends; pre-fix the order mismatch flips it / scrambles top-5.
        Assert.Equal(Argmax(cpuLogits), Argmax(cudaLogits));

        var cpuTop = TopKSet(cpuLogits, 5);
        var cudaTop = TopKSet(cudaLogits, 5);
        int overlap = 0;
        foreach (var t in cudaTop) if (cpuTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"CUDA dense Forward top-5 overlaps the CPU reference in only {overlap}/5 slots " +
            $"(maxAbs={maxAbs}). A RoPE/QK-norm ordering regression (#157) is the likely cause.");

        // With matching order, cross-backend Q4_K divergence is small (≪ the ~9-logit gap a flipped
        // order produces). 4.0 cleanly separates "same order, different backend" from "wrong order".
        Assert.True(maxAbs < 4.0f,
            $"CUDA dense Forward diverged from CPU by maxAbs={maxAbs} — far beyond cross-backend " +
            "Q4_K precision; suggests a RoPE/QK-norm ordering regression (#157).");
    }

    /// <summary>
    /// Issue #162: prompts longer than the non-flash 4096 cap must still take the fast
    /// batched-trunk path (chunked into <c>PrefillBatchChunk</c> windows, flash streaming
    /// the prior KV) and stay argmax-stable vs the bit-exact per-token loop — instead of
    /// silently dropping to the ~8× slower memory-bound per-token prefill.
    /// <para>
    /// N=5040 exercises a full + partial window (4096 + 944); N=8192 exercises the
    /// exact-multiple boundary (two full 4096 windows, the final chunk's len == chunk size)
    /// to catch the classic off-by-one in the chunk loop.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData(5040, 6144)]
    [InlineData(8192, 9216)]
    public void Qwen3_8B_ChunkedBatchedPrefill_Over4096_MatchesSequential(int promptLen, int ctx)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);

        // > 4096 tokens → forces the chunked branch. Deterministic spread across the vocab
        // via a small LCG; all ids well within Qwen3's 151936 vocab.
        var longTokens = new int[promptLen];
        uint s = 0x9E3779B9u;
        for (int i = 0; i < longTokens.Length; i++)
        {
            s = s * 1664525u + 1013904223u;
            longTokens[i] = (int)(s % 150000u) + 1;
        }

        // Disable SnapKV (a >budget prompt would otherwise route to the per-token SnapKV
        // eviction path before the batched gate). Construct under the override, then restore.
        var prevSnap = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        CudaForwardPass fwd;
        try { fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: ctx); }
        finally { Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevSnap); }
        using var _fwd = fwd;

        // Shipped defaults (flash TC on) → chunked batched path.
        fwd.BatchedPrefillEnabled = true;
        var batched = fwd.Prefill(longTokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched,
            "Chunked batched prefill did not engage for a >4096-token prompt (#162).");

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(longTokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);
        Assert.Equal(Argmax(sequential), Argmax(batched));

        var seqTop = TopKSet(sequential, 5);
        var batTop = TopKSet(batched, 5);
        int overlap = 0;
        foreach (var t in batTop) if (seqTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"Chunked batched top-5 overlaps the per-token reference in only {overlap}/5 slots.");
    }

    /// <summary>
    /// Flash off + GEMM off: Q4_K trunk runs the batched matvec GEMM-N, which is built to
    /// be bit-identical to N per-token Q4_K dp4a matvecs. Verifies the batched
    /// matvec/norm/rope/SwiGLU primitives in isolation from the argmax-stable attention.
    /// <c>PrefillGemmEnabled = false</c> is required because #156 Item C made the GEMM
    /// (dequant→fp16→cuBLAS, argmax-stable but not bit-exact) the Q4_K default — that path
    /// is covered by <see cref="Qwen3_8B_BatchedPrefill_Q4KGemm_ArgmaxStable"/>.
    /// </summary>
    [Fact]
    public void Qwen3_8B_BatchedPrefill_FlashOff_MatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        fwd.BatchedPrefillEnabled = true;
        fwd.PrefillFlashAttnEnabled = false;
        fwd.PrefillFlashTcEnabled = false;
        fwd.PrefillGemmEnabled = false; // pin the bit-exact Q4_K matvec GEMM-N, not the #156-C fp16 GEMM
        var batched = fwd.Prefill(Tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(Tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);
        Assert.Equal(Argmax(sequential), Argmax(batched));

        float maxAbs = 0f;
        int exact = 0;
        for (int i = 0; i < sequential.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(sequential[i]) == BitConverter.SingleToInt32Bits(batched[i]))
                exact++;
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        }
        Assert.True(maxAbs < 1e-2f,
            $"matvec-batched vs sequential logits diverged: maxAbs={maxAbs}, bit-exact {exact}/{sequential.Length}.");
    }

    /// <summary>
    /// Issue #156 Item C oracle: the Q4_K compute-bound prefill GEMM (dequant→fp16→cuBLAS,
    /// <c>llm_dequant_q4k_to_f16</c>, weight read once per batch) must be argmax-stable vs
    /// the bit-exact Q4_K matvec GEMM-N (weight re-streamed per token). Both run the same
    /// batched-trunk path with flash off so only the trunk-matmul dtype dispatch differs —
    /// isolating the new fp16 GEMM from the attention reassociation. Not bit-exact (fp16
    /// weight + activation rounding), so this asserts argmax equality + top-5 overlap, the
    /// same contract the Q8_0 prefill GEMM (#141) holds.
    /// </summary>
    [Fact]
    public void Qwen3_8B_BatchedPrefill_Q4KGemm_ArgmaxStable()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512)
        {
            BatchedPrefillEnabled = true,
            PrefillFlashAttnEnabled = false,
            PrefillFlashTcEnabled = false,
        };

        // Reference: bit-exact Q4_K matvec GEMM-N.
        fwd.PrefillGemmEnabled = false;
        var matvec = fwd.Prefill(Tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        // Candidate: the #156-C dequant→fp16→cuBLAS GEMM.
        fwd.ResetCache();
        fwd.PrefillGemmEnabled = true;
        var gemm = fwd.Prefill(Tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        Assert.Equal(matvec.Length, gemm.Length);
        Assert.Equal(Argmax(matvec), Argmax(gemm));

        float maxAbs = 0f;
        for (int i = 0; i < matvec.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(matvec[i] - gemm[i]));
        Assert.True(maxAbs < 1.0f,
            $"Q4_K prefill GEMM vs matvec logits diverged beyond fp16 tolerance: maxAbs={maxAbs}.");

        var matvecTop = TopKSet(matvec, 5);
        var gemmTop = TopKSet(gemm, 5);
        int overlap = 0;
        foreach (var t in gemmTop) if (matvecTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"Q4_K prefill GEMM top-5 overlaps the matvec reference in only {overlap}/5 slots.");
    }

    /// <summary>
    /// Issue #156 Item C2 oracle: the Q4_K int8 tensor-core MMQ prefill (kernel
    /// <c>llm_mmq_q4k</c>, weight read once as int8 — no fp16 dequant temp) must be
    /// argmax-stable vs the bit-exact Q4_K matvec GEMM-N (weight re-streamed per token).
    /// Both run the same batched-trunk path with flash off so only the trunk-matmul
    /// dispatch differs — isolating the new int8 MMQ from the attention reassociation.
    /// Not bit-exact (both operands int8-quantized + the asymmetric min-bias rounds
    /// through fp16 `s`), so this asserts argmax equality + top-5 overlap, the same
    /// contract the Q8_0 MMQ (#141) and the Q4_K C1 GEMM hold.
    /// </summary>
    [Fact]
    public void Qwen3_8B_BatchedPrefill_Q4KMmq_ArgmaxStable()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512)
        {
            BatchedPrefillEnabled = true,
            PrefillFlashAttnEnabled = false,
            PrefillFlashTcEnabled = false,
        };

        // Reference: bit-exact Q4_K matvec GEMM-N.
        fwd.PrefillGemmEnabled = false;
        var matvec = fwd.Prefill(Tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        // Candidate: the #156-C2 int8 MMQ (PrefillGemmEnabled gates the compute-bound
        // path; PrefillMmqEnabled selects MMQ over the C1 fp16 GEMM within it).
        fwd.ResetCache();
        fwd.PrefillGemmEnabled = true;
        fwd.PrefillMmqEnabled = true;
        var mmq = fwd.Prefill(Tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        Assert.Equal(matvec.Length, mmq.Length);
        Assert.Equal(Argmax(matvec), Argmax(mmq));

        float maxAbs = 0f;
        for (int i = 0; i < matvec.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(matvec[i] - mmq[i]));
        Assert.True(maxAbs < 1.0f,
            $"Q4_K prefill MMQ vs matvec logits diverged beyond int8 tolerance: maxAbs={maxAbs}.");

        var matvecTop = TopKSet(matvec, 5);
        var mmqTop = TopKSet(mmq, 5);
        int overlap = 0;
        foreach (var t in mmqTop) if (matvecTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"Q4_K prefill MMQ top-5 overlaps the matvec reference in only {overlap}/5 slots.");
    }
}
