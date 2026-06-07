using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #141: the all-GPU batched-trunk prefill for Gemma 4 routes its Q8_0 trunk
/// matmuls through the compute-bound cuBLAS GEMM (<see cref="CudaBackend.MatMulBatchedGemm"/>),
/// which dequantizes the weight and rounds activations to fp16 before a tensor-core
/// multiply. That is <b>not</b> bit-exact to the per-token fp32 matvec loop, so this
/// oracle asserts argmax-stability plus a tolerance envelope rather than the
/// &gt;95%-bit-exact match the pre-#141 matvec GEMM-N path satisfied.
///
/// A second case (<see cref="Gemma4_E4B_BatchedPrefill_GemmOff_MatchesSequentialBitExact"/>)
/// keeps the strict bit-exact oracle for the matvec GEMM-N path (PrefillGemmEnabled=false),
/// so the #136 batched primitives stay individually verified.
///
/// Single model instance, toggled via <see cref="CudaForwardPass.BatchedPrefillEnabled"/>
/// / <see cref="CudaForwardPass.PrefillGemmEnabled"/> with a <c>ResetCache</c> between
/// runs (two 8 GB instances would not co-reside). Silent-skips when CUDA or the GGUF
/// is absent.
/// </summary>
public sealed class Gemma4CudaBatchedPrefillTests
{
    private const string ModelFile = "gemma-4-E4B-it-Q8_0.gguf";

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

    [Fact]
    public void Gemma4_E4B_BatchedPrefill_GemmMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.HasPerLayerTokenEmbd);

        // A prompt long enough to exercise SWA windowing is not required for parity,
        // but >1 token is (batched path needs N≥2). Mixes ids across the vocab.
        var tokens = new int[] { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222, 333, 444 };

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        // Batched cuBLAS-GEMM prefill (#141). Pin MMQ + flash off so this isolates the
        // dequant→fp16→cuBLAS GEMM path (each has its own oracle below).
        fwd.BatchedPrefillEnabled = true;
        fwd.PrefillGemmEnabled = true;
        fwd.PrefillMmqEnabled = false;
        fwd.PrefillFlashAttnEnabled = false;
        fwd.PrefillFlashTcEnabled = false;
        var batched = fwd.Prefill(tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched,
            "Batched-trunk prefill did not engage — check IsGemma4BatchedPrefillSupported gating.");

        // Sequential per-token loop on the same instance (the fp32 reference).
        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);

        // The decisive parity signal: argmax must match the fp32 reference. The
        // GEMM rounds weights + activations to fp16, so logits track to fp tolerance
        // (post-softcap, |logit| ≲ FinalLogitSoftcap), not bit-for-bit.
        Assert.Equal(Argmax(sequential), Argmax(batched));

        float maxAbs = 0f;
        for (int i = 0; i < sequential.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        // fp16 trunk over 42 layers + softcap: a few tenths of a logit is expected,
        // a whole-number divergence would signal a real wiring bug.
        Assert.True(maxAbs < 1.0f,
            $"GEMM-batched vs sequential logits diverged beyond fp16 tolerance: maxAbs={maxAbs}.");

        // Top-5 set should be stable under fp16 rounding (order may shuffle below #1).
        var seqTop = TopKSet(sequential, 5);
        var batTop = TopKSet(batched, 5);
        int overlap = 0;
        foreach (var t in batTop) if (seqTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"GEMM-batched top-5 overlaps the fp32 reference in only {overlap}/5 slots; " +
            "fp16 GEMM is diverging more than rounding explains.");
    }

    [Fact]
    public void Gemma4_E4B_BatchedPrefill_MmqMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.HasPerLayerTokenEmbd);

        var tokens = new int[] { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222, 333, 444 };

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        // Int8 tensor-core MMQ prefill (#141, default on). Like the cuBLAS GEMM path
        // it int8-quantizes both operands, so it is argmax-stable to the fp32 per-token
        // reference, not bit-exact.
        fwd.BatchedPrefillEnabled = true;
        fwd.PrefillGemmEnabled = true;
        fwd.PrefillMmqEnabled = true;
        fwd.PrefillFlashAttnEnabled = false;   // isolate the MMQ matmul (flash has its own oracle)
        fwd.PrefillFlashTcEnabled = false;
        var batched = fwd.Prefill(tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched,
            "Batched-trunk MMQ prefill did not engage — check IsGemma4BatchedPrefillSupported gating.");

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);
        Assert.Equal(Argmax(sequential), Argmax(batched));

        float maxAbs = 0f;
        for (int i = 0; i < sequential.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        Assert.True(maxAbs < 1.0f,
            $"MMQ-batched vs sequential logits diverged beyond int8 tolerance: maxAbs={maxAbs}.");

        var seqTop = TopKSet(sequential, 5);
        var batTop = TopKSet(batched, 5);
        int overlap = 0;
        foreach (var t in batTop) if (seqTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"MMQ-batched top-5 overlaps the fp32 reference in only {overlap}/5 slots.");
    }

    [Fact]
    public void Gemma4_E4B_BatchedPrefill_FlashTcMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.HasPerLayerTokenEmbd);

        var tokens = new int[] { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222, 333, 444 };

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        // Tensor-core flash-attention prefill (#146). Like the half2 flash it is
        // argmax-stable (online softmax + fp16 Q/K/V/P), not bit-exact. End-to-end
        // full-model check on top of the per-kernel CudaFlashAttnTcTests parity, over
        // the real per-layer head_dim mix (256 SWA / 512 global) and KV-share tail.
        fwd.BatchedPrefillEnabled = true;
        fwd.PrefillFlashTcEnabled = true;
        var batched = fwd.Prefill(tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);
        Assert.Equal(Argmax(sequential), Argmax(batched));

        float maxAbs = 0f;
        for (int i = 0; i < sequential.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        Assert.True(maxAbs < 1.0f,
            $"TC flash prefill vs sequential logits diverged beyond fp tolerance: maxAbs={maxAbs}.");

        var seqTop5 = TopKSet(sequential, 5);
        var batTop5 = TopKSet(batched, 5);
        int ov = 0;
        foreach (var t in batTop5) if (seqTop5.Contains(t)) ov++;
        Assert.True(ov >= 4,
            $"TC flash prefill top-5 overlaps the fp32 reference in only {ov}/5 slots.");
    }

    [Fact]
    public void Gemma4_E4B_BatchedPrefill_FlashAttnMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.HasPerLayerTokenEmbd);

        var tokens = new int[] { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222, 333, 444 };

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        // Flash-attention prefill (#141, default on). Its online softmax reassociates
        // the score sum, so it is argmax-stable to the fp32 per-token reference, not
        // bit-exact. Exercises both the SWA-windowed and global attention layers.
        fwd.BatchedPrefillEnabled = true;
        fwd.PrefillFlashAttnEnabled = true;
        fwd.PrefillFlashTcEnabled = false;   // this oracle targets the half2 kernel; TC has its own
        var batched = fwd.Prefill(tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);
        Assert.Equal(Argmax(sequential), Argmax(batched));

        float maxAbs = 0f;
        for (int i = 0; i < sequential.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        Assert.True(maxAbs < 1.0f,
            $"Flash-attn prefill vs sequential logits diverged beyond fp tolerance: maxAbs={maxAbs}.");

        var seqTop = TopKSet(sequential, 5);
        var batTop = TopKSet(batched, 5);
        int overlap = 0;
        foreach (var t in batTop) if (seqTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"Flash-attn prefill top-5 overlaps the fp32 reference in only {overlap}/5 slots.");
    }

    [Fact]
    public void Gemma4_E4B_BatchedPrefill_GemmOff_MatchesSequentialBitExact()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.HasPerLayerTokenEmbd);

        var tokens = new int[] { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222, 333, 444 };

        // Pin Q8_0 matvec to the fp32-decode kernel on both sides so the matvec
        // GEMM-N batched path can be compared bit-exactly to the per-token loop
        // (the default dp4a path quantizes activations and is only argmax-stable).
        gpu.Q80Dp4aEnabled = false;

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        // Batched matvec GEMM-N path (#136), bit-exact to per-token by construction.
        // Flash attention pinned off: its online softmax reassociates the score sum
        // and is only argmax-stable, which would break this bit-exact oracle (it
        // verifies the batched matvec/norm/rope primitives, not the attention method).
        fwd.BatchedPrefillEnabled = true;
        fwd.PrefillGemmEnabled = false;
        fwd.PrefillFlashAttnEnabled = false;
        fwd.PrefillFlashTcEnabled = false;
        var batched = fwd.Prefill(tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(tokens).ToArray();
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

        // Every batched primitive is individually bit-identical to its per-token form.
        float exactFrac = (float)exact / sequential.Length;
        Assert.True(exactFrac > 0.95f,
            $"matvec-batched vs sequential only {exact}/{sequential.Length} ({exactFrac:P1}) bit-exact " +
            $"(expected >95%); maxAbs={maxAbs}.");
    }

    /// <summary>
    /// Issue #162 (SWA sub-item): a Gemma-4 prompt LONGER than the 512-token sliding window
    /// but still under the 4096 single-batch cap must produce correct output. This is the
    /// scenario the old window-sized-cache-with-absolute-indexing got wrong (positions ≥
    /// window read/wrote out of bounds). With the SWA ring (cache sized window +
    /// SwaRingHeadroom, capped at ctx) the per-token loop is itself correct again, so the
    /// batched path is checked against it. ctx=1024 makes the SWA cache full (1024 &lt;
    /// window+headroom), so this isolates the windowed-attention-past-the-window fix from
    /// the ring-wrap fix (covered by the chunked test below).
    /// </summary>
    [Fact]
    public void Gemma4_E4B_BatchedPrefill_PastWindow_MatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.SlidingWindowSize > 0 && hp.SlidingWindowSize < 700,
            $"Test assumes a sliding window < the 700-token prompt; got {hp.SlidingWindowSize}.");

        // 700 > window (512): the trailing queries' windows exclude the earliest tokens,
        // exactly the regime that the absolute-into-window-sized-cache bug corrupted.
        var tokens = MakeTokens(700);

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 1024);

        fwd.BatchedPrefillEnabled = true;   // shipped defaults (flash TC on)
        var batched = fwd.Prefill(tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);
        Assert.Equal(Argmax(sequential), Argmax(batched));

        // Logit-envelope check on top of argmax: the flash TC path rounds to fp16 over 42
        // layers + softcap, so a few tenths of a logit is expected; a whole-number gap would
        // signal a wiring bug (e.g. a ring slot reading the wrong position) that argmax alone
        // could miss when it doesn't quite flip the top token.
        float maxAbs = 0f;
        for (int i = 0; i < sequential.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        Assert.True(maxAbs < 1.5f,
            $"Past-window batched vs per-token logits diverged beyond fp16 tolerance: maxAbs={maxAbs}.");

        var seqTop = TopKSet(sequential, 5);
        var batTop = TopKSet(batched, 5);
        int overlap = 0;
        foreach (var t in batTop) if (seqTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"Past-window batched top-5 overlaps the per-token reference in only {overlap}/5 slots.");
    }

    /// <summary>
    /// Issue #162 (SWA sub-item): a Gemma-4 prompt longer than the 4096 cap must take the
    /// chunked batched path (flash streaming the prior KV) and stay argmax-stable vs the
    /// per-token loop, with both running through the SWA KV ring.
    /// <para>
    /// This is the decisive ring oracle: the per-token loop attends right after each single
    /// append (so it only ever needs ring ≥ window) while the chunked path appends a whole
    /// 4096-token chunk before any of those queries attend (so it needs ring ≥ window +
    /// chunk span). If the ring were undersized, the chunked path would overwrite an
    /// earlier query's window and diverge from the per-token reference — so agreement
    /// validates the window + SwaRingHeadroom sizing. ctx exceeds window + headroom, so the
    /// SWA cache is a true ring (positions ≥ ring size wrap) rather than a full cache.
    /// </para>
    /// <para>
    /// N=5040 exercises a full + partial chunk (4096 + 944); N=8192 the exact-multiple
    /// boundary (two full 4096 chunks).
    /// </para>
    /// </summary>
    [Theory]
    [InlineData(5040, 6144)]
    [InlineData(8192, 9216)]
    public void Gemma4_E4B_ChunkedBatchedPrefill_Over4096_MatchesSequential(int promptLen, int ctx)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.SlidingWindowSize > 0);

        var longTokens = MakeTokens(promptLen);

        // Disable SnapKV (a >budget prompt would otherwise route to the per-token SnapKV
        // eviction path before the batched gate). Construct under the override, then restore.
        var prevSnap = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        CudaForwardPass fwd;
        try { fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: ctx); }
        finally { Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevSnap); }
        using var _fwd = fwd;

        fwd.BatchedPrefillEnabled = true;   // shipped defaults (flash TC on) → chunked path
        var batched = fwd.Prefill(longTokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched,
            "Chunked batched prefill did not engage for a >4096-token Gemma 4 prompt (#162).");

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(longTokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);
        Assert.Equal(Argmax(sequential), Argmax(batched));

        // Envelope check: the chunked flash path reassociates the softmax over thousands of
        // streamed keys, so it's looser than the ≤4096 case, but a several-unit gap would
        // still flag a ring-overwrite bug (an early query reading a clobbered window slot)
        // that argmax-stability alone might not surface.
        float maxAbs = 0f;
        for (int i = 0; i < sequential.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        Assert.True(maxAbs < 2.0f,
            $"Chunked batched vs per-token logits diverged beyond fp tolerance: maxAbs={maxAbs}.");

        var seqTop = TopKSet(sequential, 5);
        var batTop = TopKSet(batched, 5);
        int overlap = 0;
        foreach (var t in batTop) if (seqTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"Chunked batched top-5 overlaps the per-token reference in only {overlap}/5 slots.");
    }

    // Deterministic spread across the vocab via a small LCG; all ids well within Gemma 4's
    // vocab. Token 2 (BOS) leads so the prompt starts in-distribution.
    private static int[] MakeTokens(int n)
    {
        var t = new int[n];
        t[0] = 2;
        uint s = 0x9E3779B9u;
        for (int i = 1; i < n; i++)
        {
            s = s * 1664525u + 1013904223u;
            t[i] = (int)(s % 100000u) + 5;
        }
        return t;
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
}
