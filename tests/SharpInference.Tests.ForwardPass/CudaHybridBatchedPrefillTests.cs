using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.Pipeline;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #123: byte-parity guard for the batched-trunk prompt prefill on the
/// pure-attention MoE CUDA hybrid path (<see cref="CudaHybridForwardPass"/>,
/// used by Qwen3-Coder-30B-A3B). The batched path
/// (<c>CudaHybridForwardPass.PrefillBatchedTrunk</c>) batches the attention trunk
/// of the GPU layers over the N prompt tokens; the FFN/MoE stage stays per-token,
/// and the CPU layers (Coder-30B splits ~29 GPU + 19 CPU on a 12 GB card) run per
/// token over the N hidden rows. It must produce bit-identical final-token logits
/// to the sequential per-token <see cref="CudaHybridForwardPass.Forward"/> loop
/// (the deterministic reference), toggling only
/// <see cref="CudaHybridForwardPass.BatchedPrefillEnabled"/>.
///
/// Skipped silently when CUDA is unavailable, the model isn't on disk, or
/// construction OOMs — but a failure INSIDE Prefill must FAIL, not skip.
/// </summary>
public sealed class CudaHybridBatchedPrefillTests
{
    private readonly ITestOutputHelper _out;
    public CudaHybridBatchedPrefillTests(ITestOutputHelper o) => _out = o;

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FindCoderPath() => FirstExisting(
        @"C:\p\sharpi\models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        @"E:\models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");

    private static string? FirstExisting(params string[] candidates)
    {
        foreach (var p in candidates)
            if (File.Exists(p)) return p;
        return null;
    }

    /// <summary>
    /// Builds the same GPU/CPU split the CLI's <c>-g -1</c> path produces:
    /// TierPlanner places as many layers on GPU as VRAM allows, the rest on CPU.
    /// </summary>
    private static LayerPlacement PlanCoder(GgufModel model, ModelHyperparams hp, CudaBackend gpu, int ctx)
    {
        var hw = HardwareProfile.Detect(gpu);
        return TierPlanner.Plan(model, hp, hw, turboQuant: false, requestedCtxSize: ctx);
    }

    /// <summary>Repeat a varied seed until the tokenizer emits ≥ <paramref name="approx"/> tokens.</summary>
    private static List<int> LongPrompt(GgufTokenizer tokenizer, int approx)
    {
        const string seed =
            "The quick brown fox jumps over the lazy dog. " +
            "Sphinx of black quartz, judge my vow. " +
            "Pack my box with five dozen liquor jugs. " +
            "How razorback-jumping frogs can level six piqued gymnasts. ";
        var sb = new System.Text.StringBuilder();
        while (true)
        {
            sb.Append(seed);
            var attempt = tokenizer.Encode(sb.ToString());
            if (attempt.Count >= approx) return attempt.ToList();
            if (sb.Length > 400_000)
                throw new InvalidOperationException("Tokenizer not packing enough tokens.");
        }
    }

    private static int FirstBitDiff(float[] a, float[] b)
    {
        for (int i = 0; i < a.Length; i++)
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                return i;
        return -1;
    }

    [Fact]
    public void Inspect_CoderPlacement()
    {
        using var gpu = TryCreate();
        if (gpu is null) { _out.WriteLine("SKIP: CUDA unavailable"); return; }
        var path = FindCoderPath();
        if (path is null) { _out.WriteLine("SKIP: Coder model not on disk"); return; }

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        int ctx = Math.Min(hp.ContextLength, 4096);
        var placement = PlanCoder(model, hp, gpu, ctx);

        _out.WriteLine($"NumLayers={hp.NumLayers} IsMoE={hp.IsMoE} NumExperts={hp.NumExperts} " +
            $"NumActiveExperts={hp.NumActiveExperts} HasSharedExpert={hp.HasSharedExpert}");
        _out.WriteLine($"IsNeoxRope={hp.IsNeoxRope} HasQkNorm={hp.HasQkNorm} UseL2QkNorm={hp.UseL2QkNorm} " +
            $"NoRopeLayerStep={hp.NoRopeLayerStep} HasAttnBias={hp.HasAttnBias} headDim={hp.HeadDim}");
        _out.WriteLine($"PLACEMENT GpuLayers={placement.GpuLayers} CpuLayers={placement.CpuLayers} ctx={placement.RecommendedCtxSize}");
    }

    /// <summary>
    /// Single-chunk parity: the batched-trunk path must produce final-token logits
    /// bit-identical to the sequential per-token Forward loop. Exercises the batched
    /// GPU attention trunk + per-token MoE FFN + the GPU→CPU N-row transfer + per-token
    /// CPU layers (Coder-30B splits across both tiers on a 12 GB card).
    /// </summary>
    [Fact]
    public void BatchedPrefill_BitwiseMatchesSequential_Coder()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCoderPath();
        if (path is null) return;

        bool prev = CudaHybridForwardPass.BatchedPrefillEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            int ctx = Math.Min(hp.ContextLength, 4096);
            var placement = PlanCoder(model, hp, gpu, ctx);

            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts! " +
                "The five boxing wizards jump quickly.");
            Assert.True(tokens.Count >= 8, $"Prompt tokenized to only {tokens.Count} tokens.");

            float[] Run(bool batched)
            {
                CudaHybridForwardPass.BatchedPrefillEnabled = batched;
                using var fwd = TryConstruct(model, gpu, hp, placement);
                if (fwd is null) return Array.Empty<float>();
                var logits = fwd.Prefill(tokens).ToArray();
                // Guard against a vacuous pass: the batched arm must actually take the
                // batched-trunk path (not silently fall back to per-token because the
                // config was gated out), else the bit-parity assertion compares the
                // per-token loop against itself.
                if (batched)
                    Assert.True(fwd.LastPrefillWasBatched,
                        "Batched arm fell back to the per-token path — the parity check would be vacuous. " +
                        $"GpuLayers={placement.GpuLayers} CpuLayers={placement.CpuLayers}.");
                return logits;
            }

            float[] seq = Run(false);
            float[] bat = Run(true);
            if (seq.Length == 0 || bat.Length == 0)
            {
                _out.WriteLine("SKIP: construction OOM'd on this box.");
                return;
            }

            Assert.Equal(seq.Length, bat.Length);
            int firstDiff = FirstBitDiff(seq, bat);
            Assert.True(firstDiff < 0,
                $"Batched-trunk prefill diverges from sequential at index {firstDiff} " +
                $"(seq={(firstDiff >= 0 ? seq[firstDiff] : 0)} bat={(firstDiff >= 0 ? bat[firstDiff] : 0)}). " +
                "PrefillBatchedTrunk must be bit-identical to the sequential Forward loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
            _out.WriteLine($"OK single-chunk N={tokens.Count} greedy={Sampler.Greedy(seq)}");
        }
        finally
        {
            CudaHybridForwardPass.BatchedPrefillEnabled = prev;
        }
    }

    /// <summary>
    /// Multi-chunk parity: prefill the prompt in two segments (<c>[0,k)</c> then
    /// <c>[k,N)</c> with <c>startPos=k</c>) and assert the final-token logits are
    /// bit-identical. Exercises <c>startPos &gt; 0</c>, cross-chunk KV continuity, and
    /// the exact-size scratch reallocation between two chunks of different length.
    /// </summary>
    [Fact]
    public void BatchedPrefill_MultiChunk_BitwiseMatchesSequential_Coder()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCoderPath();
        if (path is null) return;

        bool prev = CudaHybridForwardPass.BatchedPrefillEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            int ctx = Math.Min(hp.ContextLength, 4096);
            var placement = PlanCoder(model, hp, gpu, ctx);

            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts! " +
                "The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow.");
            int N = tokens.Count;
            Assert.True(N >= 12, $"Prompt tokenized to only {N} tokens.");
            int k1 = N / 3;

            float[] RunTwoChunks(bool batched)
            {
                CudaHybridForwardPass.BatchedPrefillEnabled = batched;
                using var fwd = TryConstruct(model, gpu, hp, placement);
                if (fwd is null) return Array.Empty<float>();
                fwd.Prefill(tokens.Take(k1).ToList(), 0);
                var logits = fwd.Prefill(tokens.Skip(k1).ToList(), k1).ToArray();
                if (batched)
                    Assert.True(fwd.LastPrefillWasBatched,
                        "Batched arm fell back to the per-token path — the parity check would be vacuous. " +
                        $"GpuLayers={placement.GpuLayers} CpuLayers={placement.CpuLayers}.");
                return logits;
            }

            float[] seq = RunTwoChunks(false);
            float[] bat = RunTwoChunks(true);
            if (seq.Length == 0 || bat.Length == 0)
            {
                _out.WriteLine("SKIP: construction OOM'd on this box.");
                return;
            }

            Assert.Equal(seq.Length, bat.Length);
            int firstDiff = FirstBitDiff(seq, bat);
            Assert.True(firstDiff < 0,
                $"Multi-chunk batched prefill diverges from sequential at index {firstDiff} " +
                $"(N={N}, split at {k1}). Cross-chunk KV continuity or the startPos>0 path " +
                "is not bit-identical to the sequential loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
            _out.WriteLine($"OK multi-chunk N={N} split={k1} greedy={Sampler.Greedy(seq)}");
        }
        finally
        {
            CudaHybridForwardPass.BatchedPrefillEnabled = prev;
        }
    }

    /// <summary>
    /// &gt;4096 parity: a prompt longer than the 4096-position shared-scores window forces
    /// the wave-based batched SDPA (<c>AttentionBatchedWave</c>, issue #118) inside the
    /// trunk. A small wave budget forces the multi-wave loop. Final-token logits must be
    /// bit-identical to the sequential per-token Forward loop (the deterministic reference).
    /// </summary>
    [Fact]
    public void BatchedPrefill_Over4096_BitwiseMatchesSequential_Coder()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCoderPath();
        if (path is null) return;

        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB");
        bool prev = CudaHybridForwardPass.BatchedPrefillEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (hp.ContextLength < 4400) { _out.WriteLine("SKIP: ctx < 4400"); return; }
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var tokens = LongPrompt(tokenizer, 4200);
            Assert.True(tokens.Count > 4096, $"Prompt only {tokens.Count} tokens; need >4096.");
            int ctx = Math.Min(hp.ContextLength, tokens.Count + 64);
            var placement = PlanCoder(model, hp, gpu, ctx);

            float[] Run(bool batched, string? budgetMb)
            {
                CudaHybridForwardPass.BatchedPrefillEnabled = batched;
                Environment.SetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB", budgetMb);
                using var fwd = TryConstruct(model, gpu, hp, placement);
                if (fwd is null) return Array.Empty<float>();
                var logits = fwd.Prefill(tokens).ToArray();
                if (batched)
                    Assert.True(fwd.LastPrefillWasBatched,
                        "Batched arm fell back to the per-token path — the parity check would be vacuous. " +
                        $"GpuLayers={placement.GpuLayers} CpuLayers={placement.CpuLayers}.");
                return logits;
            }

            float[] seq  = Run(false, null);   // per-position SDPA (reference)
            float[] wave = Run(true, "8");      // batched wave SDPA, forced multi-wave
            if (seq.Length == 0 || wave.Length == 0)
            {
                _out.WriteLine("SKIP: construction OOM'd on this box (large ctx).");
                return;
            }

            Assert.Equal(seq.Length, wave.Length);
            int firstDiff = FirstBitDiff(seq, wave);
            Assert.True(firstDiff < 0,
                $"Wave-based >4096 batched SDPA diverges from the per-position path at index {firstDiff} " +
                $"(N={tokens.Count}). AttentionBatchedWave host wiring must be bit-identical to the per-position loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(wave));
            _out.WriteLine($"OK >4096 N={tokens.Count} greedy={Sampler.Greedy(seq)}");
        }
        finally
        {
            CudaHybridForwardPass.BatchedPrefillEnabled = prev;
            Environment.SetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB", prevBudget);
        }
    }

    // Construction is the ONLY place allowed to skip (a box without the VRAM to host this
    // 30 GB MoE model under the planned split). A failure INSIDE Prefill must propagate
    // and fail the test — that is exactly the regression these oracles exist to catch.
    private static CudaHybridForwardPass? TryConstruct(
        GgufModel model, CudaBackend gpu, ModelHyperparams hp, LayerPlacement placement)
    {
        try { return new CudaHybridForwardPass(model, gpu, hp, placement); }
        catch (InvalidOperationException) { return null; } // OOM / VRAM at construction
        catch (NotSupportedException) { return null; }
    }
}
