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
/// and any CPU layers run per token over the N hidden rows. It must produce
/// bit-identical final-token logits to the sequential per-token
/// <see cref="CudaHybridForwardPass.Forward"/> loop (the deterministic reference),
/// toggling only <see cref="CudaHybridForwardPass.BatchedPrefillEnabled"/>.
///
/// NOTE: on a 12 GB card the current planner places Coder-30B as 48 GPU / 0 CPU
/// layers (CPU-MoE handles the routed experts; all attention trunks fit on GPU), so
/// these natural-placement oracles exercise the all-GPU-layer branch. The CpuLayers&gt;0
/// branch is covered deterministically by
/// <see cref="BatchedPrefill_CpuEmbedding_CpuLayersSplit_BitwiseMatchesSequential_Coder"/>
/// via a synthesized split.
///
/// Skipped silently when CUDA is unavailable, the model isn't on disk, or
/// construction OOMs — but a failure INSIDE Prefill must FAIL, not skip.
/// </summary>
public sealed class CudaHybridBatchedPrefillTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    // Issue #162: these oracles assert the batched trunk is bit-identical to the
    // sequential loop. The compute-bound attention-projection path (MMQ/GEMM) is
    // argmax-stable, NOT byte-exact, so pin it OFF for the whole class to keep the batched
    // side on the byte-exact matvec. MMQ/GEMM correctness is covered by CudaMmqQ4K/Q8_0Tests
    // + CudaGemmQ6K/Q5KTests. Restored in Dispose.
    private readonly bool _prevHybridCompute = CudaHybridForwardPass.HybridPrefillComputeEnabled;
    public CudaHybridBatchedPrefillTests(ITestOutputHelper o)
    {
        _out = o;
        CudaHybridForwardPass.HybridPrefillComputeEnabled = false;
    }
    public void Dispose() => CudaHybridForwardPass.HybridPrefillComputeEnabled = _prevHybridCompute;

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
    /// GPU attention trunk + per-token MoE FFN (on this box Coder-30B places all 48 layers
    /// on GPU with CPU-MoE, so the GPU→CPU N-row transfer + per-token CPU-layer loop are
    /// covered by the synthesized-split oracle, not here).
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
    /// Issue #162: the class pins <see cref="CudaHybridForwardPass.HybridPrefillComputeEnabled"/>
    /// OFF so the bit-parity oracles validate the byte-exact matvec batching. This test flips
    /// it ON — the path that actually ships to users — and asserts the compute-routed batched
    /// prefill (Q8_0/Q4_K → int8 MMQ, Q6_K/Q5_K → dequant→fp16 GEMM for the attention
    /// projections) stays <b>argmax-stable</b> vs the byte-exact matvec batching: shared top-5 and
    /// logits within a loose fp tolerance (the matmuls are argmax-stable, not byte-exact, so a
    /// final-token near-tie can swap the greedy pick — see the assertion note). This is the
    /// integration coverage the kernel-isolation tests
    /// (<see cref="CudaMmqQ4KTests"/>, <see cref="CudaGemmQ6KTests"/>, …) can't give — it proves
    /// the routing switch is wired correctly at model scale.
    /// </summary>
    [Fact]
    public void BatchedPrefill_ComputeRouting_ArgmaxStableVsMatvec_Coder()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCoderPath();
        if (path is null) return;

        bool prevBatched = CudaHybridForwardPass.BatchedPrefillEnabled;
        bool prevCompute = CudaHybridForwardPass.HybridPrefillComputeEnabled;
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

            float[] Run(bool compute)
            {
                CudaHybridForwardPass.BatchedPrefillEnabled = true;
                CudaHybridForwardPass.HybridPrefillComputeEnabled = compute;
                using var fwd = TryConstruct(model, gpu, hp, placement);
                if (fwd is null) return Array.Empty<float>();
                var logits = fwd.Prefill(tokens).ToArray();
                Assert.True(fwd.LastPrefillWasBatched,
                    "Batched arm fell back to per-token — the comparison would be vacuous.");
                return logits;
            }

            float[] matvec  = Run(false);   // byte-exact GEMM-N matvec
            float[] compute = Run(true);    // default-on MMQ/GEMM compute routing
            if (matvec.Length == 0 || compute.Length == 0)
            {
                _out.WriteLine("SKIP: construction OOM'd on this box.");
                return;
            }

            Assert.Equal(matvec.Length, compute.Length);

            float maxAbs = 0f;
            for (int i = 0; i < matvec.Length; i++)
                maxAbs = MathF.Max(maxAbs, MathF.Abs(matvec[i] - compute[i]));

            // Argmax-stable, not byte-exact: the int8 MMQ / fp16 GEMM routing diverges from the
            // byte-exact matvec by up to ~1.2 logits across the vocab, so when the final-token
            // top-1/top-2 fall within that band the greedy tie-break can legitimately swap. (On
            // this Coder prompt the top two are 362 @ 16.67 and 4220 @ 16.40 — a 0.27-logit tie.)
            // Assert the contract that actually holds: the two routes share their top-5 and stay
            // within fp tolerance — the same contract the Qwen3 compute-routing oracles use.
            static HashSet<int> Top5(float[] v)
            {
                var idx = new int[v.Length];
                for (int i = 0; i < idx.Length; i++) idx[i] = i;
                Array.Sort(idx, (a, b) => v[b].CompareTo(v[a]));
                var set = new HashSet<int>();
                for (int i = 0; i < 5 && i < idx.Length; i++) set.Add(idx[i]);
                return set;
            }
            var matvecTop = Top5(matvec);
            var computeTop = Top5(compute);
            int overlap = 0;
            foreach (var t in computeTop) if (matvecTop.Contains(t)) overlap++;
            Assert.True(overlap >= 4,
                $"compute-routing top-5 overlaps the byte-exact matvec in only {overlap}/5 slots " +
                $"(maxAbs={maxAbs:E2}).");
            Assert.True(maxAbs < 3.0f,
                $"compute-routing vs matvec logits diverged beyond fp tolerance: maxAbs={maxAbs:E2}.");
            _out.WriteLine($"OK compute-routing argmax-stable: N={tokens.Count} " +
                $"greedy={Sampler.Greedy(matvec)} maxAbs={maxAbs:E2}");
        }
        finally
        {
            CudaHybridForwardPass.BatchedPrefillEnabled = prevBatched;
            CudaHybridForwardPass.HybridPrefillComputeEnabled = prevCompute;
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

    /// <summary>
    /// Issue #218: batched-trunk prefill with a CPU-resident embedding (the low-VRAM / 12 GB
    /// default config). <see cref="CudaHybridForwardPass.ForceCpuResidentEmbedding"/> forces the
    /// embedding + output table onto the CPU (<c>_gpuEmbedding/_gpuOutputWeight</c> null) even
    /// though Coder-30B's Q4_K embedding fits in VRAM, so the batched path's CPU-embed staging
    /// (one pinned [N × embDim] block → single async H2D) and the <c>_gpuOutputWeight is null</c>
    /// CPU output branches are exercised. Final-token logits must be bit-identical to the
    /// sequential per-token <see cref="CudaHybridForwardPass.Forward"/> loop under the SAME
    /// (forced-CPU-embed) config — only <see cref="CudaHybridForwardPass.BatchedPrefillEnabled"/>
    /// toggles. Pre-#218 the batched arm would have fallen back to per-token (gated out); the
    /// <c>LastPrefillWasBatched</c> assertion guards against a vacuous pass.
    /// </summary>
    [Fact]
    public void BatchedPrefill_CpuEmbedding_BitwiseMatchesSequential_Coder()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCoderPath();
        if (path is null) return;

        bool prevBatched = CudaHybridForwardPass.BatchedPrefillEnabled;
        bool prevForce = CudaHybridForwardPass.ForceCpuResidentEmbedding;
        try
        {
            // Forcing embedding+output to CPU only FREES VRAM, so the planned GPU layers still
            // fit — no new OOM risk vs the GPU-embed oracle's split.
            CudaHybridForwardPass.ForceCpuResidentEmbedding = true;

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
                if (batched)
                    Assert.True(fwd.LastPrefillWasBatched,
                        "Batched arm fell back to the per-token path — the parity check would be vacuous. " +
                        "The CPU-resident-embedding gate (#218) must not block the batched path. " +
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
                $"CPU-embed batched-trunk prefill diverges from sequential at index {firstDiff} " +
                $"(seq={(firstDiff >= 0 ? seq[firstDiff] : 0)} bat={(firstDiff >= 0 ? bat[firstDiff] : 0)}). " +
                "The batched CPU-embed staging upload must be bit-identical to the per-token CpuEmbedToken loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
            _out.WriteLine($"OK cpu-embed single-chunk N={tokens.Count} greedy={Sampler.Greedy(seq)}");
        }
        finally
        {
            CudaHybridForwardPass.BatchedPrefillEnabled = prevBatched;
            CudaHybridForwardPass.ForceCpuResidentEmbedding = prevForce;
        }
    }

    /// <summary>
    /// Issue #218 multi-chunk: prefill in two segments (<c>[0,k)</c> then <c>[k,N)</c> with
    /// <c>startPos=k</c>) on the forced CPU-resident-embedding config. Exercises the batched
    /// CPU-embed upload across <c>startPos &gt; 0</c>, the exact-size pinned-embed buffer reuse
    /// between two chunks of different length, and cross-chunk KV continuity. Final-token logits
    /// must be bit-identical to the sequential per-token Forward loop under the same config.
    /// </summary>
    [Fact]
    public void BatchedPrefill_CpuEmbedding_MultiChunk_BitwiseMatchesSequential_Coder()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCoderPath();
        if (path is null) return;

        bool prevBatched = CudaHybridForwardPass.BatchedPrefillEnabled;
        bool prevForce = CudaHybridForwardPass.ForceCpuResidentEmbedding;
        try
        {
            CudaHybridForwardPass.ForceCpuResidentEmbedding = true;

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
                $"Multi-chunk CPU-embed batched prefill diverges from sequential at index {firstDiff} " +
                $"(N={N}, split at {k1}). Cross-chunk KV continuity, the startPos>0 path, or the " +
                "exact-size pinned-embed buffer reuse is not bit-identical to the sequential loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
            _out.WriteLine($"OK cpu-embed multi-chunk N={N} split={k1} greedy={Sampler.Greedy(seq)}");
        }
        finally
        {
            CudaHybridForwardPass.BatchedPrefillEnabled = prevBatched;
            CudaHybridForwardPass.ForceCpuResidentEmbedding = prevForce;
        }
    }

    /// <summary>
    /// Issue #218 with CPU layers &gt; 0. On a 12 GB card the current planner gives Coder-30B
    /// 48 GPU / 0 CPU layers (CPU-MoE handles the routed experts), so the natural-placement
    /// oracles above exercise the all-GPU-layer branch. This test SYNTHESIZES a half/half split
    /// (overriding <see cref="LayerPlacement.GpuLayers"/>/<c>CpuLayers</c>) so the batched
    /// CPU-embed staging (step 1) and the per-token CPU-layer download loop (step 3) run
    /// together — the combination the planner won't produce on this box. The synthetic split
    /// only uses LESS GPU VRAM than the full plan, so it cannot OOM where the others construct.
    /// Final-token logits must be bit-identical to the sequential per-token Forward loop.
    /// </summary>
    [Fact]
    public void BatchedPrefill_CpuEmbedding_CpuLayersSplit_BitwiseMatchesSequential_Coder()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCoderPath();
        if (path is null) return;

        bool prevBatched = CudaHybridForwardPass.BatchedPrefillEnabled;
        bool prevForce = CudaHybridForwardPass.ForceCpuResidentEmbedding;
        try
        {
            CudaHybridForwardPass.ForceCpuResidentEmbedding = true;

            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            int ctx = Math.Min(hp.ContextLength, 4096);
            var planned = PlanCoder(model, hp, gpu, ctx);
            // Force a cross-tier split so CpuLayers > 0 (the planner gives 48/0 here via CPU-MoE).
            int gpuLayers = hp.NumLayers / 2;
            var split = planned with { GpuLayers = gpuLayers, CpuLayers = hp.NumLayers - gpuLayers };

            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts!");
            Assert.True(tokens.Count >= 8, $"Prompt tokenized to only {tokens.Count} tokens.");

            float[] Run(bool batched)
            {
                CudaHybridForwardPass.BatchedPrefillEnabled = batched;
                using var fwd = TryConstruct(model, gpu, hp, split);
                if (fwd is null) return Array.Empty<float>();
                var logits = fwd.Prefill(tokens).ToArray();
                if (batched)
                    Assert.True(fwd.LastPrefillWasBatched,
                        "Batched arm fell back to the per-token path — the parity check would be vacuous. " +
                        $"GpuLayers={split.GpuLayers} CpuLayers={split.CpuLayers}.");
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
                $"CPU-embed + CpuLayers>0 batched prefill diverges from sequential at index {firstDiff} " +
                $"(GpuLayers={split.GpuLayers} CpuLayers={split.CpuLayers}). The batched CPU-embed staging " +
                "and the per-token CPU-layer download loop must together be bit-identical to the sequential loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
            _out.WriteLine($"OK cpu-embed CpuLayers>0 N={tokens.Count} split={split.GpuLayers}/{split.CpuLayers} greedy={Sampler.Greedy(seq)}");
        }
        finally
        {
            CudaHybridForwardPass.BatchedPrefillEnabled = prevBatched;
            CudaHybridForwardPass.ForceCpuResidentEmbedding = prevForce;
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
