using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.Pipeline;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #410: byte-parity guard for the batched CPU-MoE routed-expert prefill on the
/// pure-attention MoE CUDA hybrid path (<see cref="CudaHybridForwardPass"/>, Qwen3-Coder-30B).
/// With <c>SHARPI_HYBRID_BATCHED_MOE=1</c> the batched-trunk prefill replaces the per-token
/// FFN/MoE loop with the CSR-bucketed batched routed-expert scheme (one D2H of all N norm
/// rows per layer, host router + grouped gate/up/down dots, one H2D of the routed outputs).
/// It must produce bit-identical final-token logits to the sequential per-token
/// <see cref="CudaHybridForwardPass.Forward"/> loop — every tier of the grouped dots mirrors
/// the single-dot accumulation order (issues #112/#114) and the weighted reduce runs in
/// top-k order, so the parity contract is byte-exact, not just argmax-stable.
///
/// <c>SHARPI_CPU_MOE=1</c> is pinned for both arms so the pass deterministically selects
/// CPU-MoE regardless of the box's VRAM (the auto-select would otherwise pick GPU-SLRU on
/// a large card and the batched arm's non-vacuous assertion would fail spuriously).
///
/// Skipped silently when CUDA is unavailable, the model isn't on disk, or construction
/// OOMs — but a failure INSIDE Prefill must FAIL, not skip.
/// </summary>
public sealed class CudaHybridBatchedCpuMoePrefillTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    // Issue #162: pin the compute-bound attention-projection path (MMQ/GEMM, argmax-stable
    // NOT byte-exact) OFF so the oracle isolates the batched CPU-MoE stage against the
    // byte-exact trunk. Restored in Dispose.
    private readonly bool _prevHybridCompute = CudaHybridForwardPass.HybridPrefillComputeEnabled;
    public CudaHybridBatchedCpuMoePrefillTests(ITestOutputHelper o)
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
        @"C:\models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        @"E:\models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");

    private static string? FirstExisting(params string[] candidates)
    {
        foreach (var p in candidates)
            if (File.Exists(p)) return p;
        return null;
    }

    private static CudaHybridForwardPass? TryConstruct(
        GgufModel model, CudaBackend gpu, ModelHyperparams hp, LayerPlacement placement)
    {
        try { return new CudaHybridForwardPass(model, gpu, hp, placement); }
        catch (InvalidOperationException) { return null; } // OOM / VRAM at construction
        catch (NotSupportedException) { return null; }
    }

    private static int FirstBitDiff(float[] a, float[] b)
    {
        for (int i = 0; i < a.Length; i++)
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                return i;
        return -1;
    }

    /// <summary>
    /// Pin the Q8_KS activation-prepack gates (issue #410 int8 tier) OFF for the scope
    /// of a bitwise oracle: the int8 dots are argmax-stable, NOT byte-exact vs the f32
    /// per-token reference (on non-AVX-512 boxes SHARPI_Q4K_Q8K auto-enables and would
    /// fail the bit-parity assertion spuriously). Restores the previous values on dispose.
    /// </summary>
    private static IDisposable PinQ8KOff() => new EnvPin(
        ("SHARPI_Q3K_Q8K", "0"), ("SHARPI_Q8_0_Q8K", "0"), ("SHARPI_Q4K_Q8K", "0"));

    private sealed class EnvPin : IDisposable
    {
        private readonly (string Name, string? Prev)[] _saved;
        public EnvPin(params (string Name, string Value)[] pins)
        {
            _saved = new (string, string?)[pins.Length];
            for (int i = 0; i < pins.Length; i++)
            {
                _saved[i] = (pins[i].Name, Environment.GetEnvironmentVariable(pins[i].Name));
                Environment.SetEnvironmentVariable(pins[i].Name, pins[i].Value);
            }
        }
        public void Dispose()
        {
            foreach (var (name, prev) in _saved)
                Environment.SetEnvironmentVariable(name, prev);
        }
    }

    [Fact]
    public void BatchedCpuMoePrefill_BitwiseMatchesSequential_Coder()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCoderPath();
        if (path is null) return;

        bool prevBatched = CudaHybridForwardPass.BatchedPrefillEnabled;
        bool prevMoe = CudaHybridForwardPass.BatchedCpuMoePrefillEnabled;
        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        using var _q8k = PinQ8KOff();   // int8 tiers are argmax-stable, not byte-exact
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            int ctx = Math.Min(hp.ContextLength, 4096);
            var hw = HardwareProfile.Detect(gpu);
            var placement = TierPlanner.Plan(model, hp, hw, turboQuant: false, requestedCtxSize: ctx);

            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts! " +
                "The five boxing wizards jump quickly.");
            Assert.True(tokens.Count >= 8, $"Prompt tokenized to only {tokens.Count} tokens.");

            float[] Run(bool batchedMoe)
            {
                // Reference arm: everything per-token (the deterministic gold path).
                // Batched arm: batched trunk + batched CPU-MoE FFN stage.
                CudaHybridForwardPass.BatchedPrefillEnabled = batchedMoe;
                CudaHybridForwardPass.BatchedCpuMoePrefillEnabled = batchedMoe;
                using var fwd = TryConstruct(model, gpu, hp, placement);
                if (fwd is null) return Array.Empty<float>();
                var logits = fwd.Prefill(tokens).ToArray();
                if (batchedMoe)
                {
                    // Guard against a vacuous pass: the batched arm must actually run both
                    // the batched trunk AND the batched CPU-MoE FFN stage, else this compares
                    // the per-token loop against itself.
                    Assert.True(fwd.LastPrefillWasBatched,
                        "Batched arm fell back to the per-token trunk path — the parity check " +
                        $"would be vacuous. GpuLayers={placement.GpuLayers} CpuLayers={placement.CpuLayers}.");
                    Assert.True(fwd.LastPrefillUsedBatchedCpuMoe,
                        "Batched arm did not take the batched CPU-MoE FFN stage (per-token " +
                        "fallback) — the #410 parity check would be vacuous.");
                }
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
                $"Batched CPU-MoE prefill diverges from sequential at index {firstDiff} " +
                $"(seq={(firstDiff >= 0 ? seq[firstDiff] : 0)} bat={(firstDiff >= 0 ? bat[firstDiff] : 0)}). " +
                "BatchedCpuMoeFfnStage must be bit-identical to the per-token FFN/MoE loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
            _out.WriteLine($"OK N={tokens.Count} greedy={Sampler.Greedy(seq)}");
        }
        finally
        {
            CudaHybridForwardPass.BatchedPrefillEnabled = prevBatched;
            CudaHybridForwardPass.BatchedCpuMoePrefillEnabled = prevMoe;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }

    /// <summary>
    /// Multi-chunk parity (chunked admission shape, issue #183): two Prefill calls with
    /// startPos continuation must also be bit-identical to the sequential loop — the
    /// batched CPU-MoE stage is position-independent, but this pins the scratch resize
    /// (chunk sizes differ) and the KV/counter handoff between chunks.
    /// </summary>
    [Fact]
    public void BatchedCpuMoePrefill_MultiChunk_BitwiseMatchesSequential_Coder()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCoderPath();
        if (path is null) return;

        bool prevBatched = CudaHybridForwardPass.BatchedPrefillEnabled;
        bool prevMoe = CudaHybridForwardPass.BatchedCpuMoePrefillEnabled;
        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        using var _q8k = PinQ8KOff();   // int8 tiers are argmax-stable, not byte-exact
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            int ctx = Math.Min(hp.ContextLength, 4096);
            var hw = HardwareProfile.Detect(gpu);
            var placement = TierPlanner.Plan(model, hp, hw, turboQuant: false, requestedCtxSize: ctx);

            var tokens = tokenizer.Encode(
                "Sphinx of black quartz, judge my vow. Pack my box with five dozen liquor jugs. " +
                "The quick brown fox jumps over the lazy dog near the riverbank at dawn.");
            Assert.True(tokens.Count >= 12, $"Prompt tokenized to only {tokens.Count} tokens.");
            int split = tokens.Count / 2;
            var chunk1 = tokens.Take(split).ToList();
            var chunk2 = tokens.Skip(split).ToList();

            float[] Run(bool batchedMoe)
            {
                CudaHybridForwardPass.BatchedPrefillEnabled = batchedMoe;
                CudaHybridForwardPass.BatchedCpuMoePrefillEnabled = batchedMoe;
                using var fwd = TryConstruct(model, gpu, hp, placement);
                if (fwd is null) return Array.Empty<float>();
                fwd.Prefill(chunk1);
                var logits = fwd.Prefill(chunk2, split).ToArray();
                if (batchedMoe)
                    Assert.True(fwd.LastPrefillWasBatched && fwd.LastPrefillUsedBatchedCpuMoe,
                        "Batched arm fell back to a per-token path — the parity check would be vacuous.");
                return logits;
            }

            float[] seq = Run(false);
            float[] bat = Run(true);
            if (seq.Length == 0 || bat.Length == 0)
            {
                _out.WriteLine("SKIP: construction OOM'd on this box.");
                return;
            }

            int firstDiff = FirstBitDiff(seq, bat);
            Assert.True(firstDiff < 0,
                $"Multi-chunk batched CPU-MoE prefill diverges from sequential at index {firstDiff}.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
            _out.WriteLine($"OK chunks={split}+{tokens.Count - split} greedy={Sampler.Greedy(seq)}");
        }
        finally
        {
            CudaHybridForwardPass.BatchedPrefillEnabled = prevBatched;
            CudaHybridForwardPass.BatchedCpuMoePrefillEnabled = prevMoe;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }

    /// <summary>
    /// Issue #410 int8 tier: with the Q8_KS activation prepack forced ON
    /// (SHARPI_Q4K_Q8K=1 etc.), the batched CPU-MoE prefill routes the gate/up (and,
    /// dtype permitting, down) dots through the int-domain kernels. That path is
    /// argmax-stable, NOT byte-exact, vs the f32 batched path — assert the contract
    /// that actually holds: shared top-5 (≥4/5) and logits within a loose fp tolerance,
    /// the same contract the MMQ compute-routing oracles use.
    /// </summary>
    [Fact]
    public void BatchedCpuMoePrefill_Q8K_ArgmaxStableVsF32_Coder()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCoderPath();
        if (path is null) return;

        bool prevBatched = CudaHybridForwardPass.BatchedPrefillEnabled;
        bool prevMoe = CudaHybridForwardPass.BatchedCpuMoePrefillEnabled;
        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            int ctx = Math.Min(hp.ContextLength, 4096);
            var hw = HardwareProfile.Detect(gpu);
            var placement = TierPlanner.Plan(model, hp, hw, turboQuant: false, requestedCtxSize: ctx);

            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts! " +
                "The five boxing wizards jump quickly.");
            Assert.True(tokens.Count >= 8, $"Prompt tokenized to only {tokens.Count} tokens.");

            CudaHybridForwardPass.BatchedPrefillEnabled = true;
            CudaHybridForwardPass.BatchedCpuMoePrefillEnabled = true;

            float[] Run(string q8k)
            {
                using var pin = new EnvPin(
                    ("SHARPI_Q3K_Q8K", q8k), ("SHARPI_Q8_0_Q8K", q8k), ("SHARPI_Q4K_Q8K", q8k));
                using var fwd = TryConstruct(model, gpu, hp, placement);
                if (fwd is null) return Array.Empty<float>();
                var logits = fwd.Prefill(tokens).ToArray();
                Assert.True(fwd.LastPrefillWasBatched && fwd.LastPrefillUsedBatchedCpuMoe,
                    "Arm fell back to a per-token path — the comparison would be vacuous.");
                return logits;
            }

            float[] f32 = Run("0");
            float[] q8k = Run("1");
            if (f32.Length == 0 || q8k.Length == 0)
            {
                _out.WriteLine("SKIP: construction OOM'd on this box.");
                return;
            }

            Assert.Equal(f32.Length, q8k.Length);
            float maxAbs = 0f;
            for (int i = 0; i < f32.Length; i++)
                maxAbs = MathF.Max(maxAbs, MathF.Abs(f32[i] - q8k[i]));

            static HashSet<int> Top5(float[] v)
            {
                var idx = new int[v.Length];
                for (int i = 0; i < idx.Length; i++) idx[i] = i;
                Array.Sort(idx, (a, b) => v[b].CompareTo(v[a]));
                var set = new HashSet<int>();
                for (int i = 0; i < 5 && i < idx.Length; i++) set.Add(idx[i]);
                return set;
            }
            var f32Top = Top5(f32);
            var q8kTop = Top5(q8k);
            int overlap = 0;
            foreach (var t in q8kTop) if (f32Top.Contains(t)) overlap++;
            Assert.True(overlap >= 4,
                $"Q8_KS batched top-5 overlaps the f32 batched path in only {overlap}/5 slots (maxAbs={maxAbs:E2}).");
            Assert.True(maxAbs < 3.0f, $"Q8_KS batched logits deviate too far from f32 (maxAbs={maxAbs:E2}).");
            _out.WriteLine($"OK top5-overlap={overlap}/5 maxAbs={maxAbs:E2}");
        }
        finally
        {
            CudaHybridForwardPass.BatchedPrefillEnabled = prevBatched;
            CudaHybridForwardPass.BatchedCpuMoePrefillEnabled = prevMoe;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }
}
