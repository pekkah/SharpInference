using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Cross-backend correctness coverage for the CPU-MoE execution mode of
/// <see cref="CudaHybridForwardPass"/> (SHARPI_CPU_MOE=1). In that mode the GPU runs
/// attention/norms (+ shared expert if present) for the trunk layers, the post-FFN-norm
/// hidden is downloaded to the CPU, the MoE router + routed experts run on the CPU via
/// SimdKernels (shared with the CPU-tail path through <c>CpuMoeRouted</c>), and the
/// result is uploaded back to the GPU.
///
/// Parity philosophy: we do NOT assert strict greedy/argmax equality. Q4_K matvec
/// quantizes activations to Q8_1, and MoE router top-K is sensitive to that drift — the
/// GPU-attention upstream can legitimately flip an expert near the K/K+1 boundary even
/// when everything is wired correctly (see <see cref="CudaMoeTests"/>). Instead, per the
/// project's cross-backend parity guidance, we use the CPU <see cref="ForwardPass"/> as
/// the INDEPENDENT reference and assert at the logit level via top-5 vocab overlap. On a
/// short in-distribution prompt a correctly-wired CPU-MoE path overlaps heavily with the
/// pure-CPU reference; an overlap collapse signals a real wiring bug, not quant noise.
/// </summary>
public sealed class CudaHybridForwardPassCpuMoeTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FindMoEModelPath()
    {
        // Smallest first — OLMoE is the fast, no-shared-expert MoE preferred for this test.
        string[] candidates =
        {
            "models\\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
            "models\\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        };
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var c in candidates)
            {
                var p = Path.Combine(dir, c);
                if (File.Exists(p)) return p;
            }
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

    private static int[] TopK(float[] logits, int k)
    {
        var idx = new int[logits.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => logits[b].CompareTo(logits[a]));
        var top = new int[k];
        Array.Copy(idx, top, k);
        return top;
    }

    /// <summary>
    /// Runs the CPU-MoE hybrid (GPU attention + CPU routed experts) against the
    /// independent pure-CPU <see cref="ForwardPass"/> on a short coherent prompt.
    /// Asserts the hybrid output is well-formed (finite, non-degenerate range, ≥2
    /// distinct argmax over a 5-token greedy decode) and that the hybrid's top-5
    /// vocab logits overlap the CPU reference's top-5 by ≥3. Strict argmax parity is
    /// deliberately NOT required: Q4_K activation quant can flip the router top-K near
    /// the boundary; top-5 overlap validates the CPU-MoE wiring without flakiness.
    /// </summary>
    [Fact]
    public void CudaHybridForwardPass_CpuMoeMode_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return; // silent skip — no CUDA

        var path = FindMoEModelPath();
        if (path is null) return; // silent skip — no MoE model on disk

        var prev = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            Assert.True(hp.IsMoE, "Expected hp.IsMoE for the MoE model under test.");

            var tokenizer = GgufTokenizer.FromGgufModel(model);

            // GpuLayers = NumLayers is the "user wants GPU" hint; with SHARPI_CPU_MOE=1
            // the routed-expert FFN is forced onto the CPU regardless of the cut point.
            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers,
                CpuLayers: 0,
                GpuWeightBytes: 0,
                GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 2048));

            using var hybrid = new CudaHybridForwardPass(model, gpu, hp, placement);

            // Independent reference: pure-CPU forward pass over the same model handle.
            using var cpuBackend = new CpuBackend();
            using var cpu = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp);

            var prompt = "The capital of France is";
            var tokens = tokenizer.Encode(prompt);
            Assert.NotEmpty(tokens);

            var hybridLogits = hybrid.Prefill(tokens).ToArray();
            var cpuLogits = cpu.Prefill(tokens).ToArray();
            Assert.Equal(cpuLogits.Length, hybridLogits.Length);

            // Well-formedness on the hybrid prefill logits.
            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < hybridLogits.Length; i++)
            {
                float v = hybridLogits[i];
                Assert.True(float.IsFinite(v), $"Non-finite hybrid logit at vocab idx {i}: {v}");
                if (v < min) min = v;
                if (v > max) max = v;
            }
            Assert.True(max - min > 0.1f,
                $"Hybrid logit range too tight ({min:F3}..{max:F3}); the CPU-MoE FFN is " +
                "likely returning a near-zero hidden state.");

            // Anti-garble: greedy decode 5 tokens on the hybrid; ≥2 distinct argmaxes.
            ReadOnlySpan<float> dl = hybridLogits;
            var decoded = new List<int>(5);
            for (int i = 0; i < 5; i++)
            {
                int next = Sampler.Greedy(dl);
                decoded.Add(next);
                dl = hybrid.Forward(next, tokens.Count + i);
            }
            int distinct = decoded.Distinct().Count();
            Assert.True(distinct >= 2,
                $"Hybrid greedy decode produced only {distinct} distinct token(s) across 5 steps " +
                $"({string.Join(",", decoded)}); the CPU-MoE FFN may be stuck in a degenerate loop.");

            // Cross-backend top-5 overlap against the independent CPU reference.
            int hybridArgmax = Argmax(hybridLogits);
            int cpuArgmax = Argmax(cpuLogits);
            int[] hybridTop5 = TopK(hybridLogits, 5);
            int[] cpuTop5 = TopK(cpuLogits, 5);
            int overlap = 0;
            foreach (var t in hybridTop5)
                if (Array.IndexOf(cpuTop5, t) >= 0) overlap++;

            float maxAbs = 0f;
            for (int i = 0; i < cpuLogits.Length; i++)
                maxAbs = Math.Max(maxAbs, Math.Abs(cpuLogits[i] - hybridLogits[i]));

            // Diagnostic line (visible with `dotnet test --logger "console;verbosity=detailed"`).
            Console.WriteLine(
                $"[CpuMoeMode] overlap={overlap}/5 top1Match={hybridArgmax == cpuArgmax} " +
                $"maxAbsLogitDiff={maxAbs:F4} hybridArgmax={hybridArgmax} cpuArgmax={cpuArgmax} " +
                $"hybridTop5=[{string.Join(",", hybridTop5)}] cpuTop5=[{string.Join(",", cpuTop5)}]");

            Assert.True(overlap >= 3,
                $"CPU-MoE hybrid top-5 overlaps the CPU reference by only {overlap}/5 " +
                $"(maxAbsLogitDiff={maxAbs:F4}, top1Match={hybridArgmax == cpuArgmax}): " +
                $"hybrid top5=[{string.Join(",", hybridTop5)}] (argmax {hybridArgmax}), " +
                $"cpu top5=[{string.Join(",", cpuTop5)}] (argmax {cpuArgmax}). " +
                "An overlap collapse on an in-distribution prompt signals a real CPU-MoE " +
                "wiring bug (router/expert weight loading or the upload/download round-trip), " +
                "not the documented Q4_K router-boundary drift.");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prev);
        }
    }
}
