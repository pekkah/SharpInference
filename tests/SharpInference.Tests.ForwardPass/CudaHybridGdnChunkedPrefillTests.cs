using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Parity guard for the opt-in FlashQLA chunked GDN prefill on the CUDA hybrid path
/// (<see cref="CudaHybridGdnForwardPass.GdnChunkedPrefillOverride"/>). Inside the
/// batched-prefill fast path, the chunk-parallel <c>chunk_gated_delta_rule</c> kernel
/// (<see cref="CudaBackend.GdnChunkedPrefill"/>) replaces the sequential
/// <c>GdnRecurrenceScan</c>. Unlike the other batched-prefill oracles
/// (<see cref="CudaHybridGdnBatchedPrefillTests"/>) this one is NOT bit-identical:
/// the chunked form resolves the same recurrence with a different floating-point
/// reduction order, so the contract is <b>argmax-identical and numerically close</b>,
/// not bitwise. The byte-exact scan remains the default and the bitwise oracles keep
/// validating it; this test only guards the host wiring of the chunked drop-in
/// (strides/offsets/state-carry) at the model level.
///
/// Skipped silently when CUDA is unavailable or the model isn't on disk.
/// </summary>
public sealed class CudaHybridGdnChunkedPrefillTests : IDisposable
{
    // Isolate the variable under test: pin the trunk projection matmuls byte-exact
    // (GdnPrefillComputeEnabled = false, plus RawQ80WeightsEnabled = false so Q8_0 trunk
    // weights stay on the F32 matvec rather than the int8 dp4a/MMQ path) so the ONLY
    // difference between the two arms is the GDN recurrence kernel (scan vs chunked), not
    // the matmul choice (which is argmax-stable, not byte-exact). Restored in Dispose.
    private readonly bool _prevGdnCompute = CudaHybridGdnForwardPass.GdnPrefillComputeEnabled;
    private readonly bool _prevRawQ80 = CudaHybridGdnForwardPass.RawQ80WeightsEnabled;
    public CudaHybridGdnChunkedPrefillTests()
    {
        CudaHybridGdnForwardPass.GdnPrefillComputeEnabled = false;
        CudaHybridGdnForwardPass.RawQ80WeightsEnabled = false;
    }
    public void Dispose()
    {
        CudaHybridGdnForwardPass.GdnPrefillComputeEnabled = _prevGdnCompute;
        CudaHybridGdnForwardPass.RawQ80WeightsEnabled = _prevRawQ80;
    }

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FindMoePath()
    {
        string[] candidates =
        {
            @"E:\models\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf",
            @"E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf",
        };
        foreach (var p in candidates)
            if (File.Exists(p)) return p;
        return null;
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
            if (sb.Length > 200_000)
                throw new InvalidOperationException("Tokenizer not packing enough tokens.");
        }
    }

    /// <summary>
    /// The chunked-prefill GDN recurrence (<c>GdnChunkedPrefillOverride = true</c>) must
    /// produce <b>argmax-identical, finite</b> final-token logits versus the sequential
    /// scan (<c>= false</c>, the default). Both arms run under batched prefill + trunk +
    /// GDN-scan (the chunked kernel lives inside the <c>BatchedGdnScanEnabled</c> fast
    /// path); a &gt;64-token prompt forces the GPU kernel's multi-chunk state carry.
    ///
    /// <para>Why argmax + finiteness, NOT a tight per-logit tolerance like the other
    /// batched-prefill oracles: the chunked <c>chunk_gated_delta_rule</c> form resolves
    /// the same recurrence with a different FP reduction order, so it is NOT bit-identical
    /// to the scan (the kernel-level parity vs the CPU double reference —
    /// <see cref="CudaGdnKernelsTests.GdnChunkedPrefill_ModelStridesMultiChunk_MatchesCpuReference"/>
    /// — is the numeric guard, ~1e-5 max error). On an MoE model that ~1e-5 GDN delta can
    /// nudge a token across an expert-selection boundary and flip which top-k experts it
    /// routes to — a discrete change that shifts individual logits by O(1) while leaving
    /// the prediction (argmax) intact. A tight per-logit bound would be flaky on exactly
    /// the property (non-bit-exactness) this kernel is opted into for. Gross wiring errors
    /// (wrong stride/offset → garbage V, NaNs) still trip the argmax and finiteness
    /// checks. The strided multi-chunk kernel test is the rigorous numeric oracle.</para>
    /// </summary>
    [Fact]
    public void ChunkedPrefill_ArgmaxMatchesScan_Moe()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindMoePath();
        if (path is null) return;

        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        bool prevPrefill = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        bool prevTrunk = CudaHybridGdnForwardPass.BatchedTrunkEnabled;
        bool prevScan = CudaHybridGdnForwardPass.BatchedGdnScanEnabled;
        bool? prevChunked = CudaHybridGdnForwardPass.GdnChunkedPrefillOverride;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (!hp.IsMoE) return; // exercise the CPU-MoE GDN-hybrid prefill trunk
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));

            // > 64 tokens so the GPU chunked kernel's multi-chunk (GDN_CHUNK=64) state
            // carry is exercised, not just a single-chunk resolve.
            var tokens = LongPrompt(tokenizer, 96);
            Assert.True(tokens.Count > 64, $"Prompt only {tokens.Count} tokens; need >64 for multi-chunk.");

            // Hold the batched-prefill fast path on; toggle only scan vs chunked.
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = true;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = true;
            CudaHybridGdnForwardPass.BatchedGdnScanEnabled = true;

            float[] RunWith(bool chunked)
            {
                CudaHybridGdnForwardPass.GdnChunkedPrefillOverride = chunked;
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                return fwd.Prefill(tokens).ToArray();
            }

            float[] scan = RunWith(false);   // sequential GdnRecurrenceScan (reference)
            float[] chunk = RunWith(true);    // FlashQLA chunk_gated_delta_rule

            Assert.Equal(scan.Length, chunk.Length);

            // Finiteness: a gross stride/offset/state-carry bug surfaces as NaN/Inf here
            // (e.g. reading the conv-stream Q/K region as V) regardless of argmax.
            for (int i = 0; i < chunk.Length; i++)
            {
                Assert.True(float.IsFinite(scan[i]), $"scan logit non-finite at {i}: {scan[i]}");
                Assert.True(float.IsFinite(chunk[i]), $"chunked logit non-finite at {i}: {chunk[i]}");
            }

            // Load-bearing semantic check: the chunked drop-in must not change the
            // prediction. (Per-logit numeric parity is intentionally NOT asserted here —
            // see the class/method docs on MoE routing-flip amplification of the
            // non-bit-exact GDN delta; the strided kernel test is the numeric oracle.)
            Assert.Equal(Sampler.Greedy(scan), Sampler.Greedy(chunk));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevPrefill;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = prevTrunk;
            CudaHybridGdnForwardPass.BatchedGdnScanEnabled = prevScan;
            CudaHybridGdnForwardPass.GdnChunkedPrefillOverride = prevChunked;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }
}
