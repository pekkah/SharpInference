using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using Xunit;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #219: GPU-side greedy argmax. Two layers of coverage:
///   1. Kernel bit-exact parity (model-free) — <see cref="CudaBackend.Argmax"/> must match a
///      left-to-right strict-<c>&gt;</c> host scan (the <c>Sampler.Greedy</c> contract: highest
///      value wins, lowest index on an exact tie), including the winning value bit-for-bit.
///   2. Decode equivalence (model-gated) — greedy decode via <c>ForwardArgmax</c> emits the exact
///      same token stream as the legacy <c>Forward</c> + host argmax, on both the dense
///      (<c>Forward</c>) and Gemma-4 (<c>ForwardGemma4</c>) paths.
/// All tests silently no-op on hosts without CUDA (or, for the gated tests, without the model).
/// </summary>
public sealed class CudaGpuArgmaxParityTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? FindModel(params string[] candidates)
    {
        foreach (var c in candidates)
            if (File.Exists(c)) return c;
        return null;
    }

    // Host reference == Sampler.Greedy's scan (strict >, lowest-index tie-break) + the winning value.
    private static (int idx, float val) CpuArgmax(ReadOnlySpan<float> logits)
    {
        int idx = 0;
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) { max = logits[i]; idx = i; }
        return (idx, max);
    }

    private static void AssertArgmaxMatches(CudaBackend gpu, float[] logits)
    {
        var g = gpu.Upload(logits, TensorShape.D1(logits.Length));
        try
        {
            var (gIdx, gVal) = gpu.Argmax(g);
            var (cIdx, cVal) = CpuArgmax(logits);
            Assert.Equal(cIdx, gIdx);
            // The kernel only compares/forwards the original float values — no arithmetic — so the
            // winning value must be bit-identical to the host's, and equal to logits[idx].
            Assert.Equal(BitConverter.SingleToInt32Bits(cVal), BitConverter.SingleToInt32Bits(gVal));
            Assert.Equal(BitConverter.SingleToInt32Bits(logits[gIdx]), BitConverter.SingleToInt32Bits(gVal));
        }
        finally { gpu.Free(g); }
    }

    [Fact]
    public void Argmax_MaxAtVariousPositions_AndSizes()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Span block boundaries (256 threads × 256 blocks) and real vocab sizes (Qwen3 152064,
        // Gemma 4 262144) so the multi-block grid-stride + two-pass reduction is exercised.
        foreach (int n in new[] { 1, 2, 3, 255, 256, 257, 1000, 65536, 152064, 262144 })
        {
            var rng = new Random(1234 + n);
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 20 - 10);
            AssertArgmaxMatches(gpu, a);                       // random — max anywhere
            foreach (int pos in new[] { 0, n / 2, n - 1 })
            {
                var b = (float[])a.Clone();
                b[pos] = 1000f + pos;                          // a unique clear max at pos
                AssertArgmaxMatches(gpu, b);
            }
        }
    }

    [Fact]
    public void Argmax_TieBreak_LowestIndexWins()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var a = new float[5000];
        Array.Fill(a, -1f);
        a[700] = 5f;
        a[300] = 5f;   // equal maxima — index 300 (lower) must win, matching strict-> host scan
        AssertArgmaxMatches(gpu, a);
        var g = gpu.Upload(a, TensorShape.D1(a.Length));
        try { Assert.Equal(300, gpu.Argmax(g).Index); }
        finally { gpu.Free(g); }
    }

    [Fact]
    public void Argmax_AllEqual_ReturnsIndexZero()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var a = new float[8192];
        Array.Fill(a, 3.5f);
        AssertArgmaxMatches(gpu, a);   // no element beats another -> index 0
    }

    [Fact]
    public void Argmax_AllNegative()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var a = new float[4096];
        var rng = new Random(99);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(-rng.NextDouble() * 50 - 1);
        AssertArgmaxMatches(gpu, a);
    }

    [Fact]
    public void Argmax_KernelIndependentOfEngineKillSwitch()
    {
        // The SHARPI_GPU_ARGMAX gate only decides whether the engine takes the fast path; the
        // kernel itself is always correct. Toggling the backend flag must not change Argmax's result.
        using var gpu = TryCreate();
        if (gpu is null) return;
        var a = new float[2048];
        var rng = new Random(7);
        for (int i = 0; i < a.Length; i++) a[i] = (float)rng.NextDouble();
        a[123] = 9f;

        bool prev = gpu.GpuArgmaxEnabled;
        try
        {
            gpu.GpuArgmaxEnabled = false; AssertArgmaxMatches(gpu, a);
            gpu.GpuArgmaxEnabled = true;  AssertArgmaxMatches(gpu, a);
        }
        finally { gpu.GpuArgmaxEnabled = prev; }
    }

    [Fact]
    public void ArgmaxRows_BatchedParity()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Packed [rows × vocab] verify logits — each row reduced independently. Cover a few row
        // counts (k for MTP verify) and a real vocab, with a distinct max placed per row.
        foreach (int vocab in new[] { 1024, 152064 })
        foreach (int rows in new[] { 1, 2, 4, 8 })
        {
            var rng = new Random(555 + rows * 31 + vocab);
            var flat = new float[rows * vocab];
            for (int i = 0; i < flat.Length; i++) flat[i] = (float)(rng.NextDouble() * 10 - 5);
            // Put a unique, clear max in each row at a row-dependent position (incl. ties within a row).
            for (int r = 0; r < rows; r++)
            {
                int pos = (r * 37) % vocab;
                flat[r * vocab + pos] = 50f + r;
                if (vocab > 100) flat[r * vocab + (pos + 50) % vocab] = 50f + r; // tie -> lower index wins
            }

            var g = gpu.Upload(flat, TensorShape.D1(flat.Length));
            try
            {
                var rowsRes = gpu.ArgmaxRows(g, rows, vocab, vocab);
                Assert.Equal(rows, rowsRes.Length);
                for (int r = 0; r < rows; r++)
                {
                    var (cIdx, cVal) = CpuArgmax(flat.AsSpan(r * vocab, vocab));
                    Assert.Equal(cIdx, rowsRes[r].Index);
                    Assert.Equal(BitConverter.SingleToInt32Bits(cVal),
                                 BitConverter.SingleToInt32Bits(rowsRes[r].Value));
                }
            }
            finally { gpu.Free(g); }
        }
    }

    // ── Decode equivalence (model-gated) ──────────────────────────────────────

    private static void AssertDecodeEquivalence(string path, int steps = 24)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        int ctx = Math.Min(hp.ContextLength, 512);
        var tokens = tokenizer.Encode("The quick brown fox jumps over the lazy dog.").ToArray();

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: ctx);
        Assert.True(fwd.SupportsGpuArgmax, "CudaForwardPass should support GPU argmax by default.");

        // Reference: legacy full-download Forward + host Sampler.Greedy.
        var refTokens = new int[steps];
        var logits = fwd.Prefill(tokens).ToArray();
        int next = Sampler.Greedy(logits);
        for (int i = 0; i < steps; i++)
        {
            refTokens[i] = next;
            logits = fwd.Forward(next, tokens.Length + i).ToArray();
            next = Sampler.Greedy(logits);
        }

        // Candidate: the on-device argmax fast path. Fresh cache, same prompt.
        fwd.ResetCache();
        var gpuTokens = new int[steps];
        var pl = fwd.Prefill(tokens).ToArray();
        next = Sampler.Greedy(pl);                     // first token still comes from prefill logits
        for (int i = 0; i < steps; i++)
        {
            gpuTokens[i] = next;
            var (tok, val) = fwd.ForwardArgmax(next, tokens.Length + i);
            next = tok;
        }

        Assert.Equal(refTokens, gpuTokens);
    }

    [Fact]
    public void Decode_DensePath_MatchesHostArgmax()   // SmolLM2 / Qwen3 -> Forward
    {
        var path = FindModel(
            @"C:\p\sharpi\models\SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            @"E:\models\SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            @"C:\p\sharpi\models\Qwen3-8B-Q4_K_M.gguf",
            @"E:\models\Qwen3-8B-Q4_K_M.gguf");
        if (path is null) return;
        AssertDecodeEquivalence(path);
    }

    [Fact]
    public void Decode_Gemma4Path_MatchesHostArgmax()   // Gemma 4 E4B -> ForwardGemma4 (+ softcap)
    {
        var path = FindModel(
            @"E:\models\gemma-4-E4B-it-Q8_0.gguf",
            @"C:\p\sharpi\models\gemma-4-E4B-it-Q8_0.gguf");
        if (path is null) return;
        AssertDecodeEquivalence(path);
    }
}
