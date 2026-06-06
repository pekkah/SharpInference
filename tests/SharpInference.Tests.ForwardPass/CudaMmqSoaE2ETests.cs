using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #149: end-to-end check that repacking the Gemma 4 Q8_0 weights into the SoA
/// layout (<c>mmqSoa: true</c>) produces <b>bit-identical</b> logits to the interleaved
/// layout. The SoA prefill MMQ and decode dp4a kernels are each bit-identical to their
/// AoS counterparts (per <see cref="CudaMmqSoaTests"/>), so the whole forward pass —
/// batched prefill (MMQ) + greedy decode (dp4a) — must match to the bit. Two 8 GB
/// instances cannot co-reside, so this runs SoA-off, disposes, then SoA-on, and compares.
///
/// Silent no-op without CUDA or the GGUF.
/// </summary>
public sealed class CudaMmqSoaE2ETests
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
        return null;
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int best = 0; float bv = v[0];
        for (int i = 1; i < v.Length; i++) if (v[i] > bv) { bv = v[i]; best = i; }
        return best;
    }

    // Prefill the prompt, then greedily decode `nDecode` tokens, returning the logit
    // array captured at prefill and at each decode step.
    private static List<float[]> RunForward(GgufModel model, CudaBackend gpu, ModelHyperparams hp, bool soa,
                                            int[] tokens, int nDecode)
    {
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512, mmqSoa: soa);
        var captures = new List<float[]>();

        var logits = fwd.Prefill(tokens).ToArray();
        captures.Add(logits);
        int pos = tokens.Length;
        for (int i = 0; i < nDecode; i++)
        {
            int next = Argmax(logits);
            logits = fwd.Forward(next, pos).ToArray();
            captures.Add(logits);
            pos++;
        }
        return captures;
    }

    [Fact]
    public void Gemma4_E4B_MmqSoa_BitIdenticalToInterleaved()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        var tokens = new int[] { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222, 333, 444 };
        const int nDecode = 4;

        // Interleaved (default) first, then SoA — one instance at a time (8 GB each).
        var refRun = RunForward(model, gpu, hp, soa: false, tokens, nDecode);
        var soaRun = RunForward(model, gpu, hp, soa: true, tokens, nDecode);

        Assert.Equal(refRun.Count, soaRun.Count);
        float maxAbs = 0f;
        for (int step = 0; step < refRun.Count; step++)
        {
            var a = refRun[step];
            var b = soaRun[step];
            Assert.Equal(a.Length, b.Length);
            for (int i = 0; i < a.Length; i++)
                maxAbs = MathF.Max(maxAbs, MathF.Abs(a[i] - b[i]));
        }
        Console.WriteLine($"Gemma4 MMQ-SoA e2e: {refRun.Count} steps, maxAbs(SoA−AoS)={maxAbs:E3}");
        Assert.True(maxAbs == 0f,
            $"SoA weight layout changed the Gemma 4 forward output (expected bit-identical): maxAbs={maxAbs:E3}.");
    }
}
