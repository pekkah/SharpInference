using System.Diagnostics;
using SharpInference.Cpu;

namespace SharpInference.Bench;

/// <summary>
/// Manual timing harness for the Gated DeltaNet CPU prefill kernels: compares the
/// per-token sequential scan (<see cref="GdnKernels.GdnRecurrencePrefill"/>) against
/// the chunk-parallel "chunk_gated_delta_rule" form
/// (<see cref="GdnKernels.GdnRecurrenceChunkedPrefill"/>) at realistic GDN dims.
/// Run with: <c>dotnet run -c Release -- --gdn</c>.
/// </summary>
public static class GdnPrefillHarness
{
    public static void Run(string[] args)
    {
        // Qwen3.6-35B-A3B GDN block: 32 v-heads × 128 head-dim.
        const int hv = 32;
        const int d = 128;
        int[] tokenCounts = [256, 512, 1024];
        int chunkSize = GdnKernels.DefaultGdnChunkSize;

        Console.WriteLine($"GDN CPU prefill: hv={hv} d={d} chunk={chunkSize}");
        Console.WriteLine($"{"tokens",8} {"seq ms",10} {"chunk ms",10} {"speedup",9} {"max|Δ|",10}");

        var rng = new Random(12345);
        foreach (int tokens in tokenCounts)
        {
            float[] q = Rand(rng, tokens * hv * d, -0.5f, 0.5f);
            float[] k = Rand(rng, tokens * hv * d, -0.5f, 0.5f);
            float[] v = Rand(rng, tokens * hv * d, -0.5f, 0.5f);
            float[] alpha = Rand(rng, tokens * hv, -1f, 1f);
            float[] beta = Rand(rng, tokens * hv, -1f, 1f);
            float[] ssmA = Rand(rng, hv, -0.5f, -0.05f);
            float[] dtBias = Rand(rng, hv, -0.2f, 0.2f);
            float[] normW = Rand(rng, d, 0.5f, 1.5f);
            float[] z = Rand(rng, tokens * hv * d, -1f, 1f);

            float[] outSeq = new float[tokens * hv * d];
            float[] outChunk = new float[tokens * hv * d];

            Action seq = () =>
            {
                float[] s = new float[hv * d * d];
                GdnKernels.GdnRecurrencePrefill(tokens, q, k, v, alpha, beta, ssmA, dtBias, normW, z,
                    s, outSeq, hv, d);
            };
            Action chunk = () =>
            {
                float[] s = new float[hv * d * d];
                GdnKernels.GdnRecurrenceChunkedPrefill(tokens, q, k, v, alpha, beta, ssmA, dtBias, normW, z,
                    s, outChunk, hv, d, chunkSize: chunkSize);
            };

            double seqMs = TimeBest(seq);
            double chunkMs = TimeBest(chunk);

            float maxDiff = 0f;
            for (int i = 0; i < outSeq.Length; i++)
                maxDiff = MathF.Max(maxDiff, MathF.Abs(outSeq[i] - outChunk[i]));

            Console.WriteLine(
                $"{tokens,8} {seqMs,10:F3} {chunkMs,10:F3} {seqMs / chunkMs,8:F2}x {maxDiff,10:E2}");
        }
    }

    private static double TimeBest(Action a)
    {
        for (int w = 0; w < 2; w++) a();   // warm up + JIT
        double best = double.MaxValue;
        var sw = new Stopwatch();
        for (int rep = 0; rep < 5; rep++)
        {
            sw.Restart();
            a();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    private static float[] Rand(Random rng, int n, float lo, float hi)
    {
        float[] arr = new float[n];
        for (int i = 0; i < n; i++) arr[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return arr;
    }
}
