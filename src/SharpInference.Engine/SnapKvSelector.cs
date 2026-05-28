using System.Runtime.InteropServices;
using SharpInference.Cpu;

namespace SharpInference.Engine;

/// <summary>
/// SnapKV (arXiv:2404.14469) prefill-time KV-cache eviction policy.
///
/// After prefill, the K-cache for a long prompt is the dominant memory consumer
/// at long context — at 16K tokens on a 32-head Qwen3-class model the cache is
/// ~2 GiB, which is most of a 12 GB card. SnapKV exploits the observation that
/// the last <c>W</c> query tokens of the prompt are a good predictor of which
/// prompt positions matter to decode: it computes per-head attention from those
/// queries over the prompt's K cache, pools the resulting weight mass, and
/// keeps the top-K positions plus a fixed recency window.
///
/// This class produces the keep-set; <see cref="PagedKvCache.Compact"/> applies
/// it. The two together are prefill-only — decode is untouched.
///
/// See issue #51. For the v1 path we pool per-layer scores into a single global
/// keep-set (uniform across layers) because the underlying <see cref="PagedKvCache"/>
/// shares its block table across layers; per-layer eviction is queued as a
/// follow-up. Empirically, attention sparsity patterns correlate strongly across
/// layers (Liu et al. observed &gt;80% overlap on the top-256 positions between
/// adjacent layers), so the accuracy loss vs per-layer is modest.
/// </summary>
public sealed unsafe class SnapKvSelector
{
    /// <summary>Default window of recent queries used as the importance probe.</summary>
    public const int DefaultWindow = 64;

    /// <summary>Default count of trailing positions to always retain (recency window).</summary>
    public const int DefaultRecency = 64;

    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _headsPerKvGroup;
    private readonly int _kvDim;

    // Score accumulator across (layer × head × query) — one float per prompt
    // position. After AccumulateLayer(...) has been called for every layer,
    // SelectKeepSet(...) reads this and emits the keep indices.
    private float[] _scoreAccum;

    public SnapKvSelector(int numHeads, int numKvHeads, int headDim)
    {
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _headsPerKvGroup = numHeads / numKvHeads;
        _kvDim = numKvHeads * headDim;
        _scoreAccum = Array.Empty<float>();
    }

    /// <summary>Reset the score accumulator for a new prefill.</summary>
    public void Reset(int promptLen)
    {
        if (_scoreAccum.Length < promptLen)
            _scoreAccum = new float[promptLen];
        else
            Array.Clear(_scoreAccum, 0, promptLen);
    }

    /// <summary>
    /// Pool the layer's last-W query attention into the global score accumulator.
    /// <paramref name="batchQ"/> contains the layer's RoPE'd queries laid out
    /// <c>[N, numHeads*headDim]</c>; <paramref name="kPagedCache"/> holds the
    /// (also-RoPE'd) per-position K vectors. Causal masking is applied — a
    /// query at position q only scores against keys at positions ≤ q.
    /// </summary>
    /// <remarks>
    /// Per the SnapKV paper, scores are the post-softmax attention weights, not
    /// the raw dot products. That ratifies the "which positions does this query
    /// actually look at" reading rather than just "which positions have a high
    /// projection onto this query". We mirror that here.
    /// </remarks>
    public void AccumulateLayer(float* batchQ, int N, PagedKvCache cache, int layer,
                                int startPos, int window)
    {
        // Window-of-queries: last min(window, N) tokens act as the importance
        // probe. Indices into batchQ are [N - window .. N - 1].
        int W = Math.Min(window, N);
        int wStart = N - W;

        // Score buffer per query; sized to the prompt length so a causal-masked
        // softmax can run in place without further alloc.
        int promptLen = N;
        var scratch = stackalloc float[Math.Min(promptLen, 8192)];
        bool useStack = promptLen <= 8192;
        float* scoreBuf = useStack ? scratch : (float*)NativeMemory.Alloc((nuint)(promptLen * sizeof(float)));
        try
        {
            float scale = 1.0f / MathF.Sqrt(_headDim);
            for (int w = wStart; w < N; w++)
            {
                int qAbsPos = startPos + w;
                float* qVec = batchQ + (long)w * _numHeads * _headDim;

                for (int h = 0; h < _numHeads; h++)
                {
                    int kvHead = h / _headsPerKvGroup;
                    float* qHead = qVec + h * _headDim;

                    // 1. Compute causal-masked dot scores for this (query, head).
                    for (int p = 0; p < promptLen; p++)
                    {
                        int absPos = startPos + p;
                        if (absPos > qAbsPos)
                        {
                            scoreBuf[p] = float.NegativeInfinity;
                            continue;
                        }
                        float* kVec = cache.KeyAt(layer, p) + kvHead * _headDim;
                        scoreBuf[p] = SimdKernels.DotF32(qHead, kVec, _headDim) * scale;
                    }

                    // 2. Softmax in place (over only the valid prefix; the
                    //    causal-masked tail is -inf so it contributes 0).
                    SimdKernels.SoftmaxInPlace(scoreBuf, promptLen);

                    // 3. Accumulate per-position attention weight into the global
                    //    score. Pool = sum across (queries, heads, layers).
                    for (int p = 0; p < promptLen; p++) _scoreAccum[p] += scoreBuf[p];
                }
            }
        }
        finally
        {
            if (!useStack) NativeMemory.Free((void*)scoreBuf);
        }
    }

    /// <summary>
    /// Emit the keep set: union of the top-(<paramref name="budget"/> - <paramref name="recency"/>)
    /// non-recency positions by accumulated score and the trailing <paramref name="recency"/>
    /// positions. Result is sorted ascending and contains every kept position in
    /// <c>[0, promptLen)</c>.
    /// </summary>
    public int[] SelectKeepSet(int promptLen, int budget, int recency)
    {
        if (budget >= promptLen) return Identity(promptLen);
        recency = Math.Min(recency, budget);
        recency = Math.Min(recency, promptLen);

        // Recency window: always kept.
        int recencyStart = promptLen - recency;

        // From the non-recency prefix, pick the top-(budget - recency) by score.
        int pickFromPrefix = budget - recency;
        var prefixIndices = new int[recencyStart];
        for (int i = 0; i < recencyStart; i++) prefixIndices[i] = i;

        if (pickFromPrefix > 0 && pickFromPrefix < recencyStart)
        {
            // Partial sort: top-K by descending score, ties broken by position
            // (lower-index wins — biases towards the system-prompt portion of
            // the prefix, which is usually the right call for instruction-tuned
            // models). Array.Sort with a custom comparer is O(N log N); for a
            // ≤16K prompt with ≤2K budget this is microseconds, no need to use
            // a heap.
            var sorted = (int[])prefixIndices.Clone();
            var localScores = _scoreAccum;
            Array.Sort(sorted, (a, b) =>
            {
                float sa = localScores[a], sb = localScores[b];
                if (sa != sb) return sb.CompareTo(sa);   // descending score
                return a.CompareTo(b);                   // ascending position
            });
            Array.Resize(ref sorted, pickFromPrefix);
            Array.Sort(sorted);
            prefixIndices = sorted;
        }
        else if (pickFromPrefix <= 0)
        {
            prefixIndices = Array.Empty<int>();
        }
        // else pickFromPrefix >= recencyStart → keep them all (already identity).

        var keep = new int[prefixIndices.Length + recency];
        Array.Copy(prefixIndices, keep, prefixIndices.Length);
        for (int i = 0; i < recency; i++) keep[prefixIndices.Length + i] = recencyStart + i;
        return keep;
    }

    private static int[] Identity(int n)
    {
        var r = new int[n];
        for (int i = 0; i < n; i++) r[i] = i;
        return r;
    }
}

/// <summary>
/// SnapKV runtime configuration (env-backed). One read at construction time per
/// forward pass; never re-checked during decode.
/// </summary>
public readonly record struct SnapKvConfig(int Budget, int Window, int Recency)
{
    public bool Enabled => Budget > 0;

    /// <summary>
    /// Parse <c>SHARPI_SNAPKV_BUDGET</c>, <c>SHARPI_SNAPKV_WINDOW</c>,
    /// <c>SHARPI_SNAPKV_RECENCY</c>. Budget=0 (or unset) disables.
    /// </summary>
    public static SnapKvConfig FromEnvironment()
    {
        int budget  = ParseInt("SHARPI_SNAPKV_BUDGET",  0);
        int window  = ParseInt("SHARPI_SNAPKV_WINDOW",  SnapKvSelector.DefaultWindow);
        int recency = ParseInt("SHARPI_SNAPKV_RECENCY", SnapKvSelector.DefaultRecency);
        if (budget < 0) budget = 0;
        if (window < 1) window = 1;
        if (recency < 0) recency = 0;
        return new SnapKvConfig(budget, window, recency);
    }

    private static int ParseInt(string name, int defaultValue)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        return int.TryParse(raw, out int v) ? v : defaultValue;
    }
}
