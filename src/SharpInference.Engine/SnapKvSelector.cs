using System.Buffers;
using System.Runtime.InteropServices;
using SharpInference.Cpu;
using SharpInference.TurboQuant;

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
    /// Replace the internal score accumulator with externally-computed scores
    /// (e.g. produced by a GPU scoring kernel and downloaded host-side). Length
    /// must match <paramref name="promptLen"/>; the caller's pooling across
    /// (queries × heads × layers) is preserved verbatim.
    /// </summary>
    public void LoadScores(ReadOnlySpan<float> scores, int promptLen)
    {
        if (scores.Length < promptLen)
            throw new ArgumentException(
                $"LoadScores: scores.Length={scores.Length} < promptLen={promptLen}.", nameof(scores));
        if (_scoreAccum.Length < promptLen)
            _scoreAccum = new float[promptLen];
        scores[..promptLen].CopyTo(_scoreAccum);
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
    /// TurboQuant-aware overload of <see cref="AccumulateLayer(float*, int, PagedKvCache, int, int, int)"/>:
    /// reads K vectors from a <see cref="TurboQuantKvCache"/> instead of a paged FP32 cache.
    /// For positions in the TQ region the score is the dequant-dot of the rotated
    /// query against the per-position compressed block (tile or staging); for
    /// positions in the FP32 ring window the score is a plain FP32 dot product.
    /// Identical pooling/softmax/accumulation structure as the FP32 overload —
    /// only the per-position K fetch differs.
    /// </summary>
    /// <remarks>
    /// The TQ codec is lossy (~2-5% MSE-optimal bias at 3-4 bits), so the scores
    /// here have somewhat more noise than the FP32 path. Empirically that's fine
    /// for keep-set selection because the partial sort in <see cref="SelectKeepSet"/>
    /// rounds away score-rank perturbations below the budget threshold.
    /// </remarks>
    public void AccumulateLayer(float* batchQ, int N, TurboQuantKvCache cache, int layer,
                                int startPos, int window)
    {
        int W = Math.Min(window, N);
        int wStart = N - W;
        int promptLen = N;
        var scratch = stackalloc float[Math.Min(promptLen, 8192)];
        bool useStack = promptLen <= 8192;
        float* scoreBuf = useStack ? scratch : (float*)NativeMemory.Alloc((nuint)(promptLen * sizeof(float)));
        // Per-(head, position) rotated query buffer. Reused across all cached
        // positions for one query — TurboQuantOps.DequantDot expects its
        // q-side input already in the rotated domain.
        Span<float> rotatedQ = stackalloc float[_headDim];
        // Per-query TQ score scratch sized to the deepest TQ region we'll see.
        // Path: ComputeKScores writes one float per TQ position, then we copy
        // them into scoreBuf[0..tqLen). Rented from ArrayPool so AccumulateLayer
        // is allocation-free across all layers (called once per layer; selector
        // is reused across the whole prefill).
        int maxTqLen = cache.GetTqLength(layer);
        float[] tqScoreScratch = maxTqLen > 0
            ? ArrayPool<float>.Shared.Rent(maxTqLen)
            : Array.Empty<float>();
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

                    // Rotate Q once per (head, query). The same rotated vector
                    // is reused across all tqLen TQ positions via the FastScan
                    // tile/staging kernels invoked below.
                    var keyCompressor = cache.GetKeyCompressor(layer, kvHead);
                    TurboQuantOps.RotateQuery(
                        new ReadOnlySpan<float>(qHead, _headDim),
                        rotatedQ,
                        keyCompressor.SignPattern,
                        _headDim);

                    int tqLen = cache.GetTqLength(layer);

                    // 1a. TQ K-scores via the same FastScan path TqAttention uses
                    //     in decode. Single call covers tile-walk + staging tail.
                    if (tqLen > 0)
                    {
                        fixed (float* rotQPtr = rotatedQ)
                        fixed (float* tqScorePtr = tqScoreScratch)
                            cache.ComputeKScores(layer, kvHead, rotQPtr, scale, tqScorePtr);
                    }

                    // 1b. Mux TQ + FP32 + causal mask into scoreBuf in position order.
                    for (int p = 0; p < promptLen; p++)
                    {
                        int absPos = startPos + p;
                        if (absPos > qAbsPos) { scoreBuf[p] = float.NegativeInfinity; continue; }

                        if (p < tqLen)
                        {
                            // ComputeKScores already multiplied by `scale`.
                            scoreBuf[p] = tqScoreScratch[p];
                        }
                        else
                        {
                            float* kVec = cache.Fp32KeyAt(layer, p) + kvHead * _headDim;
                            scoreBuf[p] = SimdKernels.DotF32(qHead, kVec, _headDim) * scale;
                        }
                    }

                    // 2-3. Softmax in place + accumulate.
                    SimdKernels.SoftmaxInPlace(scoreBuf, promptLen);
                    for (int p = 0; p < promptLen; p++) _scoreAccum[p] += scoreBuf[p];
                }
            }
        }
        finally
        {
            if (!useStack) NativeMemory.Free((void*)scoreBuf);
            if (maxTqLen > 0) ArrayPool<float>.Shared.Return(tqScoreScratch);
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
/// <param name="Budget">Explicit budget when <see cref="IsBudgetExplicit"/> is true;
/// 0 otherwise. Backends with auto-budget support (currently
/// <c>CudaHybridGdnForwardPass</c>) substitute their own value when the env
/// var was not set.</param>
/// <param name="Window">Number of trailing queries used as the importance probe.</param>
/// <param name="Recency">Trailing positions always retained.</param>
/// <param name="IsBudgetExplicit">True iff <c>SHARPI_SNAPKV_BUDGET</c> was set to
/// any value (including <c>0</c>). False iff the env var was unset — backends
/// may pick an auto-budget instead.</param>
public readonly record struct SnapKvConfig(int Budget, int Window, int Recency, bool IsBudgetExplicit)
{
    public bool Enabled => Budget > 0;

    /// <summary>
    /// Parse <c>SHARPI_SNAPKV_BUDGET</c>, <c>SHARPI_SNAPKV_WINDOW</c>,
    /// <c>SHARPI_SNAPKV_RECENCY</c>. Budget=0 disables; budget unset leaves the
    /// decision to the backend (auto on CUDA hybrid GDN, opt-in elsewhere).
    /// </summary>
    public static SnapKvConfig FromEnvironment()
    {
        var budgetRaw = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        bool explicitBudget = int.TryParse(budgetRaw, out int budget);
        int window  = ParseInt("SHARPI_SNAPKV_WINDOW",  SnapKvSelector.DefaultWindow);
        int recency = ParseInt("SHARPI_SNAPKV_RECENCY", SnapKvSelector.DefaultRecency);
        if (budget < 0) budget = 0;
        if (window < 1) window = 1;
        if (recency < 0) recency = 0;
        return new SnapKvConfig(budget, window, recency, explicitBudget);
    }

    /// <summary>Cache size below which the SnapKV auto-budget stays disabled.
    /// A 40 MiB cache (Qwen3.6-27B-MTP at ctx=2048, bf16) gains very little
    /// from eviction — the per-token attention work is already a small slice
    /// of decode cost. The threshold scales naturally with attention-layer
    /// count, kv_dim, and configured ctx, so big-context / big-cache setups
    /// (the 12 GB target) trip it while small-context smoke tests don't.</summary>
    public const long AutoEnableMinCacheBytes = 256L * 1024 * 1024;

    /// <summary>
    /// VRAM-scaled default budget used when the env var is unset and the backend
    /// supports auto-eviction. Targets ~1/4 of the configured context window
    /// (matching the SnapKV paper's "8× compression on 16K prompts" reference
    /// point), floored at 1024 (smaller risks losing semantically important
    /// context) and capped at 4096 (beyond that the paper's accuracy curve
    /// flattens while post-eviction decode attention keeps scaling). Returns
    /// 0 — i.e. don't auto-enable — when the full cache is below
    /// <see cref="AutoEnableMinCacheBytes"/>; the user has plenty of headroom
    /// and silently introducing a lossy step is not worth it.
    /// </summary>
    public static int ComputeAutoBudget(int maxSeqLen, long fullCacheBytes)
    {
        if (fullCacheBytes > 0 && fullCacheBytes < AutoEnableMinCacheBytes) return 0;
        int candidate = maxSeqLen / 4;
        if (candidate < 1024) candidate = Math.Min(1024, maxSeqLen);
        if (candidate > 4096) candidate = 4096;
        return candidate;
    }

    private static int ParseInt(string name, int defaultValue)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        return int.TryParse(raw, out int v) ? v : defaultValue;
    }
}
