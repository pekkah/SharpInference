using System.Collections.Immutable;
using SharpInference.Core.Grammar;

namespace SharpInference.Engine;

/// <summary>
/// Token sampler: greedy, temperature, top-k, top-p (nucleus), and min-p.
/// Operates on a logits span and returns the next token ID.
/// </summary>
public static class Sampler
{
    /// <summary>
    /// Sample a token from logits using the given sampling parameters.
    /// </summary>
    public static int Sample(ReadOnlySpan<float> logits, SamplingParams p, Random? rng = null)
    {
        if (p.Temperature <= 0f)
            return Greedy(logits);

        rng ??= Random.Shared;
        int vocabSize = logits.Length;

        bool hasBias = p.LogitBias is { Count: > 0 };
        bool hasPenalty = p.RepetitionPenalty != 1.0f && p.PreviousTokens is { Count: > 0 };

        // Fast path: top-k bounds the candidate set to a handful of tokens, so there is
        // no need to softmax/normalize/sort the whole vocabulary (262144 for Gemma 4) —
        // the dominant decode-time sampling cost. Select the top-k logits in a single
        // pass, then run temperature / softmax / min-p / top-p over just those k. This is
        // the llama.cpp ordering (top-k → softmax(survivors) → top-p). Repetition penalty
        // only *demotes* the recent tokens, so it is folded in by over-selecting (see
        // SampleTopK). Logit bias can promote *arbitrary* tokens, so it keeps the full path.
        if (p.TopK > 0 && p.TopK < vocabSize && !hasBias)
            return SampleTopK(logits, p, rng, hasPenalty);

        // Copy logits so we can modify them
        Span<float> probs = vocabSize <= 4096
            ? stackalloc float[vocabSize]
            : new float[vocabSize];
        logits.CopyTo(probs);

        // Apply logit bias (additive in logit space, before temperature scaling)
        if (p.LogitBias is { Count: > 0 })
        {
            foreach (var (id, bias) in p.LogitBias)
                if ((uint)id < (uint)vocabSize)
                    probs[id] += bias;
        }

        // Repetition penalty (applied in logit space before temperature)
        if (p.RepetitionPenalty != 1.0f && p.PreviousTokens is { Count: > 0 })
        {
            foreach (int id in p.PreviousTokens)
            {
                if ((uint)id < (uint)vocabSize)
                {
                    // Positive logits are divided; negative logits are multiplied
                    if (probs[id] > 0f)
                        probs[id] /= p.RepetitionPenalty;
                    else
                        probs[id] *= p.RepetitionPenalty;
                }
            }
        }

        // Temperature scaling
        if (p.Temperature != 1.0f)
        {
            float invTemp = 1.0f / p.Temperature;
            for (int i = 0; i < vocabSize; i++)
                probs[i] *= invTemp;
        }

        // Softmax
        Softmax(probs);

        // Top-k filtering
        if (p.TopK > 0 && p.TopK < vocabSize)
        {
            ApplyTopK(probs, p.TopK);
            // Renormalize the surviving probabilities over the top-k set before min-p /
            // top-p, so the nucleus cutoff is taken over the post-top-k distribution. This
            // is the llama.cpp ordering (top-k → renormalize → top-p) and keeps this path
            // consistent with the SampleTopK fast path, which softmaxes over the k
            // survivors directly. (min-p is a ratio test, so it is unaffected by the
            // renormalization; top-p's cumulative cutoff is not.)
            Normalize(probs);
        }

        // Min-p filtering
        if (p.MinP > 0f)
            ApplyMinP(probs, p.MinP);

        // Top-p (nucleus) filtering
        if (p.TopP < 1.0f && p.TopP > 0f)
            ApplyTopP(probs, p.TopP);

        // Renormalize after filtering
        Normalize(probs);

        // Sample from the distribution
        return SampleFromDistribution(probs, rng);
    }

    /// <summary>
    /// Build the full-vocabulary, post-filter sampling distribution for <paramref name="logits"/>
    /// under <paramref name="p"/> into <paramref name="probs"/> (length must equal
    /// <c>logits.Length</c>; entries outside the kept support are 0 and the kept entries sum to 1).
    /// Applies the SAME pipeline as <see cref="Sample"/>'s slow path — logit bias, repetition
    /// penalty, temperature, softmax, top-k (+ renormalise), min-p, top-p, normalise — so a token
    /// drawn from this distribution is distributed identically to <see cref="Sample"/>.
    ///
    /// Used by distribution-preserving speculative sampling (issue #178), which needs the draft
    /// proposal probability q(x) and the residual max(0, p − q) over a SHARED support: the draft
    /// and target distributions MUST be built with the same <paramref name="p"/> for the
    /// min(1, p/q) accept ratio to be meaningful. On Temperature ≤ 0 writes a one-hot at the
    /// greedy argmax.
    /// </summary>
    public static void BuildFilteredDistribution(ReadOnlySpan<float> logits, SamplingParams p, Span<float> probs)
    {
        int vocabSize = logits.Length;
        if (probs.Length != vocabSize)
            throw new ArgumentException(
                $"probs length ({probs.Length}) must equal logits length ({vocabSize}).", nameof(probs));

        if (p.Temperature <= 0f)
        {
            probs.Clear();
            probs[Greedy(logits)] = 1f;
            return;
        }

        logits.CopyTo(probs);

        // Logit bias (additive, before temperature) — mirrors Sample's slow path.
        if (p.LogitBias is { Count: > 0 })
            foreach (var (id, bias) in p.LogitBias)
                if ((uint)id < (uint)vocabSize)
                    probs[id] += bias;

        // Repetition penalty (in logit space, before temperature).
        if (p.RepetitionPenalty != 1.0f && p.PreviousTokens is { Count: > 0 })
            foreach (int id in p.PreviousTokens)
                if ((uint)id < (uint)vocabSize)
                    probs[id] = probs[id] > 0f ? probs[id] / p.RepetitionPenalty : probs[id] * p.RepetitionPenalty;

        if (p.Temperature != 1.0f)
        {
            float invTemp = 1.0f / p.Temperature;
            for (int i = 0; i < vocabSize; i++)
                probs[i] *= invTemp;
        }

        Softmax(probs);

        if (p.TopK > 0 && p.TopK < vocabSize)
        {
            ApplyTopK(probs, p.TopK);
            Normalize(probs);
        }
        if (p.MinP > 0f)
            ApplyMinP(probs, p.MinP);
        if (p.TopP < 1.0f && p.TopP > 0f)
            ApplyTopP(probs, p.TopP);
        Normalize(probs);
    }

    /// <summary>
    /// Sample like <see cref="Sample"/> but also materialise the full post-filter distribution
    /// into <paramref name="probs"/> (see <see cref="BuildFilteredDistribution"/>) and return the
    /// drawn token, so <c>probs[token]</c> is its proposal probability q(token). On Temperature ≤ 0
    /// returns the greedy argmax. Distribution-preserving speculative sampling (issue #178) uses
    /// this to propose draft tokens while retaining q for the accept/residual test.
    /// </summary>
    public static int SampleWithDistribution(ReadOnlySpan<float> logits, SamplingParams p, Span<float> probs, Random? rng = null)
    {
        if (p.Temperature <= 0f)
        {
            // One-hot at the greedy argmax — no filter pipeline, no RNG draw.
            int argmax = Greedy(logits);
            probs.Clear();
            probs[argmax] = 1f;
            return argmax;
        }
        BuildFilteredDistribution(logits, p, probs);
        rng ??= Random.Shared;
        return SampleFromDistribution(probs, rng);
    }

    /// <summary>
    /// Draw a token from a pre-computed normalized distribution (e.g. one already built by
    /// <see cref="BuildFilteredDistribution"/>), avoiding a redundant rebuild. Used by the
    /// speculative decoder's looser-accept reject branch, where the target distribution at the
    /// rejection position is already in hand.
    /// </summary>
    public static int SampleFromProbs(ReadOnlySpan<float> probs, Random rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        return SampleFromDistribution(probs, rng);
    }

    /// <summary>
    /// Top-k-first sampling fast path (no logit bias). Selects the top-k logits in one
    /// O(vocab) pass, then applies temperature, softmax, min-p, and top-p over only those
    /// k candidates — avoiding the full-vocabulary softmax, sort, and allocation of the
    /// slow path. Probabilities are softmaxed over the k survivors (i.e. renormalised over
    /// the top-k set), which is the llama.cpp ordering (top-k → renormalise → top-p) and
    /// matches the slow path's behaviour: <see cref="Sample"/> renormalises after
    /// <see cref="ApplyTopK"/> for exactly this reason.
    ///
    /// Repetition penalty (<paramref name="hasPenalty"/>) only ever *lowers* the logits of
    /// recently seen tokens, so it cannot promote a token from outside the top-(k+W) raw
    /// logits into the post-penalty top-k (W = number of penalised tokens, bounded above by
    /// <c>PreviousTokens.Count</c>). We therefore over-select the top-(k+W) raw candidates,
    /// apply the penalty to that small set, re-sort, and keep the top-k.
    /// </summary>
    private static int SampleTopK(ReadOnlySpan<float> logits, SamplingParams p, Random rng, bool hasPenalty)
    {
        int k = p.TopK;
        int vocab = logits.Length;

        // Over-select to cover penalty demotions. W ≤ PreviousTokens.Count (using the raw
        // count, an upper bound on distinct penalised tokens, avoids a per-token dictionary
        // allocation on this hot path; the candidate set stays tiny).
        int prevCount = hasPenalty ? p.PreviousTokens!.Count : 0;
        int sel = Math.Min(vocab, k + prevCount);

        // Select the top-`sel` (idx, logit) descending by logit via threshold insertion.
        Span<int>   idx  = sel <= 256 ? stackalloc int[sel]   : new int[sel];
        Span<float> vals = sel <= 256 ? stackalloc float[sel] : new float[sel];
        vals.Fill(float.NegativeInfinity);
        idx.Fill(-1);
        for (int i = 0; i < vocab; i++)
        {
            float v = logits[i];
            if (v <= vals[sel - 1]) continue;
            int j = sel - 1;
            while (j > 0 && vals[j - 1] < v) { vals[j] = vals[j - 1]; idx[j] = idx[j - 1]; j--; }
            vals[j] = v; idx[j] = i;
        }

        // Degenerate guard: no finite logit was selected (e.g. an all -inf vector). Fall
        // back to greedy over the raw logits, which always yields a valid token index
        // rather than the Fill(-1) sentinel.
        if (idx[0] < 0 || float.IsNegativeInfinity(vals[0]))
            return Greedy(logits);

        // Apply repetition penalty to the selected candidates, then re-sort so the top-k
        // reflects the post-penalty order. Penalty is in logit space: positive logits are
        // divided, negative ones multiplied, once per occurrence in PreviousTokens (the
        // same per-occurrence compounding as the slow path).
        if (hasPenalty)
        {
            for (int t = 0; t < sel; t++)
            {
                if (idx[t] < 0) continue;
                int occ = 0;
                foreach (int id in p.PreviousTokens!)
                    if (id == idx[t]) occ++;
                if (occ == 0) continue;
                float pen = MathF.Pow(p.RepetitionPenalty, occ);
                vals[t] = vals[t] > 0f ? vals[t] / pen : vals[t] * pen;
            }
            // Insertion re-sort (sel is tiny: k + PreviousTokens.Count).
            for (int a = 1; a < sel; a++)
            {
                float v = vals[a]; int id = idx[a]; int b = a - 1;
                while (b >= 0 && vals[b] < v) { vals[b + 1] = vals[b]; idx[b + 1] = idx[b]; b--; }
                vals[b + 1] = v; idx[b + 1] = id;
            }
        }

        // Temperature + softmax over the top-k candidates (vals[0] is the max logit).
        float invTemp = p.Temperature == 1.0f ? 1.0f : 1.0f / p.Temperature;
        float max = vals[0] * invTemp;
        Span<float> probs = k <= 256 ? stackalloc float[k] : new float[k];
        float sum = 0f;
        for (int t = 0; t < k; t++)
        {
            // Tokens past the real vocab tail (k > #finite logits) stay at -inf → exp = 0.
            float e = float.IsNegativeInfinity(vals[t]) ? 0f : MathF.Exp(vals[t] * invTemp - max);
            probs[t] = e;
            sum += e;
        }
        float invSum = sum > 0f ? 1.0f / sum : 0f;
        for (int t = 0; t < k; t++) probs[t] *= invSum;

        int count = k;

        // Min-p: drop candidates below minP × max prob. Descending order ⇒ first failure
        // bounds the survivors. The ratio probs[t]/probs[0] is normalisation-invariant.
        if (p.MinP > 0f)
        {
            float thr = p.MinP * probs[0];
            for (int t = 0; t < count; t++)
                if (probs[t] < thr) { count = t; break; }
        }

        // Top-p (nucleus): smallest prefix whose cumulative prob reaches topP.
        if (p.TopP < 1.0f && p.TopP > 0f)
        {
            float cum = 0f;
            for (int t = 0; t < count; t++)
            {
                cum += probs[t];
                if (cum >= p.TopP) { count = t + 1; break; }
            }
        }

        // Sample proportionally over the kept candidates [0, count). idx[0] is the argmax
        // and always valid (guarded above), so it is the safe rounding/empty-set fallback —
        // never the Fill(-1) sentinel.
        float keptSum = 0f;
        for (int t = 0; t < count; t++) keptSum += probs[t];
        float r = (float)rng.NextDouble() * keptSum;
        float c = 0f;
        for (int t = 0; t < count; t++)
        {
            c += probs[t];
            if (r <= c && idx[t] >= 0) return idx[t];
        }
        return idx[0];
    }

    /// <summary>
    /// Greedy decoding: return the token with the highest logit.
    /// </summary>
    public static int Greedy(ReadOnlySpan<float> logits)
    {
        int maxIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > maxVal)
            {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private static void Softmax(Span<float> x)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < x.Length; i++)
            if (x[i] > max) max = x[i];

        float sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = MathF.Exp(x[i] - max);
            sum += x[i];
        }

        float invSum = 1.0f / sum;
        for (int i = 0; i < x.Length; i++)
            x[i] *= invSum;
    }

    /// <summary>
    /// Zero out all but the top-k highest probability tokens.
    /// </summary>
    private static void ApplyTopK(Span<float> probs, int k)
    {
        // Find the k-th largest value
        // For correctness (not speed), use a simple partial sort approach
        float kthValue = FindKthLargest(probs, k);

        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] < kthValue)
                probs[i] = 0f;
        }
    }

    /// <summary>
    /// Zero out tokens whose probability is less than minP * max_prob.
    /// </summary>
    private static void ApplyMinP(Span<float> probs, float minP)
    {
        float maxProb = 0;
        for (int i = 0; i < probs.Length; i++)
            if (probs[i] > maxProb) maxProb = probs[i];

        float threshold = minP * maxProb;
        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] < threshold)
                probs[i] = 0f;
        }
    }

    /// <summary>
    /// Keep only the smallest set of tokens whose cumulative probability >= topP.
    /// </summary>
    /// <remarks>
    /// Only the non-zero entries are sorted. With a large vocabulary (Gemma 4 =
    /// 262144) and top-k applied first, all but ~k entries are already zero, so
    /// sorting the full span (the old behaviour) was an O(V·logV) delegate-comparator
    /// sort + V-element allocation per token — the dominant decode-time sampling cost.
    /// Zeros never affect the cumulative sum or the cutoff (they would sort to the
    /// tail), so gathering and sorting only the survivors is bit-identical.
    /// </remarks>
    private static void ApplyTopP(Span<float> probs, float topP)
    {
        // Gather non-zero (idx, prob) survivors. After top-k this is ~k entries;
        // without top-k it can be the whole vocab (same cost as before, no regression).
        int n = 0;
        for (int i = 0; i < probs.Length; i++)
            if (probs[i] > 0f) n++;
        if (n == 0) return;

        Span<(int idx, float prob)> indexed = n <= 512
            ? stackalloc (int, float)[n]
            : new (int, float)[n];
        int j = 0;
        for (int i = 0; i < probs.Length; i++)
            if (probs[i] > 0f) indexed[j++] = (i, probs[i]);

        // Descending sort by probability (survivors only).
        MemoryExtensions.Sort(indexed, static (a, b) => b.prob.CompareTo(a.prob));

        // Find cutoff
        float cumSum = 0;
        int cutoff = indexed.Length;
        for (int i = 0; i < indexed.Length; i++)
        {
            cumSum += indexed[i].prob;
            if (cumSum >= topP)
            {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond the cutoff
        for (int i = cutoff; i < indexed.Length; i++)
            probs[indexed[i].idx] = 0f;
    }

    private static void Normalize(Span<float> probs)
    {
        float sum = 0;
        for (int i = 0; i < probs.Length; i++)
            sum += probs[i];

        if (sum > 0f && sum != 1.0f)
        {
            float invSum = 1.0f / sum;
            for (int i = 0; i < probs.Length; i++)
                probs[i] *= invSum;
        }
    }

    private static int SampleFromDistribution(ReadOnlySpan<float> probs, Random rng)
    {
        float r = (float)rng.NextDouble();
        float cumSum = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumSum += probs[i];
            if (r <= cumSum)
                return i;
        }
        // Fallback: return last non-zero token (rounding errors)
        return probs.Length - 1;
    }

    /// <summary>
    /// Sample a token proportional to the residual max(0, p[x] − q[x]) of two full-vocabulary
    /// distributions built with the SAME <see cref="SamplingParams"/> (see
    /// <see cref="BuildFilteredDistribution"/>). This is the rejection correction of speculative
    /// sampling (Leviathan et al. / Chen et al.): drawing the corrected token from the residual
    /// after a draft proposal is rejected makes the overall emitted-token distribution identical
    /// to sampling directly from <paramref name="p"/>. If the residual mass is ≈0 (numerical /
    /// q already dominated p on the kept support), falls back to sampling from <paramref name="p"/>.
    /// </summary>
    public static int ResampleResidual(ReadOnlySpan<float> p, ReadOnlySpan<float> q, Random rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (p.Length != q.Length)
            throw new ArgumentException(
                $"p and q must have equal length ({p.Length} vs {q.Length}).", nameof(q));

        float sum = 0f;
        for (int i = 0; i < p.Length; i++)
        {
            float r = p[i] - q[i];
            if (r > 0f) sum += r;
        }
        if (sum <= 0f || float.IsNaN(sum))
            return SampleFromDistribution(p, rng); // degenerate: empty residual → fall back to p

        float target = (float)rng.NextDouble() * sum;
        float cum = 0f;
        int lastPositive = -1;
        for (int i = 0; i < p.Length; i++)
        {
            float r = p[i] - q[i];
            if (r > 0f)
            {
                lastPositive = i;
                cum += r;
                if (target <= cum) return i;
            }
        }
        // Rounding fallback: the last token with positive residual (tracked above, no extra pass).
        return lastPositive >= 0 ? lastPositive : SampleFromDistribution(p, rng);
    }

    /// <summary>
    /// Find the k-th largest value in the span (1-indexed: k=1 is the largest).
    /// Simple O(n*k) selection — fine for Phase 1 correctness.
    /// </summary>
    private static float FindKthLargest(ReadOnlySpan<float> data, int k)
    {
        // Collect top-k values
        Span<float> topK = k <= 256
            ? stackalloc float[k]
            : new float[k];
        topK.Fill(float.NegativeInfinity);

        for (int i = 0; i < data.Length; i++)
        {
            float val = data[i];
            if (val > topK[k - 1])
            {
                topK[k - 1] = val;
                // Insertion sort to maintain descending order
                for (int j = k - 1; j > 0 && topK[j] > topK[j - 1]; j--)
                    (topK[j], topK[j - 1]) = (topK[j - 1], topK[j]);
            }
        }

        return topK[k - 1];
    }
}

public sealed record SamplingParams
{
    public float Temperature { get; init; } = 1.0f;
    public int TopK { get; init; } = 0;
    public float TopP { get; init; } = 1.0f;
    public float MinP { get; init; } = 0.0f;
    public float RepetitionPenalty { get; init; } = 1.0f;
    public int MaxNewTokens { get; init; } = 512;

    /// <summary>
    /// Token IDs that halt generation. <b>Replace semantics</b>: when non-null this <em>replaces</em>
    /// the tokenizer's end-of-generation set (<see cref="ITokenizer.EogTokenIds"/>) rather than
    /// adding to it — a caller that sets this to add one stop silently loses the EOG tokens unless it
    /// re-includes them. To ADD stops while keeping EOG, leave this null and use
    /// <see cref="AdditionalStopTokenIds"/> instead (issue #304).
    /// </summary>
    public int[]? StopTokenIds { get; init; }

    /// <summary>
    /// Extra stop token IDs that are <b>always unioned</b> with the effective stop set
    /// (<see cref="StopTokenIds"/> when set, otherwise <see cref="ITokenizer.EogTokenIds"/>) —
    /// never replacing it. This is the footgun-free way to add a stop, e.g. a tool-boundary token
    /// from <see cref="IToolCallAdapter.ToolBoundaryStopMarkers"/> for an agentic loop, without
    /// dropping the model's end-of-turn / EOS tokens. Null or empty = no additions. See issue #304.
    /// </summary>
    public int[]? AdditionalStopTokenIds { get; init; }

    /// <summary>
    /// Additive logit bias applied before temperature scaling.
    /// Maps token IDs to bias values in the range [-100, 100].
    /// Use -100 to effectively prevent a token; +100 to strongly favour it.
    /// </summary>
    public IReadOnlyDictionary<int, float>? LogitBias { get; init; }

    /// <summary>
    /// Recently generated token IDs for repetition penalty.
    /// Typically a sliding window of the last N generated tokens.
    /// </summary>
    public IReadOnlyList<int>? PreviousTokens { get; init; }

    /// <summary>
    /// Optional grammar/FSM constraint applied during decoding (issue #374). The engine advances it
    /// with every emitted token and, while it reports <see cref="ITokenConstraint.IsConstraining"/>,
    /// masks the logits to only the tokens it permits before sampling — forcing e.g. a tool call's
    /// arguments to satisfy the supplied JSON Schema in the model's native call syntax. The
    /// constraint is stateful and single-request. <c>null</c> (default) = unconstrained decoding,
    /// byte-identical to the pre-#374 path.
    /// <para>
    /// Honored by both the single-user <c>InferenceEngine</c> and <c>ContinuousBatchingEngine</c>
    /// (<c>SHARPI_MAX_BATCH</c>, issue #377): under batching the instance is carried per sequence and
    /// masks only that sequence's logits row each step, so concurrent unconstrained requests are
    /// unaffected. A single instance must not be shared across concurrent requests.
    /// </para>
    /// </summary>
    public ITokenConstraint? Constraint { get; init; }

    /// <summary>
    /// Maximum thinking-mode tokens before the engine forces a <c>&lt;/think&gt;</c> close.
    /// 0 = unlimited (default). No-op on models without registered think tokens, and
    /// also when the chat template was rendered with reasoning disabled (the engine
    /// never enters thinking mode in that case).
    /// </summary>
    public int MaxThinkingTokens { get; init; } = 0;

    /// <summary>
    /// Speculative decoding type. Mirrors llama.cpp's <c>--spec-type</c>:
    /// <c>Auto</c> (default) lets the engine pick (currently: enable MTP when the
    /// model ships an MTP head AND greedy AND not in thinking mode);
    /// <c>None</c> forces single-token decode; <c>Mtp</c> forces MTP self-speculative
    /// decoding (errors if the model doesn't ship an MTP head, or sampling isn't
    /// greedy, or thinking is on).
    /// </summary>
    public SpecType SpecType { get; init; } = SpecType.Auto;

    /// <summary>
    /// Max number of draft tokens per speculative step. Mirrors llama.cpp's
    /// <c>--spec-draft-n-max</c>. 0 = engine default (1 for MTP today; N=2 with
    /// issue #30 batched verify on hybrid GDN passes).
    /// </summary>
    public int SpecDraftNMax { get; init; } = 0;

    /// <summary>
    /// Min number of draft tokens per speculative step. Mirrors llama.cpp's
    /// <c>--spec-draft-n-min</c>. 0 = no minimum. With issue #30 batched verify
    /// the draft length is fixed at 1 (= one MTP forward per main pair), so
    /// this flag is currently accepted but a no-op; it gains meaning when a
    /// variable-length tree-draft lands. Issue #37.
    /// </summary>
    public int SpecDraftNMin { get; init; } = 0;

    /// <summary>
    /// Minimum draft probability for probabilistic accept under MTP verification.
    /// Mirrors llama.cpp's <c>--spec-draft-p-min</c> (llama.cpp default 0.75).
    /// <list type="bullet">
    ///   <item><c>1.0</c> (default): pure argmax-match — accept iff
    ///         <c>draft == argmax(target)</c>. Produces byte-identical output vs
    ///         <c>SHARPI_DISABLE_MTP=1</c>.</item>
    ///   <item><c>p ∈ (0, 1)</c>: probabilistic accept — accept iff
    ///         <c>softmax(target)[draft] &gt;= p</c> OR <c>draft == argmax(target)</c>.
    ///         Higher acceptance rate, but emitted tokens diverge from baseline
    ///         when the draft wins on the prob rule alone.</item>
    ///   <item><c>0.0</c> or negative: treated as 1.0 (default).</item>
    /// </list>
    /// Issue #38.
    /// </summary>
    public float SpecDraftPMin { get; init; } = 1f;

    /// <summary>
    /// Resolves the effective stop-token set for a request. <see cref="StopTokenIds"/> REPLACES
    /// <paramref name="eog"/> when non-null (replace semantics); otherwise <paramref name="eog"/>
    /// is the base. <see cref="AdditionalStopTokenIds"/> is then unioned on top (de-duplicated),
    /// never replacing — so adding a stop can't drop the EOG set. With no additions this returns
    /// the base unchanged, byte-identical to the pre-#304 behavior.
    /// </summary>
    internal ImmutableArray<int> ResolveStopSet(ImmutableArray<int> eog)
    {
        if (AdditionalStopTokenIds is not { Length: > 0 } extra)
            return StopTokenIds is { } userStops ? [.. userStops] : eog;

        var set = StopTokenIds is { } baseStops ? new HashSet<int>(baseStops) : new HashSet<int>(eog);
        foreach (int id in extra) set.Add(id);
        return [.. set];
    }
}

/// <summary>Speculative-decoding type. Mirrors llama.cpp's <c>--spec-type</c>.</summary>
public enum SpecType
{
    /// <summary>Engine picks: enable MTP when the model supports it and sampling is greedy.</summary>
    Auto = 0,
    /// <summary>Disable speculative decoding; one token per main forward.</summary>
    None = 1,
    /// <summary>Multi-Token Prediction self-speculative decoding via the model's own MTP head.</summary>
    Mtp = 2,
}
