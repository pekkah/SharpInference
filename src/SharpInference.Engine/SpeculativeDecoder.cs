using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Speculative decoding (greedy): a small draft model generates K tokens which the target model
/// verifies via a batched forward pass, accepting each where they agree and generating a
/// correction token where they first diverge.
///
/// Expected speedup: E[tokens/step] / E[target-forwards/step] where both equal
/// (1-α^(k+1))/(1-α) for acceptance rate α, but target uses batched matmuls (k tokens in one
/// Prefill-style call) reducing memory bandwidth from k×1 to approximately 1×batch.
/// Typical speedup 1.3–2× depending on model size ratio and acceptance rate.
///
/// Both target and draft must share the same tokenizer (same vocab size).
/// Note: does NOT take ownership of the forward pass instances.
/// </summary>
public sealed class SpeculativeDecoder
{
    private readonly IForwardPass _target;
    private readonly IForwardPass _draft;
    private readonly bool _batchVerify;
    private int _lookahead;

    // Generation state
    private int _nextPos;
    private float[] _savedTargetLogits;
    private float[] _savedDraftLogits;

    // Acceptance statistics
    private long _totalAccepted;
    private long _totalEmitted;

    // Phase timing (issue #207 bench reporting): cumulative wall time spent drafting,
    // batch-verifying, and committing (truncate + correction forwards) across all steps.
    private readonly System.Diagnostics.Stopwatch _phaseSw = new();

    public SpeculativeDecoder(IForwardPass target, IForwardPass draft, int lookahead = 4)
    {
        if (target.VocabSize != draft.VocabSize)
            throw new ArgumentException(
                $"Target and draft vocab sizes differ ({target.VocabSize} vs {draft.VocabSize}). " +
                "Both models must share the same tokenizer.");
        // Spec-decoding rewinds rejected draft tokens via TruncateTo(P + accepted) on
        // both passes. Models whose state is destructively updated per token (Gated
        // DeltaNet hybrid) can't honor that, so fail fast at construction rather than
        // mid-decode. See issue #20.
        if (!target.SupportsPartialRewind)
            throw new ArgumentException(
                $"Speculative decoding requires the target forward pass to support partial rewind; " +
                $"{target.GetType().Name} does not. Disable speculative decoding for this model or " +
                "use a non-GDN target pass.",
                nameof(target));
        if (!draft.SupportsPartialRewind)
            throw new ArgumentException(
                $"Speculative decoding requires the draft forward pass to support partial rewind; " +
                $"{draft.GetType().Name} does not. Disable speculative decoding for this model or " +
                "use a non-GDN draft pass.",
                nameof(draft));
        _target = target;
        _draft = draft;
        // Kill-switch (issue #207): SHARPI_SPEC_BATCH_VERIFY=0 forces the sequential
        // verify fallback even when the target implements BatchVerify. Read once at
        // construction (same pattern as the forward passes' decode toggles); the
        // capability itself is re-checked per step — it can flip after construction
        // (e.g. ForwardPass.EnableTurboQuant).
        _batchVerify = Environment.GetEnvironmentVariable("SHARPI_SPEC_BATCH_VERIFY") != "0";
        _lookahead = Math.Max(1, lookahead);
        _savedTargetLogits = new float[target.VocabSize];
        _savedDraftLogits = new float[draft.VocabSize];
    }

    /// <summary>Adaptive lookahead: increase/decrease based on recent acceptance rate.</summary>
    public int Lookahead
    {
        get => _lookahead;
        set => _lookahead = Math.Max(1, value);
    }

    /// <summary>Running acceptance rate (accepted tokens / total emitted tokens).</summary>
    public float AcceptanceRate => _totalEmitted > 0 ? (float)_totalAccepted / _totalEmitted : 0f;

    /// <summary>Cumulative milliseconds spent in the draft phase (k−1 draft forwards per step).</summary>
    public double DraftMs { get; private set; }

    /// <summary>Cumulative milliseconds spent batch-verifying with the target.</summary>
    public double VerifyMs { get; private set; }

    /// <summary>Cumulative milliseconds spent in cache truncation + correction-token forwards.</summary>
    public double CommitMs { get; private set; }

    /// <summary>
    /// Initialize the decoder after both models have processed the prompt.
    /// Call after prefilling both target and draft with the same prompt tokens.
    /// </summary>
    /// <param name="prefillLength">Number of prompt tokens (= new KV cache length).</param>
    /// <param name="targetLogits">Logits from the target's last prefill step (vocab-size span).</param>
    /// <param name="draftLogits">Logits from the draft's last prefill step (vocab-size span).</param>
    public void Initialize(int prefillLength, ReadOnlySpan<float> targetLogits, ReadOnlySpan<float> draftLogits)
    {
        _nextPos = prefillLength;
        targetLogits.CopyTo(_savedTargetLogits);
        draftLogits.CopyTo(_savedDraftLogits);
        _totalAccepted = 0;
        _totalEmitted = 0;
        DraftMs = 0;
        VerifyMs = 0;
        CommitMs = 0;
    }

    /// <summary>
    /// Decode up to <paramref name="maxTokens"/> tokens using greedy speculative decoding,
    /// invoking <paramref name="emitToken"/> for each accepted or correction token.
    /// Returns when maxTokens is reached or a stop token is generated.
    /// </summary>
    public void Decode(int maxTokens, ReadOnlySpan<int> stopTokenIds, Action<int> emitToken)
    {
        int generated = 0;
        while (generated < maxTokens)
        {
            int remaining = maxTokens - generated;
            int k = Math.Min(_lookahead, remaining);
            int[] emitted = Step(k);

            foreach (int token in emitted)
            {
                // Check stop before emitting — avoids printing the stop token itself.
                if (IsStop(token, stopTokenIds)) return;
                emitToken(token);
                generated++;
                if (generated >= maxTokens) return;
            }
        }
    }

    /// <summary>
    /// Run one speculative step: draft k tokens, batch-verify with target, accept greedily.
    /// Returns the emitted token array (accepted_count + 1 tokens, including the correction).
    /// Updates internal state (_nextPos, _savedTargetLogits, _savedDraftLogits).
    /// </summary>
    private int[] Step(int k)
    {
        int P = _nextPos;
        int vocabSize = _target.VocabSize;

        // ── Draft phase ──────────────────────────────────────────────────────────
        // d[0] is free: argmax of saved draft logits (no forward pass needed).
        // d[1..k-1] require k-1 draft Forward calls, appending d[0..k-2] to draft cache.
        _phaseSw.Restart();
        var draftTokens = new int[k];
        var draftLogitsPerPos = new float[k][];

        draftLogitsPerPos[0] = _savedDraftLogits;
        draftTokens[0] = ArgMax(_savedDraftLogits);

        for (int i = 1; i < k; i++)
        {
            var logits = _draft.Forward(draftTokens[i - 1], P + i - 1);
            draftLogitsPerPos[i] = new float[vocabSize];
            logits.CopyTo(draftLogitsPerPos[i]);
            draftTokens[i] = ArgMax(draftLogitsPerPos[i]);
        }
        // Draft cache is now at P + k - 1 (appended d[0..k-2]).
        DraftMs += _phaseSw.Elapsed.TotalMilliseconds;

        // ── Target batch-verify ──────────────────────────────────────────────────
        // Process d[0..k-1] in one batched forward pass.
        // targetLogitsFromBatch[i] = P_target(·|ctx + d[0..i]) (logits AFTER d[i]).
        // After this call, target cache is at P + k.
        _phaseSw.Restart();
        float[][] targetLogitsFromBatch = BatchVerifyTarget(draftTokens, P);
        VerifyMs += _phaseSw.Elapsed.TotalMilliseconds;

        // targetLogits[0]   = saved (before d[0])
        // targetLogits[i+1] = after d[i]  (from batch)
        // We use targetLogits[i] to verify d[i]: accept if argmax == d[i].

        // ── Greedy accept/reject ─────────────────────────────────────────────────
        int accepted = 0;
        for (int i = 0; i < k; i++)
        {
            float[] tLogits = i == 0 ? _savedTargetLogits : targetLogitsFromBatch[i - 1];
            if (ArgMax(tLogits) == draftTokens[i])
                accepted++;
            else
                break;
        }

        // targetLogits at position `accepted` (logits for deciding correction token):
        float[] correctionSourceLogits = accepted == k
            ? targetLogitsFromBatch[k - 1]  // all accepted: logits after d[k-1]
            : (accepted == 0 ? _savedTargetLogits : targetLogitsFromBatch[accepted - 1]);

        int correction = ArgMax(correctionSourceLogits);

        // Update acceptance stats
        _totalAccepted += accepted;
        _totalEmitted += accepted + 1;

        // ── Truncate caches to accepted position ─────────────────────────────────
        // Target is at P+k; truncate to P+accepted.
        _phaseSw.Restart();
        _target.TruncateTo(P + accepted);

        // Draft is at P+k-1; truncate to P+accepted.
        // For all-accepted (accepted == k): need to sync d[k-1] into draft first.
        if (accepted == k)
        {
            // Draft phase only appended d[0..k-2]. Sync d[k-1] now.
            _draft.Forward(draftTokens[k - 1], P + k - 1);
            // Draft cache is now at P+k. No truncation needed before commit.
        }
        else
        {
            _draft.TruncateTo(P + accepted);
        }

        // ── Commit correction to both caches ─────────────────────────────────────
        int commitPos = accepted == k ? P + k : P + accepted;
        var newTargetLogits = _target.Forward(correction, commitPos);
        var newDraftLogits = _draft.Forward(correction, commitPos);
        CommitMs += _phaseSw.Elapsed.TotalMilliseconds;

        // ── Update state ─────────────────────────────────────────────────────────
        _nextPos = commitPos + 1;
        newTargetLogits.CopyTo(_savedTargetLogits);
        newDraftLogits.CopyTo(_savedDraftLogits);

        // ── Build emitted token list: d[0..accepted-1] + correction ──────────────
        var emitted = new int[accepted + 1];
        for (int i = 0; i < accepted; i++) emitted[i] = draftTokens[i];
        emitted[accepted] = correction;
        return emitted;
    }

    /// <summary>
    /// Batch-verify draft tokens with the target model. Targets that report
    /// <see cref="IForwardPass.SupportsBatchVerify"/> (CPU <see cref="ForwardPass"/>, dense
    /// <c>CudaForwardPass</c> — issue #207) take the packed k-token pass, which amortizes
    /// the weight reads k× on memory-bound decode paths. Everything else (and
    /// <c>SHARPI_SPEC_BATCH_VERIFY=0</c>) falls back to k sequential Forward calls.
    /// </summary>
    private float[][] BatchVerifyTarget(int[] draftTokens, int startPos)
    {
        if (_batchVerify && _target.SupportsBatchVerify)
            return _target.BatchVerify(draftTokens, startPos);

        // Generic fallback: sequential Forward calls
        var result = new float[draftTokens.Length][];
        for (int i = 0; i < draftTokens.Length; i++)
        {
            var logits = _target.Forward(draftTokens[i], startPos + i);
            result[i] = new float[_target.VocabSize];
            logits.CopyTo(result[i]);
        }
        return result;
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        }
        return best;
    }

    private static bool IsStop(int token, ReadOnlySpan<int> stopTokenIds)
    {
        foreach (int s in stopTokenIds)
            if (token == s) return true;
        return false;
    }
}
