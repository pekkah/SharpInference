using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Speculative decoding (greedy): a small draft model proposes tokens which the target model
/// verifies via a batched forward pass, accepting each where they agree and correcting at the
/// first divergence.
///
/// Each step packs the CERTAIN next token (argmax of the saved target logits) together with
/// k−1 draft proposals into ONE batched target pass (the llama.cpp formulation): the batch
/// yields both the verification logits and the next step's saved logits, so the target runs
/// exactly one batched pass per step — no separate correction-commit forward. On memory-bound
/// decode paths the batched pass costs ~1–2× a single forward (issue #194/#207 weight
/// amortization), so the speedup is ≈ E[tokens/step] / (cost_batch/cost_forward + draft
/// overhead), with E[tokens/step] = 1 + E[accepted of k−1] for per-token acceptance α.
///
/// Both target and draft must share the same tokenizer (same vocab size).
/// Note: does NOT take ownership of the forward pass instances.
/// </summary>
public sealed class SpeculativeDecoder
{
    private readonly IForwardPass _target;
    private readonly IForwardPass? _draft;          // model-draft mode
    private readonly PromptLookupDraft? _lookup;    // prompt-lookup mode (issue #207)
    private readonly bool _batchVerify;
    private int _lookahead;

    // Generation state. Invariant at step boundaries: both caches hold exactly _nextPos
    // positions and _savedTargetLogits are the target's logits after the token at
    // _nextPos−1 (so argmax(_savedTargetLogits) is the next emitted token, by greedy
    // construction). The draft's own last logits are not part of the state — each step's
    // first proposal requires forwarding the certain token through the draft anyway.
    private int _nextPos;
    private float[] _savedTargetLogits;

    // Acceptance statistics
    private long _totalAccepted;
    private long _totalEmitted;

    // Phase timing (issue #207 bench reporting): cumulative wall time spent drafting,
    // batch-verifying, and committing (truncate + correction forwards) across all steps.
    private readonly System.Diagnostics.Stopwatch _phaseSw = new();

    /// <summary>
    /// Prompt-lookup mode (issue #207): proposals come from <paramref name="lookup"/>
    /// (n-gram matches against prompt + generated history) instead of a draft model —
    /// zero draft-forward cost, and a step with no match degrades to plain decode.
    /// Initialize with the prompt-token overload so the lookup sees the prompt.
    /// </summary>
    public SpeculativeDecoder(IForwardPass target, PromptLookupDraft lookup, int lookahead = 4)
    {
        ArgumentNullException.ThrowIfNull(lookup);
        if (!target.SupportsPartialRewind)
            throw new ArgumentException(
                $"Speculative decoding requires the target forward pass to support partial rewind; " +
                $"{target.GetType().Name} does not.",
                nameof(target));
        _target = target;
        _lookup = lookup;
        _batchVerify = Environment.GetEnvironmentVariable("SHARPI_SPEC_BATCH_VERIFY") != "0";
        _lookahead = Math.Max(1, lookahead);
        _savedTargetLogits = new float[target.VocabSize];
    }

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

    /// <summary>Cumulative milliseconds spent in cache truncation + draft-sync forwards.</summary>
    public double CommitMs { get; private set; }

    /// <summary>
    /// Initialize the decoder after both models have processed the prompt.
    /// Call after prefilling both target and draft with the same prompt tokens.
    /// </summary>
    /// <param name="prefillLength">Number of prompt tokens (= new KV cache length).</param>
    /// <param name="targetLogits">Logits from the target's last prefill step (vocab-size span).</param>
    /// <param name="draftLogits">Accepted for call-site compatibility but no longer consulted:
    /// each step's first draft proposal requires forwarding the (certain) next token through
    /// the draft anyway, which produces fresher logits than the prefill tail. The draft must
    /// still be PREFILLED before calling this — its cache has to hold the prompt.</param>
    public void Initialize(int prefillLength, ReadOnlySpan<float> targetLogits, ReadOnlySpan<float> draftLogits = default)
    {
        _nextPos = prefillLength;
        targetLogits.CopyTo(_savedTargetLogits);
        _totalAccepted = 0;
        _totalEmitted = 0;
        DraftMs = 0;
        VerifyMs = 0;
        CommitMs = 0;
    }

    /// <summary>
    /// Prompt-lookup-mode initialization: seeds the lookup history with the prompt tokens
    /// (proposals match against prompt + generated text). Call after prefilling the target.
    /// </summary>
    /// <param name="promptTokens">The prompt as fed to the target's prefill.</param>
    /// <param name="targetLogits">Logits from the target's last prefill step.</param>
    public void Initialize(IReadOnlyList<int> promptTokens, ReadOnlySpan<float> targetLogits)
    {
        if (_lookup is null)
            throw new InvalidOperationException(
                "Prompt-token initialization is only valid in prompt-lookup mode; " +
                "use Initialize(prefillLength, targetLogits, draftLogits) with a draft model.");
        ArgumentNullException.ThrowIfNull(promptTokens);
        _lookup.Reset(promptTokens);
        Initialize(promptTokens.Count, targetLogits);
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
    /// Run one speculative step (llama.cpp formulation): pack the certain next token with
    /// k−1 draft proposals, batch-verify all k in ONE target pass, accept greedily.
    /// Returns the emitted token array (1 + accepted tokens). Updates internal state
    /// (_nextPos, _savedTargetLogits).
    /// </summary>
    private int[] Step(int k)
    {
        int P = _nextPos;

        // tokens[0] is CERTAIN (greedy argmax of the saved target logits — it would be
        // emitted by plain decode too); tokens[1..] are draft proposals chained from it.
        int n0 = ArgMax(_savedTargetLogits);

        // ── Draft phase ──────────────────────────────────────────────────────────
        _phaseSw.Restart();
        int[] tokens;
        if (_lookup is not null)
        {
            // Prompt-lookup: tokens[0] is certain, so it joins the history before
            // matching; proposals are whatever the tail n-gram match yields (possibly
            // none — then the step is a plain single-token decode).
            _lookup.Append(n0);
            int[] proposals = _lookup.Propose(k - 1);
            tokens = new int[1 + proposals.Length];
            tokens[0] = n0;
            proposals.CopyTo(tokens, 1);
        }
        else
        {
            // Model draft: k−1 draft forwards propose tokens[1..k-1]; the draft cache
            // advances to P + k - 1 (appended tokens[0..k-2]).
            tokens = new int[k];
            tokens[0] = n0;
            for (int i = 1; i < k; i++)
            {
                var logits = _draft!.Forward(tokens[i - 1], P + i - 1);
                tokens[i] = ArgMax(logits);
            }
        }
        DraftMs += _phaseSw.Elapsed.TotalMilliseconds;

        // ── Target batch-verify ──────────────────────────────────────────────────
        // One packed pass over tokens[0..k-1] at positions P..P+k-1.
        // batch[i] = logits AFTER tokens[i]. Target cache advances to P + k.
        _phaseSw.Restart();
        float[][] batch = BatchVerifyTarget(tokens, P);
        VerifyMs += _phaseSw.Elapsed.TotalMilliseconds;

        // ── Greedy accept ────────────────────────────────────────────────────────
        // tokens[i] (i ≥ 1) is accepted iff the target's logits after tokens[i-1]
        // pick it. tokens[0] needs no check (it IS the target's pick).
        int accepted = 0;
        for (int i = 1; i < tokens.Length; i++)
        {
            if (ArgMax(batch[i - 1]) == tokens[i]) accepted++;
            else break;
        }

        _totalAccepted += accepted;
        _totalEmitted += accepted + 1;

        // ── Roll both caches back to the last emitted token ──────────────────────
        // Emitted: tokens[0..accepted] → new length newPos. The batch already holds the
        // logits after the last emitted token (batch[accepted]) — they seed the next step,
        // so NO separate correction forward is needed on the target.
        _phaseSw.Restart();
        int newPos = P + 1 + accepted;
        _target.TruncateTo(newPos);

        if (_lookup is not null)
        {
            // No draft cache to manage; record the accepted proposals (tokens[0] was
            // appended before matching) so future tail n-grams can match them.
            for (int i = 1; i <= accepted; i++) _lookup.Append(tokens[i]);
        }
        else if (accepted == tokens.Length - 1)
        {
            // Fully accepted: the draft never processed tokens[k-1] (its cache is at
            // P+k-1 = newPos-1). Sync it so the next step's draft chain starts at newPos.
            _draft!.Forward(tokens[^1], P + tokens.Length - 1);
        }
        else
        {
            _draft!.TruncateTo(newPos);
        }
        CommitMs += _phaseSw.Elapsed.TotalMilliseconds;

        // ── Update state ─────────────────────────────────────────────────────────
        _nextPos = newPos;
        batch[accepted].CopyTo(_savedTargetLogits, 0);

        // ── Emitted token list: the certain token + accepted proposals ───────────
        var emitted = new int[accepted + 1];
        for (int i = 0; i <= accepted; i++) emitted[i] = tokens[i];
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

    // Greedy selection delegates to the engine's single argmax implementation —
    // spec-decode parity contracts hinge on draft selection, verification, and
    // production sampling all breaking ties identically (first max wins).
    private static int ArgMax(ReadOnlySpan<float> logits) => Sampler.Greedy(logits);

    private static bool IsStop(int token, ReadOnlySpan<int> stopTokenIds)
    {
        foreach (int s in stopTokenIds)
            if (token == s) return true;
        return false;
    }
}
