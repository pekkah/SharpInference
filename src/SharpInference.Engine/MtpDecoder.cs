using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Multi-Token Prediction (MTP / NEXTN) self-speculative decoder for hybrid
/// GDN models that ship an MTP head (e.g. Qwen3.6-27B-MTP). The draft model
/// is the MTP head of the same network — not a separate weights file — so
/// the vocab, tokenizer, and trunk state are all shared.
///
/// <para><b>Algorithm (v1, sequential N=1).</b></para>
///
/// The main pass and the MTP head each maintain their own KV/state caches
/// that advance in lockstep, one position per emitted token. Per iteration:
///
/// <list type="number">
///   <item><c>t1 = argmax(saved_main_logits)</c> — already correct by greedy
///         construction (= argmax of the previous step's main logits).
///         <c>t1</c> is emitted as the first token of the iter.</item>
///   <item>MTP draft: <c>mtp_logits = mtp.Forward(t1, P, h_prev)</c>; advances
///         the MTP KV cache by one position. <c>t2_draft = argmax(mtp_logits)</c>.</item>
///   <item>Main verify: <c>l_main = main.Forward(t1, P)</c>; advances the
///         main caches (KV + GDN) by one position. <c>t2_target = argmax(l_main)</c>.</item>
///   <item>Accept iff <c>t2_target == t2_draft</c>; emit
///         (<c>t2_draft</c> on accept, else <c>t2_target</c>) as the second token.</item>
///   <item>Commit: <c>main.Forward(t2_emitted, P+1)</c> + <c>mtp.Forward(t2_emitted, P+1, h)</c>.
///         These also produce <c>saved_main_logits</c> and the new <c>h_prev</c>
///         for the next iter.</item>
/// </list>
///
/// <para><b>Speedup envelope.</b></para>
/// Sequential N=1 emits 2 tokens per 2 main forwards + 2 MTP forwards. With
/// MTP ~1/64 the cost of a main forward, the per-iteration wall time is
/// roughly the same as 2 baseline forwards — i.e. v1 is near-baseline, not
/// 1.3×. Hitting the issue #25 acceptance criterion (≥1.3×) requires a
/// batched main verify pass (Phase 7 optimization) plus a per-token GDN
/// snapshot ring (Phase 11.7 / Risk #6). Both are tracked as follow-ups.
///
/// <para><b>State invariants this class relies on.</b></para>
/// <list type="bullet">
///   <item>The main pass exposes <see cref="IForwardPass.LastHidden"/> — the
///         post-trunk pre-final-norm hidden of the most recent <see cref="IForwardPass.Forward"/>
///         call. Refreshed in lockstep with the cache.</item>
///   <item>The MTP head shares the main pass's lm_head (output projection)
///         and embedding table; only the per-block weights differ.</item>
///   <item>This decoder does NOT take ownership of <paramref name="fwd"/>.</item>
/// </list>
/// </summary>
public sealed class MtpDecoder
{
    private readonly IForwardPass _fwd;

    private int _nextPos;
    private readonly float[] _savedMainLogits;
    private float[] _savedHidden;

    // Acceptance statistics (per Decode call cumulative)
    private long _totalDraftsEmitted;
    private long _totalDraftsAccepted;

    public MtpDecoder(IForwardPass fwd)
    {
        ArgumentNullException.ThrowIfNull(fwd);
        if (!fwd.HasMtpHead)
            throw new ArgumentException(
                $"MtpDecoder requires the forward pass to ship an MTP head; " +
                $"{fwd.GetType().Name} reports HasMtpHead == false.", nameof(fwd));

        _fwd = fwd;
        _savedMainLogits = new float[fwd.VocabSize];
        // _savedHidden is lazily sized in Initialize so this constructor doesn't
        // depend on prior LastHidden state. EmbeddingDim isn't exposed on the
        // interface.
        _savedHidden = [];
    }

    /// <summary>Number of MTP drafts emitted as accepted tokens.</summary>
    public long TotalDraftsAccepted => _totalDraftsAccepted;

    /// <summary>Total MTP drafts considered (= total iterations).</summary>
    public long TotalDraftsEmitted => _totalDraftsEmitted;

    /// <summary>Acceptance rate over the cumulative run (0..1).</summary>
    public float AcceptanceRate => _totalDraftsEmitted > 0
        ? (float)_totalDraftsAccepted / _totalDraftsEmitted
        : 0f;

    /// <summary>
    /// Initialise the decoder after the forward pass has consumed the prompt.
    /// Call after <see cref="IForwardPass.Prefill"/> (or the final prompt-token
    /// <see cref="IForwardPass.Forward"/>) returns.
    /// </summary>
    /// <param name="nextPosition">Position of the first token to emit (= prompt length).</param>
    /// <param name="lastMainLogits">Logits from the final prompt-token forward (predicts <paramref name="nextPosition"/>).</param>
    public void Initialize(int nextPosition, ReadOnlySpan<float> lastMainLogits)
    {
        if (lastMainLogits.Length != _savedMainLogits.Length)
            throw new ArgumentException(
                $"lastMainLogits length {lastMainLogits.Length} != vocab size {_savedMainLogits.Length}.",
                nameof(lastMainLogits));

        _nextPos = nextPosition;
        lastMainLogits.CopyTo(_savedMainLogits);
        // Snapshot the main pass's current LastHidden so the first draft can use it.
        var h = _fwd.LastHidden;
        if (h.IsEmpty)
            throw new InvalidOperationException(
                "LastHidden is empty — the main forward pass has not produced any hidden " +
                "state yet. Call Prefill or Forward before Initialize.");
        if (_savedHidden.Length != h.Length)
            _savedHidden = new float[h.Length];
        h.CopyTo(_savedHidden);
        _totalDraftsEmitted = 0;
        _totalDraftsAccepted = 0;
    }

    /// <summary>
    /// Decode up to <paramref name="maxTokens"/> tokens. Calls <paramref name="emitToken"/>
    /// for every accepted or correction token. Stops when a token in
    /// <paramref name="stopTokenIds"/> is generated (and does NOT emit the stop token).
    /// Dispatches to the batched N=2 verify path (<see cref="DecodeBatched"/>) when the
    /// underlying forward pass implements <see cref="IForwardPass.BatchForward2"/>;
    /// otherwise falls back to the sequential N=1 algorithm.
    /// </summary>
    /// <param name="pMin">Min draft probability for probabilistic accept under MTP
    /// verification (llama.cpp's <c>--spec-draft-p-min</c>, issue #38). <c>1.0</c>
    /// (default) is byte-perfect argmax-match — accept iff <c>draft == argmax(target)</c>.
    /// <c>p ∈ (0, 1)</c> also accepts when <c>softmax(target)[draft] &gt;= p</c>; the
    /// emitted token then equals the draft, which can differ from baseline greedy.
    /// <c>0.0</c> or negative is treated as <c>1.0</c>.</param>
    public void Decode(int maxTokens, ReadOnlySpan<int> stopTokenIds, Action<int> emitToken,
                       float pMin = 1f,
                       CancellationToken ct = default)
    {
        // Treat 0 and negative as the default (argmax-match) for back-compat with
        // callers that left SpecDraftPMin at the previous default of 0f.
        if (pMin <= 0f) pMin = 1f;
        if (_fwd.SupportsBatchVerify)
        {
            DecodeBatched(maxTokens, stopTokenIds, emitToken, pMin, ct);
            return;
        }
        DecodeSequential(maxTokens, stopTokenIds, emitToken, pMin, ct);
    }

    private void DecodeSequential(int maxTokens, ReadOnlySpan<int> stopTokenIds, Action<int> emitToken,
                                  float pMin, CancellationToken ct)
    {
        bool trace = Environment.GetEnvironmentVariable("SHARPI_TRACE_MTP") == "1";
        int generated = 0;
        while (generated < maxTokens)
        {
            ct.ThrowIfCancellationRequested();

            // Each iter emits up to 2 tokens.
            int remaining = maxTokens - generated;

            int t1 = ArgMax(_savedMainLogits);
            if (IsStop(t1, stopTokenIds)) return;
            emitToken(t1); generated++;
            if (generated >= maxTokens) return;

            // ── MTP draft (uses saved hidden + just-emitted t1) ──────
            int P = _nextPos;
            ReadOnlySpan<float> mtpLogits = _fwd.MtpForward(t1, P, _savedHidden);
            int t2Draft = ArgMax(mtpLogits);
            float t2DraftLogit = mtpLogits[t2Draft];
            _totalDraftsEmitted++;

            // ── Main verify (advances main caches through t1) ────────
            ReadOnlySpan<float> mainLogits = _fwd.Forward(t1, P);
            int t2Target = ArgMax(mainLogits);

            bool accept = AcceptDraft(t2Draft, t2Target, mainLogits, pMin,
                                      out float draftProbInMain);
            int t2 = accept ? t2Draft : t2Target;
            if (accept) _totalDraftsAccepted++;

            if (trace)
            {
                float draftLogitInMain = mainLogits[t2Draft];
                float mainTopLogit = mainLogits[t2Target];
                Console.Error.WriteLine(
                    $"[mtp] P={P} t1={t1} t2_draft={t2Draft}(draft_logit={t2DraftLogit:F3}, main_logit_at_draft={draftLogitInMain:F3}, p={draftProbInMain:F3}) " +
                    $"t2_target={t2Target}(main_logit={mainTopLogit:F3}) " +
                    $"{(accept ? "ACCEPT" : "reject")}");
            }

            if (IsStop(t2, stopTokenIds))
            {
                // Don't emit the stop, but DO sync the main pass's hidden +
                // logits so a follow-up call resumes from a consistent state.
                _fwd.LastHidden.CopyTo(_savedHidden);
                mainLogits.CopyTo(_savedMainLogits);
                _nextPos = P + 1;
                return;
            }
            emitToken(t2); generated++;
            if (generated >= maxTokens)
            {
                _fwd.LastHidden.CopyTo(_savedHidden);
                mainLogits.CopyTo(_savedMainLogits);
                _nextPos = P + 1;
                return;
            }

            // ── Commit t2 to BOTH caches so they stay in lockstep ────
            // Order matters: the MTP commit at position P+1 needs prev_hidden
            // = h@P (the hidden from BEFORE main consumed t2 at position P+1).
            // After the t1 verify Forward above, _fwd.LastHidden already holds
            // h@P — call MtpForward FIRST, then run main commit which will
            // overwrite LastHidden with h@P+1 for the next iter.
            //
            // The mismatch case still gets a valid MTP cache update at P+1:
            // it's conditioned on t2 (the actually-emitted token, draft or
            // target) so subsequent mtp forwards see consistent K/V history.
            _ = _fwd.MtpForward(t2, P + 1, _fwd.LastHidden);
            ReadOnlySpan<float> mainLogitsAfter = _fwd.Forward(t2, P + 1);

            // ── Update saved state for next iter ─────────────────────
            // After main.Forward(t2, P+1), LastHidden = h@P+1 — exactly the
            // "previous hidden" the next iter's MTP draft will want.
            _fwd.LastHidden.CopyTo(_savedHidden);
            mainLogitsAfter.CopyTo(_savedMainLogits);
            _nextPos = P + 2;
        }
    }

    /// <summary>
    /// Batched N=2 verify (issue #30). Per iter:
    ///   t1 = argmax(saved_main_logits); emit t1
    ///   t2_draft = argmax(MtpForward(t1, P, h_prev))
    ///   BatchForward2(t1, t2_draft, P) → (l@P+1, l@P+2), LastHiddenT1 = h@P
    ///   if argmax(l@P+1) == t2_draft: accept; saved_main_logits = l@P+2; emit t2_draft
    ///   else: reject; RestoreBatchSnapshot(P+1); MtpTruncateTo(P+1);
    ///         Forward(t2_target, P+1) → l@P+2; saved_main_logits = l@P+2; emit t2_target
    ///   MtpForward(t2_emitted, P+1, LastHiddenT1)  # commit MTP at P+1
    ///   _savedHidden = LastHidden   # h@P+1 ready for next iter
    /// </summary>
    private void DecodeBatched(int maxTokens, ReadOnlySpan<int> stopTokenIds, Action<int> emitToken,
                               float pMin, CancellationToken ct)
    {
        bool trace = Environment.GetEnvironmentVariable("SHARPI_TRACE_MTP") == "1";
        // Per-iter copy of h@P (LastHiddenT1) so MtpForward's scratch can't
        // disturb the slice between batched verify and MTP commit. Sized to the
        // embedding dim (length of _savedHidden); allocated once outside the loop.
        Span<float> hAtPCopy = new float[_savedHidden.Length];
        int generated = 0;
        while (generated < maxTokens)
        {
            ct.ThrowIfCancellationRequested();

            // ── Token 1: argmax of last main logits (greedy correctness) ──
            int t1 = ArgMax(_savedMainLogits);
            if (IsStop(t1, stopTokenIds)) return;
            emitToken(t1); generated++;
            if (generated >= maxTokens) return;

            // ── MTP draft for position P+1 ────────────────────────────
            int P = _nextPos;
            ReadOnlySpan<float> mtpLogits = _fwd.MtpForward(t1, P, _savedHidden);
            int t2Draft = ArgMax(mtpLogits);
            float t2DraftLogit = mtpLogits[t2Draft];
            _totalDraftsEmitted++;

            // ── Batched main verify (advances main caches through t1 + t2_draft) ─
            _fwd.BatchForward2(t1, t2Draft, P,
                out ReadOnlySpan<float> l_atPplus1,
                out ReadOnlySpan<float> l_atPplus2);
            int t2Target = ArgMax(l_atPplus1);

            // Snapshot LastHiddenT1 — h@P (t1's pre-output-norm hidden). Needed for
            // the MTP commit at P+1 regardless of accept/reject. We copy out now
            // because a subsequent Forward (reject path) doesn't overwrite it but
            // the value's tied to the batched forward's scratch.
            var hAtP = _fwd.LastHiddenT1;
            if (hAtP.Length != _savedHidden.Length)
                throw new InvalidOperationException(
                    $"LastHiddenT1 length {hAtP.Length} != EmbeddingDim {_savedHidden.Length}.");
            // Use a local copy so MtpForward (which writes its own scratch) can't
            // disturb the slice between now and the commit call.
            hAtP.CopyTo(hAtPCopy);

            ReadOnlySpan<float> mainLogitsAfter;
            int t2;
            bool accept = AcceptDraft(t2Draft, t2Target, l_atPplus1, pMin,
                                      out float draftProbInMain);
            if (accept)
            {
                _totalDraftsAccepted++;
                // Emit the draft (which equals argmax on argmax-match, or differs
                // from argmax on a prob-only accept under pMin < 1.0). The batched
                // forward has already advanced both caches through t2_draft, so
                // l_at_P+2 is the next iter's saved_main_logits.
                t2 = t2Draft;
                mainLogitsAfter = l_atPplus2;
            }
            else
            {
                t2 = t2Target;
                _fwd.RestoreBatchSnapshot(P + 1);
                _fwd.MtpTruncateTo(P + 1);
                mainLogitsAfter = _fwd.Forward(t2Target, P + 1);
            }

            if (trace)
            {
                float draftLogitInMain = l_atPplus1[t2Draft];
                float mainTopLogit = l_atPplus1[t2Target];
                Console.Error.WriteLine(
                    $"[mtp-batch] P={P} t1={t1} t2_draft={t2Draft}(draft_logit={t2DraftLogit:F3}, main_logit_at_draft={draftLogitInMain:F3}, p={draftProbInMain:F3}) " +
                    $"t2_target={t2Target}(main_logit={mainTopLogit:F3}) " +
                    $"{(accept ? "ACCEPT" : "reject")}");
            }

            if (IsStop(t2, stopTokenIds))
            {
                // Keep state consistent for a follow-up call.
                _fwd.LastHidden.CopyTo(_savedHidden);
                mainLogitsAfter.CopyTo(_savedMainLogits);
                _nextPos = P + 2;
                return;
            }
            emitToken(t2); generated++;
            if (generated >= maxTokens)
            {
                _fwd.LastHidden.CopyTo(_savedHidden);
                mainLogitsAfter.CopyTo(_savedMainLogits);
                _nextPos = P + 2;
                return;
            }

            // ── MTP commit at P+1 ────────────────────────────────────
            // prevHidden = h@P (the hidden that came out of the trunk for t1).
            _ = _fwd.MtpForward(t2, P + 1, hAtPCopy);

            // Update saved state for next iter. _fwd.LastHidden = h@P+1.
            _fwd.LastHidden.CopyTo(_savedHidden);
            mainLogitsAfter.CopyTo(_savedMainLogits);
            _nextPos = P + 2;
        }
    }

    /// <summary>
    /// MTP draft acceptance check (issue #38). Always accepts when the draft is the
    /// verifier's argmax. Under <paramref name="pMin"/> &lt; 1.0, ALSO accepts when
    /// the draft's softmax probability under <paramref name="mainLogits"/> meets the
    /// threshold. The softmax is computed in numerically-stable max-shift form; only
    /// the draft's probability is materialised (no full distribution write).
    /// </summary>
    /// <param name="draft">Token id proposed by the MTP head.</param>
    /// <param name="target">Argmax of <paramref name="mainLogits"/> (= the greedy
    /// correction token).</param>
    /// <param name="mainLogits">Verifier logits at the position the draft predicts.</param>
    /// <param name="pMin">Acceptance threshold; <c>1.0</c> = argmax-match only.</param>
    /// <param name="draftProb">Computed <c>softmax(mainLogits)[draft]</c>, or <c>0</c>
    /// when no softmax was needed (pMin &gt;= 1.0). Surfaced for trace logging.</param>
    private static bool AcceptDraft(int draft, int target, ReadOnlySpan<float> mainLogits,
                                    float pMin, out float draftProb)
    {
        if (draft == target)
        {
            draftProb = 0f;
            return true;
        }
        if (pMin >= 1.0f)
        {
            draftProb = 0f;
            return false;
        }
        draftProb = SoftmaxProbAt(mainLogits, draft);
        return draftProb >= pMin;
    }

    /// <summary>
    /// Numerically-stable softmax probability of <paramref name="idx"/> under
    /// <paramref name="logits"/>: one max-pass, one exp-sum-pass, then the single
    /// ratio. O(2V) reads; vocab fits in L2 so latency is negligible per MTP iter.
    /// </summary>
    private static float SoftmaxProbAt(ReadOnlySpan<float> logits, int idx)
    {
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        double sumExp = 0;
        for (int i = 0; i < logits.Length; i++)
            sumExp += Math.Exp(logits[i] - max);

        if (sumExp <= 0 || double.IsNaN(sumExp)) return 0f;
        return (float)(Math.Exp(logits[idx] - max) / sumExp);
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        }
        return best;
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
