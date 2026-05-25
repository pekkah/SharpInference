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
    /// </summary>
    public void Decode(int maxTokens, ReadOnlySpan<int> stopTokenIds, Action<int> emitToken,
                       CancellationToken ct = default)
    {
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
            _totalDraftsEmitted++;

            // ── Main verify (advances main caches through t1) ────────
            ReadOnlySpan<float> mainLogits = _fwd.Forward(t1, P);
            int t2Target = ArgMax(mainLogits);

            int t2 = (t2Target == t2Draft) ? t2Draft : t2Target;
            if (t2Target == t2Draft) _totalDraftsAccepted++;

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
            // The mismatch case still needs an MTP cache update at position
            // P+1 conditioned on the corrected token (t2_target), so the cache
            // state matches what a re-derivation would produce. (mtp cache
            // before this was advanced via t1 at position P — that's correct.)
            ReadOnlySpan<float> mainLogitsAfter = _fwd.Forward(t2, P + 1);
            _ = _fwd.MtpForward(t2, P + 1, _fwd.LastHidden);

            // ── Update saved state for next iter ─────────────────────
            _fwd.LastHidden.CopyTo(_savedHidden);
            mainLogitsAfter.CopyTo(_savedMainLogits);
            _nextPos = P + 2;
        }
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
