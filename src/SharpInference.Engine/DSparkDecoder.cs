using System.Diagnostics;
using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// DSpark speculative decoder (docs/dspark-plan.md, PR #413): drives an
/// <see cref="IDSparkDraft"/> block drafter against a rewindable target pass
/// with hidden-state taps enabled.
///
/// Per step (folded verify, same shape as <see cref="MtpDecoder"/>'s batched
/// loop): emit the certain token t1 = argmax(saved logits) at position P, draft
/// one block anchored on it, trim the proposal to its confident prefix, verify
/// [t1, drafts…] in ONE batched target pass, accept the leading argmax-matching
/// prefix, rewind the target's KV to the commit point via
/// <see cref="IForwardPass.TruncateTo"/>, feed the newly verified positions'
/// taps to the draft, and save the verifier's logits at the commit point as the
/// next step's certain token. Greedy-parity: every emitted token equals
/// argmax(target logits), byte-identical to plain greedy decode.
///
/// The caller must <see cref="IForwardPass.EnableHiddenTaps"/> on the target
/// with the draft's <c>TargetLayerIds</c> BEFORE prefilling, then call
/// <see cref="Initialize"/> with the prefill logits. Temperature-0 only (the
/// CLI gates it exactly like MTP).
/// </summary>
public sealed class DSparkDecoder
{
    private readonly IForwardPass _target;
    private readonly IDSparkDraft _draft;

    private int _nextPos;
    private readonly float[] _savedLogits;
    // #219 fast-path carry: on the argmax verify path only the next certain
    // token's index is carried, not full logits (mirrors MtpDecoder).
    private int _savedArgmax;
    private bool _hasSavedArgmax;
    private float[] _tapScratch = [];

    private long _totalDraftsEmitted;
    private long _totalDraftsAccepted;
    private readonly Stopwatch _phaseSw = new();

    /// <summary>Milliseconds spent producing draft blocks.</summary>
    public double DraftMs { get; private set; }

    /// <summary>Milliseconds spent in batched target verification.</summary>
    public double VerifyMs { get; private set; }

    /// <summary>Milliseconds spent in rollback + draft context refresh.</summary>
    public double CommitMs { get; private set; }

    public long TotalDraftsEmitted => _totalDraftsEmitted;
    public long TotalDraftsAccepted => _totalDraftsAccepted;
    public float AcceptanceRate =>
        _totalDraftsEmitted > 0 ? (float)_totalDraftsAccepted / _totalDraftsEmitted : 0f;

    public DSparkDecoder(IForwardPass target, IDSparkDraft draft)
    {
        _target = target;
        _draft = draft;
        if (target.VocabSize != draft.VocabSize)
            throw new ArgumentException(
                $"Target vocab ({target.VocabSize}) != draft vocab ({draft.VocabSize}); " +
                "the DSpark head must be trained for this target model.");
        if (!target.SupportsPartialRewind)
            throw new ArgumentException(
                "DSpark requires a target with partial-rewind TruncateTo " +
                "(rejected draft positions are rolled back mid-sequence).", nameof(target));
        if (!target.SupportsHiddenTaps)
            throw new ArgumentException(
                "DSpark requires a target with hidden-state taps " +
                "(check IForwardPass.SupportsHiddenTaps).", nameof(target));
        _savedLogits = new float[target.VocabSize];
    }

    /// <summary>
    /// Resolve the per-step verify-length cap: explicit flag (&gt;= 1) &gt;
    /// SHARPI_DSPARK_VERIFY_LEN &gt; 0 (uncapped — confidence trim decides).
    /// </summary>
    public static int ResolveVerifyLen(int flagValue)
    {
        if (flagValue >= 1) return flagValue;
        var s = Environment.GetEnvironmentVariable("SHARPI_DSPARK_VERIFY_LEN");
        if (s is not null && int.TryParse(s, out var v) && v >= 1) return v;
        return 0;
    }

    /// <summary>
    /// Resolve the confidence floor: explicit flag (&gt;= 0) &gt;
    /// SHARPI_DSPARK_MIN_CONFIDENCE &gt; 0 (disabled — verify the whole block).
    /// Pass a negative flag value for "unset".
    /// </summary>
    public static float ResolveMinConfidence(float flagValue)
    {
        if (flagValue >= 0f) return flagValue;
        var s = Environment.GetEnvironmentVariable("SHARPI_DSPARK_MIN_CONFIDENCE");
        if (s is not null && float.TryParse(s,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var v)
            && v >= 0f && v <= 1f)
            return v;
        return 0f;
    }

    /// <summary>
    /// Prime the decoder after prefill: <paramref name="promptLength"/> tokens
    /// are in the target's cache (with taps captured for positions
    /// 0..promptLength-1) and <paramref name="lastMainLogits"/> predicts the
    /// token at position promptLength. Feeds all prompt taps to the draft.
    /// </summary>
    public void Initialize(int promptLength, ReadOnlySpan<float> lastMainLogits)
    {
        if (lastMainLogits.Length != _savedLogits.Length)
            throw new ArgumentException(
                $"Expected {_savedLogits.Length} logits, got {lastMainLogits.Length}.",
                nameof(lastMainLogits));
        if (promptLength < 1)
            throw new ArgumentOutOfRangeException(nameof(promptLength));
        if (_target.HiddenTapDim != _draft.TapDim)
            throw new InvalidOperationException(
                $"Target tap dim {_target.HiddenTapDim} != draft tap dim {_draft.TapDim}. " +
                "EnableHiddenTaps must be called with the draft's TargetLayerIds before prefill.");
        if (_target.HiddenTapsAt(promptLength - 1).IsEmpty)
            throw new InvalidOperationException(
                "No hidden taps captured for the prompt — EnableHiddenTaps must be " +
                "called BEFORE Prefill.");

        lastMainLogits.CopyTo(_savedLogits);
        _hasSavedArgmax = false;
        _nextPos = promptLength;
        _draft.ResetContext();
        AppendTapsRange(0, promptLength);

        _totalDraftsEmitted = 0;
        _totalDraftsAccepted = 0;
        DraftMs = VerifyMs = CommitMs = 0;
    }

    /// <summary>
    /// Generate up to <paramref name="maxTokens"/> tokens, invoking
    /// <paramref name="emitToken"/> per token. Stop tokens are never emitted.
    /// </summary>
    /// <param name="minConfidence">Confidence floor: draft positions whose
    /// predicted acceptance probability falls below it are trimmed from the
    /// verify batch (0 = verify the full block).</param>
    /// <param name="verifyLenCap">Hard cap on drafts verified per step
    /// (0 = block size).</param>
    public void Decode(int maxTokens, ReadOnlySpan<int> stopTokenIds, Action<int> emitToken,
        float minConfidence = 0f, int verifyLenCap = 0, CancellationToken ct = default)
    {
        int generated = 0;
        while (generated < maxTokens)
        {
            ct.ThrowIfCancellationRequested();

            int P = _nextPos;
            // The draft's ctx window can be smaller than the target's — the
            // commit below appends taps up to newPos, so stop where either ends.
            if (P >= _target.MaxSeqLen || P >= _draft.MaxContext) return;

            // ── Certain token: argmax of the saved logits ─────────────
            int t1 = _hasSavedArgmax ? _savedArgmax : ArgMax(_savedLogits);
            if (IsStop(t1, stopTokenIds)) return;
            emitToken(t1); generated++;

            // ── Hard caps on drafts this step (remaining token budget,
            // batch capacity, both context windows, verify-length flag).
            // A step never verifies tokens it can't emit. ───────────────
            int maxDrafts = Math.Min(maxTokens - generated, _target.MaxBatchVerifyTokens - 1);
            maxDrafts = Math.Min(maxDrafts, _target.MaxSeqLen - P - 1);
            maxDrafts = Math.Min(maxDrafts, _draft.MaxContext - P - 1);
            if (verifyLenCap > 0) maxDrafts = Math.Min(maxDrafts, verifyLenCap);
            maxDrafts = Math.Max(maxDrafts, 0);

            // ── Draft one block anchored at (t1, P) — skipped when the caps
            // already forbid any draft (a full block forward would be wasted).
            int kDraft = 0;
            DSparkProposal proposal = default;
            if (maxDrafts > 0)
            {
                _phaseSw.Restart();
                proposal = _draft.ProposeBlock(t1, P);
                DraftMs += _phaseSw.Elapsed.TotalMilliseconds;

                // Validate the proposal shape up front: a default(DSparkProposal) or a
                // confidence array shorter than the tokens would otherwise surface as an
                // opaque NRE / IndexOutOfRange deep in the trim loop.
                if (proposal.Tokens is null)
                    throw new InvalidOperationException(
                        $"{_draft.GetType().Name}.ProposeBlock returned a default proposal (null Tokens).");
                if (minConfidence > 0f
                    && (proposal.Confidences is null || proposal.Confidences.Length < proposal.Tokens.Length))
                    throw new InvalidOperationException(
                        $"{_draft.GetType().Name}.ProposeBlock returned {proposal.Confidences?.Length ?? 0} " +
                        $"confidences for {proposal.Tokens.Length} tokens; the confidence trim needs one per draft.");

                kDraft = proposal.Tokens.Length;
                if (minConfidence > 0f)
                {
                    int confident = 0;
                    while (confident < kDraft && proposal.Confidences[confident] >= minConfidence)
                        confident++;
                    kDraft = confident;
                }
                kDraft = Math.Min(kDraft, maxDrafts);
            }

            var tokens = new int[1 + kDraft];
            tokens[0] = t1;
            for (int i = 0; i < kDraft; i++) tokens[i + 1] = proposal.Tokens[i];
            _totalDraftsEmitted += kDraft;

            // ── ONE batched target verify over tokens[0..k]. Greedy verify only
            //    needs per-position argmaxes, so passes with a device-side row
            //    argmax (#219) skip the k×vocab logits download entirely. ──
            _phaseSw.Restart();
            bool argmaxFast = _target.SupportsBatchVerify && _target.SupportsBatchVerifyArgmax;
            float[][]? batch = null;
            (int Index, float Value)[]? verifyArgmax = null;
            if (argmaxFast) verifyArgmax = _target.BatchVerifyArgmax(tokens, P);
            else            batch = VerifyTokens(tokens, P);
            VerifyMs += _phaseSw.Elapsed.TotalMilliseconds;

            int Target(int pos) => argmaxFast ? verifyArgmax![pos].Index : ArgMax(batch![pos]);

            // ── Greedy accept: leading argmax-matching drafts. An accepted
            //    STOP draft ends the chain, EXCLUDED from `a` (neither emitted
            //    nor committed), mirroring MtpDecoder's stop contract. ──
            int a = 0;
            bool stopHit = false;
            for (int i = 1; i <= kDraft; i++)
            {
                if (tokens[i] != Target(i - 1)) break;
                if (IsStop(tokens[i], stopTokenIds)) { stopHit = true; break; }
                a++;
            }
            _totalDraftsAccepted += a;
            int newPos = P + 1 + a;

            // ── Roll back the rejected tail; sync the draft's context ──
            _phaseSw.Restart();
            _target.TruncateTo(newPos);
            AppendTapsRange(P, a + 1);
            CommitMs += _phaseSw.Elapsed.TotalMilliseconds;

            // ── Emit accepted drafts ──────────────────────────────────
            for (int i = 1; i <= a; i++)
            {
                emitToken(tokens[i]); generated++;
            }

            // Position `a`'s verify result predicts newPos: the correction on a
            // reject, the continuation on full accept, or the (un-emitted) stop.
            if (argmaxFast) { _savedArgmax = verifyArgmax![a].Index; _hasSavedArgmax = true; }
            else            { batch![a].CopyTo(_savedLogits, 0); _hasSavedArgmax = false; }
            _nextPos = newPos;
            if (stopHit) return;
        }
    }

    /// <summary>
    /// Batched verify with a sequential fallback for passes whose
    /// <see cref="IForwardPass.SupportsBatchVerify"/> is (transiently) false.
    /// Both paths leave the target cache at startPos + tokens.Length and
    /// capture taps for every processed position.
    /// </summary>
    private float[][] VerifyTokens(int[] tokens, int startPos)
    {
        if (_target.SupportsBatchVerify)
            return _target.BatchVerify(tokens, startPos);

        var result = new float[tokens.Length][];
        for (int i = 0; i < tokens.Length; i++)
        {
            var logits = _target.Forward(tokens[i], startPos + i);
            result[i] = new float[_savedLogits.Length];
            logits.CopyTo(result[i]);
        }
        return result;
    }

    /// <summary>
    /// Feed target taps for positions [startPos, startPos+count) to the draft,
    /// in bounded chunks (a whole prompt's taps would be hundreds of MB flat).
    /// </summary>
    private void AppendTapsRange(int startPos, int count)
    {
        int tapDim = _draft.TapDim;
        const int ChunkRows = 128;
        int scratchLen = Math.Min(count, ChunkRows) * tapDim;
        if (_tapScratch.Length < scratchLen) _tapScratch = new float[scratchLen];

        int done = 0;
        while (done < count)
        {
            int rows = Math.Min(ChunkRows, count - done);
            for (int r = 0; r < rows; r++)
            {
                var tap = _target.HiddenTapsAt(startPos + done + r);
                if (tap.Length != tapDim)
                    throw new InvalidOperationException(
                        $"Missing hidden tap for position {startPos + done + r} " +
                        $"(got {tap.Length} floats, expected {tapDim}).");
                tap.CopyTo(_tapScratch.AsSpan(r * tapDim, tapDim));
            }
            _draft.AppendContext(_tapScratch.AsSpan(0, rows * tapDim), startPos + done, rows);
            done += rows;
        }
    }

    private static int ArgMax(ReadOnlySpan<float> logits) => Sampler.Greedy(logits);

    private static bool IsStop(int token, ReadOnlySpan<int> stopTokenIds) =>
        stopTokenIds.IndexOf(token) >= 0;
}
