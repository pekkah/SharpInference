using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Multi-Token Prediction (MTP / NEXTN) self-speculative decoder for hybrid
/// GDN models that ship an MTP head (e.g. Qwen3.6-27B-MTP). The draft model
/// is the MTP head of the same network — not a separate weights file — so
/// the vocab, tokenizer, and trunk state are all shared.
///
/// <para><b>Primary algorithm — folded k-token batched verify (issues #30/#207).</b>
/// Per step, the certain next token (argmax of the saved main logits) plus a
/// chain of <c>draftN</c> MTP proposals run through ONE
/// <see cref="IForwardPass.BatchVerify"/> pass, which amortizes the trunk's
/// weight reads k× on memory-bound decode. Leading drafts that match the
/// verifier's argmax are emitted; on a rejection the trunk rolls back via the
/// per-token GDN snapshot ring (<see cref="IForwardPass.RestoreBatchSnapshot"/>)
/// and the correction token rides into the NEXT step's batch — the target never
/// runs a separate correction forward. See <see cref="DecodeBatched"/>.</para>
///
/// <para><b>Fallback algorithm — sequential N=1</b> (<see cref="DecodeSequential"/>),
/// used when the pass doesn't support batched verify (e.g. SnapKV-compacted
/// cache, issue #130) or under <c>SHARPI_DISABLE_BATCH_VERIFY=1</c>. Per iter:</para>
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
/// <para>The sequential form emits 2 tokens per 2 main forwards — near-baseline
/// throughput (the pre-#30 state). The batched form is what realizes the issue
/// #25 ≥1.3× criterion.</para>
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
    // #219: on the greedy verify fast path (pMin=1.0 + BatchVerifyArgmax) we carry only the saved
    // next-token's argmax index, not its full logits. _hasSavedArgmax says which is current; every
    // full-logits write to _savedMainLogits clears it, every argmax-path save sets it.
    private int _savedMainArgmax;
    private bool _hasSavedArgmax;

    // Acceptance statistics (per Decode call cumulative)
    private long _totalDraftsEmitted;
    private long _totalDraftsAccepted;

    // Phase timing (issue #207 bench reporting, mirrors SpeculativeDecoder):
    // cumulative wall time in the MTP draft chain + KV refresh forwards, the
    // batched main verify, and rollback/state sync. Uniform slowdown across all
    // three phases under VRAM pressure indicates WDDM paging, not slow kernels.
    private readonly System.Diagnostics.Stopwatch _phaseSw = new();

    /// <summary>Cumulative ms in MTP head forwards (draft chain + KV refresh).</summary>
    public double DraftMs { get; private set; }

    /// <summary>Cumulative ms in the batched main verify passes.</summary>
    public double VerifyMs { get; private set; }

    /// <summary>Cumulative ms in snapshot rollback + saved-state sync.</summary>
    public double CommitMs { get; private set; }

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
        _hasSavedArgmax = false;
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
        DraftMs = 0;
        VerifyMs = 0;
        CommitMs = 0;
    }

    /// <summary>
    /// Resolve the effective MTP draft-chain length (proposed tokens per step) from the
    /// llama.cpp-parity <c>--spec-draft-n-max</c> value. <c>&lt; 1</c> (unset) falls back
    /// to <c>SHARPI_MTP_DRAFT_N</c>, then to the built-in default of 3 (a 4-token verify
    /// batch — the measured 27B optimum once the 4-input CPU FFN kernel landed (issue
    /// #209): the dominant CPU mmap FFN now amortizes its weight read across four draft
    /// tokens, moving the optimum out from the old pairwise k=2 to k=4. Deeper chains
    /// need ring slots too: <see cref="GdnStateCache.ResolveMtpBatchMax"/> defaults to 4
    /// (raise <c>SHARPI_MTP_BATCH_MAX</c> together for k &gt; 4). The result is further
    /// clamped per step against <see cref="IForwardPass.MaxBatchVerifyTokens"/>.
    /// Shared by <see cref="InferenceEngine"/> and the CLI so both resolve identically.
    /// </summary>
    public static int ResolveDraftN(int specDraftNMax)
    {
        if (specDraftNMax >= 1) return specDraftNMax;
        var s = Environment.GetEnvironmentVariable("SHARPI_MTP_DRAFT_N");
        if (s is not null && int.TryParse(s, out var v) && v >= 1) return v;
        return 3;
    }

    /// <summary>
    /// Decode up to <paramref name="maxTokens"/> tokens. Calls <paramref name="emitToken"/>
    /// for every accepted or correction token. Stops when a token in
    /// <paramref name="stopTokenIds"/> is generated (and does NOT emit the stop token).
    /// Dispatches to the folded k-token batched verify path (<see cref="DecodeBatched"/>,
    /// issues #30/#207) when the underlying forward pass implements
    /// <see cref="IForwardPass.BatchVerify"/>; otherwise falls back to the sequential
    /// N=1 algorithm.
    /// </summary>
    /// <param name="pMin">Min draft probability for probabilistic accept under MTP
    /// verification (llama.cpp's <c>--spec-draft-p-min</c>, issue #38). <c>1.0</c>
    /// (default) is byte-perfect argmax-match — accept iff <c>draft == argmax(target)</c>.
    /// <c>p ∈ (0, 1)</c> also accepts when <c>softmax(target)[draft] &gt;= p</c>; the
    /// emitted token then equals the draft, which can differ from baseline greedy.
    /// <c>0.0</c> or negative is treated as <c>1.0</c>.</param>
    /// <param name="draftN">MTP draft-chain length: proposed tokens per step (the verify
    /// batch is <c>1 + draftN</c> wide — the certain token rides in the batch). Clamped
    /// per step to <see cref="IForwardPass.MaxBatchVerifyTokens"/>, the remaining token
    /// budget, and the context window. <c>1</c> reproduces the legacy N=2 behaviour.</param>
    public void Decode(int maxTokens, ReadOnlySpan<int> stopTokenIds, Action<int> emitToken,
                       float pMin = 1f,
                       int draftN = 1,
                       CancellationToken ct = default)
    {
        // Treat 0 and negative as the default (argmax-match) for back-compat with
        // callers that left SpecDraftPMin at the previous default of 0f.
        if (pMin <= 0f) pMin = 1f;
        draftN = Math.Max(1, draftN);
        // Initial dispatch; DecodeBatched additionally re-checks the capability per
        // step (#130: it flips false when the KV cache is SnapKV-compacted) and
        // hands off to the sequential loop if it ever turns off mid-decode.
        bool batched = _fwd.SupportsBatchVerify
            && Environment.GetEnvironmentVariable("SHARPI_DISABLE_BATCH_VERIFY") != "1";
        if (Environment.GetEnvironmentVariable("SHARPI_TRACE_MTP") == "1")
            Console.Error.WriteLine(
                $"[mtp] batched-verify {(batched ? $"ON (draftN={draftN}, maxBatch={_fwd.MaxBatchVerifyTokens})" : "OFF (cache compacted / unsupported config / disabled)")}");
        // Surface a silent capability clamp (the pre-#30 code threw for draftN > 1;
        // a server operator asking for a deeper chain than the snapshot ring allows
        // should see WHY throughput matches a shallower setting).
        if (batched && draftN > _fwd.MaxBatchVerifyTokens - 1)
            Console.Error.WriteLine(
                $"[mtp] requested draft chain {draftN} exceeds the snapshot-ring capacity; " +
                $"clamping to {_fwd.MaxBatchVerifyTokens - 1} draft(s)/step " +
                "(raise SHARPI_MTP_BATCH_MAX to reserve more ring slots).");
        if (batched)
        {
            DecodeBatched(maxTokens, stopTokenIds, emitToken, pMin, draftN, ct);
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

            int t1 = _hasSavedArgmax ? _savedMainArgmax : ArgMax(_savedMainLogits);
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
                _hasSavedArgmax = false;
                _nextPos = P + 1;
                return;
            }
            emitToken(t2); generated++;
            if (generated >= maxTokens)
            {
                _fwd.LastHidden.CopyTo(_savedHidden);
                mainLogits.CopyTo(_savedMainLogits);
                _hasSavedArgmax = false;
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
            _hasSavedArgmax = false;
            _nextPos = P + 2;
        }
    }

    /// <summary>
    /// Folded k-token batched verify (issues #30 / #207 goal 4 — the llama.cpp /
    /// SpeculativeDecoder formulation). Per step, with k = 1 + min(draftN, budget):
    /// <code>
    ///   t1 = argmax(saved_main_logits); emit t1                       // CERTAIN token
    ///   tokens = [t1, d1..d_{k-1}]   d_i chained via MtpForward      // d1 from h@P-1,
    ///                                                                 // d_{i>1} from MtpLastHidden
    ///   batch = BatchVerify(tokens, P)        // ONE pass; batch[i] = logits after tokens[i]
    ///   a = count of leading d_i with Accept(d_i, batch[i-1])
    ///   if a &lt; k-1: RestoreBatchSnapshot(P + 1 + a)               // GDN ring + KV rewind
    ///   MtpTruncateTo(P+1); re-run MtpForward for d_1..d_a with trunk hiddens (HiddenAt)
    ///   emit d_1..d_a
    ///   saved_main_logits = batch[a]          // the correction (or next certain token)
    ///   _savedHidden = HiddenAt(P + a); _nextPos = P + 1 + a
    /// </code>
    /// The rejected-path correction token is argmax(batch[a]) — it rides into the NEXT
    /// step's batch as t1, so the trunk runs exactly one batched pass per step and no
    /// separate correction forward exists (the #207 lesson: a per-step commit forward
    /// erases the batching win). At draftN=1 this emits exactly the legacy N=2
    /// sequence: every emitted token is argmax of main logits when pMin == 1.
    /// </summary>
    private void DecodeBatched(int maxTokens, ReadOnlySpan<int> stopTokenIds, Action<int> emitToken,
                               float pMin, int draftN, CancellationToken ct)
    {
        bool trace = Environment.GetEnvironmentVariable("SHARPI_TRACE_MTP") == "1";
        // Draft-chain hidden scratch: MtpLastHidden points into pass scratch that the
        // next MtpForward overwrites, so the chain copies it out between calls.
        var chainHidden = new float[_savedHidden.Length];
        int generated = 0;
        while (generated < maxTokens)
        {
            ct.ThrowIfCancellationRequested();

            int P = _nextPos;
            // Step sizing: the batch holds t1 + the draft chain, bounded by the
            // remaining token budget (a step never verifies tokens it can't emit),
            // the pass's snapshot-ring capacity, and the context window.
            int kEff = Math.Min(1 + draftN, maxTokens - generated);
            kEff = Math.Min(kEff, _fwd.MaxBatchVerifyTokens);
            kEff = Math.Min(kEff, _fwd.MaxSeqLen - P);

            // Per-step capability re-check (#130: SnapKV compaction turns it off).
            // kEff < 2 also routes here: a 1-token "batch" is just a plain decode
            // step, which the sequential loop already implements.
            if (kEff < 2 || !_fwd.SupportsBatchVerify)
            {
                DecodeSequential(maxTokens - generated, stopTokenIds, emitToken, pMin, ct);
                return;
            }

            // ── Token 1: argmax of last main logits (greedy correctness) ──
            int t1 = _hasSavedArgmax ? _savedMainArgmax : ArgMax(_savedMainLogits);
            if (IsStop(t1, stopTokenIds)) return;
            emitToken(t1); generated++;

            // ── MTP draft chain: kEff−1 proposals ─────────────────────
            // d1 is conditioned on the trunk's h@P-1; deeper drafts self-chain on
            // the MTP block's own output hidden (NEXTN/EAGLE-style — verification
            // corrects any quality loss, so this only affects acceptance rate).
            _phaseSw.Restart();
            var tokens = new int[kEff];
            tokens[0] = t1;
            ReadOnlySpan<float> prevH = _savedHidden;
            for (int i = 1; i < kEff; i++)
            {
                ReadOnlySpan<float> mtpLogits = _fwd.MtpForward(tokens[i - 1], P + i - 1, prevH);
                tokens[i] = ArgMax(mtpLogits);
                if (i < kEff - 1)
                {
                    var selfH = _fwd.MtpLastHidden;
                    if (selfH.Length != chainHidden.Length)
                        throw new InvalidOperationException(
                            $"MtpLastHidden length {selfH.Length} != EmbeddingDim {chainHidden.Length}; " +
                            "the forward pass must capture the MTP block output for chained drafting.");
                    selfH.CopyTo(chainHidden);
                    prevH = chainHidden;
                }
            }
            _totalDraftsEmitted += kEff - 1;
            DraftMs += _phaseSw.Elapsed.TotalMilliseconds;

            // ── ONE batched main verify over tokens[0..kEff) ──────────
            // #219: a pure-greedy verify (pMin >= 1.0) on a GPU pass fetches only the per-position
            // argmaxes (k*8 bytes) instead of k full-vocab logits vectors. pMin < 1.0 still needs
            // the distributions (AcceptDraft's softmax branch), so it keeps the full download.
            bool argmaxFast = pMin >= 1.0f && _fwd.SupportsBatchVerifyArgmax;
            _phaseSw.Restart();
            float[][]? batch = null;
            (int Index, float Value)[]? verifyArgmax = null;
            if (argmaxFast) verifyArgmax = _fwd.BatchVerifyArgmax(tokens, P);
            else            batch = _fwd.BatchVerify(tokens, P);
            VerifyMs += _phaseSw.Elapsed.TotalMilliseconds;

            // Verifier's greedy correction at draft position `pos` (argmax of the logits there).
            int Target(int pos) => argmaxFast ? verifyArgmax![pos].Index : ArgMax(batch![pos]);

            // ── Greedy accept: count leading agreeing drafts ──────────
            // An accepted STOP draft also ends the chain here, EXCLUDED from `a`:
            // like the sequential path, the stop token is neither emitted nor
            // committed — clamping acceptance before the rollback below keeps
            // every piece of state (trunk KV, GDN ring restore, MTP KV, hidden
            // history, _nextPos) consistent at newPos, where the pre-#208-review
            // emit-loop stop check returned with the caches stranded past _nextPos.
            int a = 0;
            bool stopHit = false;
            for (int i = 1; i < kEff; i++)
            {
                int target = Target(i - 1);
                bool accept = argmaxFast
                    ? tokens[i] == target                              // pMin>=1.0: argmax-match only
                    : AcceptDraft(tokens[i], target, batch![i - 1], pMin, out _);
                if (!accept) break;
                if (IsStop(tokens[i], stopTokenIds)) { stopHit = true; break; }
                a++;
            }
            _totalDraftsAccepted += a;
            int newPos = P + 1 + a;

            if (trace)
                Console.Error.WriteLine(
                    $"[mtp-batch] P={P} k={kEff} t1={t1} " +
                    $"drafts=[{string.Join(",", tokens[1..])}] accepted={a}/{kEff - 1}" +
                    (a < kEff - 1 ? $" correction={Target(a)}" : ""));

            _phaseSw.Restart();
            // ── Roll back the rejected tail (no-op when fully accepted) ──
            if (newPos < P + kEff)
                _fwd.RestoreBatchSnapshot(newPos);
            CommitMs += _phaseSw.Elapsed.TotalMilliseconds;

            // ── MTP KV refresh ────────────────────────────────────────
            // The chain wrote MTP K/V at P (trunk-quality) and P+1..P+kEff-2
            // (self-hidden quality). Rewind to P+1 and rewrite the ACCEPTED
            // positions with the trunk hiddens the verify produced, so future
            // drafts attend over trunk-quality K/V — the same per-position
            // contract the legacy N=2 commit kept. The rejected-path correction's
            // MTP entry is written by the NEXT step's first chain call.
            _phaseSw.Restart();
            _fwd.MtpTruncateTo(P + 1);
            for (int i = 1; i <= a; i++)
                _ = _fwd.MtpForward(tokens[i], P + i, HiddenAtChecked(P + i - 1));
            DraftMs += _phaseSw.Elapsed.TotalMilliseconds;

            // ── Emit accepted drafts (stop drafts never reach here — the accept
            //    loop clamps `a` before the first accepted stop) ────────
            for (int i = 1; i <= a; i++)
            {
                emitToken(tokens[i]); generated++;
            }

            // ── Saved state for the next step ─────────────────────────
            // batch[a] predicts position newPos: it is the next certain token —
            // the correction on a reject, the chain continuation on full accept,
            // or the (un-emitted, un-committed) stop on stopHit. All caches sit
            // exactly at newPos, so a follow-up call resumes consistently.
            if (argmaxFast) { _savedMainArgmax = verifyArgmax![a].Index; _hasSavedArgmax = true; }
            else            { batch![a].CopyTo(_savedMainLogits, 0); _hasSavedArgmax = false; }
            HiddenAtChecked(newPos - 1).CopyTo(_savedHidden);
            _nextPos = newPos;
            if (stopHit) return;
        }
    }

    /// <summary>
    /// <see cref="IForwardPass.HiddenAt"/> with an invariant check: the batched verify
    /// must have populated the hidden-history slot for every batch position.
    /// </summary>
    private ReadOnlySpan<float> HiddenAtChecked(int position)
    {
        var h = _fwd.HiddenAt(position);
        if (h.Length != _savedHidden.Length)
            throw new InvalidOperationException(
                $"HiddenAt({position}) returned {h.Length} floats, expected {_savedHidden.Length}. " +
                "BatchVerify must write the per-position trunk hiddens into the MTP hidden " +
                "history (the PrefillMtp contract, issue #33).");
        return h;
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
