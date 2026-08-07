using System.Text;
using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Regression coverage for <see cref="InferenceEngine"/>'s prefix-cache reuse path
/// (issues #20 and #21). Three scenarios:
///
/// <list type="number">
///   <item>A forward pass that does NOT support partial rewind AND has no snapshot
///         facility (e.g. unknown future backend) must not be asked to
///         <see cref="IForwardPass.TruncateTo"/> to an intermediate length; the
///         engine must fall back to a full reset (issue #20).</item>
///   <item>A forward pass that DOES support partial rewind keeps the existing
///         page-aligned prefix-cache fast path — guards against accidentally
///         disabling it for every backend.</item>
///   <item>A forward pass that does NOT support partial rewind but DOES expose
///         <see cref="IForwardPass.CaptureSnapshot"/> /
///         <see cref="IForwardPass.SnapshotLength"/> (GDN hybrid; issue #21)
///         can reuse state across chat turns when the new prompt extends the
///         previous full sequence — the engine calls <c>TruncateTo(snapLen)</c>
///         instead of resetting.</item>
/// </list>
/// </summary>
public sealed class InferenceEnginePrefixCacheTests
{
    private const int Eos = 99;

    /// <summary>
    /// Drives two sequential <c>GenerateAsync</c> calls with prompts that share a
    /// 32-token page-aligned prefix (2 × <see cref="PagedKvCache.PageSize"/>). On a
    /// rewind-incompatible pass the engine must not propagate the
    /// <see cref="NotSupportedException"/> from <see cref="IForwardPass.TruncateTo"/>.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_OnRewindIncompatiblePass_DoesNotPropagateTruncateThrow()
    {
        var tokenizer = new MultiTurnTokenizer();
        var fwd = new RewindIncompatibleForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // Turn 1: 32 prompt tokens. After this _prevTokens has 32 entries.
        await Drain(engine.GenerateAsync("turn1", sp));

        // Turn 2: 48 prompt tokens, first 32 shared with turn 1. FindCacheablePrefix
        // returns 32; without the fix the engine calls TruncateTo(32) which throws.
        await Drain(engine.GenerateAsync("turn2", sp));

        Assert.False(
            fwd.PartialTruncateAttempted,
            "InferenceEngine called TruncateTo with a partial length on a rewind-incompatible pass.");
        // ResetCache is the cold-start branch the engine takes whenever prefixLen == 0. On a
        // rewind-incompatible pass the gate forces prefixLen to 0 even when prefixes match,
        // so observing ResetCache here confirms the gate fired (no partial TruncateTo was tried).
        Assert.True(fwd.ResetCacheCalled, "Engine should reach the ResetCache branch when the prefix gate forces prefixLen to 0.");
    }

    /// <summary>
    /// Same two-turn pattern on a rewind-capable pass — the engine should call
    /// <see cref="IForwardPass.TruncateTo"/> with the matched prefix length and
    /// only re-prefill the new suffix.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_OnRewindCapablePass_ReusesPrefixOnSecondCall()
    {
        var tokenizer = new MultiTurnTokenizer();
        var fwd = new RewindCapableForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await Drain(engine.GenerateAsync("turn1", sp));
        await Drain(engine.GenerateAsync("turn2", sp));

        Assert.Equal(32, fwd.LastTruncateLength);
        // Second prefill should cover only the 16-token suffix, not all 48 tokens.
        Assert.Equal(16, fwd.LastPrefillLength);
        Assert.Equal(32, fwd.LastPrefillStartPos);
    }

    /// <summary>
    /// Issue #22: rewind-incompatible passes must surface <see cref="IInferenceEngine.PrefixCacheEnabled"/>
    /// as <c>false</c>, and <see cref="IInferenceEngine.PrefillTokensReused"/> must remain at zero across
    /// turns because the engine is forced down the full-reset branch.
    /// </summary>
    [Fact]
    public async Task PrefixCacheState_OnRewindIncompatiblePass_ReportsDisabledAndZeroReused()
    {
        var tokenizer = new MultiTurnTokenizer();
        var fwd = new RewindIncompatibleForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        Assert.False(engine.PrefixCacheEnabled);
        Assert.Equal(0, engine.PrefillTokensReused);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        await Drain(engine.GenerateAsync("turn1", sp));
        await Drain(engine.GenerateAsync("turn2", sp));

        Assert.Equal(0, engine.PrefillTokensReused);
    }

    /// <summary>
    /// Issue #22: rewind-capable passes report <see cref="IInferenceEngine.PrefixCacheEnabled"/> true
    /// and accumulate the matched-prefix length into <see cref="IInferenceEngine.PrefillTokensReused"/>
    /// after a cache hit. The fixture's two-turn prompt shares a 32-token prefix.
    /// </summary>
    [Fact]
    public async Task PrefixCacheState_OnRewindCapablePass_ReportsEnabledAndAccumulatesReused()
    {
        var tokenizer = new MultiTurnTokenizer();
        var fwd = new RewindCapableForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        Assert.True(engine.PrefixCacheEnabled);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        await Drain(engine.GenerateAsync("turn1", sp));
        // First turn is a cold start — no prefix to reuse.
        Assert.Equal(0, engine.PrefillTokensReused);

        await Drain(engine.GenerateAsync("turn2", sp));
        // Second turn shares a 32-token prefix with turn 1.
        Assert.Equal(32, engine.PrefillTokensReused);
    }

    /// <summary>
    /// Issue #21: a non-rewindable pass with a snapshot facility (GDN hybrid)
    /// should still reuse cached state across chat-continuation turns by way
    /// of <see cref="IForwardPass.CaptureSnapshot"/> at end-of-decode and
    /// <see cref="IForwardPass.TruncateTo"/> on the snapshot length next turn.
    ///
    /// Turn 1: 32 prompt tokens; after the loop (1 generated token + EOS),
    /// the stub captures a snapshot at the full-sequence length.
    /// Turn 2: 33 prompt tokens whose first 33 tokens are exactly the turn-1
    /// full sequence (32 prompt + 1 generated). The engine must call
    /// TruncateTo(snapLen) — not propagate any throw, not call ResetCache.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_OnSnapshotCapablePass_RestoresSnapshotOnContinuation()
    {
        var tokenizer = new SnapshotMultiTurnTokenizer();
        var fwd = new SnapshotCapableForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await Drain(engine.GenerateAsync("turn1", sp));

        // After turn 1, _prevTokens is the 32-token prompt + the 1 generated token.
        // The stub generated token (id 50) is not in stopIds, so it's appended; the
        // loop then iterates once more and hits MaxNewTokens, exiting cleanly.
        // CaptureSnapshot records the GdnStateCache.Length-equivalent (mock).
        Assert.Equal(33, fwd.LastCapturedSnapshotLength);

        await Drain(engine.GenerateAsync("turn2", sp));

        // Engine should have asked the snapshot-capable pass to restore at length 33.
        Assert.Equal(33, fwd.LastTruncateLength);
        Assert.False(
            fwd.ResetCalledAfter,
            "ResetCache must not run on turn 2 — the snapshot branch already prepared the cache.");
        // Suffix prefill covers only the tokens after the snapshot length.
        Assert.True(fwd.LastPrefillLength > 0);
        Assert.Equal(33, fwd.LastPrefillStartPos);
    }

    // ── Issue #102: canonical-history prefix snapshot path ─────────────────

    /// <summary>
    /// Snapshot-only backend with a canonical-history hint: the engine must split
    /// prefill into two stages (canonical | generation prep) and capture the snapshot
    /// at the canonical boundary — NOT at end-of-decode. That captured length is
    /// what the next turn's prefix match keys off.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_CanonicalPrefix_SnapshotCapturedAtCanonicalBoundary()
    {
        var tokenizer = new CanonicalChatTokenizer();
        var fwd = new SnapshotCapableForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // generating prompt = canonical (24 tokens) + assistant prep (8 tokens) = 32
        await Drain(GenerateAsync(engine, "turn1_full", canonical: "turn1_canon", sp));

        // Exactly two Prefill calls: stage 1 covers the canonical portion (24 tokens
        // at startPos=0), stage 2 covers the generation prep (8 tokens at startPos=24).
        Assert.Equal(2, fwd.PrefillCalls.Count);
        Assert.Equal((24, 0), fwd.PrefillCalls[0]);
        Assert.Equal((8, 24), fwd.PrefillCalls[1]);

        // One snapshot captured, at the canonical boundary (not at 32 + 1 = 33).
        Assert.Single(fwd.CaptureSnapshotCalls);
        Assert.Equal(24, fwd.CaptureSnapshotCalls[0]);
        Assert.Equal(24, fwd.SnapshotLength);
    }

    /// <summary>
    /// Two-turn flow: turn 2's generating prompt starts with turn 1's canonical history,
    /// then appends turn 1's scrubbed assistant response + a new user message + assistant
    /// prep. The engine should restore from turn 1's snapshot (at canonical boundary)
    /// and prefill only the tail.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_CanonicalPrefix_TurnTwoRestoresFromCanonicalSnapshot()
    {
        var tokenizer = new CanonicalChatTokenizer();
        var fwd = new SnapshotCapableForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // Turn 1: snapshot is captured at 24 (canonical boundary).
        await Drain(GenerateAsync(engine, "turn1_full", canonical: "turn1_canon", sp));
        fwd.PrefillCalls.Clear();
        fwd.CaptureSnapshotCalls.Clear();
        long reusedAfterTurn1 = engine.PrefillTokensReused;

        // Turn 2: generating prompt's first 24 tokens equal turn 1's canonical → restore.
        // Turn 2's own canonical is 40 tokens, full prompt is 48. Expected:
        //   TruncateTo(24)        — restore from turn 1's snapshot
        //   Prefill(16 @ 24)      — stage 1: prefill canonical tail (positions 24..40)
        //   CaptureSnapshot at 40 — turn 2's canonical boundary
        //   Prefill(8 @ 40)       — stage 2: generation prep
        await Drain(GenerateAsync(engine, "turn2_full", canonical: "turn2_canon", sp));

        Assert.Equal(24, fwd.LastTruncateLength);
        Assert.False(fwd.ResetCalledAfter,
            "Snapshot restore must skip ResetCache when the canonical prefix matches.");
        Assert.Equal(2, fwd.PrefillCalls.Count);
        Assert.Equal((16, 24), fwd.PrefillCalls[0]);
        Assert.Equal((8, 40),  fwd.PrefillCalls[1]);
        Assert.Single(fwd.CaptureSnapshotCalls);
        Assert.Equal(40, fwd.CaptureSnapshotCalls[0]);
        // 24 prompt tokens skipped on turn 2 — observable via PrefillTokensReused.
        Assert.Equal(reusedAfterTurn1 + 24, engine.PrefillTokensReused);
    }

    /// <summary>
    /// Rewindable backend with a canonical hint: the canonical-snapshot split is
    /// gated on <see cref="IForwardPass.SupportsSnapshot"/>, so a rewindable backend
    /// behaves exactly as it would without the hint — single-shot prefill and
    /// <c>_prevTokens = fullSeq</c> baseline (legacy path). The hint becomes
    /// observable on snapshot-only backends like the GDN hybrids. This test pins
    /// that contract: rewindable backends keep the legacy behavior intact, so the
    /// canonical-snapshot work doesn't regress page-aligned reuse on the rewindable
    /// fast path that already shipped (issue #102 design note).
    /// </summary>
    [Fact]
    public async Task GenerateAsync_CanonicalPrefix_RewindableKeepsLegacyFullSeqBaseline()
    {
        var tokenizer = new CanonicalChatTokenizer();
        var fwd = new RewindCapableForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // Turn 1 cold start: full prompt prefilled in one call. No split.
        await Drain(GenerateAsync(engine, "turn1_full", canonical: "turn1_canon", sp));
        Assert.Equal(32, fwd.LastPrefillLength);
        Assert.Equal(0,  fwd.LastPrefillStartPos);
        Assert.Equal(0, engine.PrefillTokensReused);

        // Turn 2's tokens diverge from turn 1's fullSeq at position 24 (turn 2's
        // canonical extends with [100..116); turn 1's fullSeq has the assistant prep
        // [200..208) at the same offset). Shared prefix = 24, page-aligned down to 16.
        // The point: rewindable backend reuses what the legacy fullSeq baseline allows —
        // no canonical-aware split, no canonical-only _prevTokens, just FindCacheablePrefix.
        await Drain(GenerateAsync(engine, "turn2_full", canonical: "turn2_canon", sp));
        Assert.Equal(PagedKvCache.PageSize, fwd.LastTruncateLength);
        Assert.Equal(PagedKvCache.PageSize, engine.PrefillTokensReused);
    }

    // ── Issue #106: MTP runs also use canonical / snapshot reuse ───────────

    /// <summary>
    /// Issue #106 turn 1 of the canonical-snapshot path on an MTP-capable pass.
    /// With the <c>!useMtp</c> gate removed, MTP runs must also split prefill at the
    /// canonical boundary, capture the snapshot there (not at end-of-decode), and
    /// drive <c>PrefillMtp</c> over the full prompt at <c>startPos = 0</c>.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_Mtp_CanonicalPrefix_SnapshotCapturedAtCanonicalBoundary()
    {
        var tokenizer = new CanonicalChatTokenizer();
        var fwd = new SnapshotMtpForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await Drain(GenerateAsync(engine, "turn1_full", canonical: "turn1_canon", sp));

        Assert.Equal(2, fwd.PrefillCalls.Count);
        Assert.Equal((24, 0), fwd.PrefillCalls[0]);
        Assert.Equal((8, 24), fwd.PrefillCalls[1]);

        // Exactly one snapshot, at the canonical boundary — NOT at 32 (end-of-prefill)
        // and NOT at 33 (end-of-decode). The MTP path's end-of-decode capture is now
        // gated by !useCanonicalSnapshot so it doesn't clobber the canonical snapshot.
        Assert.Single(fwd.CaptureSnapshotCalls);
        Assert.Equal(24, fwd.CaptureSnapshotCalls[0]);
        Assert.Equal(24, fwd.SnapshotLength);

        // PrefillMtp covers the whole prompt at startPos = prefixLen = 0.
        Assert.Single(fwd.PrefillMtpCalls);
        Assert.Equal((32, 0), fwd.PrefillMtpCalls[0]);
    }

    /// <summary>
    /// Issue #106 turn 2: snapshot restore on an MTP run must also rewind the MTP KV
    /// cache (mock's <c>MtpTruncateTo</c> fires from inside <c>TruncateTo</c>) and the
    /// follow-up <c>PrefillMtp</c> must be called with <c>startPos = snapLen</c> (not 0).
    /// </summary>
    [Fact]
    public async Task GenerateAsync_Mtp_CanonicalPrefix_TurnTwoRestoresAndCallsPrefillMtpAtSnapLen()
    {
        var tokenizer = new CanonicalChatTokenizer();
        var fwd = new SnapshotMtpForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await Drain(GenerateAsync(engine, "turn1_full", canonical: "turn1_canon", sp));
        fwd.PrefillCalls.Clear();
        fwd.CaptureSnapshotCalls.Clear();
        fwd.PrefillMtpCalls.Clear();
        fwd.TruncateCalls.Clear();
        fwd.MtpTruncateCalls.Clear();
        long reusedAfterTurn1 = engine.PrefillTokensReused;

        await Drain(GenerateAsync(engine, "turn2_full", canonical: "turn2_canon", sp));

        Assert.Contains(24, fwd.TruncateCalls);
        Assert.Contains(24, fwd.MtpTruncateCalls);
        Assert.False(fwd.ResetCalledAfter,
            "Snapshot restore must skip ResetCache when the canonical prefix matches.");

        // Two-stage prefill at the new canonical boundary (40).
        Assert.Equal(2, fwd.PrefillCalls.Count);
        Assert.Equal((16, 24), fwd.PrefillCalls[0]);
        Assert.Equal((8, 40),  fwd.PrefillCalls[1]);
        Assert.Single(fwd.CaptureSnapshotCalls);
        Assert.Equal(40, fwd.CaptureSnapshotCalls[0]);

        // PrefillMtp covers the [24..48) tail at startPos = snapLen.
        Assert.Single(fwd.PrefillMtpCalls);
        Assert.Equal((24, 24), fwd.PrefillMtpCalls[0]);

        Assert.Equal(reusedAfterTurn1 + 24, engine.PrefillTokensReused);
    }

    /// <summary>
    /// Issue #106 legacy-snapshot path on an MTP run: with no canonical hint, the
    /// engine still captures a snapshot at end-of-decode and on the next turn
    /// restores via <c>TruncateTo(snapLen)</c> + <c>PrefillMtp(suffix, startPos = snapLen)</c>.
    /// Pre-#106, the engine's <c>!useMtp</c> gate skipped the snapshot-match branch for
    /// MTP runs and re-prefilled the whole prompt every turn — wasted ~95 s per
    /// round-trip on long Carnice agentic loops per the issue.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_Mtp_LegacySnapshot_TurnTwoRestoresAndCallsPrefillMtpAtSnapLen()
    {
        var tokenizer = new MtpLegacyMultiTurnTokenizer();
        var fwd = new SnapshotMtpForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // Turn 1: 32-token prompt, immediate-EOS decode → end-of-decode snapshot at 32.
        await Drain(engine.GenerateAsync("turn1", sp));
        Assert.Equal(32, fwd.SnapshotLength);
        Assert.Single(fwd.PrefillMtpCalls);
        Assert.Equal((32, 0), fwd.PrefillMtpCalls[0]);

        fwd.PrefillCalls.Clear();
        fwd.PrefillMtpCalls.Clear();
        fwd.TruncateCalls.Clear();
        fwd.MtpTruncateCalls.Clear();

        // Turn 2: 34-token prompt whose first 32 match turn 1. Snapshot-match fires.
        await Drain(engine.GenerateAsync("turn2", sp));

        Assert.Contains(32, fwd.TruncateCalls);
        Assert.False(fwd.ResetCalledAfter,
            "Snapshot restore must skip ResetCache when the legacy snapshot matches.");
        Assert.Single(fwd.PrefillCalls);
        Assert.Equal((2, 32), fwd.PrefillCalls[0]);
        Assert.Single(fwd.PrefillMtpCalls);
        Assert.Equal((2, 32), fwd.PrefillMtpCalls[0]);
    }

    /// <summary>
    /// A canonical hint that doesn't tokenize to a strict token-prefix of the
    /// generating prompt must be rejected silently and the legacy single-prefill +
    /// end-of-decode snapshot path used.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_CanonicalPrefix_MismatchedHintFallsBackToLegacy()
    {
        var tokenizer = new CanonicalChatTokenizer();
        var fwd = new SnapshotCapableForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // canonical = "mismatch" → tokenizes to [9,9,9,9] which is NOT a prefix of
        // the generating prompt's tokens. Engine should fall through to legacy path.
        await Drain(GenerateAsync(engine, "turn1_full", canonical: "mismatch", sp));

        // Single prefill (no split), single snapshot capture at end-of-decode (33 = 32 prompt + 1 sampled).
        Assert.Single(fwd.PrefillCalls);
        Assert.Equal((32, 0), fwd.PrefillCalls[0]);
        Assert.Single(fwd.CaptureSnapshotCalls);
        Assert.Equal(33, fwd.CaptureSnapshotCalls[0]);
    }

    /// <summary>
    /// Issue #212: with the 2-slot prefix cache enabled (SHARPI_PREFIX_SLOTS=2), a short
    /// interleaved auxiliary request is served from the bounded scratch slot (slot 1) and does
    /// NOT evict the long resident prefix in the owned slot (slot 0) — so the next long request
    /// still reuses it. This is the exact agentic-client (Claude Code) pattern #212 targets.
    /// Slot activations: long → owned, aux → scratch, long → owned; the second long reuses 48
    /// (3-page-aligned) tokens of the first long's prefix.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_TwoSlot_ShortAuxRequestDoesNotEvictLongPrefix()
    {
        using var _ = new EnvScope(("SHARPI_PREFIX_SLOTS", "2"), ("SHARPI_PREFIX_SCRATCH_TOKENS", "48"));

        var tokenizer = new InterleavedAgenticTokenizer();
        var fwd = new MultiSlotForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        Assert.True(engine.PrefixCacheEnabled);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await Drain(engine.GenerateAsync("long", sp));   // cold → owned slot 0
        await Drain(engine.GenerateAsync("aux", sp));    // short, no match → scratch slot 1
        await Drain(engine.GenerateAsync("long", sp));   // matches slot 0 → reuse

        // The three requests bound owned, scratch, owned in order.
        Assert.Equal([0, 1, 0], fwd.ActivatedIds);

        // The aux turn never touched the owned slot: it was reset/prefilled exactly once (the
        // first long turn), then truncated+prefilled for the third (reuse) turn — never by aux.
        Assert.Equal(1, fwd.Owned.Resets);
        Assert.Equal([(64, 0), (16, 48)], fwd.Owned.Prefills);
        Assert.Contains(48, fwd.Owned.Truncates);

        // The aux request lived entirely in the scratch slot.
        Assert.Single(fwd.Scratch);
        Assert.Equal([(16, 0)], fwd.Scratch[0].Prefills);

        // Only the third turn reused a prefix (48 page-aligned tokens of the long prefix).
        Assert.Equal(48, engine.PrefillTokensReused);
    }

    /// <summary>
    /// Issue #212 contrast (the bug being fixed): with the default single-slot cache, the short
    /// interleaved aux request overwrites the one resident sequence, so the next long request
    /// finds no reusable prefix and re-prefills from scratch — 0 reuse.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_SingleSlot_ShortAuxRequestEvictsLongPrefix()
    {
        var tokenizer = new InterleavedAgenticTokenizer();
        var fwd = new MultiSlotForwardPass();
        // No SHARPI_PREFIX_SLOTS env → single-slot (default) behavior, even though fwd CAN do 2.
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        await Drain(engine.GenerateAsync("long", sp));
        await Drain(engine.GenerateAsync("aux", sp));
        await Drain(engine.GenerateAsync("long", sp));

        // Single-slot: the engine never binds a non-owned slot...
        Assert.Empty(fwd.ActivatedIds);
        // ...and the aux turn evicted the long prefix, so the second long turn reused nothing.
        Assert.Equal(0, engine.PrefillTokensReused);
    }

    /// <summary>
    /// Issue #212 (review follow-up): a scratch request that resets+overwrites its slot's KV but
    /// fails mid-decode must NOT leave the slot's token shadow describing the destroyed KV — else
    /// the next request could SelectSlot-match that stale prefix and TruncateTo into mismatched KV
    /// (silent garbage). The engine nulls the active slot's shadow before mutating its KV and only
    /// re-writes it on a complete decode, so the post-failure request reuses nothing.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_TwoSlot_ScratchFailedMidDecode_DoesNotReuseDestroyedPrefix()
    {
        using var _ = new EnvScope(("SHARPI_PREFIX_SLOTS", "2"), ("SHARPI_PREFIX_SCRATCH_TOKENS", "48"));

        var tokenizer = new InterleavedAgenticTokenizer();
        var fwd = new MultiSlotForwardPass { EmitNonStopFirst = true }; // force a decode Forward
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };

        // B1: completes on the scratch slot; its transcript becomes scratch's shadow.
        await Drain(engine.GenerateAsync("aux32", sp));

        // B2: a disjoint short request resets the scratch KV, then throws mid-decode.
        fwd.FailNextDecode = true;
        await Assert.ThrowsAnyAsync<Exception>(async () => await Drain(engine.GenerateAsync("aux2", sp)));

        long reusedBefore = engine.PrefillTokensReused;

        // B3: same prompt as B1. If B2's failure had left the stale B1 shadow in place, B3 would
        // reuse a 16-token page of it — into KV that B2 reset+overwrote. The shadow-null fix makes
        // B3 reuse nothing.
        await Drain(engine.GenerateAsync("aux32", sp));

        Assert.Equal(reusedBefore, engine.PrefillTokensReused);
    }

    /// <summary>
    /// Issue #452: the MTP path stays single-slot — it skips SelectSlot and runs on the owned
    /// slot 0 — so <c>activeSlotIdx</c> is never set. Gating the pre-prefill shadow invalidation
    /// on a bound slot therefore skipped MTP entirely: slot 0's shadow kept describing the PRIOR
    /// sequence while the request reset and overwrote slot 0's KV. Downgrading <c>_prevTokens</c>
    /// was not enough, because the next MTP request re-seeds <c>_prevTokens</c> from that stale
    /// <c>_slotTokens[0]</c>, resurrecting a prefix whose KV is gone. The engine now invalidates
    /// the slot it MUTATES rather than the one it bound, so the post-abort request reuses nothing.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_TwoSlot_MtpFailedMidDecode_DoesNotReuseDestroyedPrefix()
    {
        using var _ = new EnvScope(("SHARPI_PREFIX_SLOTS", "2"), ("SHARPI_PREFIX_SCRATCH_TOKENS", "48"));

        var tokenizer = new InterleavedAgenticTokenizer();
        var fwd = new MultiSlotForwardPass { MtpHead = true, EmitNonStopFirst = true };
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        Assert.True(engine.PrefixCacheEnabled);
        // MaxNewTokens=2: the first emitted token comes from the prefill logits, so the decoder
        // reaches MtpForward + Forward — where the mid-decode failure lands.
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 2, SpecType = SpecType.Mtp };

        // A: completes on the owned slot; its transcript becomes slot 0's shadow.
        await Drain(engine.GenerateAsync("aux32", sp));
        // MTP never binds a non-owned slot, so nothing was activated.
        Assert.Empty(fwd.ActivatedIds);

        // B: a disjoint prompt resets slot 0's KV, then throws mid-decode.
        fwd.FailNextDecode = true;
        await Assert.ThrowsAnyAsync<Exception>(async () => await Drain(engine.GenerateAsync("aux2", sp)));

        long reusedBefore = engine.PrefillTokensReused;

        // C: same prompt as A. With the stale slot-0 shadow in place, C would page-match A's
        // transcript and TruncateTo into KV that B reset and overwrote.
        await Drain(engine.GenerateAsync("aux32", sp));

        Assert.Equal(reusedBefore, engine.PrefillTokensReused);
    }

    /// <summary>
    /// Issue #451 companion: the engine's DSpark end-of-decode shadow write used to hard-code
    /// slot 0 even though DSpark, unlike MTP, takes the ordinary SelectSlot bind and can run
    /// entirely in the bounded scratch slot. That combination is unreachable in practice only
    /// because <see cref="InferenceEngine.AttachDSparkDraft"/> rejects it outright — the tap
    /// buffer is position-indexed against a single KV region. Pin that guard: it is the sole
    /// reason the shadow write can't be reached with the wrong slot, so silently dropping it
    /// would re-open #451 rather than merely relaxing a restriction.
    /// </summary>
    [Fact]
    public void AttachDSparkDraft_WithMultiSlotPrefixCache_IsRejected()
    {
        using var _ = new EnvScope(("SHARPI_PREFIX_SLOTS", "2"), ("SHARPI_PREFIX_SCRATCH_TOKENS", "48"));

        var fwd = new MultiSlotForwardPass { Taps = true };
        using var engine = new InferenceEngine(fwd, new InterleavedAgenticTokenizer(), "mock",
            thinkTokenId: -1, endThinkTokenId: -1);

        var ex = Assert.Throws<InvalidOperationException>(
            () => engine.AttachDSparkDraft(new StubDSparkDraft()));
        Assert.Contains("multi-slot", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Issue #450: once prefill completes, KV <c>[0, tokens.Length)</c> is valid — decode only
    /// appends beyond it. A request aborted mid-decode must therefore hand the next one the WHOLE
    /// prompt, not just the prefix it happened to inherit. Turn 2 reuses turn 1's 32-token prefix
    /// and prefills a 16-token suffix before being cancelled; turn 3 extends turn 2's prompt, so
    /// it can reuse all 48 of turn 2's prefilled tokens. Before this change the cancelled request
    /// retained only its inherited 32, charging turn 3 a re-prefill of 16 tokens whose KV was
    /// intact all along — the common agentic pattern of disconnect-then-continue.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_CancelledMidDecode_KeepsFullPromptForNextRequest()
    {
        var tokenizer = new MultiTurnTokenizer();
        var fwd = new RewindCapableForwardPass { DecodeTokenId = 50 };
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        // Turn 1 (cold): 32-token prompt becomes the cached sequence.
        await Drain(engine.GenerateAsync("turn1", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 }));

        // Turn 2: 48 tokens = turn 1's 32 + 16 fresh. Reuses 32, prefills the 16-token suffix,
        // then cancels from inside the first decode Forward — never reaching end-of-decode.
        using var cts = new CancellationTokenSource();
        fwd.CancelOnDecode = cts;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => Drain(engine.GenerateAsync("turn2", new SamplingParams { Temperature = 0f, MaxNewTokens = 64 }, cts.Token)));
        fwd.CancelOnDecode = null;

        long reusedAfterCancel = engine.PrefillTokensReused;

        // Turn 3: 64 tokens = turn 2's 48 + 16 more. All 48 of turn 2's prompt are still backed
        // by valid KV, so the suffix prefill must start at 48 — not 32.
        await Drain(engine.GenerateAsync("turn3", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 }));

        Assert.Equal(48, fwd.LastTruncateLength);
        Assert.Equal(48, fwd.LastPrefillStartPos);
        Assert.Equal(16, fwd.LastPrefillLength);
        Assert.Equal(reusedAfterCancel + 48, engine.PrefillTokensReused);
    }

    /// <summary>
    /// A request cancelled mid-decode must leave the prefix it reused still reusable. The
    /// engine invalidates the token shadow before mutating KV, but only positions
    /// &gt;= prefixLen are ever overwritten — so the shadow is downgraded to
    /// <c>tokens[..prefixLen]</c>, not cleared. Clearing it (the previous behavior) charged
    /// the next request a full re-prefill of a prompt whose KV was still intact, which is
    /// the common agentic-client pattern: disconnect mid-generation, immediately retry.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_CancelledMidDecode_KeepsReusedPrefixForNextRequest()
    {
        var tokenizer = new MultiTurnTokenizer();
        // Non-stop decode token so the decode loop keeps running until cancellation.
        var fwd = new RewindCapableForwardPass { DecodeTokenId = 50 };
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        // Turn 1 (cold): 32-token prompt + 1 decoded token becomes the cached sequence.
        await Drain(engine.GenerateAsync("turn1", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 }));

        // Turn 2: 48-token prompt sharing turn 1's 32-token prefix. It reuses that prefix,
        // prefills the 16-token suffix, then is cancelled from inside the first decode
        // Forward — so it never reaches the end-of-decode shadow write.
        using var cts = new CancellationTokenSource();
        fwd.CancelOnDecode = cts;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => Drain(engine.GenerateAsync("turn2", new SamplingParams { Temperature = 0f, MaxNewTokens = 64 }, cts.Token)));
        fwd.CancelOnDecode = null;

        long reusedAfterCancel = engine.PrefillTokensReused;

        // Turn 3: the retry. KV positions [0, 32) were never touched by turn 2, so the
        // same 32-token prefix must still be reused — suffix-only prefill, not 48 @ 0.
        await Drain(engine.GenerateAsync("turn2", new SamplingParams { Temperature = 0f, MaxNewTokens = 1 }));

        Assert.Equal(32, fwd.LastTruncateLength);
        Assert.Equal(16, fwd.LastPrefillLength);
        Assert.Equal(32, fwd.LastPrefillStartPos);
        Assert.Equal(reusedAfterCancel + 32, engine.PrefillTokensReused);
    }

    /// <summary>
    /// A TurboQuant pass is rewindable only down to its FP32 recent window; older KV is
    /// compressed in place and <c>TruncateTo</c> throws for it. The engine must clamp the
    /// prefix candidate against <see cref="IForwardPass.MinRewindLength"/>: below the floor
    /// (a long chat whose system prompt carries injected memory or a timestamp, so the
    /// prompt diverges early) it falls back to a full reset instead of letting the throw
    /// reach the user; at or above it, reuse stays on — so the fix can't degenerate into
    /// "TurboQuant disables the prefix cache". turn1/turn2 share a 32-token prefix.
    /// </summary>
    [Theory]
    // floor, truncate (-1 = never called), prefill len, prefill startPos, tokens reused
    [InlineData(64, -1, 48, 0,  0)]   // below floor  → full re-prefill of turn 2
    [InlineData(16, 32, 16, 32, 32)]  // above floor  → suffix-only prefill
    public async Task GenerateAsync_TqRewindFloor_ClampsPrefixReuse(
        int floor, int expectTruncate, int expectPrefillLen, int expectPrefillStart, long expectReused)
    {
        var fwd = new TqCompressedForwardPass(compressedLen: floor);
        using var engine = new InferenceEngine(
            fwd, new MultiTurnTokenizer(), "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await Drain(engine.GenerateAsync("turn1", sp));
        await Drain(engine.GenerateAsync("turn2", sp));

        Assert.Equal(expectTruncate, fwd.LastTruncateLength);
        Assert.Equal(expectPrefillLen, fwd.LastPrefillLength);
        Assert.Equal(expectPrefillStart, fwd.LastPrefillStartPos);
        Assert.Equal(expectReused, engine.PrefillTokensReused);
    }

    private static IAsyncEnumerable<string> GenerateAsync(
        InferenceEngine engine, string prompt, string canonical, SamplingParams sp)
    {
        // Adapt the IInferenceEngine GenerateChunksAsync to the test's text-only Drain helper.
        return Inner(engine.GenerateChunksAsync(prompt, sp, default, canonical));

        static async IAsyncEnumerable<string> Inner(IAsyncEnumerable<GenerateChunk> chunks)
        {
            await foreach (var c in chunks)
                if (c.Kind == GenerateChunkKind.Text) yield return c.Text;
        }
    }

    private static async Task Drain(IAsyncEnumerable<string> stream)
    {
        await foreach (var _ in stream) { }
    }

    /// <summary>
    /// Hand-rolled tokenizer that models the chat-template scenario for issue #102:
    /// each turn has a "canonical" rendering (history only) and a "full" rendering
    /// (history + assistant prep + inline &lt;think&gt; injection). Canonical is a
    /// strict token-level prefix of full. Turn 2's canonical extends turn 1's by
    /// 16 tokens (the scrubbed assistant response + new user message).
    ///   turn1_canon → tokens [0..24)             (24 tokens)
    ///   turn1_full  → tokens [0..24) + [200..208) (32 tokens = canon + assistant prep)
    ///   turn2_canon → tokens [0..24) + [100..116) (40 tokens = turn1_canon + new content)
    ///   turn2_full  → turn2_canon + [200..208)    (48 tokens)
    ///   mismatch    → [9,9,9,9]                   (not a prefix of anything else)
    /// </summary>
    private sealed class CanonicalChatTokenizer : ITokenizer
    {
        public int VocabSize => 300;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;

        public IReadOnlyList<int> Encode(string text) => text switch
        {
            "turn1_canon" => Enumerable.Range(0, 24).ToArray(),
            "turn1_full"  => Enumerable.Range(0, 24).Concat(Enumerable.Range(200, 8)).ToArray(),
            "turn2_canon" => Enumerable.Range(0, 24).Concat(Enumerable.Range(100, 16)).ToArray(),
            "turn2_full"  => Enumerable.Range(0, 24).Concat(Enumerable.Range(100, 16))
                                                    .Concat(Enumerable.Range(200, 8)).ToArray(),
            "mismatch"    => [9, 9, 9, 9],
            _             => throw new ArgumentException($"unknown prompt: {text}", nameof(text)),
        };

        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    // ── Mocks ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Hand-rolled tokenizer that returns two distinct token sequences sharing a
    /// 32-token (two-page) prefix:
    ///   "turn1" → [0..32)
    ///   "turn2" → [0..32) followed by [100..116)
    /// </summary>
    private sealed class MultiTurnTokenizer : ITokenizer
    {
        public int VocabSize => 200;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;

        public IReadOnlyList<int> Encode(string text)
        {
            // 32-token shared prefix for any prompt; "turn2" appends 16 fresh tokens and
            // "turn3" (issue #450) extends turn2 by a further 16 — so a request that reuses
            // all of turn2's 48 prefilled tokens is distinguishable from one that reuses 32.
            var prefix = Enumerable.Range(0, 32).ToArray();
            if (text == "turn2")
                return prefix.Concat(Enumerable.Range(100, 16)).ToArray();
            if (text == "turn3")
                return prefix.Concat(Enumerable.Range(100, 16)).Concat(Enumerable.Range(200, 16)).ToArray();
            return prefix;
        }

        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    /// <summary>
    /// Models the Qwen3.6 GDN hybrid contract: <see cref="TruncateTo"/> only accepts
    /// length 0 or the current length. Any other call records the attempt and throws.
    /// </summary>
    private sealed class RewindIncompatibleForwardPass : IForwardPass
    {
        private readonly float[] _logits = new float[200];
        private int _length;

        public bool PartialTruncateAttempted { get; private set; }
        public bool ResetCacheCalled { get; private set; }

        public bool SupportsPartialRewind => false;

        public int VocabSize => 200;
        public int MaxSeqLen => 4096;

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            _length = position + 1;
            return EosLogits();
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            _length = startPos + tokens.Count;
            return EosLogits();
        }

        public void TruncateTo(int length)
        {
            if (length == _length || length == 0)
            {
                if (length == 0) _length = 0;
                return;
            }
            PartialTruncateAttempted = true;
            throw new NotSupportedException(
                $"RewindIncompatibleForwardPass.TruncateTo({length}): only length == 0 or current ({_length}) is supported.");
        }

        public void ResetCache()
        {
            ResetCacheCalled = true;
            _length = 0;
        }

        public void Dispose() { }

        private ReadOnlySpan<float> EosLogits()
        {
            Array.Clear(_logits);
            _logits[Eos] = 1.0f;
            return _logits;
        }
    }

    /// <summary>
    /// Tokenizer paired with <see cref="SnapshotCapableForwardPass"/>. Generates
    /// a 32-token prefix; for "turn2" extends by 1 sampled-token-slot + 1 fresh
    /// token so the snapshot length (33) is a strict prefix of the new prompt
    /// and at least one suffix token remains.
    /// </summary>
    private sealed class SnapshotMultiTurnTokenizer : ITokenizer
    {
        public const int SampledTokenId = 50;

        public int VocabSize => 200;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;

        public IReadOnlyList<int> Encode(string text)
        {
            // turn1: 32-token prefix (matches the GDN-hybrid 'apply chat template' result).
            var prefix = Enumerable.Range(0, 32).ToArray();
            if (text == "turn2")
            {
                // turn2 = [prefix(32) , SampledTokenId , fresh-token(110)]
                // The engine's full-sequence for turn1 is [prefix(32) , SampledTokenId],
                // length 33. turn2's first 33 tokens must equal that; the 34th token is
                // the fresh user-message extension that drives a non-empty prefill.
                return prefix.Concat([SampledTokenId, 110]).ToArray();
            }
            return prefix;
        }

        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    /// <summary>
    /// Models a GDN-hybrid contract: <see cref="SupportsPartialRewind"/> is false
    /// (recurrent state is destructively updated), but the pass exposes
    /// <see cref="CaptureSnapshot"/> / <see cref="SnapshotLength"/> so the engine
    /// can reuse state across chat turns whose prompt extends the prior transcript.
    /// </summary>
    private sealed class SnapshotCapableForwardPass : IForwardPass
    {
        private readonly float[] _logits = new float[200];
        private int _length;

        public int LastCapturedSnapshotLength { get; private set; } = -1;
        public int LastTruncateLength { get; private set; } = -1;
        public int LastPrefillLength { get; private set; } = -1;
        public int LastPrefillStartPos { get; private set; } = -1;
        public bool ResetCalledAfter { get; private set; }
        // Cumulative call history for tests that need to verify split-prefill behavior
        // (the canonical-snapshot path issues two Prefill calls and one mid-prefill
        // CaptureSnapshot per request).
        public List<(int Length, int StartPos)> PrefillCalls { get; } = [];
        public List<int> CaptureSnapshotCalls { get; } = [];

        public bool SupportsPartialRewind => false;
        public bool SupportsSnapshot => true;
        public int VocabSize => 200;
        public int MaxSeqLen => 4096;

        public int SnapshotLength { get; private set; } = -1;

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            _length = position + 1;
            return SampledLogits();
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            LastPrefillLength = tokens.Count;
            LastPrefillStartPos = startPos;
            PrefillCalls.Add((tokens.Count, startPos));
            _length = startPos + tokens.Count;
            return SampledLogits();
        }

        public void TruncateTo(int length)
        {
            // Mirror GDN semantics: accept 0, current, and SnapshotLength.
            // Record every call so the test can verify the engine took the
            // snapshot branch even when current == SnapshotLength (the trivial
            // no-op overlap that the real GDN pass would also accept).
            LastTruncateLength = length;
            if (length == 0) { _length = 0; SnapshotLength = -1; return; }
            if (length == _length) return;
            if (length == SnapshotLength && SnapshotLength >= 0)
            {
                _length = length;
                return;
            }
            throw new NotSupportedException(
                $"SnapshotCapableForwardPass.TruncateTo({length}): only 0, current ({_length}), or SnapshotLength ({SnapshotLength}) supported.");
        }

        public void ResetCache()
        {
            // After a successful TruncateTo(snapLen), the engine must NOT call ResetCache.
            // Record only when reset happens after the snapshot was captured — that's
            // the actual regression we're guarding against.
            if (SnapshotLength >= 0)
                ResetCalledAfter = true;
            _length = 0;
            SnapshotLength = -1;
        }

        public void CaptureSnapshot()
        {
            LastCapturedSnapshotLength = _length;
            CaptureSnapshotCalls.Add(_length);
            SnapshotLength = _length;
        }

        public void Dispose() { }

        private ReadOnlySpan<float> SampledLogits()
        {
            // Force a non-stop token so the decode loop runs once and the snapshot
            // is captured at "prompt + 1 generated token" length, mirroring real
            // generation. EOS is the only default stop id.
            Array.Clear(_logits);
            _logits[SnapshotMultiTurnTokenizer.SampledTokenId] = 1.0f;
            return _logits;
        }
    }

    /// <summary>
    /// Issue #106: turn 1's 32-token prompt; turn 2's prompt is the same 32 tokens
    /// plus 2 fresh tokens (50 then 110). With the MTP path's immediate-EOS decode
    /// the snapshot is captured at length 32; turn 2's first 32 match exactly so
    /// snapshot reuse fires. The fresh tokens drive a non-empty Prefill + PrefillMtp
    /// at <c>startPos = 32</c>, exercising the snapshot-restored branch.
    /// </summary>
    private sealed class MtpLegacyMultiTurnTokenizer : ITokenizer
    {
        public int VocabSize => 200;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;

        public IReadOnlyList<int> Encode(string text)
        {
            var prefix = Enumerable.Range(0, 32).ToArray();
            return text == "turn2"
                ? prefix.Concat([50, 110]).ToArray()
                : prefix;
        }

        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    /// <summary>
    /// MTP-capable mock forward pass (issue #106). Reports <c>HasMtpHead = true</c>
    /// and exposes the snapshot + MTP-KV-truncation behaviour the real
    /// <see cref="HybridGdnForwardPass"/> / <see cref="CudaHybridGdnForwardPass"/> ship:
    /// <list type="bullet">
    ///   <item><c>TruncateTo(snapLen)</c> internally calls <c>MtpTruncateTo(snapLen)</c>
    ///         so the engine doesn't need to know about MTP KV bookkeeping.</item>
    ///   <item>Returns EOS-favoured logits so <see cref="MtpDecoder"/> exits on the
    ///         first iter — keeps the tests focused on prefill / PrefillMtp / snapshot
    ///         plumbing rather than MTP decode minutiae.</item>
    ///   <item>Records every <c>Prefill</c>, <c>PrefillMtp</c>, <c>TruncateTo</c>,
    ///         <c>MtpTruncateTo</c>, and <c>CaptureSnapshot</c> call.</item>
    /// </list>
    /// </summary>
    private sealed class SnapshotMtpForwardPass : IForwardPass
    {
        private readonly float[] _logits;
        private readonly float[] _lastHidden;
        private int _length;
        private int _mtpLength;

        public int LastTruncateLength { get; private set; } = -1;
        public bool ResetCalledAfter { get; private set; }
        public List<(int Length, int StartPos)> PrefillCalls { get; } = [];
        public List<(int Length, int StartPos)> PrefillMtpCalls { get; } = [];
        public List<int> CaptureSnapshotCalls { get; } = [];
        public List<int> TruncateCalls { get; } = [];
        public List<int> MtpTruncateCalls { get; } = [];

        public bool SupportsPartialRewind => false;
        public bool SupportsSnapshot => true;
        public bool HasMtpHead => true;
        public int VocabSize => 200;
        public int MaxSeqLen => 4096;
        public int SnapshotLength { get; private set; } = -1;
        public ReadOnlySpan<float> LastHidden => _lastHidden;

        public SnapshotMtpForwardPass()
        {
            _logits = new float[VocabSize];
            _logits[Eos] = 1f;
            // Non-zero hidden so MtpDecoder.Initialize doesn't reject IsEmpty.
            _lastHidden = new float[16];
            _lastHidden[0] = 1f;
        }

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            _length = position + 1;
            return _logits;
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            PrefillCalls.Add((tokens.Count, startPos));
            _length = startPos + tokens.Count;
            return _logits;
        }

        public void TruncateTo(int length)
        {
            LastTruncateLength = length;
            TruncateCalls.Add(length);
            if (length == 0)
            {
                _length = 0;
                _mtpLength = 0;
                SnapshotLength = -1;
                return;
            }
            if (length == _length) return;
            if (length == SnapshotLength && SnapshotLength >= 0)
            {
                _length = length;
                // Mirror the real pass: snapshot restore also rewinds MTP KV.
                _mtpLength = length;
                MtpTruncateCalls.Add(length);
                return;
            }
            throw new NotSupportedException(
                $"SnapshotMtpForwardPass.TruncateTo({length}): only 0, current ({_length}), or SnapshotLength ({SnapshotLength}) supported.");
        }

        public void ResetCache()
        {
            if (SnapshotLength >= 0) ResetCalledAfter = true;
            _length = 0;
            _mtpLength = 0;
            SnapshotLength = -1;
        }

        public void CaptureSnapshot()
        {
            CaptureSnapshotCalls.Add(_length);
            SnapshotLength = _length;
        }

        public void PrefillMtp(IReadOnlyList<int> tokens, int startPos = 0)
        {
            PrefillMtpCalls.Add((tokens.Count, startPos));
            _mtpLength = startPos + tokens.Count;
        }

        public void MtpResetCache() { _mtpLength = 0; }

        public void MtpTruncateTo(int length)
        {
            MtpTruncateCalls.Add(length);
            _mtpLength = length;
        }

        public void Dispose() { }
    }

    /// <summary>
    /// Rewind-capable forward pass that records the arguments to the last
    /// <see cref="TruncateTo"/> and <see cref="Prefill"/> calls so the test can
    /// confirm the prefix path was taken.
    /// </summary>
    private sealed class RewindCapableForwardPass : IForwardPass
    {
        private readonly float[] _logits = new float[200];

        // Sentinel -1 means "never called" — distinguishes a missing call from a TruncateTo(0).
        public int LastTruncateLength { get; private set; } = -1;
        public int LastPrefillLength { get; private set; } = -1;
        public int LastPrefillStartPos { get; private set; } = -1;

        /// Token the greedy sampler will pick. Defaults to EOS so decode stops after one
        /// step; set to a non-stop id to keep the decode loop running.
        public int DecodeTokenId { get; init; } = Eos;

        /// When set, the first decode <see cref="Forward"/> trips this source — the
        /// deterministic way to land a cancellation mid-decode.
        public CancellationTokenSource? CancelOnDecode { get; set; }

        public bool SupportsPartialRewind => true;

        public int VocabSize => 200;
        public int MaxSeqLen => 4096;

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            CancelOnDecode?.Cancel();
            return NextLogits();
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            LastPrefillLength = tokens.Count;
            LastPrefillStartPos = startPos;
            return NextLogits();
        }

        public void TruncateTo(int length) => LastTruncateLength = length;
        public void ResetCache() { }
        public void Dispose() { }

        private ReadOnlySpan<float> NextLogits()
        {
            Array.Clear(_logits);
            _logits[DecodeTokenId] = 1.0f;
            return _logits;
        }
    }

    /// <summary>
    /// Models the TurboQuant rewind contract shared by the CUDA/Vulkan dense passes and
    /// <c>TurboQuantKvCache</c>: partial rewind is supported, but positions below
    /// <see cref="IForwardPass.MinRewindLength"/> are compressed in place and cannot be
    /// restored, so <see cref="TruncateTo"/> throws for them — a below-floor call fails
    /// the test by propagating, no tracking flag needed. The floor stays pinned across
    /// turns (the real passes clear it in ResetCache); holding it constant keeps the
    /// fixture on the one axis under test.
    /// </summary>
    private sealed class TqCompressedForwardPass(int compressedLen) : IForwardPass
    {
        private readonly float[] _logits = new float[200];

        // Sentinel -1 means "never called".
        public int LastTruncateLength { get; private set; } = -1;
        public int LastPrefillLength { get; private set; } = -1;
        public int LastPrefillStartPos { get; private set; } = -1;

        public bool SupportsPartialRewind => true;
        public int MinRewindLength => compressedLen;

        public int VocabSize => 200;
        public int MaxSeqLen => 4096;

        public ReadOnlySpan<float> Forward(int token, int position) => EosLogits();

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            LastPrefillLength = tokens.Count;
            LastPrefillStartPos = startPos;
            return EosLogits();
        }

        public void TruncateTo(int length)
        {
            LastTruncateLength = length;
            if (length < compressedLen)
                throw new NotSupportedException(
                    $"TruncateTo({length}) cannot rewind into the TQ-compressed region " +
                    $"(tqCompressedLen={compressedLen}).");
        }

        public void ResetCache() { }
        public void Dispose() { }

        private ReadOnlySpan<float> EosLogits()
        {
            Array.Clear(_logits);
            _logits[Eos] = 1.0f;
            return _logits;
        }
    }

    /// <summary>
    /// Issue #212 tokenizer modelling the agentic interleave: a long stable request, a short
    /// auxiliary request with disjoint tokens, then the long request again.
    ///   "long" → [0..64)                 (64 tokens; the stable system-prefix-like sequence)
    ///   "aux"  → [500..516)              (16 tokens; disjoint, no shared prefix with "long")
    /// </summary>
    private sealed class InterleavedAgenticTokenizer : ITokenizer
    {
        public int VocabSize => 600;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;

        public IReadOnlyList<int> Encode(string text) => text switch
        {
            "long"  => Enumerable.Range(0, 64).ToArray(),
            "aux"   => Enumerable.Range(500, 16).ToArray(),
            // 32-token (>1 page) scratch-sized prompts for the mid-decode-failure test, where a
            // reusable prefix requires length > PageSize. "aux2" is disjoint from "aux32".
            "aux32"  => Enumerable.Range(500, 32).ToArray(),
            "aux2"   => Enumerable.Range(600, 32).ToArray(),
            _       => throw new ArgumentException($"unknown prompt: {text}", nameof(text)),
        };

        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    /// <summary>
    /// Rewind-capable forward pass that ALSO implements <see cref="IMultiSlotKvCache"/> (issue
    /// #212). Each <see cref="Slot"/> records the Prefill/TruncateTo/ResetCache calls routed to it
    /// while it is the active (bound) slot, so a test can assert which slot served each request and
    /// that a scratch-bound request never touched the owned slot. Slot 0 is the owned cache; each
    /// <see cref="AllocateScratchSlot"/> hands out the next id.
    /// </summary>
    private sealed class MultiSlotForwardPass : IForwardPass, IMultiSlotKvCache
    {
        private const int HiddenDim = 16;
        private readonly float[] _logits = new float[600];
        private readonly Slot _owned = new(0);
        private readonly List<Slot> _scratch = [];
        private Slot _active;

        public MultiSlotForwardPass() => _active = _owned;

        public sealed class Slot(int id) : ISequenceKvCache
        {
            public int Id { get; } = id;
            public int Length;
            public int Capacity = int.MaxValue;
            public List<(int Length, int StartPos)> Prefills { get; } = [];
            public List<int> Truncates { get; } = [];
            public int Resets;
            public void Dispose() { }
        }

        public Slot Owned => _owned;
        public IReadOnlyList<Slot> Scratch => _scratch;
        /// <summary>Ids of slots bound via ActivateSlot, in order (empty in single-slot mode).</summary>
        public List<int> ActivatedIds { get; } = [];

        /// <summary>When set, Prefill returns a non-stop token so the decode loop runs at least
        /// one Forward (the EOS-on-prefill default breaks before any Forward).</summary>
        public bool EmitNonStopFirst;
        /// <summary>One-shot: the next Forward throws, simulating a mid-decode failure/cancel.</summary>
        public bool FailNextDecode;
        private const int NonStopToken = 5;

        /// <summary>Issue #452: advertise an MTP head so the engine takes the single-slot MTP
        /// path (no SelectSlot bind) while still being rewindable + multi-slot capable — the
        /// exact combination a dense MTP model with SHARPI_PREFIX_SLOTS=2 presents.</summary>
        public bool MtpHead { get; init; }
        /// <summary>Issue #451: DSpark needs hidden taps and batched verify from the target.</summary>
        public bool Taps { get; init; }

        private readonly float[] _hidden = CreateHidden();
        private static float[] CreateHidden()
        {
            // Non-zero: MtpDecoder.Initialize rejects an empty/zero hidden.
            var h = new float[HiddenDim];
            h[0] = 1f;
            return h;
        }

        // ── IForwardPass: route to the active slot ──
        public bool SupportsPartialRewind => true;
        public int VocabSize => 600;
        public int MaxSeqLen => 100_000;

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            if (FailNextDecode)
            {
                FailNextDecode = false;
                throw new InvalidOperationException("simulated mid-decode failure");
            }
            _active.Length = position + 1;
            return EosLogits();
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            _active.Prefills.Add((tokens.Count, startPos));
            _active.Length = startPos + tokens.Count;
            return EmitNonStopFirst ? OneHot(NonStopToken) : EosLogits();
        }

        public void TruncateTo(int length)
        {
            _active.Truncates.Add(length);
            _active.Length = length;
        }

        public void ResetCache()
        {
            _active.Resets++;
            _active.Length = 0;
        }

        public void Dispose() { }

        // ── MTP head (issue #452) ──
        // Just enough surface for MtpDecoder.DecodeSequential: a hidden to save, an MTP
        // draft forward, and the KV-side no-ops. The draft always proposes EOS, so it is
        // rejected and decode advances one token per iter through Forward — which is where
        // FailNextDecode lands the mid-decode abort.
        public bool HasMtpHead => MtpHead;
        public ReadOnlySpan<float> LastHidden => MtpHead || Taps ? _hidden : default;
        public ReadOnlySpan<float> MtpForward(int token, int position, ReadOnlySpan<float> prevHidden)
            => EosLogits();
        public void PrefillMtp(IReadOnlyList<int> tokens, int startPos = 0) { }
        public void MtpResetCache() { }
        public void MtpTruncateTo(int length) { }

        // ── Hidden taps + batched verify (issue #451, DSpark) ──
        public bool SupportsHiddenTaps => Taps;
        public bool SupportsBatchVerify => Taps;
        public int HiddenTapDim => HiddenDim;
        public void EnableHiddenTaps(ReadOnlySpan<int> layerIds) { }

        public ReadOnlySpan<float> HiddenTapsAt(int position)
            => position >= 0 && position < _active.Length ? _hidden : default;

        public float[][] BatchVerify(int[] tokens, int startPos)
        {
            _active.Length = startPos + tokens.Length;
            var rows = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++)
            {
                rows[i] = new float[VocabSize];
                rows[i][Eos] = 1f;
            }
            return rows;
        }

        // ── IMultiSlotKvCache ──
        public bool SupportsMultiSlotPrefix => true;
        public ISequenceKvCache OwnedSlot => _owned;

        public ISequenceKvCache AllocateScratchSlot(int capacityTokens)
        {
            var s = new Slot(_scratch.Count + 1) { Capacity = capacityTokens };
            _scratch.Add(s);
            return s;
        }

        public void ActivateSlot(ISequenceKvCache slot)
        {
            _active = (Slot)slot;
            ActivatedIds.Add(_active.Id);
        }

        public void DeactivateSlot() => _active = _owned;

        private ReadOnlySpan<float> EosLogits() => OneHot(Eos);

        private ReadOnlySpan<float> OneHot(int token)
        {
            Array.Clear(_logits);
            _logits[token] = 1.0f;
            return _logits;
        }
    }

    /// <summary>
    /// Inert DSpark draft sized to match <see cref="MultiSlotForwardPass"/>, so the attach-time
    /// capability checks all pass and the multi-slot rejection (issue #451) is what trips.
    /// </summary>
    private sealed class StubDSparkDraft : IDSparkDraft
    {
        public int BlockSize => 2;
        public int VocabSize => 600;
        public int TapDim => 16;
        public int ContextLength { get; private set; }
        public int MaxContext => int.MaxValue;

        public void AppendContext(ReadOnlySpan<float> taps, int startPos, int count)
            => ContextLength = startPos + count;

        public DSparkProposal ProposeBlock(int anchorToken, int anchorPos)
            => new(new int[BlockSize], new float[BlockSize]);

        public void TruncateContext(int length) => ContextLength = Math.Min(ContextLength, length);
        public void ResetContext() => ContextLength = 0;
        public void Dispose() { }
    }

    /// <summary>
    /// Sets environment variables for the duration of a test and restores their prior values on
    /// dispose. The engine reads SHARPI_PREFIX_* only in its constructor; other engine tests use
    /// non-multi-slot fakes, so a stray concurrent read of these vars is inert.
    /// </summary>
    private sealed class EnvScope : IDisposable
    {
        private readonly (string Key, string? Prior)[] _saved;

        public EnvScope(params (string Key, string? Value)[] vars)
        {
            _saved = new (string, string?)[vars.Length];
            for (int i = 0; i < vars.Length; i++)
            {
                _saved[i] = (vars[i].Key, Environment.GetEnvironmentVariable(vars[i].Key));
                Environment.SetEnvironmentVariable(vars[i].Key, vars[i].Value);
            }
        }

        public void Dispose()
        {
            foreach (var (key, prior) in _saved)
                Environment.SetEnvironmentVariable(key, prior);
        }
    }
}
