using System.Runtime.CompilerServices;
using System.Threading.Channels;
using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Top-level inference engine. Wraps a forward pass + tokenizer, applies pre-formatted
/// prompts (caller applies chat template), and provides serialized async generation.
/// One request runs at a time; concurrent callers block in arrival order.
///
/// Prefix caching: if successive prompts share a page-aligned token prefix, the KV cache
/// for those positions is reused and only the new suffix is prefilled — eliminating
/// redundant computation for repeated system prompts.
///
/// Reasoning support: when constructed with <c>thinkTokenId</c> / <c>endThinkTokenId</c>
/// for a model that emits <c>&lt;think&gt;...&lt;/think&gt;</c>, the engine splits output
/// into <see cref="GenerateChunkKind.Thinking"/> and <see cref="GenerateChunkKind.Text"/>
/// chunks. Boundary tokens themselves are consumed and never appear in chunk text.
/// </summary>
public sealed class InferenceEngine : IInferenceEngine, IDisposable, IAsyncDisposable
{
    private readonly IForwardPass _fwd;
    private readonly ITokenizer _tokenizer;
    private readonly IDisposable[] _owned;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly int _thinkTokenId;
    private readonly int _endThinkTokenId;

    // Cancelled by Dispose/DisposeAsync; linked into every generation's token so a
    // shutdown stops the in-flight background worker promptly rather than letting it
    // run to MaxNewTokens (issue #132).
    private readonly CancellationTokenSource _shutdownCts = new();

    // Prefix caching state (guarded by _gate — only accessed during generation).
    private int[]? _prevTokens;

    // Observability counters (updated via Interlocked).
    private int _pendingCount;
    private int _activeCount;
    private long _prefillTokensReused;

    // 0 = live, 1 = disposed. Interlocked so Dispose/DisposeAsync are single-shot even
    // if both are called (or called concurrently).
    private int _disposed;

    // Upper bound on how long Dispose waits for an in-flight generation to drain before
    // giving up. The worker is cancelled (via _shutdownCts) and cooperatively checks
    // cancellation between decode tokens and prefill chunks, so the real wait is one
    // forward/prefill-chunk; this is only a backstop against a wedged backend or an
    // abandoned (never-disposed) enumerator stranding the gate. On timeout the forward
    // pass is leaked rather than freed (see DisposeCore). Overridable by tests via
    // InternalsVisibleTo to exercise the timeout path without a 10s wait.
    internal TimeSpan _disposeDrainTimeout = TimeSpan.FromSeconds(10);

    public string ModelId { get; }

    /// <summary>Number of callers blocked waiting to acquire the generation gate.</summary>
    public int QueueDepth => _pendingCount;

    /// <summary>1 while a generation is in progress, 0 otherwise.</summary>
    public int ActiveRequests => _activeCount;

    /// <inheritdoc/>
    /// <remarks>
    /// True when the underlying pass supports partial rewind (page-aligned reuse across
    /// any matching prefix) OR a single-slot end-of-prefill snapshot (issue #21 / #102,
    /// used for chat-continuation reuse on GDN-hybrid passes via a caller-supplied
    /// canonical-history hint). A <c>false</c> here is the only configuration where the
    /// engine cannot reuse KV state across turns; reuse on a <c>true</c> backend is still
    /// gated per-request on a matching prefix being available.
    /// </remarks>
    public bool PrefixCacheEnabled => _fwd.SupportsPartialRewind || _fwd.SupportsSnapshot;

    /// <inheritdoc/>
    public long PrefillTokensReused => Interlocked.Read(ref _prefillTokensReused);

    /// <param name="fwd">Forward pass implementation (CPU / GPU / Hybrid). Owned by this engine.</param>
    /// <param name="tokenizer">Tokenizer matching the model vocabulary.</param>
    /// <param name="modelId">Human-readable model identifier returned in API responses.</param>
    /// <param name="thinkTokenId">
    /// Token ID of the model's <c>&lt;think&gt;</c> marker, or <c>-1</c> if the model has no reasoning
    /// stream. When <c>-1</c>, all chunks are emitted as <see cref="GenerateChunkKind.Text"/>.
    /// </param>
    /// <param name="endThinkTokenId">
    /// Token ID of the model's <c>&lt;/think&gt;</c> marker, or <c>-1</c>. Must be paired with
    /// a non-negative <paramref name="thinkTokenId"/> to enable reasoning-stream splitting.
    /// </param>
    /// <param name="owned">Additional disposable resources owned by this engine (backend, model handle, etc.).</param>
    public InferenceEngine(
        IForwardPass fwd,
        ITokenizer tokenizer,
        string modelId,
        int thinkTokenId,
        int endThinkTokenId,
        params IDisposable[] owned)
    {
        _fwd = fwd;
        _tokenizer = tokenizer;
        ModelId = modelId;
        _thinkTokenId = thinkTokenId;
        _endThinkTokenId = endThinkTokenId;
        _owned = owned;

        if (!fwd.SupportsPartialRewind)
        {
            // Issue #102: GDN-hybrid passes report SupportsPartialRewind=false but still
            // expose a single-slot snapshot via CaptureSnapshot/SnapshotLength. The engine
            // uses it for chat-continuation reuse when callers pass a canonicalHistoryPrefix,
            // so flagging the cache as universally "disabled" is misleading. Distinguish
            // the two flavours so server logs match what actually happens at request time.
            if (fwd.SupportsSnapshot)
            {
                Console.Error.WriteLine(
                    $"[InferenceEngine] partial-rewind prefix cache disabled — {fwd.GetType().Name} " +
                    "uses destructive recurrent state. Snapshot-based prefix reuse is available across " +
                    "chat turns when the caller supplies a canonical-history hint (chat completion endpoints do).");
            }
            else
            {
                Console.Error.WriteLine(
                    $"[InferenceEngine] prefix cache disabled — {fwd.GetType().Name} reports SupportsPartialRewind == false " +
                    "and exposes no snapshot facility. Multi-turn requests will re-prefill the full prompt.");
            }
        }
    }

    /// <summary>
    /// Back-compat constructor preserving the original positional signature
    /// (no reasoning-token IDs). Equivalent to passing <c>thinkTokenId = -1</c>.
    /// </summary>
    public InferenceEngine(
        IForwardPass fwd,
        ITokenizer tokenizer,
        string modelId,
        params IDisposable[] owned)
        : this(fwd, tokenizer, modelId, thinkTokenId: -1, endThinkTokenId: -1, owned)
    {
    }

    /// <summary>
    /// Prompt-prefill batch size, in tokens. A single <see cref="IForwardPass.Prefill"/> call is
    /// opaque to the request's <see cref="CancellationToken"/> — the engine only checks <c>ct</c>
    /// between decode tokens — so a large-prompt prefill would otherwise run to completion even
    /// after the client has disconnected, pinning the engine gate and a CPU/GPU core for the dead
    /// request. Splitting the prompt into chunks and checking <c>ct</c> between them makes prefill
    /// cooperatively cancellable, bounding the post-disconnect compute to (at most) one chunk.
    /// <para>
    /// Chunking is numerically identical to a single prefill: a transformer forward produces one
    /// independent output row per token (GEMM output rows don't mix across the token batch) and the
    /// GDN recurrent state carries across calls via the cache — the same multi-call pattern the
    /// prefix-reuse and canonical-snapshot prefill paths already rely on. Tunable via
    /// <c>SHARPI_PREFILL_CHUNK</c> (set very large to effectively disable chunking, e.g. for
    /// prefill throughput benchmarking).
    /// </para>
    /// </summary>
    private static readonly int PrefillChunkSize =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPI_PREFILL_CHUNK"), out int c) && c > 0
            ? c
            : 512;

    /// <summary>
    /// Prefill the absolute token range <c>[from, to)</c> (where <c>tokens[i]</c> sits at cache
    /// position <c>i</c>) in <see cref="PrefillChunkSize"/>-token chunks, checking
    /// <paramref name="ct"/> before each chunk so a client disconnect aborts prefill promptly.
    /// Returns the logits for the last processed token (the only ones a caller consumes).
    /// </summary>
    private ReadOnlySpan<float> PrefillChunked(int[] tokens, int from, int to, CancellationToken ct)
    {
        ReadOnlySpan<float> logits = default;
        for (int pos = from; pos < to; pos += PrefillChunkSize)
        {
            ct.ThrowIfCancellationRequested();
            int len = Math.Min(PrefillChunkSize, to - pos);
            logits = _fwd.Prefill(new ArraySegment<int>(tokens, pos, len), pos);
        }
        return logits;
    }

    /// <summary>
    /// Cancellation-checkpointed twin of <see cref="PrefillChunked"/> for the MTP attention KV
    /// cache (issue #33). Same chunking rationale; populates one chunk per call so a disconnect
    /// during MTP-prompt population is honored within one chunk.
    /// </summary>
    private void PrefillMtpChunked(int[] tokens, int from, int to, CancellationToken ct)
    {
        for (int pos = from; pos < to; pos += PrefillChunkSize)
        {
            ct.ThrowIfCancellationRequested();
            int len = Math.Min(PrefillChunkSize, to - pos);
            _fwd.PrefillMtp(new ArraySegment<int>(tokens, pos, len), pos);
        }
    }

    /// <summary>
    /// Finds the longest page-aligned prefix shared between the new token array and the cached
    /// previous token array, returning its length (0 if no reusable prefix exists).
    /// </summary>
    private int FindCacheablePrefix(int[] tokens)
    {
        if (_prevTokens == null || tokens.Length <= PagedKvCache.PageSize)
            return 0;

        // Compare up to all-but-last-page tokens (need at least one page to bother).
        int maxCompare = Math.Min(tokens.Length - 1, _prevTokens.Length);
        int match = 0;
        while (match < maxCompare && tokens[match] == _prevTokens[match])
            match++;

        // Align down to page boundary (must be at least one full page).
        int aligned = (match / PagedKvCache.PageSize) * PagedKvCache.PageSize;
        return aligned;
    }

    /// <summary>
    /// Back-compat string-stream view of <see cref="GenerateChunksAsync"/>: yields only
    /// user-facing answer text, suppressing reasoning chunks. Equivalent to the default
    /// interface-method implementation on <see cref="IInferenceEngine"/> but callable
    /// directly on the concrete type.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var c in GenerateChunksAsync(prompt, sp, ct).WithCancellation(ct).ConfigureAwait(false))
        {
            if (c.Kind == GenerateChunkKind.Text)
                yield return c.Text;
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default,
        string? canonicalHistoryPrefix = null)
    {
        // Link the caller's token with the engine-shutdown token so Dispose can stop this
        // generation's background worker (issue #132). Everything below uses `ct` — the
        // linked token — so cancellation from either source aborts decode and releases the
        // gate via the finally blocks. Reading _shutdownCts.Token can race a concurrent
        // Dispose that already disposed it (we passed the _disposed check above, then got
        // pre-empted); surface that as the engine's ObjectDisposedException, not the
        // CancellationTokenSource's, so callers see a consistent object name.
        CancellationTokenSource linkedCts;
        try
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(InferenceEngine));
            linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, _shutdownCts.Token);
        }
        catch (ObjectDisposedException)
        {
            throw new ObjectDisposedException(nameof(InferenceEngine));
        }
        using var _linkedCts = linkedCts;
        ct = linkedCts.Token;

        Interlocked.Increment(ref _pendingCount);
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        Interlocked.Decrement(ref _pendingCount);
        Interlocked.Increment(ref _activeCount);
        try
        {
            var channel = Channel.CreateUnbounded<GenerateChunk>(
                new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

            // Capture reasoning-token IDs into locals to avoid repeated field reads in the hot loop.
            int thinkId = _thinkTokenId;
            int endThinkId = _endThinkTokenId;
            bool thinkingEnabled = thinkId >= 0 && endThinkId >= 0;

            // Run the blocking CPU generation on a thread-pool thread.
            var genTask = Task.Run(() =>
            {
                try
                {
                    var tokens = _tokenizer.Encode(prompt).ToArray();
                    var rng = new Random();
                    System.Collections.Immutable.ImmutableArray<int> stopIds =
                        sp.StopTokenIds is { } userStops ? [.. userStops] : _tokenizer.EogTokenIds;

                    // Issue #102: canonical-history prefix resolution. The endpoint passes a
                    // chat-template render of just the message history (add_generation_prompt=false);
                    // tokenize it and verify it's a strict token-level prefix of the generating
                    // prompt. The validated boundary becomes (a) the position at which to capture
                    // the GDN snapshot mid-prefill, and (b) the tokens stored as _prevTokens so the
                    // next turn's generating prompt — which starts with that same canonical history
                    // and then appends scrubbed assistant content + the next user turn — matches.
                    // Empty / mismatched canonical → fall through to legacy behavior (snapshot at
                    // end-of-decode, _prevTokens stores generating + decoded).
                    int canonicalLen = 0;
                    if (!string.IsNullOrEmpty(canonicalHistoryPrefix))
                    {
                        var canonTokens = _tokenizer.Encode(canonicalHistoryPrefix).ToArray();
                        if (canonTokens.Length > 0
                            && canonTokens.Length < tokens.Length
                            && tokens.AsSpan(0, canonTokens.Length).SequenceEqual(canonTokens))
                        {
                            canonicalLen = canonTokens.Length;
                        }
                        else if (Environment.GetEnvironmentVariable("SHARPI_TRACE_SNAPSHOT") == "1")
                        {
                            Console.Error.WriteLine(
                                $"[InferenceEngine] canonical-history prefix is not a strict token prefix of the generating prompt " +
                                $"(canon.Len={canonTokens.Length}, prompt.Len={tokens.Length}); " +
                                "falling back to end-of-decode snapshot.");
                        }
                    }

                    // Track the full generated sequence (prompt + decoded tokens) so the
                    // next turn's prefix match has the right baseline — issue #21. Sized
                    // up front: prompt + MaxNewTokens is a safe upper bound. List<int> is
                    // cheap (vs. native alloc) and just shadows the tokens we'd lose to
                    // the channel writer.
                    var fullSeq = new List<int>(tokens.Length + Math.Max(1, sp.MaxNewTokens));
                    fullSeq.AddRange(tokens);

                    // MTP self-speculative decoding (issue #25): when the forward pass
                    // ships an MTP head AND we're in greedy, non-reasoning mode AND
                    // sp.SpecType doesn't forbid it, drive decode through MtpDecoder
                    // instead of the per-token sampling loop below. The MTP path
                    // emits one extra MTP-drafted token per main forward.
                    //
                    // Decided BEFORE prefix-reuse (issue #33): MTP requires the MTP KV
                    // cache to cover every prompt position via PrefillMtp, which only
                    // works when Prefill starts from position 0. Prefix reuse is
                    // therefore skipped on MTP runs.
                    //
                    // sp.SpecType wires the llama.cpp-compatible --spec-type flag:
                    //   Auto  → enable when HasMtpHead && greedy && !thinking
                    //   None  → off (always)
                    //   Mtp   → on; surface a clear error if any prerequisite is missing
                    //
                    // SHARPI_DISABLE_MTP=1 is a back-compat off-switch that wins over Auto
                    // and Mtp (so existing benchmarking scripts that set it keep working).
                    bool mtpEnvDisabled = Environment.GetEnvironmentVariable("SHARPI_DISABLE_MTP") == "1";
                    bool useMtp;
                    switch (sp.SpecType)
                    {
                        case SpecType.None:
                            useMtp = false;
                            break;
                        case SpecType.Mtp:
                            if (mtpEnvDisabled)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=Mtp conflicts with SHARPI_DISABLE_MTP=1. " +
                                    "Unset the env var or use SpecType.None.");
                            if (!_fwd.HasMtpHead)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=Mtp requires a model with an MTP head. " +
                                    $"{_fwd.GetType().Name} reports HasMtpHead=false (no nextn tensors in the GGUF).");
                            if (sp.Temperature > 0f)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=Mtp requires greedy sampling (Temperature=0). " +
                                    "MTP verification is greedy (argmax match); sampling support is not yet implemented.");
                            if (thinkingEnabled)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=Mtp is incompatible with reasoning mode. " +
                                    "Pass --no-thinking (or render the chat template with enable_thinking=false).");
                            useMtp = true;
                            break;
                        default: // Auto
                            useMtp = _fwd.HasMtpHead
                                && !mtpEnvDisabled
                                && sp.Temperature <= 0f
                                && !thinkingEnabled;
                            break;
                    }

                    // --spec-draft-n-max parity with llama.cpp (issue #30): the MTP
                    // draft-chain length per step. Unset (0) resolves via
                    // SHARPI_MTP_DRAFT_N → built-in default; MtpDecoder clamps per
                    // step against the pass's snapshot-ring capacity
                    // (MaxBatchVerifyTokens), so over-asking degrades gracefully.
                    int mtpDraftN = MtpDecoder.ResolveDraftN(sp.SpecDraftNMax);

                    // Prefix cache decision: two branches.
                    //   (a) Rewindable attention pass — existing FindCacheablePrefix path,
                    //       page-aligned KV reuse.
                    //   (b) Non-rewindable GDN hybrid with a snapshot — issue #21: try
                    //       exact-match against the most-recent snapshot length so the
                    //       held GDN recurrent state can be restored. Works for MTP
                    //       runs too (issue #106) — PrefillMtp accepts startPos > 0 and
                    //       TruncateTo soft-truncates the MTP KV alongside the trunk.
                    // If neither branch hits, fall back to a full ResetCache + Prefill.
                    int prefixLen = 0;
                    if (_fwd.SupportsPartialRewind)
                    {
                        int candidate = FindCacheablePrefix(tokens);
                        if (candidate > 0)
                        {
                            _fwd.TruncateTo(candidate);
                            prefixLen = candidate;
                        }
                    }
                    else if (_fwd.SupportsSnapshot && _fwd.SnapshotLength > 0 && _prevTokens != null)
                    {
                        int snapLen = _fwd.SnapshotLength;
                        // The pair (snapshot @ snapLen, _prevTokens) is maintained as an
                        // invariant: on the canonical path both are written together immediately
                        // after CaptureSnapshot, on the legacy path both are written together at
                        // end-of-decode. snapLen must be a strict prefix of the new prompt (need
                        // at least one suffix token to drive the decoder) AND a token-level
                        // prefix of _prevTokens — the latter is the actual reuse precondition.
                        if (snapLen <= tokens.Length - 1 && snapLen <= _prevTokens.Length
                            && tokens.AsSpan(0, snapLen).SequenceEqual(_prevTokens.AsSpan(0, snapLen)))
                        {
                            _fwd.TruncateTo(snapLen);
                            prefixLen = snapLen;
                        }
                    }
                    if (prefixLen == 0)
                    {
                        _fwd.ResetCache();
                    }
                    else
                    {
                        // Issue #22 observability: account reused tokens regardless of
                        // mechanism (attention partial-rewind or GDN snapshot).
                        Interlocked.Add(ref _prefillTokensReused, prefixLen);
                    }

                    // Prefill: process all prompt tokens (or just the suffix after the cached prefix).
                    //
                    // Issue #102 split: when a canonical-history boundary lies strictly between
                    // prefixLen and the end of the prompt, run the prefill in two stages with a
                    // CaptureSnapshot in between. That snapshot sits at the canonical boundary so
                    // the next turn — whose generating prompt starts with the same canonical
                    // history plus a scrubbed assistant response and a fresh user turn — can
                    // restore it. Issue #106: also applies on MTP runs; the sticky hidden
                    // history buffer + MTP KV soft-truncate make PrefillMtp(startPos=snapLen)
                    // viable. Skipped for backends that don't expose a snapshot (the snapshot
                    // call is a no-op, but skipping the split keeps the prefill in one shot
                    // for cache efficiency).
                    bool useCanonicalSnapshot =
                        canonicalLen > prefixLen
                        && canonicalLen < tokens.Length
                        && _fwd.SupportsSnapshot;

                    // Reused by the MTP path below for PrefillMtp(suffixTokens, prefixLen).
                    int[] suffixTokens = prefixLen > 0 ? tokens[prefixLen..] : tokens;

                    ReadOnlySpan<float> logits;
                    if (useCanonicalSnapshot)
                    {
                        PrefillChunked(tokens, prefixLen, canonicalLen, ct);
                        _fwd.CaptureSnapshot();
                        // Pair _prevTokens with the snapshot atomically: stage-2 failure or
                        // mid-decode cancellation now leaves snapshot + _prevTokens consistent
                        // with each other (both describe this turn's canonical state). Without
                        // the immediate write, a stage-2 throw would leave the snapshot pointing
                        // at this turn's canonical while _prevTokens still described the prior
                        // turn — the next request could pass the snapshot-match check and
                        // TruncateTo to state that doesn't correspond to its prefix.
                        _prevTokens = tokens[..canonicalLen];
                        logits = PrefillChunked(tokens, canonicalLen, tokens.Length, ct);
                    }
                    else
                    {
                        if (suffixTokens.Length > 0)
                            logits = PrefillChunked(tokens, prefixLen, tokens.Length, ct);
                        else
                            logits = _fwd.Forward(tokens[^1], tokens.Length - 1);
                    }

                    if (useMtp)
                    {
                        // Initialize captures the main logits + LastHidden BEFORE
                        // PrefillMtp's MtpForward calls overwrite the shared _logits
                        // scratch buffer (issue #33). PrefillMtp does not touch
                        // _lastHidden, so the captured hidden remains h_{N-1}.
                        var mtpDec = new MtpDecoder(_fwd);
                        mtpDec.Initialize(tokens.Length, logits);

                        // Issue #33: populate the MTP KV cache for the prompt so the
                        // first decode-step MTP attention sees the full prompt context.
                        // ~1.6%/token overhead — only paid on MTP-enabled runs.
                        if (suffixTokens.Length > 0)
                            PrefillMtpChunked(tokens, prefixLen, tokens.Length, ct);

                        var textDecMtp = new Utf8StreamDecoder();

                        mtpDec.Decode(sp.MaxNewTokens, stopIds.AsSpan(), tok =>
                        {
                            fullSeq.Add(tok);
                            var bytes = _tokenizer.DecodeBytes(tok);
                            var chunk = textDecMtp.Append(bytes);
                            if (chunk.Length > 0)
                                channel.Writer.TryWrite(
                                    new GenerateChunk(GenerateChunkKind.Text, chunk));
                        }, pMin: sp.SpecDraftPMin, draftN: mtpDraftN, ct: ct);

                        var textFlushMtp = textDecMtp.Flush();
                        if (textFlushMtp.Length > 0)
                            channel.Writer.TryWrite(
                                new GenerateChunk(GenerateChunkKind.Text, textFlushMtp));

                        if (Environment.GetEnvironmentVariable("SHARPI_TRACE_MTP") == "1"
                            && mtpDec.TotalDraftsEmitted > 0)
                        {
                            Console.Error.WriteLine(
                                $"[InferenceEngine] MTP: {mtpDec.TotalDraftsAccepted}/{mtpDec.TotalDraftsEmitted} " +
                                $"drafts accepted ({mtpDec.AcceptanceRate:P1}); " +
                                $"phase ms draft={mtpDec.DraftMs:F0} verify={mtpDec.VerifyMs:F0} commit={mtpDec.CommitMs:F0}");
                        }

                        // End-of-decode snapshot — see the non-MTP twin below for the
                        // !useCanonicalSnapshot rationale.
                        if (!useCanonicalSnapshot && !ct.IsCancellationRequested)
                        {
                            _fwd.CaptureSnapshot();
                            _prevTokens = fullSeq.ToArray();
                        }
                        channel.Writer.TryComplete();
                        return;
                    }

                    // Decode loop. Separate stateful UTF-8 decoders for the answer stream and
                    // the thinking stream so multi-byte characters in either stream reassemble
                    // independently — the two streams never share decoder state.
                    var textDec = new Utf8StreamDecoder();
                    var thinkDec = new Utf8StreamDecoder();

                    // Seed inThinking from the prompt itself (issue #92). Qwen3.6 and other
                    // reasoning models append a bare `<think>` token to the generation prompt
                    // via their chat template, so the model is already inside a reasoning
                    // block before the first sampled token. Without this scan the engine
                    // would route the model's "Here's a thinking process:" preamble into
                    // the content stream instead of the reasoning stream.
                    bool inThinking = false;
                    if (thinkingEnabled)
                    {
                        foreach (int tok in tokens)
                        {
                            if (tok == thinkId) inThinking = true;
                            else if (tok == endThinkId) inThinking = false;
                        }
                    }
                    int thinkingCount = 0;

                    for (int i = 0; i < sp.MaxNewTokens; i++)
                    {
                        ct.ThrowIfCancellationRequested();

                        // Force-inject </think> when the model has burned through its reasoning
                        // budget. Mirrors the CLI's --max-thinking-tokens path: the forced close
                        // token still routes through the boundary branch below and is fed back
                        // into forward(...) so the model continues from its post-think state.
                        int next;
                        if (inThinking && sp.MaxThinkingTokens > 0 && thinkingCount >= sp.MaxThinkingTokens
                            && thinkingEnabled && endThinkId > 0)
                        {
                            next = endThinkId;
                        }
                        else
                        {
                            next = sp.Temperature <= 0f
                                ? Sampler.Greedy(logits)
                                : Sampler.Sample(logits, sp, rng);
                        }

                        // Record every emitted/consumed token (stop tokens included) so the
                        // post-decode snapshot of _prevTokens reflects the full transcript.
                        // Issue #21: chat-continuation prompts for turn N+1 typically extend
                        // turn N's full transcript, not just turn N's prompt.
                        fullSeq.Add(next);

                        if (stopIds.Contains(next)) break;

                        // Counter update mirrors RunCommand.DecodeLoop: reset on each <think>
                        // open, otherwise increment whenever inThinking was true on entry to
                        // this iteration. That includes the </think> boundary token itself —
                        // so N content tokens of reasoning trips the force-close on iteration N+1.
                        if (thinkingEnabled && next == thinkId) thinkingCount = 0;
                        else if (inThinking) thinkingCount++;

                        // Reasoning boundary tokens: flip state, consume the token, do NOT emit.
                        // Both directions require the *opposite* state — a malformed second
                        // <think> mid-reasoning, or a stray </think> with no open block, falls
                        // through to the content path below rather than silently corrupting state.
                        if (thinkingEnabled && next == thinkId && !inThinking)
                        {
                            // Flush any pending text bytes before entering thinking mode.
                            var textTail = textDec.Flush();
                            if (textTail.Length > 0)
                                channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, textTail));
                            inThinking = true;
                            logits = _fwd.Forward(next, tokens.Length + i);
                            continue;
                        }
                        if (thinkingEnabled && next == endThinkId && inThinking)
                        {
                            // Flush thinking-decoder tail as a final Thinking chunk.
                            var thinkTail = thinkDec.Flush();
                            if (thinkTail.Length > 0)
                                channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, thinkTail));
                            inThinking = false;
                            logits = _fwd.Forward(next, tokens.Length + i);
                            continue;
                        }

                        var bytes = _tokenizer.DecodeBytes(next);
                        if (inThinking)
                        {
                            var chunk = thinkDec.Append(bytes);
                            if (chunk.Length > 0)
                                channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, chunk));
                        }
                        else
                        {
                            var chunk = textDec.Append(bytes);
                            if (chunk.Length > 0)
                                channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, chunk));
                        }

                        logits = _fwd.Forward(next, tokens.Length + i);
                    }

                    // End-of-loop: flush both decoders defensively (whichever was active).
                    var textFlush = textDec.Flush();
                    if (textFlush.Length > 0)
                        channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, textFlush));
                    var thinkFlush = thinkDec.Flush();
                    if (thinkFlush.Length > 0)
                        channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, thinkFlush));

                    // End-of-decode snapshot for the legacy path only — the canonical path
                    // already captured + paired _prevTokens during the split prefill above.
                    // ct.ThrowIfCancellationRequested earlier ensures we don't reach here on a
                    // cancelled token; skip capture in that case to avoid stale state.
                    if (!useCanonicalSnapshot && !ct.IsCancellationRequested)
                    {
                        _fwd.CaptureSnapshot();
                        _prevTokens = fullSeq.ToArray();
                    }

                    channel.Writer.TryComplete();
                }
                catch (Exception ex)
                {
                    channel.Writer.TryComplete(ex);
                }
            }, ct);

            try
            {
                await foreach (var chunk in channel.Reader.ReadAllAsync(ct).ConfigureAwait(false))
                    yield return chunk;
            }
            finally
            {
                // Issue #109: the background generation task drives the (non-thread-safe)
                // forward pass + shared KV cache. We must wait for it to fully stop before
                // releasing the gate — otherwise a cancelled/abandoned consumer (e.g. an
                // agentic client that disconnects mid-decode and immediately fires the next
                // request) would throw out of the await foreach above and release the gate
                // while genTask is still inside _fwd.Forward(...), letting the next request
                // reset/prefill the same cache concurrently and corrupt it (observed hang).
                // genTask routes all exceptions through the channel (surfaced by the foreach),
                // so awaiting it here only re-throws on the unobserved cancellation path,
                // which we swallow — the original cause already propagates from the foreach.
                try
                {
                    await genTask.ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                }
            }
        }
        finally
        {
            Interlocked.Decrement(ref _activeCount);
            _gate.Release();
        }
    }

    /// <summary>
    /// Synchronous dispose. Prefer <see cref="DisposeAsync"/> on async shutdown paths
    /// (e.g. a Generic Host's <c>StopAsync</c>); this overload blocks the calling thread
    /// while the in-flight generation drains.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // Stop any in-flight worker, then wait for it to release the gate before freeing
        // _fwd — see DrainNote on DisposeAsync.
        _shutdownCts.Cancel();
        // Don't release after acquiring: we're about to dispose the gate, and releasing it
        // would let a queued waiter whose cancellation hasn't yet propagated slip in and
        // touch _fwd as it's freed. Holding the permit guarantees exclusivity through teardown.
        bool drained = _gate.Wait(_disposeDrainTimeout);
        DisposeCore(drained);
    }

    /// <summary>
    /// Asynchronous dispose. Signals shutdown and awaits the in-flight generation worker's
    /// completion before freeing the engine-owned <see cref="IForwardPass"/>, so a host
    /// that cancels and disposes mid-decode gets a clean teardown instead of an access
    /// violation from the worker touching freed KV-cache memory (issue #132).
    /// </summary>
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _shutdownCts.Cancel();
        // DrainNote: the gate is held for the entire lifetime of a GenerateChunksAsync call
        // — across the background worker AND its drain (the finally awaits genTask before
        // releasing) — so re-acquiring it guarantees no worker is still inside _fwd.Forward /
        // the shared PagedKvCache. Cancelling _shutdownCts above makes the active worker exit
        // at its next per-token / per-prefill-chunk cancellation checkpoint, so this normally
        // returns in well under one forward. The timeout is only a backstop against a wedged
        // backend or a consumer that abandoned its enumerator without disposing it (which
        // would otherwise strand the gate forever).
        // See Dispose(): keep the permit held through teardown rather than releasing it.
        bool drained = await _gate.WaitAsync(_disposeDrainTimeout).ConfigureAwait(false);
        DisposeCore(drained);
    }

    /// <summary>
    /// Frees engine-owned resources. <paramref name="drained"/> is <c>false</c> only when the
    /// dispose drain timed out — i.e. a worker may still be live inside <c>_fwd</c> / the KV
    /// cache. In that case we deliberately leak the forward pass and owned handles rather than
    /// free them out from under the worker, which is the very access violation (#132) this fix
    /// exists to prevent: a leaked allocation on a shutdown that is already wedged is strictly
    /// better than a native crash.
    /// </summary>
    private void DisposeCore(bool drained)
    {
        if (!drained)
        {
            Console.Error.WriteLine(
                $"[InferenceEngine] dispose timed out after {_disposeDrainTimeout.TotalSeconds:0}s waiting " +
                "for the in-flight generation to drain; leaking the forward pass instead of freeing it under a " +
                "live worker. Ensure consumers dispose their generation enumerators and that the backend is responsive.");
            return;
        }
        _fwd.Dispose();
        foreach (var d in _owned)
            d.Dispose();
        _gate.Dispose();
        _shutdownCts.Dispose();
    }
}
