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
public sealed class InferenceEngine : IInferenceEngine, IDisposable
{
    private readonly IForwardPass _fwd;
    private readonly ITokenizer _tokenizer;
    private readonly IDisposable[] _owned;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly int _thinkTokenId;
    private readonly int _endThinkTokenId;

    // Prefix caching state (guarded by _gate — only accessed during generation).
    private int[]? _prevTokens;

    // Observability counters (updated via Interlocked).
    private int _pendingCount;
    private int _activeCount;
    private long _prefillTokensReused;

    private bool _disposed;

    public string ModelId { get; }

    /// <summary>Number of callers blocked waiting to acquire the generation gate.</summary>
    public int QueueDepth => _pendingCount;

    /// <summary>1 while a generation is in progress, 0 otherwise.</summary>
    public int ActiveRequests => _activeCount;

    /// <inheritdoc/>
    public bool PrefixCacheEnabled => _fwd.SupportsPartialRewind;

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
            Console.Error.WriteLine(
                $"[InferenceEngine] prefix cache disabled — {fwd.GetType().Name} reports SupportsPartialRewind == false. Multi-turn requests will re-prefill the full prompt.");
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
        [EnumeratorCancellation] CancellationToken ct = default)
    {
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
                    var stopIds = sp.StopTokenIds ?? [_tokenizer.EosTokenId];

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

                    // --spec-draft-n-max parity with llama.cpp. Sharpi's MTP decoder
                    // is sequential N=1 today; issue #30 (Phase-7 batched verify + per-token
                    // GDN snapshot ring) is what unlocks N>1. Reject larger values up front
                    // so users don't silently get the same throughput they'd get without
                    // the flag.
                    if (useMtp && sp.SpecDraftNMax > 1)
                        throw new InvalidOperationException(
                            $"SamplingParams.SpecDraftNMax={sp.SpecDraftNMax} is not yet supported. " +
                            "Sharpi's MTP path is sequential N=1; issue #30 (Phase-7 batched verify) " +
                            "lifts this. Pass --spec-draft-n-max 1 (or omit) until then.");

                    // Prefix cache decision: two branches.
                    //   (a) Rewindable attention pass — existing FindCacheablePrefix path,
                    //       page-aligned KV reuse.
                    //   (b) Non-rewindable GDN hybrid with a snapshot — issue #21: try
                    //       exact-match against the most-recent snapshot length so the
                    //       held GDN recurrent state can be restored.
                    // If neither branch hits, fall back to a full ResetCache + Prefill.
                    int prefixLen = 0;
                    if (!useMtp && _fwd.SupportsPartialRewind)
                    {
                        int candidate = FindCacheablePrefix(tokens);
                        if (candidate > 0)
                        {
                            _fwd.TruncateTo(candidate);
                            prefixLen = candidate;
                        }
                    }
                    else if (!useMtp && _fwd.SnapshotLength > 0 && _prevTokens != null)
                    {
                        int snapLen = _fwd.SnapshotLength;
                        // snapLen must be a strict prefix of the new prompt (need at least
                        // one suffix token to drive the decoder) AND a prefix of the
                        // previously decoded sequence (so the snapshot matches the state
                        // at that token position).
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
                    ReadOnlySpan<float> logits;
                    int[] suffixTokens = prefixLen > 0 ? tokens[prefixLen..] : tokens;
                    if (suffixTokens.Length > 0)
                        logits = _fwd.Prefill(suffixTokens, prefixLen);
                    else
                        logits = _fwd.Forward(tokens[^1], tokens.Length - 1);

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
                            _fwd.PrefillMtp(suffixTokens, prefixLen);

                        var textDecMtp = new Utf8StreamDecoder();

                        mtpDec.Decode(sp.MaxNewTokens, stopIds.AsSpan(), tok =>
                        {
                            fullSeq.Add(tok);
                            var bytes = _tokenizer.DecodeBytes(tok);
                            var chunk = textDecMtp.Append(bytes);
                            if (chunk.Length > 0)
                                channel.Writer.TryWrite(
                                    new GenerateChunk(GenerateChunkKind.Text, chunk));
                        }, pMin: sp.SpecDraftPMin, ct: ct);

                        var textFlushMtp = textDecMtp.Flush();
                        if (textFlushMtp.Length > 0)
                            channel.Writer.TryWrite(
                                new GenerateChunk(GenerateChunkKind.Text, textFlushMtp));

                        if (Environment.GetEnvironmentVariable("SHARPI_TRACE_MTP") == "1"
                            && mtpDec.TotalDraftsEmitted > 0)
                        {
                            Console.Error.WriteLine(
                                $"[InferenceEngine] MTP: {mtpDec.TotalDraftsAccepted}/{mtpDec.TotalDraftsEmitted} " +
                                $"drafts accepted ({mtpDec.AcceptanceRate:P1})");
                        }

                        if (!ct.IsCancellationRequested)
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

                    // Capture end-of-decode snapshot (issue #21) for the GDN-hybrid passes
                    // and record the full sequence for prefix matching on the next turn.
                    // ct.ThrowIfCancellationRequested above ensures we don't reach here on
                    // a cancelled token — skip capture in that case to avoid stale state.
                    if (!ct.IsCancellationRequested)
                    {
                        _fwd.CaptureSnapshot();
                        // Write _prevTokens AFTER capturing the snapshot so a subsequent
                        // turn's prefix check sees the matching state.
                        _prevTokens = fullSeq.ToArray();
                    }

                    channel.Writer.TryComplete();
                }
                catch (Exception ex)
                {
                    channel.Writer.TryComplete(ex);
                }
            }, ct);

            await foreach (var chunk in channel.Reader.ReadAllAsync(ct).ConfigureAwait(false))
                yield return chunk;

            await genTask.ConfigureAwait(false); // re-throw any generation exception
        }
        finally
        {
            Interlocked.Decrement(ref _activeCount);
            _gate.Release();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _fwd.Dispose();
        foreach (var d in _owned)
            d.Dispose();
        _gate.Dispose();
    }
}
