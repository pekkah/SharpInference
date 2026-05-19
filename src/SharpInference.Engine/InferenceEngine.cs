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

    private bool _disposed;

    public string ModelId { get; }

    /// <summary>Number of callers blocked waiting to acquire the generation gate.</summary>
    public int QueueDepth => _pendingCount;

    /// <summary>1 while a generation is in progress, 0 otherwise.</summary>
    public int ActiveRequests => _activeCount;

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

                    // Prefix cache check: reuse K/V for matching prefix, skip its prefill.
                    int prefixLen = FindCacheablePrefix(tokens);
                    if (prefixLen > 0)
                    {
                        // Soft-truncate: discard positions >= prefixLen, keep prefix K/V.
                        _fwd.TruncateTo(prefixLen);
                    }
                    else
                    {
                        _fwd.ResetCache();
                    }

                    // Prefill: process all prompt tokens (or just the suffix after the cached prefix).
                    ReadOnlySpan<float> logits;
                    int[] suffixTokens = prefixLen > 0 ? tokens[prefixLen..] : tokens;
                    if (suffixTokens.Length > 0)
                        logits = _fwd.Prefill(suffixTokens, prefixLen);
                    else
                        logits = _fwd.Forward(tokens[^1], tokens.Length - 1);

                    _prevTokens = tokens;

                    // Decode loop. Separate stateful UTF-8 decoders for the answer stream and
                    // the thinking stream so multi-byte characters in either stream reassemble
                    // independently — the two streams never share decoder state.
                    var textDec = new Utf8StreamDecoder();
                    var thinkDec = new Utf8StreamDecoder();
                    bool inThinking = false;
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
