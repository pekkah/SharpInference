using System.Runtime.CompilerServices;
using System.Threading.Channels;
using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Continuous batching inference engine: admits multiple concurrent requests and decodes them
/// in lock-step batches, amortizing weight reads across N sequences per decode step.
///
/// Flow:
///   1. Caller enqueues requests via <see cref="GenerateChunksAsync"/>.
///   2. Background batcher admits pending requests one at a time (prefilling each individually).
///   3. All active sequences are decoded together in a single <see cref="ForwardPass.BatchForwardMulti"/> call.
///   4. Sequences that hit EOS or max tokens are retired; their caches are returned to the pool.
///
/// Not supported for MoE models or when TurboQuant KV cache is enabled.
///
/// Reasoning support: when constructed with <c>thinkTokenId</c> / <c>endThinkTokenId</c>,
/// the engine splits each sequence's output into <see cref="GenerateChunkKind.Thinking"/>
/// and <see cref="GenerateChunkKind.Text"/> chunks. Per-sequence state tracks the current
/// reasoning mode and uses independent UTF-8 decoders so multi-byte chars in either stream
/// reassemble cleanly.
/// </summary>
public sealed class ContinuousBatchingEngine : IInferenceEngine, IDisposable
{
    private readonly ForwardPass _fwd;
    private readonly ITokenizer _tokenizer;
    private readonly int _maxBatchSize;
    private readonly int _thinkTokenId;
    private readonly int _endThinkTokenId;

    private readonly Channel<PendingRequest> _queue =
        Channel.CreateUnbounded<PendingRequest>(new UnboundedChannelOptions { SingleWriter = false, SingleReader = true });
    private readonly Task _batcherTask;
    private volatile bool _disposed;

    // Observability counters (updated via Interlocked).
    private int _pendingCount;
    private int _activeCount;

    private sealed class PendingRequest(string prompt, SamplingParams sp, CancellationToken ct, Channel<GenerateChunk> output)
    {
        public readonly string Prompt = prompt;
        public readonly SamplingParams Sp = sp;
        public readonly CancellationToken Ct = ct;
        public readonly Channel<GenerateChunk> Output = output;
    }

    private sealed class ActiveSeq
    {
        public required int CurrentToken;
        public required int Position;       // position at which CurrentToken will be decoded
        public required PagedKvCache Cache;
        public required SamplingParams Sp;
        public required Channel<GenerateChunk> Output;
        public required System.Collections.Immutable.ImmutableArray<int> StopIds;
        public required Random Rng;
        public required CancellationToken Ct;
        public int TokenCount;
        // Per-sequence stateful UTF-8 decoders: reassembles multi-byte characters
        // split across tokens (CJK, emoji, smart quotes). Independent decoders for
        // the answer stream and reasoning stream so neither pollutes the other.
        public Utf8StreamDecoder TextDec = new();
        public Utf8StreamDecoder ThinkDec = new();
        public bool InThinking;
        public int ThinkingCount;  // tokens accumulated in the current <think> block (resets on each <think> open)
    }

    /// <param name="fwd">Forward pass implementation. Owned externally — caller disposes.</param>
    /// <param name="tokenizer">Tokenizer matching the model vocabulary.</param>
    /// <param name="modelId">Human-readable model identifier returned in API responses.</param>
    /// <param name="maxBatchSize">Maximum concurrent decode sequences. Clamped to ≥ 1.</param>
    /// <param name="thinkTokenId">
    /// Token ID of the model's <c>&lt;think&gt;</c> marker, or <c>-1</c> if the model has no reasoning
    /// stream. When <c>-1</c>, all chunks are emitted as <see cref="GenerateChunkKind.Text"/>.
    /// </param>
    /// <param name="endThinkTokenId">
    /// Token ID of the model's <c>&lt;/think&gt;</c> marker, or <c>-1</c>. Must be paired with
    /// a non-negative <paramref name="thinkTokenId"/> to enable reasoning-stream splitting.
    /// </param>
    public ContinuousBatchingEngine(
        ForwardPass fwd,
        ITokenizer tokenizer,
        string modelId,
        int maxBatchSize = 8,
        int thinkTokenId = -1,
        int endThinkTokenId = -1)
    {
        _fwd = fwd;
        _tokenizer = tokenizer;
        ModelId = modelId;
        _maxBatchSize = Math.Max(1, maxBatchSize);
        _thinkTokenId = thinkTokenId;
        _endThinkTokenId = endThinkTokenId;
        _batcherTask = Task.Run(BatcherLoop);
    }

    public string ModelId { get; }

    /// <summary>Number of requests queued but not yet being generated.</summary>
    public int QueueDepth => _pendingCount;

    /// <summary>Number of requests currently in the active decode batch.</summary>
    public int ActiveRequests => _activeCount;

    /// <summary>
    /// Continuous batching does not (yet) share KV state across requests — each admitted
    /// sequence allocates its own <see cref="PagedKvCache"/> — so prefix caching is always
    /// off here.
    /// </summary>
    public bool PrefixCacheEnabled => false;

    /// <inheritdoc/>
    public long PrefillTokensReused => 0;

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
    /// <remarks>
    /// <paramref name="canonicalHistoryPrefix"/> is accepted for interface parity but
    /// ignored on this engine — continuous batching gives each admitted sequence its
    /// own freshly allocated <see cref="PagedKvCache"/> and never reuses state across
    /// requests, so a snapshot hint has nothing to attach to (issue #102).
    /// </remarks>
    public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default,
        string? canonicalHistoryPrefix = null)
    {
        _ = canonicalHistoryPrefix; // intentionally ignored; see XML remarks
        var channel = Channel.CreateUnbounded<GenerateChunk>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

        Interlocked.Increment(ref _pendingCount);
        await _queue.Writer.WriteAsync(new PendingRequest(prompt, sp, ct, channel), ct)
            .ConfigureAwait(false);

        await foreach (var chunk in channel.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            yield return chunk;
    }

    private async Task BatcherLoop()
    {
        var active = new List<ActiveSeq>(_maxBatchSize);
        var tokensBuf = new int[_maxBatchSize];
        var posBuf = new int[_maxBatchSize];
        var cacheBuf = new PagedKvCache[_maxBatchSize];
        bool thinkingEnabled = _thinkTokenId >= 0 && _endThinkTokenId >= 0;

        while (!_disposed)
        {
            // Admit pending requests into available batch slots
            while (active.Count < _maxBatchSize && _queue.Reader.TryRead(out var req))
            {
                Interlocked.Decrement(ref _pendingCount);
                try
                {
                    AdmitRequest(req, active, thinkingEnabled);
                }
                catch (Exception ex)
                {
                    req.Output.Writer.TryComplete(ex);
                }
            }

            if (active.Count == 0)
            {
                // No active work — wait for a new request
                try
                {
                    bool hasMore = await _queue.Reader.WaitToReadAsync().ConfigureAwait(false);
                    if (!hasMore) break;
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                continue;
            }

            // Build batched inputs
            int n = active.Count;
            for (int i = 0; i < n; i++)
            {
                tokensBuf[i] = active[i].CurrentToken;
                posBuf[i] = active[i].Position;
                cacheBuf[i] = active[i].Cache;
            }

            // Batched decode step (shares weight reads across N sequences)
            float[][] logitsBatch = _fwd.BatchForwardMulti(
                tokensBuf[..n], posBuf[..n], cacheBuf[..n]);

            // Process results in reverse order so RemoveAt indices stay valid
            for (int i = n - 1; i >= 0; i--)
            {
                var seq = active[i];

                // Force-inject </think> on budget overrun. Mirrors InferenceEngine /
                // RunCommand.DecodeLoop: the forced close token threads through the
                // boundary branch below and is fed back as the next CurrentToken so the
                // model continues from its post-think state on the next batched step.
                int next;
                if (thinkingEnabled && seq.InThinking && seq.Sp.MaxThinkingTokens > 0
                    && seq.ThinkingCount >= seq.Sp.MaxThinkingTokens && _endThinkTokenId > 0)
                {
                    next = _endThinkTokenId;
                }
                else
                {
                    next = seq.Sp.Temperature <= 0f
                        ? Sampler.Greedy(logitsBatch[i])
                        : Sampler.Sample(logitsBatch[i], seq.Sp, seq.Rng);
                }

                bool done = seq.StopIds.Contains(next)
                    || seq.TokenCount >= seq.Sp.MaxNewTokens
                    || seq.Ct.IsCancellationRequested;

                if (done)
                {
                    FlushAndComplete(seq);
                    seq.Cache.Dispose();
                    active.RemoveAt(i);
                    Interlocked.Decrement(ref _activeCount);
                }
                else
                {
                    // Counter update mirrors RunCommand.DecodeLoop: reset on each <think>
                    // open, otherwise increment whenever seq.InThinking was true on entry to
                    // this step. That includes the </think> boundary token — so N reasoning
                    // tokens trip the force-close on step N+1.
                    if (thinkingEnabled && next == _thinkTokenId) seq.ThinkingCount = 0;
                    else if (seq.InThinking) seq.ThinkingCount++;

                    // Reasoning boundary tokens: flip state, consume the token, do NOT emit content.
                    // Each direction requires the opposite state, so malformed double <think>
                    // or orphan </think> falls through to the content path below.
                    if (thinkingEnabled && next == _thinkTokenId && !seq.InThinking)
                    {
                        var textTail = seq.TextDec.Flush();
                        if (textTail.Length > 0)
                            seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, textTail));
                        seq.InThinking = true;
                    }
                    else if (thinkingEnabled && next == _endThinkTokenId && seq.InThinking)
                    {
                        var thinkTail = seq.ThinkDec.Flush();
                        if (thinkTail.Length > 0)
                            seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, thinkTail));
                        seq.InThinking = false;
                    }
                    else
                    {
                        var bytes = _tokenizer.DecodeBytes(next);
                        if (seq.InThinking)
                        {
                            var chunk = seq.ThinkDec.Append(bytes);
                            if (chunk.Length > 0)
                                seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, chunk));
                        }
                        else
                        {
                            var chunk = seq.TextDec.Append(bytes);
                            if (chunk.Length > 0)
                                seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, chunk));
                        }
                    }

                    seq.CurrentToken = next;
                    seq.Position++;
                    seq.TokenCount++;
                }
            }
        }

        // Drain: complete any remaining active sequences
        foreach (var seq in active)
        {
            FlushAndComplete(seq);
            seq.Cache.Dispose();
            Interlocked.Decrement(ref _activeCount);
        }
        active.Clear();
    }

    private static void FlushAndComplete(ActiveSeq seq)
    {
        var textTail = seq.TextDec.Flush();
        if (textTail.Length > 0)
            seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, textTail));
        var thinkTail = seq.ThinkDec.Flush();
        if (thinkTail.Length > 0)
            seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, thinkTail));
        seq.Output.Writer.TryComplete();
    }

    private void AdmitRequest(PendingRequest req, List<ActiveSeq> active, bool thinkingEnabled)
    {
        if (req.Ct.IsCancellationRequested)
        {
            req.Output.Writer.TryComplete();
            return;
        }

        var tokens = _tokenizer.Encode(req.Prompt).ToArray();
        if (tokens.Length == 0)
        {
            req.Output.Writer.TryComplete();
            return;
        }

        var cache = _fwd.CreateCache();
        ReadOnlySpan<float> logits = _fwd.PrefillWithCache(tokens, cache);

        // Stop on ANY end-of-generation token, not just the configured EOS — matches the
        // single-user InferenceEngine path. A model with an alternate end token (e.g. Gemma's
        // <eos>, distinct from its <turn|> EOS) would otherwise decode it as text and run on.
        System.Collections.Immutable.ImmutableArray<int> stopIds =
            req.Sp.StopTokenIds is { } userStops ? [.. userStops] : _tokenizer.EogTokenIds;
        var rng = new Random();

        int firstToken = req.Sp.Temperature <= 0f
            ? Sampler.Greedy(logits)
            : Sampler.Sample(logits, req.Sp, rng);

        if (stopIds.Contains(firstToken))
        {
            req.Output.Writer.TryComplete();
            cache.Dispose();
            return;
        }

        // Seed InThinking from the prompt itself (issue #92). Qwen3.6 and other
        // reasoning models append a bare `<think>` token to the generation prompt
        // via their chat template, so the model is already inside a reasoning
        // block before the first sampled token.
        bool promptInThinking = false;
        if (thinkingEnabled)
        {
            foreach (int tok in tokens)
            {
                if (tok == _thinkTokenId) promptInThinking = true;
                else if (tok == _endThinkTokenId) promptInThinking = false;
            }
        }

        var seq = new ActiveSeq
        {
            CurrentToken = firstToken,
            Position = tokens.Length,
            Cache = cache,
            Sp = req.Sp,
            Output = req.Output,
            StopIds = stopIds,
            Rng = rng,
            Ct = req.Ct,
            TokenCount = 1,
            InThinking = promptInThinking,
        };

        // Route the first sampled token through the same state machine as the decode loop.
        // A `<think>` here without an open block opens one; a `</think>` while in a prompt-
        // seeded block closes it. A stray `</think>` outside a block falls through to
        // content (same fall-through as decode).
        if (thinkingEnabled && firstToken == _thinkTokenId && !seq.InThinking)
        {
            seq.InThinking = true;
        }
        else if (thinkingEnabled && firstToken == _endThinkTokenId && seq.InThinking)
        {
            seq.InThinking = false;
        }
        else
        {
            var bytes = _tokenizer.DecodeBytes(firstToken);
            if (seq.InThinking)
            {
                var firstChunk = seq.ThinkDec.Append(bytes);
                if (firstChunk.Length > 0)
                    req.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, firstChunk));
            }
            else
            {
                var firstChunk = seq.TextDec.Append(bytes);
                if (firstChunk.Length > 0)
                    req.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, firstChunk));
            }
        }

        active.Add(seq);
        Interlocked.Increment(ref _activeCount);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _queue.Writer.TryComplete();
    }
}
