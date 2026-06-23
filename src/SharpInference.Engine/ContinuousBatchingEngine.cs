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
///   2. Background batcher admits pending requests under a KV token budget (issue #183 Gap 3:
///      admission backpressure — a burst of long prompts queues instead of exhausting memory).
///   3. Admitted prompts prefill in chunks interleaved with decode steps (issue #183 Gap 1:
///      a long prompt no longer stalls every active sequence). When several prompts are
///      prefilling at once, their chunks run as one packed forward pass
///      (<see cref="ForwardPass.PrefillPackedMulti"/>, issue #183 Gap 2) so weight reads are
///      amortized across prompts exactly like decode batching.
///   4. All active sequences are decoded together in a single <see cref="ForwardPass.BatchForwardMulti"/> call.
///   5. Sequences that hit EOS or max tokens are retired; their caches are returned to the pool.
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
    private readonly IBatchedForwardPass _fwd;
    private readonly ITokenizer _tokenizer;
    private readonly int _maxBatchSize;
    private readonly int _thinkTokenId;
    private readonly int _endThinkTokenId;
    private readonly bool _thinkingEnabled;

    // Issue #183 Gap 1: tokens of prompt prefilled per batcher iteration. Between chunks
    // every active sequence advances one decode step. 0 disables chunking (a prompt
    // prefills in one blocking call — the pre-#183 behavior). SnapKV also disables
    // chunking: its prefill eviction only runs on a fresh full-prompt prefill.
    private readonly int _prefillChunkTokens;
    private readonly bool _chunkingEnabled;

    // Issue #183 Gap 3: max total KV tokens committed across admitted sequences. Each
    // sequence reserves promptTokens + MaxNewTokens (clamped to MaxSeqLen) at admission
    // and releases the reservation when it retires. long.MaxValue = unlimited.
    private readonly long _kvTokenBudget;
    private long _committedTokens; // batcher-thread only

    private readonly Channel<PendingRequest> _queue =
        Channel.CreateUnbounded<PendingRequest>(new UnboundedChannelOptions { SingleWriter = false, SingleReader = true });
    private readonly Task _batcherTask;
    private volatile bool _disposed;

    // Observability counters (updated via Interlocked).
    private int _pendingCount;
    private int _activeCount;

    // Set once (Interlocked) the first time a constraint-bearing request is seen, so the
    // tool-grammar-ignored warning (issue #374) is emitted at most once per engine.
    private int _warnedConstraintIgnored;

    private sealed class PendingRequest(string prompt, SamplingParams sp, CancellationToken ct, Channel<GenerateChunk> output)
    {
        public readonly string Prompt = prompt;
        public readonly SamplingParams Sp = sp;
        public readonly CancellationToken Ct = ct;
        public readonly Channel<GenerateChunk> Output = output;
        public int[]? Tokens; // memoized tokenization (admission may retry under backpressure)
    }

    /// <summary>A sequence whose prompt is being prefilled chunk-by-chunk.</summary>
    private sealed class PrefillingSeq
    {
        public required PendingRequest Req;
        public required int[] Tokens;
        public required ISequenceKvCache Cache;
        public required long ProjectedTokens; // KV budget reservation, released on retire
        public int Consumed;                  // prompt tokens prefilled so far
    }

    private sealed class ActiveSeq
    {
        public required int CurrentToken;
        public required int Position;       // position at which CurrentToken will be decoded
        public required ISequenceKvCache Cache;
        public required SamplingParams Sp;
        public required Channel<GenerateChunk> Output;
        public required System.Collections.Immutable.ImmutableArray<int> StopIds;
        public required Random Rng;
        public required CancellationToken Ct;
        public required long ProjectedTokens;
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
    /// <param name="prefillChunkTokens">
    /// Prompt tokens prefilled per batcher iteration (issue #183 Gap 1); active sequences
    /// decode one step between chunks. <c>0</c> = prefill each prompt in one blocking call.
    /// <c>-1</c> = auto (issue #189): <c>64</c> when the forward pass's dequant-once weight
    /// cache covers the model — small chunks are then nearly free, so a low decode-stall
    /// chunk is safe — otherwise <c>256</c> to keep the per-chunk re-dequant amortized.
    /// </param>
    /// <param name="kvBudgetBytes">
    /// KV-memory budget gating admission (issue #183 Gap 3). <c>0</c> = auto (half of
    /// available system RAM), negative = unlimited, positive = explicit byte budget.
    /// </param>
    public ContinuousBatchingEngine(
        IBatchedForwardPass fwd,
        ITokenizer tokenizer,
        string modelId,
        int maxBatchSize = 8,
        int thinkTokenId = -1,
        int endThinkTokenId = -1,
        int prefillChunkTokens = -1,
        long kvBudgetBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(fwd);
        ArgumentNullException.ThrowIfNull(tokenizer);
        _fwd = fwd;
        _tokenizer = tokenizer;
        ModelId = modelId;
        _maxBatchSize = Math.Max(1, maxBatchSize);
        _thinkTokenId = thinkTokenId;
        _endThinkTokenId = endThinkTokenId;
        _thinkingEnabled = thinkTokenId >= 0 && endThinkTokenId >= 0;

        // -1 = auto (issue #189): a small chunk minimizes decode stall but normally collapses
        // prefill throughput (per-chunk weight re-dequant); pick it only when the dequant-once
        // cache covers the model so small chunks re-pay no dequant. Otherwise keep 256.
        _prefillChunkTokens = prefillChunkTokens >= 0
            ? prefillChunkTokens
            : (fwd.PrefillDequantCacheActive ? 64 : 256);
        _chunkingEnabled = _prefillChunkTokens > 0 && !fwd.SnapKvEnabled;

        long budgetBytes = kvBudgetBytes switch
        {
            < 0 => long.MaxValue,
            0 => GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 2,
            _ => kvBudgetBytes,
        };
        _kvTokenBudget = budgetBytes == long.MaxValue
            ? long.MaxValue
            : Math.Max(1, budgetBytes / Math.Max(1, fwd.KvBytesPerToken));

        _batcherTask = Task.Run(BatcherLoop);
    }

    public string ModelId { get; }

    /// <summary>Number of requests queued but not yet being generated.</summary>
    public int QueueDepth => _pendingCount;

    /// <summary>Number of requests currently being prefilled or decoded.</summary>
    public int ActiveRequests => _activeCount;

    /// <summary>Total KV token budget gating admission (issue #183 Gap 3).</summary>
    public long KvTokenBudget => _kvTokenBudget;

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

        // Grammar-constrained decoding (issue #374) is not wired into the batched sampler — each
        // sequence is sampled directly from its logits with no per-token mask/advance. Rather than
        // silently drop the constraint (the request would generate unconstrained tool arguments with
        // no signal), warn once so an operator who set SHARPI_TOOL_GRAMMAR alongside SHARPI_MAX_BATCH
        // knows the two don't yet compose. Single-user InferenceEngine honors the constraint.
        if (sp.Constraint is not null && Interlocked.Exchange(ref _warnedConstraintIgnored, 1) == 0)
            Console.Error.WriteLine(
                "[ContinuousBatchingEngine] tool-grammar constraint is ignored under continuous " +
                "batching (SHARPI_MAX_BATCH); tool-call arguments will be generated unconstrained. " +
                "Run without batching to use SHARPI_TOOL_GRAMMAR (issue #374).");

        var channel = Channel.CreateUnbounded<GenerateChunk>(
            new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

        Interlocked.Increment(ref _pendingCount);
        try
        {
            await _queue.Writer.WriteAsync(new PendingRequest(prompt, sp, ct, channel), ct)
                .ConfigureAwait(false);
        }
        catch
        {
            // Disposed concurrently (writer completed) or caller-cancelled before the
            // write landed — undo the counter so QueueDepth doesn't drift.
            Interlocked.Decrement(ref _pendingCount);
            throw;
        }

        await foreach (var chunk in channel.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            yield return chunk;
    }

    private async Task BatcherLoop()
    {
        var pending = new Queue<PendingRequest>();
        var prefilling = new List<PrefillingSeq>();
        var active = new List<ActiveSeq>(_maxBatchSize);
        var tokensBuf = new int[_maxBatchSize];
        var posBuf = new int[_maxBatchSize];
        var cacheBuf = new ISequenceKvCache[_maxBatchSize];

        // An exception the per-request handlers didn't isolate (e.g. a backend failure
        // inside BatchForwardMulti) must not kill the batcher silently: without the
        // catch + finally-drain below, every in-flight caller would hang forever on a
        // never-completed channel and the per-sequence caches would leak.
        Exception? fatal = null;
        try
        {
        while (!_disposed)
        {
            // Issue #302: rebind the backend's thread-affine context (CUDA) before any forward
            // work this iteration. The loop's `await WaitToReadAsync().ConfigureAwait(false)` on
            // the idle path can resume the continuation on a different thread-pool thread than the
            // one that ran the previous step, so a single bind before the loop wouldn't hold. The
            // call is a no-op on CPU/Vulkan and free after the first call per thread.
            _fwd.BindToCurrentThread();

            // Pull everything queued so far into the local pending queue. Channels have
            // no peek, and admission backpressure needs to inspect the head request's
            // size without consuming it — hence the local FIFO.
            while (_queue.Reader.TryRead(out var queued))
                pending.Enqueue(queued);

            AdmitPending(pending, prefilling, active);

            if (active.Count == 0 && prefilling.Count == 0)
            {
                // No active work — wait for a new request. (Admission always starts the
                // head request when nothing is running, so pending is empty here.)
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

            // Advance prefilling prompts by one chunk, then give every active sequence
            // one decode step — the interleave that keeps decode from stalling.
            if (prefilling.Count > 0)
                RunPrefillStep(prefilling, active);

            if (active.Count == 0)
                continue;

            // Build batched inputs
            int n = active.Count;
            for (int i = 0; i < n; i++)
            {
                tokensBuf[i] = active[i].CurrentToken;
                posBuf[i] = active[i].Position;
                cacheBuf[i] = active[i].Cache;
            }

            // Batched decode step (shares weight reads across N sequences). When EVERY active
            // sequence is plain greedy this step (temp ≤ 0, not force-closing </think>), take the
            // on-device argmax tail (#205/#206): it returns just the per-seq (token, logit) and
            // skips the full N×vocab logits D2H + host split. A single sampled or force-closing
            // seq reverts the whole step to the full-logits path (the argmax buffer can't sample).
            bool allGreedy = _fwd.SupportsBatchedGpuArgmax;
            for (int i = 0; i < n && allGreedy; i++)
            {
                var s = active[i];
                bool forcedClose = _thinkingEnabled && s.InThinking && s.Sp.MaxThinkingTokens > 0
                                   && s.ThinkingCount >= s.Sp.MaxThinkingTokens && _endThinkTokenId > 0;
                if (s.Sp.Temperature > 0f || forcedClose) allGreedy = false;
            }

            float[][]? logitsBatch = null;
            (int Token, float Logit)[]? argmaxBatch = null;
            if (allGreedy)
                argmaxBatch = _fwd.BatchForwardMultiArgmax(tokensBuf[..n], posBuf[..n], cacheBuf[..n]);
            else
                logitsBatch = _fwd.BatchForwardMulti(tokensBuf[..n], posBuf[..n], cacheBuf[..n]);

            // Process results in reverse order so RemoveAt indices stay valid
            for (int i = n - 1; i >= 0; i--)
            {
                var seq = active[i];

                // Force-inject </think> on budget overrun. Mirrors InferenceEngine /
                // RunCommand.DecodeLoop: the forced close token threads through the
                // boundary branch below and is fed back as the next CurrentToken so the
                // model continues from its post-think state on the next batched step.
                int next;
                if (argmaxBatch is not null)
                {
                    // All-greedy fast path: the on-device argmax already picked each token (the
                    // gate above excluded any sampled / force-closing seq from this branch).
                    next = argmaxBatch[i].Token;
                }
                else if (_thinkingEnabled && seq.InThinking && seq.Sp.MaxThinkingTokens > 0
                    && seq.ThinkingCount >= seq.Sp.MaxThinkingTokens && _endThinkTokenId > 0)
                {
                    next = _endThinkTokenId;
                }
                else
                {
                    next = seq.Sp.Temperature <= 0f
                        ? Sampler.Greedy(logitsBatch![i])
                        : Sampler.Sample(logitsBatch![i], seq.Sp, seq.Rng);
                }

                bool done = seq.StopIds.Contains(next)
                    || seq.TokenCount >= seq.Sp.MaxNewTokens
                    || seq.Ct.IsCancellationRequested;

                if (done)
                {
                    FlushAndComplete(seq);
                    RetireSeq(seq);
                    active.RemoveAt(i);
                }
                else
                {
                    // Counter update mirrors RunCommand.DecodeLoop: reset on each <think>
                    // open, otherwise increment whenever seq.InThinking was true on entry to
                    // this step. That includes the </think> boundary token — so N reasoning
                    // tokens trip the force-close on step N+1.
                    if (_thinkingEnabled && next == _thinkTokenId) seq.ThinkingCount = 0;
                    else if (seq.InThinking) seq.ThinkingCount++;

                    // Reasoning boundary tokens are ALWAYS consumed and never emitted. State flips
                    // only on a valid transition, so a malformed double-open or an orphan close is
                    // swallowed rather than leaking its literal marker as text — e.g. a bare Gemma 4
                    // <channel|> close with no preceding open (issue #304).
                    if (_thinkingEnabled && next == _thinkTokenId)
                    {
                        if (!seq.InThinking)
                        {
                            var textTail = seq.TextDec.Flush();
                            if (textTail.Length > 0)
                                seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, textTail));
                            seq.InThinking = true;
                        }
                    }
                    else if (_thinkingEnabled && next == _endThinkTokenId)
                    {
                        if (seq.InThinking)
                        {
                            var thinkTail = seq.ThinkDec.Flush();
                            if (thinkTail.Length > 0)
                                seq.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, thinkTail));
                            seq.InThinking = false;
                        }
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
        }
        catch (Exception ex)
        {
            fatal = ex;
        }
        finally
        {
            // Drain: complete everything still in flight — with the fatal exception when
            // the loop died, so callers observe the failure instead of hanging. Pull any
            // requests still sitting in the channel first (written between our last
            // TryRead pass and the writer completing).
            while (_queue.Reader.TryRead(out var stranded))
                pending.Enqueue(stranded);
            foreach (var req in pending)
            {
                Interlocked.Decrement(ref _pendingCount);
                req.Output.Writer.TryComplete(fatal);
            }
            pending.Clear();
            foreach (var p in prefilling)
            {
                p.Req.Output.Writer.TryComplete(fatal);
                p.Cache.Dispose();
                Interlocked.Decrement(ref _activeCount);
            }
            prefilling.Clear();
            foreach (var seq in active)
            {
                if (fatal is null) FlushAndComplete(seq);
                else seq.Output.Writer.TryComplete(fatal);
                seq.Cache.Dispose();
                Interlocked.Decrement(ref _activeCount);
            }
            active.Clear();
        }
    }

    /// <summary>
    /// Moves requests from the local pending queue into the prefilling set while batch
    /// slots are free and the KV token budget allows (issue #183 Gap 3). FIFO: a
    /// too-large head request blocks the queue until running sequences retire — except
    /// when nothing is running, where it is always admitted so a single oversized
    /// request can't deadlock the engine (it then fails in prefill if it truly can't fit).
    /// </summary>
    private void AdmitPending(Queue<PendingRequest> pending, List<PrefillingSeq> prefilling, List<ActiveSeq> active)
    {
        while (pending.Count > 0 && active.Count + prefilling.Count < _maxBatchSize)
        {
            var req = pending.Peek();
            if (req.Ct.IsCancellationRequested)
            {
                pending.Dequeue();
                Interlocked.Decrement(ref _pendingCount);
                req.Output.Writer.TryComplete();
                continue;
            }

            int[] tokens;
            try
            {
                tokens = req.Tokens ??= _tokenizer.Encode(req.Prompt).ToArray();
            }
            catch (Exception ex)
            {
                pending.Dequeue();
                Interlocked.Decrement(ref _pendingCount);
                req.Output.Writer.TryComplete(ex);
                continue;
            }
            if (tokens.Length == 0)
            {
                pending.Dequeue();
                Interlocked.Decrement(ref _pendingCount);
                req.Output.Writer.TryComplete();
                continue;
            }

            long projected = Math.Min(
                tokens.Length + Math.Max(0L, req.Sp.MaxNewTokens), _fwd.MaxSeqLen);
            if (_committedTokens + projected > _kvTokenBudget
                && (active.Count > 0 || prefilling.Count > 0))
            {
                break; // backpressure: re-evaluated next iteration as sequences retire
            }

            pending.Dequeue();
            Interlocked.Decrement(ref _pendingCount);

            ISequenceKvCache cache;
            try
            {
                cache = _fwd.CreateCache();
            }
            catch (Exception ex)
            {
                req.Output.Writer.TryComplete(ex);
                continue;
            }

            _committedTokens += projected;
            // Surface the prompt-token count out-of-band so endpoints can report
            // usage.prompt_tokens / input_tokens without re-tokenizing (issue #150).
            // Emitted at admission, before any text/thinking chunk for this sequence.
            req.Output.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Usage, "", tokens.Length));
            prefilling.Add(new PrefillingSeq
            {
                Req = req,
                Tokens = tokens,
                Cache = cache,
                ProjectedTokens = projected,
            });
            Interlocked.Increment(ref _activeCount);
        }
    }

    /// <summary>
    /// Advances prefilling prompts by one chunk budget (issue #183 Gap 1). With several
    /// prompts in flight the chunk budget is split across them and run as one packed
    /// forward pass (Gap 2) so weight reads amortize across prompts. Prompts whose final
    /// chunk completes are sampled and promoted to the active decode set.
    /// </summary>
    private void RunPrefillStep(List<PrefillingSeq> prefilling, List<ActiveSeq> active)
    {
        // Cancellation sweep between chunks — a benefit chunking adds for free:
        // a cancelled long-prompt request stops mid-prefill instead of running to the end.
        for (int i = prefilling.Count - 1; i >= 0; i--)
        {
            var p = prefilling[i];
            if (p.Req.Ct.IsCancellationRequested)
            {
                p.Req.Output.Writer.TryComplete();
                DropPrefilling(prefilling, i);
            }
        }
        if (prefilling.Count == 0) return;

        if (!_chunkingEnabled)
        {
            // Unchunked path (prefillChunkTokens == 0, or SnapKV active — its prefill
            // eviction only runs on a whole-prompt startPos==0 prefill). One prompt per
            // iteration, blocking, exactly the pre-#183 behavior.
            var p = prefilling[0];
            float[] logits;
            try
            {
                logits = _fwd.PrefillWithCache(p.Tokens, p.Cache).ToArray();
            }
            catch (Exception ex)
            {
                p.Req.Output.Writer.TryComplete(ex);
                DropPrefilling(prefilling, 0);
                return;
            }
            prefilling.RemoveAt(0);
            ActivateSeq(p, logits, active);
            return;
        }

        // Split this iteration's chunk budget across all prefilling prompts (≥1 token
        // each). The budget bounds the decode stall per iteration regardless of how
        // many prompts are prefilling, because the packed pass amortizes weights.
        int sCount = prefilling.Count;
        int perSeq = Math.Max(1, _prefillChunkTokens / sCount);

        var chunks = new ReadOnlyMemory<int>[sCount];
        var startPos = new int[sCount];
        var caches = new ISequenceKvCache[sCount];
        var wantLogits = new bool[sCount];
        var takes = new int[sCount];
        for (int s = 0; s < sCount; s++)
        {
            var p = prefilling[s];
            int take = Math.Min(p.Tokens.Length - p.Consumed, perSeq);
            chunks[s] = p.Tokens.AsMemory(p.Consumed, take);
            startPos[s] = p.Consumed;
            caches[s] = p.Cache;
            wantLogits[s] = p.Consumed + take == p.Tokens.Length;
            takes[s] = take;
        }

        float[]?[] logitsPerSeq;
        try
        {
            if (sCount == 1)
            {
                var p = prefilling[0];
                var segment = new ArraySegment<int>(p.Tokens, p.Consumed, takes[0]);
                var logits = _fwd.PrefillWithCache(segment, p.Cache, startPos: p.Consumed);
                logitsPerSeq = [wantLogits[0] ? logits.ToArray() : null];
            }
            else
            {
                logitsPerSeq = _fwd.PrefillPackedMulti(chunks, startPos, caches, wantLogits);
            }
        }
        catch (Exception ex)
        {
            // A failed packed pass leaves the involved caches in an indeterminate state;
            // fail every involved request rather than guessing which ones survived.
            for (int i = prefilling.Count - 1; i >= 0; i--)
            {
                prefilling[i].Req.Output.Writer.TryComplete(ex);
                DropPrefilling(prefilling, i);
            }
            return;
        }

        for (int s = sCount - 1; s >= 0; s--)
        {
            var p = prefilling[s];
            p.Consumed += takes[s];
            if (logitsPerSeq[s] is { } logits)
            {
                prefilling.RemoveAt(s);
                ActivateSeq(p, logits, active);
            }
        }
    }

    /// <summary>Removes prefilling[i], disposing its cache and releasing its budget reservation.</summary>
    private void DropPrefilling(List<PrefillingSeq> prefilling, int i)
    {
        var p = prefilling[i];
        p.Cache.Dispose();
        _committedTokens -= p.ProjectedTokens;
        prefilling.RemoveAt(i);
        Interlocked.Decrement(ref _activeCount);
    }

    /// <summary>Releases a retired decode sequence's cache and budget reservation.</summary>
    private void RetireSeq(ActiveSeq seq)
    {
        seq.Cache.Dispose();
        _committedTokens -= seq.ProjectedTokens;
        Interlocked.Decrement(ref _activeCount);
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

    /// <summary>
    /// Promotes a fully prefilled prompt to the active decode set: samples the first
    /// token from the prefill logits, seeds the reasoning state machine, and emits the
    /// first chunk. A first token that is already a stop token completes the request
    /// immediately.
    /// </summary>
    private void ActivateSeq(PrefillingSeq p, float[] logits, List<ActiveSeq> active)
    {
        var req = p.Req;

        // Stop on ANY end-of-generation token, not just the configured EOS — matches the
        // single-user InferenceEngine path. A model with an alternate end token (e.g. Gemma's
        // <eos>, distinct from its <turn|> EOS) would otherwise decode it as text and run on.
        // StopTokenIds replaces this set; AdditionalStopTokenIds is unioned on top (issue #304).
        System.Collections.Immutable.ImmutableArray<int> stopIds =
            req.Sp.ResolveStopSet(_tokenizer.EogTokenIds);
        var rng = new Random();

        int firstToken = req.Sp.Temperature <= 0f
            ? Sampler.Greedy(logits)
            : Sampler.Sample(logits, req.Sp, rng);

        if (stopIds.Contains(firstToken))
        {
            req.Output.Writer.TryComplete();
            p.Cache.Dispose();
            _committedTokens -= p.ProjectedTokens;
            Interlocked.Decrement(ref _activeCount);
            return;
        }

        // Seed InThinking from the prompt itself (issue #92). Qwen3.6 and other
        // reasoning models append a bare `<think>` token to the generation prompt
        // via their chat template, so the model is already inside a reasoning
        // block before the first sampled token.
        bool promptInThinking = false;
        if (_thinkingEnabled)
        {
            foreach (int tok in p.Tokens)
            {
                if (tok == _thinkTokenId) promptInThinking = true;
                else if (tok == _endThinkTokenId) promptInThinking = false;
            }
        }

        var seq = new ActiveSeq
        {
            CurrentToken = firstToken,
            Position = p.Tokens.Length,
            Cache = p.Cache,
            Sp = req.Sp,
            Output = req.Output,
            StopIds = stopIds,
            Rng = rng,
            Ct = req.Ct,
            ProjectedTokens = p.ProjectedTokens,
            TokenCount = 1,
            InThinking = promptInThinking,
        };

        // Route the first sampled token through the same always-consume state machine as the
        // decode loop: a reasoning boundary token is consumed and never emitted, with the state
        // flip gated on the current InThinking. A bare `<channel|>` close (or a double-open) is
        // swallowed rather than leaking its literal marker as text — issue #304.
        if (_thinkingEnabled && firstToken == _thinkTokenId)
        {
            if (!seq.InThinking) seq.InThinking = true;
        }
        else if (_thinkingEnabled && firstToken == _endThinkTokenId)
        {
            if (seq.InThinking) seq.InThinking = false;
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
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _queue.Writer.TryComplete();
    }
}
