using System.Runtime.CompilerServices;
using System.Threading.Channels;
using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Continuous batching inference engine: admits multiple concurrent requests and decodes them
/// in lock-step batches, amortizing weight reads across N sequences per decode step.
///
/// Flow:
///   1. Caller enqueues requests via <see cref="GenerateAsync"/>.
///   2. Background batcher admits pending requests one at a time (prefilling each individually).
///   3. All active sequences are decoded together in a single <see cref="ForwardPass.BatchForwardMulti"/> call.
///   4. Sequences that hit EOS or max tokens are retired; their caches are returned to the pool.
///
/// Not supported for MoE models or when TurboQuant KV cache is enabled.
/// </summary>
public sealed class ContinuousBatchingEngine : IInferenceEngine, IDisposable
{
    private readonly ForwardPass _fwd;
    private readonly ITokenizer _tokenizer;
    private readonly int _maxBatchSize;

    private readonly Channel<PendingRequest> _queue =
        Channel.CreateUnbounded<PendingRequest>(new UnboundedChannelOptions { SingleWriter = false, SingleReader = true });
    private readonly Task _batcherTask;
    private volatile bool _disposed;

    // Observability counters (updated via Interlocked).
    private int _pendingCount;
    private int _activeCount;

    private sealed class PendingRequest(string prompt, SamplingParams sp, CancellationToken ct, Channel<string> output)
    {
        public readonly string Prompt = prompt;
        public readonly SamplingParams Sp = sp;
        public readonly CancellationToken Ct = ct;
        public readonly Channel<string> Output = output;
    }

    private sealed class ActiveSeq
    {
        public required int CurrentToken;
        public required int Position;       // position at which CurrentToken will be decoded
        public required PagedKvCache Cache;
        public required SamplingParams Sp;
        public required Channel<string> Output;
        public required int[] StopIds;
        public required Random Rng;
        public required CancellationToken Ct;
        public int TokenCount;
        // Per-sequence stateful UTF-8 decoder: reassembles multi-byte characters
        // split across tokens (CJK, emoji, smart quotes).
        public Utf8StreamDecoder StreamDec = new();
    }

    public ContinuousBatchingEngine(
        ForwardPass fwd,
        ITokenizer tokenizer,
        string modelId,
        int maxBatchSize = 8)
    {
        _fwd = fwd;
        _tokenizer = tokenizer;
        ModelId = modelId;
        _maxBatchSize = Math.Max(1, maxBatchSize);
        _batcherTask = Task.Run(BatcherLoop);
    }

    public string ModelId { get; }

    /// <summary>Number of requests queued but not yet being generated.</summary>
    public int QueueDepth => _pendingCount;

    /// <summary>Number of requests currently in the active decode batch.</summary>
    public int ActiveRequests => _activeCount;

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var channel = Channel.CreateUnbounded<string>(
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

        while (!_disposed)
        {
            // Admit pending requests into available batch slots
            while (active.Count < _maxBatchSize && _queue.Reader.TryRead(out var req))
            {
                Interlocked.Decrement(ref _pendingCount);
                try
                {
                    AdmitRequest(req, active);
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
                int next = seq.Sp.Temperature <= 0f
                    ? Sampler.Greedy(logitsBatch[i])
                    : Sampler.Sample(logitsBatch[i], seq.Sp, seq.Rng);

                bool done = seq.StopIds.Contains(next)
                    || seq.TokenCount >= seq.Sp.MaxNewTokens
                    || seq.Ct.IsCancellationRequested;

                if (done)
                {
                    var tail = seq.StreamDec.Flush();
                    if (tail.Length > 0)
                        seq.Output.Writer.TryWrite(tail);
                    seq.Output.Writer.TryComplete();
                    seq.Cache.Dispose();
                    active.RemoveAt(i);
                    Interlocked.Decrement(ref _activeCount);
                }
                else
                {
                    var chunk = seq.StreamDec.Append(_tokenizer.DecodeBytes(next));
                    if (chunk.Length > 0)
                        seq.Output.Writer.TryWrite(chunk);
                    seq.CurrentToken = next;
                    seq.Position++;
                    seq.TokenCount++;
                }
            }
        }

        // Drain: complete any remaining active sequences
        foreach (var seq in active)
        {
            var tail = seq.StreamDec.Flush();
            if (tail.Length > 0)
                seq.Output.Writer.TryWrite(tail);
            seq.Output.Writer.TryComplete();
            seq.Cache.Dispose();
            Interlocked.Decrement(ref _activeCount);
        }
        active.Clear();
    }

    private void AdmitRequest(PendingRequest req, List<ActiveSeq> active)
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

        var stopIds = req.Sp.StopTokenIds ?? [_tokenizer.EosTokenId];
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
        };
        var firstChunk = seq.StreamDec.Append(_tokenizer.DecodeBytes(firstToken));
        if (firstChunk.Length > 0)
            req.Output.Writer.TryWrite(firstChunk);

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
