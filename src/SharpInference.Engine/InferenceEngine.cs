using System.Runtime.CompilerServices;
using System.Threading.Channels;
using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Top-level inference engine. Wraps a forward pass + tokenizer, applies pre-formatted
/// prompts (caller applies chat template), and provides serialized async generation.
/// One request runs at a time; concurrent callers block in arrival order.
/// </summary>
public sealed class InferenceEngine : IInferenceEngine, IDisposable
{
    private readonly IForwardPass _fwd;
    private readonly ITokenizer _tokenizer;
    private readonly IDisposable[] _owned;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _disposed;

    public string ModelId { get; }

    /// <param name="fwd">Forward pass implementation (CPU / GPU / Hybrid). Owned by this engine.</param>
    /// <param name="tokenizer">Tokenizer matching the model vocabulary.</param>
    /// <param name="modelId">Human-readable model identifier returned in API responses.</param>
    /// <param name="owned">Additional disposable resources owned by this engine (backend, model handle, etc.).</param>
    public InferenceEngine(
        IForwardPass fwd,
        ITokenizer tokenizer,
        string modelId,
        params IDisposable[] owned)
    {
        _fwd = fwd;
        _tokenizer = tokenizer;
        ModelId = modelId;
        _owned = owned;
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            var channel = Channel.CreateUnbounded<string>(
                new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

            // Run the blocking CPU generation on a thread-pool thread.
            var genTask = Task.Run(() =>
            {
                try
                {
                    _fwd.ResetCache();
                    var tokens = _tokenizer.Encode(prompt);
                    var rng = new Random();
                    var stopIds = sp.StopTokenIds ?? [_tokenizer.EosTokenId];

                    // Prefill — sequential forward on all prompt tokens
                    ReadOnlySpan<float> logits = default;
                    for (int i = 0; i < tokens.Count; i++)
                    {
                        ct.ThrowIfCancellationRequested();
                        logits = _fwd.Forward(tokens[i], i);
                    }

                    // Decode loop
                    for (int i = 0; i < sp.MaxNewTokens; i++)
                    {
                        ct.ThrowIfCancellationRequested();

                        int next = sp.Temperature <= 0f
                            ? Sampler.Greedy(logits)
                            : Sampler.Sample(logits, sp, rng);

                        if (stopIds.Contains(next)) break;

                        channel.Writer.TryWrite(_tokenizer.Decode([next]));
                        logits = _fwd.Forward(next, tokens.Count + i);
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
