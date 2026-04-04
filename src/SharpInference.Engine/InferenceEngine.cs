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
/// </summary>
public sealed class InferenceEngine : IInferenceEngine, IDisposable
{
    private readonly IForwardPass _fwd;
    private readonly ITokenizer _tokenizer;
    private readonly IDisposable[] _owned;
    private readonly SemaphoreSlim _gate = new(1, 1);

    // Prefix caching state (guarded by _gate — only accessed during generation).
    private int[]? _prevTokens;

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

                    // Decode loop
                    for (int i = 0; i < sp.MaxNewTokens; i++)
                    {
                        ct.ThrowIfCancellationRequested();

                        int next = sp.Temperature <= 0f
                            ? Sampler.Greedy(logits)
                            : Sampler.Sample(logits, sp, rng);

                        if (stopIds.Contains(next)) break;

                        channel.Writer.TryWrite(_tokenizer.Decode([next]));
                        logits = _fwd.Forward(next, tokens.Length + i);
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
