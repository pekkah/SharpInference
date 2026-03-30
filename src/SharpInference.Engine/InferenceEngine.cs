using SharpInference.Core;
using SharpInference.Pipeline;

namespace SharpInference.Engine;

/// <summary>
/// Top-level inference engine. Orchestrates the forward pass, speculative decoding,
/// sampling, and token streaming across the memory hierarchy.
/// </summary>
public sealed class InferenceEngine : IAsyncDisposable
{
    private readonly ModelGraph _model;
    private readonly IComputeBackend _backend;
    private readonly MemoryHierarchy _memory;
    private readonly Prefetcher _prefetcher;
    private readonly KvCache _kvCache;

    public InferenceEngine(
        ModelGraph model,
        IComputeBackend backend,
        MemoryHierarchy memory)
    {
        _model = model;
        _backend = backend;
        _memory = memory;
        _prefetcher = new Prefetcher(memory);
        _kvCache = new KvCache(model.Hyperparams);
    }

    /// <summary>
    /// Run a single forward pass and return logits for the last token position.
    /// </summary>
    public ValueTask<Tensor> ForwardAsync(
        ReadOnlyMemory<int> tokens,
        int position,
        CancellationToken ct = default)
    {
        // TODO: embed ? n × transformer layers ? norm ? unembed
        throw new NotImplementedException();
    }

    /// <summary>
    /// Generate tokens autoregressively, yielding each token ID as it is sampled.
    /// </summary>
    public async IAsyncEnumerable<int> GenerateAsync(
        ReadOnlyMemory<int> prompt,
        SamplingParams sampling,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        // TODO: prefill phase + decode loop with speculative decoding
        await Task.CompletedTask;
        yield break;
    }

    public async ValueTask DisposeAsync()
    {
        _prefetcher.Dispose();
        _kvCache.Dispose();
        _backend.Dispose();
        await _memory.DisposeAsync();
    }
}
