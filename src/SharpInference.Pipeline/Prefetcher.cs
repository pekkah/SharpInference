using SharpInference.Core;

namespace SharpInference.Pipeline;

/// <summary>
/// Asynchronous weight prefetcher.
/// Analyses the upcoming layer schedule and issues DMA transfers
/// before weights are needed, hiding I/O latency.
/// </summary>
public sealed class Prefetcher : IDisposable
{
    private readonly MemoryHierarchy _memory;
    private readonly Channel<PrefetchRequest> _queue;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _worker;

    public Prefetcher(MemoryHierarchy memory, int queueDepth = 4)
    {
        _memory = memory;
        _queue = System.Threading.Channels.Channel.CreateBounded<PrefetchRequest>(queueDepth);
        _worker = Task.Run(RunAsync);
    }

    public ValueTask EnqueueAsync(PrefetchRequest request, CancellationToken ct = default) =>
        _queue.Writer.WriteAsync(request, ct);

    private async Task RunAsync()
    {
        await foreach (var req in _queue.Reader.ReadAllAsync(_cts.Token))
        {
            await _memory.PromoteToGpuAsync(req.TensorName, _cts.Token);
        }
    }

    public void Dispose()
    {
        _cts.Cancel();
        _worker.Wait();
        _cts.Dispose();
    }
}

public readonly record struct PrefetchRequest(string TensorName, int Priority = 0);
