using System.Threading.Channels;

namespace SharpInference.Engine;

/// <summary>
/// Background prefetcher for MoE expert weights.
/// After the router selects experts for the current token, callers enqueue
/// the predicted experts for the next token (or next layer). The background
/// worker calls <see cref="ExpertSlotManager.Preload"/> so experts are
/// GPU-resident before they are needed, hiding upload latency.
/// </summary>
public sealed class MoEPrefetcher : IDisposable
{
    private readonly ExpertSlotManager _slotManager;
    private readonly Channel<PrefetchBatch> _channel;
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _worker;

    public MoEPrefetcher(ExpertSlotManager slotManager, int queueDepth = 8)
    {
        _slotManager = slotManager;
        _channel = Channel.CreateBounded<PrefetchBatch>(new BoundedChannelOptions(queueDepth)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = false
        });
        _worker = Task.Run(RunAsync);
    }

    /// <summary>
    /// Enqueue a prefetch request for <paramref name="expertIds"/> in
    /// <paramref name="layer"/>. Non-blocking: drops oldest if queue is full.
    /// </summary>
    public void EnqueuePrefetch(int layer, ReadOnlySpan<int> expertIds)
    {
        var batch = new PrefetchBatch(layer, expertIds.ToArray());
        _channel.Writer.TryWrite(batch); // DropOldest if full — safe to ignore
    }

    private async Task RunAsync()
    {
        try
        {
            await foreach (var batch in _channel.Reader.ReadAllAsync(_cts.Token))
            {
                foreach (int expertId in batch.ExpertIds)
                {
                    if (_cts.Token.IsCancellationRequested) return;
                    _slotManager.Preload(batch.Layer, expertId);
                }
            }
        }
        catch (OperationCanceledException) { }
    }

    public void Dispose()
    {
        _cts.Cancel();
        _channel.Writer.TryComplete();
        try { _worker.Wait(TimeSpan.FromSeconds(2)); } catch { }
        _cts.Dispose();
    }
}

/// <summary>A batch of experts to prefetch for a specific transformer layer.</summary>
public readonly record struct PrefetchBatch(int Layer, int[] ExpertIds);
