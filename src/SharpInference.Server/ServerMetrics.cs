using System.Diagnostics;

namespace SharpInference.Server;

/// <summary>
/// In-process counters scraped by <c>/metrics</c> and exposed to anyone who injects
/// the service. Registered as a singleton by
/// <see cref="ServiceCollectionExtensions.AddSharpInference"/>, so counters live for
/// the process lifetime and are shared across every endpoint.
/// </summary>
public sealed class ServerMetrics
{
    private readonly Stopwatch _uptime = Stopwatch.StartNew();
    private long _totalRequests;
    private long _totalTokens;

    /// <summary>Wall-clock time since the metrics instance was constructed.</summary>
    public TimeSpan Uptime => _uptime.Elapsed;

    /// <summary>Lifetime count of inference requests admitted to the engine.</summary>
    public long TotalRequests => Interlocked.Read(ref _totalRequests);

    /// <summary>Lifetime count of tokens emitted by the engine (text + reasoning).</summary>
    public long TotalTokens => Interlocked.Read(ref _totalTokens);

    /// <summary>Increments <see cref="TotalRequests"/> by one. Called once per inbound HTTP request.</summary>
    public void RecordRequest() => Interlocked.Increment(ref _totalRequests);

    /// <summary>Adds <paramref name="count"/> to <see cref="TotalTokens"/>.</summary>
    public void RecordTokens(long count) => Interlocked.Add(ref _totalTokens, count);
}
