using System.Diagnostics;
using Microsoft.AspNetCore.Http;
using SharpInference.Engine;

namespace SharpInference.Server.Endpoints;

public static class HealthEndpoints
{
    private static readonly Stopwatch s_uptime = Stopwatch.StartNew();
    private static long s_totalRequests;
    private static long s_totalTokens;

    public static IEndpointRouteBuilder MapHealthEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapGet("/health", (IInferenceEngine engine) =>
            Results.Ok(new HealthStatus("ok", engine.ModelId,
                (long)s_uptime.Elapsed.TotalSeconds)));

        app.MapGet("/metrics", HandleMetrics);
        return app;
    }

    internal static void RecordRequest() =>
        System.Threading.Interlocked.Increment(ref s_totalRequests);

    internal static void RecordTokens(long count) =>
        System.Threading.Interlocked.Add(ref s_totalTokens, count);

    private static Task HandleMetrics(HttpContext ctx, IInferenceEngine engine)
    {
        ctx.Response.ContentType = "text/plain; version=0.0.4";
        var uptime = s_uptime.Elapsed.TotalSeconds;
        double tps = uptime > 0 ? s_totalTokens / uptime : 0;
        return ctx.Response.WriteAsync(
            $"# HELP sharpi_requests_total Total inference requests served\n" +
            $"# TYPE sharpi_requests_total counter\n" +
            $"sharpi_requests_total {s_totalRequests}\n" +
            $"# HELP sharpi_tokens_generated_total Total tokens generated\n" +
            $"# TYPE sharpi_tokens_generated_total counter\n" +
            $"sharpi_tokens_generated_total {s_totalTokens}\n" +
            $"# HELP sharpi_uptime_seconds Server uptime in seconds\n" +
            $"# TYPE sharpi_uptime_seconds gauge\n" +
            $"sharpi_uptime_seconds {(long)uptime}\n" +
            $"# HELP sharpi_tokens_per_second Lifetime-average tokens generated per second\n" +
            $"# TYPE sharpi_tokens_per_second gauge\n" +
            $"sharpi_tokens_per_second {tps:F2}\n" +
            $"# HELP sharpi_queue_depth Number of requests waiting to start generation\n" +
            $"# TYPE sharpi_queue_depth gauge\n" +
            $"sharpi_queue_depth {engine.QueueDepth}\n" +
            $"# HELP sharpi_active_requests Number of requests currently generating tokens\n" +
            $"# TYPE sharpi_active_requests gauge\n" +
            $"sharpi_active_requests {engine.ActiveRequests}\n" +
            $"# HELP sharpi_prefix_cache_enabled 1 if the engine's prefix-cache reuse path is active, 0 if disabled (e.g. GDN hybrid models)\n" +
            $"# TYPE sharpi_prefix_cache_enabled gauge\n" +
            $"sharpi_prefix_cache_enabled {(engine.PrefixCacheEnabled ? 1 : 0)}\n" +
            $"# HELP sharpi_prefill_tokens_reused_total Total prompt tokens skipped via the prefix-cache fast path\n" +
            $"# TYPE sharpi_prefill_tokens_reused_total counter\n" +
            $"sharpi_prefill_tokens_reused_total {engine.PrefillTokensReused}\n",
            ctx.RequestAborted);
    }
}

public sealed record HealthStatus(string Status, string Model, long UptimeSeconds);
