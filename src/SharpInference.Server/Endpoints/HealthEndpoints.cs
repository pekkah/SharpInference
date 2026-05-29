using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using SharpInference.Engine;

namespace SharpInference.Server.Endpoints;

public static class HealthEndpoints
{
    public static IEndpointRouteBuilder MapHealthEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapGet("/health", (IInferenceEngine engine, ServerMetrics metrics) =>
            Results.Ok(new HealthStatus("ok", engine.ModelId,
                (long)metrics.Uptime.TotalSeconds)));

        app.MapGet("/metrics", HandleMetrics);
        return app;
    }

    private static Task HandleMetrics(HttpContext ctx, IInferenceEngine engine, ServerMetrics metrics)
    {
        ctx.Response.ContentType = "text/plain; version=0.0.4";
        double uptime = metrics.Uptime.TotalSeconds;
        long totalRequests = metrics.TotalRequests;
        long totalTokens = metrics.TotalTokens;
        double tps = uptime > 0 ? totalTokens / uptime : 0;
        return ctx.Response.WriteAsync(
            $"# HELP sharpi_requests_total Total inference requests served\n" +
            $"# TYPE sharpi_requests_total counter\n" +
            $"sharpi_requests_total {totalRequests}\n" +
            $"# HELP sharpi_tokens_generated_total Total tokens generated\n" +
            $"# TYPE sharpi_tokens_generated_total counter\n" +
            $"sharpi_tokens_generated_total {totalTokens}\n" +
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
