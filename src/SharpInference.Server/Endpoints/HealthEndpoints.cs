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

    private static Task HandleMetrics(HttpContext ctx)
    {
        ctx.Response.ContentType = "text/plain; version=0.0.4";
        var uptime = (long)s_uptime.Elapsed.TotalSeconds;
        return ctx.Response.WriteAsync(
            $"# HELP sharpi_requests_total Total inference requests served\n" +
            $"# TYPE sharpi_requests_total counter\n" +
            $"sharpi_requests_total {s_totalRequests}\n" +
            $"# HELP sharpi_tokens_generated_total Total tokens generated\n" +
            $"# TYPE sharpi_tokens_generated_total counter\n" +
            $"sharpi_tokens_generated_total {s_totalTokens}\n" +
            $"# HELP sharpi_uptime_seconds Server uptime in seconds\n" +
            $"# TYPE sharpi_uptime_seconds gauge\n" +
            $"sharpi_uptime_seconds {uptime}\n",
            ctx.RequestAborted);
    }
}

public sealed record HealthStatus(string Status, string Model, long UptimeSeconds);
