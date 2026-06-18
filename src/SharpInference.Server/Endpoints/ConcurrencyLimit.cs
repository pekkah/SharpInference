using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;

namespace SharpInference.Server.Endpoints;

/// <summary>
/// Process-wide admission gate for generation requests (issue #109). The single-user
/// <see cref="SharpInference.Engine.InferenceEngine"/> serializes overlapping requests on an
/// internal lock, so without this gate a concurrent request silently blocks for the full
/// duration of the in-flight one — an agentic client reads that as an indefinite hang. When
/// <see cref="SharpInferenceServerOptions.MaxConcurrentRequests"/> is set, this caps the number
/// of in-flight generation requests and the endpoints fast-reject the overflow with HTTP 429.
/// </summary>
internal sealed class RequestConcurrencyGate
{
    private readonly SemaphoreSlim? _sem;

    /// <summary>Configured ceiling (0 when disabled).</summary>
    public int Limit { get; }

    public RequestConcurrencyGate(int? maxConcurrent)
    {
        // A non-positive limit is treated as "disabled" rather than "reject everything", which
        // would wedge the server — the only sensible interpretation of 0/negative here.
        Limit = maxConcurrent.GetValueOrDefault();
        _sem = Limit > 0 ? new SemaphoreSlim(Limit, Limit) : null;
    }

    /// <summary>Whether admission control is active. When false the gate is a pure passthrough.</summary>
    public bool Enabled => _sem is not null;

    /// <summary>Non-blocking acquire — returns false immediately when at capacity (no queuing).</summary>
    public bool TryEnter() => _sem!.Wait(0);

    public void Exit() => _sem!.Release();
}

/// <summary>
/// Wires <see cref="RequestConcurrencyGate"/> onto a generation endpoint as an endpoint filter.
/// The filter brackets the entire request (including a streaming response, since the handler's
/// Task completes only when the stream is fully written), so the slot is held for the request's
/// whole lifetime and released in a finally.
/// </summary>
internal static class ConcurrencyLimitExtensions
{
    public static RouteHandlerBuilder WithConcurrencyLimit(this RouteHandlerBuilder builder) =>
        builder.AddEndpointFilter(async (ctx, next) =>
        {
            var gate = ctx.HttpContext.RequestServices.GetRequiredService<RequestConcurrencyGate>();
            if (!gate.Enabled)
                return await next(ctx);

            if (!gate.TryEnter())
                return BusyResult(ctx.HttpContext, gate.Limit);

            try
            {
                return await next(ctx);
            }
            finally
            {
                gate.Exit();
            }
        });

    private static IResult BusyResult(HttpContext http, int limit)
    {
        // Set the advisory header before returning — the result executes (and starts the
        // response) afterwards, so the header is still mutable here.
        http.Response.Headers.RetryAfter = "1";
        var error = new ErrorResponse(
            "rate_limit_error",
            $"The inference engine is at capacity (max {limit} concurrent request(s)). Retry shortly, " +
            "or start the server with SHARPI_MAX_BATCH>1 to enable continuous batching for concurrent requests.");
        return TypedResults.Json(error, SharpInferenceJsonContext.Default.ErrorResponse,
            statusCode: StatusCodes.Status429TooManyRequests);
    }
}
