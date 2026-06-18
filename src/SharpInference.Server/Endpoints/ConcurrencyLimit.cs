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
internal sealed class RequestConcurrencyGate : IDisposable
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

    /// <summary>Non-blocking acquire — returns false immediately when at capacity (no queuing).
    /// A no-op (returns true) when disabled, so callers don't have to special-case it.</summary>
    public bool TryEnter() => _sem?.Wait(0) ?? true;

    public void Exit() => _sem?.Release();

    /// <summary>Disposes the underlying semaphore on host shutdown (the DI container disposes
    /// this singleton).</summary>
    public void Dispose() => _sem?.Dispose();
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

            // The gated generation handlers write the response and return a bare Task (they do
            // not return an IResult), so `await next(ctx)` completes only after the response —
            // including a streaming SSE body — is fully written. The finally then releases the
            // slot exactly once on every path: normal completion, an exception out of next(), or
            // a mid-stream client abort (which the streaming handlers catch and return from
            // normally). A future handler that instead RETURNS an IResult would release the slot
            // before the result executes — revisit this (e.g. release via Response.OnCompleted)
            // if that ever changes.
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
