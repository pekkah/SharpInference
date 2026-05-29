using Microsoft.AspNetCore.Routing;
using SharpInference.Server.Endpoints;

namespace SharpInference.Server;

/// <summary>
/// Composite map-endpoints extension for hosts that want every SharpInference HTTP API
/// in one call. Individual <c>Map…Endpoints()</c> extensions remain available for hosts
/// that want only a subset (e.g. <c>MapOpenAiEndpoints()</c> alone behind an auth filter).
/// </summary>
public static class EndpointRouteBuilderExtensions
{
    /// <summary>
    /// Maps the OpenAI chat completions + models endpoints, the Anthropic <c>/v1/messages</c>
    /// endpoint, the OpenAI Responses endpoint, and the <c>/health</c> + <c>/metrics</c>
    /// observability endpoints onto <paramref name="endpoints"/>.
    /// </summary>
    public static IEndpointRouteBuilder MapSharpInference(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapOpenAiEndpoints();
        endpoints.MapAnthropicEndpoints();
        endpoints.MapResponsesEndpoints();
        endpoints.MapHealthEndpoints();
        return endpoints;
    }
}
