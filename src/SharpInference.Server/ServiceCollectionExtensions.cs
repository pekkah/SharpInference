using Microsoft.AspNetCore.Http.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using SharpInference.Engine;

namespace SharpInference.Server;

/// <summary>
/// DI registration entry points for the SharpInference HTTP API.
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>Default configuration section name bound to <see cref="SharpInferenceServerOptions"/>.</summary>
    public const string DefaultConfigurationSection = "SharpInference";

    /// <summary>
    /// Registers the SharpInference engine, chat-template renderer, metrics counters, and
    /// JSON source-gen context. The engine itself is constructed lazily on first request,
    /// so the call returns immediately even when <see cref="SharpInferenceServerOptions.ModelPath"/>
    /// points at a multi-gigabyte GGUF file.
    /// </summary>
    /// <param name="services">Service collection to mutate.</param>
    /// <param name="configure">
    /// Optional inline configuration. Runs after any prior <c>Configure&lt;SharpInferenceServerOptions&gt;</c>
    /// call (e.g. binding from <see cref="IConfiguration"/>) so callers can override individual fields.
    /// </param>
    public static IServiceCollection AddSharpInference(
        this IServiceCollection services,
        Action<SharpInferenceServerOptions>? configure = null)
    {
        services.AddOptions<SharpInferenceServerOptions>();
        if (configure is not null)
            services.Configure(configure);

        // TryAdd: a test or downstream module may already have registered a fake/replacement
        // for any of these services. We never overwrite an existing registration here.
        services.TryAddSingleton<ServerMetrics>();

        // Request admission gate (issue #109). Resolved lazily so the options object is fully
        // bound (config + the host's inline Configure, which runs after AddSharpInference) by
        // the time the first request constructs it. Disabled (passthrough) unless
        // MaxConcurrentRequests is set.
        services.TryAddSingleton(sp => new Endpoints.RequestConcurrencyGate(
            sp.GetRequiredService<IOptions<SharpInferenceServerOptions>>().Value.MaxConcurrentRequests));
        services.TryAddSingleton<ChatTemplateRenderer>(sp =>
        {
            var opts = sp.GetRequiredService<IOptions<SharpInferenceServerOptions>>().Value;
            return new ChatTemplateRenderer(opts.Architecture);
        });

        services.TryAddSingleton<IInferenceEngine>(sp =>
        {
            var opts = sp.GetRequiredService<IOptions<SharpInferenceServerOptions>>().Value;
            var loaded = (opts.EngineFactory ?? (s => InferenceEngineLoader.Load(opts)))(sp);

            // Hand the single-user engine the host's logger so its per-request perf trace
            // (Debug level) flows through the configured logging pipeline rather than stderr.
            if (loaded.Engine is InferenceEngine ie)
                ie.Logger = sp.GetService<ILoggerFactory>()?.CreateLogger("SharpInference.Engine");

            // Reconfigure the renderer with the model's actual arch + Jinja template now
            // that we have them. Done here rather than as a separate DI registration so
            // resolving ChatTemplateRenderer doesn't transitively trigger model loading
            // — important for tests that override IInferenceEngine but expect the
            // renderer to use the safe fallback path.
            sp.GetRequiredService<ChatTemplateRenderer>().Configure(
                loaded.Architecture, loaded.ChatTemplate, loaded.ToolBoundaryStopTokenIds);

            return loaded.Engine;
        });

        // Wire the source-gen JSON context into ASP.NET Core's JSON pipeline so
        // POST bodies and SSE deltas are AOT-compatible.
        services.Configure<JsonOptions>(o =>
            o.SerializerOptions.TypeInfoResolverChain.Insert(0, SharpInferenceJsonContext.Default));

        return services;
    }

    /// <summary>
    /// Convenience overload that binds <see cref="SharpInferenceServerOptions"/> from the supplied
    /// <see cref="IConfiguration"/> section before applying <paramref name="configure"/>.
    /// </summary>
    /// <param name="services">Service collection to mutate.</param>
    /// <param name="configuration">
    /// Configuration root (or sub-section) holding a <see cref="DefaultConfigurationSection"/>
    /// child. Pass <c>builder.Configuration</c> for the typical case.
    /// </param>
    /// <param name="configure">Optional inline tweaks applied after the configuration bind.</param>
    public static IServiceCollection AddSharpInference(
        this IServiceCollection services,
        IConfiguration configuration,
        Action<SharpInferenceServerOptions>? configure = null)
    {
        services.Configure<SharpInferenceServerOptions>(configuration.GetSection(DefaultConfigurationSection));
        return services.AddSharpInference(configure);
    }
}
