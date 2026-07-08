using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using SharpInference.Core;
using SharpInference.Core.Grammar;
using SharpInference.Engine;

namespace SharpInference.Server.Endpoints;

public static class OpenAiEndpoints
{
    public static IEndpointRouteBuilder MapOpenAiEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/chat/completions", HandleChatCompletion).WithConcurrencyLimit();
        app.MapGet("/v1/models", HandleListModels);
        return app;
    }

    private static async Task HandleChatCompletion(
        HttpContext ctx,
        IInferenceEngine engine,
        ChatTemplateRenderer chatTemplate,
        ServerMetrics metrics,
        IOptions<SharpInferenceServerOptions> options)
    {
        var opts = options.Value;
        ChatCompletionRequest? req;
        try
        {
            req = await ctx.Request.ReadFromJsonAsync(SharpInferenceJsonContext.Default.ChatCompletionRequest, ctx.RequestAborted);
        }
        catch (JsonException ex)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", ex.Message),
                    SharpInferenceJsonContext.Default.ErrorResponse), ctx.RequestAborted);
            return;
        }
        catch (Microsoft.AspNetCore.Http.BadHttpRequestException ex)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", ex.Message),
                    SharpInferenceJsonContext.Default.ErrorResponse), ctx.RequestAborted);
            return;
        }

        if (req is null || req.Messages is null || req.Messages.Length == 0)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", "messages array is required"),
                    SharpInferenceJsonContext.Default.ErrorResponse), ctx.RequestAborted);
            return;
        }

        metrics.RecordRequest();

        // When enable_thinking is absent the default depends on the model family — Gemma 4
        // defaults reasoning off (see ChatTemplateRenderer.ModelDefaultsThinkingOff) while ChatML
        // models default on; an explicit flag always wins. Server-level DisableThinking
        // (SHARPI_NO_THINKING) forces reasoning off regardless — for agentic clients that never
        // send the per-request flag.
        bool enableThinking = (req.EnableThinking ?? !chatTemplate.ModelDefaultsThinkingOff)
                              && !options.Value.DisableThinking;

        // preserve_thinking (per-request) or SHARPI_PRESERVE_THINKING (server-wide) keeps prior
        // assistant turns' reasoning in the rendered history instead of the default
        // ChatTemplate.ScrubAssistantThinking strip — see SharpInferenceServerOptions.PreserveThinking.
        // Server-level DisableThinking still wins: an operator who has decided this deployment never
        // exposes <think> content to the model shouldn't have historical reasoning re-injected either.
        bool preserveThinking = (req.PreserveThinking ?? options.Value.PreserveThinking)
                                && !options.Value.DisableThinking;
        var adapter = chatTemplate.ToolCallAdapter;

        // Tool-aware rendering: if either tool definitions or a history-side
        // tool_call / tool message is present, route through the rich-message path
        // so the chat template can inject the tool schema and replay prior calls.
        bool toolsActive = req.Tools is { Length: > 0 } || HasToolMessages(req.Messages);
        string prompt;
        // Issue #102: render a history-only canonical view of the same messages so the
        // engine can snapshot at the chat-history boundary instead of after the inline
        // <think> injection. The next turn's generating prompt starts with this exact
        // canonical render plus the scrubbed assistant content + new user turn, so
        // storing it as the snapshot anchor is what makes prefix reuse possible across
        // multi-turn agentic loops.
        string? canonicalHistoryPrefix;
        // Image content parts (issue #253) collected (base64 → bytes) and replaced by the
        // model's <|image|> placeholder; the engine splices each image's soft tokens at it.
        var images = new List<byte[]>();
        try
        {
            if (toolsActive)
            {
                var (richMessages, tools) = BuildRichMessageList(req, adapter, images, preserveThinking);
                prompt = chatTemplate.Format(richMessages, enableThinking, tools);
                canonicalHistoryPrefix = chatTemplate.Format(richMessages, enableThinking, tools, addGenerationPrompt: false);
            }
            else
            {
                var messages = BuildMessageList(req.Messages, req.ResponseFormat?.Type, images, preserveThinking);
                prompt = chatTemplate.Format(messages, enableThinking);
                canonicalHistoryPrefix = chatTemplate.Format(messages, enableThinking, addGenerationPrompt: false);
            }
        }
        catch (ImageContentException ex)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", ex.Message),
                    SharpInferenceJsonContext.Default.ErrorResponse), ctx.RequestAborted);
            return;
        }

        if (images.Count > 0 && !engine.SupportsImageInput)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new ErrorResponse("invalid_request_error",
                        "This model is not configured for image input. Start the server with a Gemma 4 model and " +
                        "SHARPI_MMPROJ pointing at its gemma4uv mmproj, on a CPU (NGpuLayers=0) or full-CUDA " +
                        "(NGpuLayers=-1) backend."),
                    SharpInferenceJsonContext.Default.ErrorResponse), ctx.RequestAborted);
            return;
        }

        // Safety: only forward the canonical hint when it's a string-prefix of the generating
        // prompt. A non-Jinja fallback that diverges, or an exotic template that interleaves
        // generation prep mid-history, would fail the engine's token-prefix check too — but
        // skipping the hint here saves a tokenizer call on the engine side and keeps the path
        // identical to legacy when the template doesn't fit the assumption.
        if (canonicalHistoryPrefix is not null && !prompt.StartsWith(canonicalHistoryPrefix, StringComparison.Ordinal))
            canonicalHistoryPrefix = null;

        // Parse logit_bias: OpenAI sends {"tokenId": biasValue} with string keys
        IReadOnlyDictionary<int, float>? logitBias = null;
        if (req.LogitBias is { Count: > 0 })
        {
            var d = new Dictionary<int, float>(req.LogitBias.Count);
            foreach (var (k, v) in req.LogitBias)
                if (int.TryParse(k, out int id)) d[id] = v;
            if (d.Count > 0) logitBias = d;
        }

        var sp = SamplingParamsBuilder.Build(opts,
            temperature: req.Temperature,
            topP:        req.TopP,
            maxTokens:   req.MaxTokens,
            maxThinking: req.MaxThinkingTokens,
            logitBias:   logitBias,
            thinkingDisabled: !enableThinking);

        // Tool-active agentic turn: add the model family's tool-boundary stop (Gemma 4:
        // <|tool_response>) so generation halts the instant the tool calls finish rather than
        // running on into a hallucinated trailing turn. Additive — keeps the EOG set (issue #304).
        if (toolsActive && chatTemplate.ToolBoundaryStopTokenIds is { Count: > 0 } toolStops)
            sp = sp with { AdditionalStopTokenIds = [.. toolStops] };

        // Schema/grammar-constrained tool-argument decoding (issue #374): opt-in, and skipped when
        // the client forces no tool use (tool_choice:"none"). Restricts the sampler to tokens that
        // keep the arguments schema-conformant in the model's native call syntax.
        ITokenConstraint? toolConstraint = null;
        if (toolsActive && req.Tools is { Length: > 0 } && ToolGrammarHelper.Enabled(opts)
            && !IsToolChoiceNone(req.ToolChoice))
        {
            var schemas = ToolGrammarHelper.ToSchemas(
                req.Tools.Select(t => (t.Function?.Name, t.Function?.Parameters)));
            toolConstraint = chatTemplate.BuildToolArgumentConstraint(schemas);
        }

        // Whole-turn JSON-Schema-constrained output (issue #423 follow-up): the SharpInference
        // analogue of OpenAI/llama.cpp's response_format:json_schema (and llama.cpp's flat
        // json_object.schema extension). An explicit schema request is a hard requirement, so
        // failures return 400 rather than silently generating unconstrained output.
        ITokenConstraint? jsonSchemaConstraint = null;
        JsonElement? requestedSchema = null;
        if (req.ResponseFormat?.Type == "json_schema")
        {
            // Unlike "json_object" (below), a schema is REQUIRED here -- the client explicitly asked
            // for schema-constrained output, so a missing json_schema.schema is a client error, not a
            // silent no-op (issue #423 follow-up: an explicit schema request is a hard requirement).
            if (req.ResponseFormat.JsonSchema?.Schema is { } s)
                requestedSchema = s;
            else
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync(
                    JsonSerializer.Serialize(new ErrorResponse("invalid_request_error",
                            "response_format.type is \"json_schema\" but no json_schema.schema was provided."),
                        SharpInferenceJsonContext.Default.ErrorResponse), ctx.RequestAborted);
                return;
            }
        }
        else if (req.ResponseFormat?.Type == "json_object")
        {
            // llama.cpp's flat extension: an optional schema alongside the plain "valid JSON" request.
            // Absent here just means "valid JSON, no shape enforced" -- the pre-existing behavior.
            requestedSchema = req.ResponseFormat.Schema;
        }
        if (requestedSchema is { } schemaElement)
        {
            if (chatTemplate.Grammar is not { } vocab)
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync(
                    JsonSerializer.Serialize(new ErrorResponse("invalid_request_error",
                            "response_format.json_schema requires a loaded model vocabulary, which isn't available for this engine."),
                        SharpInferenceJsonContext.Default.ErrorResponse), ctx.RequestAborted);
                return;
            }
            try
            {
                var schemaObject = ToolSchema.FromOpenAiFunction("_", schemaElement).Arguments;
                jsonSchemaConstraint = new JsonSchemaOutputConstraint(vocab, schemaObject);
            }
            catch (ArgumentException ex)
            {
                ctx.Response.StatusCode = 400;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync(
                    JsonSerializer.Serialize(new ErrorResponse("invalid_request_error",
                            $"response_format.json_schema could not be compiled: {ex.Message}"),
                        SharpInferenceJsonContext.Default.ErrorResponse), ctx.RequestAborted);
                return;
            }
        }

        // Caller-supplied whole-turn output constraint (issue #423): independent of tool schemas,
        // AND-composed with the tool-argument and json-schema constraints above rather than one
        // overriding the others.
        var outputConstraint = opts.OutputConstraintFactory?.Invoke(ctx.RequestServices);
        if (TokenConstraints.Combine(outputConstraint, jsonSchemaConstraint, toolConstraint) is { } constraint)
            sp = sp with { Constraint = constraint };

        var requestId = $"chatcmpl-{Guid.NewGuid():N}";
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (req.Stream == true)
        {
            await HandleStreaming(ctx, engine, metrics, adapter, toolsActive, prompt, canonicalHistoryPrefix, images, sp, requestId, created);
        }
        else
        {
            await HandleNonStreaming(ctx, engine, metrics, adapter, toolsActive, prompt, canonicalHistoryPrefix, images, sp, requestId, created);
        }
    }

    private static async Task HandleNonStreaming(
        HttpContext ctx, IInferenceEngine engine, ServerMetrics metrics, IToolCallAdapter adapter,
        bool toolsActive, string prompt, string? canonicalHistoryPrefix, IReadOnlyList<byte[]> images, SamplingParams sp, string requestId, long created)
    {
        var textSb = new StringBuilder();
        var reasoningSb = new StringBuilder();
        int textTokens = 0;
        int reasoningTokens = 0;
        int promptTokens = 0;
        bool truncatedByMaxTokens = false;

        try
        {
            await foreach (var c in ImageContent.Generate(engine, prompt, canonicalHistoryPrefix, images, sp, ctx.RequestAborted))
            {
                if (c.Kind == GenerateChunkKind.Usage)
                {
                    promptTokens = c.PromptTokens;
                }
                else if (c.Kind == GenerateChunkKind.Stop)
                {
                    truncatedByMaxTokens = c.TruncatedByMaxTokens;
                }
                else if (c.Kind == GenerateChunkKind.Thinking)
                {
                    reasoningSb.Append(c.Text);
                    reasoningTokens++;
                }
                else
                {
                    textSb.Append(c.Text);
                    textTokens++;
                }
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // A residual generation error (e.g. an image-count mismatch from a user-typed
            // <|image|>, or a projection failure) would otherwise surface as a bare 500 with no
            // body. The response hasn't started yet on the non-streaming path, so return a
            // structured error the OpenAI SDK can parse.
            ctx.Response.StatusCode = 500;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new ErrorResponse("internal_error", ex.Message),
                    SharpInferenceJsonContext.Default.ErrorResponse), ctx.RequestAborted);
            return;
        }

        int completionTokens = textTokens + reasoningTokens;
        metrics.RecordTokens(completionTokens);

        var rawText = textSb.ToString();
        IReadOnlyList<ParsedToolCall> parsedCalls;
        string plainText;
        bool truncatedCall = false;
        if (toolsActive)
        {
            var pr = adapter.Parse(rawText);
            plainText = pr.PlainText;
            parsedCalls = pr.Calls;
            truncatedCall = pr.Truncated;
        }
        else
        {
            // No tools declared on this request → don't try to interpret the output as
            // structured calls. Surfaces raw text identically to the pre-tools behaviour.
            plainText = rawText;
            parsedCalls = [];
        }

        OaiToolCall[]? toolCalls = null;
        string? content = plainText;
        if (parsedCalls.Count > 0)
        {
            toolCalls = parsedCalls
                .Select(c => new OaiToolCall(
                    Id: $"call_{Guid.NewGuid():N}",
                    Type: "function",
                    Function: new OaiToolCallFunction(c.Name, JinjaChatTemplate.SerializeToJson(c.Arguments))))
                .ToArray();
            // OpenAI returns content: null when a tool_calls array is present and there
            // was no accompanying text; an empty string would be a wire-shape mismatch.
            if (plainText.Length == 0) content = null;
        }
        // truncatedCall: model hit max_tokens / EOS inside an unterminated tool call — the
        // partial was surfaced as content by Parse, so report length instead of tool_calls.
        // truncatedByMaxTokens: no tool call in progress, but the budget was exhausted on
        // plain text/thinking. Mirrors the streaming path.
        string finishReason = DetermineFinishReason(parsedCalls.Count > 0, truncatedCall, truncatedByMaxTokens);

        var message = new OaiAssistantMessage(
            "assistant",
            content,
            reasoningSb.Length > 0 ? reasoningSb.ToString() : null,
            toolCalls);
        var usage = new ChatUsage(
            promptTokens, completionTokens, promptTokens + completionTokens,
            reasoningTokens > 0 ? new CompletionTokensDetails(reasoningTokens) : null);

        var response = new ChatCompletionResponse(
            requestId, "chat.completion", created, engine.ModelId,
            [new CompletionChoice(0, message, finishReason)],
            usage);

        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(
            JsonSerializer.Serialize(response, SharpInferenceJsonContext.Default.ChatCompletionResponse),
            ctx.RequestAborted);
    }

    private static async Task HandleStreaming(
        HttpContext ctx, IInferenceEngine engine, ServerMetrics metrics, IToolCallAdapter adapter,
        bool toolsActive, string prompt, string? canonicalHistoryPrefix, IReadOnlyList<byte[]> images, SamplingParams sp, string requestId, long created)
    {
        ctx.Response.ContentType = "text/event-stream";
        ctx.Response.Headers.CacheControl = "no-cache";
        ctx.Response.Headers.Connection = "keep-alive";

        // First chunk: role delta
        var firstChunk = new ChatCompletionChunk(
            requestId, "chat.completion.chunk", created, engine.ModelId,
            [new ChunkChoice(0, new ChunkDelta("assistant", null), null)]);
        await WriteEvent(ctx.Response, JsonSerializer.Serialize(firstChunk, SharpInferenceJsonContext.Default.ChatCompletionChunk));

        int maxOpenLen = adapter.MaxOpenTagLength;
        bool inToolCall = false;
        int toolCallContentStart = -1;
        var toolCallBuf = new StringBuilder();
        string pendingText = "";
        int toolCallIndex = 0;       // monotonic index into delta.tool_calls
        bool hasToolCalls = false;
        long tokenCount = 0;

        async Task WriteContentDelta(string text)
        {
            if (text.Length == 0) return;
            // OpenAI's wire shape carries the assistant's intent in tool_calls once a call has
            // been emitted; any trailing plain text is model control noise (e.g. Gemma's
            // <|tool_response> turn marker) and must not leak as a content delta (issue #150).
            // Preamble before the first call is unaffected (hasToolCalls is still false then).
            if (hasToolCalls) return;
            var chunk = new ChatCompletionChunk(
                requestId, "chat.completion.chunk", created, engine.ModelId,
                [new ChunkChoice(0, new ChunkDelta(null, text), null)]);
            await WriteEvent(ctx.Response, JsonSerializer.Serialize(chunk, SharpInferenceJsonContext.Default.ChatCompletionChunk));
        }

        async Task EmitToolCallsFromBlock(string blockContent)
        {
            var calls = adapter.ParseBlock(blockContent);
            foreach (var tc in calls)
            {
                hasToolCalls = true;
                var delta = new OaiToolCallDelta(
                    Index: toolCallIndex,
                    Id: $"call_{Guid.NewGuid():N}",
                    Type: "function",
                    Function: new OaiToolCallFunction(tc.Name, JinjaChatTemplate.SerializeToJson(tc.Arguments)));
                var chunk = new ChatCompletionChunk(
                    requestId, "chat.completion.chunk", created, engine.ModelId,
                    [new ChunkChoice(0, new ChunkDelta(null, null, null, [delta]), null)]);
                await WriteEvent(ctx.Response, JsonSerializer.Serialize(chunk, SharpInferenceJsonContext.Default.ChatCompletionChunk));
                toolCallIndex++;
            }
        }

        async Task ProcessTextChunk(string chunk)
        {
            if (inToolCall)
            {
                toolCallBuf.Append(chunk);
                string buf = toolCallBuf.ToString();
                int closeIdx = adapter.FindCloseMarker(buf, toolCallContentStart, out int afterClose);
                if (closeIdx >= 0)
                {
                    string block = buf[toolCallContentStart..closeIdx];
                    string remaining = buf[afterClose..];
                    toolCallBuf.Clear();
                    toolCallContentStart = -1;
                    inToolCall = false;
                    await EmitToolCallsFromBlock(block);
                    if (remaining.Length > 0)
                        await ProcessTextChunk(remaining);
                }
                return;
            }

            pendingText += chunk;
            int openIdx = adapter.FindOpenMarker(pendingText, 0, out int contentStart);
            if (openIdx >= 0)
            {
                if (openIdx > 0)
                    await WriteContentDelta(pendingText[..openIdx]);

                inToolCall = true;
                toolCallBuf.Clear();
                toolCallBuf.Append(pendingText, contentStart, pendingText.Length - contentStart);
                toolCallContentStart = 0;
                pendingText = "";

                if (toolCallBuf.Length > 0)
                    await ProcessTextChunk("");
                return;
            }

            int safeLen = Math.Max(0, pendingText.Length - (maxOpenLen - 1));
            if (safeLen > 0)
            {
                await WriteContentDelta(pendingText[..safeLen]);
                pendingText = pendingText[safeLen..];
            }
        }

        bool truncatedToolCall = false;
        bool truncatedByMaxTokens = false;
        try
        {
            await foreach (var c in ImageContent.Generate(engine, prompt, canonicalHistoryPrefix, images, sp, ctx.RequestAborted))
            {
                // Out-of-band usage chunk (issue #150): not a generated token and carries no
                // text. The streaming response doesn't emit a usage object (OpenAI gates that
                // behind stream_options.include_usage, unsupported here), so just skip it.
                if (c.Kind == GenerateChunkKind.Usage) continue;
                if (c.Kind == GenerateChunkKind.Stop)
                {
                    truncatedByMaxTokens = c.TruncatedByMaxTokens;
                    continue;
                }
                tokenCount++;
                if (c.Kind == GenerateChunkKind.Thinking)
                {
                    var delta = new ChunkDelta(null, null, c.Text);
                    var chunk = new ChatCompletionChunk(
                        requestId, "chat.completion.chunk", created, engine.ModelId,
                        [new ChunkChoice(0, delta, null)]);
                    await WriteEvent(ctx.Response, JsonSerializer.Serialize(chunk, SharpInferenceJsonContext.Default.ChatCompletionChunk));
                }
                else if (toolsActive)
                {
                    await ProcessTextChunk(c.Text);
                }
                else
                {
                    // No tools declared → skip the buffering state machine and forward each
                    // chunk as a separate content delta (clients rely on the streaming cadence).
                    await WriteContentDelta(c.Text);
                }
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // The SSE response is already committed (200 + role delta), so the status can't change.
            // Swallow so the finally + final chunk terminate the stream cleanly instead of the TCP
            // connection being reset mid-stream. Common image errors are rejected at parse time (400).
            // Log it — otherwise a genuine mid-stream failure is invisible (the client sees a clean
            // finish_reason "stop" with truncated text).
            ctx.RequestServices.GetService<ILoggerFactory>()?
                .CreateLogger("SharpInference.Server.Endpoints")
                .LogError(ex, "Generation failed after the streaming response was committed; the SSE stream is terminated without an error frame.");
        }
        finally
        {
            // Flush any remaining buffered text (cannot contain a partial open marker now).
            if (!inToolCall && pendingText.Length > 0)
            {
                try { await WriteContentDelta(pendingText); }
                catch (OperationCanceledException) { /* client disconnected */ }
                catch (IOException) { /* response stream closed */ }
                pendingText = "";
            }

            // Stream ended mid-tool-call (model hit max_tokens / EOS before emitting the close
            // marker). Surface the buffered bytes so the client sees the truncated output rather
            // than getting an empty response, and signal length-based truncation via finish_reason.
            if (inToolCall && toolCallBuf.Length > 0)
            {
                truncatedToolCall = true;
                try { await WriteContentDelta(toolCallBuf.ToString()); }
                catch (OperationCanceledException) { /* client disconnected */ }
                catch (IOException) { /* response stream closed */ }
                toolCallBuf.Clear();
            }
        }

        // Final chunk with finish_reason
        var finishReason = DetermineFinishReason(hasToolCalls, truncatedToolCall, truncatedByMaxTokens);
        var finalChunk = new ChatCompletionChunk(
            requestId, "chat.completion.chunk", created, engine.ModelId,
            [new ChunkChoice(0, new ChunkDelta(null, null), finishReason)]);
        await WriteEvent(ctx.Response, JsonSerializer.Serialize(finalChunk, SharpInferenceJsonContext.Default.ChatCompletionChunk));
        await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);

        metrics.RecordTokens(tokenCount);
    }

    private static Task HandleListModels(HttpContext ctx, IInferenceEngine engine)
    {
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var response = new ModelsResponse("list", [new ModelInfo(engine.ModelId, "model", created, "sharpi")]);
        ctx.Response.ContentType = "application/json";
        return ctx.Response.WriteAsync(
            JsonSerializer.Serialize(response, SharpInferenceJsonContext.Default.ModelsResponse),
            ctx.RequestAborted);
    }

    private static List<(string role, string content)> BuildMessageList(
        OaiMessage[] messages, string? responseFormatType, List<byte[]> images, bool preserveThinking)
    {
        var list = new List<(string, string)>(messages.Length + 1);
        if (responseFormatType == "json_object")
            list.Add(("system", "Respond with valid JSON only. Do not include any text outside the JSON object."));
        foreach (var m in messages)
        {
            var role = m.Role ?? "user";
            var content = FlattenContent(m.Content, images);
            if (role == "assistant")
                content = preserveThinking
                    ? ChatTemplate.InjectThinking(content, m.ReasoningContent)
                    : ChatTemplate.ScrubAssistantThinking(content);
            list.Add((role, content));
        }
        return list;
    }

    /// <summary>
    /// Flatten an OpenAI message <c>content</c> (a plain string, or an array of content parts)
    /// to text, decoding any <c>image_url</c> / <c>input_image</c> parts into
    /// <paramref name="images"/> (in document order) and replacing each with the model's
    /// <c>&lt;|image|&gt;</c> placeholder (issue #253). Only base64 data URLs are supported.
    /// </summary>
    private static string FlattenContent(JsonElement? contentEl, List<byte[]> images)
    {
        if (contentEl is null) return "";
        var el = contentEl.Value;
        if (el.ValueKind == JsonValueKind.String) return el.GetString() ?? "";
        if (el.ValueKind != JsonValueKind.Array) return "";

        var sb = new StringBuilder();
        foreach (var part in el.EnumerateArray())
        {
            // Content parts must be objects; a bare primitive would make TryGetProperty throw
            // (it requires an object receiver), so skip non-objects rather than 500.
            if (part.ValueKind != JsonValueKind.Object) continue;
            var type = part.TryGetProperty("type", out var t) ? t.GetString() : null;
            if (type == "text")
            {
                if (part.TryGetProperty("text", out var tv))
                    sb.Append(tv.GetString());
            }
            else if (type is "image_url" or "input_image")
            {
                // image_url is either {url:"data:..."} (chat completions) or a bare string url.
                string url = "";
                if (part.TryGetProperty("image_url", out var iu))
                    url = iu.ValueKind == JsonValueKind.String
                        ? iu.GetString() ?? ""
                        : iu.ValueKind == JsonValueKind.Object && iu.TryGetProperty("url", out var u) ? u.GetString() ?? "" : "";
                if (string.IsNullOrEmpty(url))
                    throw new ImageContentException($"{type} content part is missing a base64 'image_url'.");
                ImageContent.CheckCap(images.Count);
                images.Add(ImageContent.FromDataUrl(url));
                sb.Append(ImageContent.Placeholder);
            }
        }
        return sb.ToString();
    }

    /// <summary>True when OpenAI <c>tool_choice</c> is the string <c>"none"</c> (no tool use forced
    /// off) — the one value for which argument-grammar constraint must not engage.</summary>
    private static bool IsToolChoiceNone(JsonElement? toolChoice) =>
        toolChoice is { ValueKind: JsonValueKind.String } tc
        && string.Equals(tc.GetString(), "none", StringComparison.Ordinal);

    private static bool HasToolMessages(OaiMessage[]? messages)
    {
        if (messages is null) return false;
        foreach (var m in messages)
            if (m.Role == "tool" || m.ToolCalls is { Length: > 0 })
                return true;
        return false;
    }

    /// <summary>
    /// Builds rich message dictionaries and converts OpenAI tool definitions to the
    /// <c>{type:"function", function:{...}}</c> shape the Jinja chat template expects.
    /// Mirrors <see cref="AnthropicEndpoints"/>' BuildRichMessageList but for the
    /// OpenAI wire shape: tool definitions live at top level, assistant tool calls
    /// arrive as a <c>tool_calls</c> array, and tool results arrive as a separate
    /// <c>role:"tool"</c> message with <c>tool_call_id</c>.
    /// </summary>
    private static (List<Dictionary<string, object?>> messages, List<object?>? tools)
        BuildRichMessageList(ChatCompletionRequest req, IToolCallAdapter adapter, List<byte[]> images, bool preserveThinking)
    {
        var messages = new List<Dictionary<string, object?>>();

        foreach (var m in req.Messages!)
        {
            var role = m.Role ?? "user";
            var content = FlattenContent(m.Content, images);

            if (role == "tool")
            {
                messages.Add(adapter.RenderToolResult(m.ToolCallId ?? "", content));
                continue;
            }

            if (role == "assistant")
            {
                string textStr = preserveThinking
                    ? ChatTemplate.InjectThinking(content, m.ReasoningContent)
                    : ChatTemplate.ScrubAssistantThinking(content);
                var msg = new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["role"]    = "assistant",
                    ["content"] = textStr,
                };
                if (m.ToolCalls is { Length: > 0 })
                {
                    var toolCalls = new List<object?>();
                    foreach (var tc in m.ToolCalls)
                    {
                        // Source-gen JSON sets non-nullable record properties to null when the
                        // payload omits them — guard so malformed history doesn't NRE the request.
                        toolCalls.Add(new Dictionary<string, object?>(StringComparer.Ordinal)
                        {
                            ["id"]   = tc.Id,
                            ["type"] = tc.Type,
                            ["function"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                            {
                                ["name"]      = tc.Function?.Name,
                                // OpenAI stringifies arguments; pass through as-is.
                                ["arguments"] = tc.Function?.Arguments,
                            },
                        });
                    }
                    msg["tool_calls"] = (object?)toolCalls;
                }
                messages.Add(msg);
                continue;
            }

            // system / user / other roles
            messages.Add(new(StringComparer.Ordinal) { ["role"] = role, ["content"] = content });
        }

        List<object?>? toolsList = null;
        if (req.Tools is { Length: > 0 })
        {
            toolsList = req.Tools
                .Select(t => (object?)new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["type"] = t.Type,
                    ["function"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                    {
                        ["name"]        = t.Function?.Name,
                        ["description"] = t.Function?.Description,
                        ["parameters"]  = t.Function?.Parameters,
                    },
                })
                .ToList();
        }

        return (messages, toolsList);
    }

    private static async Task WriteEvent(HttpResponse response, string data)
    {
        await response.WriteAsync($"data: {data}\n\n", response.HttpContext.RequestAborted);
        await response.Body.FlushAsync(response.HttpContext.RequestAborted);
    }

    /// <summary>Shared priority order for the streaming and non-streaming handlers: a tool call
    /// in progress always wins, then any form of budget truncation, else a natural stop.</summary>
    private static string DetermineFinishReason(bool hasToolCalls, bool truncatedCall, bool truncatedByMaxTokens) =>
        hasToolCalls ? "tool_calls"
        : truncatedCall || truncatedByMaxTokens ? "length"
        : "stop";
}

// ── Request / Response types ──────────────────────────────────────────────────

public sealed record ChatCompletionRequest(
    string? Model,
    OaiMessage[]? Messages,
    int? MaxTokens,
    float? Temperature,
    float? TopP,
    bool? Stream,
    Dictionary<string, float>? LogitBias,
    ResponseFormat? ResponseFormat,
    [property: JsonPropertyName("enable_thinking")] bool? EnableThinking = null,
    [property: JsonPropertyName("max_thinking_tokens")] int? MaxThinkingTokens = null,
    OaiTool[]? Tools = null,
    [property: JsonPropertyName("tool_choice")] JsonElement? ToolChoice = null,
    // When true, prior assistant turns' reasoning_content is re-inlined as a leading <think>
    // block instead of being stripped (SharpInferenceServerOptions.PreserveThinking is the
    // server-wide default when this is absent). Off by default, matching the pre-existing
    // ChatTemplate.ScrubAssistantThinking behavior.
    [property: JsonPropertyName("preserve_thinking")] bool? PreserveThinking = null);

/// <summary>
/// Message in an OpenAI <c>/v1/chat/completions</c> request. Both single-string
/// <c>content</c> and structured fields are supported; an assistant message echoing
/// a prior tool call uses <see cref="ToolCalls"/> in place of (or alongside) text,
/// and a <c>role: "tool"</c> message carries <see cref="ToolCallId"/> + the result
/// text in <see cref="Content"/>.
/// </summary>
public sealed record OaiMessage(
    string? Role,
    // string for plain text, or an array of content parts ({type:"text"} / {type:"image_url"})
    // for multimodal requests (issue #253). JsonElement accepts both shapes via source-gen.
    JsonElement? Content,
    [property: JsonPropertyName("tool_call_id")] string? ToolCallId = null,
    [property: JsonPropertyName("tool_calls")] OaiToolCall[]? ToolCalls = null,
    string? Name = null,
    // Reasoning from a prior assistant turn, as echoed back by a client that captured it from
    // this same field on the response (see OaiAssistantMessage.ReasoningContent). Only consulted
    // when the request sets preserve_thinking; ChatTemplate.InjectThinking re-inlines it as a
    // leading <think> block ahead of Content.
    [property: JsonPropertyName("reasoning_content")] string? ReasoningContent = null);

/// <summary>
/// OpenAI tool definition. <c>function.parameters</c> is the JSON Schema; we keep it
/// as a raw <see cref="JsonElement"/> so it round-trips through the Jinja template unchanged.
/// <see cref="Function"/> is nullable because source-gen deserialization sets a missing
/// JSON object to <c>null</c> even on a non-nullable property; the endpoint guards on it.
/// </summary>
public sealed record OaiTool(string Type, OaiToolFunction? Function);

public sealed record OaiToolFunction(
    string Name,
    string? Description,
    JsonElement? Parameters);

/// <summary>
/// Single tool-call entry. Emitted on assistant messages (in responses) and accepted
/// on history-side assistant messages (in requests). OpenAI's spec uses a stringly
/// JSON-encoded <c>arguments</c> field.
/// </summary>
public sealed record OaiToolCall(
    string Id,
    string Type,
    OaiToolCallFunction? Function);

public sealed record OaiToolCallFunction(string Name, string Arguments);

public sealed record ChatCompletionResponse(
    string Id,
    string Object,
    long Created,
    string Model,
    CompletionChoice[] Choices,
    ChatUsage Usage);

public sealed record CompletionChoice(int Index, OaiAssistantMessage Message, string? FinishReason);
public sealed record OaiAssistantMessage(
    string Role,
    string? Content,
    [property: JsonPropertyName("reasoning_content")] string? ReasoningContent = null,
    [property: JsonPropertyName("tool_calls")] OaiToolCall[]? ToolCalls = null);
public sealed record ChatUsage(
    int PromptTokens,
    int CompletionTokens,
    int TotalTokens,
    [property: JsonPropertyName("completion_tokens_details")] CompletionTokensDetails? CompletionTokensDetails = null);
public sealed record CompletionTokensDetails(
    [property: JsonPropertyName("reasoning_tokens")] int ReasoningTokens);

public sealed record ChatCompletionChunk(
    string Id,
    string Object,
    long Created,
    string Model,
    ChunkChoice[] Choices);

public sealed record ChunkChoice(int Index, ChunkDelta Delta, string? FinishReason);
public sealed record ChunkDelta(
    string? Role,
    string? Content,
    [property: JsonPropertyName("reasoning_content")] string? ReasoningContent = null,
    [property: JsonPropertyName("tool_calls")] OaiToolCallDelta[]? ToolCalls = null);

/// <summary>
/// Per-chunk tool-call delta. OpenAI streams partial tool calls in array index order
/// — clients reconstruct each call by concatenating the <c>function.arguments</c> JSON
/// fragments across deltas sharing the same <see cref="Index"/>. We emit the full call
/// in a single delta (matching how the engine surfaces a complete <c>tool_use</c> block).
/// </summary>
public sealed record OaiToolCallDelta(
    int Index,
    string? Id,
    string? Type,
    OaiToolCallFunction? Function);

public sealed record ModelsResponse(string Object, ModelInfo[] Data);
public sealed record ModelInfo(string Id, string Object, long Created, string OwnedBy);

/// <summary>
/// OpenAI <c>response_format</c> (issue #423 follow-up). <c>type:"json_schema"</c> follows OpenAI's
/// own nested wire shape (<see cref="JsonSchema"/>.Schema); <c>type:"json_object"</c> additionally
/// accepts a flat <see cref="Schema"/> (llama.cpp's <c>response_format.schema</c> extension) for a
/// schema without the OpenAI envelope's <c>name</c>/<c>strict</c>. Either shape, when a schema is
/// present, constrains the whole response via <see cref="SharpInference.Core.Grammar.JsonSchemaOutputConstraint"/>.
/// </summary>
public sealed record ResponseFormat(string? Type, JsonSchemaSpec? JsonSchema = null, JsonElement? Schema = null);

/// <summary>OpenAI's <c>response_format.json_schema</c> envelope. Only <see cref="Schema"/> is
/// consumed; <see cref="Name"/>/<see cref="Strict"/> round-trip but aren't validated (matching
/// llama.cpp's own server behavior).</summary>
public sealed record JsonSchemaSpec(string? Name, JsonElement? Schema, bool? Strict = null);

public sealed record ErrorResponse(string Type, string Message);
