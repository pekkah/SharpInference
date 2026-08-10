using System.Text;
using System.Text.Json;
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

public static class AnthropicEndpoints
{
    public static IEndpointRouteBuilder MapAnthropicEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/messages", HandleMessages).WithConcurrencyLimit();
        return app;
    }

    private static async Task HandleMessages(
        HttpContext ctx,
        IInferenceEngine engine,
        ChatTemplateRenderer chatTemplate,
        ServerMetrics metrics,
        IOptions<SharpInferenceServerOptions> options)
    {
        var opts = options.Value;
        AnthropicMessageRequest? req;
        try
        {
            req = await ctx.Request.ReadFromJsonAsync(SharpInferenceJsonContext.Default.AnthropicMessageRequest, ctx.RequestAborted);
        }
        catch (JsonException ex)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new AErrorResponse("invalid_request_error", ex.Message),
                    SharpInferenceJsonContext.Default.AErrorResponse), ctx.RequestAborted);
            return;
        }
        catch (Microsoft.AspNetCore.Http.BadHttpRequestException ex)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new AErrorResponse("invalid_request_error", ex.Message),
                    SharpInferenceJsonContext.Default.AErrorResponse), ctx.RequestAborted);
            return;
        }

        if (req is null || req.Messages is null || req.Messages.Length == 0)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new AErrorResponse("invalid_request_error", "messages array is required"),
                    SharpInferenceJsonContext.Default.AErrorResponse), ctx.RequestAborted);
            return;
        }

        metrics.RecordRequest();

        // Anthropic-style thinking control: {"type":"disabled"} turns it off, {"type":"enabled"}
        // turns it on. When the field is absent, the default depends on the model family — Gemma 4
        // defaults reasoning off (see ChatTemplateRenderer.ModelDefaultsThinkingOff) while ChatML
        // models default on. An explicit value always wins over the family default. BudgetTokens,
        // when present, maps to SamplingParams.MaxThinkingTokens — the engine force-closes the
        // <think> block once that many reasoning tokens have streamed.
        // Server-level DisableThinking (SHARPI_NO_THINKING) forces reasoning off regardless of
        // the per-request thinking flag — for agentic clients (e.g. Claude Code) that never send
        // thinking:{"type":"disabled"}.
        bool? requestedThinking = req.Thinking?.Type switch
        {
            "enabled"  => true,
            "disabled" => false,
            _          => null,
        };
        bool enableThinking = (requestedThinking ?? !chatTemplate.ModelDefaultsThinkingOff)
                              && !options.Value.DisableThinking;

        // preserve_thinking (per-request) or SHARPI_PRESERVE_THINKING (server-wide) keeps prior
        // assistant turns' thinking content blocks in the rendered history instead of the default
        // ChatTemplate.ScrubAssistantThinking strip — see SharpInferenceServerOptions.PreserveThinking.
        // Server-level DisableThinking still wins: an operator who has decided this deployment never
        // exposes <think> content to the model shouldn't have historical reasoning re-injected either.
        bool preserveThinking = (req.PreserveThinking ?? options.Value.PreserveThinking)
                                && !options.Value.DisableThinking;

        var adapter = chatTemplate.ToolCallAdapter;

        // Image content blocks (issue #253) are collected (base64 → bytes) and replaced by the
        // model's <|image|> placeholder in the rendered prompt; the engine splices each image's
        // projected soft tokens at the matching placeholder during prefill.
        var images = new List<byte[]>();
        string prompt;
        // Issue #102: render history-only canonical alongside the generating prompt so the
        // engine can snapshot at the chat-template's history boundary and reuse KV across
        // multi-turn /v1/messages calls (the long-running pain point on agentic loops).
        string? canonicalHistoryPrefix;
        try
        {
            if (req.Tools is { Length: > 0 })
            {
                var (richMessages, tools) = BuildRichMessageList(req, adapter, images, preserveThinking);
                prompt = chatTemplate.Format(richMessages, enableThinking, tools);
                canonicalHistoryPrefix = chatTemplate.Format(richMessages, enableThinking, tools, addGenerationPrompt: false);
            }
            else
            {
                var messages = BuildMessageList(req, images, preserveThinking);
                prompt = chatTemplate.Format(messages, enableThinking);
                canonicalHistoryPrefix = chatTemplate.Format(messages, enableThinking, addGenerationPrompt: false);
            }
        }
        catch (ImageContentException ex)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new AErrorResponse("invalid_request_error", ex.Message),
                    SharpInferenceJsonContext.Default.AErrorResponse), ctx.RequestAborted);
            return;
        }
        catch (ChatTemplateException ex)
        {
            // See the matching handler in OpenAiEndpoints: a template that rejects its message
            // list is reporting bad input, not a server fault.
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new AErrorResponse("invalid_request_error", ex.Message),
                    SharpInferenceJsonContext.Default.AErrorResponse), ctx.RequestAborted);
            return;
        }

        if (images.Count > 0 && !engine.SupportsImageInput)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new AErrorResponse("invalid_request_error",
                        "This model is not configured for image input. Start the server with a Gemma 4 model and " +
                        "SHARPI_MMPROJ pointing at its gemma4uv mmproj, on a CPU (NGpuLayers=0) or full-CUDA " +
                        "(NGpuLayers=-1) backend."),
                    SharpInferenceJsonContext.Default.AErrorResponse), ctx.RequestAborted);
            return;
        }

        // Same guard as OpenAiEndpoints — only forward the hint if it's a string-prefix
        // of the generating prompt.
        if (canonicalHistoryPrefix is not null && !prompt.StartsWith(canonicalHistoryPrefix, StringComparison.Ordinal))
            canonicalHistoryPrefix = null;

        var sp = SamplingParamsBuilder.Build(opts,
            temperature: req.Temperature,
            topP:        req.TopP,
            topK:        req.TopK,
            repPenalty:  req.RepetitionPenalty,
            penaltyLastN: req.PenaltyLastN,
            maxTokens:   req.MaxTokens,
            maxThinking: req.Thinking?.BudgetTokens,
            thinkingDisabled: !enableThinking);

        // Tool-active turn: add the model family's tool-boundary stop (Gemma 4: <|tool_response>)
        // so generation halts when the tool calls finish instead of running on. Additive — keeps
        // the EOG set (issue #304). Gated on req.Tools to match THIS endpoint's tool-aware
        // rendering condition above (Anthropic only routes through the rich/tool path when tools
        // are declared); OpenAI also folds in tool messages in history, mirroring its wider gate.
        if (req.Tools is { Length: > 0 } && chatTemplate.ToolBoundaryStopTokenIds is { Count: > 0 } toolStops)
            sp = sp with { AdditionalStopTokenIds = [.. toolStops] };

        // Schema/grammar-constrained tool-argument decoding (issue #374): opt-in. Restricts the
        // sampler to tokens that keep the arguments schema-conformant in the model's native syntax.
        ITokenConstraint? toolConstraint = null;
        if (req.Tools is { Length: > 0 } && ToolGrammarHelper.Enabled(opts))
        {
            var schemas = ToolGrammarHelper.ToSchemas(req.Tools.Select(t => (t.Name, t.InputSchema)));
            toolConstraint = chatTemplate.BuildToolArgumentConstraint(schemas);
        }

        // Caller-supplied whole-turn output constraint (issue #423): independent of tool schemas,
        // AND-composed with the tool-argument constraint above rather than one overriding the other.
        var outputConstraint = opts.OutputConstraintFactory?.Invoke(ctx.RequestServices);
        if (TokenConstraints.Combine(outputConstraint, toolConstraint) is { } constraint)
            sp = sp with { Constraint = constraint };

        var msgId = $"msg_{Guid.NewGuid():N}";
        var modelId = engine.ModelId;

        if (req.Stream == true)
        {
            await HandleStreaming(ctx, engine, metrics, adapter, prompt, canonicalHistoryPrefix, images, sp, msgId, modelId);
        }
        else
        {
            await HandleNonStreaming(ctx, engine, metrics, adapter, prompt, canonicalHistoryPrefix, images, sp, msgId, modelId);
        }
    }

    private static async Task HandleNonStreaming(
        HttpContext ctx, IInferenceEngine engine, ServerMetrics metrics, IToolCallAdapter adapter,
        string prompt, string? canonicalHistoryPrefix, IReadOnlyList<byte[]> images, SamplingParams sp, string msgId, string modelId)
    {
        var thinkingSb = new StringBuilder();
        var textSb = new StringBuilder();
        int totalOutputTokens = 0;
        int promptTokens = 0;
        bool truncatedByMaxTokens = false;

        try
        {
            await foreach (var chunk in ImageContent.Generate(engine, prompt, canonicalHistoryPrefix, images, sp, ctx.RequestAborted))
            {
                if (chunk.Kind == GenerateChunkKind.Usage)
                {
                    promptTokens = chunk.PromptTokens;
                    continue;
                }
                if (chunk.Kind == GenerateChunkKind.Stop)
                {
                    truncatedByMaxTokens = chunk.TruncatedByMaxTokens;
                    continue;
                }
                totalOutputTokens++;
                if (chunk.Kind == GenerateChunkKind.Thinking)
                    thinkingSb.Append(chunk.Text);
                else
                    textSb.Append(chunk.Text);
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Residual generation error (e.g. image-count mismatch from a user-typed <|image|>, or a
            // projection failure) — return a structured error instead of a bare 500. The response
            // hasn't started on the non-streaming path. Common image errors are rejected at parse (400).
            ctx.Response.StatusCode = 500;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new AErrorResponse("internal_error", ex.Message),
                    SharpInferenceJsonContext.Default.AErrorResponse), ctx.RequestAborted);
            return;
        }

        metrics.RecordTokens(totalOutputTokens);

        var contentList = new List<AContent>();

        if (thinkingSb.Length > 0)
        {
            var thinking = thinkingSb.ToString();
            contentList.Add(new AContent("thinking", Thinking: thinking, Signature: MakeSignatureStub(thinking)));
        }

        var rawText = textSb.ToString();
        var (plainText, toolCalls, truncatedCall) = ParseToolCalls(adapter, rawText);

        if (plainText.Length > 0)
            contentList.Add(new AContent("text", Text: plainText));

        foreach (var (id, name, argsJson) in toolCalls)
        {
            JsonElement inputEl;
            try { using var d = JsonDocument.Parse(argsJson); inputEl = d.RootElement.Clone(); }
            catch (JsonException) { inputEl = EmptyJsonObject; }
            contentList.Add(new AContent("tool_use", Id: id, Name: name, Input: inputEl));
        }

        // Anthropic requires at least one content block.
        if (contentList.Count == 0)
            contentList.Add(new AContent("text", Text: ""));

        // A trailing unterminated tool call was surfaced as text (not a tool_use block); report
        // max_tokens so the client sees it was cut off. Mirrors the streaming path.
        var stopReason = DetermineStopReason(toolCalls.Count > 0, truncatedCall, truncatedByMaxTokens);

        var response = new AnthropicMessageResponse(
            msgId, "message", "assistant",
            contentList.ToArray(),
            modelId, stopReason,
            new AUsage(promptTokens, totalOutputTokens));

        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(
            JsonSerializer.Serialize(response, SharpInferenceJsonContext.Default.AnthropicMessageResponse),
            ctx.RequestAborted);
    }

    private static async Task HandleStreaming(
        HttpContext ctx, IInferenceEngine engine, ServerMetrics metrics, IToolCallAdapter adapter,
        string prompt, string? canonicalHistoryPrefix, IReadOnlyList<byte[]> images, SamplingParams sp, string msgId, string modelId)
    {
        ctx.Response.ContentType = "text/event-stream";
        ctx.Response.Headers.CacheControl = "no-cache";
        ctx.Response.Headers.Connection = "keep-alive";

        // message_start
        var startMsg = new AMessageStartEvent("message_start",
            new AMessageStartInner(msgId, "message", "assistant", modelId, "max_tokens", new AUsage(0, 0)));
        await WriteAnthropicEvent(ctx.Response, "message_start",
            JsonSerializer.Serialize(startMsg, SharpInferenceJsonContext.Default.AMessageStartEvent));

        // Per Anthropic's protocol, thinking (when present) is index 0 and text is index 1.
        // If no thinking chunks ever arrive, the text block takes index 0. Open each block
        // lazily — on its first chunk — so an all-text response still produces a single
        // block at index 0 (matching the pre-thinking wire shape).
        bool thinkingOpen = false;
        bool thinkingClosed = false;
        bool textOpen = false;
        int textIndex = 0;
        int nextBlockIndex = 0; // increments as blocks are opened
        var thinkingSb = new StringBuilder();
        int outputTokens = 0;
        int promptTokens = 0;
        bool hasToolCalls = false;

        // Tool-call streaming state machine — adapter-driven so the open/close markers
        // match the loaded model's wire format (Qwen3 <tool_call>, Qwen3-Coder <function=>,
        // Llama-3 <|python_tag|>, DeepSeek-R1 <|tool_calls_begin|>, ...).
        int maxOpenLen = adapter.MaxOpenTagLength;
        bool inToolCall = false;
        int toolCallContentStart = -1; // index into toolCallBuf where call content begins
        var toolCallBuf = new StringBuilder();
        string pendingText = "";

        async Task FlushTextDelta(string text)
        {
            if (text.Length == 0) return;
            // Once a tool_use block has been emitted, any trailing plain text is model control
            // noise (e.g. Gemma's <|tool_response> turn marker) — not assistant content — so it
            // must not open a stray text block (issue #150). Preamble before the first call is
            // unaffected (hasToolCalls is still false then). Mirrors the OpenAI streaming guard.
            if (hasToolCalls) return;
            if (!textOpen)
            {
                // Close thinking block first if it was opened without a follow-up text block.
                if (thinkingOpen && !thinkingClosed)
                {
                    var sigDelta = new AContentBlockDeltaEvent("content_block_delta", 0,
                        new AContentDelta("signature_delta", Signature: MakeSignatureStub(thinkingSb.ToString())));
                    await WriteAnthropicEvent(ctx.Response, "content_block_delta",
                        JsonSerializer.Serialize(sigDelta, SharpInferenceJsonContext.Default.AContentBlockDeltaEvent));
                    thinkingClosed = true;
                    await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                        JsonSerializer.Serialize(new AContentBlockStopEvent("content_block_stop", 0),
                            SharpInferenceJsonContext.Default.AContentBlockStopEvent));
                    nextBlockIndex = 1;
                }

                textIndex = nextBlockIndex;
                var textStart = new AContentBlockStartEvent("content_block_start", textIndex,
                    new AContentBlock("text", Text: ""));
                await WriteAnthropicEvent(ctx.Response, "content_block_start",
                    JsonSerializer.Serialize(textStart, SharpInferenceJsonContext.Default.AContentBlockStartEvent));
                textOpen = true;
            }

            var delta = new AContentBlockDeltaEvent("content_block_delta", textIndex,
                new AContentDelta("text_delta", Text: text));
            await WriteAnthropicEvent(ctx.Response, "content_block_delta",
                JsonSerializer.Serialize(delta, SharpInferenceJsonContext.Default.AContentBlockDeltaEvent));
        }

        async Task EmitToolCallsFromBlock(string blockContent)
        {
            var calls = adapter.ParseBlock(blockContent);
            foreach (var tc in calls)
            {
                string name    = tc.Name;
                string argsJson = JinjaChatTemplate.SerializeToJson(tc.Arguments);

                // Close any open text block at the first tool call.
                if (textOpen)
                {
                    await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                        JsonSerializer.Serialize(new AContentBlockStopEvent("content_block_stop", textIndex),
                            SharpInferenceJsonContext.Default.AContentBlockStopEvent));
                    textOpen = false;
                    nextBlockIndex = textIndex + 1;
                }

                var id = $"toolu_{Guid.NewGuid():N}";
                int toolIdx = nextBlockIndex++;
                hasToolCalls = true;

                var toolStart = new AContentBlockStartEvent("content_block_start", toolIdx,
                    new AContentBlock("tool_use", Id: id, Name: name, Input: EmptyJsonObject));
                await WriteAnthropicEvent(ctx.Response, "content_block_start",
                    JsonSerializer.Serialize(toolStart, SharpInferenceJsonContext.Default.AContentBlockStartEvent));

                var inputDelta = new AContentBlockDeltaEvent("content_block_delta", toolIdx,
                    new AContentDelta("input_json_delta", PartialJson: argsJson));
                await WriteAnthropicEvent(ctx.Response, "content_block_delta",
                    JsonSerializer.Serialize(inputDelta, SharpInferenceJsonContext.Default.AContentBlockDeltaEvent));

                await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                    JsonSerializer.Serialize(new AContentBlockStopEvent("content_block_stop", toolIdx),
                        SharpInferenceJsonContext.Default.AContentBlockStopEvent));
            }
        }

        // Process a text-stream chunk through the tool-call detection state machine.
        // Adapter-driven so a different model family (e.g. Qwen3-Coder, Llama-3) plugs
        // its own open/close marker scan in without changes here.
        async Task ProcessTextChunk(string chunk)
        {
            if (inToolCall)
            {
                toolCallBuf.Append(chunk);
                string buf = toolCallBuf.ToString();
                int closeIdx = adapter.FindCloseMarker(buf, toolCallContentStart, out int afterClose);
                if (closeIdx >= 0)
                {
                    // Block content is whatever lies between the open marker's contentStart
                    // and the close marker's start index. afterClose points past the marker.
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
                    await FlushTextDelta(pendingText[..openIdx]);

                // Capture whatever the adapter considers "block content" — for most adapters
                // that's everything past the open marker, but Qwen3-Coder's name lives inside
                // the open marker so its contentStart points AT the marker, not past it.
                inToolCall = true;
                toolCallBuf.Clear();
                toolCallBuf.Append(pendingText, contentStart, pendingText.Length - contentStart);
                toolCallContentStart = 0;
                pendingText = "";

                // The newly buffered region may itself already contain a complete close marker
                // (e.g. when a single chunk delivered the whole call). Re-enter to check.
                if (toolCallBuf.Length > 0)
                    await ProcessTextChunk("");
                return;
            }

            // No open marker found: flush everything except the last (maxOpenLen - 1) chars
            // which might be the start of a partial marker match.
            int safeLen = Math.Max(0, pendingText.Length - (maxOpenLen - 1));
            if (safeLen > 0)
            {
                await FlushTextDelta(pendingText[..safeLen]);
                pendingText = pendingText[safeLen..];
            }
        }

        bool truncatedToolCall = false;
        bool truncatedByMaxTokens = false;
        try
        {
            await foreach (var chunk in ImageContent.Generate(engine, prompt, canonicalHistoryPrefix, images, sp, ctx.RequestAborted))
            {
                // Out-of-band usage chunk (issue #150): not a generated token. Capture the
                // prompt-token count for the terminal message_delta usage and skip the rest of
                // the per-token handling.
                if (chunk.Kind == GenerateChunkKind.Usage)
                {
                    promptTokens = chunk.PromptTokens;
                    continue;
                }
                if (chunk.Kind == GenerateChunkKind.Stop)
                {
                    truncatedByMaxTokens = chunk.TruncatedByMaxTokens;
                    continue;
                }
                outputTokens++;

                if (chunk.Kind == GenerateChunkKind.Thinking)
                {
                    if (textOpen) continue; // discard stray thinking after text has started

                    if (!thinkingOpen)
                    {
                        var thinkingStart = new AContentBlockStartEvent("content_block_start", 0,
                            new AContentBlock("thinking", Thinking: ""));
                        await WriteAnthropicEvent(ctx.Response, "content_block_start",
                            JsonSerializer.Serialize(thinkingStart, SharpInferenceJsonContext.Default.AContentBlockStartEvent));
                        thinkingOpen = true;
                        nextBlockIndex = 1;
                    }

                    thinkingSb.Append(chunk.Text);
                    var delta = new AContentBlockDeltaEvent("content_block_delta", 0,
                        new AContentDelta("thinking_delta", Thinking: chunk.Text));
                    await WriteAnthropicEvent(ctx.Response, "content_block_delta",
                        JsonSerializer.Serialize(delta, SharpInferenceJsonContext.Default.AContentBlockDeltaEvent));
                }
                else // Text (may contain <tool_call> tags)
                {
                    await ProcessTextChunk(chunk.Text);
                }
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // SSE response already committed (message_start sent) — status can't change. Swallow so
            // the finally + message_delta/message_stop terminate the stream cleanly instead of the
            // TCP connection resetting mid-stream. Common image errors are rejected at parse (400).
            // Log it — otherwise a genuine mid-stream failure is invisible (the client sees a clean
            // end_turn with truncated text).
            ctx.RequestServices.GetService<ILoggerFactory>()?
                .CreateLogger("SharpInference.Server.Endpoints")
                .LogError(ex, "Generation failed after the streaming response was committed; the SSE stream is terminated without an error frame.");
        }
        finally
        {
            // Flush any remaining buffered text (cannot contain a partial tool_call open tag).
            if (!inToolCall && pendingText.Length > 0)
            {
                try { await FlushTextDelta(pendingText); }
                catch (OperationCanceledException) { /* client disconnected */ }
                catch (IOException) { /* response stream closed */ }
                pendingText = "";
            }

            // Stream ended mid-tool-call: surface the buffered bytes so the client sees the
            // truncated output (rather than an empty assistant turn) and signal max_tokens.
            if (inToolCall && toolCallBuf.Length > 0)
            {
                truncatedToolCall = true;
                try { await FlushTextDelta(toolCallBuf.ToString()); }
                catch (OperationCanceledException) { /* client disconnected */ }
                catch (IOException) { /* response stream closed */ }
                toolCallBuf.Clear();
            }

            // Close thinking block if it was opened but never got a following text block.
            if (thinkingOpen && !thinkingClosed)
            {
                try
                {
                    var sigDelta = new AContentBlockDeltaEvent("content_block_delta", 0,
                        new AContentDelta("signature_delta", Signature: MakeSignatureStub(thinkingSb.ToString())));
                    await WriteAnthropicEvent(ctx.Response, "content_block_delta",
                        JsonSerializer.Serialize(sigDelta, SharpInferenceJsonContext.Default.AContentBlockDeltaEvent));
                    await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                        JsonSerializer.Serialize(new AContentBlockStopEvent("content_block_stop", 0),
                            SharpInferenceJsonContext.Default.AContentBlockStopEvent));
                }
                catch (OperationCanceledException) { /* client disconnected */ }
                catch (IOException) { /* response stream closed */ }
            }
            if (textOpen)
            {
                try
                {
                    await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                        JsonSerializer.Serialize(new AContentBlockStopEvent("content_block_stop", textIndex),
                            SharpInferenceJsonContext.Default.AContentBlockStopEvent));
                }
                catch (OperationCanceledException) { /* client disconnected */ }
                catch (IOException) { /* response stream closed */ }
            }
        }

        // Anthropic responses always include a terminal text block, even if empty.
        // Cover cases: (a) no chunks at all → empty text at index 0;
        //              (b) thinking-only (model hit budget) → empty text at nextBlockIndex.
        if (!textOpen && !hasToolCalls)
        {
            try
            {
                int idx = nextBlockIndex;
                var textStart = new AContentBlockStartEvent("content_block_start", idx,
                    new AContentBlock("text", Text: ""));
                await WriteAnthropicEvent(ctx.Response, "content_block_start",
                    JsonSerializer.Serialize(textStart, SharpInferenceJsonContext.Default.AContentBlockStartEvent));
                await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                    JsonSerializer.Serialize(new AContentBlockStopEvent("content_block_stop", idx),
                        SharpInferenceJsonContext.Default.AContentBlockStopEvent));
            }
            catch (OperationCanceledException) { /* client disconnected */ }
            catch (IOException) { /* response stream closed */ }
        }

        // message_delta
        var stopReason = DetermineStopReason(hasToolCalls, truncatedToolCall, truncatedByMaxTokens);
        var msgDelta = new AMessageDeltaEvent("message_delta",
            new AMessageDelta(stopReason, null), new AUsage(promptTokens, outputTokens));
        await WriteAnthropicEvent(ctx.Response, "message_delta",
            JsonSerializer.Serialize(msgDelta, SharpInferenceJsonContext.Default.AMessageDeltaEvent));

        // message_stop
        await WriteAnthropicEvent(ctx.Response, "message_stop",
            JsonSerializer.Serialize(new ATypeOnly("message_stop"), SharpInferenceJsonContext.Default.ATypeOnly));

        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        metrics.RecordTokens(outputTokens);
    }

    /// <summary>
    /// Produces a deterministic placeholder for the Anthropic <c>signature</c> field on
    /// thinking blocks. The real API uses an HMAC so clients can prove a thinking block
    /// wasn't tampered with before being echoed back in a follow-up turn. We don't do
    /// round-trip validation yet, so a stable but non-cryptographic stub is enough —
    /// the field's presence is what most clients check. Bounded length keeps the wire
    /// payload small for long reasoning traces.
    /// </summary>
    private static string MakeSignatureStub(string thinking)
    {
        // Stable hash → base64 → truncate. Deterministic in-process; not portable across runs
        // because string.GetHashCode is randomized per process — that's fine for an opaque token.
        var bytes = Encoding.UTF8.GetBytes($"sharpi-v1:{thinking.Length}:{thinking.GetHashCode()}");
        var b64 = Convert.ToBase64String(bytes);
        return b64.Length > 32 ? b64[..32] : b64;
    }

    /// <summary>Shared priority order for the streaming and non-streaming handlers: a tool call
    /// in progress always wins, then any form of budget truncation, else a natural end_turn.</summary>
    private static string DetermineStopReason(bool hasToolCalls, bool truncatedCall, bool truncatedByMaxTokens) =>
        hasToolCalls ? "tool_use"
        : truncatedCall || truncatedByMaxTokens ? "max_tokens"
        : "end_turn";

    /// <summary>Pre-built empty JSON object used for the <c>input</c> field of tool_use blocks.</summary>
    private static readonly JsonElement EmptyJsonObject = JsonDocument.Parse("{}").RootElement.Clone();

    /// <summary>
    /// Extracts tool calls from raw model output via the adapter for the loaded model.
    /// Returns the plain text (with tool-call blocks stripped) and a list of parsed
    /// calls tagged with Anthropic-style <c>toolu_</c> identifiers.
    /// </summary>
    private static (string text, List<(string id, string name, string argsJson)> toolCalls, bool truncated)
        ParseToolCalls(IToolCallAdapter adapter, string output)
    {
        var pr = adapter.Parse(output);
        var toolCalls = pr.Calls
            .Select(c => ($"toolu_{Guid.NewGuid():N}", c.Name, JinjaChatTemplate.SerializeToJson(c.Arguments)))
            .ToList();
        return (pr.PlainText, toolCalls, pr.Truncated);
    }

    /// <summary>
    /// Builds the simple (string-content) message list used when no tools are present.
    /// Handles both plain-string and array-of-content-blocks message formats.
    /// </summary>
    private static List<(string role, string content)> BuildMessageList(
        AnthropicMessageRequest req, List<byte[]> images, bool preserveThinking)
    {
        var list = new List<(string, string)>();
        var systemText = ExtractTextContent(req.System);
        if (!string.IsNullOrEmpty(systemText))
            list.Add(("system", systemText));
        foreach (var m in req.Messages!)
        {
            var role = m.Role ?? "user";
            if (role == "assistant")
            {
                var (text, thinking) = ExtractAssistantTextAndThinking(m.Content, images);
                var content = preserveThinking
                    ? ChatTemplate.InjectThinking(text, thinking)
                    : ChatTemplate.ScrubAssistantThinking(text);
                list.Add((role, content));
            }
            else
            {
                list.Add((role, ExtractTextAndImages(m.Content, images)));
            }
        }
        return list;
    }

    /// <summary>
    /// Like <see cref="ExtractTextAndImages"/> but also pulls out <c>thinking</c> content blocks
    /// separately from <c>text</c> — Anthropic carries an assistant turn's reasoning as its own
    /// block type, not inlined into the text, so the two channels have to be split before
    /// <see cref="ChatTemplate.ScrubAssistantThinking"/> / <see cref="ChatTemplate.InjectThinking"/>
    /// can decide what happens to the reasoning independently of the answer text.
    /// </summary>
    private static (string text, string? thinking) ExtractAssistantTextAndThinking(
        JsonElement? contentEl, List<byte[]> images)
    {
        if (contentEl is null) return ("", null);
        var el = contentEl.Value;
        if (el.ValueKind == JsonValueKind.String) return (el.GetString() ?? "", null);
        if (el.ValueKind != JsonValueKind.Array) return ("", null);

        var textSb = new StringBuilder();
        var thinkingSb = new StringBuilder();
        foreach (var block in el.EnumerateArray())
        {
            if (block.ValueKind != JsonValueKind.Object) continue;
            var type = block.TryGetProperty("type", out var t) ? t.GetString() : null;
            if (type == "text")
            {
                if (block.TryGetProperty("text", out var tv))
                    textSb.Append(tv.GetString());
            }
            else if (type == "thinking")
            {
                if (block.TryGetProperty("thinking", out var tk))
                    thinkingSb.Append(tk.GetString());
            }
            else if (type == "image")
            {
                ImageContent.CheckCap(images.Count);
                images.Add(ParseAnthropicImageBlock(block));
                textSb.Append(ImageContent.Placeholder);
            }
        }
        return (textSb.ToString(), thinkingSb.Length > 0 ? thinkingSb.ToString() : null);
    }

    /// <summary>
    /// Like <see cref="ExtractTextContent"/> but also handles Anthropic <c>image</c> content
    /// blocks (issue #253): each base64 <c>source</c> is decoded into <paramref name="images"/>
    /// (in document order) and replaced by the model's <c>&lt;|image|&gt;</c> placeholder so the
    /// engine can splice its soft tokens at the matching position.
    /// </summary>
    private static string ExtractTextAndImages(JsonElement? contentEl, List<byte[]> images)
    {
        if (contentEl is null) return "";
        var el = contentEl.Value;
        if (el.ValueKind == JsonValueKind.String) return el.GetString() ?? "";
        if (el.ValueKind != JsonValueKind.Array) return "";

        var sb = new StringBuilder();
        foreach (var block in el.EnumerateArray())
        {
            // A content array may legally hold only objects; a bare primitive element would make
            // TryGetProperty throw (it requires an object receiver), so skip non-objects rather
            // than 500. The model just sees the surrounding text.
            if (block.ValueKind != JsonValueKind.Object) continue;
            var type = block.TryGetProperty("type", out var t) ? t.GetString() : null;
            if (type == "text")
            {
                if (block.TryGetProperty("text", out var tv))
                    sb.Append(tv.GetString());
            }
            else if (type == "image")
            {
                ImageContent.CheckCap(images.Count);
                images.Add(ParseAnthropicImageBlock(block));
                sb.Append(ImageContent.Placeholder);
            }
        }
        return sb.ToString();
    }

    private static byte[] ParseAnthropicImageBlock(JsonElement block)
    {
        if (!block.TryGetProperty("source", out var source) || source.ValueKind != JsonValueKind.Object)
            throw new ImageContentException("image content block is missing an object 'source'.");
        var srcType = source.TryGetProperty("type", out var st) ? st.GetString() : null;
        if (srcType != "base64")
            throw new ImageContentException(
                $"unsupported image source type '{srcType}'; only base64 image sources are supported (URL fetch is not).");
        if (!source.TryGetProperty("data", out var data) || data.ValueKind != JsonValueKind.String)
            throw new ImageContentException("image source is missing base64 'data'.");
        return ImageContent.FromBase64(data.GetString() ?? "");
    }

    /// <summary>
    /// Builds rich message dictionaries and converts Anthropic tool definitions to
    /// the OpenAI/Qwen3 function format expected by the Jinja chat template.
    /// Handles multi-content messages: tool_use content blocks become <c>tool_calls</c>
    /// entries on assistant messages; tool_result blocks become the role/content shape
    /// the adapter specifies (defaults to role="tool").
    /// </summary>
    private static (List<Dictionary<string, object?>> messages, List<object?>? tools)
        BuildRichMessageList(AnthropicMessageRequest req, IToolCallAdapter adapter, List<byte[]> images, bool preserveThinking)
    {
        var messages = new List<Dictionary<string, object?>>();

        var systemText = ExtractTextContent(req.System);
        if (!string.IsNullOrEmpty(systemText))
            messages.Add(new() { ["role"] = "system", ["content"] = systemText });

        foreach (var m in req.Messages!)
        {
            var role = m.Role ?? "user";

            if (m.Content is null || m.Content.Value.ValueKind == JsonValueKind.Undefined)
            {
                messages.Add(new() { ["role"] = role, ["content"] = "" });
                continue;
            }

            var contentEl = m.Content.Value;

            if (contentEl.ValueKind == JsonValueKind.String)
            {
                var str = contentEl.GetString() ?? "";
                if (role == "assistant")
                    str = preserveThinking ? str : ChatTemplate.ScrubAssistantThinking(str);
                messages.Add(new() { ["role"] = role, ["content"] = str });
            }
            else if (contentEl.ValueKind == JsonValueKind.Array)
            {
                if (role == "assistant")
                {
                    var textSb = new StringBuilder();
                    var thinkingSb = new StringBuilder();
                    var toolCalls = new List<object?>();

                    foreach (var block in contentEl.EnumerateArray())
                    {
                        if (block.ValueKind != JsonValueKind.Object) continue;
                        var type = block.TryGetProperty("type", out var t) ? t.GetString() : null;
                        if (type == "text")
                        {
                            if (block.TryGetProperty("text", out var tv))
                                textSb.Append(tv.GetString());
                        }
                        else if (type == "thinking")
                        {
                            if (block.TryGetProperty("thinking", out var tk))
                                thinkingSb.Append(tk.GetString());
                        }
                        else if (type == "tool_use")
                        {
                            var name = block.TryGetProperty("name", out var n) ? n.GetString() : null;
                            var argsJson = block.TryGetProperty("input", out var inp)
                                ? inp.GetRawText()
                                : "{}";
                            // Template checks tool_call.arguments is string → output directly.
                            toolCalls.Add(new Dictionary<string, object?>
                            {
                                ["name"]      = name,
                                ["arguments"] = argsJson,
                            });
                        }
                    }

                    string textStr = preserveThinking
                        ? ChatTemplate.InjectThinking(textSb.ToString(), thinkingSb.Length > 0 ? thinkingSb.ToString() : null)
                        : ChatTemplate.ScrubAssistantThinking(textSb.ToString());
                    var msg = new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = textStr };
                    if (toolCalls.Count > 0) msg["tool_calls"] = (object?)toolCalls;
                    messages.Add(msg);
                }
                else // user message — may contain tool_result blocks
                {
                    var textSb = new StringBuilder();

                    foreach (var block in contentEl.EnumerateArray())
                    {
                        if (block.ValueKind != JsonValueKind.Object) continue;
                        var type = block.TryGetProperty("type", out var t) ? t.GetString() : null;
                        if (type == "text")
                        {
                            if (block.TryGetProperty("text", out var tv))
                                textSb.Append(tv.GetString());
                        }
                        else if (type == "image")
                        {
                            // Issue #253: decode + collect the image, emit a placeholder inline.
                            ImageContent.CheckCap(images.Count);
                            images.Add(ParseAnthropicImageBlock(block));
                            textSb.Append(ImageContent.Placeholder);
                        }
                        else if (type == "tool_result")
                        {
                            // tool_result.content can be a string or an array of text blocks.
                            string resultContent = "";
                            if (block.TryGetProperty("content", out var c))
                            {
                                resultContent = c.ValueKind == JsonValueKind.String
                                    ? c.GetString() ?? ""
                                    : ExtractTextFromArray(c);
                            }
                            string toolUseId = block.TryGetProperty("tool_use_id", out var tid)
                                ? tid.GetString() ?? ""
                                : "";
                            messages.Add(adapter.RenderToolResult(toolUseId, resultContent));
                        }
                    }

                    // Any accompanying plain text goes as a separate user message after tool results.
                    if (textSb.Length > 0)
                        messages.Add(new() { ["role"] = "user", ["content"] = textSb.ToString() });
                }
            }
        }

        // Convert Anthropic tool definitions → OpenAI/Qwen3 function format.
        List<object?>? toolsList = null;
        if (req.Tools is { Length: > 0 })
        {
            toolsList = req.Tools
                .Select(tool => (object?)new Dictionary<string, object?>
                {
                    ["type"] = "function",
                    ["function"] = new Dictionary<string, object?>
                    {
                        ["name"]        = tool.Name,
                        ["description"] = tool.Description,
                        ["parameters"]  = tool.InputSchema,  // JsonElement — rendered via ToJson in template
                    },
                })
                .ToList();
        }

        return (messages, toolsList);
    }

    private static string? ExtractTextContent(JsonElement? contentEl)
    {
        if (contentEl is null) return null;
        var el = contentEl.Value;
        return el.ValueKind switch
        {
            JsonValueKind.String => el.GetString(),
            JsonValueKind.Array  => ExtractTextFromArray(el),
            _                    => null,
        };
    }

    private static string ExtractTextFromArray(JsonElement arr)
    {
        var sb = new StringBuilder();
        foreach (var block in arr.EnumerateArray())
        {
            if (block.TryGetProperty("type", out var t) && t.GetString() == "text"
                && block.TryGetProperty("text", out var text))
                sb.Append(text.GetString());
        }
        return sb.ToString();
    }

    private static async Task WriteAnthropicEvent(HttpResponse response, string eventType, string data)
    {
        await response.WriteAsync($"event: {eventType}\ndata: {data}\n\n", response.HttpContext.RequestAborted);
        await response.Body.FlushAsync(response.HttpContext.RequestAborted);
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

public sealed record AnthropicMessageRequest(
    string? Model,
    AnthropicMessage[]? Messages,
    int MaxTokens,
    // Anthropic allows `system` to be either a plain string or an array of text content
    // blocks (with cache_control) — Claude Code sends the array form. Accept both as a raw
    // JsonElement and flatten via ExtractTextContent.
    JsonElement? System,
    bool? Stream,
    float? Temperature,
    float? TopP,
    int? TopK,
    AnthropicThinking? Thinking = null,
    AnthropicTool[]? Tools = null,
    // When true, a prior assistant turn's `thinking` content block is re-inlined as a leading
    // <think> block instead of being dropped (SharpInferenceServerOptions.PreserveThinking is
    // the server-wide default when this is absent). Off by default, matching the pre-existing
    // ChatTemplate.ScrubAssistantThinking behavior — and the pre-existing behavior of silently
    // dropping `thinking` blocks entirely on the tool-active path. Maps to the wire field
    // `preserve_thinking` via the server's snake_case JSON naming policy.
    bool? PreserveThinking = null,
    // Repetition penalty and the window all three penalties look back over. Not part of
    // Anthropic's schema — an extension mirroring the OpenAI surface, previously reachable only
    // as a host-level default. 1.0 disables the penalty; PenaltyLastN 0 means the whole request.
    // Absent = the host default. Wire fields `repetition_penalty` / `penalty_last_n` via the
    // snake_case naming policy.
    float? RepetitionPenalty = null,
    int? PenaltyLastN = null);

public sealed record AnthropicThinking(string? Type, int? BudgetTokens);

/// <summary>
/// A single tool definition in an Anthropic /v1/messages request.
/// <c>InputSchema</c> is the JSON Schema describing the tool's parameters
/// (equivalent to OpenAI's <c>parameters</c> field) — passed as a raw
/// <see cref="JsonElement"/> so it round-trips through the Jinja template unchanged.
/// </summary>
public sealed record AnthropicTool(
    string? Name,
    string? Description,
    JsonElement? InputSchema);

/// <summary>
/// A single message in an Anthropic /v1/messages request.
/// <c>Content</c> may be either a plain string or an array of typed content blocks
/// (e.g. <c>text</c>, <c>tool_use</c>, <c>tool_result</c>).
/// Using <see cref="JsonElement"/> lets us handle both shapes without a custom converter.
/// </summary>
public sealed record AnthropicMessage(string? Role, JsonElement? Content);

public sealed record AnthropicMessageResponse(
    string Id,
    string Type,
    string Role,
    AContent[] Content,
    string Model,
    string StopReason,
    AUsage Usage);

/// <summary>
/// Heterogeneous content block — covers <c>text</c>, <c>thinking</c>, and <c>tool_use</c>
/// shapes via optional fields. Null-valued fields are stripped from JSON output (see
/// <see cref="SharpInferenceJsonContext"/>'s <c>WhenWritingNull</c> setting), so the wire format is
/// byte-identical to Anthropic's per-type shapes. A single record keeps the response
/// array NativeAOT-friendly (no polymorphic serialization needed).
/// </summary>
public sealed record AContent(
    string Type,
    string? Text = null,
    string? Thinking = null,
    string? Signature = null,
    string? Id = null,
    string? Name = null,
    JsonElement? Input = null);

public sealed record AUsage(int InputTokens, int OutputTokens);

// Streaming event types
public sealed record AMessageStartEvent(string Type, AMessageStartInner Message);
public sealed record AMessageStartInner(string Id, string Type, string Role, string Model, string StopReason, AUsage Usage);
public sealed record AContentBlockStartEvent(string Type, int Index, AContentBlock ContentBlock);

/// <summary>
/// Content-block envelope emitted on <c>content_block_start</c>. Covers <c>text</c>,
/// <c>thinking</c>, and <c>tool_use</c> starts. Null fields are omitted via
/// <c>WhenWritingNull</c> so each block type emits only its relevant fields.
/// </summary>
public sealed record AContentBlock(
    string Type,
    string? Text = null,
    string? Thinking = null,
    string? Id = null,
    string? Name = null,
    JsonElement? Input = null);

public sealed record AContentBlockDeltaEvent(string Type, int Index, AContentDelta Delta);

/// <summary>
/// Streaming delta envelope. Covers <c>text_delta</c>, <c>thinking_delta</c>,
/// <c>signature_delta</c>, and <c>input_json_delta</c> via four optional payload fields.
/// Only the field matching <see cref="Type"/> is populated; the rest are null and omitted.
/// </summary>
public sealed record AContentDelta(
    string Type,
    string? Text = null,
    string? Thinking = null,
    string? Signature = null,
    string? PartialJson = null);

public sealed record AContentBlockStopEvent(string Type, int Index);
public sealed record AMessageDeltaEvent(string Type, AMessageDelta Delta, AUsage Usage);
public sealed record AMessageDelta(string StopReason, string? StopSequence);
public sealed record ATypeOnly(string Type);
public sealed record AErrorResponse(string Type, string Message);
