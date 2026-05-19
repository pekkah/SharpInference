using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using SharpInference.Engine;

namespace SharpInference.Server.Endpoints;

public static class AnthropicEndpoints
{
    public static IEndpointRouteBuilder MapAnthropicEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/messages", HandleMessages);
        return app;
    }

    private static async Task HandleMessages(
        HttpContext ctx,
        IInferenceEngine engine)
    {
        AnthropicMessageRequest? req;
        try
        {
            req = await ctx.Request.ReadFromJsonAsync(AppJsonContext.Default.AnthropicMessageRequest, ctx.RequestAborted);
        }
        catch
        {
            ctx.Response.StatusCode = 400;
            return;
        }

        if (req is null || req.Messages is null || req.Messages.Length == 0)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new AErrorResponse("invalid_request_error", "messages array is required"),
                    AppJsonContext.Default.AErrorResponse), ctx.RequestAborted);
            return;
        }

        HealthEndpoints.RecordRequest();

        var modelArch = Environment.GetEnvironmentVariable("SHARPI_ARCH") ?? "qwen2";
        // Anthropic-style thinking control: {"type":"disabled"} turns it off; absence or any
        // other value (including {"type":"enabled"}) leaves it on. BudgetTokens is accepted on
        // the wire but currently advisory — SamplingParams does not yet enforce a thinking-token
        // ceiling. When that engine-side knob lands, plumb req.Thinking.BudgetTokens into it.
        bool enableThinking = req.Thinking?.Type != "disabled";
        var messages = BuildMessageList(req);
        var prompt = ChatTemplate.Format(messages, modelArch, enableThinking);

        var sp = new SamplingParams
        {
            Temperature = req.Temperature ?? 1.0f,
            TopP = req.TopP ?? 1.0f,
            MaxNewTokens = req.MaxTokens,
        };

        var msgId = $"msg_{Guid.NewGuid():N}";
        var modelId = engine.ModelId;

        if (req.Stream == true)
        {
            await HandleStreaming(ctx, engine, prompt, sp, msgId, modelId);
        }
        else
        {
            await HandleNonStreaming(ctx, engine, prompt, sp, msgId, modelId);
        }
    }

    private static async Task HandleNonStreaming(
        HttpContext ctx, IInferenceEngine engine, string prompt, SamplingParams sp,
        string msgId, string modelId)
    {
        var thinkingSb = new StringBuilder();
        var textSb = new StringBuilder();
        int totalOutputTokens = 0;

        await foreach (var chunk in engine.GenerateChunksAsync(prompt, sp, ctx.RequestAborted))
        {
            totalOutputTokens++;
            if (chunk.Kind == GenerateChunkKind.Thinking)
                thinkingSb.Append(chunk.Text);
            else
                textSb.Append(chunk.Text);
        }

        HealthEndpoints.RecordTokens(totalOutputTokens);

        AContent[] content;
        if (thinkingSb.Length > 0)
        {
            var thinking = thinkingSb.ToString();
            content =
            [
                new AContent("thinking", Text: null, Thinking: thinking, Signature: MakeSignatureStub(thinking)),
                new AContent("text", Text: textSb.ToString()),
            ];
        }
        else
        {
            content = [new AContent("text", Text: textSb.ToString())];
        }

        var response = new AnthropicMessageResponse(
            msgId, "message", "assistant",
            content,
            modelId, "end_turn",
            new AUsage(0, totalOutputTokens));

        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(
            JsonSerializer.Serialize(response, AppJsonContext.Default.AnthropicMessageResponse),
            ctx.RequestAborted);
    }

    private static async Task HandleStreaming(
        HttpContext ctx, IInferenceEngine engine, string prompt, SamplingParams sp,
        string msgId, string modelId)
    {
        ctx.Response.ContentType = "text/event-stream";
        ctx.Response.Headers.CacheControl = "no-cache";
        ctx.Response.Headers.Connection = "keep-alive";

        // message_start
        var startMsg = new AMessageStartEvent("message_start",
            new AMessageStartInner(msgId, "message", "assistant", modelId, "max_tokens", new AUsage(0, 0)));
        await WriteAnthropicEvent(ctx.Response, "message_start",
            JsonSerializer.Serialize(startMsg, AppJsonContext.Default.AMessageStartEvent));

        // Per Anthropic's protocol, thinking (when present) is index 0 and text is index 1.
        // If no thinking chunks ever arrive, the text block takes index 0. Open each block
        // lazily — on its first chunk — so an all-text response still produces a single
        // block at index 0 (matching the pre-thinking wire shape).
        bool thinkingOpen = false;
        bool thinkingClosed = false;
        bool textOpen = false;
        int textIndex = 0;
        var thinkingSb = new StringBuilder();
        int outputTokens = 0;

        try
        {
            await foreach (var chunk in engine.GenerateChunksAsync(prompt, sp, ctx.RequestAborted))
            {
                outputTokens++;

                if (chunk.Kind == GenerateChunkKind.Thinking)
                {
                    // Skip stray thinking chunks that arrive after we've already opened text
                    // (shouldn't happen with well-formed reasoning, but bail gracefully).
                    if (textOpen) continue;

                    if (!thinkingOpen)
                    {
                        var thinkingStart = new AContentBlockStartEvent("content_block_start", 0,
                            new AContentBlock("thinking", Text: null, Thinking: ""));
                        await WriteAnthropicEvent(ctx.Response, "content_block_start",
                            JsonSerializer.Serialize(thinkingStart, AppJsonContext.Default.AContentBlockStartEvent));
                        thinkingOpen = true;
                    }

                    thinkingSb.Append(chunk.Text);
                    var delta = new AContentBlockDeltaEvent("content_block_delta", 0,
                        new AContentDelta("thinking_delta", Thinking: chunk.Text));
                    await WriteAnthropicEvent(ctx.Response, "content_block_delta",
                        JsonSerializer.Serialize(delta, AppJsonContext.Default.AContentBlockDeltaEvent));
                }
                else // Text
                {
                    if (!textOpen)
                    {
                        // First close the thinking block (if any) with a signature_delta then stop.
                        // thinkingClosed flips before the stop write so cancellation between the
                        // two writes doesn't re-emit the signature_delta from the finally block.
                        if (thinkingOpen && !thinkingClosed)
                        {
                            var sigDelta = new AContentBlockDeltaEvent("content_block_delta", 0,
                                new AContentDelta("signature_delta", Signature: MakeSignatureStub(thinkingSb.ToString())));
                            await WriteAnthropicEvent(ctx.Response, "content_block_delta",
                                JsonSerializer.Serialize(sigDelta, AppJsonContext.Default.AContentBlockDeltaEvent));
                            thinkingClosed = true;
                            await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                                JsonSerializer.Serialize(new AContentBlockStopEvent("content_block_stop", 0),
                                    AppJsonContext.Default.AContentBlockStopEvent));
                            textIndex = 1;
                        }

                        var textStart = new AContentBlockStartEvent("content_block_start", textIndex,
                            new AContentBlock("text", Text: "", Thinking: null));
                        await WriteAnthropicEvent(ctx.Response, "content_block_start",
                            JsonSerializer.Serialize(textStart, AppJsonContext.Default.AContentBlockStartEvent));
                        textOpen = true;
                    }

                    var delta = new AContentBlockDeltaEvent("content_block_delta", textIndex,
                        new AContentDelta("text_delta", Text: chunk.Text));
                    await WriteAnthropicEvent(ctx.Response, "content_block_delta",
                        JsonSerializer.Serialize(delta, AppJsonContext.Default.AContentBlockDeltaEvent));
                }
            }
        }
        finally
        {
            // Close whatever blocks are open. Order matters: thinking (index 0) first if it
            // was opened without a follow-up text block (model output ended mid-reasoning or
            // was cancelled); then the text block.
            if (thinkingOpen && !thinkingClosed)
            {
                try
                {
                    var sigDelta = new AContentBlockDeltaEvent("content_block_delta", 0,
                        new AContentDelta("signature_delta", Signature: MakeSignatureStub(thinkingSb.ToString())));
                    await WriteAnthropicEvent(ctx.Response, "content_block_delta",
                        JsonSerializer.Serialize(sigDelta, AppJsonContext.Default.AContentBlockDeltaEvent));
                    await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                        JsonSerializer.Serialize(new AContentBlockStopEvent("content_block_stop", 0),
                            AppJsonContext.Default.AContentBlockStopEvent));
                }
                catch { /* response already aborted */ }
            }
            if (textOpen)
            {
                try
                {
                    await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                        JsonSerializer.Serialize(new AContentBlockStopEvent("content_block_stop", textIndex),
                            AppJsonContext.Default.AContentBlockStopEvent));
                }
                catch { /* response already aborted */ }
            }
        }

        // Anthropic responses always include a terminal text block, even if empty. Cover
        // two cases here: (a) no chunks at all → empty text at index 0; (b) thinking-only
        // (model hit budget or ended mid-`<think>`) → empty text at index 1.
        if (!textOpen)
        {
            try
            {
                int idx = thinkingOpen ? 1 : 0;
                var textStart = new AContentBlockStartEvent("content_block_start", idx,
                    new AContentBlock("text", Text: "", Thinking: null));
                await WriteAnthropicEvent(ctx.Response, "content_block_start",
                    JsonSerializer.Serialize(textStart, AppJsonContext.Default.AContentBlockStartEvent));
                await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                    JsonSerializer.Serialize(new AContentBlockStopEvent("content_block_stop", idx),
                        AppJsonContext.Default.AContentBlockStopEvent));
            }
            catch { /* response already aborted */ }
        }

        // message_delta
        var msgDelta = new AMessageDeltaEvent("message_delta",
            new AMessageDelta("end_turn", null), new AUsage(0, outputTokens));
        await WriteAnthropicEvent(ctx.Response, "message_delta",
            JsonSerializer.Serialize(msgDelta, AppJsonContext.Default.AMessageDeltaEvent));

        // message_stop
        await WriteAnthropicEvent(ctx.Response, "message_stop",
            JsonSerializer.Serialize(new ATypeOnly("message_stop"), AppJsonContext.Default.ATypeOnly));

        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        HealthEndpoints.RecordTokens(outputTokens);
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

    private static List<(string role, string content)> BuildMessageList(AnthropicMessageRequest req)
    {
        var list = new List<(string, string)>();
        if (req.System is { Length: > 0 })
            list.Add(("system", req.System));
        foreach (var m in req.Messages!)
        {
            var role = m.Role ?? "user";
            var content = m.Content ?? "";
            if (role == "assistant")
                content = ChatTemplate.ScrubAssistantThinking(content);
            list.Add((role, content));
        }
        return list;
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
    string? System,
    bool? Stream,
    float? Temperature,
    float? TopP,
    int? TopK,
    AnthropicThinking? Thinking = null);

public sealed record AnthropicThinking(string? Type, int? BudgetTokens);

public sealed record AnthropicMessage(string? Role, string? Content);

public sealed record AnthropicMessageResponse(
    string Id,
    string Type,
    string Role,
    AContent[] Content,
    string Model,
    string StopReason,
    AUsage Usage);

/// <summary>
/// Heterogeneous content block — covers both <c>text</c> and <c>thinking</c> shapes via
/// optional fields. Null-valued fields are stripped from JSON output (see
/// <see cref="AppJsonContext"/>'s <c>WhenWritingNull</c> setting), so the wire format
/// is byte-identical to Anthropic's: a text block emits only <c>{type,text}</c> and a
/// thinking block emits only <c>{type,thinking,signature}</c>. Using a single record
/// keeps the response array NativeAOT-friendly (no polymorphic serialization needed).
/// </summary>
public sealed record AContent(string Type, string? Text = null, string? Thinking = null, string? Signature = null);

public sealed record AUsage(int InputTokens, int OutputTokens);

// Streaming event types
public sealed record AMessageStartEvent(string Type, AMessageStartInner Message);
public sealed record AMessageStartInner(string Id, string Type, string Role, string Model, string StopReason, AUsage Usage);
public sealed record AContentBlockStartEvent(string Type, int Index, AContentBlock ContentBlock);

/// <summary>
/// Content-block envelope emitted on <c>content_block_start</c>. Same Option-A pattern as
/// <see cref="AContent"/>: a text start emits <c>{type:"text",text:""}</c>, a thinking start
/// emits <c>{type:"thinking",thinking:""}</c>, and the other field is omitted via null.
/// </summary>
public sealed record AContentBlock(string Type, string? Text = null, string? Thinking = null);

public sealed record AContentBlockDeltaEvent(string Type, int Index, AContentDelta Delta);

/// <summary>
/// Streaming delta envelope. Covers <c>text_delta</c>, <c>thinking_delta</c>, and
/// <c>signature_delta</c> via three optional payload fields — only the one matching
/// <see cref="Type"/> is populated, the rest are null and omitted from JSON.
/// Replaces the old text-only <c>ATextDelta</c>.
/// </summary>
public sealed record AContentDelta(string Type, string? Text = null, string? Thinking = null, string? Signature = null);

public sealed record AContentBlockStopEvent(string Type, int Index);
public sealed record AMessageDeltaEvent(string Type, AMessageDelta Delta, AUsage Usage);
public sealed record AMessageDelta(string StopReason, string? StopSequence);
public sealed record ATypeOnly(string Type);
public sealed record AErrorResponse(string Type, string Message);
