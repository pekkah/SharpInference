using SharpInference.Core;

namespace SharpInference.Tests.Core;

/// <summary>
/// Per-adapter unit tests covering each family's wire format against representative
/// model outputs. Streaming tests use the open/close marker API directly so they
/// also lock in the contract the server's streaming state machine depends on.
/// </summary>
public sealed class ToolCallAdapterTests
{
    // ── Registry ──────────────────────────────────────────────────────────────

    [Fact]
    public void Registry_ResolvesQwenArchitectures()
    {
        Assert.IsType<QwenToolCallAdapter>(ToolCallAdapterRegistry.Get("qwen2"));
        Assert.IsType<QwenToolCallAdapter>(ToolCallAdapterRegistry.Get("qwen3"));
        Assert.IsType<QwenToolCallAdapter>(ToolCallAdapterRegistry.Get("qwen3moe"));
        Assert.IsType<QwenToolCallAdapter>(ToolCallAdapterRegistry.Get("qwen35moe"));
    }

    [Fact]
    public void Registry_ResolvesQwenCoder() =>
        Assert.IsType<QwenCoderToolCallAdapter>(ToolCallAdapterRegistry.Get("qwen3coder"));

    [Fact]
    public void Registry_ResolvesLlamaFamilies()
    {
        Assert.IsType<LlamaToolCallAdapter>(ToolCallAdapterRegistry.Get("llama"));
        Assert.IsType<LlamaToolCallAdapter>(ToolCallAdapterRegistry.Get("llama4"));
    }

    [Fact]
    public void Registry_ResolvesDeepSeek() =>
        Assert.IsType<DeepSeekToolCallAdapter>(ToolCallAdapterRegistry.Get("deepseek2"));

    [Fact]
    public void Registry_UnknownArchFallsBackToDefault()
    {
        var a = ToolCallAdapterRegistry.Get("never-heard-of-it");
        Assert.Same(ToolCallAdapterRegistry.DefaultAdapter, a);
    }

    [Fact]
    public void Registry_NullOrEmptyArchFallsBackToDefault()
    {
        Assert.Same(ToolCallAdapterRegistry.DefaultAdapter, ToolCallAdapterRegistry.Get(null));
        Assert.Same(ToolCallAdapterRegistry.DefaultAdapter, ToolCallAdapterRegistry.Get(""));
    }

    // ── Qwen (wrapper) adapter ────────────────────────────────────────────────

    [Fact]
    public void Qwen_Parse_JsonCall()
    {
        var a = new QwenToolCallAdapter("qwen3moe");
        var raw = "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>";
        var (plain, calls) = a.Parse(raw);
        Assert.Equal("", plain);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls[0].Name);
        Assert.Equal("Paris", calls[0].Arguments["city"]);
    }

    [Fact]
    public void Qwen_Parse_XmlFunctionCall()
    {
        // Qwen3.6 alt payload: <function=name><parameter=k>v</parameter></function>
        var a = new QwenToolCallAdapter("qwen3moe");
        var raw = "<tool_call><function=read_file><parameter=path>/etc/passwd</parameter></function></tool_call>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Equal("read_file", calls[0].Name);
        Assert.Equal("/etc/passwd", calls[0].Arguments["path"]);
    }

    [Fact]
    public void Qwen_Parse_TextBeforeAndAfterToolCall()
    {
        var a = new QwenToolCallAdapter("qwen3moe");
        var raw = "Let me check.<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call> Done.";
        var (plain, calls) = a.Parse(raw);
        Assert.Equal("Let me check. Done.", plain);
        Assert.Single(calls);
    }

    [Fact]
    public void Qwen_FindMarkers_RoundTripsBlock()
    {
        var a = new QwenToolCallAdapter("qwen3moe");
        var buf = "noise<tool_call>{\"name\":\"x\",\"arguments\":{\"k\":1}}</tool_call>tail";
        int open = a.FindOpenMarker(buf, 0, out int contentStart);
        Assert.Equal(5, open);
        Assert.Equal(5 + "<tool_call>".Length, contentStart);
        int close = a.FindCloseMarker(buf, contentStart, out int afterClose);
        Assert.True(close > contentStart);
        Assert.Equal(close + "</tool_call>".Length, afterClose);

        var block = buf[contentStart..close];
        var calls = a.ParseBlock(block);
        Assert.Single(calls);
        Assert.Equal("x", calls[0].Name);
    }

    // ── Qwen3-Coder adapter (closes #95) ──────────────────────────────────────

    [Fact]
    public void QwenCoder_Parse_BareFunctionBlock()
    {
        var a = new QwenCoderToolCallAdapter();
        var raw = "<function=get_weather><parameter=city>Paris</parameter></function>";
        var (plain, calls) = a.Parse(raw);
        Assert.Equal("", plain);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls[0].Name);
        Assert.Equal("Paris", calls[0].Arguments["city"]);
    }

    [Fact]
    public void QwenCoder_Parse_TextBeforeFunctionBlock()
    {
        var a = new QwenCoderToolCallAdapter();
        var raw = "I'll check.\n<function=lookup><parameter=q>x</parameter></function>";
        var (plain, calls) = a.Parse(raw);
        Assert.Equal("I'll check.\n", plain);
        Assert.Single(calls);
    }

    [Fact]
    public void QwenCoder_Parse_MultipleFunctionBlocks()
    {
        var a = new QwenCoderToolCallAdapter();
        var raw = "<function=a><parameter=k>1</parameter></function>"
               + "<function=b><parameter=k>2</parameter></function>";
        var (_, calls) = a.Parse(raw);
        Assert.Equal(2, calls.Count);
        Assert.Equal("a", calls[0].Name);
        Assert.Equal("b", calls[1].Name);
    }

    [Fact]
    public void QwenCoder_Streaming_BlockIncludesOpenMarker()
    {
        // The name is inside the open marker, so the streaming block MUST contain it.
        var a = new QwenCoderToolCallAdapter();
        var buf = "<function=ls><parameter=path>/</parameter></function>";
        int open = a.FindOpenMarker(buf, 0, out int contentStart);
        Assert.Equal(0, open);
        Assert.Equal(0, contentStart);   // contentStart == openIdx → block keeps the marker
        int close = a.FindCloseMarker(buf, contentStart, out int afterClose);
        var block = buf[contentStart..close];
        Assert.StartsWith("<function=", block);
        var calls = a.ParseBlock(block);
        Assert.Single(calls);
        Assert.Equal("ls", calls[0].Name);
        Assert.Equal(afterClose, buf.Length);   // we consumed everything
    }

    // ── Llama adapter ─────────────────────────────────────────────────────────

    [Fact]
    public void Llama_Parse_PythonTagBlock_WithEom()
    {
        var a = new LlamaToolCallAdapter();
        var raw = "<|python_tag|>{\"name\":\"get_weather\",\"parameters\":{\"city\":\"Paris\"}}<|eom_id|>";
        var (plain, calls) = a.Parse(raw);
        Assert.Equal("", plain);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls[0].Name);
        Assert.Equal("Paris", calls[0].Arguments["city"]);
    }

    [Fact]
    public void Llama_Parse_PythonTagBlock_WithEot()
    {
        // Some short tool outputs close with <|eot_id|> instead of <|eom_id|>.
        var a = new LlamaToolCallAdapter();
        var raw = "<|python_tag|>{\"name\":\"x\",\"parameters\":{}}<|eot_id|>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Equal("x", calls[0].Name);
    }

    [Fact]
    public void Llama_Parse_AcceptsArgumentsKey()
    {
        // Some fine-tunes use the OpenAI "arguments" key instead of "parameters".
        var a = new LlamaToolCallAdapter();
        var raw = "<|python_tag|>{\"name\":\"x\",\"arguments\":{\"k\":1}}<|eom_id|>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Equal(1L, calls[0].Arguments["k"]);
    }

    [Fact]
    public void Llama_RenderToolResult_UsesIpythonRole()
    {
        var a = new LlamaToolCallAdapter();
        var msg = a.RenderToolResult("call_1", "result text");
        Assert.Equal("ipython", msg["role"]);
        Assert.Equal("result text", msg["content"]);
    }

    // ── DeepSeek adapter ──────────────────────────────────────────────────────

    [Fact]
    public void DeepSeek_Parse_SingleInnerCall()
    {
        var a = new DeepSeekToolCallAdapter();
        var raw = "<|tool_calls_begin|>"
               + "<|tool_call_begin|>get_weather<|tool_sep|>{\"city\":\"Paris\"}<|tool_call_end|>"
               + "<|tool_calls_end|>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls[0].Name);
        Assert.Equal("Paris", calls[0].Arguments["city"]);
    }

    [Fact]
    public void DeepSeek_Parse_MultipleInnerCalls()
    {
        var a = new DeepSeekToolCallAdapter();
        var raw = "<|tool_calls_begin|>"
               + "<|tool_call_begin|>a<|tool_sep|>{\"k\":1}<|tool_call_end|>"
               + "<|tool_call_begin|>b<|tool_sep|>{\"k\":2}<|tool_call_end|>"
               + "<|tool_calls_end|>";
        var (_, calls) = a.Parse(raw);
        Assert.Equal(2, calls.Count);
        Assert.Equal("a", calls[0].Name);
        Assert.Equal("b", calls[1].Name);
    }

    [Fact]
    public void DeepSeek_Parse_PlainTextStaysPlain()
    {
        var a = new DeepSeekToolCallAdapter();
        var (plain, calls) = a.Parse("just an answer with no tool call");
        Assert.Equal("just an answer with no tool call", plain);
        Assert.Empty(calls);
    }

    // ── Gemma 4 adapter ───────────────────────────────────────────────────────

    [Fact]
    public void Registry_ResolvesGemma4() =>
        Assert.IsType<Gemma4ToolCallAdapter>(ToolCallAdapterRegistry.Get("gemma4"));

    [Fact]
    public void Gemma4_Parse_StringArgument()
    {
        var a = new Gemma4ToolCallAdapter();
        var raw = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>";
        var (plain, calls) = a.Parse(raw);
        Assert.Equal("", plain);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls[0].Name);
        Assert.Equal("Paris", calls[0].Arguments["city"]);
    }

    [Fact]
    public void Gemma4_Parse_MixedScalarArguments()
    {
        var a = new Gemma4ToolCallAdapter();
        // string, int, double, bool — Gemma's bare/quoted scalar mix.
        var raw = "<|tool_call>call:book{name:<|\"|>Ada<|\"|>,count:3,ratio:1.5,vip:true}<tool_call|>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Equal("Ada", calls[0].Arguments["name"]);
        Assert.Equal(3L, calls[0].Arguments["count"]);
        Assert.Equal(1.5d, calls[0].Arguments["ratio"]);
        Assert.Equal(true, calls[0].Arguments["vip"]);
    }

    [Fact]
    public void Gemma4_Parse_StringWithCommaAndBraces()
    {
        // Structural chars inside a <|"|>-quoted string must NOT split the argument.
        var a = new Gemma4ToolCallAdapter();
        var raw = "<|tool_call>call:say{text:<|\"|>a, b, {c}<|\"|>,n:1}<tool_call|>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Equal("a, b, {c}", calls[0].Arguments["text"]);
        Assert.Equal(1L, calls[0].Arguments["n"]);
    }

    [Fact]
    public void Gemma4_Parse_NestedObjectAndArray()
    {
        var a = new Gemma4ToolCallAdapter();
        var raw = "<|tool_call>call:q{filter:{min:0,max:10},tags:[<|\"|>x<|\"|>,<|\"|>y<|\"|>]}<tool_call|>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        var filter = Assert.IsType<Dictionary<string, object?>>(calls[0].Arguments["filter"]);
        Assert.Equal(0L, filter["min"]);
        Assert.Equal(10L, filter["max"]);
        var tags = Assert.IsType<List<object?>>(calls[0].Arguments["tags"]);
        Assert.Equal(new object?[] { "x", "y" }, tags);
    }

    [Fact]
    public void Gemma4_Parse_NoArguments()
    {
        var a = new Gemma4ToolCallAdapter();
        var raw = "<|tool_call>call:get_time{}<tool_call|>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Equal("get_time", calls[0].Name);
        Assert.Empty(calls[0].Arguments);
    }

    [Fact]
    public void Gemma4_Parse_TextBeforeAndMultipleCalls()
    {
        var a = new Gemma4ToolCallAdapter();
        var raw = "Let me check.<|tool_call>call:a{k:1}<tool_call|>"
                + "<|tool_call>call:b{k:2}<tool_call|>";
        var (plain, calls) = a.Parse(raw);
        Assert.Equal("Let me check.", plain);
        Assert.Equal(2, calls.Count);
        Assert.Equal("a", calls[0].Name);
        Assert.Equal("b", calls[1].Name);
        Assert.Equal(2L, calls[1].Arguments["k"]);
    }

    [Fact]
    public void Gemma4_Parse_PlainTextStaysPlain()
    {
        var a = new Gemma4ToolCallAdapter();
        var (plain, calls) = a.Parse("Paris is the capital of France.");
        Assert.Equal("Paris is the capital of France.", plain);
        Assert.Empty(calls);
    }

    [Fact]
    public void Gemma4_FindMarkers_RoundTripsBlock()
    {
        var a = new Gemma4ToolCallAdapter();
        var buf = "pre<|tool_call>call:x{k:<|\"|>v<|\"|>}<tool_call|>post";
        int open = a.FindOpenMarker(buf, 0, out int contentStart);
        Assert.Equal(3, open);
        Assert.Equal(3 + "<|tool_call>".Length, contentStart);
        int close = a.FindCloseMarker(buf, contentStart, out int afterClose);
        Assert.True(close > contentStart);
        Assert.Equal(close + "<tool_call|>".Length, afterClose);

        var block = buf[contentStart..close];
        var calls = a.ParseBlock(block);
        Assert.Single(calls);
        Assert.Equal("x", calls[0].Name);
        Assert.Equal("v", calls[0].Arguments["k"]);
    }

    [Fact]
    public void Gemma4_Parse_MissingCloseMarker_StillParses()
    {
        // Model stopped before emitting <tool_call|>.
        var a = new Gemma4ToolCallAdapter();
        var raw = "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Equal("Paris", calls[0].Arguments["city"]);
    }

    [Fact]
    public void Gemma4_RenderToolResult_CarriesToolCallId()
    {
        var a = new Gemma4ToolCallAdapter();
        var msg = a.RenderToolResult("call_42", "sunny, 21C");
        Assert.Equal("tool", msg["role"]);
        Assert.Equal("sunny, 21C", msg["content"]);
        Assert.Equal("call_42", msg["tool_call_id"]);
    }

    [Fact]
    public void Gemma4_Parse_NullAndNegativeAndBarewordScalars()
    {
        var a = new Gemma4ToolCallAdapter();
        var raw = "<|tool_call>call:f{a:null,b:-3,c:-1.5,d:pending}<tool_call|>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Null(calls[0].Arguments["a"]);
        Assert.Equal(-3L, calls[0].Arguments["b"]);
        Assert.Equal(-1.5d, calls[0].Arguments["c"]);
        Assert.Equal("pending", calls[0].Arguments["d"]);   // unparseable bareword → raw string
    }

    [Fact]
    public void Gemma4_Parse_QuotedKey()
    {
        // Keys are normally bare, but the parser also accepts a <|"|>-quoted key.
        var a = new Gemma4ToolCallAdapter();
        var raw = "<|tool_call>call:f{<|\"|>weird key<|\"|>:<|\"|>v<|\"|>}<tool_call|>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Equal("v", calls[0].Arguments["weird key"]);
    }

    [Fact]
    public void Gemma4_Parse_UnterminatedQuote_TakesRemainder()
    {
        // Missing closing <|"|> → value absorbs the rest of the block (documented tolerance).
        var a = new Gemma4ToolCallAdapter();
        var raw = "<|tool_call>call:f{msg:<|\"|>hello world<tool_call|>";
        var (_, calls) = a.Parse(raw);
        Assert.Single(calls);
        Assert.Equal("f", calls[0].Name);
        Assert.Equal("hello world", calls[0].Arguments["msg"]);
    }

    [Fact]
    public void Gemma4_FindMarkers_ReturnNegativeWhenAbsent()
    {
        // The streaming state machine relies on the -1 / out=-1 not-found contract.
        var a = new Gemma4ToolCallAdapter();
        Assert.Equal(-1, a.FindOpenMarker("plain text, no markers", 0, out int contentStart));
        Assert.Equal(-1, contentStart);
        Assert.Equal(-1, a.FindCloseMarker("still nothing here", 0, out int afterClose));
        Assert.Equal(-1, afterClose);
    }
}
