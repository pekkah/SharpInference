using System.Text;
using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Tests for <see cref="InferenceEngine.GenerateChunksAsync"/> — the typed chunk stream
/// that splits output into <see cref="GenerateChunkKind.Text"/> vs
/// <see cref="GenerateChunkKind.Thinking"/> at <c>&lt;think&gt;</c> / <c>&lt;/think&gt;</c>
/// boundaries. Uses synthetic mock implementations of <see cref="ITokenizer"/> and
/// <see cref="IForwardPass"/> so the tests run without a real model file.
/// </summary>
public sealed class InferenceEngineChunkTests
{
    // Token IDs used by ScriptedTokenizer / ScriptedForwardPass below.
    // Token 0 = "Hi", 1 = " there", 2 = "<think>", 3 = "</think>", 4 = "X", 5 = "Y", 6 = EOS.
    private const int TokHi = 0;
    private const int TokThere = 1;
    private const int TokThink = 2;
    private const int TokEndThink = 3;
    private const int TokX = 4;
    private const int TokY = 5;
    private const int TokEos = 6;

    // ── No reasoning (thinkTokenId = -1) ─────────────────────────────────

    [Fact]
    public async Task GenerateChunksAsync_NoThinking_AllChunksAreText()
    {
        // Model emits: "Hi" " there" EOS — no <think> tokens involved.
        var scripted = new int[] { TokHi, TokThere, TokEos };
        var tokenizer = new ScriptedTokenizer();
        var fwd = new ScriptedForwardPass(scripted, tokenizer.VocabSize);
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var chunks = new List<GenerateChunk>();
        await foreach (var c in engine.GenerateChunksAsync("seed", sp))
            chunks.Add(c);

        // The engine leads with one out-of-band Usage chunk carrying the prompt-token count
        // (issue #150) — ScriptedTokenizer.Encode returns a single token. The rest are Text.
        var usage = Assert.Single(chunks, c => c.Kind == GenerateChunkKind.Usage);
        Assert.Equal(1, usage.PromptTokens);
        Assert.All(chunks.Where(c => c.Kind != GenerateChunkKind.Usage),
            c => Assert.Equal(GenerateChunkKind.Text, c.Kind));
        var joined = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));
        Assert.Equal("Hi there", joined);
    }

    [Fact]
    public async Task GenerateAsync_BackCompat_MatchesJoinedTextChunks()
    {
        var scripted = new int[] { TokHi, TokThere, TokEos };
        var tokenizer = new ScriptedTokenizer();
        var fwd = new ScriptedForwardPass(scripted, tokenizer.VocabSize);
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var sb = new StringBuilder();
        await foreach (var s in engine.GenerateAsync("seed", sp))
            sb.Append(s);

        // Back-compat string stream must equal the concatenation of Text chunks.
        Assert.Equal("Hi there", sb.ToString());
    }

    [Fact]
    public async Task GenerateChunksAsync_StopsOnAlternateEogToken()
    {
        // EogTokenIds = {EOS=6, alternate=Y=5}. Model emits: Hi, Y(alt-EOG), X.
        // Generation must halt at the alternate EOG token — only "Hi" is emitted; neither the
        // stop token nor anything after it appears. This is the Gemma 4 <eos>-vs-<turn|> fix:
        // without EogTokenIds the engine would stop only on EOS (6) and decode Y as text.
        var scripted = new int[] { TokHi, TokY, TokX };
        var tokenizer = new ScriptedTokenizer { EogOverride = [TokEos, TokY] };
        var fwd = new ScriptedForwardPass(scripted, tokenizer.VocabSize);
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var sb = new StringBuilder();
        await foreach (var s in engine.GenerateAsync("seed", sp))
            sb.Append(s);

        Assert.Equal("Hi", sb.ToString());
    }

    // ── With reasoning ───────────────────────────────────────────────────

    [Fact]
    public async Task GenerateChunksAsync_WithThinking_SplitsReasoningAndAnswer()
    {
        // Model emits: <think> X </think> Y EOS
        // Expected stream: Thinking("X"), Text("Y") — boundary tokens never appear.
        var scripted = new int[] { TokThink, TokX, TokEndThink, TokY, TokEos };
        var tokenizer = new ScriptedTokenizer();
        var fwd = new ScriptedForwardPass(scripted, tokenizer.VocabSize);
        using var engine = new InferenceEngine(
            fwd, tokenizer, "mock",
            thinkTokenId: TokThink, endThinkTokenId: TokEndThink);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var chunks = new List<GenerateChunk>();
        await foreach (var c in engine.GenerateChunksAsync("seed", sp))
            chunks.Add(c);

        var thinkingText = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Thinking).Select(c => c.Text));
        var answerText = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));

        Assert.Equal("X", thinkingText);
        Assert.Equal("Y", answerText);

        // Boundary literals must never appear in any chunk's content.
        foreach (var c in chunks)
        {
            Assert.DoesNotContain("<think>", c.Text, StringComparison.Ordinal);
            Assert.DoesNotContain("</think>", c.Text, StringComparison.Ordinal);
        }
    }

    [Fact]
    public async Task GenerateAsync_WithThinking_SuppressesThinkingFromBackCompatStream()
    {
        // <think> X </think> Y — back-compat string stream must yield only "Y".
        var scripted = new int[] { TokThink, TokX, TokEndThink, TokY, TokEos };
        var tokenizer = new ScriptedTokenizer();
        var fwd = new ScriptedForwardPass(scripted, tokenizer.VocabSize);
        using var engine = new InferenceEngine(
            fwd, tokenizer, "mock",
            thinkTokenId: TokThink, endThinkTokenId: TokEndThink);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var sb = new StringBuilder();
        await foreach (var s in engine.GenerateAsync("seed", sp))
            sb.Append(s);

        Assert.Equal("Y", sb.ToString());
    }

    [Fact]
    public async Task GenerateChunksAsync_MaxThinkingTokens_ForcesCloseAndYieldsTextAfterBudget()
    {
        // Budget=3 means: after three reasoning content tokens (count reaches 3), the next
        // iteration force-injects </think> instead of sampling. The forced close token still
        // flows through forward(), advancing the scripted sequence to the post-think Y token.
        //
        // Script index → step:
        //   0 <think>  prefill returns this; iter 0 samples it, resets count to 0, enters thinking.
        //   1 X        iter 1 samples; count=1, emits Thinking("X").
        //   2 X        iter 2 samples; count=2, emits Thinking("X").
        //   3 X        iter 3 samples; count=3, emits Thinking("X").
        //   4 X        iter 4 — budget hit, force-injects </think>; logits at index 4 are ignored.
        //              count goes 3→4 (inThinking was true on entry), boundary branch flips inThinking off.
        //   5 Y        iter 5 samples; emits Text("Y").
        //   6 EOS      iter 6 samples; break.
        var scripted = new int[] { TokThink, TokX, TokX, TokX, TokX, TokY, TokEos };
        var tokenizer = new ScriptedTokenizer();
        var fwd = new ScriptedForwardPass(scripted, tokenizer.VocabSize);
        using var engine = new InferenceEngine(
            fwd, tokenizer, "mock",
            thinkTokenId: TokThink, endThinkTokenId: TokEndThink);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 20, MaxThinkingTokens = 3 };
        var chunks = new List<GenerateChunk>();
        await foreach (var c in engine.GenerateChunksAsync("seed", sp))
            chunks.Add(c);

        var thinkingText = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Thinking).Select(c => c.Text));
        var answerText = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));

        // Exactly three X's were admitted before the force-close fired.
        Assert.Equal("XXX", thinkingText);
        // The post-think continuation reached the user.
        Assert.Equal("Y", answerText);
        // Boundary literals stay out of chunk payloads.
        foreach (var c in chunks)
        {
            Assert.DoesNotContain("</think>", c.Text, StringComparison.Ordinal);
            Assert.DoesNotContain("<think>", c.Text, StringComparison.Ordinal);
        }
    }

    [Fact]
    public async Task GenerateChunksAsync_MaxThinkingTokensZero_DoesNotForceClose()
    {
        // Sanity check: with the budget disabled (0 = unlimited, the default), the engine
        // never force-injects, so a long reasoning trace streams through untouched.
        var scripted = new int[] { TokThink, TokX, TokX, TokX, TokX, TokX, TokEndThink, TokY, TokEos };
        var tokenizer = new ScriptedTokenizer();
        var fwd = new ScriptedForwardPass(scripted, tokenizer.VocabSize);
        using var engine = new InferenceEngine(
            fwd, tokenizer, "mock",
            thinkTokenId: TokThink, endThinkTokenId: TokEndThink);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 20, MaxThinkingTokens = 0 };
        var chunks = new List<GenerateChunk>();
        await foreach (var c in engine.GenerateChunksAsync("seed", sp))
            chunks.Add(c);

        var thinkingText = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Thinking).Select(c => c.Text));
        var answerText = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));
        Assert.Equal("XXXXX", thinkingText);
        Assert.Equal("Y", answerText);
    }

    // ── Prompt-seeded thinking state (issue #92) ─────────────────────────

    [Fact]
    public async Task GenerateChunksAsync_PromptEndsWithThinkToken_RoutesFirstTokenToThinking()
    {
        // Qwen3.6 chat template auto-appends `<think>` to the generation prompt, so the
        // model starts already inside thinking mode. Engine must seed inThinking from the
        // prompt tokens — otherwise reasoning content leaks into the Text stream.
        // Model emits: X </think> Y EOS. With the prompt ending in <think>, X must arrive
        // as Thinking and Y as Text.
        var scripted = new int[] { TokX, TokEndThink, TokY, TokEos };
        var tokenizer = new ScriptedTokenizer
        {
            PromptTokens = [TokHi, TokThink],
        };
        var fwd = new ScriptedForwardPass(scripted, tokenizer.VocabSize);
        using var engine = new InferenceEngine(
            fwd, tokenizer, "mock",
            thinkTokenId: TokThink, endThinkTokenId: TokEndThink);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var chunks = new List<GenerateChunk>();
        await foreach (var c in engine.GenerateChunksAsync("seed", sp))
            chunks.Add(c);

        var thinkingText = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Thinking).Select(c => c.Text));
        var answerText = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));

        Assert.Equal("X", thinkingText);
        Assert.Equal("Y", answerText);
    }

    [Fact]
    public async Task GenerateChunksAsync_PromptOpensAndClosesThink_StartsOutsideThinking()
    {
        // A prompt that contains a balanced `<think>...</think>` block must NOT leave
        // the engine in thinking mode at decode start — the scan must track both tokens.
        var scripted = new int[] { TokY, TokEos };
        var tokenizer = new ScriptedTokenizer
        {
            PromptTokens = [TokHi, TokThink, TokX, TokEndThink],
        };
        var fwd = new ScriptedForwardPass(scripted, tokenizer.VocabSize);
        using var engine = new InferenceEngine(
            fwd, tokenizer, "mock",
            thinkTokenId: TokThink, endThinkTokenId: TokEndThink);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var chunks = new List<GenerateChunk>();
        await foreach (var c in engine.GenerateChunksAsync("seed", sp))
            chunks.Add(c);

        var thinkingText = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Thinking).Select(c => c.Text));
        var answerText = string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));

        Assert.Equal("", thinkingText);
        Assert.Equal("Y", answerText);
    }

    // ── Mocks ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Tiny <see cref="ITokenizer"/> stub with a 7-token vocabulary. Decodes each
    /// token to a fixed string and pretends the prompt encodes to a single dummy
    /// token (id 0) so the engine's prefill path has something to chew on.
    /// </summary>
    private sealed class ScriptedTokenizer : ITokenizer
    {
        private static readonly string[] Vocab =
        [
            "Hi", " there", "<think>", "</think>", "X", "Y", "<eos>",
        ];

        public int VocabSize => Vocab.Length;
        public int BosTokenId => 0;
        public int EosTokenId => TokEos;
        public int UnknownTokenId => 0;
        public int PadTokenId => TokEos;
        public bool AddBosToken => false;

        /// <summary>When set, overrides the end-of-generation set (default is just EOS).</summary>
        public System.Collections.Immutable.ImmutableArray<int>? EogOverride { get; set; }
        public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => EogOverride ?? [EosTokenId];

        public int[] PromptTokens { get; set; } = [TokHi];

        public IReadOnlyList<int> Encode(string text) => PromptTokens;

        public string Decode(IEnumerable<int> tokens)
        {
            var sb = new StringBuilder();
            foreach (var id in tokens)
                if ((uint)id < (uint)Vocab.Length) sb.Append(Vocab[id]);
            return sb.ToString();
        }

        public byte[] DecodeBytes(int token)
        {
            if ((uint)token >= (uint)Vocab.Length) return [];
            return Encoding.UTF8.GetBytes(Vocab[token]);
        }
    }

    /// <summary>
    /// Scripted <see cref="IForwardPass"/> that returns one-hot logits favoring the
    /// next token in a pre-determined sequence on every call. Greedy sampling will
    /// pick that token deterministically.
    /// </summary>
    private sealed class ScriptedForwardPass : IForwardPass
    {
        private readonly int[] _sequence;
        private readonly int _vocabSize;
        private readonly float[] _logits;
        private int _step;

        public ScriptedForwardPass(int[] sequence, int vocabSize)
        {
            _sequence = sequence;
            _vocabSize = vocabSize;
            _logits = new float[vocabSize];
        }

        public int VocabSize => _vocabSize;
        public int MaxSeqLen => 4096;

        private ReadOnlySpan<float> EmitNext()
        {
            Array.Clear(_logits);
            if (_step < _sequence.Length)
            {
                int id = _sequence[_step++];
                _logits[id] = 1.0f;
            }
            else
            {
                // Past the script — return EOS to terminate.
                _logits[TokEos] = 1.0f;
            }
            return _logits;
        }

        public ReadOnlySpan<float> Forward(int token, int position) => EmitNext();
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => EmitNext();
        public void TruncateTo(int length) { }
        public void ResetCache() { _step = 0; }
        public bool SupportsPartialRewind => true;
        public void Dispose() { }
    }
}
