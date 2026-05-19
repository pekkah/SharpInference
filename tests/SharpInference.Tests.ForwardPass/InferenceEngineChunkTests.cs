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

        Assert.All(chunks, c => Assert.Equal(GenerateChunkKind.Text, c.Kind));
        var joined = string.Concat(chunks.Select(c => c.Text));
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

        public IReadOnlyList<int> Encode(string text) => [TokHi];

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
        public void Dispose() { }
    }
}
