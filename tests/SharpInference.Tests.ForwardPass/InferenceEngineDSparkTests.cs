using System.Text;
using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// DSpark in the single-user engine (docs/dspark-plan.md Phase 6, PR #413):
/// greedy no-thinking requests with an attached <see cref="IDSparkDraft"/> must
/// decode through <see cref="DSparkDecoder"/> and emit EXACTLY what the plain
/// per-token path emits; sampled requests fall back to the plain loop; the
/// attach-time validations catch misconfiguration. Pure-logic fakes — no model
/// files (mirrors InferenceEngineChunkTests' scripted harness).
/// </summary>
public sealed class InferenceEngineDSparkTests
{
    private const int Vocab = 6;   // "A" "B" "C" "D" "E" "<eos>"
    private const int EmbDim = 4;
    private const int Block = 3;
    private const int TokEos = 5;

    /// <summary>Deterministic chain target with taps: greedy next after t is (t+1) % Vocab.</summary>
    private sealed class TapChainPass : IForwardPass
    {
        public int CacheLen;
        public int TapHighWater;
        public int BatchVerifyCalls;
        private int _tapLayers;

        public int VocabSize => Vocab;
        public int MaxSeqLen => 256;
        public bool SupportsPartialRewind => true;
        public bool SupportsBatchVerify => true;
        public bool SupportsHiddenTaps => true;
        public int HiddenTapDim => _tapLayers * EmbDim;

        public void EnableHiddenTaps(ReadOnlySpan<int> layerIds) => _tapLayers = layerIds.Length;

        public ReadOnlySpan<float> HiddenTapsAt(int position)
        {
            if (_tapLayers == 0 || position < 0 || position >= TapHighWater) return default;
            var row = new float[HiddenTapDim];
            Array.Fill(row, position + 0.5f);
            return row;
        }

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            Assert.Equal(CacheLen, position);   // appends exactly at the cache end
            CacheLen = position + 1;
            if (CacheLen > TapHighWater) TapHighWater = CacheLen;
            return Logits((token + 1) % Vocab);
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            Assert.Equal(CacheLen, startPos);   // suffix prefill resumes at the cache end
            CacheLen = startPos + tokens.Count;
            if (CacheLen > TapHighWater) TapHighWater = CacheLen;
            return Logits((tokens[^1] + 1) % Vocab);
        }

        public float[][] BatchVerify(int[] tokens, int startPos)
        {
            Assert.Equal(CacheLen, startPos);
            BatchVerifyCalls++;
            CacheLen = startPos + tokens.Length;
            if (CacheLen > TapHighWater) TapHighWater = CacheLen;
            var result = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++)
                result[i] = Logits((tokens[i] + 1) % Vocab);
            return result;
        }

        public void TruncateTo(int length)
        {
            Assert.True(length <= CacheLen);
            CacheLen = length;
        }

        public void ResetCache() => CacheLen = 0;
        public void Dispose() { }

        private static float[] Logits(int next)
        {
            // Strongly peaked so even sampled decode (temp 0.7 fallback test) is
            // deterministic — a 1.0 one-hot LOGIT would leave the winner only
            // ~45% probability after softmax at that temperature.
            var l = new float[Vocab];
            l[next] = 100f;
            return l;
        }
    }

    /// <summary>Correct-chain draft; counts proposals so tests can prove engagement.</summary>
    private sealed class ChainDraft : IDSparkDraft
    {
        public int ProposeCalls;
        public int ContextLength { get; private set; }
        public int BlockSize => Block;
        public int VocabSize => Vocab;
        public int MaxContext => int.MaxValue;
        public int TapDim { get; init; } = 2 * EmbDim;

        public void AppendContext(ReadOnlySpan<float> taps, int startPos, int count)
        {
            Assert.Equal(ContextLength, startPos);
            Assert.Equal(count * TapDim, taps.Length);
            ContextLength = startPos + count;
        }

        public DSparkProposal ProposeBlock(int anchorToken, int anchorPos)
        {
            Assert.Equal(ContextLength, anchorPos);
            ProposeCalls++;
            var chain = new int[Block];
            for (int j = 0; j < Block; j++) chain[j] = (anchorToken + 1 + j) % Vocab;
            var ones = new float[Block];
            Array.Fill(ones, 1f);
            return new DSparkProposal(chain, ones);
        }

        public void TruncateContext(int length) => ContextLength = Math.Min(ContextLength, length);
        public void ResetContext() => ContextLength = 0;
        public void Dispose() { }
    }

    private sealed class ChainTokenizer : ITokenizer
    {
        private static readonly string[] Strings = ["A", "B", "C", "D", "E", "<eos>"];
        public int VocabSize => Strings.Length;
        public int BosTokenId => 0;
        public int EosTokenId => TokEos;
        public int UnknownTokenId => 0;
        public int PadTokenId => TokEos;
        public bool AddBosToken => false;
        public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => [TokEos];
        public (int Open, int Close) ReasoningOverride { get; init; } = (-1, -1);
        public (int Open, int Close) ReasoningTokens => ReasoningOverride;
        public IReadOnlyList<int> Encode(string text) => [0];   // prompt = "A"
        public string Decode(IEnumerable<int> tokens)
        {
            var sb = new StringBuilder();
            foreach (var id in tokens)
                if ((uint)id < (uint)Strings.Length) sb.Append(Strings[id]);
            return sb.ToString();
        }
        public byte[] DecodeBytes(int token) =>
            (uint)token >= (uint)Strings.Length ? [] : Encoding.UTF8.GetBytes(Strings[token]);
    }

    private static async Task<string> GenerateText(InferenceEngine engine, SamplingParams sp)
    {
        var sb = new StringBuilder();
        await foreach (var s in engine.GenerateAsync("seed", sp))
            sb.Append(s);
        return sb.ToString();
    }

    [Fact]
    public async Task Auto_WithAttachedDraft_MatchesPlainDecode()
    {
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };

        // Baseline: same fakes, no draft attached → plain per-token decode.
        string baseline;
        using (var engine = new InferenceEngine(new TapChainPass(), new ChainTokenizer(), "mock",
                   thinkTokenId: -1, endThinkTokenId: -1))
        {
            baseline = await GenerateText(engine, sp);
        }
        Assert.Equal("BCDE", baseline);   // chain A→B→C→D→E→<eos>, stop not emitted

        // DSpark: taps enabled before any request, draft attached → identical text.
        var fwd = new TapChainPass();
        ((IForwardPass)fwd).EnableHiddenTaps([0, 2]);
        var draft = new ChainDraft();
        using (var engine = new InferenceEngine(fwd, new ChainTokenizer(), "mock",
                   thinkTokenId: -1, endThinkTokenId: -1))
        {
            engine.AttachDSparkDraft(draft);
            var text = await GenerateText(engine, sp);
            Assert.Equal(baseline, text);
        }
        Assert.True(draft.ProposeCalls > 0, "The draft never proposed — the DSpark path didn't engage.");
        Assert.True(fwd.BatchVerifyCalls > 0);
    }

    /// <summary>
    /// Thinking-CAPABLE model (reasoning ids registered) + a request rendered with
    /// enable_thinking=false: DSpark must engage (the per-request gate, not the
    /// model-static one) and any stray boundary token in the chain must be consumed,
    /// never emitted — output identical to the plain path, which routes the same
    /// markers through the think/text splitter.
    /// </summary>
    [Fact]
    public async Task ThinkingModel_ThinkingDisabled_EngagesDSparkAndSwallowsMarkers()
    {
        // Chain A→B→C→D→E→eos with tokens 2 ("C") / 3 ("D") registered as the
        // reasoning boundary pair: the plain path consumes them as <think>/</think>
        // markers, so visible text is "BE".
        var tokenizer = new ChainTokenizer { ReasoningOverride = (2, 3) };
        var spOn = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };

        string baseline;
        using (var engine = new InferenceEngine(new TapChainPass(), tokenizer, "mock",
                   thinkTokenId: 2, endThinkTokenId: 3))
        {
            baseline = await GenerateText(engine, spOn);
        }
        Assert.Equal("BE", baseline);

        var fwd = new TapChainPass();
        ((IForwardPass)fwd).EnableHiddenTaps([0, 2]);
        var draft = new ChainDraft();
        var spOff = new SamplingParams { Temperature = 0f, MaxNewTokens = 10, ThinkingDisabled = true };
        using (var engine = new InferenceEngine(fwd, tokenizer, "mock",
                   thinkTokenId: 2, endThinkTokenId: 3))
        {
            engine.AttachDSparkDraft(draft);
            var text = await GenerateText(engine, spOff);
            Assert.Equal(baseline, text);
        }
        Assert.True(draft.ProposeCalls > 0,
            "ThinkingDisabled did not unlock the DSpark path on a thinking-capable model.");

        // Without the per-request opt-out the model-static gate keeps DSpark off.
        var fwd2 = new TapChainPass();
        ((IForwardPass)fwd2).EnableHiddenTaps([0, 2]);
        var draft2 = new ChainDraft();
        using (var engine = new InferenceEngine(fwd2, tokenizer, "mock",
                   thinkTokenId: 2, endThinkTokenId: 3))
        {
            engine.AttachDSparkDraft(draft2);
            var text = await GenerateText(engine, spOn);
            Assert.Equal(baseline, text);
        }
        Assert.Equal(0, draft2.ProposeCalls);
    }

    [Fact]
    public async Task SpecTypeDSpark_ThinkingModel_WithoutOptOut_Throws()
    {
        var tokenizer = new ChainTokenizer { ReasoningOverride = (2, 3) };
        var fwd = new TapChainPass();
        ((IForwardPass)fwd).EnableHiddenTaps([0, 2]);
        using var engine = new InferenceEngine(fwd, tokenizer, "mock",
            thinkTokenId: 2, endThinkTokenId: 3);
        engine.AttachDSparkDraft(new ChainDraft());
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4, SpecType = SpecType.DSpark };

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var _ in engine.GenerateAsync("seed", sp)) { }
        });
    }

    [Fact]
    public async Task Sampled_WithAttachedDraft_FallsBackToPlainLoop()
    {
        var fwd = new TapChainPass();
        ((IForwardPass)fwd).EnableHiddenTaps([0, 2]);
        var draft = new ChainDraft();
        using var engine = new InferenceEngine(fwd, new ChainTokenizer(), "mock",
            thinkTokenId: -1, endThinkTokenId: -1);
        engine.AttachDSparkDraft(draft);

        // One-hot logits make sampling deterministic regardless of temperature.
        var sp = new SamplingParams { Temperature = 0.7f, MaxNewTokens = 10, TopK = 0, TopP = 1f, MinP = 0f };
        var text = await GenerateText(engine, sp);

        Assert.Equal("BCDE", text);
        Assert.Equal(0, draft.ProposeCalls);
    }

    [Fact]
    public async Task SpecTypeDSpark_WithoutAttachedDraft_Throws()
    {
        using var engine = new InferenceEngine(new TapChainPass(), new ChainTokenizer(), "mock",
            thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4, SpecType = SpecType.DSpark };

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var _ in engine.GenerateAsync("seed", sp)) { }
        });
    }

    [Fact]
    public async Task SpecTypeDSpark_Sampled_Throws()
    {
        var fwd = new TapChainPass();
        ((IForwardPass)fwd).EnableHiddenTaps([0, 2]);
        using var engine = new InferenceEngine(fwd, new ChainTokenizer(), "mock",
            thinkTokenId: -1, endThinkTokenId: -1);
        engine.AttachDSparkDraft(new ChainDraft());
        var sp = new SamplingParams { Temperature = 0.7f, MaxNewTokens = 4, SpecType = SpecType.DSpark };

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var _ in engine.GenerateAsync("seed", sp)) { }
        });
    }

    [Fact]
    public void Attach_WithoutTapsEnabled_Throws()
    {
        using var engine = new InferenceEngine(new TapChainPass(), new ChainTokenizer(), "mock",
            thinkTokenId: -1, endThinkTokenId: -1);
        // HiddenTapDim is 0 until EnableHiddenTaps — the attach must reject.
        Assert.Throws<InvalidOperationException>(() => engine.AttachDSparkDraft(new ChainDraft()));
    }

    [Fact]
    public void Attach_Twice_Throws()
    {
        var fwd = new TapChainPass();
        ((IForwardPass)fwd).EnableHiddenTaps([0, 2]);
        using var engine = new InferenceEngine(fwd, new ChainTokenizer(), "mock",
            thinkTokenId: -1, endThinkTokenId: -1);
        engine.AttachDSparkDraft(new ChainDraft());
        Assert.Throws<InvalidOperationException>(() => engine.AttachDSparkDraft(new ChainDraft()));
    }

    [Fact]
    public void Attach_VocabMismatch_Throws()
    {
        var fwd = new TapChainPass();
        ((IForwardPass)fwd).EnableHiddenTaps([0, 2]);
        using var engine = new InferenceEngine(fwd, new ChainTokenizer(), "mock",
            thinkTokenId: -1, endThinkTokenId: -1);
        Assert.Throws<InvalidOperationException>(
            () => engine.AttachDSparkDraft(new WrongVocabDraft()));
    }

    private sealed class WrongVocabDraft : IDSparkDraft
    {
        public int BlockSize => Block;
        public int VocabSize => Vocab + 1;
        public int TapDim => 2 * EmbDim;
        public int ContextLength => 0;
        public int MaxContext => int.MaxValue;
        public void AppendContext(ReadOnlySpan<float> taps, int startPos, int count) { }
        public DSparkProposal ProposeBlock(int anchorToken, int anchorPos) => default;
        public void TruncateContext(int length) { }
        public void ResetContext() { }
        public void Dispose() { }
    }
}
