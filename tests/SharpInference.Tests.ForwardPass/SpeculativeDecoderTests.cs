using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Construction-time contract for <see cref="SpeculativeDecoder"/>. Speculative decoding
/// rewinds rejected draft tokens via <see cref="IForwardPass.TruncateTo"/>; any forward
/// pass whose state is destructively updated per token (Gated DeltaNet hybrid models,
/// <see cref="IForwardPass.SupportsPartialRewind"/> == false) is rejected up front so the
/// failure surfaces at construction rather than mid-decode. See issue #20.
/// </summary>
public sealed class SpeculativeDecoderTests
{
    [Fact]
    public void Ctor_RewindIncompatibleTarget_ThrowsWithParamNameTarget()
    {
        var target = new MockForwardPass(vocabSize: 16, supportsPartialRewind: false);
        var draft = new MockForwardPass(vocabSize: 16, supportsPartialRewind: true);

        var ex = Assert.Throws<ArgumentException>(() => new SpeculativeDecoder(target, draft));
        Assert.Equal("target", ex.ParamName);
    }

    [Fact]
    public void Ctor_RewindIncompatibleDraft_ThrowsWithParamNameDraft()
    {
        var target = new MockForwardPass(vocabSize: 16, supportsPartialRewind: true);
        var draft = new MockForwardPass(vocabSize: 16, supportsPartialRewind: false);

        var ex = Assert.Throws<ArgumentException>(() => new SpeculativeDecoder(target, draft));
        Assert.Equal("draft", ex.ParamName);
    }

    [Fact]
    public void Ctor_BothRewindCapable_ConstructsSuccessfully()
    {
        var target = new MockForwardPass(vocabSize: 16, supportsPartialRewind: true);
        var draft = new MockForwardPass(vocabSize: 16, supportsPartialRewind: true);

        // No throw expected. Lookahead defaults to 4 (clamped below in the ctor for ≤ 0).
        var decoder = new SpeculativeDecoder(target, draft);
        Assert.Equal(4, decoder.Lookahead);
    }

    [Fact]
    public void Ctor_VocabSizeMismatch_ThrowsArgumentExceptionRegardlessOfRewindSupport()
    {
        // Regression guard: the vocab-size check (pre-existing) and the partial-rewind check
        // (new in #20) coexist. The vocab-size check runs first.
        var target = new MockForwardPass(vocabSize: 16, supportsPartialRewind: false);
        var draft = new MockForwardPass(vocabSize: 32, supportsPartialRewind: false);

        var ex = Assert.Throws<ArgumentException>(() => new SpeculativeDecoder(target, draft));
        Assert.Contains("vocab", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    private sealed class MockForwardPass : IForwardPass
    {
        public MockForwardPass(int vocabSize, bool supportsPartialRewind)
        {
            VocabSize = vocabSize;
            SupportsPartialRewind = supportsPartialRewind;
        }

        public int VocabSize { get; }
        public int MaxSeqLen => 4096;
        public bool SupportsPartialRewind { get; }

        public ReadOnlySpan<float> Forward(int token, int position) => new float[VocabSize];
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => new float[VocabSize];
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public void Dispose() { }
    }
}
