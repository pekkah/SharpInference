using System.Text;
using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Regression tests for issue #302: <see cref="InferenceEngine"/> must drive the forward pass
/// from a single, dedicated, long-lived thread (not arbitrary <c>Task.Run</c> thread-pool
/// threads), and must call <see cref="IForwardPass.BindToCurrentThread"/> on that thread before
/// any <c>Forward</c>/<c>Prefill</c> call. CUDA contexts are thread-affine; the old pool-thread
/// model deadlocked the first CUDA call in non-interactive sessions. These tests use a recording
/// mock so they assert the threading contract without a real model or GPU.
/// </summary>
public sealed class InferenceEngineThreadAffinityTests
{
    [Fact]
    public async Task GenerateChunksAsync_RunsForwardOnDedicatedNonPoolThread_AfterBind()
    {
        var tokenizer = new MiniTokenizer();
        var fwd = new RecordingForwardPass([1, 2, MiniTokenizer.Eos], tokenizer.VocabSize);
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        int callerThreadId = Environment.CurrentManagedThreadId;

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        await foreach (var _ in engine.GenerateChunksAsync("seed", sp)) { }

        // BindToCurrentThread must have run, before the first forward, on the SAME thread that
        // then issued every Forward/Prefill call.
        Assert.True(fwd.BindCalled, "BindToCurrentThread was never called");
        Assert.True(fwd.BindCalledBeforeFirstForward, "a forward ran before the context was bound");
        Assert.NotEmpty(fwd.ForwardThreadIds);
        Assert.All(fwd.ForwardThreadIds, id => Assert.Equal(fwd.BindThreadId, id));

        // The forward pass must NOT run on the caller's thread nor on a thread-pool thread — the
        // whole point of #302 is a stable, owned thread the CUDA context is bound to.
        Assert.NotEqual(callerThreadId, fwd.BindThreadId);
        Assert.False(fwd.AnyForwardOnThreadPoolThread,
            "forward ran on a thread-pool thread — CUDA context affinity is not guaranteed there");
    }

    [Fact]
    public async Task GenerateChunksAsync_SuccessiveRequests_ShareOneStableEngineThread()
    {
        var tokenizer = new MiniTokenizer();
        var fwd = new RecordingForwardPass([1, MiniTokenizer.Eos], tokenizer.VocabSize);
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };

        await foreach (var _ in engine.GenerateChunksAsync("seed", sp)) { }
        int firstThread = fwd.BindThreadId;

        await foreach (var _ in engine.GenerateChunksAsync("seed", sp)) { }

        // Every forward across both requests ran on the one engine thread — confirming a single
        // owned thread rather than a fresh pool thread per request.
        Assert.NotEqual(0, firstThread);
        Assert.All(fwd.ForwardThreadIds, id => Assert.Equal(firstThread, id));
    }

    [Fact]
    public async Task GenerateChunksAsync_BindFailure_SurfacesThroughChannel_DoesNotHang()
    {
        // A backend whose context bind throws must fault the request's stream, not hang the
        // consumer forever (the failure mode #302 is about). The dedicated thread survives so
        // later requests still work.
        var tokenizer = new MiniTokenizer();
        var fwd = new RecordingForwardPass([1, MiniTokenizer.Eos], tokenizer.VocabSize) { ThrowOnBind = true };
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var _ in engine.GenerateChunksAsync("seed", sp)) { }
        });
    }

    // ── Mocks ─────────────────────────────────────────────────────────────

    /// <summary>2-token-plus-EOS tokenizer; the prompt encodes to a single dummy token.</summary>
    private sealed class MiniTokenizer : ITokenizer
    {
        public const int Eos = 2;
        private static readonly string[] Vocab = ["a", "b", "<eos>"];

        public int VocabSize => Vocab.Length;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => [Eos];

        public IReadOnlyList<int> Encode(string text) => [0];
        public string Decode(IEnumerable<int> tokens)
        {
            var sb = new StringBuilder();
            foreach (var id in tokens)
                if ((uint)id < (uint)Vocab.Length) sb.Append(Vocab[id]);
            return sb.ToString();
        }
        public byte[] DecodeBytes(int token) =>
            (uint)token < (uint)Vocab.Length ? Encoding.UTF8.GetBytes(Vocab[token]) : [];
    }

    /// <summary>
    /// Scripted forward pass that records which thread <see cref="BindToCurrentThread"/> and each
    /// <c>Forward</c>/<c>Prefill</c> ran on, plus whether the bind preceded the first forward.
    /// </summary>
    private sealed class RecordingForwardPass : IForwardPass
    {
        private readonly int[] _sequence;
        private readonly int _vocabSize;
        private readonly float[] _logits;
        private readonly object _gate = new();
        private int _step;

        public RecordingForwardPass(int[] sequence, int vocabSize)
        {
            _sequence = sequence;
            _vocabSize = vocabSize;
            _logits = new float[vocabSize];
        }

        public bool ThrowOnBind { get; init; }

        public bool BindCalled { get; private set; }
        public int BindThreadId { get; private set; }
        public bool BindCalledBeforeFirstForward { get; private set; } = true;
        public bool AnyForwardOnThreadPoolThread { get; private set; }
        public List<int> ForwardThreadIds { get; } = [];

        public int VocabSize => _vocabSize;
        public int MaxSeqLen => 4096;
        public bool SupportsPartialRewind => true;

        public void BindToCurrentThread()
        {
            if (ThrowOnBind)
                throw new InvalidOperationException("simulated cuCtxSetCurrent failure");
            lock (_gate)
            {
                BindCalled = true;
                BindThreadId = Environment.CurrentManagedThreadId;
            }
        }

        private ReadOnlySpan<float> EmitNext()
        {
            lock (_gate)
            {
                if (!BindCalled) BindCalledBeforeFirstForward = false;
                if (Thread.CurrentThread.IsThreadPoolThread) AnyForwardOnThreadPoolThread = true;
                ForwardThreadIds.Add(Environment.CurrentManagedThreadId);
            }

            Array.Clear(_logits);
            int id = _step < _sequence.Length ? _sequence[_step++] : MiniTokenizer.Eos;
            _logits[id] = 1.0f;
            return _logits;
        }

        public ReadOnlySpan<float> Forward(int token, int position) => EmitNext();
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => EmitNext();
        public void TruncateTo(int length) { }
        public void ResetCache() { _step = 0; }
        public void Dispose() { }
    }
}
