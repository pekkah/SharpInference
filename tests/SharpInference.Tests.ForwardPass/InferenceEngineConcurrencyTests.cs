using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Regression tests for issue #109: the single-user <see cref="InferenceEngine"/> must never
/// drive its (non-thread-safe) <see cref="IForwardPass"/> from two requests at once. When an
/// agentic client cancels a request mid-decode and immediately fires another, the engine used
/// to release the serialization gate while the background generation task was still inside the
/// forward pass — letting the next request reset/prefill the shared KV cache concurrently,
/// corrupting state and hanging.
/// </summary>
public sealed class InferenceEngineConcurrencyTests
{
    [Fact]
    public async Task CancelMidDecode_DoesNotLetNextRequestEnterForwardPassConcurrently()
    {
        var fwd = new GatedForwardPass();
        var tokenizer = new SingleTokenTokenizer();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock");

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 64 };

        // Request A: starts generating and parks inside the forward pass (gate held).
        using var ctsA = new CancellationTokenSource();
        var aTask = Task.Run(async () =>
        {
            try
            {
                await foreach (var _ in engine.GenerateChunksAsync("seed", sp, ctsA.Token))
                {
                }
            }
            catch (OperationCanceledException)
            {
                // Expected: A is cancelled mid-flight.
            }
        });

        // Wait until A is actually inside a (blocked) forward-pass call, holding the gate.
        Assert.True(fwd.EnteredFirstCall.Wait(TimeSpan.FromSeconds(5)),
            "Request A never reached the forward pass.");

        // Request B: queues behind A on the serialization gate. It must not touch the
        // forward pass until A's background generation task has fully stopped.
        using var ctsB = new CancellationTokenSource();
        var bTask = Task.Run(async () =>
        {
            try
            {
                await foreach (var _ in engine.GenerateChunksAsync("seed", sp, ctsB.Token))
                {
                }
            }
            catch (OperationCanceledException)
            {
            }
        });

        // Give B a moment to park on the gate (it issues no forward-pass call while queued).
        await Task.Delay(150);

        // Cancel A while its generation task is still blocked inside the forward pass.
        ctsA.Cancel();

        // If the gate is released before A's generation task unwinds, B will enter the
        // forward pass while A is still inside it — the issue #109 deadlock window.
        bool concurrencyObserved = fwd.ConcurrencyDetected.Wait(TimeSpan.FromSeconds(2));

        // Let every blocked forward-pass call complete so both requests can finish.
        fwd.ReleaseAll();
        ctsB.Cancel();
        await Task.WhenAll(aTask, bTask).WaitAsync(TimeSpan.FromSeconds(10));

        Assert.False(concurrencyObserved,
            "Request B entered the forward pass while request A was still inside it (issue #109).");
        Assert.Equal(1, fwd.MaxConcurrentCalls);
    }

    [Fact]
    public async Task LargePrompt_CancelledDuringPrefill_StopsAfterCurrentChunk()
    {
        // A single _fwd.Prefill call is opaque to cancellation; the engine chunks the prompt and
        // checks the token between chunks so a client disconnect aborts a long prefill promptly.
        // This forward pass cancels the request from inside the first chunk's Prefill, simulating
        // a disconnect mid-prefill — the engine must then skip the remaining chunks.
        const int promptLen = 4000; // > one PrefillChunkSize (512) → multiple chunks
        using var cts = new CancellationTokenSource();
        var fwd = new CancelOnFirstPrefillForwardPass(cts);
        var tokenizer = new FixedLengthTokenizer(promptLen);
        using var engine = new InferenceEngine(fwd, tokenizer, "mock");

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 8 };

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in engine.GenerateChunksAsync("x", sp, cts.Token))
            {
            }
        });

        // Only the first chunk ran before cancellation was observed; the rest were skipped.
        Assert.Equal(1, fwd.PrefillCalls);
        Assert.True(fwd.TotalTokensPrefilled < promptLen,
            $"prefilled {fwd.TotalTokensPrefilled} of {promptLen} tokens — prefill was not chunked/cancellable.");
    }

    /// <summary>
    /// Issue #132: <see cref="InferenceEngine.DisposeAsync"/> must not free the engine-owned
    /// forward pass while a background generation worker is still inside it. The worker is
    /// parked inside a <c>Prefill</c> call (holding the gate); disposing then must (a) block
    /// until the worker drains and (b) never call <c>_fwd.Dispose()</c> while a call is in flight.
    /// </summary>
    [Fact]
    public async Task DisposeAsync_DrainsInFlightWorker_BeforeFreeingForwardPass()
    {
        var fwd = new DrainTrackingForwardPass();
        var tokenizer = new SingleTokenTokenizer();
        var engine = new InferenceEngine(fwd, tokenizer, "mock");
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 64 };

        using var cts = new CancellationTokenSource();
        var genTask = Task.Run(async () =>
        {
            try
            {
                await foreach (var _ in engine.GenerateChunksAsync("seed", sp, cts.Token))
                {
                }
            }
            catch (OperationCanceledException)
            {
                // Expected: shutdown cancels the in-flight generation.
            }
        });

        // Park the worker inside the forward pass, holding the serialization gate.
        Assert.True(fwd.EnteredForward.Wait(TimeSpan.FromSeconds(5)),
            "worker never entered the forward pass");

        // Dispose while the worker is still inside Prefill. DisposeAsync signals shutdown
        // (so the worker will unwind at its next checkpoint) and waits for the gate.
        var disposeTask = engine.DisposeAsync().AsTask();

        // It must NOT complete yet — the worker is still inside the forward pass, and
        // freeing _fwd now is exactly the 0xC0000005 window from #132.
        var settledEarly = await Task.WhenAny(disposeTask, Task.Delay(500));
        Assert.NotSame(disposeTask, settledEarly);
        Assert.False(fwd.Disposed, "forward pass was freed before the in-flight worker drained");

        // Release the parked call; the worker returns, observes shutdown cancellation,
        // drains, releases the gate, and DisposeAsync proceeds to free the forward pass.
        fwd.Release();
        await disposeTask.WaitAsync(TimeSpan.FromSeconds(10));
        await genTask.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.True(fwd.Disposed, "forward pass was never disposed");
        Assert.False(fwd.DisposedWhileActive,
            "forward pass was disposed while a worker call was in flight (issue #132)");
    }

    [Fact]
    public async Task GenerateChunksAsync_AfterDispose_Throws()
    {
        var fwd = new DrainTrackingForwardPass();
        var engine = new InferenceEngine(fwd, new SingleTokenTokenizer(), "mock");
        await engine.DisposeAsync();

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        await Assert.ThrowsAsync<ObjectDisposedException>(async () =>
        {
            await foreach (var _ in engine.GenerateChunksAsync("seed", sp))
            {
            }
        });

        // Dispose is single-shot and safe to call again (sync and async), in any order.
        await engine.DisposeAsync();
        engine.Dispose();
    }

    /// <summary>Tokenizer whose prompt always encodes to <paramref name="length"/> copies of token 0.</summary>
    private sealed class FixedLengthTokenizer(int length) : ITokenizer
    {
        private readonly int[] _tokens = new int[length];
        public int VocabSize => 2;
        public int BosTokenId => 0;
        public int EosTokenId => 1;
        public int UnknownTokenId => 0;
        public int PadTokenId => 1;
        public bool AddBosToken => false;
        public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => [EosTokenId];
        public IReadOnlyList<int> Encode(string text) => _tokens;
        public string Decode(IEnumerable<int> tokens) => "a";
        public byte[] DecodeBytes(int token) => "a"u8.ToArray();
    }

    /// <summary>
    /// Forward pass that cancels the supplied token source from inside its first <see cref="Prefill"/>
    /// call (simulating a client disconnect mid-prefill) and records how many prefill chunks it saw.
    /// </summary>
    private sealed class CancelOnFirstPrefillForwardPass(CancellationTokenSource cts) : IForwardPass
    {
        private readonly float[] _logits = [1.0f, 0.0f];
        public int PrefillCalls { get; private set; }
        public int TotalTokensPrefilled { get; private set; }

        public int VocabSize => 2;
        public int MaxSeqLen => 16384;

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            PrefillCalls++;
            TotalTokensPrefilled += tokens.Count;
            cts.Cancel(); // disconnect arrives while this chunk is being prefilled
            return _logits;
        }

        public ReadOnlySpan<float> Forward(int token, int position) => _logits;
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public bool SupportsPartialRewind => true;
        public void Dispose() { }
    }

    /// <summary>
    /// Tokenizer that maps any prompt to a single token and decodes everything to "a".
    /// Enough to drive the engine's prefill + decode loop without a real model.
    /// </summary>
    private sealed class SingleTokenTokenizer : ITokenizer
    {
        public int VocabSize => 2;
        public int BosTokenId => 0;
        public int EosTokenId => 1;
        public int UnknownTokenId => 0;
        public int PadTokenId => 1;
        public bool AddBosToken => false;
        public System.Collections.Immutable.ImmutableArray<int> EogTokenIds => [EosTokenId];

        public IReadOnlyList<int> Encode(string text) => [0];
        public string Decode(IEnumerable<int> tokens) => "a";
        public byte[] DecodeBytes(int token) => "a"u8.ToArray();
    }

    /// <summary>
    /// Forward pass that records the maximum number of concurrent in-flight calls and blocks
    /// every <see cref="Prefill"/> / <see cref="Forward"/> on a release gate so a test can park
    /// a request inside the forward pass deterministically. Always emits a non-EOS token so the
    /// decode loop keeps running until cancelled.
    /// </summary>
    private sealed class GatedForwardPass : IForwardPass
    {
        private readonly float[] _logits = [1.0f, 0.0f]; // greedy → token 0 (non-EOS)
        private readonly Lock _lock = new();
        private readonly ManualResetEventSlim _release = new(false);

        public readonly ManualResetEventSlim EnteredFirstCall = new(false);
        public readonly ManualResetEventSlim ConcurrencyDetected = new(false);

        private int _active;
        public int MaxConcurrentCalls { get; private set; }

        public int VocabSize => 2;
        public int MaxSeqLen => 4096;

        private void Enter()
        {
            lock (_lock)
            {
                _active++;
                if (_active > MaxConcurrentCalls)
                    MaxConcurrentCalls = _active;
                if (_active >= 2)
                    ConcurrencyDetected.Set();
            }
        }

        private void Exit()
        {
            lock (_lock)
                _active--;
        }

        private ReadOnlySpan<float> Blocking()
        {
            Enter();
            try
            {
                EnteredFirstCall.Set();
                // Safety timeout so a regressed engine can't hang the test run forever.
                _release.Wait(TimeSpan.FromSeconds(15));
                return _logits;
            }
            finally
            {
                Exit();
            }
        }

        public void ReleaseAll() => _release.Set();

        public ReadOnlySpan<float> Forward(int token, int position) => Blocking();
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => Blocking();

        // Non-blocking cache mutations still count toward concurrency — a second request
        // calling ResetCache while the first is mid-Forward is exactly the corruption we guard against.
        public void ResetCache()
        {
            Enter();
            Exit();
        }

        public void TruncateTo(int length)
        {
            Enter();
            Exit();
        }

        public bool SupportsPartialRewind => true;
        public void Dispose() => _release.Dispose();
    }

    /// <summary>
    /// Forward pass that parks every call on a release gate (like <see cref="GatedForwardPass"/>)
    /// but additionally records whether <see cref="Dispose"/> was ever called while a call was
    /// still in flight — the use-after-free the #132 drain-on-dispose fix prevents.
    /// </summary>
    private sealed class DrainTrackingForwardPass : IForwardPass
    {
        private readonly float[] _logits = [1.0f, 0.0f]; // greedy → token 0 (non-EOS)
        private readonly Lock _lock = new();
        private readonly ManualResetEventSlim _release = new(false);

        public readonly ManualResetEventSlim EnteredForward = new(false);

        private int _active;
        public bool Disposed { get; private set; }
        public bool DisposedWhileActive { get; private set; }

        public int VocabSize => 2;
        public int MaxSeqLen => 4096;

        private ReadOnlySpan<float> Blocking()
        {
            lock (_lock) _active++;
            try
            {
                EnteredForward.Set();
                _release.Wait(TimeSpan.FromSeconds(15));
                return _logits;
            }
            finally
            {
                lock (_lock) _active--;
            }
        }

        public void Release() => _release.Set();

        public ReadOnlySpan<float> Forward(int token, int position) => Blocking();
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => Blocking();
        public void ResetCache() { }
        public void TruncateTo(int length) { }
        public bool SupportsPartialRewind => true;

        public void Dispose()
        {
            lock (_lock)
            {
                if (_active > 0) DisposedWhileActive = true;
                Disposed = true;
            }
            _release.Dispose();
        }
    }
}
