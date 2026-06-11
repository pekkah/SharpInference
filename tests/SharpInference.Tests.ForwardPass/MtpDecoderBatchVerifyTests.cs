using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Scripted-pass coverage for the folded k-token MTP batched verify loop
/// (<see cref="MtpDecoder"/>, issues #30 / #207 goal 4). A deterministic fake
/// <see cref="IForwardPass"/> scripts the trunk's greedy chain and the MTP head's
/// drafts (with injectable disagreement positions), and asserts the decoder's
/// cache-length contracts on every call — so accept/reject bookkeeping bugs fail
/// loudly here without any model file.
///
/// Invariant under pMin = 1 (argmax-match accept): the emitted sequence equals
/// the trunk's greedy chain REGARDLESS of what the MTP head drafts — rejections
/// only cost speed. Every test asserts that first.
/// </summary>
public sealed class MtpDecoderBatchVerifyTests
{
    private const int Vocab = 16;
    private const int EmbDim = 4;

    /// <summary>Deterministic trunk chain: next token after t is (t + 1) % Vocab.</summary>
    private static int NextTarget(int t) => (t + 1) % Vocab;

    private static float[] Logits(int next)
    {
        var l = new float[Vocab];
        l[next] = 1f;
        return l;
    }

    /// <summary>
    /// Scripted MTP-capable pass. Trunk chain is <see cref="NextTarget"/>; the MTP
    /// head drafts the same chain except at positions listed in
    /// <see cref="RejectAtPositions"/> (where it proposes t+2, which the verify
    /// rejects). Asserts the MtpDecoder's length contracts on every call.
    /// </summary>
    private sealed class ScriptedMtpPass : IForwardPass
    {
        public bool BatchVerifySupported = true;
        public int MaxBatch = 8;
        public HashSet<int> RejectAtPositions = new();

        public int MainLen;            // trunk cache length
        public int MtpLen;             // MTP KV cache length
        public int HistLen;            // hidden-history length

        public int ForwardCalls;
        public int BatchVerifyCalls;
        public readonly List<int> BatchVerifyKs = new();
        public readonly List<int> RestoreCalls = new();
        // (position, prevHiddenMarker) for every MtpForward — markers: trunk hidden
        // at pos p encodes p; MTP self-hidden at draft pos p encodes -(p + 1000).
        public readonly List<(int Pos, float PrevMarker)> MtpCalls = new();

        private readonly float[] _lastHidden = new float[EmbDim];
        private readonly float[] _mtpSelfHidden = new float[EmbDim];
        private readonly Dictionary<int, float[]> _hist = new();

        public ScriptedMtpPass(int prefillLen)
        {
            MainLen = prefillLen;
            MtpLen = prefillLen;
            HistLen = prefillLen;
            _lastHidden[0] = prefillLen - 1;   // trunk hidden marker for h@prefillLen-1
        }

        public int VocabSize => Vocab;
        public int MaxSeqLen => 4096;
        public bool HasMtpHead => true;
        public bool SupportsBatchVerify => BatchVerifySupported;
        public int MaxBatchVerifyTokens => MaxBatch;
        public ReadOnlySpan<float> LastHidden => _lastHidden;
        public ReadOnlySpan<float> MtpLastHidden => _mtpSelfHidden;

        public ReadOnlySpan<float> HiddenAt(int position)
        {
            if (position < 0 || position >= HistLen) return default;
            return _hist.TryGetValue(position, out var h) ? h : default;
        }

        private void WriteHist(int position)
        {
            var h = new float[EmbDim];
            h[0] = position;
            _hist[position] = h;
            if (HistLen < position + 1) HistLen = position + 1;
        }

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            Assert.Equal(MainLen, position);
            ForwardCalls++;
            MainLen++;
            WriteHist(position);
            _lastHidden[0] = position;
            return Logits(NextTarget(token));
        }

        public float[][] BatchVerify(int[] tokens, int startPos)
        {
            Assert.True(BatchVerifySupported, "BatchVerify called while unsupported");
            Assert.True(tokens.Length <= MaxBatch,
                $"BatchVerify k={tokens.Length} exceeds MaxBatchVerifyTokens={MaxBatch}");
            Assert.Equal(MainLen, startPos);
            BatchVerifyCalls++;
            BatchVerifyKs.Add(tokens.Length);
            var result = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++)
            {
                result[i] = Logits(NextTarget(tokens[i]));
                WriteHist(startPos + i);
            }
            MainLen += tokens.Length;
            _lastHidden[0] = startPos + tokens.Length - 1;
            return result;
        }

        public void RestoreBatchSnapshot(int lengthAfter)
        {
            RestoreCalls.Add(lengthAfter);
            Assert.True(lengthAfter < MainLen,
                "RestoreBatchSnapshot must rewind, not extend, the trunk cache.");
            MainLen = lengthAfter;
            if (MtpLen > lengthAfter) MtpLen = lengthAfter;
            if (HistLen > lengthAfter) HistLen = lengthAfter;
        }

        public ReadOnlySpan<float> MtpForward(int token, int position, ReadOnlySpan<float> prevHidden)
        {
            Assert.Equal(MtpLen, position);
            MtpCalls.Add((position, prevHidden[0]));
            MtpLen++;
            _mtpSelfHidden[0] = -(position + 1000);
            int draft = RejectAtPositions.Contains(position + 1)
                ? (token + 2) % Vocab
                : NextTarget(token);
            return Logits(draft);
        }

        public void MtpTruncateTo(int length)
        {
            if (MtpLen > length) MtpLen = length;
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => new float[Vocab];
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public void Dispose() { }
    }

    private static List<int> Decode(ScriptedMtpPass pass, int prefillLen, int firstToken,
                                    int maxTokens, int draftN, int[]? stops = null)
    {
        var dec = new MtpDecoder(pass);
        dec.Initialize(prefillLen, Logits(firstToken));
        var emitted = new List<int>();
        dec.Decode(maxTokens, stops ?? [], emitted.Add, pMin: 1f, draftN: draftN);
        return emitted;
    }

    private static List<int> TargetChain(int firstToken, int count)
    {
        var chain = new List<int>(count) { firstToken };
        while (chain.Count < count) chain.Add(NextTarget(chain[^1]));
        return chain;
    }

    [Fact]
    public void AllAccept_EmitsTargetChain_OneBatchPerStep_NoForward()
    {
        var pass = new ScriptedMtpPass(prefillLen: 10);
        var emitted = Decode(pass, 10, firstToken: 3, maxTokens: 12, draftN: 3);

        Assert.Equal(TargetChain(3, 12), emitted);
        // 12 tokens at 4 tokens/step (all drafts accepted) = 3 batched passes.
        Assert.Equal(3, pass.BatchVerifyCalls);
        Assert.All(pass.BatchVerifyKs, k => Assert.Equal(4, k));
        Assert.Empty(pass.RestoreCalls);
        Assert.Equal(0, pass.ForwardCalls);   // the fold: zero per-token trunk forwards
    }

    [Fact]
    public void RejectEveryDraft_StillEmitsTargetChain_WithoutCorrectionForwards()
    {
        var pass = new ScriptedMtpPass(prefillLen: 10);
        // Reject the FIRST draft of every step: drafts predict positions 11, 12, ...
        for (int p = 0; p < 64; p++) pass.RejectAtPositions.Add(p);

        var emitted = Decode(pass, 10, firstToken: 3, maxTokens: 8, draftN: 3);

        // pMin=1 invariant: corrections ride into the next batch, output unchanged.
        Assert.Equal(TargetChain(3, 8), emitted);
        // Every step emits only t1 → one batch per emitted token, EXCEPT the final
        // token: with 1 budget left the step degrades to the sequential tail, which
        // emits the pending argmax without any trunk pass. Every batched step rolls
        // back to P+1 (zero accepted drafts); the target never runs Forward.
        Assert.Equal(7, pass.BatchVerifyCalls);
        Assert.Equal(7, pass.RestoreCalls.Count);
        Assert.Equal(0, pass.ForwardCalls);
    }

    [Fact]
    public void MidChainReject_RollsBackToAcceptedBoundary()
    {
        var pass = new ScriptedMtpPass(prefillLen: 10);
        // Step 1 verifies positions [10, 14); drafts predict 11, 12, 13.
        // Reject the draft predicting 13 → 2 drafts accepted, rollback to 13.
        pass.RejectAtPositions.Add(13);

        var emitted = Decode(pass, 10, firstToken: 3, maxTokens: 6, draftN: 3);

        Assert.Equal(TargetChain(3, 6), emitted);
        Assert.Equal(13, Assert.Single(pass.RestoreCalls));
    }

    [Fact]
    public void StopToken_NotEmitted_DecodeEnds_StateConsistent()
    {
        var pass = new ScriptedMtpPass(prefillLen: 10);
        // Chain from 3: 3, 4, 5, 6, ... — stop at 6. Step 1 verifies [10, 14):
        // t1=3@10, drafts 4@11, 5@12, 6@13; the accepted stop (6) clamps
        // acceptance at a=2 → rollback to 13, emit 4 and 5, end decode.
        var emitted = Decode(pass, 10, firstToken: 3, maxTokens: 12, draftN: 3, stops: [6]);
        Assert.Equal(new List<int> { 3, 4, 5 }, emitted);

        // The accepted-stop boundary must leave EVERY cache exactly at the last
        // emitted position + 1 (13) — the stop is neither emitted nor committed.
        // Pre-#208-review the trunk/MTP caches were stranded at P+1+a past
        // _nextPos, poisoning the GDN recurrence for any follow-up use.
        Assert.Equal(13, Assert.Single(pass.RestoreCalls));
        Assert.Equal(13, pass.MainLen);
        Assert.True(pass.MtpLen <= 13,
            $"MTP KV must not hold positions past the stop boundary (len={pass.MtpLen}).");
        Assert.Equal(13, pass.HistLen);
    }

    [Fact]
    public void CapabilityOff_FallsBackToSequential()
    {
        var pass = new ScriptedMtpPass(prefillLen: 10) { BatchVerifySupported = false };
        var emitted = Decode(pass, 10, firstToken: 3, maxTokens: 6, draftN: 3);

        Assert.Equal(TargetChain(3, 6), emitted);
        Assert.Equal(0, pass.BatchVerifyCalls);
        Assert.True(pass.ForwardCalls > 0, "sequential fallback must drive Forward");
    }

    [Fact]
    public void MaxBatchVerifyTokens_ClampsTheChain()
    {
        var pass = new ScriptedMtpPass(prefillLen: 10) { MaxBatch = 2 };
        var emitted = Decode(pass, 10, firstToken: 3, maxTokens: 8, draftN: 5);

        Assert.Equal(TargetChain(3, 8), emitted);
        // Every batch clamped to the ring capacity (the fake also hard-asserts ≤ MaxBatch).
        Assert.All(pass.BatchVerifyKs, k => Assert.Equal(2, k));
    }

    [Fact]
    public void DraftN1_MatchesLegacyTwoTokenShape()
    {
        var pass = new ScriptedMtpPass(prefillLen: 10);
        var emitted = Decode(pass, 10, firstToken: 3, maxTokens: 8, draftN: 1);

        Assert.Equal(TargetChain(3, 8), emitted);
        Assert.All(pass.BatchVerifyKs, k => Assert.Equal(2, k));
    }

    [Fact]
    public void DraftChain_SelfChains_AndRefreshUsesTrunkHiddens()
    {
        var pass = new ScriptedMtpPass(prefillLen: 10);
        _ = Decode(pass, 10, firstToken: 3, maxTokens: 4, draftN: 3);

        // One step: chain at positions 10, 11, 12 then (all-accept) refresh at 11, 12, 13.
        Assert.Equal(6, pass.MtpCalls.Count);

        // Chain call 1 (pos 10): prevHidden = trunk h@9 (marker 9).
        Assert.Equal((10, 9f), pass.MtpCalls[0]);
        // Chain calls 2-3 self-chain on the MTP block hidden (marker -(pos-1 + 1000)).
        Assert.Equal((11, -1010f), pass.MtpCalls[1]);
        Assert.Equal((12, -1011f), pass.MtpCalls[2]);
        // Refresh rewrites accepted positions with TRUNK hiddens h@10..h@12.
        Assert.Equal((11, 10f), pass.MtpCalls[3]);
        Assert.Equal((12, 11f), pass.MtpCalls[4]);
        Assert.Equal((13, 12f), pass.MtpCalls[5]);
    }

    [Fact]
    public void RemainingBudget_ShrinksTheLastBatch()
    {
        var pass = new ScriptedMtpPass(prefillLen: 10);
        // maxTokens=6 with k=4 steps: step 1 emits 4, step 2 may batch at most 2.
        var emitted = Decode(pass, 10, firstToken: 3, maxTokens: 6, draftN: 3);

        Assert.Equal(TargetChain(3, 6), emitted);
        Assert.Equal(new List<int> { 4, 2 }, pass.BatchVerifyKs);
    }
}
