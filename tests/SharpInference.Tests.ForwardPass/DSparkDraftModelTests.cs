using System.Buffers.Binary;
using System.Text;
using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Numerical and contract tests for <see cref="DSparkDraftModel"/> against an
/// independent scalar reference derived from the DeepSpec Python implementation
/// (deepspec/modeling/dspark/qwen3.py + markov_head.py + common.py):
/// fused = RMSNorm_hidden_norm(fc @ tap) per context position; per layer the
/// context K gets per-head RMSNorm then NEOX RoPE at its absolute position;
/// the block ([embed(anchor), embed(mask)...]) runs a qwen3-style layer stack
/// with BIDIRECTIONAL attention over (all context ++ whole block), GQA head
/// mapping h -> h / (heads / kvHeads), scale headDim^-0.5; base logits from
/// lm_head, then the vanilla Markov head re-biases sequentially
/// (bias[v] = dot(markov_w2[v], markov_w1[prev]), greedy argmax chain seeded
/// with the anchor token); the confidence head scores
/// sigmoid(W . [hidden_j || markov_w1[prev_j]] + b) with prev_0 = anchor.
/// Weights come from a tiny synthetic safetensors checkpoint written to a temp
/// file; BF16 tensors are round-tripped through bf16 truncation before use so
/// the reference and the implementation consume identical values.
/// </summary>
public sealed class DSparkDraftModelTests
{
    private const int Hidden = 8;
    private const int HeadDim = 4;
    private const int Heads = 2;
    private const int KvHeads = 1;
    private const int Interm = 16;
    private const int Layers = 2;
    private const int Block = 3;
    private const int Vocab = 32;
    private const int Rank = 4;
    private const int MaskToken = 31;
    private const int TapDim = 16; // target_layer_ids [0, 2] x hidden_size 8
    private const float Eps = 1e-6f;
    private const float Theta = 10000f;
    private const int MaxCtx = 16;

    // ── Tests ────────────────────────────────────────────────────────────

    [Fact]
    public void ProposeBlock_MatchesNaiveReference()
    {
        using var head = new SyntheticHead(withConfidence: true);
        using var model = head.CreateModel();

        Assert.Equal(Block, model.BlockSize);
        Assert.Equal(Vocab, model.VocabSize);
        Assert.Equal(TapDim, model.TapDim);
        Assert.Equal(MaskToken, model.MaskTokenId);
        Assert.Equal(new[] { 0, 2 }, model.TargetLayerIds);
        Assert.Equal(0, model.ContextLength);

        const int ctx = 5;
        var taps = MakeTaps(ctx, seed: 12345);
        model.AppendContext(taps, 0, ctx);
        Assert.Equal(ctx, model.ContextLength);

        var proposal = model.ProposeBlock(anchorToken: 7, anchorPos: ctx);
        var (refTokens, refConf) = ReferencePropose(head, taps, ctx, anchorToken: 7, anchorPos: ctx);

        Assert.Equal(Block, proposal.Tokens.Length);
        Assert.Equal(Block, proposal.Confidences.Length);
        Assert.Equal(refTokens, proposal.Tokens);
        for (int j = 0; j < Block; j++)
            Assert.True(Math.Abs(proposal.Confidences[j] - refConf[j]) <= 2e-3,
                $"confidence[{j}]: impl={proposal.Confidences[j]}, ref={refConf[j]}");
    }

    [Fact]
    public void AppendContext_Incremental_EqualsBatch()
    {
        using var head = new SyntheticHead(withConfidence: true);
        var taps = MakeTaps(5, seed: 777);

        using var batch = head.CreateModel();
        batch.AppendContext(taps, 0, 5);

        using var incremental = head.CreateModel();
        incremental.AppendContext(taps.AsSpan(0, 3 * TapDim), 0, 3);
        incremental.AppendContext(taps.AsSpan(3 * TapDim), 3, 2);
        Assert.Equal(5, incremental.ContextLength);

        var pBatch = batch.ProposeBlock(anchorToken: 4, anchorPos: 5);
        var pIncremental = incremental.ProposeBlock(anchorToken: 4, anchorPos: 5);

        Assert.Equal(pBatch.Tokens, pIncremental.Tokens);
        Assert.Equal(pBatch.Confidences, pIncremental.Confidences);
    }

    [Fact]
    public void TruncateContext_Reappend_EqualsFresh()
    {
        using var head = new SyntheticHead(withConfidence: true);
        var taps = MakeTaps(5, seed: 999);

        using var fresh = head.CreateModel();
        fresh.AppendContext(taps, 0, 5);
        var expected = fresh.ProposeBlock(anchorToken: 9, anchorPos: 5);

        using var truncated = head.CreateModel();
        truncated.AppendContext(taps, 0, 5);
        truncated.TruncateContext(3);
        Assert.Equal(3, truncated.ContextLength);
        truncated.AppendContext(taps.AsSpan(3 * TapDim), 3, 2);
        var actual = truncated.ProposeBlock(anchorToken: 9, anchorPos: 5);

        Assert.Equal(expected.Tokens, actual.Tokens);
        Assert.Equal(expected.Confidences, actual.Confidences);
    }

    [Fact]
    public void ContractViolations_Throw()
    {
        using var head = new SyntheticHead(withConfidence: true);
        using var model = head.CreateModel();
        var taps2 = MakeTaps(2, seed: 42);

        // AppendContext must be contiguous: startPos != ContextLength (0).
        Assert.Throws<InvalidOperationException>(() => model.AppendContext(taps2, 1, 2));

        model.AppendContext(taps2, 0, 2);

        // ProposeBlock anchorPos must equal ContextLength (2).
        Assert.Throws<InvalidOperationException>(() => model.ProposeBlock(1, 5));
        Assert.Throws<InvalidOperationException>(() => model.ProposeBlock(1, 1));

        // Wrong taps length for the given count.
        Assert.Throws<ArgumentException>(() => model.AppendContext(new float[TapDim - 1], 2, 1));

        // Context overflow past maxContextLength.
        var big = MakeTaps(MaxCtx, seed: 43);
        Assert.Throws<InvalidOperationException>(() => model.AppendContext(big, 2, MaxCtx));

        // TruncateContext outside [0, ContextLength].
        Assert.Throws<ArgumentOutOfRangeException>(() => model.TruncateContext(3));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.TruncateContext(-1));

        // Anchor token outside the vocab.
        Assert.Throws<ArgumentOutOfRangeException>(() => model.ProposeBlock(Vocab, 2));
    }

    [Fact]
    public void ConfidenceHeadAbsent_AllConfidencesOne()
    {
        using var head = new SyntheticHead(withConfidence: false);
        using var model = head.CreateModel();

        const int ctx = 4;
        var taps = MakeTaps(ctx, seed: 2026);
        model.AppendContext(taps, 0, ctx);

        var proposal = model.ProposeBlock(anchorToken: 3, anchorPos: ctx);
        Assert.All(proposal.Confidences, c => Assert.Equal(1f, c));

        // Tokens still follow the Markov-corrected argmax chain.
        var (refTokens, refConf) = ReferencePropose(head, taps, ctx, anchorToken: 3, anchorPos: ctx);
        Assert.Equal(refTokens, proposal.Tokens);
        Assert.All(refConf, c => Assert.Equal(1.0, c));
    }

    // ── Synthetic checkpoint ─────────────────────────────────────────────

    private readonly record struct TensorSpec(string Name, int[] Shape, float[] Data, bool Bf16);

    /// <summary>
    /// Deterministic tiny DSpark head: generates all weights with a fixed
    /// xorshift stream, keeps them as the reference ground truth (BF16 tensors
    /// pre-rounded), and writes a temp .safetensors file the real loader reads.
    /// </summary>
    // Internal (not private): CudaDSparkDraftModelTests reuses the same synthetic
    // checkpoint to compare the CUDA draft backbone against this CPU model.
    internal sealed class SyntheticHead : IDisposable
    {
        public string FilePath { get; }
        public DSparkConfig Config { get; }

        public readonly float[] Fc, HiddenNorm, FinalNorm, LmHead, Embed, MarkovW1, MarkovW2;
        public readonly float[]? ConfW;
        public readonly float ConfB;
        public readonly float[][] Wq, Wk, Wv, Wo, QNorm, KNorm, InNorm, FfnNorm, WGate, WUp, WDown;

        public SyntheticHead(bool withConfidence)
        {
            var rng = new XorShift(0x9E3779B9u);
            Fc = Rand(rng, Hidden * TapDim);
            HiddenNorm = RandNorm(rng, Hidden);
            FinalNorm = RandNorm(rng, Hidden);
            LmHead = Rand(rng, Vocab * Hidden);
            Embed = Bf16RoundTrip(Rand(rng, Vocab * Hidden));
            MarkovW1 = Bf16RoundTrip(Rand(rng, Vocab * Rank));
            MarkovW2 = Rand(rng, Vocab * Rank);

            Wq = new float[Layers][]; Wk = new float[Layers][]; Wv = new float[Layers][];
            Wo = new float[Layers][]; QNorm = new float[Layers][]; KNorm = new float[Layers][];
            InNorm = new float[Layers][]; FfnNorm = new float[Layers][];
            WGate = new float[Layers][]; WUp = new float[Layers][]; WDown = new float[Layers][];
            for (int l = 0; l < Layers; l++)
            {
                Wq[l] = Rand(rng, Heads * HeadDim * Hidden);
                Wk[l] = Rand(rng, KvHeads * HeadDim * Hidden);
                Wv[l] = Rand(rng, KvHeads * HeadDim * Hidden);
                Wo[l] = Rand(rng, Hidden * Heads * HeadDim);
                QNorm[l] = RandNorm(rng, HeadDim);
                KNorm[l] = RandNorm(rng, HeadDim);
                InNorm[l] = RandNorm(rng, Hidden);
                FfnNorm[l] = RandNorm(rng, Hidden);
                WGate[l] = Rand(rng, Interm * Hidden);
                WUp[l] = Rand(rng, Interm * Hidden);
                WDown[l] = Rand(rng, Hidden * Interm);
            }

            if (withConfidence)
            {
                ConfW = Rand(rng, Hidden + Rank);
                ConfB = rng.Next();
            }

            var tensors = new List<TensorSpec>
            {
                new("fc.weight", [Hidden, TapDim], Fc, false),
                new("hidden_norm.weight", [Hidden], HiddenNorm, false),
                new("norm.weight", [Hidden], FinalNorm, false),
                new("lm_head.weight", [Vocab, Hidden], LmHead, false),
                new("embed_tokens.weight", [Vocab, Hidden], Embed, true),
                new("markov_head.markov_w1.weight", [Vocab, Rank], MarkovW1, true),
                new("markov_head.markov_w2.weight", [Vocab, Rank], MarkovW2, false),
            };
            if (withConfidence)
            {
                tensors.Add(new("confidence_head.proj.weight", [1, Hidden + Rank], ConfW!, false));
                tensors.Add(new("confidence_head.proj.bias", [1], [ConfB], false));
            }
            for (int l = 0; l < Layers; l++)
            {
                tensors.Add(new($"layers.{l}.self_attn.q_proj.weight", [Heads * HeadDim, Hidden], Wq[l], false));
                tensors.Add(new($"layers.{l}.self_attn.k_proj.weight", [KvHeads * HeadDim, Hidden], Wk[l], false));
                tensors.Add(new($"layers.{l}.self_attn.v_proj.weight", [KvHeads * HeadDim, Hidden], Wv[l], false));
                tensors.Add(new($"layers.{l}.self_attn.o_proj.weight", [Hidden, Heads * HeadDim], Wo[l], false));
                tensors.Add(new($"layers.{l}.self_attn.q_norm.weight", [HeadDim], QNorm[l], false));
                tensors.Add(new($"layers.{l}.self_attn.k_norm.weight", [HeadDim], KNorm[l], false));
                tensors.Add(new($"layers.{l}.input_layernorm.weight", [Hidden], InNorm[l], false));
                tensors.Add(new($"layers.{l}.post_attention_layernorm.weight", [Hidden], FfnNorm[l], false));
                tensors.Add(new($"layers.{l}.mlp.gate_proj.weight", [Interm, Hidden], WGate[l], false));
                tensors.Add(new($"layers.{l}.mlp.up_proj.weight", [Interm, Hidden], WUp[l], false));
                tensors.Add(new($"layers.{l}.mlp.down_proj.weight", [Hidden, Interm], WDown[l], false));
            }

            FilePath = Path.Combine(Path.GetTempPath(), $"dspark-test-{Guid.NewGuid():N}.safetensors");
            WriteSafetensors(FilePath, tensors);

            string confFlag = withConfidence ? "true" : "false";
            Config = DSparkConfig.FromJson($$"""
                {
                    "hidden_size": {{Hidden}},
                    "head_dim": {{HeadDim}},
                    "num_attention_heads": {{Heads}},
                    "num_key_value_heads": {{KvHeads}},
                    "intermediate_size": {{Interm}},
                    "num_hidden_layers": {{Layers}},
                    "block_size": {{Block}},
                    "mask_token_id": {{MaskToken}},
                    "target_layer_ids": [0, 2],
                    "num_target_layers": 4,
                    "markov_rank": {{Rank}},
                    "markov_head_type": "vanilla",
                    "enable_confidence_head": {{confFlag}},
                    "confidence_head_with_markov": {{confFlag}},
                    "vocab_size": {{Vocab}},
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10000.0,
                    "max_position_embeddings": 64
                }
                """);
        }

        public DSparkDraftModel CreateModel(int maxContextLength = MaxCtx)
        {
            using var st = SafetensorsLoader.Open(FilePath);
            return new DSparkDraftModel(Config, st, maxContextLength);
        }

        public CudaDSparkDraftModel CreateCudaModel(Cuda.CudaBackend gpu, int maxContextLength = MaxCtx)
        {
            using var st = SafetensorsLoader.Open(FilePath);
            return new CudaDSparkDraftModel(Config, st, gpu, maxContextLength);
        }

        public void Dispose()
        {
            try { File.Delete(FilePath); } catch (IOException) { }
        }
    }

    private static void WriteSafetensors(string path, IReadOnlyList<TensorSpec> tensors)
    {
        var header = new StringBuilder();
        header.Append('{');
        long offset = 0;
        for (int t = 0; t < tensors.Count; t++)
        {
            var (name, shape, data, bf16) = tensors[t];
            long bytes = (long)data.Length * (bf16 ? 2 : 4);
            if (t > 0) header.Append(',');
            header.Append('"').Append(name).Append("\":{\"dtype\":\"").Append(bf16 ? "BF16" : "F32")
                .Append("\",\"shape\":[").Append(string.Join(',', shape))
                .Append("],\"data_offsets\":[").Append(offset).Append(',').Append(offset + bytes).Append("]}");
            offset += bytes;
        }
        header.Append('}');

        byte[] headerBytes = Encoding.UTF8.GetBytes(header.ToString());
        using var fs = File.Create(path);
        Span<byte> scratch = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(scratch, (ulong)headerBytes.Length);
        fs.Write(scratch);
        fs.Write(headerBytes);
        foreach (var spec in tensors)
        {
            if (spec.Bf16)
            {
                foreach (float v in spec.Data)
                {
                    BinaryPrimitives.WriteUInt16LittleEndian(scratch,
                        (ushort)(BitConverter.SingleToInt32Bits(v) >> 16));
                    fs.Write(scratch[..2]);
                }
            }
            else
            {
                foreach (float v in spec.Data)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(scratch, v);
                    fs.Write(scratch[..4]);
                }
            }
        }
    }

    // ── Deterministic weight/tap generation ──────────────────────────────

    private sealed class XorShift(uint seed)
    {
        private uint _s = seed;

        /// <summary>Uniform in [-0.2, 0.2].</summary>
        public float Next()
        {
            _s ^= _s << 13; _s ^= _s >> 17; _s ^= _s << 5;
            return ((_s >> 8) * (1f / 16777216f) - 0.5f) * 0.4f;
        }
    }

    private static float[] Rand(XorShift rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = rng.Next();
        return a;
    }

    /// <summary>Norm weights near 1.0 (in [0.95, 1.05]).</summary>
    private static float[] RandNorm(XorShift rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 1f + rng.Next() * 0.25f;
        return a;
    }

    /// <summary>
    /// Truncate to bf16 (drop the low 16 mantissa bits) and back, so values
    /// written as BF16 are exactly representable and the reference sees the
    /// same numbers the implementation dequantizes.
    /// </summary>
    private static float[] Bf16RoundTrip(float[] a)
    {
        for (int i = 0; i < a.Length; i++)
            a[i] = BitConverter.Int32BitsToSingle((BitConverter.SingleToInt32Bits(a[i]) >> 16) << 16);
        return a;
    }

    internal static float[] MakeTaps(int count, uint seed)
    {
        var rng = new XorShift(seed);
        var taps = new float[count * TapDim];
        for (int i = 0; i < taps.Length; i++) taps[i] = rng.Next() * 5f; // ~[-1, 1]
        return taps;
    }

    // ── Naive scalar reference (double accumulation) ─────────────────────

    private static (int[] Tokens, double[] Confidences) ReferencePropose(
        SyntheticHead h, float[] taps, int ctxCount, int anchorToken, int anchorPos)
    {
        const int qDim = Heads * HeadDim;
        const int kvDim = KvHeads * HeadDim;
        const int kvGroup = Heads / KvHeads;
        double scale = 1.0 / Math.Sqrt(HeadDim);

        // Context K/V per layer: fused = RMSNorm_hidden_norm(fc @ tap), then
        // K = rope(perHeadRms(k_proj @ fused), pos), V = v_proj @ fused.
        var kCtx = new double[Layers][][];
        var vCtx = new double[Layers][][];
        for (int l = 0; l < Layers; l++) { kCtx[l] = new double[ctxCount][]; vCtx[l] = new double[ctxCount][]; }
        for (int p = 0; p < ctxCount; p++)
        {
            var tap = Row(taps, p, TapDim);
            var fused = Rms(MatVec(h.Fc, tap, Hidden, TapDim), h.HiddenNorm);
            for (int l = 0; l < Layers; l++)
            {
                var k = MatVec(h.Wk[l], fused, kvDim, Hidden);
                PerHeadRms(k, h.KNorm[l], KvHeads);
                RopeRef(k, p, KvHeads);
                kCtx[l][p] = k;
                vCtx[l][p] = MatVec(h.Wv[l], fused, kvDim, Hidden);
            }
        }

        // Block inputs: [embed(anchor), embed(mask) x (Block-1)].
        var x = new double[Block][];
        x[0] = Row(h.Embed, anchorToken, Hidden);
        for (int j = 1; j < Block; j++) x[j] = Row(h.Embed, MaskToken, Hidden);

        for (int l = 0; l < Layers; l++)
        {
            // Attention.
            var resid = new double[Block][];
            var q = new double[Block][];
            var kB = new double[Block][];
            var vB = new double[Block][];
            for (int j = 0; j < Block; j++)
            {
                resid[j] = (double[])x[j].Clone();
                var n = Rms(x[j], h.InNorm[l]);
                q[j] = MatVec(h.Wq[l], n, qDim, Hidden);
                PerHeadRms(q[j], h.QNorm[l], Heads);
                RopeRef(q[j], anchorPos + j, Heads);
                kB[j] = MatVec(h.Wk[l], n, kvDim, Hidden);
                PerHeadRms(kB[j], h.KNorm[l], KvHeads);
                RopeRef(kB[j], anchorPos + j, KvHeads);
                vB[j] = MatVec(h.Wv[l], n, kvDim, Hidden);
            }

            for (int j = 0; j < Block; j++)
            {
                // Bidirectional GQA attention over (all context ++ whole block).
                var attn = new double[qDim];
                for (int hh = 0; hh < Heads; hh++)
                {
                    int qOff = hh * HeadDim;
                    int kvOff = (hh / kvGroup) * HeadDim;
                    var s = new double[ctxCount + Block];
                    for (int c = 0; c < ctxCount; c++)
                    {
                        double dot = 0;
                        for (int d = 0; d < HeadDim; d++) dot += q[j][qOff + d] * kCtx[l][c][kvOff + d];
                        s[c] = dot * scale;
                    }
                    for (int c = 0; c < Block; c++)
                    {
                        double dot = 0;
                        for (int d = 0; d < HeadDim; d++) dot += q[j][qOff + d] * kB[c][kvOff + d];
                        s[ctxCount + c] = dot * scale;
                    }
                    SoftmaxRef(s);
                    for (int c = 0; c < ctxCount; c++)
                        for (int d = 0; d < HeadDim; d++) attn[qOff + d] += s[c] * vCtx[l][c][kvOff + d];
                    for (int c = 0; c < Block; c++)
                        for (int d = 0; d < HeadDim; d++) attn[qOff + d] += s[ctxCount + c] * vB[c][kvOff + d];
                }
                var o = MatVec(h.Wo[l], attn, Hidden, qDim);
                for (int i = 0; i < Hidden; i++) x[j][i] = resid[j][i] + o[i];
            }

            // FFN (SwiGLU): x += down(silu(gate(n)) * up(n)).
            for (int j = 0; j < Block; j++)
            {
                var n = Rms(x[j], h.FfnNorm[l]);
                var gate = MatVec(h.WGate[l], n, Interm, Hidden);
                var up = MatVec(h.WUp[l], n, Interm, Hidden);
                var act = new double[Interm];
                for (int i = 0; i < Interm; i++)
                    act[i] = gate[i] / (1.0 + Math.Exp(-gate[i])) * up[i];
                var down = MatVec(h.WDown[l], act, Hidden, Interm);
                for (int i = 0; i < Hidden; i++) x[j][i] += down[i];
            }
        }

        // Final norm, base logits, sequential Markov correction (greedy).
        var hidden = new double[Block][];
        for (int j = 0; j < Block; j++) hidden[j] = Rms(x[j], h.FinalNorm);

        var tokens = new int[Block];
        int prev = anchorToken;
        for (int j = 0; j < Block; j++)
        {
            var logits = MatVec(h.LmHead, hidden[j], Vocab, Hidden);
            var w1 = Row(h.MarkovW1, prev, Rank);
            for (int v = 0; v < Vocab; v++)
            {
                double bias = 0;
                for (int r = 0; r < Rank; r++) bias += (double)h.MarkovW2[v * Rank + r] * w1[r];
                logits[v] += bias;
            }
            tokens[j] = ArgMaxRef(logits);
            prev = tokens[j];
        }

        // Confidence: sigmoid(W . [hidden_j || markov_w1[prev_j]] + b),
        // prev_0 = anchor, prev_j = tokens[j-1]. All 1.0 without the head.
        var conf = new double[Block];
        if (h.ConfW is null)
        {
            Array.Fill(conf, 1.0);
        }
        else
        {
            for (int j = 0; j < Block; j++)
            {
                int prevJ = j == 0 ? anchorToken : tokens[j - 1];
                double logit = h.ConfB;
                for (int i = 0; i < Hidden; i++) logit += (double)h.ConfW[i] * hidden[j][i];
                var w1 = Row(h.MarkovW1, prevJ, Rank);
                for (int r = 0; r < Rank; r++) logit += (double)h.ConfW[Hidden + r] * w1[r];
                conf[j] = 1.0 / (1.0 + Math.Exp(-logit));
            }
        }

        return (tokens, conf);
    }

    private static double[] MatVec(float[] w, double[] x, int rows, int cols)
    {
        var r = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            double sum = 0;
            int b = i * cols;
            for (int c = 0; c < cols; c++) sum += (double)w[b + c] * x[c];
            r[i] = sum;
        }
        return r;
    }

    private static double[] Row(float[] table, int row, int cols)
    {
        var r = new double[cols];
        for (int i = 0; i < cols; i++) r[i] = table[row * cols + i];
        return r;
    }

    /// <summary>RMSNorm: x * (1 / sqrt(mean(x^2) + eps)) * w.</summary>
    private static double[] Rms(double[] x, float[] w)
    {
        double ss = 0;
        foreach (double v in x) ss += v * v;
        double scale = 1.0 / Math.Sqrt(ss / x.Length + Eps);
        var r = new double[x.Length];
        for (int i = 0; i < x.Length; i++) r[i] = x[i] * scale * w[i];
        return r;
    }

    /// <summary>Per-head RMSNorm over headDim slices with a shared [headDim] weight.</summary>
    private static void PerHeadRms(double[] x, float[] w, int heads)
    {
        for (int h = 0; h < heads; h++)
        {
            int off = h * HeadDim;
            double ss = 0;
            for (int i = 0; i < HeadDim; i++) ss += x[off + i] * x[off + i];
            double scale = 1.0 / Math.Sqrt(ss / HeadDim + Eps);
            for (int i = 0; i < HeadDim; i++) x[off + i] = x[off + i] * scale * w[i];
        }
    }

    /// <summary>
    /// NEOX / rotate_half RoPE: pair (i, i + headDim/2) rotated by
    /// pos * theta^(-2i/headDim). Angles are computed in float to match the
    /// implementation's precomputed table exactly.
    /// </summary>
    private static void RopeRef(double[] x, int pos, int heads)
    {
        const int half = HeadDim / 2;
        for (int h = 0; h < heads; h++)
        {
            int off = h * HeadDim;
            for (int i = 0; i < half; i++)
            {
                float freq = 1.0f / MathF.Pow(Theta, 2.0f * i / HeadDim);
                float angle = pos * freq;
                double c = MathF.Cos(angle);
                double s = MathF.Sin(angle);
                double x0 = x[off + i], x1 = x[off + i + half];
                x[off + i] = x0 * c - x1 * s;
                x[off + i + half] = x0 * s + x1 * c;
            }
        }
    }

    private static void SoftmaxRef(double[] s)
    {
        double max = double.NegativeInfinity;
        foreach (double v in s) if (v > max) max = v;
        double sum = 0;
        for (int i = 0; i < s.Length; i++) { s[i] = Math.Exp(s[i] - max); sum += s[i]; }
        for (int i = 0; i < s.Length; i++) s[i] /= sum;
    }

    private static int ArgMaxRef(double[] x)
    {
        int idx = 0;
        double max = x[0];
        for (int i = 1; i < x.Length; i++)
            if (x[i] > max) { max = x[i]; idx = i; }
        return idx;
    }
}
