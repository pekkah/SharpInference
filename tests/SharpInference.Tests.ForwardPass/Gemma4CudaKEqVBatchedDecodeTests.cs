using System.Buffers.Binary;
using System.Text;
using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #276: coverage for the Gemma 4 12B-global <c>attention_k_eq_v</c> path in the BATCHED
/// decode (<c>GpuLayerBatchedDecodeGemma4</c> — <c>if (kEqV) CopyDevice(vAll, kAll)</c>, V reuses the
/// raw K projection). That branch is 12B-global-only and had no runnable test: the only local 12B is
/// Q4_0 (not GEMM-N-batchable → single-user fallback), and the E4B Q8_0 fixture
/// (<see cref="Gemma4CudaBatchForwardMultiTests"/>) exercises per-layer head_dim / SWA / shared-KV /
/// PLE but never k_eq_v (E4B carries attn_v). A GEMM-N-batchable 12B doesn't exist locally and a
/// Q8_0 12B (~13 GB) wouldn't fit the 4070 Ti's 12 GB at a useful context — so this uses the issue's
/// option 2: a tiny SYNTHETIC F32 Gemma-4 GGUF that triggers the k_eq_v global-layer geometry.
///
/// <para>The fixture is intentionally minimal — all-global layers (no <c>sliding_window_pattern</c>),
/// no PLE / QK-norm / sandwich norms / softcap. A layer that omits <c>attn_v</c> takes the k_eq_v
/// branch (<c>_wv[layer]</c> null → V = raw K projection, then V-norm); a layer that keeps
/// <c>attn_v</c> takes the normal real-V branch. F32 weights make it GEMM-N-batchable, so
/// <see cref="CudaForwardPass.SupportsContinuousBatching"/> is true. Random weights are fine: the
/// oracle compares the batched decode to the single-user <c>ForwardGemma4</c> loop on the SAME model,
/// so both paths read the same weights — it isolates the batched-vs-single-token wiring of the V=rawK
/// copy + attention, which is exactly the uncovered branch. Argmax-stable within the fp32-WS
/// tolerance, mirroring the E4B oracles. Silent-skips when CUDA is absent.</para>
///
/// <para>Coverage: the all-k_eq_v fixture (every layer copies) and the MIXED fixture (one real-V
/// layer + one k_eq_v layer) together exercise the per-layer <c>kEqV = … &amp;&amp; _wv[layer] is null</c>
/// dispatch — the real risk in <c>GpuLayerBatchedDecodeGemma4</c>. The per-layer dispatch keys on
/// attn_v presence, NOT on SWA, so an all-global mixed model reaches both branches without the SWA
/// machinery. The SWA-vs-global axis is orthogonal and already covered by the E4B oracles
/// (<see cref="Gemma4CudaBatchForwardMultiTests"/>); the realistic 12B pairing of k_eq_v-on-global +
/// real-V-on-SWA is therefore not reproduced here, but each branch and their selection are.</para>
/// </summary>
public sealed class Gemma4CudaKEqVBatchedDecodeTests : IDisposable
{
    // Tiny but valid Gemma-4 geometry. embDim = numHeads*headDim keeps the projections square; all
    // dims are clean powers of two. kvHeads < heads exercises GQA in the k_eq_v attention.
    private const int EmbDim = 256, NumHeads = 4, HeadDim = 64, NumKvHeads = 2, FfnDim = 512;
    private const int Vocab = 256, NumLayers = 2, Context = 256;
    private const int QDim = NumHeads * HeadDim;     // 256
    private const int KvDim = NumKvHeads * HeadDim;  // 128

    private readonly List<string> _tempFiles = new();
    public void Dispose()
    {
        foreach (var f in _tempFiles)
            try { File.Delete(f); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best-effort temp cleanup */ }
    }

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0; float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static float MaxAbs(float[] a, float[] b)
    {
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private static int Overlap(float[] a, float[] b, int k)
    {
        static HashSet<int> TopK(float[] x, int kk)
        {
            var idx = new int[x.Length];
            for (int i = 0; i < idx.Length; i++) idx[i] = i;
            Array.Sort(idx, (p, q) => x[q].CompareTo(x[p]));
            var s = new HashSet<int>();
            for (int i = 0; i < kk && i < idx.Length; i++) s.Add(idx[i]);
            return s;
        }
        var ra = TopK(a, k);
        int o = 0;
        foreach (var t in TopK(b, k)) if (ra.Contains(t)) o++;
        return o;
    }

    /// <summary>Argmax parity tolerant of a precision-driven near-tie (single-token MatMul vs batched
    /// WS matvec round differently), accepted ONLY when provably a near-tie in the reference.</summary>
    private static void AssertArgmaxOrNearTie(float[] reference, float[] candidate, float tieEps, string label)
    {
        int rArg = Argmax(reference), cArg = Argmax(candidate);
        if (rArg == cArg) return;
        float gap = MathF.Abs(reference[rArg] - reference[cArg]);
        Assert.True(gap < tieEps,
            $"{label}: batched argmax {cArg} != single-user {rArg}, NOT a near-tie (reference gap {gap:F3} ≥ {tieEps:F1}) " +
            "— a real k_eq_v wiring divergence (V=rawK copy / per-layer attention), not fp32-WS rounding.");
    }

    /// <summary>The synthetic model loads as a k_eq_v Gemma-4 model that supports continuous batching,
    /// and the k_eq_v invariant holds: every layer is global (not SWA) and carries no attn_v tensor.</summary>
    [Fact]
    public void SyntheticKEqV_Loads_AsBatchableGemma4_WithNoAttnV()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = WriteSyntheticGguf(seed: 1234);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.True(hp.AttentionKEqV, "fixture must set _sharpi.attention_k_eq_v.");
        Assert.NotNull(hp.LayerHeadDim);                 // → _isGemma4Like
        Assert.NotNull(hp.IsSwaLayer);
        for (int i = 0; i < NumLayers; i++)
        {
            Assert.False(hp.IsSwaLayer![i], $"layer {i} must be global (no sliding_window_pattern → all global).");
            Assert.Null(model.FindTensor($"blk.{i}.attn_v.weight")); // k_eq_v omits V → kEqV triggers
        }

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: Context);
        Assert.True(fwd.SupportsContinuousBatching,
            "F32 all-global k_eq_v Gemma-4 should support continuous batching (#195/#276).");
    }

    /// <summary>Headline #276 oracle: one batched <see cref="CudaForwardPass.BatchForwardMulti"/>
    /// decode step over two sequences must reproduce two independent single-user
    /// prefill+decode passes — driving the k_eq_v V=rawK copy + per-sequence attention through the
    /// batched path and comparing it to the single-token <c>ForwardGemma4</c> loop on the same model.</summary>
    [Fact]
    public void SyntheticKEqV_BatchedDecode_N2_MatchesSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = WriteSyntheticGguf(seed: 99);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: Context);
        Assert.True(fwd.SupportsContinuousBatching);

        int[] promptA = { 5, 17, 200, 33, 41, 9 };
        int[] promptB = { 7, 100, 13, 222, 4, 88, 150 };

        // Single-user references: prefill, greedy token, one decode step (single-token k_eq_v path).
        fwd.ResetCache();
        int tokA = Argmax(fwd.Prefill(promptA));
        float[] refA = fwd.Forward(tokA, promptA.Length).ToArray();
        fwd.ResetCache();
        int tokB = Argmax(fwd.Prefill(promptB));
        float[] refB = fwd.Forward(tokB, promptB.Length).ToArray();

        // Batched: per-sequence caches, one batched decode step (batched k_eq_v path).
        using var cacheA = fwd.CreateCache();
        using var cacheB = fwd.CreateCache();
        fwd.PrefillWithCache(promptA, cacheA);
        fwd.PrefillWithCache(promptB, cacheB);

        float[][] batch = fwd.BatchForwardMulti(
            [tokA, tokB], [promptA.Length, promptB.Length], [cacheA, cacheB]);
        Assert.Equal(2, batch.Length);

        AssertArgmaxOrNearTie(refA, batch[0], tieEps: 0.5f, "Seq A");
        Assert.True(Overlap(refA, batch[0], 5) >= 4, $"Seq A top-5 overlap (maxAbs={MaxAbs(refA, batch[0])}).");
        Assert.True(MaxAbs(refA, batch[0]) < 0.5f, $"Seq A maxAbs={MaxAbs(refA, batch[0])}.");

        AssertArgmaxOrNearTie(refB, batch[1], tieEps: 0.5f, "Seq B");
        Assert.True(Overlap(refB, batch[1], 5) >= 4, $"Seq B top-5 overlap (maxAbs={MaxAbs(refB, batch[1])}).");
        Assert.True(MaxAbs(refB, batch[1]) < 0.5f, $"Seq B maxAbs={MaxAbs(refB, batch[1])}.");
    }

    /// <summary>Mixed per-layer dispatch: one real-V layer (attn_v present → <c>kEqV</c> false →
    /// <c>BatchDecodeMatMul(vAll, _wv[layer], …)</c>) and one k_eq_v layer (attn_v absent → copy) in
    /// the SAME model. The batched decode must select the right branch per layer and still match the
    /// single-user loop — guards the <c>kEqV = … &amp;&amp; _wv[layer] is null</c> per-layer predicate
    /// that the all-k_eq_v fixture can't (every layer is identical there).</summary>
    [Fact]
    public void SyntheticKEqV_MixedRealVAndKEqV_BatchedDecode_MatchesSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Layer 0 omits attn_v (k_eq_v branch); layer 1 keeps it (real-V branch). The k_eq_v layer is
        // FIRST so a broken K→V copy propagates through the remaining layer — a stronger discriminator.
        bool[] layerHasV = { false, true };
        var path = WriteSyntheticGguf(seed: 4242, layerHasV);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.True(hp.AttentionKEqV);
        Assert.Null(model.FindTensor("blk.0.attn_v.weight"));    // k_eq_v layer
        Assert.NotNull(model.FindTensor("blk.1.attn_v.weight")); // real-V layer
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: Context);
        Assert.True(fwd.SupportsContinuousBatching);

        int[] promptA = { 5, 17, 200, 33, 41, 9 };
        int[] promptB = { 7, 100, 13, 222, 4, 88, 150 };

        fwd.ResetCache();
        int tokA = Argmax(fwd.Prefill(promptA));
        float[] refA = fwd.Forward(tokA, promptA.Length).ToArray();
        fwd.ResetCache();
        int tokB = Argmax(fwd.Prefill(promptB));
        float[] refB = fwd.Forward(tokB, promptB.Length).ToArray();

        using var cacheA = fwd.CreateCache();
        using var cacheB = fwd.CreateCache();
        fwd.PrefillWithCache(promptA, cacheA);
        fwd.PrefillWithCache(promptB, cacheB);

        float[][] batch = fwd.BatchForwardMulti(
            [tokA, tokB], [promptA.Length, promptB.Length], [cacheA, cacheB]);

        AssertArgmaxOrNearTie(refA, batch[0], tieEps: 0.5f, "Seq A (mixed)");
        Assert.True(MaxAbs(refA, batch[0]) < 0.5f, $"Seq A maxAbs={MaxAbs(refA, batch[0])}.");
        AssertArgmaxOrNearTie(refB, batch[1], tieEps: 0.5f, "Seq B (mixed)");
        Assert.True(MaxAbs(refB, batch[1]) < 0.5f, $"Seq B maxAbs={MaxAbs(refB, batch[1])}.");
    }

    /// <summary>Two batched decode steps (positions advance) must track the single-user continuation —
    /// catches a k_eq_v KV-append / position bug that a single step would miss (the first step's
    /// K-as-V is reused, a second token appended at the new position).</summary>
    [Fact]
    public void SyntheticKEqV_BatchedDecode_TwoSteps_MatchSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = WriteSyntheticGguf(seed: 7);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: Context);

        int[] promptA = { 5, 17, 200, 33, 41, 9 };
        int[] promptB = { 7, 100, 13, 222, 4, 88, 150 };

        fwd.ResetCache();
        int a0 = Argmax(fwd.Prefill(promptA));
        float[] refA1 = fwd.Forward(a0, promptA.Length).ToArray();
        int a1 = Argmax(refA1);
        float[] refA2 = fwd.Forward(a1, promptA.Length + 1).ToArray();
        fwd.ResetCache();
        int b0 = Argmax(fwd.Prefill(promptB));
        float[] refB1 = fwd.Forward(b0, promptB.Length).ToArray();
        int b1 = Argmax(refB1);
        float[] refB2 = fwd.Forward(b1, promptB.Length + 1).ToArray();

        using var cacheA = fwd.CreateCache();
        using var cacheB = fwd.CreateCache();
        fwd.PrefillWithCache(promptA, cacheA);
        fwd.PrefillWithCache(promptB, cacheB);

        var step1 = fwd.BatchForwardMulti([a0, b0], [promptA.Length, promptB.Length], [cacheA, cacheB]);
        AssertArgmaxOrNearTie(refA1, step1[0], tieEps: 0.5f, "Seq A step1");
        AssertArgmaxOrNearTie(refB1, step1[1], tieEps: 0.5f, "Seq B step1");
        int ba1 = Argmax(step1[0]), bb1 = Argmax(step1[1]);

        var step2 = fwd.BatchForwardMulti(
            [ba1, bb1], [promptA.Length + 1, promptB.Length + 1], [cacheA, cacheB]);
        if (ba1 == a1)
        {
            AssertArgmaxOrNearTie(refA2, step2[0], tieEps: 0.5f, "Seq A step2");
            Assert.True(MaxAbs(refA2, step2[0]) < 0.5f, $"Seq A step2 maxAbs={MaxAbs(refA2, step2[0])}.");
        }
        if (bb1 == b1)
        {
            AssertArgmaxOrNearTie(refB2, step2[1], tieEps: 0.5f, "Seq B step2");
            Assert.True(MaxAbs(refB2, step2[1]) < 0.5f, $"Seq B step2 maxAbs={MaxAbs(refB2, step2[1])}.");
        }
    }

    // ── Synthetic GGUF fixture writer ──────────────────────────────────────────────────────────

    /// <summary>Writes a tiny all-global F32 Gemma-4 GGUF to a temp file and returns its path. A layer
    /// <c>l</c> omits its <c>attn_v.weight</c> (→ k_eq_v) unless <paramref name="layerHasV"/>[l] is
    /// true; the default (null) omits every attn_v, i.e. all layers k_eq_v. Scalar metadata only (an
    /// all-global model needs no per-layer arrays), so the minimal scalar writer below suffices.
    /// Weights are small deterministic pseudo-random values so the forward pass stays finite.</summary>
    private string WriteSyntheticGguf(int seed, bool[]? layerHasV = null)
    {
        var tensors = new List<(string name, long[] dims, float[] data)>();

        // Global tensors. GGUF dim order is [in, out] (ne0 = innermost = columns).
        tensors.Add(("token_embd.weight", new long[] { EmbDim, Vocab }, RandF32(Vocab * EmbDim, seed + 1)));
        tensors.Add(("output_norm.weight", new long[] { EmbDim }, OnesF32(EmbDim)));
        tensors.Add(("output.weight", new long[] { EmbDim, Vocab }, RandF32(Vocab * EmbDim, seed + 2)));

        for (int l = 0; l < NumLayers; l++)
        {
            int b = seed + 100 * (l + 1);
            tensors.Add(($"blk.{l}.attn_norm.weight", new long[] { EmbDim }, OnesF32(EmbDim)));
            tensors.Add(($"blk.{l}.attn_q.weight", new long[] { EmbDim, QDim }, RandF32(QDim * EmbDim, b + 1)));
            tensors.Add(($"blk.{l}.attn_k.weight", new long[] { EmbDim, KvDim }, RandF32(KvDim * EmbDim, b + 2)));
            // A k_eq_v layer omits attn_v (V reuses the raw K projection); a real-V layer includes it.
            if (layerHasV is not null && layerHasV[l])
                tensors.Add(($"blk.{l}.attn_v.weight", new long[] { EmbDim, KvDim }, RandF32(KvDim * EmbDim, b + 7)));
            tensors.Add(($"blk.{l}.attn_output.weight", new long[] { QDim, EmbDim }, RandF32(EmbDim * QDim, b + 3)));
            tensors.Add(($"blk.{l}.ffn_norm.weight", new long[] { EmbDim }, OnesF32(EmbDim)));
            tensors.Add(($"blk.{l}.ffn_gate.weight", new long[] { EmbDim, FfnDim }, RandF32(FfnDim * EmbDim, b + 4)));
            tensors.Add(($"blk.{l}.ffn_up.weight", new long[] { EmbDim, FfnDim }, RandF32(FfnDim * EmbDim, b + 5)));
            tensors.Add(($"blk.{l}.ffn_down.weight", new long[] { FfnDim, EmbDim }, RandF32(EmbDim * FfnDim, b + 6)));
        }

        var metadata = new (string key, GgufValueType type, object value)[]
        {
            ("general.architecture", GgufValueType.String, "gemma4"),
            ("general.name", GgufValueType.String, "synthetic-gemma4-keqv"),
            ("gemma4.block_count", GgufValueType.Int32, NumLayers),
            ("gemma4.embedding_length", GgufValueType.Int32, EmbDim),
            ("gemma4.feed_forward_length", GgufValueType.Int32, FfnDim),
            ("gemma4.context_length", GgufValueType.Int32, Context),
            ("gemma4.vocab_size", GgufValueType.UInt64, (ulong)Vocab),
            ("gemma4.attention.head_count", GgufValueType.Int32, NumHeads),
            ("gemma4.attention.head_count_kv", GgufValueType.Int32, NumKvHeads),
            ("gemma4.attention.key_length", GgufValueType.Int32, HeadDim),
            ("gemma4.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-6f),
            ("gemma4.rope.freq_base", GgufValueType.Float32, 10_000f),
            ("gemma4.attention.sliding_window", GgufValueType.Int32, 0),
            ("gemma4.embedding_length_per_layer_input", GgufValueType.Int32, 0),
            ("gemma4.final_logit_softcapping", GgufValueType.Float32, 0f),
            // Explicit, though GgufModel.Open also auto-injects this for gemma4 when attn_v count <
            // attn_k count (the real-12B V-deficit heuristic); set it directly so the fixture's intent
            // is self-evident and a mixed model with one attn_v still flags k_eq_v.
            ("_sharpi.attention_k_eq_v", GgufValueType.Bool, true),
        };

        var path = Path.Combine(Path.GetTempPath(), $"sharpi_gemma4_keqv_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);
        using var fs = File.Create(path);
        var w = new ScalarGgufWriter(fs);

        const int alignment = 32;
        w.WriteHeader(version: 3, tensorCount: (ulong)tensors.Count, metadataKvCount: (ulong)metadata.Length);
        foreach (var (key, type, value) in metadata)
            w.WriteMetadataKv(key, type, value);

        // Tensor infos carry offsets into the (aligned) data section; align each tensor to 32 B so the
        // layout matches a real GGUF and every F32 block starts aligned.
        ulong offset = 0;
        foreach (var (name, dims, data) in tensors)
        {
            w.WriteTensorInfo(name, dims, DType.Float32, offset);
            offset += AlignUp((ulong)data.Length * sizeof(float), alignment);
        }
        w.PadToAlignment(alignment); // data section starts aligned

        foreach (var (_, _, data) in tensors)
        {
            long before = w.BytesWritten;
            w.WriteFloats(data);
            w.PadTo(before + (long)AlignUp((ulong)data.Length * sizeof(float), alignment));
        }
        return path;
    }

    private static ulong AlignUp(ulong v, int a) => (v + (ulong)a - 1) / (ulong)a * (ulong)a;

    private static float[] OnesF32(int n)
    {
        var a = new float[n];
        Array.Fill(a, 1f);
        return a;
    }

    // Deterministic small weights in ~[-0.05, 0.05] (xorshift), so the forward stays finite under the
    // Gemma sqrt(embDim) embedding scale and the seed varies the model per test.
    private static float[] RandF32(int n, int seed)
    {
        var a = new float[n];
        uint s = (uint)seed | 1u;
        for (int i = 0; i < n; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            a[i] = ((s & 0xFFFF) / 65535f - 0.5f) * 0.1f;
        }
        return a;
    }

    /// <summary>Minimal scalar-metadata GGUF writer (no array values — an all-global Gemma-4 needs
    /// none). Mirrors the layout the <c>GgufModel</c> reader expects: header, metadata KVs, tensor
    /// infos, 32-byte-aligned data section.</summary>
    private sealed class ScalarGgufWriter(Stream stream)
    {
        public long BytesWritten { get; private set; }

        public void WriteHeader(uint version, ulong tensorCount, ulong metadataKvCount)
        {
            WriteUInt32(0x46554747); // "GGUF"
            WriteUInt32(version);
            WriteUInt64(tensorCount);
            WriteUInt64(metadataKvCount);
        }

        public void WriteTensorInfo(string name, long[] dims, DType dtype, ulong offset)
        {
            WriteString(name);
            WriteUInt32((uint)dims.Length);
            foreach (var d in dims) WriteUInt64((ulong)d);
            WriteUInt32((uint)dtype);
            WriteUInt64(offset);
        }

        public void WriteMetadataKv(string key, GgufValueType type, object value)
        {
            WriteString(key);
            WriteUInt32((uint)type);
            switch (type)
            {
                case GgufValueType.Int32:   WriteUInt32((uint)(int)value); break;
                case GgufValueType.UInt32:  WriteUInt32((uint)value); break;
                case GgufValueType.UInt64:  WriteUInt64((ulong)value); break;
                case GgufValueType.Float32: WriteFloat((float)value); break;
                case GgufValueType.Bool:    WriteByte((bool)value ? (byte)1 : (byte)0); break;
                case GgufValueType.String:  WriteString((string)value); break;
                default: throw new NotSupportedException($"ScalarGgufWriter: unsupported metadata type {type}.");
            }
        }

        public void PadToAlignment(int alignment)
        {
            long rem = BytesWritten % alignment;
            if (rem != 0) PadBytes(alignment - (int)rem);
        }

        public void PadTo(long targetBytes)
        {
            if (targetBytes > BytesWritten) PadBytes((int)(targetBytes - BytesWritten));
        }

        public void WriteFloats(float[] data)
        {
            Span<byte> buf = stackalloc byte[4];
            foreach (var f in data)
            {
                BinaryPrimitives.WriteSingleLittleEndian(buf, f);
                stream.Write(buf);
                BytesWritten += 4;
            }
        }

        private void PadBytes(int n)
        {
            Span<byte> zeros = stackalloc byte[64];
            zeros.Clear();
            while (n > 0)
            {
                int chunk = Math.Min(n, zeros.Length);
                stream.Write(zeros[..chunk]);
                BytesWritten += chunk;
                n -= chunk;
            }
        }

        private void WriteString(string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            WriteUInt64((ulong)bytes.Length);
            stream.Write(bytes);
            BytesWritten += bytes.Length;
        }

        private void WriteByte(byte v) { stream.WriteByte(v); BytesWritten += 1; }

        private void WriteUInt32(uint v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(buf, v);
            stream.Write(buf);
            BytesWritten += 4;
        }

        private void WriteUInt64(ulong v)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(buf, v);
            stream.Write(buf);
            BytesWritten += 8;
        }

        private void WriteFloat(float v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteSingleLittleEndian(buf, v);
            stream.Write(buf);
            BytesWritten += 4;
        }
    }
}
