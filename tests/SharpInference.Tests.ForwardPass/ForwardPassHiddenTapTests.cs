using System.Buffers.Binary;
using System.Text;
using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Hidden-state taps on the CPU <see cref="Engine.ForwardPass"/> (DSpark / EAGLE-3-style
/// draft conditioning, PR #413 spec). DSpark's draft head conditions on the TARGET model's
/// intermediate hidden states from a fixed set of layers (config <c>target_layer_ids</c>;
/// layer i's tap is that layer's output — HF's <c>hidden_states[i+1]</c> convention),
/// captured for every processed position during (a) prefill and (b) the k+1-token verify
/// forward, indexed by absolute position and overwritten when a later pass re-processes
/// the same position.
///
/// <para>These tests exercise the three capture sites — sequential <c>Forward</c> (RunTrunk),
/// batched <c>Prefill</c> (PrefillCore), and <c>BatchVerify</c> — against each other on a tiny
/// synthetic all-F32 LLAMA-architecture GGUF (emb=32, 4 heads / 2 KV heads, head_dim=8,
/// ffn=64, vocab=64, 3 layers), plus the <c>EnableHiddenTaps</c> validation contract, the
/// unpopulated-position contract of <c>HiddenTapsAt</c>, and overwrite-after-rewind semantics
/// (<c>TruncateTo</c> does not clear taps; re-processing a position overwrites its slot).
/// Random weights are fine: both passes read the same weights, so the comparison isolates the
/// tap wiring. Batched GEMM rounds slightly differently from the sequential matvec path, hence
/// the 1e-4 element tolerance.</para>
/// </summary>
public sealed class ForwardPassHiddenTapTests : IDisposable
{
    // Tiny but valid llama geometry. embDim = numHeads*headDim keeps the Q projection square.
    private const int EmbDim = 32, NumHeads = 4, HeadDim = 8, NumKvHeads = 2, FfnDim = 64;
    private const int Vocab = 64, NumLayers = 3, Context = 128;
    private const int QDim = NumHeads * HeadDim;     // 32
    private const int KvDim = NumKvHeads * HeadDim;  // 16
    private const float Tol = 1e-4f;

    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (var f in _tempFiles)
            try { File.Delete(f); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { /* best-effort temp cleanup */ }
    }

    // ── Tests ──────────────────────────────────────────────────────────────────────────────

    /// <summary>Taps captured by the batched <c>Prefill</c> path must match the sequential
    /// <c>Forward</c> path per position: same shape (HiddenTapDim = 2×EmbDim for two tapped
    /// layers), element-wise within tolerance, same argmax. Two independent pass instances
    /// over the same GGUF so KV caches don't interfere.</summary>
    [Fact]
    public void Taps_SequentialForward_vs_Prefill_Match()
    {
        var path = WriteSyntheticLlamaGguf(seed: 42);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var backend = new CpuBackend();
        using var fwdA = new Engine.ForwardPass(model, backend, hp);
        using var fwdB = new Engine.ForwardPass(model, backend, hp);

        int[] tapLayers = [0, 2];
        fwdA.EnableHiddenTaps(tapLayers);
        fwdB.EnableHiddenTaps(tapLayers);
        Assert.Equal(2 * EmbDim, fwdA.HiddenTapDim);
        Assert.Equal(2 * EmbDim, fwdB.HiddenTapDim);

        int[] tokens = [3, 5, 7, 9];

        // Pass A: sequential single-token forwards (RunTrunk capture site).
        float[] seqLogits = [];
        for (int i = 0; i < tokens.Length; i++)
            seqLogits = fwdA.Forward(tokens[i], i).ToArray();

        // Pass B: one batched prefill (PrefillCore capture site).
        float[] preLogits = fwdB.Prefill(tokens).ToArray();

        // Prefill returns the LAST token's logits — same computation as the final Forward.
        Assert.Equal(Argmax(seqLogits), Argmax(preLogits));

        for (int p = 0; p < tokens.Length; p++)
        {
            float[] a = fwdA.HiddenTapsAt(p).ToArray();
            float[] b = fwdB.HiddenTapsAt(p).ToArray();
            Assert.Equal(2 * EmbDim, a.Length);
            Assert.Equal(2 * EmbDim, b.Length);
            AssertClose(a, b, Tol, $"tap row at position {p}");
            Assert.Equal(Argmax(a), Argmax(b));
        }
    }

    /// <summary>Taps captured by <c>BatchVerify</c> (the speculative-decoding verify forward —
    /// DSpark consumes exactly these for the accepted positions) must exist for the verified
    /// positions and match a reference pass running the same 5 tokens sequentially.</summary>
    [Fact]
    public void Taps_BatchVerify_Match()
    {
        var path = WriteSyntheticLlamaGguf(seed: 77);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var backend = new CpuBackend();
        using var fwdA = new Engine.ForwardPass(model, backend, hp);
        using var fwdB = new Engine.ForwardPass(model, backend, hp);

        int[] tapLayers = [0, 2];
        fwdA.EnableHiddenTaps(tapLayers);
        fwdB.EnableHiddenTaps(tapLayers);

        // Pass A: prefill the context, then a 2-token batched verify at startPos 3
        // (N=2, non-MoE → the batched BatchVerify core, not the sequential fallback).
        int[] prompt = [3, 5, 7];
        _ = fwdA.Prefill(prompt);
        float[][] verify = fwdA.BatchVerify([9, 11], startPos: 3);
        Assert.Equal(2, verify.Length);
        Assert.Equal(Vocab, verify[0].Length);

        // Pass B: same 5 tokens through sequential Forward.
        int[] all = [3, 5, 7, 9, 11];
        for (int i = 0; i < all.Length; i++)
            _ = fwdB.Forward(all[i], i);

        for (int p = 3; p <= 4; p++)
        {
            float[] a = fwdA.HiddenTapsAt(p).ToArray();
            float[] b = fwdB.HiddenTapsAt(p).ToArray();
            Assert.Equal(2 * EmbDim, a.Length);
            Assert.Equal(2 * EmbDim, b.Length);
            AssertClose(a, b, Tol, $"verify tap row at position {p}");
        }
    }

    /// <summary>HiddenTapsAt returns an empty span for positions that have not been captured:
    /// every position before any forward, and positions at/after the high-water mark after.</summary>
    [Fact]
    public void TapsAt_Unpopulated_ReturnsEmpty()
    {
        var path = WriteSyntheticLlamaGguf(seed: 11);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        fwd.EnableHiddenTaps([1]);
        Assert.True(fwd.HiddenTapsAt(0).IsEmpty);

        int[] tokens = [3, 5, 7];
        for (int i = 0; i < tokens.Length; i++)
            _ = fwd.Forward(tokens[i], i);

        for (int p = 0; p < tokens.Length; p++)
            Assert.Equal(fwd.HiddenTapDim, fwd.HiddenTapsAt(p).Length); // 0..2 populated

        Assert.True(fwd.HiddenTapsAt(3).IsEmpty);   // beyond the high-water mark
        Assert.True(fwd.HiddenTapsAt(-1).IsEmpty);  // negative position
    }

    /// <summary>EnableHiddenTaps rejects non-strictly-increasing ids, out-of-range ids, and an
    /// empty id list — and a valid call still succeeds after the failed attempts.</summary>
    [Fact]
    public void EnableHiddenTaps_Validation()
    {
        var path = WriteSyntheticLlamaGguf(seed: 13);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        Assert.Throws<ArgumentOutOfRangeException>(() => fwd.EnableHiddenTaps([1, 1]));           // repeated id
        Assert.Throws<ArgumentOutOfRangeException>(() => fwd.EnableHiddenTaps([2, 0]));           // decreasing
        Assert.Throws<ArgumentOutOfRangeException>(() => fwd.EnableHiddenTaps([0, NumLayers]));   // id >= NumLayers
        Assert.Throws<ArgumentOutOfRangeException>(() => fwd.EnableHiddenTaps([-1]));             // negative id
        Assert.Throws<ArgumentException>(() => fwd.EnableHiddenTaps([]));                         // empty

        fwd.EnableHiddenTaps([0, 1, 2]); // full-range strictly-increasing ids are accepted
        Assert.Equal(NumLayers * EmbDim, fwd.HiddenTapDim);
    }

    /// <summary>TruncateTo does not clear taps; re-processing a position overwrites its slot.
    /// After rewinding to length 1, forwarding token 9 at position 1 must replace token 5's tap
    /// row while leaving position 0's row (not re-processed) bit-identical and position 2's row
    /// (stale but still within the high-water mark) readable.</summary>
    [Fact]
    public void Taps_OverwrittenAfterRewind()
    {
        var path = WriteSyntheticLlamaGguf(seed: 21);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        fwd.EnableHiddenTaps([0, 2]);

        int[] tokens = [3, 5, 7];
        for (int i = 0; i < tokens.Length; i++)
            _ = fwd.Forward(tokens[i], i);

        float[] oldTap0 = fwd.HiddenTapsAt(0).ToArray();
        float[] oldTap1 = fwd.HiddenTapsAt(1).ToArray(); // token 5's computation
        Assert.Equal(2 * EmbDim, oldTap1.Length);

        fwd.TruncateTo(1);
        _ = fwd.Forward(9, 1); // re-process position 1 with a DIFFERENT token

        float[] newTap1 = fwd.HiddenTapsAt(1).ToArray(); // token 9's computation
        Assert.Equal(2 * EmbDim, newTap1.Length);
        float maxAbs = MaxAbsDiff(oldTap1, newTap1);
        Assert.True(maxAbs > Tol,
            $"tap row at position 1 was not overwritten by the rewound forward (maxAbsDiff={maxAbs}).");

        // Position 0 was not re-processed — its row must be untouched (bit-identical).
        Assert.True(fwd.HiddenTapsAt(0).SequenceEqual(oldTap0),
            "tap row at position 0 changed although the position was never re-processed.");

        // Position 2's stale row stays readable (taps survive TruncateTo until overwritten).
        Assert.Equal(2 * EmbDim, fwd.HiddenTapsAt(2).Length);
    }

    /// <summary>A plain CPU pass (no SnapKV, no TurboQuant) reports tap support, with
    /// HiddenTapDim 0 and empty tap rows before <c>EnableHiddenTaps</c> is called.</summary>
    [Fact]
    public void SupportsHiddenTaps_TrueOnPlainCpuPass()
    {
        var path = WriteSyntheticLlamaGguf(seed: 31);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        Assert.True(fwd.SupportsHiddenTaps);
        Assert.Equal(0, fwd.HiddenTapDim);           // no taps enabled yet
        Assert.True(fwd.HiddenTapsAt(0).IsEmpty);    // and nothing captured
    }

    // ── Helpers ────────────────────────────────────────────────────────────────────────────

    private static int Argmax(ReadOnlySpan<float> values)
    {
        int best = 0; float bestVal = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > bestVal) { bestVal = values[i]; best = i; }
        return best;
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private static void AssertClose(float[] a, float[] b, float tol, string context)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            // NaN diffs fail here too: NaN <= tol is false.
            Assert.True(d <= tol, $"{context}: element {i} differs by {d} (a={a[i]}, b={b[i]}, tol={tol}).");
        }
    }

    // ── Synthetic GGUF fixture writer ──────────────────────────────────────────────────────

    /// <summary>Writes a tiny all-F32 LLAMA-architecture GGUF to a temp file and returns its
    /// path. Weights are small deterministic pseudo-random values (xorshift) so the forward
    /// pass stays finite; norm weights are 1.0. Scalar metadata only.</summary>
    private string WriteSyntheticLlamaGguf(int seed)
    {
        var tensors = new List<(string name, long[] dims, float[] data)>();

        // GGUF dim order is [in, out] (ne0 = innermost = columns).
        tensors.Add(("token_embd.weight", new long[] { EmbDim, Vocab }, RandF32(Vocab * EmbDim, seed + 1)));
        tensors.Add(("output_norm.weight", new long[] { EmbDim }, OnesF32(EmbDim)));
        tensors.Add(("output.weight", new long[] { EmbDim, Vocab }, RandF32(Vocab * EmbDim, seed + 2)));

        for (int l = 0; l < NumLayers; l++)
        {
            int b = seed + 100 * (l + 1);
            tensors.Add(($"blk.{l}.attn_norm.weight", new long[] { EmbDim }, OnesF32(EmbDim)));
            tensors.Add(($"blk.{l}.attn_q.weight", new long[] { EmbDim, QDim }, RandF32(QDim * EmbDim, b + 1)));
            tensors.Add(($"blk.{l}.attn_k.weight", new long[] { EmbDim, KvDim }, RandF32(KvDim * EmbDim, b + 2)));
            tensors.Add(($"blk.{l}.attn_v.weight", new long[] { EmbDim, KvDim }, RandF32(KvDim * EmbDim, b + 3)));
            tensors.Add(($"blk.{l}.attn_output.weight", new long[] { QDim, EmbDim }, RandF32(EmbDim * QDim, b + 4)));
            tensors.Add(($"blk.{l}.ffn_norm.weight", new long[] { EmbDim }, OnesF32(EmbDim)));
            tensors.Add(($"blk.{l}.ffn_gate.weight", new long[] { EmbDim, FfnDim }, RandF32(FfnDim * EmbDim, b + 5)));
            tensors.Add(($"blk.{l}.ffn_up.weight", new long[] { EmbDim, FfnDim }, RandF32(FfnDim * EmbDim, b + 6)));
            tensors.Add(($"blk.{l}.ffn_down.weight", new long[] { FfnDim, EmbDim }, RandF32(EmbDim * FfnDim, b + 7)));
        }

        var metadata = new (string key, GgufValueType type, object value)[]
        {
            ("general.architecture", GgufValueType.String, "llama"),
            ("general.name", GgufValueType.String, "synthetic-llama-hidden-taps"),
            ("llama.block_count", GgufValueType.Int32, NumLayers),
            ("llama.context_length", GgufValueType.Int32, Context),
            ("llama.embedding_length", GgufValueType.Int32, EmbDim),
            ("llama.feed_forward_length", GgufValueType.Int32, FfnDim),
            ("llama.vocab_size", GgufValueType.UInt64, (ulong)Vocab),
            ("llama.attention.head_count", GgufValueType.Int32, NumHeads),
            ("llama.attention.head_count_kv", GgufValueType.Int32, NumKvHeads),
            ("llama.attention.key_length", GgufValueType.Int32, HeadDim),
            ("llama.attention.layer_norm_rms_epsilon", GgufValueType.Float32, 1e-5f),
            ("llama.rope.freq_base", GgufValueType.Float32, 10_000f),
        };

        var path = Path.Combine(Path.GetTempPath(), $"sharpi_llama_hiddentaps_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);
        using var fs = File.Create(path);
        var w = new ScalarGgufWriter(fs);

        const int alignment = 32;
        w.WriteHeader(version: 3, tensorCount: (ulong)tensors.Count, metadataKvCount: (ulong)metadata.Length);
        foreach (var (key, type, value) in metadata)
            w.WriteMetadataKv(key, type, value);

        // Tensor infos carry offsets into the (aligned) data section; align each tensor to 32 B
        // so the layout matches a real GGUF and every F32 block starts aligned.
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

    // Deterministic small weights in ~[-0.05, 0.05] (xorshift) so the forward stays finite
    // and the seed varies the model per test.
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

    /// <summary>Minimal scalar-metadata GGUF writer (no array values — a plain llama fixture
    /// needs none). Mirrors the layout the <c>GgufModel</c> reader expects: header, metadata
    /// KVs, tensor infos, 32-byte-aligned data section.</summary>
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
