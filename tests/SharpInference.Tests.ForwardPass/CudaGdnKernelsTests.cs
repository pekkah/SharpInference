using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Parity tests for the NVRTC kernels backing the qwen35moe GPU GDN block
/// (<see cref="CudaBackend.GdnConv1dDecode"/>,
/// <see cref="CudaBackend.GdnL2NormPerHead"/>,
/// <see cref="CudaBackend.GdnTileHeads"/>,
/// <see cref="CudaBackend.GdnRecurrenceDecode"/>,
/// <see cref="CudaBackend.SiLUInPlace"/>). Each kernel is exercised at the
/// shape the real model uses (where applicable) and cross-checked against the
/// CPU reference in <see cref="GdnKernels"/>. Skipped silently on hosts
/// without CUDA.
/// </summary>
public sealed unsafe class CudaGdnKernelsTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static float[] RandomArray(Random rng, int count, float lo, float hi)
    {
        var a = new float[count];
        for (int i = 0; i < count; i++) a[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return a;
    }

    [Fact]
    public void SiLUInPlace_MatchesScalarReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int N = 513;
        var rng = new Random(123);
        var x = RandomArray(rng, N, -3f, 3f);

        var expected = new float[N];
        for (int i = 0; i < N; i++)
            expected[i] = x[i] / (1f + MathF.Exp(-x[i]));

        var gpuX = gpu.Upload(x, TensorShape.D1(N));
        gpu.SiLUInPlace(gpuX);
        gpu.Synchronize();
        var result = new float[N];
        gpu.Download(gpuX, result);
        gpu.Free(gpuX);

        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 1e-5f,
                $"SiLUInPlace mismatch at [{i}]: gpu={result[i]} cpu={expected[i]}");
    }

    [Fact]
    public void GdnConv1dDecode_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Use the model's real shape.
        const int Channels = 8192;
        const int Kernel = 4;
        int stateLen = (Kernel - 1) * Channels;
        int weightLen = Kernel * Channels;

        var rng = new Random(777);
        var x = RandomArray(rng, Channels, -1f, 1f);
        var state = RandomArray(rng, stateLen, -0.5f, 0.5f);
        var weight = RandomArray(rng, weightLen, -0.3f, 0.3f);

        // CPU reference: copy state so we can compare against the GPU-mutated state.
        var stateCpu = (float[])state.Clone();
        var outputCpu = new float[Channels];
        GdnKernels.CausalDepthwiseConv1dDecode(x, stateCpu, weight, outputCpu, Channels, Kernel);

        var gpuX = gpu.Upload(x, TensorShape.D1(Channels));
        var gpuState = gpu.Upload(state, TensorShape.D1(stateLen));
        var gpuWeight = gpu.Upload(weight, TensorShape.D1(weightLen));
        var gpuOutput = gpu.Allocate(TensorShape.D1(Channels));

        gpu.GdnConv1dDecode(gpuX, gpuState, gpuWeight, gpuOutput, Channels, Kernel);
        gpu.Synchronize();

        var outputGpu = new float[Channels];
        var stateGpu = new float[stateLen];
        gpu.Download(gpuOutput, outputGpu);
        gpu.Download(gpuState, stateGpu);

        for (int i = 0; i < Channels; i++)
            Assert.True(MathF.Abs(outputGpu[i] - outputCpu[i]) < 1e-4f,
                $"conv1d output mismatch at [{i}]: gpu={outputGpu[i]} cpu={outputCpu[i]}");
        for (int i = 0; i < stateLen; i++)
            Assert.True(MathF.Abs(stateGpu[i] - stateCpu[i]) < 1e-5f,
                $"conv1d state mismatch at [{i}]: gpu={stateGpu[i]} cpu={stateCpu[i]}");

        gpu.Free(gpuX); gpu.Free(gpuState); gpu.Free(gpuWeight); gpu.Free(gpuOutput);
    }

    [Fact]
    public void GdnL2NormPerHead_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Model's real shape: 16 K-heads × 128 head_dim = 2048 floats.
        const int NumHeads = 16;
        const int HeadDim = 128;
        const float Eps = 1e-6f;
        int N = NumHeads * HeadDim;

        var rng = new Random(31337);
        var x = RandomArray(rng, N, -2f, 2f);

        var expected = (float[])x.Clone();
        GdnKernels.L2NormPerHead(expected, NumHeads, HeadDim, Eps);

        var gpuX = gpu.Upload(x, TensorShape.D1(N));
        gpu.GdnL2NormPerHead(gpuX, elementOffset: 0, NumHeads, HeadDim, Eps);
        gpu.Synchronize();
        var result = new float[N];
        gpu.Download(gpuX, result);
        gpu.Free(gpuX);

        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 1e-5f,
                $"L2 norm mismatch at [{i}]: gpu={result[i]} cpu={expected[i]}");
    }

    [Fact]
    public void GdnL2NormPerHead_RespectsElementOffset()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Lay out [Q‖K‖V] in a single buffer; verify normalization only touches the K slice.
        const int Slice = 2048;   // K-heads × head_dim
        const int NumHeads = 16;
        const int HeadDim = 128;
        int Total = 3 * Slice;

        var rng = new Random(42);
        var data = RandomArray(rng, Total, -1f, 1f);

        // CPU: norm only the middle slice (offset = Slice).
        var expected = (float[])data.Clone();
        GdnKernels.L2NormPerHead(expected.AsSpan(Slice, Slice), NumHeads, HeadDim, 1e-6f);

        var gpuD = gpu.Upload(data, TensorShape.D1(Total));
        gpu.GdnL2NormPerHead(gpuD, elementOffset: Slice, NumHeads, HeadDim, 1e-6f);
        gpu.Synchronize();
        var result = new float[Total];
        gpu.Download(gpuD, result);
        gpu.Free(gpuD);

        // Untouched halves must be byte-identical to input.
        for (int i = 0; i < Slice; i++) Assert.Equal(data[i], result[i]);
        for (int i = 2 * Slice; i < Total; i++) Assert.Equal(data[i], result[i]);
        // Middle slice must match the CPU norm.
        for (int i = Slice; i < 2 * Slice; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 1e-5f,
                $"L2 norm offset mismatch at [{i}]: gpu={result[i]} cpu={expected[i]}");
    }

    [Fact]
    public void GdnTileHeads_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int SrcHeads = 16;
        const int Repeat = 2;
        const int HeadDim = 128;
        int srcLen = SrcHeads * HeadDim;
        int dstLen = SrcHeads * Repeat * HeadDim;

        var rng = new Random(54321);
        var src = RandomArray(rng, srcLen, -1f, 1f);

        var expected = new float[dstLen];
        GdnKernels.TileHeads(src, expected, SrcHeads, Repeat, HeadDim);

        var gpuSrc = gpu.Upload(src, TensorShape.D1(srcLen));
        var gpuDst = gpu.Allocate(TensorShape.D1(dstLen));
        gpu.GdnTileHeads(gpuSrc, 0, gpuDst, 0, SrcHeads, Repeat, HeadDim);
        gpu.Synchronize();
        var result = new float[dstLen];
        gpu.Download(gpuDst, result);
        gpu.Free(gpuSrc); gpu.Free(gpuDst);

        Assert.Equal(expected, result);
    }

    [Fact]
    public void GdnRecurrenceDecode_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Use the model's real shape: 32 v-heads × 128 head_dim.
        const int Hv = 32;
        const int D = 128;
        int qkv = Hv * D;
        int stateLen = Hv * D * D;

        var rng = new Random(7919);
        var q = RandomArray(rng, qkv, -0.5f, 0.5f);
        var k = RandomArray(rng, qkv, -0.5f, 0.5f);
        var v = RandomArray(rng, qkv, -0.5f, 0.5f);
        var alphaIn = RandomArray(rng, Hv, -0.3f, 0.3f);
        var beta = RandomArray(rng, Hv, -0.3f, 0.3f);
        var ssmA = RandomArray(rng, Hv, -0.5f, -0.01f);     // ssmA is negative
        var dtBias = RandomArray(rng, Hv, -0.1f, 0.1f);
        var normW = RandomArray(rng, D, 0.5f, 1.5f);
        var z = RandomArray(rng, qkv, -1f, 1f);

        // Initialize state to a small random matrix (not all zero — exercises decay).
        var state = RandomArray(rng, stateLen, -0.1f, 0.1f);

        // CPU reference.
        var stateCpu = (float[])state.Clone();
        var outputCpu = new float[qkv];
        GdnKernels.GdnRecurrenceDecode(q, k, v, alphaIn, beta, ssmA, dtBias, normW, z,
            stateCpu, outputCpu, Hv, D);

        // GPU.
        var gpuState = gpu.Upload(state, TensorShape.D1(stateLen));
        var gpuQ = gpu.Upload(q, TensorShape.D1(qkv));
        var gpuK = gpu.Upload(k, TensorShape.D1(qkv));
        var gpuV = gpu.Upload(v, TensorShape.D1(qkv));
        var gpuAlpha = gpu.Upload(alphaIn, TensorShape.D1(Hv));
        var gpuBeta = gpu.Upload(beta, TensorShape.D1(Hv));
        var gpuSsmA = gpu.Upload(ssmA, TensorShape.D1(Hv));
        var gpuDtBias = gpu.Upload(dtBias, TensorShape.D1(Hv));
        var gpuNormW = gpu.Upload(normW, TensorShape.D1(D));
        var gpuZ = gpu.Upload(z, TensorShape.D1(qkv));
        var gpuOut = gpu.Allocate(TensorShape.D1(qkv));

        gpu.GdnRecurrenceDecode(gpuState, gpuQ, gpuK, gpuV,
            gpuAlpha, gpuBeta, gpuSsmA, gpuDtBias, gpuNormW, gpuZ, gpuOut,
            Hv, D);
        gpu.Synchronize();

        var outputGpu = new float[qkv];
        var stateGpu = new float[stateLen];
        gpu.Download(gpuOut, outputGpu);
        gpu.Download(gpuState, stateGpu);

        gpu.Free(gpuState); gpu.Free(gpuQ); gpu.Free(gpuK); gpu.Free(gpuV);
        gpu.Free(gpuAlpha); gpu.Free(gpuBeta); gpu.Free(gpuSsmA); gpu.Free(gpuDtBias);
        gpu.Free(gpuNormW); gpu.Free(gpuZ); gpu.Free(gpuOut);

        // The recurrence accumulates over 128 inner-product terms × 32 heads. With single-precision
        // arithmetic, max observed error is around 1e-3 on the post-norm + silu-gated output.
        int badCount = 0;
        float maxErr = 0f;
        for (int i = 0; i < qkv; i++)
        {
            float err = MathF.Abs(outputGpu[i] - outputCpu[i]);
            maxErr = MathF.Max(maxErr, err);
            if (err > 1e-3f) badCount++;
        }
        Assert.True(badCount == 0,
            $"GdnRecurrenceDecode: {badCount} / {qkv} outputs exceed 1e-3 (max err = {maxErr}).");

        // State update should also match (rank-1 update on 32 × 128 × 128 = 524,288 floats).
        int badStateCount = 0;
        float maxStateErr = 0f;
        for (int i = 0; i < stateLen; i++)
        {
            float err = MathF.Abs(stateGpu[i] - stateCpu[i]);
            maxStateErr = MathF.Max(maxStateErr, err);
            if (err > 1e-4f) badStateCount++;
        }
        Assert.True(badStateCount == 0,
            $"GdnRecurrenceDecode state: {badStateCount} / {stateLen} entries exceed 1e-4 (max err = {maxStateErr}).");
    }

    [Fact]
    public void GdnChunkedPrefill_ModelStridesMultiChunk_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Reproduce the EXACT buffer layout CudaHybridGdnForwardPass.GdnBlockBatched
        // feeds the kernel — the one thing the contiguous-stride test above doesn't
        // cover: q/k from tiled [nTok, valueDim] head buffers, v read in place from the
        // silu'd conv output [nTok, convCh] at vHeadOff = 2*keyDim (stride convCh), z
        // from [nTok, valueDim]. n_tok = 136 spans 3 GDN_CHUNK(64) blocks → exercises
        // the multi-chunk state carry under non-trivial strides. Real qwen35moe dims.
        const int Hv = 32, D = 128, nTok = 136;
        const int numKHeads = 16;
        int valueDim = Hv * D;                 // 4096
        int keyDim = numKHeads * D;            // 2048
        int convCh = 2 * keyDim + valueDim;    // 8192 (Q‖K‖V joint conv stream)
        int vHeadOff = 2 * keyDim;             // 4096
        int qkv = Hv * D, stateLen = Hv * D * D;

        var rng = new Random(0x57121D);
        // Logical per-token q/k/v/z (contiguous [nTok, Hv*D]) for the CPU reference.
        var q = RandomArray(rng, nTok * qkv, -0.5f, 0.5f);
        var k = RandomArray(rng, nTok * qkv, -0.5f, 0.5f);
        var v = RandomArray(rng, nTok * qkv, -0.5f, 0.5f);
        var alpha = RandomArray(rng, nTok * Hv, -0.3f, 0.3f);
        var beta = RandomArray(rng, nTok * Hv, -0.3f, 0.3f);
        var ssmA = RandomArray(rng, Hv, -0.5f, -0.01f);
        var dtBias = RandomArray(rng, Hv, -0.1f, 0.1f);
        var normW = RandomArray(rng, D, 0.5f, 1.5f);
        var z = RandomArray(rng, nTok * qkv, -1f, 1f);
        var state0 = RandomArray(rng, stateLen, -0.1f, 0.1f);

        // CPU double-precision reference (the gold standard).
        var stateCpu = (float[])state0.Clone();
        var outCpu = new float[nTok * qkv];
        GdnKernels.GdnRecurrencePrefill(nTok, q, k, v, alpha, beta, ssmA, dtBias, normW, z,
            stateCpu, outCpu, Hv, D);

        // Strided V buffer [nTok, convCh]: place each token's v at [t*convCh + vHeadOff].
        // The conv-stream Q/K regions (before vHeadOff) are filled with noise the kernel
        // must NOT read for V — a stride/offset bug would pull this garbage into the scan.
        var vStrided = RandomArray(rng, nTok * convCh, 5f, 6f);   // distinctive out-of-range filler
        for (int t = 0; t < nTok; t++)
            Array.Copy(v, t * qkv, vStrided, t * convCh + vHeadOff, qkv);

        var gSt = gpu.Upload(state0, TensorShape.D1(stateLen));
        var gQ = gpu.Upload(q, TensorShape.D1(nTok * qkv));        // qStride = valueDim
        var gK = gpu.Upload(k, TensorShape.D1(nTok * qkv));        // kStride = valueDim
        var gV = gpu.Upload(vStrided, TensorShape.D1(nTok * convCh));
        var gA = gpu.Upload(alpha, TensorShape.D1(nTok * Hv));
        var gB = gpu.Upload(beta, TensorShape.D1(nTok * Hv));
        var gSA = gpu.Upload(ssmA, TensorShape.D1(Hv));
        var gDB = gpu.Upload(dtBias, TensorShape.D1(Hv));
        var gNW = gpu.Upload(normW, TensorShape.D1(D));
        var gZ = gpu.Upload(z, TensorShape.D1(nTok * qkv));        // zStride = valueDim
        var gO = gpu.Allocate(TensorShape.D1(nTok * qkv));

        gpu.GdnChunkedPrefill(gSt, gQ, gK, gV, gA, gB, gSA, gDB, gNW, gZ, gO,
            Hv, D, 1e-6f,
            qStride: valueDim, kStride: valueDim, vStride: convCh, vHeadOff: vHeadOff,
            zStride: valueDim, oStride: valueDim, nTok: nTok);
        gpu.Synchronize();

        var outGpu = new float[nTok * qkv];
        var stateGpu = new float[stateLen];
        gpu.Download(gO, outGpu);
        gpu.Download(gSt, stateGpu);
        gpu.Free(gSt); gpu.Free(gQ); gpu.Free(gK); gpu.Free(gV); gpu.Free(gA); gpu.Free(gB);
        gpu.Free(gSA); gpu.Free(gDB); gpu.Free(gNW); gpu.Free(gZ); gpu.Free(gO);

        int badOut = 0; float maxOut = 0f;
        for (int i = 0; i < outGpu.Length; i++)
        {
            float err = MathF.Abs(outGpu[i] - outCpu[i]);
            maxOut = MathF.Max(maxOut, err);
            if (err > 3e-3f + 3e-3f * MathF.Abs(outCpu[i])) badOut++;
        }
        Assert.True(badOut == 0,
            $"GdnChunkedPrefill (model strides, 3 chunks) output: {badOut} entries exceed tol (max err {maxOut}).");

        int badState = 0; float maxState = 0f;
        for (int i = 0; i < stateLen; i++)
        {
            float err = MathF.Abs(stateGpu[i] - stateCpu[i]);
            maxState = MathF.Max(maxState, err);
            if (err > 3e-3f + 3e-3f * MathF.Abs(stateCpu[i])) badState++;
        }
        Assert.True(badState == 0,
            $"GdnChunkedPrefill (model strides, 3 chunks) state: {badState} entries exceed tol (max err {maxState}).");
    }

    [Fact]
    public void GdnChunkedPrefill_MatchesCpuSequentialReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Real GDN shape; n_tok spans more than one GDN_CHUNK (64) to exercise the
        // multi-chunk state carry.
        const int Hv = 32;
        const int D = 128;
        const int nTok = 70;
        int qkv = Hv * D;                 // per-token q/k/v/z width
        int stateLen = Hv * D * D;

        var rng = new Random(20260614);
        var q = RandomArray(rng, nTok * qkv, -0.5f, 0.5f);
        var k = RandomArray(rng, nTok * qkv, -0.5f, 0.5f);
        var v = RandomArray(rng, nTok * qkv, -0.5f, 0.5f);
        var alpha = RandomArray(rng, nTok * Hv, -0.3f, 0.3f);
        var beta = RandomArray(rng, nTok * Hv, -0.3f, 0.3f);
        var ssmA = RandomArray(rng, Hv, -0.5f, -0.01f);
        var dtBias = RandomArray(rng, Hv, -0.1f, 0.1f);
        var normW = RandomArray(rng, D, 0.5f, 1.5f);
        var z = RandomArray(rng, nTok * qkv, -1f, 1f);
        var state = RandomArray(rng, stateLen, -0.1f, 0.1f);

        // CPU reference: the sequential per-token scan (the byte-parity oracle).
        var stateCpu = (float[])state.Clone();
        var outputCpu = new float[nTok * qkv];
        GdnKernels.GdnRecurrencePrefill(nTok, q, k, v, alpha, beta, ssmA, dtBias, normW, z,
            stateCpu, outputCpu, Hv, D);

        // GPU chunked prefill. Contiguous [nTok, Hv*D] layout → strides = Hv*D, vHeadOff = 0.
        var gpuState = gpu.Upload(state, TensorShape.D1(stateLen));
        var gpuQ = gpu.Upload(q, TensorShape.D1(nTok * qkv));
        var gpuK = gpu.Upload(k, TensorShape.D1(nTok * qkv));
        var gpuV = gpu.Upload(v, TensorShape.D1(nTok * qkv));
        var gpuAlpha = gpu.Upload(alpha, TensorShape.D1(nTok * Hv));
        var gpuBeta = gpu.Upload(beta, TensorShape.D1(nTok * Hv));
        var gpuSsmA = gpu.Upload(ssmA, TensorShape.D1(Hv));
        var gpuDtBias = gpu.Upload(dtBias, TensorShape.D1(Hv));
        var gpuNormW = gpu.Upload(normW, TensorShape.D1(D));
        var gpuZ = gpu.Upload(z, TensorShape.D1(nTok * qkv));
        var gpuOut = gpu.Allocate(TensorShape.D1(nTok * qkv));

        gpu.GdnChunkedPrefill(gpuState, gpuQ, gpuK, gpuV, gpuAlpha, gpuBeta, gpuSsmA, gpuDtBias,
            gpuNormW, gpuZ, gpuOut, Hv, D, 1e-6f,
            qStride: qkv, kStride: qkv, vStride: qkv, vHeadOff: 0, zStride: qkv, oStride: qkv, nTok: nTok);
        gpu.Synchronize();

        var outputGpu = new float[nTok * qkv];
        var stateGpu = new float[stateLen];
        gpu.Download(gpuOut, outputGpu);
        gpu.Download(gpuState, stateGpu);

        gpu.Free(gpuState); gpu.Free(gpuQ); gpu.Free(gpuK); gpu.Free(gpuV);
        gpu.Free(gpuAlpha); gpu.Free(gpuBeta); gpu.Free(gpuSsmA); gpu.Free(gpuDtBias);
        gpu.Free(gpuNormW); gpu.Free(gpuZ); gpu.Free(gpuOut);

        // Chunked vs sequential: FP reduction order differs, so compare with a relative
        // tolerance (the chunked form resolves the same recurrence over 70 tokens).
        int badOut = 0; float maxOut = 0f;
        for (int i = 0; i < outputGpu.Length; i++)
        {
            float err = MathF.Abs(outputGpu[i] - outputCpu[i]);
            maxOut = MathF.Max(maxOut, err);
            if (err > 3e-3f + 3e-3f * MathF.Abs(outputCpu[i])) badOut++;
        }
        Assert.True(badOut == 0, $"GdnChunkedPrefill output: {badOut} entries exceed tol (max err {maxOut}).");

        int badState = 0; float maxState = 0f;
        for (int i = 0; i < stateLen; i++)
        {
            float err = MathF.Abs(stateGpu[i] - stateCpu[i]);
            maxState = MathF.Max(maxState, err);
            if (err > 3e-3f + 3e-3f * MathF.Abs(stateCpu[i])) badState++;
        }
        Assert.True(badState == 0, $"GdnChunkedPrefill state: {badState} entries exceed tol (max err {maxState}).");
    }
}
