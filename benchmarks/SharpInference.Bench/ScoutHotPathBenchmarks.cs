using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Bench;

/// <summary>
/// Micro-benchmarks for individual hot-path operations at Llama-4-Scout dimensions.
/// Uses synthetic Q4_K data so no model file is needed.
/// </summary>
[MemoryDiagnoser]
[WarmupCount(3)]
[IterationCount(5)]
public unsafe class ScoutQ4KMicroBenchmarks
{
    // Llama-4-Scout-17B-16E dimensions
    private const int EmbDim = 5120;
    private const int NumHeads = 40;
    private const int NumKvHeads = 8;
    private const int HeadDim = 128;
    private const int QDim = NumHeads * HeadDim;   // 5120
    private const int KvDim = NumKvHeads * HeadDim; // 1024
    private const int ExpertDim = 3072;

    // Q4_K: 256 elements per block, 144 bytes per block
    private const int QK_K = 256;
    private const int BytesPerBlockQ4K = 144;
    private const int BytesPerBlockQ6K = 210;

    // Weight buffers (Q4_K layout)
    private byte* _qWeight;       // [QDim, EmbDim]
    private byte* _kWeight;       // [KvDim, EmbDim]
    private byte* _vWeight;       // [KvDim, EmbDim]
    private byte* _expertGateW;   // [ExpertDim, EmbDim]
    private byte* _expertUpW;     // [ExpertDim, EmbDim]
    private byte* _expertDownW;   // [EmbDim, ExpertDim]
    private byte* _outputW;       // [VocabSlice, EmbDim] — smaller slice for micro-bench

    // Weight buffers (Q6_K layout — used for attn_output and expert_down in Q4_K_M)
    private byte* _attnOutW_Q6K;  // [EmbDim, EmbDim]
    private byte* _expDownW_Q6K;  // [EmbDim, ExpertDim]

    // Float buffers
    private float* _input;
    private float* _output1;
    private float* _output2;
    private float* _normWeight;
    private float* _gate;
    private float* _up;

    private const int VocabSlice = 4096; // small slice for output projection bench

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        _input = Alloc(EmbDim);
        _output1 = Alloc(Math.Max(QDim, Math.Max(ExpertDim, VocabSlice)));
        _output2 = Alloc(Math.Max(QDim, ExpertDim));
        _normWeight = Alloc(EmbDim);
        _gate = Alloc(ExpertDim);
        _up = Alloc(ExpertDim);

        // Fill float buffers with random data
        FillRandom(_input, EmbDim, rng);
        FillRandom(_normWeight, EmbDim, rng);
        FillRandom(_gate, ExpertDim, rng);
        FillRandom(_up, ExpertDim, rng);

        // Allocate and fill Q4_K weight buffers
        _qWeight = AllocQ4K(QDim, EmbDim, rng);
        _kWeight = AllocQ4K(KvDim, EmbDim, rng);
        _vWeight = AllocQ4K(KvDim, EmbDim, rng);
        _expertGateW = AllocQ4K(ExpertDim, EmbDim, rng);
        _expertUpW = AllocQ4K(ExpertDim, EmbDim, rng);
        _expertDownW = AllocQ4K(EmbDim, ExpertDim, rng);
        _outputW = AllocQ4K(VocabSlice, EmbDim, rng);

        // Allocate and fill Q6_K weight buffers
        _attnOutW_Q6K = AllocQ6K(EmbDim, EmbDim, rng);
        _expDownW_Q6K = AllocQ6K(EmbDim, ExpertDim, rng);
    }

    // ================================================================
    //  Single-row dot product
    // ================================================================

    [Benchmark(Description = "DotQ4K single row (5120 cols)")]
    public float DotQ4K_SingleRow()
    {
        return SimdKernels.DotQ4K(_qWeight, _input, EmbDim);
    }

    // ================================================================
    //  MatVec at different Scout projection sizes
    // ================================================================

    [Benchmark(Description = "MatVec Q4K Q-proj [5120,5120]")]
    public void MatVec_QProj()
    {
        SimdKernels.MatVecQ4K(_output1, _qWeight, _input, QDim, EmbDim);
    }

    [Benchmark(Description = "MatVec Q4K K-proj [1024,5120]")]
    public void MatVec_KProj()
    {
        SimdKernels.MatVecQ4K(_output1, _kWeight, _input, KvDim, EmbDim);
    }

    [Benchmark(Description = "MatVec Q4K expert gate [3072,5120]")]
    public void MatVec_ExpertGate()
    {
        SimdKernels.MatVecQ4K(_output1, _expertGateW, _input, ExpertDim, EmbDim);
    }

    [Benchmark(Description = "MatVec Q4K expert down [5120,3072]")]
    public void MatVec_ExpertDown()
    {
        SimdKernels.MatVecQ4K(_output1, _expertDownW, _input, EmbDim, ExpertDim);
    }

    [Benchmark(Description = "MatVec Q4K output proj [4096,5120]")]
    public void MatVec_OutputProj()
    {
        SimdKernels.MatVecQ4K(_output1, _outputW, _input, VocabSlice, EmbDim);
    }

    // ================================================================
    //  K+V: separate vs fused (MatVecDual)
    // ================================================================

    [Benchmark(Description = "K+V separate (2× MatVec)")]
    public void KV_Separate()
    {
        SimdKernels.MatVecQ4K(_output1, _kWeight, _input, KvDim, EmbDim);
        SimdKernels.MatVecQ4K(_output2, _vWeight, _input, KvDim, EmbDim);
    }

    [Benchmark(Description = "K+V fused (MatVecDual)")]
    public void KV_Fused()
    {
        SimdKernels.MatVecDual(_output1, _kWeight, _output2, _vWeight,
            _input, KvDim, EmbDim, DType.Q4_K, DType.Q4_K);
    }

    // ================================================================
    //  Expert gate+up: separate vs fused
    // ================================================================

    [Benchmark(Description = "Expert gate+up separate (2× MatVec)")]
    public void ExpertGateUp_Separate()
    {
        SimdKernels.MatVecQ4K(_output1, _expertGateW, _input, ExpertDim, EmbDim);
        SimdKernels.MatVecQ4K(_output2, _expertUpW, _input, ExpertDim, EmbDim);
    }

    [Benchmark(Description = "Expert gate+up fused (MatVecDual)")]
    public void ExpertGateUp_Fused()
    {
        SimdKernels.MatVecDual(_output1, _expertGateW, _output2, _expertUpW,
            _input, ExpertDim, EmbDim, DType.Q4_K, DType.Q4_K);
    }

    // ================================================================
    //  Weighted accumulate: scalar vs SIMD
    // ================================================================

    [Benchmark(Description = "WeightedAdd scalar (5120 elems)")]
    public void WeightedAdd_Scalar()
    {
        float weight = 0.75f;
        for (int i = 0; i < EmbDim; i++)
            _output1[i] += weight * _output2[i];
    }

    [Benchmark(Description = "WeightedAdd SIMD (5120 elems)")]
    public void WeightedAdd_Simd()
    {
        SimdKernels.WeightedAddInPlace(_output1, _output2, 0.75f, EmbDim);
    }

    // ================================================================
    //  Q6_K operations (used for attn_output, expert_down in Q4_K_M)
    // ================================================================

    [Benchmark(Description = "DotQ6K single row (5120 cols)")]
    public float DotQ6K_SingleRow()
    {
        return SimdKernels.DotQ6K(_attnOutW_Q6K, _input, EmbDim);
    }

    [Benchmark(Description = "MatVec Q6K attn_output [5120,5120]")]
    public void MatVec_Q6K_AttnOut()
    {
        SimdKernels.MatVecQ6K(_output1, _attnOutW_Q6K, _input, EmbDim, EmbDim);
    }

    [Benchmark(Description = "MatVec Q6K expert down [5120,3072]")]
    public void MatVec_Q6K_ExpertDown()
    {
        SimdKernels.MatVecQ6K(_output1, _expDownW_Q6K, _input, EmbDim, ExpertDim);
    }

    // ================================================================
    //  Element-wise ops at Scout dimensions
    // ================================================================

    [Benchmark(Description = "RmsNorm (5120 elems)")]
    public void RmsNorm_EmbDim()
    {
        SimdKernels.RmsNorm(_output1, _input, _normWeight, EmbDim, 1e-5f);
    }

    [Benchmark(Description = "SiLuMul (3072 elems)")]
    public void SiLuMul_ExpertDim()
    {
        SimdKernels.SiLuMul(_gate, _up, ExpertDim);
    }

    // ================================================================
    //  Helpers
    // ================================================================

    private static float* Alloc(int count)
    {
        return (float*)NativeMemory.AllocZeroed((nuint)((long)count * sizeof(float)));
    }

    private static void FillRandom(float* buf, int count, Random rng)
    {
        for (int i = 0; i < count; i++)
            buf[i] = (float)(rng.NextDouble() * 2 - 1);
    }

    /// <summary>Allocate a [rows, cols] Q4_K weight matrix with valid block structure.</summary>
    private static byte* AllocQ4K(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / QK_K;
        int bytesPerRow = blocksPerRow * BytesPerBlockQ4K;
        long totalBytes = (long)rows * bytesPerRow;
        var ptr = (byte*)NativeMemory.AllocZeroed((nuint)totalBytes);

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* block = ptr + (long)r * bytesPerRow + b * BytesPerBlockQ4K;
                // d = 0.01 as FP16
                block[0] = 0x1E; block[1] = 0x21;
                // dmin = 0.005 as FP16
                block[2] = 0x14; block[3] = 0x19;
                for (int i = 4; i < BytesPerBlockQ4K; i++)
                    block[i] = (byte)rng.Next(256);
            }
        }
        return ptr;
    }

    /// <summary>Allocate a [rows, cols] Q6_K weight matrix with valid block structure.</summary>
    private static byte* AllocQ6K(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / QK_K;
        int bytesPerRow = blocksPerRow * BytesPerBlockQ6K;
        long totalBytes = (long)rows * bytesPerRow;
        var ptr = (byte*)NativeMemory.AllocZeroed((nuint)totalBytes);

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* block = ptr + (long)r * bytesPerRow + b * BytesPerBlockQ6K;
                // ql[128] + qh[64] + sc[16] + d[2] = 210
                // d = 0.01 as FP16 at offset 208
                block[208] = 0x1E; block[209] = 0x21;
                // Fill ql, qh, sc with random data
                for (int i = 0; i < 208; i++)
                    block[i] = (byte)rng.Next(256);
            }
        }
        return ptr;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.Free(_input);
        NativeMemory.Free(_output1);
        NativeMemory.Free(_output2);
        NativeMemory.Free(_normWeight);
        NativeMemory.Free(_gate);
        NativeMemory.Free(_up);
        NativeMemory.Free(_qWeight);
        NativeMemory.Free(_kWeight);
        NativeMemory.Free(_vWeight);
        NativeMemory.Free(_expertGateW);
        NativeMemory.Free(_expertUpW);
        NativeMemory.Free(_expertDownW);
        NativeMemory.Free(_outputW);
        NativeMemory.Free(_attnOutW_Q6K);
        NativeMemory.Free(_expDownW_Q6K);
    }
}
