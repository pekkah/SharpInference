using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #146: validates the <c>mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32</c>
/// tensor-core building block (kernel <c>llm_mma_test_m16n8k16_f32</c>,
/// <see cref="CudaBackend.MmaTestM16N8K16"/>) in isolation before it is used by the
/// TC flash-attention prefill. A known A[16×16] and B[16×8] are multiplied on the
/// tensor cores (fp16 multiplicands, fp32 accumulate) and compared against a CPU
/// fp16-rounded reference. A wrong A/B/C fragment→lane→register map silently
/// produces garbage, so this is the go/no-go gate for the fragment layouts.
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaMmaPrimitiveTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    [Fact]
    public void MmaM16N8K16_F32_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int M = 16, K = 16, N = 8;
        var rng = new Random(20260606);

        // A[16×16] row-major, B[16×8] K-major (b[k*N+n] = B[k][n]). Keep magnitudes
        // modest so fp16 rounding of the inputs stays the only meaningful error.
        var a = new float[M * K];
        var b = new float[K * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU reference with inputs rounded to fp16 (the tensor core rounds A,B to
        // fp16 before the multiply; accumulation is fp32 on both sides).
        var cRef = new float[M * N];
        for (int i = 0; i < M; i++)
            for (int n = 0; n < N; n++)
            {
                float acc = 0f;
                for (int k = 0; k < K; k++)
                    acc += (float)(Half)a[i * K + k] * (float)(Half)b[k * N + n];
                cRef[i * N + n] = acc;
            }

        var gpuA = gpu.Upload(a, TensorShape.D1(a.Length));
        var gpuB = gpu.Upload(b, TensorShape.D1(b.Length));
        var gpuC = gpu.Allocate(TensorShape.D1(M * N));

        gpu.MmaTestM16N8K16(gpuA, gpuB, gpuC);
        gpu.Synchronize();

        var cGpu = new float[M * N];
        gpu.Download(gpuC, cGpu);
        gpu.Free(gpuA);
        gpu.Free(gpuB);
        gpu.Free(gpuC);

        // Per-element magnitude ~ sqrt(K) for ±1 inputs. fp16 mantissa is ~2^-11, so
        // the dominant error is the input rounding; a few 1e-3 absolute is expected.
        float maxAbs = 0f;
        int mismatches = 0;
        for (int i = 0; i < cRef.Length; i++)
        {
            float diff = MathF.Abs(cGpu[i] - cRef[i]);
            maxAbs = MathF.Max(maxAbs, diff);
            if (diff > 1e-2f) mismatches++;
        }
        Console.WriteLine($"mma m16n8k16: maxAbs={maxAbs:E3} mismatches={mismatches}/{cRef.Length}");
        // Dump the matrices on failure so a fragment-layout bug is diagnosable.
        if (mismatches > 0)
        {
            Console.WriteLine("  row  | gpu vs ref");
            for (int i = 0; i < M; i++)
            {
                var line = new System.Text.StringBuilder($"  q{i,2}: ");
                for (int n = 0; n < N; n++)
                    line.Append($"{cGpu[i * N + n],8:F3}/{cRef[i * N + n],-8:F3} ");
                Console.WriteLine(line.ToString());
            }
        }
        Assert.True(mismatches == 0,
            $"mma.sync m16n8k16 fragment layout produced wrong results: {mismatches}/{cRef.Length} elements off, maxAbs={maxAbs:E3}.");
    }
}
