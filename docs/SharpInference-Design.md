# SharpInference — Design Document

**A .NET 10 / C# 14 LLM inference engine optimized for consumer desktop hardware**

Version: 0.1.0-draft
Date: 2026-03-30
Author: Pekka (with Claude)

---

## 1. Vision and Goals

SharpInference is an experimental LLM inference engine written entirely in modern .NET, designed to extract maximum performance from consumer desktop hardware. The project proves that C# with Vulkan compute, aggressive memory tiering, and cutting-edge KV cache compression can compete with native C++ inference engines on hardware that most developers already own.

### 1.1 Target Hardware Profile

| Component | Specification | Role |
|-----------|--------------|------|
| GPU | NVIDIA 12GB VRAM (e.g. RTX 3060/4070) | Hot compute + weight cache |
| System RAM | 64GB DDR4/DDR5 | Warm weight storage + KV cache overflow |
| Storage | NVMe SSD (5–7 GB/s) | Cold weight storage, model loading |
| CPU | Modern x86-64 with AVX2/AVX-512 | Expert FFN compute, preprocessing |
| Bus | PCIe 4.0 x16 (~25 GB/s) | GPU ↔ RAM data transfer |

### 1.2 Design Principles

1. **Data logistics over compute optimization.** The GPU has surplus FLOPS. The bottleneck is always memory bandwidth — getting the right weights to the right place at the right time. Every design decision optimizes for data movement.
2. **Never let the GPU stall.** All data transfers (RAM→VRAM, SSD→RAM) must overlap with compute. The GPU should always have work to do.
3. **Correct first, fast second.** Every acceleration technique is validated against a reference CPU implementation before deployment.
4. **NativeAOT from day one.** Trim and AOT analyzers enabled throughout development. Release builds produce a single statically-linked binary with zero JIT overhead.
5. **No managed heap allocations in the hot path.** All performance-critical memory uses `NativeMemory`, `Span<T>`, Vulkan buffers, or pinned CUDA/host allocations.

### 1.3 Non-Goals

- Training or fine-tuning (inference only).
- Multi-GPU tensor parallelism (single GPU + CPU offload).
- Mobile or embedded targets (desktop Linux/Windows only).
- Replacing llama.cpp for general use (this is an experimental/research project).

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    API Server Layer                          │
│         Anthropic Messages API compatible HTTP server        │
│         ASP.NET Core Minimal API / NativeAOT compatible     │
├─────────────────────────────────────────────────────────────┤
│                   Orchestration Layer                        │
│     Speculative decoding · Sampling · Token streaming        │
├──────────┬──────────────┬───────────────────────────────────┤
│ Pipeline │  TurboQuant   │        Compute Backends          │
│ Manager  │  KV Cache     │  ┌─────────┐  ┌──────────────┐  │
│          │  Compression  │  │ Vulkan   │  │ CPU (SIMD)   │  │
│ Tier L1  │               │  │ Compute  │  │ AVX2/AVX-512 │  │
│ Tier L2  │  3-bit keys   │  │ Vortice  │  │ Fallback     │  │
│ Tier L3  │  2-bit values │  └─────────┘  └──────────────┘  │
├──────────┴──────────────┴───────────────────────────────────┤
│                     Core Layer                               │
│     GGUF parser · Tokenizer · Tensor types · Model graphs    │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Layer Responsibilities

**Core Layer** — Model loading, tokenization, tensor storage, and the abstract computation graph. No hardware-specific code. Pure C# with `Span<T>` and `NativeMemory`.

**Compute Backends** — Pluggable backends that execute tensor operations. The Vulkan backend handles GPU dispatch via Vortice.Vulkan. The CPU backend provides a SIMD-optimized fallback using `System.Runtime.Intrinsics`. Both implement the same `IComputeBackend` interface.

**TurboQuant** — KV cache compression using the TurboQuant algorithm (ICLR 2026). Precomputed Lloyd-Max codebooks, randomized Hadamard rotation, 3-bit packing. Both CPU and Vulkan shader implementations.

**Pipeline Manager** — Three-tier memory hierarchy (VRAM → pinned RAM → NVMe), async prefetching via `Channel<T>`, expert cache with SLRU eviction, and `io_uring` integration for SSD reads.

**Orchestration Layer** — Speculative decoding, temperature/top-p/top-k sampling, stop sequence detection, token-by-token streaming via `IAsyncEnumerable<Token>`.

**API Server Layer** — Anthropic Messages API compatible HTTP server. Accepts `/v1/messages` requests, returns streaming SSE responses. Also exposes an OpenAI-compatible `/v1/chat/completions` endpoint for broader tooling compatibility.

---

## 3. Core Layer

### 3.1 GGUF Parser

The GGUF format is a flat binary container: a header with key-value metadata, tensor descriptors, then raw tensor data. The parser uses `MemoryMappedFile` for zero-copy access to tensor data on disk.

```csharp
public sealed class GgufModel : IDisposable
{
    public GgufMetadata Metadata { get; }
    public IReadOnlyList<GgufTensorInfo> Tensors { get; }

    // Zero-copy view into memory-mapped file
    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor);

    // Load specific tensor into target memory (VRAM, pinned RAM, etc.)
    public void LoadTensor(GgufTensorInfo tensor, Span<byte> destination);

    public static GgufModel Open(string path);
}

public readonly record struct GgufTensorInfo(
    string Name,
    GgufType DataType,     // F32, F16, Q4_K_M, Q8_0, etc.
    ReadOnlyMemory<int> Shape,
    long FileOffset,
    long ByteSize);
```

Key design decisions:
- Memory-mapped I/O via `MemoryMappedFile` for lazy loading — the OS handles paging from SSD.
- Tensor data is never copied into managed memory — `GetTensorData` returns a span directly into the mapped region.
- Metadata parsed eagerly on open (small), tensor data accessed lazily (large).

### 3.2 Tokenizer

Tokenization is delegated to `Microsoft.ML.Tokenizers`, which supports BPE and SentencePiece models natively. No custom tokenizer implementation.

```csharp
public interface ITokenizer
{
    ReadOnlyMemory<int> Encode(ReadOnlySpan<char> text);
    string Decode(ReadOnlySpan<int> tokens);
    int VocabularySize { get; }
    int BosToken { get; }
    int EosToken { get; }
}
```

### 3.3 Tensor Types

All tensors are unmanaged memory with type-safe wrappers. No `float[]` arrays — everything is `NativeMemory` or device buffers.

```csharp
// Unmanaged tensor on CPU
public readonly ref struct CpuTensor<T> where T : unmanaged
{
    public readonly Span<T> Data;
    public readonly ReadOnlySpan<int> Shape;
    public readonly int Stride;

    // No managed heap allocation — backed by NativeMemory
}

// Handle to a Vulkan storage buffer on GPU
public readonly record struct GpuTensor(
    VkBuffer Buffer,
    VkDeviceMemory Memory,
    long ByteSize,
    ReadOnlyMemory<int> Shape,
    TensorFormat Format);
```

### 3.4 Model Graph

The model is represented as a static graph of layers, constructed from GGUF metadata. This graph drives both the forward pass and the tier placement decisions.

```csharp
public sealed class ModelGraph
{
    public ModelArchitecture Architecture { get; }  // LLaMA, Qwen, Mistral, etc.
    public int NumLayers { get; }
    public int HiddenDim { get; }
    public int NumHeads { get; }
    public int NumKvHeads { get; }     // GQA support
    public int VocabSize { get; }
    public int MaxSeqLen { get; }
    public bool IsMoe { get; }
    public int? NumExperts { get; }
    public int? NumActiveExperts { get; }

    public IReadOnlyList<LayerWeights> Layers { get; }
    public EmbeddingWeights Embedding { get; }
    public OutputWeights Output { get; }
}
```

---

## 4. Compute Backends

### 4.1 Interface

```csharp
public interface IComputeBackend : IDisposable
{
    // Core operations
    void MatVecMul(TensorRef output, TensorRef matrix, TensorRef vector);
    void MatVecMulDequant(TensorRef output, TensorRef quantMatrix, TensorRef vector,
                          QuantFormat format);
    void RmsNorm(TensorRef output, TensorRef input, TensorRef weight, float epsilon);
    void RoPE(TensorRef qk, int position, int headDim, float ropeTheta);
    void Softmax(TensorRef output, TensorRef input);
    void SiLU(TensorRef output, TensorRef input);
    void ElementwiseMul(TensorRef output, TensorRef a, TensorRef b);
    void ResidualAdd(TensorRef output, TensorRef a, TensorRef b);

    // Attention
    void Attention(TensorRef output, TensorRef q, TensorRef k, TensorRef v,
                   KvCache cache, int position, AttentionMask? mask);

    // Memory management
    TensorRef Allocate(ReadOnlySpan<int> shape, TensorFormat format);
    void Free(TensorRef tensor);
    void CopyToDevice(TensorRef dst, ReadOnlySpan<byte> src);
    void CopyFromDevice(Span<byte> dst, TensorRef src);
}
```

### 4.2 CPU Backend

Reference implementation using `System.Runtime.Intrinsics` for SIMD acceleration.

```
SharpInference.Cpu/
├── CpuBackend.cs              # IComputeBackend implementation
├── Simd/
│   ├── MatVecAvx2.cs          # AVX2 FP32 matrix-vector multiply
│   ├── MatVecAvx512.cs        # AVX-512 path (runtime feature check)
│   ├── DequantQ4K.cs          # Q4_K_M dequantization with SIMD
│   ├── HadamardAvx2.cs        # Walsh-Hadamard transform for TurboQuant
│   └── SimdHelper.cs          # Runtime ISA detection, dispatch
└── Reference/
    └── ScalarOps.cs            # Naive scalar fallback for validation
```

Runtime dispatch pattern:

```csharp
public static class MatVec
{
    public static void Multiply(Span<float> output, ReadOnlySpan<float> matrix,
                                ReadOnlySpan<float> vector, int rows, int cols)
    {
        if (Avx512F.IsSupported)
            MatVecAvx512.Execute(output, matrix, vector, rows, cols);
        else if (Avx2.IsSupported)
            MatVecAvx2.Execute(output, matrix, vector, rows, cols);
        else
            ScalarOps.MatVec(output, matrix, vector, rows, cols);
    }
}
```

### 4.3 Vulkan Compute Backend

GPU acceleration via Vortice.Vulkan Vulkan. All inference operations are compute shaders dispatched from C#.

```
SharpInference.Vulkan/
├── VulkanBackend.cs            # IComputeBackend implementation
├── VulkanDevice.cs             # Device init, queue families, memory types
├── VulkanBufferPool.cs         # Suballocator for storage buffers
├── PipelineCache.cs            # Compute pipeline compilation + caching
├── CommandScheduler.cs         # Double-buffered command buffer submission
├── DescriptorManager.cs        # Bindless descriptor set management
└── Shaders/
    ├── matmul_f16.comp         # FP16 matrix-vector multiply
    ├── matmul_dequant_q4k.comp # Fused Q4_K_M dequant + matmul
    ├── rmsnorm.comp            # RMSNorm
    ├── rope.comp               # Rotary position embedding
    ├── softmax.comp            # Softmax (online numerically stable)
    ├── silu.comp               # SiLU activation
    ├── attention.comp          # Fused attention kernel
    ├── tq_quantize.comp        # TurboQuant: rotate + quantize + pack
    └── tq_dequant_dot.comp     # TurboQuant: fused unpack + dot product
```

Key design decisions:

- **Bindless descriptors** for weight buffers. All model weights are uploaded to a single large storage buffer (or array of buffers). Shader accesses weights via `buffer_reference` or descriptor indexing with a push-constant offset. No descriptor set rebinding between layers.
- **Double-buffered command submission.** While the GPU executes command buffer N, the CPU records command buffer N+1. This hides CPU-side recording latency.
- **Compute queue separation.** If the device supports async compute queues, DMA transfers (RAM→VRAM for expert prefetching) run on the transfer queue while compute runs on the compute queue. `VkSemaphore` synchronization between them.
- **Shader compilation at startup.** All SPIR-V shaders are compiled from GLSL at build time via `glslangValidator`, embedded as resources, and loaded into `VkPipeline` objects during initialization. No runtime shader compilation.

### 4.4 Backend Selection

```csharp
public static class BackendFactory
{
    public static IComputeBackend Create(BackendPreference preference = BackendPreference.Auto)
    {
        return preference switch
        {
            BackendPreference.Vulkan => new VulkanBackend(),
            BackendPreference.Cpu => new CpuBackend(),
            BackendPreference.Auto => VulkanBackend.IsAvailable()
                ? new VulkanBackend()
                : new CpuBackend(),
            _ => throw new ArgumentOutOfRangeException(nameof(preference))
        };
    }
}
```

---

## 5. TurboQuant KV Cache Compression

### 5.1 Background

The KV cache stores key and value vectors for every token at every layer. For an 8B model at 32K context in FP16, this alone consumes ~4.6GB of VRAM. TurboQuant (Zandieh et al., ICLR 2026) compresses this to 3 bits per value with near-zero accuracy loss, achieving ~6x memory reduction.

The algorithm is data-oblivious: it requires no training, no calibration data, and no model-specific tuning. It works by:

1. Applying a random orthogonal rotation (Walsh-Hadamard transform + sign flips) to each KV vector.
2. The rotation induces a concentrated Beta distribution on each coordinate, regardless of input data.
3. Applying Lloyd-Max optimal scalar quantization per coordinate using precomputed codebooks.
4. Packing the quantized indices into a compact bit representation.

### 5.2 Data Layout

```
Block: 128 values → 52 bytes (3-bit quantization)

┌──────────────┬───────────────────────────────┐
│ FP16 norm    │ Packed 3-bit indices           │
│ (2 bytes)    │ (48 bytes = 128 × 3 bits)     │
├──────────────┼───────────────────────────────┤
│ Padding      │                               │
│ (2 bytes)    │ Total: 52 bytes per block      │
└──────────────┴───────────────────────────────┘

Compression: 128 × 2 bytes (FP16) = 256 bytes → 52 bytes = 4.9x
```

### 5.3 Codebook Generation

Lloyd-Max codebooks are computed offline for the Beta distribution induced by the Hadamard rotation. These are small lookup tables (8 centroids for 3-bit, 16 for 4-bit) embedded as compile-time constants.

```csharp
public static class TurboQuantCodebooks
{
    // 3-bit Lloyd-Max centroids for Beta(d/2, d/2) distribution, d=128
    // Computed via iterative convergence (~178 iterations)
    public static ReadOnlySpan<float> Centroids3Bit => new float[]
    {
        // 8 centroids, precomputed
        -1.1503f, -0.7186f, -0.3579f, -0.0638f,
         0.0638f,  0.3579f,  0.7186f,  1.1503f
    };

    // Decision boundaries (midpoints between centroids)
    public static ReadOnlySpan<float> Boundaries3Bit => new float[]
    {
        -0.9345f, -0.5383f, -0.2109f,
         0.0000f,
         0.2109f,  0.5383f,  0.9345f
    };

    // 4-bit codebook (16 centroids)
    public static ReadOnlySpan<float> Centroids4Bit => /* ... */;
}
```

### 5.4 Core Operations

```csharp
public static class TurboQuant
{
    /// <summary>
    /// Quantize a KV vector to 3-bit TurboQuant representation.
    /// Called once per token per layer on KV cache write.
    /// </summary>
    public static void Quantize(
        ReadOnlySpan<float> input,         // d floats (e.g. 128)
        Span<byte> output,                 // 52 bytes packed
        ReadOnlySpan<float> signPattern,   // deterministic sign flips (precomputed)
        ReadOnlySpan<float> codebook,      // Lloyd-Max centroids
        ReadOnlySpan<float> boundaries,    // decision boundaries
        int dim)
    {
        Span<float> rotated = stackalloc float[dim];

        // Step 1: Walsh-Hadamard transform + sign flip
        WalshHadamard.Transform(input, rotated, dim);
        ApplySignFlip(rotated, signPattern);

        // Step 2: Compute and store norm
        float norm = ComputeL2Norm(rotated);
        BinaryPrimitives.WriteHalfLittleEndian(output, (Half)norm);

        // Step 3: Normalize and quantize each coordinate
        float invNorm = 1.0f / norm;
        for (int i = 0; i < dim; i++)
        {
            float normalized = rotated[i] * invNorm;
            int index = FindNearestBoundary(normalized, boundaries); // 0..7 for 3-bit
            PackBits3(output, offset: 2, i, index);
        }
    }

    /// <summary>
    /// Dequantize and compute dot product in one fused operation.
    /// Called during attention scoring — never fully materializes decompressed cache.
    /// </summary>
    public static float DequantDot(
        ReadOnlySpan<byte> quantized,      // 52 bytes packed
        ReadOnlySpan<float> query,         // d floats (current query vector)
        ReadOnlySpan<float> signPattern,
        ReadOnlySpan<float> codebook,
        int dim)
    {
        float norm = BinaryPrimitives.ReadHalfLittleEndian(quantized).ToSingle();
        float dot = 0f;

        // Fused: unpack index → lookup centroid → multiply by query → accumulate
        Span<float> rotatedQuery = stackalloc float[dim];
        WalshHadamard.Transform(query, rotatedQuery, dim);
        ApplySignFlip(rotatedQuery, signPattern);

        for (int i = 0; i < dim; i++)
        {
            int index = UnpackBits3(quantized, offset: 2, i);
            float reconstructed = codebook[index] * norm;
            dot += reconstructed * rotatedQuery[i];
        }

        return dot;
    }
}
```

### 5.5 Adaptive Precision Strategy

Not all tokens and not all tensor types need the same precision.

| Data | Precision | Rationale |
|------|-----------|-----------|
| Recent tokens (last 128–256) | Full FP16 | Attention focuses heavily on recent context |
| Older key vectors | TQ 3–4 bit | Keys have higher magnitude variance, need more bits |
| Older value vectors | TQ 2–3 bit | Values are more uniform, tolerate aggressive compression |
| Residual window | Configurable | Trade memory for quality on a per-model basis |

The K/V magnitude ratio is profiled per model during warmup. Community findings (llama.cpp #20969) show:
- K/V ratio < 10x → 3-bit uniform works well
- K/V ratio 10–60x → 4-bit keys, 3-bit values
- K/V ratio > 100x → 5+ bit keys or mixed precision

```csharp
public sealed class TurboQuantKvCache
{
    private readonly int _fullPrecisionWindow;   // recent tokens in FP16
    private readonly int _keyBits;               // 3 or 4
    private readonly int _valueBits;             // 2 or 3

    // FP16 ring buffer for recent tokens
    private readonly FP16RingBuffer _recentKeys;
    private readonly FP16RingBuffer _recentValues;

    // Compressed storage for older tokens
    private readonly PackedBuffer _compressedKeys;
    private readonly PackedBuffer _compressedValues;
}
```

### 5.6 Vulkan Compute Shaders

Two fused shaders handle the GPU path:

**`tq_quantize.comp`** — Runs on every KV cache write. Each workgroup processes one 128-dimensional vector: applies WHT via shared memory, normalizes, quantizes against codebook LUT, and packs bits into output buffer.

**`tq_dequant_dot.comp`** — Runs during attention scoring. Each workgroup computes the dot product between a query vector and one compressed KV entry without fully materializing the decompressed vector. This is the critical kernel — it runs for every (query, cached-key) pair during attention.

### 5.7 Memory Impact

Example: Qwen3 8B at Q4_K_M weights, 32K context:

| Configuration | KV Cache Size | Total VRAM (weights + KV + overhead) |
|--------------|---------------|--------------------------------------|
| FP16 KV cache | ~4.6 GB | ~9.8 GB (tight on 12GB) |
| TQ3 keys + TQ2 values | ~0.9 GB | ~6.1 GB (room for 64K+ context) |

This is the difference between "barely fits at 32K" and "comfortably runs 64K+ with VRAM to spare for expert caching."

---

## 6. Memory Hierarchy and Pipeline

### 6.1 Three-Tier Architecture

```
┌─────────────────────────────────────┐
│ L1: GPU VRAM (12 GB)               │
│ ~1 TB/s internal bandwidth          │
│                                     │
│ Residents:                          │
│   - Embedding table                 │
│   - LM head / output projection    │
│   - Attention QKV weights           │
│   - TurboQuant KV cache            │
│   - MoE router weights             │
│   - Expert slot cache (SLRU)        │
├─────────────────────────────────────┤
│ L2: Pinned System RAM (48–56 GB)   │  ← cudaHostAlloc / VK mapped memory
│ ~25 GB/s to GPU via PCIe 4.0       │
│                                     │
│ Residents:                          │
│   - Expert FFN weights (all)        │
│   - Dense model overflow layers     │
│   - KV cache overflow (if needed)   │
├─────────────────────────────────────┤
│ L3: NVMe SSD                       │  ← io_uring async reads
│ ~6 GB/s sequential read             │
│                                     │
│ Residents:                          │
│   - Cold experts (models > 64 GB)   │
│   - Full model file (mmap'd)        │
└─────────────────────────────────────┘
```

### 6.2 Tier Placement Algorithm

During model load, the tier placement profiler assigns each tensor to a tier based on priority:

```
Priority 1 (always VRAM):
  - Embedding table
  - Output projection / LM head
  - MoE router weights
  - RMSNorm weights (tiny, used every layer)

Priority 2 (VRAM if space permits):
  - Attention Q, K, V, O projection weights
  - KV cache (TurboQuant compressed)

Priority 3 (pinned RAM, DMA to VRAM on demand):
  - FFN gate/up/down weights (dense models)
  - MoE expert weights (all experts)

Priority 4 (NVMe, promote to RAM on access):
  - Cold expert weights when total model > RAM capacity
```

```csharp
public sealed class TierPlacementPlanner
{
    public TierAssignment Plan(ModelGraph model, HardwareProfile hardware)
    {
        var assignment = new TierAssignment();
        long vramBudget = hardware.VramBytes - ReserveForKvCache(model, hardware);
        long ramBudget = hardware.RamBytes - ReserveForOs();

        // Priority 1: always VRAM
        foreach (var tensor in model.GetEmbeddingTensors())
            assignment.Assign(tensor, Tier.Vram, ref vramBudget);

        foreach (var tensor in model.GetRouterTensors())
            assignment.Assign(tensor, Tier.Vram, ref vramBudget);

        // Priority 2: VRAM if fits
        foreach (var tensor in model.GetAttentionTensors())
        {
            if (vramBudget >= tensor.ByteSize)
                assignment.Assign(tensor, Tier.Vram, ref vramBudget);
            else
                assignment.Assign(tensor, Tier.PinnedRam, ref ramBudget);
        }

        // Priority 3: pinned RAM
        foreach (var tensor in model.GetExpertTensors())
        {
            if (ramBudget >= tensor.ByteSize)
                assignment.Assign(tensor, Tier.PinnedRam, ref ramBudget);
            else
                assignment.Assign(tensor, Tier.Nvme);
        }

        return assignment;
    }
}
```

### 6.3 VRAM Expert Slot Cache

For MoE models, a fixed number of expert-sized slots are reserved in VRAM. Experts are cached using an SLRU (Segmented LRU) eviction policy, exploiting the observation that MoE routing is heavily skewed — approximately 15–20% of experts handle ~80% of tokens.

```csharp
public sealed class ExpertSlotCache
{
    private readonly int _slotCount;           // e.g., 32–64 slots
    private readonly long _slotByteSize;       // size of one expert's weights
    private readonly VkBuffer _slotBuffer;     // contiguous VRAM allocation
    private readonly SlruPolicy _eviction;

    /// <summary>
    /// Returns the VRAM offset if the expert is cached (hit), or
    /// evicts the coldest slot and returns it for DMA fill (miss).
    /// </summary>
    public ExpertCacheResult Lookup(int layerIndex, int expertId);

    /// <summary>
    /// Async DMA fill from pinned RAM into an evicted slot.
    /// Returns a fence that the compute queue waits on before dispatch.
    /// </summary>
    public ValueTask<VkFence> FillAsync(ExpertCacheResult miss,
                                         ReadOnlyMemory<byte> pinnedSource,
                                         CancellationToken ct);
}
```

### 6.4 Async Prefetching Pipeline

The router layer reveals which experts are needed before the expert FFN computation begins. This lookahead drives predictive prefetching.

```csharp
public sealed class PrefetchPipeline : IAsyncDisposable
{
    private readonly Channel<PrefetchRequest> _channel;
    private readonly ExpertSlotCache _vramCache;
    private readonly PinnedMemoryPool _ramPool;
    private readonly IoUringReader _nvmeReader;    // optional, for L3 tier
    private readonly Task _consumerTask;

    public PrefetchPipeline(ExpertSlotCache vramCache, PinnedMemoryPool ramPool)
    {
        _channel = Channel.CreateBounded<PrefetchRequest>(
            new BoundedChannelOptions(32) { SingleWriter = false, SingleReader = true });

        _consumerTask = Task.Factory.StartNew(
            ConsumeLoop, TaskCreationOptions.LongRunning);
    }

    /// <summary>
    /// Called by router immediately after expert selection.
    /// Non-blocking — just enqueues the prefetch request.
    /// </summary>
    public void RequestPrefetch(int layerIndex, ReadOnlySpan<int> selectedExperts)
    {
        foreach (int expertId in selectedExperts)
        {
            var result = _vramCache.Lookup(layerIndex, expertId);
            if (result.IsHit) continue;  // already in VRAM

            _channel.Writer.TryWrite(new PrefetchRequest(layerIndex, expertId, result));
        }
    }

    private async Task ConsumeLoop()
    {
        await foreach (var req in _channel.Reader.ReadAllAsync())
        {
            if (_ramPool.Contains(req.LayerIndex, req.ExpertId))
            {
                // L2 hit: DMA from pinned RAM → VRAM slot
                var source = _ramPool.GetPinned(req.LayerIndex, req.ExpertId);
                await _vramCache.FillAsync(req.CacheResult, source, CancellationToken.None);
            }
            else if (_nvmeReader is not null)
            {
                // L3: async read from SSD → pinned RAM → VRAM
                var ram = _ramPool.AllocateSlot(req.LayerIndex, req.ExpertId);
                await _nvmeReader.ReadAsync(req.FileOffset, ram, CancellationToken.None);
                await _vramCache.FillAsync(req.CacheResult, ram, CancellationToken.None);
            }
        }
    }
}
```

### 6.5 io_uring Integration

For NVMe reads that bypass the OS page cache, a thin `io_uring` interop layer provides fully async, zero-copy SSD access. This is only needed when models exceed RAM capacity.

```csharp
/// <summary>
/// Minimal io_uring wrapper for async NVMe reads.
/// ~200 lines of P/Invoke to Linux io_uring syscalls.
/// </summary>
public sealed class IoUringReader : IDisposable
{
    private readonly int _ringFd;
    private readonly int _fileFd;

    [LibraryImport("libc", EntryPoint = "io_uring_setup")]
    private static partial int IoUringSetup(uint entries, ref IoUringParams p);

    public ValueTask<int> ReadAsync(long fileOffset, Memory<byte> destination,
                                     CancellationToken ct);
}
```

### 6.6 Pipelined Dense Model Inference

For dense models that don't fit in VRAM, the pipeline double-buffers layer weights:

```
Time →
GPU:   [compute layer N  ] [compute layer N+1] [compute layer N+2] ...
DMA:      [load layer N+1]    [load layer N+2]    [load layer N+3] ...
              ↑ overlapped ↑
```

Two VRAM buffers are allocated, each large enough for one layer. While the GPU computes on buffer A, DMA fills buffer B from pinned RAM. They swap each layer.

```csharp
public sealed class DoubleBufferedLayerStreamer
{
    private readonly GpuTensor _bufferA;
    private readonly GpuTensor _bufferB;
    private int _activeBuffer;       // 0 = A computing, B loading; 1 = swapped

    public async ValueTask StreamLayerAsync(int layerIndex, IComputeBackend gpu,
                                             PinnedMemoryPool ram, CancellationToken ct)
    {
        var computeBuffer = _activeBuffer == 0 ? _bufferA : _bufferB;
        var loadBuffer = _activeBuffer == 0 ? _bufferB : _bufferA;

        // Issue DMA for next layer (non-blocking)
        var dmaFence = gpu.BeginDmaAsync(
            source: ram.GetLayerWeights(layerIndex + 1),
            destination: loadBuffer);

        // Compute current layer on GPU (uses computeBuffer which was loaded last iteration)
        gpu.DispatchLayerForward(layerIndex, computeBuffer);

        // Wait for both to complete
        await gpu.WaitFenceAsync(dmaFence, ct);

        _activeBuffer ^= 1;  // swap
    }
}
```

### 6.7 MoE Inference Flow

For MoE models, the flow combines routing, prefetching, and split compute:

```
For each token:
  1. GPU: compute attention (weights resident in VRAM)
  2. GPU: compute router → selected expert IDs
  3. Pipeline: issue async prefetch for selected experts
  4. GPU: compute previous token's expert FFN (already in VRAM from prior prefetch)
  5. Sync: wait for current prefetch if not yet complete
  6. Advance: current experts become "previous" for next token
```

The activation vector is small (hidden_dim floats, e.g., 4096 × 4 = 16KB), so in cases where an expert misses both VRAM and the DMA pipeline, falling back to CPU-side compute via PCIe round-trip is acceptable: send the 16KB activation to CPU, compute the expert FFN in RAM using AVX2, return the 16KB result. This is the same strategy llama.cpp uses.

---

## 7. Speculative Decoding

### 7.1 Concept

A small draft model (e.g., SmolLM2 1.7B or the target model's smallest variant) runs entirely in VRAM and generates N candidate tokens speculatively. The large target model then verifies all N tokens in a single batched forward pass. If K out of N candidates match, we've generated K+1 tokens for the cost of one large-model pass plus N cheap draft passes.

### 7.2 Implementation

```csharp
public sealed class SpeculativeDecoder
{
    private readonly InferenceEngine _draftModel;   // small, fast, fully in VRAM
    private readonly InferenceEngine _targetModel;  // large, may use offloading
    private readonly int _specTokenCount;            // N candidates per round (typically 4)

    public async IAsyncEnumerable<int> GenerateAsync(
        ReadOnlyMemory<int> prompt,
        SamplingParams sampling,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var context = prompt;

        while (true)
        {
            // Draft: generate N candidate tokens (fast)
            Span<int> candidates = stackalloc int[_specTokenCount];
            Span<float> draftLogits = stackalloc float[_specTokenCount * _draftModel.VocabSize];
            for (int i = 0; i < _specTokenCount; i++)
                candidates[i] = _draftModel.ForwardAndSample(context, sampling);

            // Verify: single batched forward pass of target model
            var targetLogits = _targetModel.ForwardBatch(context, candidates);

            // Accept/reject using standard speculative sampling
            int accepted = VerifyAndAccept(candidates, draftLogits, targetLogits, sampling);

            for (int i = 0; i <= accepted; i++)
            {
                yield return candidates[i];
                if (candidates[i] == _targetModel.EosToken) yield break;
            }

            // Advance context
            context = AppendTokens(context, candidates[..(accepted + 1)]);
        }
    }
}
```

### 7.3 VRAM Budget for Speculative Decoding

The draft model must coexist with the target model's VRAM-resident components. SmolLM2 1.7B at Q4_K_M is ~1GB, leaving 11GB for the target model's attention layers, KV cache, and expert slot cache. This is tight but workable.

---

## 8. Forward Pass

### 8.1 Dense Transformer Forward Pass

The core inference loop for a standard transformer decoder:

```csharp
public sealed class DenseForwardPass
{
    private readonly ModelGraph _model;
    private readonly IComputeBackend _backend;
    private readonly KvCacheManager _kvCache;

    public void Forward(ReadOnlySpan<int> tokens, int startPos, Span<float> logitsOut)
    {
        // Embed
        var hidden = _backend.Allocate(stackalloc int[] { tokens.Length, _model.HiddenDim },
                                        TensorFormat.F16);
        _backend.Embed(hidden, tokens, _model.Embedding);

        // Transformer layers
        for (int i = 0; i < _model.NumLayers; i++)
        {
            ref var layer = ref _model.Layers[i];
            var residual = _backend.Clone(hidden);

            // Pre-attention norm
            _backend.RmsNorm(hidden, hidden, layer.AttnNorm, _model.RmsEpsilon);

            // Self-attention with GQA
            _backend.Attention(hidden, hidden, _kvCache, layer.Attn, startPos);

            // Residual
            _backend.ResidualAdd(hidden, hidden, residual);
            _backend.Clone(residual, hidden);

            // Pre-FFN norm
            _backend.RmsNorm(hidden, hidden, layer.FfnNorm, _model.RmsEpsilon);

            // Feed-forward: gate * SiLU(up) then down
            _backend.FeedForward(hidden, hidden, layer.Ffn);

            // Residual
            _backend.ResidualAdd(hidden, hidden, residual);
        }

        // Final norm + output projection
        _backend.RmsNorm(hidden, hidden, _model.FinalNorm, _model.RmsEpsilon);
        _backend.MatVecMul(logitsOut, _model.Output.Weight, hidden);
    }
}
```

### 8.2 MoE Forward Pass

The MoE variant replaces the dense FFN with a routed expert dispatch:

```csharp
public void MoeForward(/* ... */)
{
    // ... same attention path as dense ...

    // MoE FFN
    for (int i = 0; i < _model.NumLayers; i++)
    {
        // Router: small linear → softmax → top-K expert selection
        Span<int> selectedExperts = stackalloc int[_model.NumActiveExperts];
        Span<float> expertWeights = stackalloc float[_model.NumActiveExperts];
        _backend.Route(selectedExperts, expertWeights, hidden, _model.Layers[i].Router);

        // Prefetch next experts (async, non-blocking)
        _prefetchPipeline.RequestPrefetch(i, selectedExperts);

        // Compute selected experts (from VRAM cache or CPU fallback)
        _backend.MoeExpertFfn(hidden, hidden, selectedExperts, expertWeights,
                               _expertCache, _model.Layers[i]);
    }
}
```

---

## 9. API Server Layer

### 9.1 Overview

SharpInference exposes an HTTP API server compatible with the Anthropic Messages API, enabling drop-in use with existing client libraries, SDKs, and tools that target the Anthropic API. An OpenAI-compatible endpoint is also provided for broader ecosystem compatibility.

The server is built with ASP.NET Core Minimal APIs and is fully NativeAOT compatible.

### 9.2 Supported Endpoints

#### Anthropic Messages API

```
POST /v1/messages
```

Accepts the Anthropic Messages API format:

```json
{
  "model": "sharpinference-qwen3-30b-a3b",
  "max_tokens": 1024,
  "messages": [
    { "role": "user", "content": "Explain how TurboQuant works." }
  ],
  "stream": true,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stop_sequences": ["\n\nHuman:"],
  "system": "You are a helpful assistant."
}
```

Streaming responses use SSE (Server-Sent Events) matching the Anthropic wire format:

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message","role":"assistant","content":[],"model":"sharpinference-qwen3-30b-a3b","usage":{"input_tokens":42}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"TurboQuant"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" is a"}}

...

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":128}}

event: message_stop
data: {"type":"message_stop"}
```

#### OpenAI Chat Completions API

```
POST /v1/chat/completions
```

Standard OpenAI format for compatibility with tools expecting the OpenAI API shape (LangChain, LlamaIndex, Continue.dev, etc.):

```json
{
  "model": "sharpinference-qwen3-30b-a3b",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello" }
  ],
  "stream": true,
  "temperature": 0.7
}
```

#### Model Information

```
GET /v1/models
```

Returns loaded models and their capabilities:

```json
{
  "data": [
    {
      "id": "sharpinference-qwen3-30b-a3b",
      "object": "model",
      "owned_by": "local",
      "capabilities": {
        "max_context": 65536,
        "quantization": "Q4_K_M",
        "kv_cache": "turboquant-3bit",
        "speculative_decoding": true,
        "draft_model": "smollm2-1.7b"
      }
    }
  ]
}
```

#### Health and Metrics

```
GET /health              # liveness check
GET /metrics             # Prometheus-compatible metrics
```

Metrics include: tokens/second, VRAM usage, RAM usage, expert cache hit rate, TurboQuant compression ratio, queue depth, active requests.

### 9.3 Server Architecture

```csharp
public static class ServerProgram
{
    public static async Task Main(string[] args)
    {
        var builder = WebApplication.CreateSlimBuilder(args);

        // JSON source generation for NativeAOT compatibility
        builder.Services.ConfigureHttpJsonOptions(options =>
            options.SerializerOptions.TypeInfoResolverChain.Add(ApiJsonContext.Default));

        // Core services
        builder.Services.AddSingleton<InferenceEngine>();
        builder.Services.AddSingleton<ModelRegistry>();
        builder.Services.AddSingleton<TokenStreamingService>();

        var app = builder.Build();

        // Anthropic Messages API
        app.MapPost("/v1/messages", HandleAnthropicMessages);

        // OpenAI Chat Completions API
        app.MapPost("/v1/chat/completions", HandleOpenAiChatCompletions);

        // Model info
        app.MapGet("/v1/models", HandleListModels);

        // Health
        app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

        await app.RunAsync();
    }
}
```

### 9.4 Streaming Implementation

Token streaming is implemented using `IAsyncEnumerable<T>` from the inference engine, converted to SSE at the HTTP layer:

```csharp
public static class AnthropicHandler
{
    public static async Task HandleAnthropicMessages(
        HttpContext context,
        InferenceEngine engine,
        MessagesRequest request)
    {
        if (!request.Stream)
        {
            // Non-streaming: collect all tokens, return complete response
            var response = await engine.GenerateCompleteAsync(request);
            await context.Response.WriteAsJsonAsync(response, ApiJsonContext.Default.MessagesResponse);
            return;
        }

        // Streaming: SSE
        context.Response.ContentType = "text/event-stream";
        context.Response.Headers.CacheControl = "no-cache";

        var writer = new SseWriter(context.Response);
        var messageId = IdGenerator.NewMessageId();

        // message_start
        await writer.WriteEventAsync("message_start", new MessageStartEvent(messageId, request.Model));

        // content_block_start
        await writer.WriteEventAsync("content_block_start", new ContentBlockStartEvent(0));

        // Stream tokens
        int outputTokens = 0;
        string? stopReason = null;

        await foreach (var token in engine.GenerateStreamAsync(request, context.RequestAborted))
        {
            if (token.IsStop)
            {
                stopReason = token.StopReason;  // "end_turn" or "stop_sequence"
                break;
            }

            await writer.WriteEventAsync("content_block_delta",
                new ContentBlockDeltaEvent(0, new TextDelta(token.Text)));
            outputTokens++;
        }

        // content_block_stop
        await writer.WriteEventAsync("content_block_stop", new ContentBlockStopEvent(0));

        // message_delta
        await writer.WriteEventAsync("message_delta",
            new MessageDeltaEvent(stopReason ?? "end_turn", outputTokens));

        // message_stop
        await writer.WriteEventAsync("message_stop", new MessageStopEvent());
    }
}
```

### 9.5 Request/Response Models

All models use `System.Text.Json` source generation for NativeAOT compatibility:

```csharp
// Anthropic Messages API request
public sealed record MessagesRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("max_tokens")] int MaxTokens,
    [property: JsonPropertyName("messages")] IReadOnlyList<Message> Messages,
    [property: JsonPropertyName("stream")] bool Stream = false,
    [property: JsonPropertyName("temperature")] float? Temperature = null,
    [property: JsonPropertyName("top_p")] float? TopP = null,
    [property: JsonPropertyName("top_k")] int? TopK = null,
    [property: JsonPropertyName("stop_sequences")] IReadOnlyList<string>? StopSequences = null,
    [property: JsonPropertyName("system")] string? System = null);

public sealed record Message(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] JsonElement Content);  // string or array

// Source-generated JSON context
[JsonSerializable(typeof(MessagesRequest))]
[JsonSerializable(typeof(MessagesResponse))]
[JsonSerializable(typeof(MessageStartEvent))]
[JsonSerializable(typeof(ContentBlockDeltaEvent))]
[JsonSerializable(typeof(OpenAiChatRequest))]
[JsonSerializable(typeof(OpenAiChatResponse))]
[JsonSerializable(typeof(ModelListResponse))]
public partial class ApiJsonContext : JsonSerializerContext { }
```

### 9.6 Configuration

```json
{
  "SharpInference": {
    "Server": {
      "Host": "0.0.0.0",
      "Port": 8080
    },
    "Models": [
      {
        "Id": "qwen3-30b-a3b",
        "Path": "/models/Qwen3-30B-A3B-Q4_K_M.gguf",
        "Backend": "auto",
        "KvCacheType": "turboquant-3bit",
        "MaxContext": 65536
      }
    ],
    "SpeculativeDecoding": {
      "Enabled": true,
      "DraftModel": "/models/SmolLM2-1.7B-Q4_K_M.gguf",
      "CandidateCount": 4
    },
    "Hardware": {
      "VramReserveMb": 512,
      "PinnedRamMaxMb": 49152,
      "ExpertCacheSlots": 48,
      "EnableIoUring": true
    }
  }
}
```

---

## 10. Project Structure

```
SharpInference/
├── src/
│   ├── SharpInference.Core/
│   │   ├── Gguf/
│   │   │   ├── GgufModel.cs
│   │   │   ├── GgufMetadata.cs
│   │   │   ├── GgufTensorInfo.cs
│   │   │   └── GgufTypes.cs
│   │   ├── Tokenizer/
│   │   │   ├── ITokenizer.cs
│   │   │   └── MlTokenizerAdapter.cs
│   │   ├── Tensors/
│   │   │   ├── CpuTensor.cs
│   │   │   ├── GpuTensor.cs
│   │   │   ├── TensorFormat.cs
│   │   │   └── NativeMemoryPool.cs
│   │   ├── Model/
│   │   │   ├── ModelGraph.cs
│   │   │   ├── ModelArchitecture.cs
│   │   │   ├── LayerWeights.cs
│   │   │   └── ModelLoader.cs
│   │   └── IComputeBackend.cs
│   │
│   ├── SharpInference.Cpu/
│   │   ├── CpuBackend.cs
│   │   ├── Simd/
│   │   │   ├── MatVecAvx2.cs
│   │   │   ├── MatVecAvx512.cs
│   │   │   ├── DequantQ4K.cs
│   │   │   ├── HadamardAvx2.cs
│   │   │   └── SimdHelper.cs
│   │   └── Reference/
│   │       └── ScalarOps.cs
│   │
│   ├── SharpInference.Vulkan/
│   │   ├── VulkanBackend.cs
│   │   ├── VulkanDevice.cs
│   │   ├── VulkanBufferPool.cs
│   │   ├── PipelineCache.cs
│   │   ├── CommandScheduler.cs
│   │   └── DescriptorManager.cs
│   │
│   ├── SharpInference.TurboQuant/
│   │   ├── TurboQuant.cs              # Core quantize/dequant
│   │   ├── TurboQuantCodebooks.cs     # Precomputed Lloyd-Max tables
│   │   ├── WalshHadamard.cs           # WHT transform (scalar + SIMD)
│   │   ├── BitPacking.cs              # 3-bit / 4-bit pack/unpack
│   │   ├── TurboQuantKvCache.cs       # Adaptive KV cache manager
│   │   └── MagnitudeProfiler.cs       # K/V ratio analysis per model
│   │
│   ├── SharpInference.Pipeline/
│   │   ├── TierPlacementPlanner.cs
│   │   ├── ExpertSlotCache.cs
│   │   ├── PrefetchPipeline.cs
│   │   ├── DoubleBufferedLayerStreamer.cs
│   │   ├── PinnedMemoryPool.cs
│   │   └── IoUring/
│   │       ├── IoUringReader.cs
│   │       └── IoUringInterop.cs
│   │
│   ├── SharpInference.Engine/
│   │   ├── InferenceEngine.cs         # Top-level generate API
│   │   ├── DenseForwardPass.cs
│   │   ├── MoeForwardPass.cs
│   │   ├── SpeculativeDecoder.cs
│   │   ├── Sampling/
│   │   │   ├── ISampler.cs
│   │   │   ├── TemperatureSampler.cs
│   │   │   ├── TopKTopPSampler.cs
│   │   │   └── RepetitionPenalty.cs
│   │   └── Streaming/
│   │       └── TokenStreamingService.cs
│   │
│   ├── SharpInference.Server/
│   │   ├── Program.cs                 # ASP.NET Core Minimal API entry
│   │   ├── Handlers/
│   │   │   ├── AnthropicHandler.cs    # POST /v1/messages
│   │   │   ├── OpenAiHandler.cs       # POST /v1/chat/completions
│   │   │   └── ModelsHandler.cs       # GET /v1/models
│   │   ├── Models/
│   │   │   ├── Anthropic/
│   │   │   │   ├── MessagesRequest.cs
│   │   │   │   ├── MessagesResponse.cs
│   │   │   │   └── StreamEvents.cs
│   │   │   ├── OpenAi/
│   │   │   │   ├── ChatRequest.cs
│   │   │   │   └── ChatResponse.cs
│   │   │   └── ApiJsonContext.cs       # Source-generated JSON
│   │   ├── Middleware/
│   │   │   ├── RequestLogging.cs
│   │   │   └── ErrorHandling.cs
│   │   ├── Sse/
│   │   │   └── SseWriter.cs
│   │   └── Configuration/
│   │       └── ServerConfig.cs
│   │
│   └── SharpInference.Cli/
│       └── Program.cs                 # Interactive chat REPL + bench runner
│
├── shaders/
│   ├── matmul_f16.comp
│   ├── matmul_dequant_q4k.comp
│   ├── rmsnorm.comp
│   ├── rope.comp
│   ├── softmax.comp
│   ├── silu.comp
│   ├── attention.comp
│   ├── tq_quantize.comp
│   └── tq_dequant_dot.comp
│
├── codebooks/
│   ├── lloyd_max_3bit_d128.json
│   ├── lloyd_max_3bit_d256.json
│   ├── lloyd_max_4bit_d128.json
│   └── lloyd_max_4bit_d256.json
│
├── tests/
│   ├── SharpInference.Tests.Core/
│   │   ├── GgufParserTests.cs
│   │   └── TokenizerTests.cs
│   ├── SharpInference.Tests.ForwardPass/
│   │   ├── SmolLm2ReferenceTests.cs    # logit comparison vs llama.cpp
│   │   └── Qwen3ReferenceTests.cs
│   ├── SharpInference.Tests.TurboQuant/
│   │   ├── CodebookTests.cs
│   │   ├── HadamardTests.cs
│   │   ├── QuantizeRoundtripTests.cs
│   │   ├── MseValidationTests.cs       # MSE matches paper ±1%
│   │   └── DequantDotTests.cs
│   ├── SharpInference.Tests.Pipeline/
│   │   ├── ExpertCacheTests.cs
│   │   ├── PrefetchPipelineTests.cs
│   │   └── TierPlacementTests.cs
│   └── SharpInference.Tests.Server/
│       ├── AnthropicApiTests.cs
│       ├── OpenAiApiTests.cs
│       └── SseStreamingTests.cs
│
├── benchmarks/
│   └── SharpInference.Benchmarks/
│       ├── MatVecBenchmark.cs
│       ├── TurboQuantBenchmark.cs
│       ├── ForwardPassBenchmark.cs
│       └── E2EInferenceBenchmark.cs
│
├── docs/
│   └── ARCHITECTURE.md                 # This document
│
├── Directory.Build.props               # Shared build properties, AOT analyzers
├── SharpInference.sln
└── README.md
```

---

## 11. Build Configuration

### 11.1 Shared Build Properties

```xml
<!-- Directory.Build.props -->
<Project>
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <LangVersion>14</LangVersion>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <InvariantGlobalization>true</InvariantGlobalization>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>

    <!-- NativeAOT readiness from day one -->
    <IsTrimmable>true</IsTrimmable>
    <EnableTrimAnalyzer>true</EnableTrimAnalyzer>
    <EnableAotAnalyzer>true</EnableAotAnalyzer>
    <SuppressTrimAnalysisWarnings>false</SuppressTrimAnalysisWarnings>

    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
  </PropertyGroup>
</Project>
```

### 11.2 Server Publish Profiles

```xml
<!-- Development: fast compile, JIT, full debugging -->
<PropertyGroup>
  <PublishReadyToRun>true</PublishReadyToRun>
  <SelfContained>true</SelfContained>
</PropertyGroup>
```

```xml
<!-- Release: NativeAOT, single binary, max performance -->
<PropertyGroup>
  <PublishAot>true</PublishAot>
  <OptimizationPreference>Speed</OptimizationPreference>
  <IlcInstructionSet>native</IlcInstructionSet>
  <StripSymbols>true</StripSymbols>
  <SelfContained>true</SelfContained>
</PropertyGroup>
```

### 11.3 Shader Compilation

GLSL compute shaders are compiled to SPIR-V at build time via an MSBuild target:

```xml
<Target Name="CompileShaders" BeforeTargets="Build">
  <Exec Command="glslangValidator -V %(ShaderFiles.Identity) -o %(ShaderFiles.Identity).spv"
        WorkingDirectory="$(SolutionDir)shaders/" />
</Target>
```

SPIR-V bytecode is embedded as assembly resources and loaded at runtime.

### 11.4 Dependencies

| Package | Purpose |
|---------|---------|
| `Vortice.Vulkan` | GPU compute dispatch (zero-dependency, includes VMA bindings) |
| `Microsoft.ML.Tokenizers` | BPE / SentencePiece tokenization |
| `System.IO.Pipelines` | Async data flow primitives |
| `BenchmarkDotNet` | Performance measurement |
| `Microsoft.Extensions.Logging` | Structured logging |
| `Microsoft.AspNetCore.App` | HTTP server (Minimal API) |

---

## 12. Implementation Phases

### Phase 1: CPU Reference Implementation ✅

**Goal:** Correct output, then maximum CPU performance.

- [x] GGUF parser with memory-mapped tensor access (zero-copy via `MemoryMappedFile`)
- [x] Tokenizer integration via `Microsoft.ML.Tokenizers` (BPE with special token handling)
- [x] Q4_K and Q6_K scalar dequantization matching ggml-quants.c
- [x] Full LLaMA-family forward pass: GQA attention, SwiGLU FFN, interleaved RoPE
- [x] FP32 KV cache with per-layer buffers
- [x] Temperature / top-k / top-p / min-p sampling
- [x] AVX2 SIMD: fused dequant-matvec with multi-accumulator FMA chains
- [x] Multi-threaded MatVec via `Parallel.For` (24 threads)
- [x] Batched prefill with layer-by-layer cache-hot weight reuse
- [x] OpenBLAS GEMM integration for large-batch prefill
- [x] CLI chat REPL (`sharpi-cli`) with llama.cpp-compatible flags
- [x] Reference test: output matches llama.cpp greedy decode token-for-token
- [x] BenchmarkDotNet harness measuring decode and prefill throughput

**Results:** 48.6 t/s decode (matches llama.cpp 45.1 t/s on same hardware). 85 tests passing.

**Target model:** SmolLM2 1.7B (dense, Apache 2.0) ✅

### Phase 2: Vulkan GPU Acceleration ✅

**Goal:** Competitive single-GPU speed for VRAM-fitting models.

- [x] Vortice.Vulkan device initialization and dedicated compute queue
- [x] GPU buffer management: device-local VRAM, staging transfers, cached download buffers
- [x] Compute shader pipeline: GLSL→SPIR-V via glslc, descriptor sets, push constants
- [x] Compute shaders: MatVecQ4K, MatVecQ6K, MatVecF32, RMSNorm, RoPE, softmax, SiLU, attention, embedding lookup, KV append
- [x] Batched command buffer: all ~240 dispatches per token in one submission with memory barriers
- [x] FP32 KV cache in VRAM (per-layer, device-local)
- [x] GPU attention with shared-memory parallel reduction (no atomics, no PCIe round-trips)
- [x] All weights resident in VRAM (Q4_K raw, Q6_K raw, F32 norms)
- [x] Zero managed allocation per decode token
- [x] GPU forward pass validated against CPU token-for-token
- [x] NativeAOT-ready CLI with IlcOptimizationPreference=Speed, IlcInstructionSet=native

**Results:** 87.4 t/s decode on RTX 4070 Ti (1.80× faster than CPU, 250% of ≥35 t/s target).
Optimized from initial 68.7 t/s (+28%) via shared-memory block caching in Q4_K/Q6_K shaders,
atomic-free attention reduction, descriptor set caching, fence-based sync, and staging buffer reuse.

**Target model:** SmolLM2 1.7B ✅ (Qwen3 8B scaling — Phase 2b)

### Phase 3: TurboQuant KV Cache Compression (Weeks 7–10)

**Goal:** 4–6x KV cache reduction, enabling 64K+ context on 12GB VRAM.

- [ ] Lloyd-Max codebook generation (offline script, embed as constants)
- [ ] Walsh-Hadamard transform (scalar reference + AVX2 SIMD)
- [ ] 3-bit quantize / dequant with bit packing
- [ ] Fused dequant-dot-product (CPU and Vulkan shader)
- [ ] Adaptive precision: FP16 recent window + TQ compressed history
- [ ] K/V magnitude profiler for per-model bit budget selection
- [ ] MSE validation tests matching paper within 1%
- [ ] Needle-in-a-haystack test at 8K / 16K / 32K / 64K

**Target model:** Qwen3 8B with 64K context

### Phase 4: Hybrid GPU/CPU Offloading (Weeks 11–14)

**Goal:** Run dense models larger than VRAM.

- [ ] Tier placement planner (auto-assign tensors to VRAM / RAM / NVMe)
- [ ] Pinned memory pool (`cuMemAllocHost` or Vulkan host-visible)
- [ ] Double-buffered layer streaming (DMA overlapped with compute)
- [ ] CPU backend with AVX2/AVX-512 SIMD matmul for overflow layers
- [ ] `io_uring` interop for async NVMe reads (Linux)
- [ ] Profiling: measure PCIe utilization, GPU stall time

**Target model:** Llama 3.1 70B at Q4_K_M

### Phase 5: MoE-Aware Inference with Prefetching (Weeks 15–20)

**Goal:** Fast MoE inference exploiting routing sparsity.

- [ ] MoE router computation on GPU
- [ ] Expert slot cache in VRAM with SLRU eviction
- [ ] `Channel<T>` async prefetch pipeline
- [ ] Router-driven predictive prefetching (1-token lookahead)
- [ ] CPU fallback for expert cache misses (PCIe round-trip activation)
- [ ] NVMe cold expert promotion path
- [ ] Expert access frequency profiling and cache hit rate metrics

**Target model:** Qwen3 30B-A3B → GPT-OSS 120B

### Phase 6: Speculative Decoding (Weeks 21–23)

**Goal:** 2–3x throughput improvement via draft-verify pipeline.

- [ ] Draft model co-loaded in VRAM alongside target model
- [ ] Speculative candidate generation
- [ ] Batched verification pass
- [ ] Accept/reject with proper probability adjustment
- [ ] Adaptive candidate count based on acceptance rate

**Draft model:** SmolLM2 1.7B

### Phase 7: API Server (Weeks 24–26)

**Goal:** Drop-in Anthropic and OpenAI API compatible server.

- [ ] ASP.NET Core Minimal API with NativeAOT compatibility
- [ ] Anthropic Messages API: `POST /v1/messages` (streaming + non-streaming)
- [ ] OpenAI Chat Completions API: `POST /v1/chat/completions`
- [ ] SSE streaming with proper event types
- [ ] Model listing: `GET /v1/models`
- [ ] Health and Prometheus metrics endpoints
- [ ] Source-generated JSON serialization throughout
- [ ] Integration tests validating wire format compatibility
- [ ] Configuration via `appsettings.json` and CLI arguments

---

## 13. Validation Strategy

### 13.1 Correctness

Every phase validates against llama.cpp as the reference implementation:

1. **Logit comparison.** Run identical prompt through both engines at temperature 0. Compare raw logit vectors. Tolerance: max absolute difference < 0.01 for FP16, < 0.1 for Q4.
2. **Output identity.** For greedy decoding (temperature 0), both engines must produce identical token sequences for the first 100 tokens.
3. **TurboQuant MSE.** Quantize-dequant round-trip MSE must match paper values within 1% for random unit vectors at d=128 and d=256.
4. **Needle-in-a-haystack.** Insert a unique fact at various positions in a long context. Model must retrieve it correctly at 8K, 16K, 32K, and 64K with TurboQuant enabled.

### 13.2 Reference Benchmarks (llama.cpp baseline)

These are community-reported llama.cpp benchmarks on hardware comparable to our target profile. They establish the performance floor SharpInference must reach and the ceiling it aims to approach.

All benchmarks use llama.cpp with CUDA backend unless noted. "PP" = prompt processing (prefill), "TG" = token generation (decode). Context length is noted where available.

#### RTX 3060 12GB — Models Fully in VRAM

| Model | Quant | Context | PP (t/s) | TG (t/s) | Source |
|-------|-------|---------|----------|----------|--------|
| Llama 2 7B | Q4_0 | 512 | ~1,490 | ~52 | localscore.ai, llama.cpp #15013 |
| Llama 3.1 8B | Q4_K_M | 4K | ~1,490 | ~38–52 | localscore.ai, practicalwebtools |
| Qwen3 8B | Q4_K_M | 8K | — | ~40+ | localllm.in (Ollama benchmark) |

Key observation: on the RTX 3060 12GB, 8B-class dense models at Q4_K_M produce approximately 38–52 tokens/second for generation and ~1,500 t/s for prompt processing when fully resident in VRAM. This is the primary benchmark SharpInference Phase 2 must approach.

#### RTX 3060 12GB — MoE Models with CPU Offload

| Model | Quant | Context | Config | TG (t/s) | Source |
|-------|-------|---------|--------|----------|--------|
| GPT-OSS 20B (MoE) | MXFP4 | 32K | -ngl 99, -ncmoe 2 | ~60 | llama.cpp #15396 (Ryzen 7 5700X, 32GB DDR4) |
| GPT-OSS 120B (MoE) | Q4_K_XL | 16K | -ncmoe 32 | ~12 | llama.cpp #15396 (RTX 3060, 32GB RAM, barely fit) |
| Qwen3-Coder 30B-A3B | Q6 | 8K | MoE offload | ~12 | arsturn.com (user reports, 12GB GPU) |

Key observation: MoE models with CPU expert offloading on a 12GB GPU + 32–64GB RAM achieve 12–60 t/s depending on model size and how many experts are offloaded. The GPT-OSS 20B result of 60 t/s on an RTX 3060 is particularly relevant — it's a MoE model with CPU-side expert computation over PCIe.

#### RTX 3060 12GB — Dense Models with Hybrid CPU/GPU Split

| Model | Quant | Context | Config | TG (t/s) | Source |
|-------|-------|---------|--------|----------|--------|
| Qwen3 8B | Q4_K_M | 8K | 25 of 36 layers in VRAM | ~8 | localllm.in (partial offload) |
| Llama 3.3 70B | Q4_K_M | 4K | 50–60% GPU offload | ~3–5 | willitrunai.com |

Key observation: partial GPU offloading for dense models causes severe performance degradation (40 t/s → 8 t/s when just 11 layers spill to RAM). This is the bottleneck SharpInference's pipelined double-buffered layer streaming (Phase 4) aims to improve.

#### Comparison Point: RTX 3090 24GB (Full VRAM)

| Model | Quant | Context | TG (t/s) | Source |
|-------|-------|---------|----------|--------|
| Qwen3 30B-A3B (MoE) | Q4_K | 32K | ~87 | hardware-corner.net |
| GPT-OSS 20B (MoE) | MXFP4 | 32K | ~75–128 | llama.cpp #15396 |

These represent what's possible if you could fit everything in VRAM — useful as an upper bound target for our caching and prefetching optimizations.

#### Comparison Point: Apple Silicon (Unified Memory)

| Model | Hardware | Quant | TG (t/s) | Source |
|-------|----------|-------|----------|--------|
| Llama 3.1 8B | M3 Pro 18GB | Q4_K_M | ~15–28 | localaimaster.com |
| Qwen3.5-397B (MoE) | M3 Max 48GB | 2-bit experts | ~5.5 | Dan Woods / flash-moe |
| Qwen3-Coder 30B-A3B | M4 Max | Q4 | ~100+ | arsturn.com (user reports) |

These demonstrate the advantage of unified memory for large models. Our pipelined offloading strategy aims to narrow this gap on discrete GPU hardware.

### 13.3 SharpInference Performance Targets

Based on the reference benchmarks above, these are concrete targets per phase:

| Phase | Model | Configuration | llama.cpp Baseline | SharpInference Target | Actual | Notes |
|-------|-------|---------------|-------------------|----------------------|--------|-------|
| 1 | SmolLM2 1.7B Q4_K_M | CPU only | 45.1 TG t/s | Match llama.cpp | **48.6 TG t/s** ✅ | AVX2 SIMD, fused dequant-matvec |
| 2 | SmolLM2 1.7B Q4_K_M | Full VRAM, RTX 4070 Ti | ~40–52 TG t/s | ≥ 35 TG t/s (≥80%) | **87.4 TG t/s** ✅ | Vulkan compute shaders, 250% of target |
| 3 | Qwen3 8B Q4_K_M + TQ3 | Full VRAM, 64K ctx | N/A (doesn't fit with FP16 KV) | ≥ 30 TG t/s | TurboQuant enables what llama.cpp can't do at this context on 12GB |
| 4 | Llama 3.1 70B Q4_K_M | Hybrid GPU+CPU | ~3–5 TG t/s (naive offload) | ≥ 5 TG t/s | Pipelined streaming should match or beat naive split |
| 5 | Qwen3 30B-A3B Q4_K_M | MoE offload, 12GB + 64GB RAM | ~12 TG t/s (estimated) | ≥ 15 TG t/s | Prefetch + expert cache should beat naive offload |
| 5+6 | Qwen3 30B-A3B + speculative | MoE + SmolLM2 draft | ~12 TG t/s (no spec) | ≥ 25 effective TG t/s | ~2x from speculative decoding on top of Phase 5 |

**Stretch targets (if all optimizations compose well):**

| Scenario | Optimistic Target | Rationale |
|----------|------------------|-----------|
| GPT-OSS 120B MoE, 64GB RAM | ≥ 8 TG t/s | Expert caching + prefetch + 64GB RAM vs 32GB in reference |
| Qwen3 8B, 128K context + TQ3 | ≥ 20 TG t/s | TurboQuant keeps KV cache in VRAM; llama.cpp Q4 KV can't match here |

### 13.4 Benchmark Methodology

All SharpInference benchmarks will use:

- **Model:** as specified per phase target
- **Prompt:** standardized 512-token input (matching llama-bench pp512 convention)
- **Generation:** 128 tokens output (matching llama-bench tg128 convention)
- **Context:** as specified, with KV cache warm from prompt
- **Measurement:** median of 5 runs after 1 warmup run
- **Tool:** `BenchmarkDotNet` for micro-benchmarks, wall-clock for end-to-end
- **Comparison:** llama.cpp run on identical hardware with equivalent settings, same GGUF file

Results will be reported as:

```
SharpInference v0.X.0 — Qwen3 8B Q4_K_M, RTX 3060 12GB, 64GB DDR4
PP512:  XXXX t/s (llama.cpp: YYYY t/s, ratio: 0.XX)
TG128:  XX.X t/s (llama.cpp: YY.Y t/s, ratio: 0.XX)
VRAM:   X.XX GB  (llama.cpp: Y.YY GB)
```

---

## 14. Future Directions

These are explicitly out of scope for the initial implementation but are noted as potential extensions:

- **AMD ROCm / RDNA support** via Vulkan compute (should largely work, needs testing).
- **Multi-GPU** via Vulkan device groups or explicit multi-device management.
- **LoRA adapter hot-loading** for serving multiple fine-tunes from a single base model.
- **Continuous batching** for multi-user serving (requires PagedAttention-style KV cache management).
- **Weight quantization with TurboQuant** in addition to KV cache (the `turboquant-model` project demonstrates this path).
- **Apple Silicon / Metal backend** via MoltenVK or a native Metal compute backend.
- **Tool use / function calling** in the API server layer.
- **OpenAI Responses API** compatibility.

---

## 15. References

1. Zandieh et al. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." ICLR 2026. arXiv:2504.19874.
2. Alizadeh et al. "LLM in a flash: Efficient Large Language Model Inference with Limited Memory." ACL 2024. arXiv:2312.11514.
3. Karpathy, A. "llm.c" — minimal C implementation of GPT-2 training/inference. github.com/karpathy/llm.c.
4. llama.cpp — reference C++ LLM inference engine. github.com/ggml-org/llama.cpp.
5. Doctor Shotgun. "Performant local mixture-of-experts CPU inference with GPU acceleration in llama.cpp." HuggingFace Blog, February 2026.
6. llama.cpp Discussion #20969: "TurboQuant — Extreme KV Cache Quantization." Community implementation notes and K/V magnitude findings.
7. Dan Woods. "Autoresearching Apple's LLM in a Flash to run Qwen 397B locally." github.com/danveloper/flash-moe.
8. GGUF specification. HuggingFace documentation.
