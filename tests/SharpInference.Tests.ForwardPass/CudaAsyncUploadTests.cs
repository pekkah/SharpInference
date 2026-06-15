using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Tests for the issue #78 async CUDA upload stream: dedicated upload stream,
/// event-based readiness, and cross-stream ordering when a consumer kernel
/// reads a prefetched tensor.
///
/// Silently skips on hosts without CUDA, same pattern as the other Cuda* tests.
/// </summary>
public sealed class CudaAsyncUploadTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    /// <summary>
    /// Smoke-test the event signal/wait semantics: UploadBackground returns a
    /// handle with a real CUDA event; after a stream sync the event must be
    /// signaled (IsUploadComplete returns true); the data downloaded from the
    /// device matches the source bytes.
    /// </summary>
    [Fact]
    public void UploadBackground_EventSignalsAfterStreamSync()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int N = 1024;
        var data = new float[N];
        for (int i = 0; i < N; i++) data[i] = i * 0.5f - 100f;

        var pending = gpu.UploadBackground(data, TensorShape.D1(N));

        // Cross-stream fence: make the compute stream wait for the upload event.
        gpu.WaitForUpload(pending);
        // Reading back forces a full stream sync — once that returns, the event
        // must have signaled (the compute stream waited on it, the wait drained).
        var roundTrip = new float[N];
        gpu.Download(pending.Tensor, roundTrip);

        Assert.True(gpu.IsUploadComplete(pending), "Upload event should be signaled after stream sync.");
        for (int i = 0; i < N; i++)
            Assert.Equal(data[i], roundTrip[i]);

        gpu.ReleaseUploadHandle(pending);
        gpu.Free(pending.Tensor);
    }

    /// <summary>
    /// Several concurrent UploadBackground calls each get distinct events that
    /// all signal independently. Verifies the per-call event tracking model:
    /// destroying one handle's event doesn't affect another's.
    /// </summary>
    [Fact]
    public void UploadBackground_MultipleHandles_AllSignal()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int N = 512;
        const int Batch = 5;
        var handles = new CudaUploadHandle[Batch];
        var sources = new float[Batch][];
        for (int b = 0; b < Batch; b++)
        {
            sources[b] = new float[N];
            for (int i = 0; i < N; i++) sources[b][i] = b * 100f + i;
            handles[b] = gpu.UploadBackground(sources[b], TensorShape.D1(N));
            Assert.NotEqual(nint.Zero, handles[b].UploadEvent);
        }

        // Distinct events: no two handles share the same cudaEvent_t.
        for (int i = 0; i < Batch; i++)
            for (int j = i + 1; j < Batch; j++)
                Assert.NotEqual(handles[i].UploadEvent, handles[j].UploadEvent);

        // Fence all uploads behind the compute stream, then read each back.
        for (int b = 0; b < Batch; b++) gpu.WaitForUpload(handles[b]);
        for (int b = 0; b < Batch; b++)
        {
            var rt = new float[N];
            gpu.Download(handles[b].Tensor, rt);
            for (int i = 0; i < N; i++) Assert.Equal(sources[b][i], rt[i]);
            Assert.True(gpu.IsUploadComplete(handles[b]));
        }

        for (int b = 0; b < Batch; b++)
        {
            gpu.ReleaseUploadHandle(handles[b]);
            gpu.Free(handles[b].Tensor);
        }
    }

    /// <summary>
    /// Cross-stream ordering: a consumer kernel (MatMul reading the prefetched
    /// weights) on the compute stream must produce the same output as if the
    /// weights had been uploaded synchronously. Without the upload-event ->
    /// compute-stream fence, the MatMul could read pre-DMA garbage.
    /// </summary>
    [Fact]
    public void UploadBackground_MatMul_ProducesIdenticalOutputToSyncPath()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int Rows = 64;
        const int Cols = 512;
        var rng = new Random(20260530);
        var weights = new float[Rows * Cols];
        var input   = new float[Cols];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < Cols; i++)            input[i]   = (float)(rng.NextDouble() * 2 - 1);

        // Sync reference path.
        var wSync = gpu.Upload(weights, TensorShape.D1(weights.Length));
        var xSync = gpu.Upload(input, TensorShape.D1(Cols));
        var ySync = gpu.Allocate(TensorShape.D1(Rows));
        gpu.MatMul(ySync, wSync, xSync, DType.Float32);
        var refOut = new float[Rows];
        gpu.Download(ySync, refOut);

        // Async upload path: UploadBackground -> WaitForUpload -> MatMul.
        var wAsync = gpu.UploadBackground(weights, TensorShape.D1(weights.Length));
        gpu.WaitForUpload(wAsync);
        var xAsync = gpu.Upload(input, TensorShape.D1(Cols));
        var yAsync = gpu.Allocate(TensorShape.D1(Rows));
        gpu.MatMul(yAsync, wAsync.Tensor, xAsync, DType.Float32);
        var asyncOut = new float[Rows];
        gpu.Download(yAsync, asyncOut);

        for (int r = 0; r < Rows; r++)
            Assert.Equal(refOut[r], asyncOut[r]);

        gpu.ReleaseUploadHandle(wAsync);
        gpu.Free(wAsync.Tensor); gpu.Free(xAsync); gpu.Free(yAsync);
        gpu.Free(wSync); gpu.Free(xSync); gpu.Free(ySync);
    }

    /// <summary>
    /// UploadBackgroundRaw byte path: bytes uploaded async match a sync DownloadRaw,
    /// confirming the dtype tagging and raw-byte staging are wired correctly.
    /// </summary>
    [Fact]
    public void UploadBackgroundRaw_ProducesIdenticalBytesToSyncPath()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int Bytes = 4096;
        var rng = new Random(20260530);
        var src = new byte[Bytes];
        rng.NextBytes(src);

        // Upload as F32 so we can compare via the float Download path (Bytes/4 floats).
        var pending = gpu.UploadBackgroundRaw(src, TensorShape.D1(Bytes / 4), DType.Float32);
        gpu.WaitForUpload(pending);

        var rt = new float[Bytes / 4];
        gpu.Download(pending.Tensor, rt);

        var asBytes = new byte[Bytes];
        Buffer.BlockCopy(rt, 0, asBytes, 0, Bytes);
        for (int i = 0; i < Bytes; i++) Assert.Equal(src[i], asBytes[i]);

        gpu.ReleaseUploadHandle(pending);
        gpu.Free(pending.Tensor);
    }

    /// <summary>
    /// Issue #217 staging ring: issuing more uploads than the ring's slot count forces
    /// slot reuse — the drain-then-re-record of a backend-owned fence + staging-buffer reuse
    /// the ring exists to make safe. Every upload must still round-trip its own distinct
    /// payload; a stale staging slot, a wrong-fence drain, or an off-by-one in the slot index
    /// would corrupt the tensors uploaded after the first wrap. (The other tests issue ≤5
    /// uploads and never wrap, so this is the only coverage of the reuse path.)
    /// </summary>
    [Fact]
    public void UploadBackground_RingWrap_AllPayloadsRoundTrip()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int N = 64;    // > the 32-slot ring → at least one full wrap (each slot reused)
        const int Len = 257; // distinct, multi-element payload per upload
        var handles = new CudaUploadHandle[N];
        var sources = new float[N][];
        for (int u = 0; u < N; u++)
        {
            var s = new float[Len];
            for (int i = 0; i < Len; i++) s[i] = u * 1000f + i; // unique per (upload, index)
            sources[u] = s;
            handles[u] = gpu.UploadBackground(s, TensorShape.D1(Len));
        }

        for (int u = 0; u < N; u++)
        {
            gpu.WaitForUpload(handles[u]);
            var rt = new float[Len];
            gpu.Download(handles[u].Tensor, rt);
            for (int i = 0; i < Len; i++)
                Assert.Equal(sources[u][i], rt[i]);
        }

        for (int u = 0; u < N; u++)
        {
            gpu.ReleaseUploadHandle(handles[u]);
            gpu.Free(handles[u].Tensor);
        }
    }

    /// <summary>
    /// Backend.UploadStream is nint.Zero until the first background upload, then
    /// becomes non-zero and is reused across subsequent calls. Validates the
    /// lazy-init contract documented on the property.
    /// </summary>
    [Fact]
    public void UploadStream_LazyInitOnFirstBackgroundUpload()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        Assert.Equal(nint.Zero, gpu.UploadStream);

        var data = new float[16];
        var first = gpu.UploadBackground(data, TensorShape.D1(16));
        Assert.NotEqual(nint.Zero, gpu.UploadStream);
        nint streamAfterFirst = gpu.UploadStream;

        var second = gpu.UploadBackground(data, TensorShape.D1(16));
        Assert.Equal(streamAfterFirst, gpu.UploadStream);

        gpu.WaitForUpload(first); gpu.WaitForUpload(second);
        gpu.ReleaseUploadHandle(first); gpu.ReleaseUploadHandle(second);
        gpu.Free(first.Tensor); gpu.Free(second.Tensor);
    }

    /// <summary>
    /// End-to-end: CudaExpertSlotManager.Preload populates the cache via the
    /// async path; a subsequent GetOrLoad observes the slot as resident
    /// (single profiler hit, no extra miss) and the resulting MatMul output
    /// matches the path where the same expert was loaded synchronously.
    /// Requires an MoE GGUF on disk; silently skipped otherwise.
    /// </summary>
    [Fact]
    public void CudaExpertSlotManager_PreloadThenGetOrLoad_HitsCache_AndOutputMatchesSync()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindFirstExisting("models\\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        if (!hp.IsMoE) return;

        // Two slot managers, same expert: one loaded sync via GetOrLoad, the
        // other preloaded async then accessed via GetOrLoad.
        var dtypesSync = new Dictionary<nint, DType>();
        using var syncSlots = new CudaExpertSlotManager(gpu, model, hp, slotCapacity: 4, dtypesSync);

        var dtypesAsync = new Dictionary<nint, DType>();
        using var asyncSlots = new CudaExpertSlotManager(gpu, model, hp, slotCapacity: 4, dtypesAsync);

        var syncSlot = syncSlots.GetOrLoad(layer: 0, expertId: 0);
        Assert.Equal(1, syncSlots.Profiler.TotalMisses);
        Assert.Equal(0, syncSlots.Profiler.TotalHits);

        // Async path: Preload (no profiler update) then GetOrLoad (hit).
        asyncSlots.Preload(layer: 0, expertId: 0);
        Assert.Equal(0, asyncSlots.Profiler.TotalMisses);
        Assert.Equal(0, asyncSlots.Profiler.TotalHits);

        var asyncSlot = asyncSlots.GetOrLoad(layer: 0, expertId: 0);
        Assert.Equal(0, asyncSlots.Profiler.TotalMisses);
        Assert.Equal(1, asyncSlots.Profiler.TotalHits);

        // Run MatMul against each expert's Gate weight with the same input and
        // compare the downloaded outputs element-by-element.
        int cols = hp.EmbeddingDim;
        int rows = hp.ExpertIntermediateDim;
        var rng = new Random(20260530);
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var gpuIn = gpu.Upload(input, TensorShape.D1(cols));

        var ySync  = gpu.Allocate(TensorShape.D1(rows));
        var yAsync = gpu.Allocate(TensorShape.D1(rows));
        gpu.MatMul(ySync,  syncSlot.Gate,  gpuIn);
        gpu.MatMul(yAsync, asyncSlot.Gate, gpuIn);
        var refOut   = new float[rows]; gpu.Download(ySync,  refOut);
        var asyncOut = new float[rows]; gpu.Download(yAsync, asyncOut);

        for (int r = 0; r < rows; r++)
            Assert.Equal(refOut[r], asyncOut[r]);

        gpu.Free(gpuIn); gpu.Free(ySync); gpu.Free(yAsync);
    }

    private static string? FindFirstExisting(params string[] candidates)
    {
        foreach (var c in candidates)
            if (Path.IsPathRooted(c) && File.Exists(c)) return c;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var c in candidates)
            {
                if (Path.IsPathRooted(c)) continue;
                var p = Path.Combine(dir, c);
                if (File.Exists(p)) return p;
            }
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
