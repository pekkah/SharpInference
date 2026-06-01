using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Phase-6 smoke tests for the Gemma 4 CUDA plumbing in <see cref="CudaForwardPass"/>.
/// These cover the inert (construction-only) layer:
/// <list type="bullet">
///   <item><b>PLE stays CPU-resident</b> — the per_layer_token_embd table is held by
///         CPU mmap reference and is NOT uploaded; the post-construction "free VRAM"
///         delta must be smaller than the PLE table's raw byte size.</item>
///   <item><b>Dispose handles KV aliasing without double-free</b> — Gemma 4's
///         <c>shared_kv_layers</c> tail aliases the source layer's K/V handles, and
///         Dispose must skip them or hit a CUDA AccessViolation.</item>
///   <item><b>Phase-8 guard fires</b> — Forward must throw NotImplementedException
///         since the per-layer SWA/global / dual-RoPE / PLE-gather wiring is Phase 8.</item>
/// </list>
///
/// All tests silently no-op when CUDA isn't available OR the Gemma 4 E4B Q8 GGUF
/// isn't on disk — mirroring the rest of the Cuda* test files. They are slow
/// (model load is ~8 GB through the GGUF mmap + ~3 GB of GPU weight uploads on the
/// first construction) but only run when the artefact is present.
/// </summary>
public sealed class Gemma4CudaPlumbingTests
{
    private const string ModelFile = "gemma-4-E4B-it-Q8_0.gguf";

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    /// <summary>
    /// Locate the Gemma 4 E4B Q8_0 GGUF. Returns null when not present so the
    /// caller can silent-skip. Search order matches the project convention
    /// (small models in <c>C:\p\sharpi\models</c>, large in <c>E:\models</c>).
    /// </summary>
    private static string? FindModelPath()
    {
        string[] absoluteCandidates =
        {
            $@"E:\models\{ModelFile}",
            $@"C:\p\sharpi\models\{ModelFile}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Reflection accessor for private fields on <see cref="CudaForwardPass"/>.
    /// Keeping the fields private (they're inert plumbing, not API) and using
    /// reflection here avoids spamming the public surface with test-only members.
    /// The trim/AOT analyzer suppression is safe because this is a test-only path
    /// — the production code never reflects on these fields.
    /// </summary>
    [UnconditionalSuppressMessage("Trimming", "IL2075",
        Justification = "Test-only reflection on a fixed type; not used in NativeAOT publishing.")]
    private static object? GetField(CudaForwardPass instance, string fieldName)
    {
        var fi = typeof(CudaForwardPass).GetField(fieldName,
            BindingFlags.Instance | BindingFlags.NonPublic);
        return fi?.GetValue(instance);
    }

    [Fact]
    public void Gemma4_CudaForwardPass_ConstructorLoadsPleAsCpuResident()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Pre-condition: the GGUF really is a PLE-bearing Gemma 4. If the loader
        // didn't flip these flags the rest of the assertion is meaningless.
        Assert.True(hp.HasPerLayerTokenEmbd,
            "Gemma 4 E4B GGUF should advertise PLE — ModelHyperparams.HasPerLayerTokenEmbd is false.");
        Assert.NotNull(hp.LayerHeadDim);
        Assert.NotNull(hp.KvSourceLayer);

        var pleInfo = model.FindTensor("per_layer_token_embd.weight");
        Assert.NotNull(pleInfo);
        long pleBytes = pleInfo!.Value.ByteSize;
        Assert.True(pleBytes > 1L << 30,
            $"PLE table should be > 1 GiB to make this test meaningful; was {pleBytes:N0} bytes.");

        // Sum total on-disk GGUF tensor bytes (mirrors EstimateGpuTensorBytes's
        // raw-upload assumption for Q8_0 / Q4_K / Q6_K / F32, which is correct for
        // every Gemma 4 tensor). This is the upper bound for what VRAM consumption
        // could be if the planner uploaded everything.
        long totalTensorBytes = 0;
        foreach (var t in model.Tensors) totalTensorBytes += t.ByteSize;

        // Capture free VRAM before constructing CudaForwardPass. After construction
        // the drop must NOT include the PLE bytes — that's the load-bearing
        // assertion: consumed << totalTensorBytes when PLE is excluded.
        long freeBefore = (long)gpu.FreeVramBytes;

        // Keep ctx tight so the matching KV-cache allocation doesn't dominate the
        // delta and mask a PLE upload. Even 512 positions × 42 layers × 1024 dims
        // × 4 bytes ≈ 88 MiB worst case — well under the PLE table size.
        int ctx = 512;

        using (var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: ctx))
        {
            // The Phase-6 PLE plumbing landed: the CPU-resident handles must exist.
            var cpuPleField = GetField(fwd, "_cpuPleTokenEmbed");
            Assert.NotNull(cpuPleField);

            // per_layer_model_proj is preloaded as a float[] for the Phase 8 hot path.
            var cpuProjField = GetField(fwd, "_cpuPerLayerModelProj");
            Assert.NotNull(cpuProjField);
            Assert.IsType<float[]>(cpuProjField);

            // Post-norm GPU arrays exist (small, GPU-resident is intentional).
            Assert.NotNull(GetField(fwd, "_wPostAttnNorm"));
            Assert.NotNull(GetField(fwd, "_wPostFfwNorm"));
            // Per-layer output scale array exists (CPU-side).
            Assert.NotNull(GetField(fwd, "_layerOutputScale"));

            long freeAfter = (long)gpu.FreeVramBytes;
            long consumed  = freeBefore - freeAfter;

            // The load-bearing assertion: VRAM consumption must be at least 1 GiB
            // SHY of the total tensor bytes — i.e. roughly PLE was excluded.
            // Allow generous slack (1 GiB) for KV cache, scratch, and pool overhead.
            // If PLE had been uploaded, consumed would be ≈ totalTensorBytes; the
            // gap proves the table didn't reach VRAM.
            long gap = totalTensorBytes - consumed;
            Assert.True(gap >= pleBytes - (1L << 30),
                $"Expected GPU consumption to be ≥ {pleBytes - (1L << 30):N0} bytes BELOW the " +
                $"total tensor footprint ({totalTensorBytes:N0}); was only {gap:N0} bytes lower " +
                $"(consumed={consumed:N0}). PLE table ({pleBytes:N0} bytes) appears to have " +
                "reached VRAM — check that the Phase-6 plumbing kept it CPU-resident and that " +
                "TierPlanner's PLE exclusion is in effect.");
        }
    }

    [Fact]
    public void Gemma4_CudaForwardPass_DisposeWithKvAliasNoDoubleFree()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Pre-condition: there must actually be aliased layers in the source GGUF.
        // If shared_kv_layers were 0 this test would pass trivially without
        // covering the alias-skip path.
        Assert.NotNull(hp.KvSourceLayer);
        int aliasedCount = 0;
        for (int i = 0; i < hp.NumLayers; i++)
            if (hp.KvSourceLayer![i] >= 0) aliasedCount++;
        Assert.True(aliasedCount > 0,
            "Gemma 4 should report at least one KV-aliased layer (shared_kv_layers > 0). " +
            "Test cannot exercise the double-free guard without one.");

        // Construct, immediately dispose — if Dispose attempts to Free an aliased
        // K/V handle twice we'll either crash with AccessViolationException or
        // surface a CUDA error here. Any exception escaping `using` fails the test.
        var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);
        fwd.Dispose();

        // Smoke: backend should still be live + responsive afterwards. A botched
        // double-free can leave the CUDA context in an unrecoverable state where
        // every subsequent op fails — proxy that with a tiny allocate/free round trip.
        var probe = gpu.Allocate(TensorShape.D1(64));
        gpu.Free(probe);
    }

    [Fact]
    public void Gemma4_CudaForwardPass_ForwardReturnsFiniteLogits()
    {
        // Phase 8 successor of the now-removed NotImplementedException guard.
        // Single forward must produce VocabSize finite logits — the load-bearing
        // smoke check that Forward no longer throws and the pipeline runs to
        // completion. Deeper coherence + CPU↔CUDA parity live in
        // Gemma4CudaForwardPassTests.
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.NotNull(hp.LayerHeadDim);

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        var logits = fwd.Forward(1, 0);
        Assert.Equal(hp.VocabSize, logits.Length);
        for (int i = 0; i < logits.Length; i++)
            Assert.True(float.IsFinite(logits[i]),
                $"Non-finite logit at vocab idx {i}: {logits[i]}");
    }
}
