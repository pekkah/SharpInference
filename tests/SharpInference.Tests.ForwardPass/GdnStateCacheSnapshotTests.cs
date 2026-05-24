using System.Runtime.InteropServices;

using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #21 coverage for <see cref="GdnStateCache.SnapshotInto"/> /
/// <see cref="GdnStateCache.RestoreFrom"/>. The hybrid GDN forward passes capture
/// one of these at end-of-decode so the inference engine can reuse the recurrent
/// state across chat-continuation turns (which would otherwise be impossible — the
/// recurrence is destructively updated and <c>SupportsPartialRewind</c> stays
/// <c>false</c>).
/// </summary>
public sealed unsafe class GdnStateCacheSnapshotTests
{
    // Tiny shape so the snapshot byte count is small but every code path
    // (multiple GDN layers, both conv and scan blocks, length tracking) exercises.
    private static LayerType[] TinyLayerTypes() =>
    [
        LayerType.GatedDeltaNet,
        LayerType.Attention,
        LayerType.GatedDeltaNet,
        LayerType.GatedDeltaNet,
    ];

    private static GdnConfig TinyGdn() => new(
        NumKHeads: 1,
        NumVHeads: 2,
        HeadDim: 4,
        InnerSize: 8,
        ConvKernel: 3,
        FullAttentionInterval: 2);

    [Fact]
    public void Snapshot_RoundTrip_PreservesLengthAndPerLayerBuffers()
    {
        using var cache = new GdnStateCache(TinyLayerTypes(), TinyGdn());

        Assert.Equal(3, cache.NumGdnLayers);
        Assert.True(cache.ConvStateFloatsPerLayer > 0);
        Assert.True(cache.ScanStateFloatsPerLayer > 0);

        // Mutate the cache: write distinctive sentinel patterns into both buffers
        // of every GDN layer, advance _length a few times.
        int convLen = cache.ConvStateFloatsPerLayer;
        int scanLen = cache.ScanStateFloatsPerLayer;
        for (int g = 0; g < cache.NumGdnLayers; g++)
        {
            float* cp = cache.ConvStateAt(g);
            float* sp = cache.ScanStateAt(g);
            for (int i = 0; i < convLen; i++)
                cp[i] = (g + 1) * 100f + i * 0.5f;
            for (int i = 0; i < scanLen; i++)
                sp[i] = (g + 1) * -100f - i * 0.25f;
        }
        cache.IncrementPosition();
        cache.IncrementPosition();
        cache.IncrementPosition();
        Assert.Equal(3, cache.Length);

        // Snapshot into a managed byte array so we can also poke the header.
        long snapshotBytes = cache.SnapshotBytes;
        Assert.True(snapshotBytes > 0);
        var buf = new byte[snapshotBytes];
        fixed (byte* bp = buf)
        {
            cache.SnapshotInto(bp, buf.Length);

            // Header sanity: first int is _length, second int is padding (zero).
            Assert.Equal(3, *(int*)bp);
            Assert.Equal(0, *(int*)(bp + sizeof(int)));
        }

        // Wipe the cache. After this, every byte and the length are back to zero.
        cache.Reset();
        Assert.Equal(0, cache.Length);
        for (int g = 0; g < cache.NumGdnLayers; g++)
        {
            Assert.Equal(0f, cache.ConvStateAt(g)[0]);
            Assert.Equal(0f, cache.ScanStateAt(g)[scanLen - 1]);
        }

        // Restore from the snapshot and verify byte-identical recovery.
        fixed (byte* bp = buf)
            cache.RestoreFrom(bp, buf.Length);
        Assert.Equal(3, cache.Length);
        for (int g = 0; g < cache.NumGdnLayers; g++)
        {
            float* cp = cache.ConvStateAt(g);
            float* sp = cache.ScanStateAt(g);
            for (int i = 0; i < convLen; i++)
                Assert.Equal((g + 1) * 100f + i * 0.5f, cp[i]);
            for (int i = 0; i < scanLen; i++)
                Assert.Equal((g + 1) * -100f - i * 0.25f, sp[i]);
        }
    }

    [Fact]
    public void SnapshotBytes_MatchesDocumentedFormula()
    {
        using var cache = new GdnStateCache(TinyLayerTypes(), TinyGdn());

        long expected =
            sizeof(int) /* length */
          + sizeof(int) /* pad */
          + (long)cache.NumGdnLayers * (
                (long)cache.ConvStateFloatsPerLayer * sizeof(float)
              + (long)cache.ScanStateFloatsPerLayer * sizeof(float));
        Assert.Equal(expected, cache.SnapshotBytes);
    }

    [Fact]
    public void SnapshotInto_TooSmallDst_Throws()
    {
        using var cache = new GdnStateCache(TinyLayerTypes(), TinyGdn());

        // 1 byte below the required size triggers the size guard.
        long needed = cache.SnapshotBytes;
        var buf = new byte[needed];
        fixed (byte* bp = buf)
        {
            byte* p = bp;
            Assert.Throws<ArgumentException>(() => cache.SnapshotInto(p, needed - 1));
        }
    }

    [Fact]
    public void RestoreFrom_TooSmallSrc_Throws()
    {
        using var cache = new GdnStateCache(TinyLayerTypes(), TinyGdn());

        long needed = cache.SnapshotBytes;
        var buf = new byte[needed];
        fixed (byte* bp = buf)
        {
            byte* p = bp;
            Assert.Throws<ArgumentException>(() => cache.RestoreFrom(p, needed - 1));
        }
    }

    [Fact]
    public void SnapshotInto_NullDst_Throws()
    {
        using var cache = new GdnStateCache(TinyLayerTypes(), TinyGdn());
        Assert.Throws<ArgumentNullException>(() => cache.SnapshotInto(null, cache.SnapshotBytes));
    }

    [Fact]
    public void RestoreFrom_NullSrc_Throws()
    {
        using var cache = new GdnStateCache(TinyLayerTypes(), TinyGdn());
        Assert.Throws<ArgumentNullException>(() => cache.RestoreFrom(null, cache.SnapshotBytes));
    }

    [Fact]
    public void ConvKernel1_DegenerateLayout_SnapshotsScanOnly()
    {
        // ConvKernel == 1 ⇒ ConvStateFloatsPerLayer == 0; per-layer conv pointers
        // are null. SnapshotInto / RestoreFrom must skip those blocks cleanly.
        var gdn = new GdnConfig(
            NumKHeads: 1,
            NumVHeads: 1,
            HeadDim: 2,
            InnerSize: 2,
            ConvKernel: 1,
            FullAttentionInterval: 4);
        using var cache = new GdnStateCache([LayerType.GatedDeltaNet], gdn);

        Assert.Equal(0, cache.ConvStateFloatsPerLayer);
        Assert.True(cache.ConvStateAt(0) == null);

        // Write into the scan buffer only.
        float* sp = cache.ScanStateAt(0);
        for (int i = 0; i < cache.ScanStateFloatsPerLayer; i++)
            sp[i] = i + 1f;
        cache.IncrementPosition();

        long snapshotBytes = cache.SnapshotBytes;
        // Header (8) + scan only.
        Assert.Equal(
            sizeof(int) + sizeof(int) +
            (long)cache.ScanStateFloatsPerLayer * sizeof(float),
            snapshotBytes);

        var buf = new byte[snapshotBytes];
        fixed (byte* bp = buf)
            cache.SnapshotInto(bp, buf.Length);

        cache.Reset();
        Assert.Equal(0, cache.Length);

        fixed (byte* bp = buf)
            cache.RestoreFrom(bp, buf.Length);
        Assert.Equal(1, cache.Length);
        for (int i = 0; i < cache.ScanStateFloatsPerLayer; i++)
            Assert.Equal(i + 1f, sp[i]);
    }
}
