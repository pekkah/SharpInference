using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// CPU-only unit tests for <see cref="CudaHybridGdnForwardPass.MergePageAlignedRanges"/> — the
/// page-align + sort + merge step extracted from the #390 register-in-place MoE pin path. The
/// arithmetic (floor start, ceil end, coalesce overlapping/adjacent ranges) is off-by-one prone,
/// so it is exercised here without a GPU. GGUF expert tensors are 32-byte aligned (not
/// page-aligned), so two adjacent tensors can share a page; merging guarantees no page is
/// registered twice (a double <c>cudaHostRegister</c> would silently leave that range pageable).
/// </summary>
public sealed class MergePageAlignedRangesTests
{
    private const long Page = 4096;

    [Fact]
    public void SingleRange_RoundsOutToWholePages()
    {
        // ptr=100, bytes=50 → spans [100,150), rounds out to [0,4096).
        var merged = CudaHybridGdnForwardPass.MergePageAlignedRanges(
            new List<(long ptr, long bytes)> { (100, 50) }, Page);

        Assert.Single(merged);
        Assert.Equal((0L, 4096L), merged[0]);
    }

    [Fact]
    public void TwoRangesSharingAPage_MergeIntoOne()
    {
        // (0,4000) → [0,4096) and (4090,100) → [4096,8192). They are exactly adjacent at 4096 and
        // the first also touches the page the second starts in, so both coalesce into [0,8192).
        var merged = CudaHybridGdnForwardPass.MergePageAlignedRanges(
            new List<(long ptr, long bytes)> { (0, 4000), (4090, 100) }, Page);

        Assert.Single(merged);
        Assert.Equal((0L, 8192L), merged[0]);
    }

    [Fact]
    public void TwoRangesFarApart_StaySeparate()
    {
        // (0,50) → [0,4096) and (1_000_000,50) → [999_424,1_003_520). No shared/adjacent page.
        var merged = CudaHybridGdnForwardPass.MergePageAlignedRanges(
            new List<(long ptr, long bytes)> { (0, 50), (1_000_000, 50) }, Page);

        Assert.Equal(2, merged.Count);
        Assert.Equal((0L, 4096L), merged[0]);
        Assert.Equal((999_424L, 1_003_520L), merged[1]);
    }

    [Fact]
    public void ExactlyAdjacentPageAlignedRanges_Coalesce()
    {
        // (0,4096) → [0,4096) and (4096,4096) → [4096,8192). Adjacent (start == prev end) → merge.
        var merged = CudaHybridGdnForwardPass.MergePageAlignedRanges(
            new List<(long ptr, long bytes)> { (0, 4096), (4096, 4096) }, Page);

        Assert.Single(merged);
        Assert.Equal((0L, 8192L), merged[0]);
    }

    [Fact]
    public void NonPositiveBytes_AreSkipped()
    {
        // A zero-byte and a negative-byte entry are dropped; only the real range survives.
        var merged = CudaHybridGdnForwardPass.MergePageAlignedRanges(
            new List<(long ptr, long bytes)> { (0, 0), (4096, -10), (8192, 50) }, Page);

        Assert.Single(merged);
        Assert.Equal((8192L, 12288L), merged[0]);
    }

    [Fact]
    public void UnsortedInput_IsSortedAndMerged()
    {
        // Out-of-order input with a shared page in the middle: the far-late range first, then two
        // that share page 0, then an early standalone. Result must be sorted ascending and merged.
        var merged = CudaHybridGdnForwardPass.MergePageAlignedRanges(
            new List<(long ptr, long bytes)>
            {
                (1_000_000, 50),  // [999_424, 1_003_520)
                (4090, 100),      // [4096, 8192)
                (0, 4000),        // [0, 4096)  — coalesces with the previous into [0, 8192)
            }, Page);

        Assert.Equal(2, merged.Count);
        Assert.Equal((0L, 8192L), merged[0]);
        Assert.Equal((999_424L, 1_003_520L), merged[1]);
    }

    [Theory]
    // Small page size (16) makes the floor/ceil rounding easy to read by eye.
    [InlineData(0, 1, 0, 16)]          // 1 byte → one whole page
    [InlineData(15, 1, 0, 16)]         // last byte of page 0 → [0,16)
    [InlineData(16, 1, 16, 32)]        // first byte of page 1 → [16,32)
    [InlineData(10, 20, 0, 32)]        // spans pages 0 and 1 → [0,32)
    public void SmallPage_RoundsCorrectly(long ptr, long bytes, long expStart, long expEnd)
    {
        var merged = CudaHybridGdnForwardPass.MergePageAlignedRanges(
            new List<(long ptr, long bytes)> { (ptr, bytes) }, 16);

        Assert.Single(merged);
        Assert.Equal((expStart, expEnd), merged[0]);
    }

    [Fact]
    public void HighBitPointer_AlignsViaUnsigned()
    {
        // A host pointer with bit 63 set is negative as a signed long; signed division would floor
        // the wrong way (truncate toward zero) and misorder it. The unsigned alignment must still
        // floor the start down to the page boundary and ceil the end up (Gemini #390 review).
        long ptr = unchecked((long)0x8000_0000_0000_0010UL);  // page base + 16 bytes
        var merged = CudaHybridGdnForwardPass.MergePageAlignedRanges(
            new List<(long ptr, long bytes)> { (ptr, 32) }, Page);

        Assert.Single(merged);
        long expStart = unchecked((long)0x8000_0000_0000_0000UL);  // floored to the page boundary
        long expEnd   = unchecked((long)0x8000_0000_0000_1000UL);  // ceiled to the next page
        Assert.Equal((expStart, expEnd), merged[0]);
        // end − start is the correct positive byte count (one page) regardless of sign.
        Assert.Equal(4096L, merged[0].end - merged[0].start);
    }

    [Fact]
    public void EmptyInput_ProducesNoRanges()
    {
        var merged = CudaHybridGdnForwardPass.MergePageAlignedRanges(
            new List<(long ptr, long bytes)>(), Page);

        Assert.Empty(merged);
    }
}
