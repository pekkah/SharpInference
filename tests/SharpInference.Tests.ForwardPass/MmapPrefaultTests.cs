using System.Runtime.InteropServices;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Unit tests for the issue #221 mmap pre-fault helper. The gating decision is tested
/// as a pure function (no memory, no environment); a handful of integration tests
/// exercise the real sweep over small <see cref="NativeMemory"/> buffers and the
/// SHARPI_PREFAULT kill switch / RAM-fit skip (which must bail out before touching the
/// claimed bytes — verified by claiming far more than is actually allocated).
/// </summary>
public sealed class MmapPrefaultTests
{
    private const long Gib = 1L << 30;

    // ── Pure gating decision ────────────────────────────────────────────────

    [Fact]
    public void ShouldRun_NoBytes_IsFalse()
    {
        Assert.False(MmapPrefault.ShouldRun(null, 0, 16 * Gib, MmapPrefault.RamGate.FitsInRam, out var reason));
        Assert.Contains("no mapped weights", reason);
    }

    [Fact]
    public void ShouldRun_ModeZero_IsDisabled()
    {
        Assert.False(MmapPrefault.ShouldRun("0", 4 * Gib, 64 * Gib, MmapPrefault.RamGate.FitsInRam, out var reason));
        Assert.Contains("disabled", reason);
    }

    [Fact]
    public void ShouldRun_ModeOne_ForcesEvenWhenOverRam()
    {
        // Force bypasses the RAM-fit heuristic entirely.
        Assert.True(MmapPrefault.ShouldRun("1", 100 * Gib, 8 * Gib, MmapPrefault.RamGate.FitsInRam, out var reason));
        Assert.Contains("forced", reason);
    }

    [Fact]
    public void ShouldRun_Auto_FitsInRam_Runs()
    {
        Assert.True(MmapPrefault.ShouldRun(null, 4 * Gib, 16 * Gib, MmapPrefault.RamGate.FitsInRam, out _));
    }

    [Fact]
    public void ShouldRun_Auto_ExceedsEightyPercent_Skips()
    {
        // 14 GiB mapped > 80% of 16 GiB (= 12.8 GiB) → skip rather than thrash.
        Assert.False(MmapPrefault.ShouldRun(null, 14 * Gib, 16 * Gib, MmapPrefault.RamGate.FitsInRam, out var reason));
        Assert.Contains("exceeds", reason);
    }

    [Fact]
    public void ShouldRun_Auto_ExactlyEightyPercent_Runs()
    {
        // Boundary: the gate uses a strict '>' so exactly 80% still runs.
        long avail = 16 * Gib;
        Assert.True(MmapPrefault.ShouldRun(null, avail / 10 * 8, avail, MmapPrefault.RamGate.FitsInRam, out _));
    }

    [Fact]
    public void ShouldRun_AlwaysGate_IgnoresRamHeuristic()
    {
        // The fully-CPU-resident passes prefault regardless of the 80% threshold.
        Assert.True(MmapPrefault.ShouldRun(null, 100 * Gib, 16 * Gib, MmapPrefault.RamGate.Always, out _));
    }

    [Fact]
    public void ShouldRun_UnknownRam_DoesNotSkip()
    {
        // availRamBytes <= 0 means "couldn't measure" — don't skip on a guess.
        Assert.True(MmapPrefault.ShouldRun(null, 100 * Gib, 0, MmapPrefault.RamGate.FitsInRam, out _));
    }

    // ── Integration: real sweep over small buffers ──────────────────────────

    [Fact]
    public unsafe void Run_SmallBuffers_FaultsAndReportsBytes()
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_PREFAULT");
        Environment.SetEnvironmentVariable("SHARPI_PREFAULT", null); // auto
        const long sizeA = 1 << 20; // 1 MiB (spans many pages + a chunk boundary downstream)
        const long sizeB = 64 << 10; // 64 KiB
        void* a = NativeMemory.Alloc((nuint)sizeA);
        void* b = NativeMemory.Alloc((nuint)sizeB);
        try
        {
            new Span<byte>(a, (int)sizeA).Fill(1);
            new Span<byte>(b, (int)sizeB).Fill(2);

            var regions = new List<(nint, long)> { ((nint)a, sizeA), ((nint)b, sizeB) };
            var result = MmapPrefault.Run("test", regions, MmapPrefault.RamGate.Always);

            Assert.True(result.Ran);
            Assert.Equal(sizeA + sizeB, result.Bytes);
        }
        finally
        {
            NativeMemory.Free(a);
            NativeMemory.Free(b);
            Environment.SetEnvironmentVariable("SHARPI_PREFAULT", prev);
        }
    }

    [Fact]
    public unsafe void Run_NullAndZeroRegions_AreSkipped()
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_PREFAULT");
        Environment.SetEnvironmentVariable("SHARPI_PREFAULT", null);
        const long size = 4096;
        void* a = NativeMemory.Alloc((nuint)size);
        try
        {
            new Span<byte>(a, (int)size).Clear();
            var regions = new List<(nint, long)>
            {
                (0, size),        // null ptr → ignored
                ((nint)a, 0),     // zero bytes → ignored
                ((nint)a, size),  // the only real region
            };
            var result = MmapPrefault.Run("test", regions, MmapPrefault.RamGate.Always);

            Assert.True(result.Ran);
            Assert.Equal(size, result.Bytes); // only the valid region counts
        }
        finally
        {
            NativeMemory.Free(a);
            Environment.SetEnvironmentVariable("SHARPI_PREFAULT", prev);
        }
    }

    [Fact]
    public unsafe void Run_Disabled_SkipsWithoutTouching()
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_PREFAULT");
        Environment.SetEnvironmentVariable("SHARPI_PREFAULT", "0");
        // Tiny real allocation, but the region claims 1 TiB: if Run tried to stride-read
        // it the process would fault. The kill switch must bail out before any access.
        void* a = NativeMemory.Alloc(16);
        try
        {
            var regions = new List<(nint, long)> { ((nint)a, 1L << 40) };
            var result = MmapPrefault.Run("test", regions, MmapPrefault.RamGate.Always);

            Assert.False(result.Ran);
            Assert.Contains("disabled", result.Reason);
        }
        finally
        {
            NativeMemory.Free(a);
            Environment.SetEnvironmentVariable("SHARPI_PREFAULT", prev);
        }
    }

    [Fact]
    public unsafe void Run_AutoExceedsRam_SkipsWithoutTouching()
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_PREFAULT");
        Environment.SetEnvironmentVariable("SHARPI_PREFAULT", null); // auto
        void* a = NativeMemory.Alloc(16);
        try
        {
            // 1 PiB claimed > 80% of any real machine's RAM → skipped before reading.
            var regions = new List<(nint, long)> { ((nint)a, 1L << 50) };
            var result = MmapPrefault.Run("test", regions, MmapPrefault.RamGate.FitsInRam);

            Assert.False(result.Ran);
            Assert.Contains("exceeds", result.Reason);
        }
        finally
        {
            NativeMemory.Free(a);
            Environment.SetEnvironmentVariable("SHARPI_PREFAULT", prev);
        }
    }
}
