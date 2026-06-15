using System.Diagnostics;
using System.Runtime.InteropServices;

namespace SharpInference.Engine;

/// <summary>
/// Forces memory-mapped, CPU-resident weight pages resident <i>before</i> the first
/// forward pass, so the first request does not stall on demand paging (issue #221).
///
/// <para>The mmap'd GGUF is paged in lazily by the OS: without a warm-up, the first
/// request faults every weight page one at a time on the critical path, running ~5×
/// slower than warm steady-state on CPU-MoE configs (the same penalty #210's bench
/// protocol works around by running every cell twice). A parallel sequential sweep
/// pulls the same bytes in at memory/SSD bandwidth — a 15-20 GiB expert region warms
/// in seconds at NVMe speeds.</para>
///
/// <para>Two-step warm-up: (1) a best-effort OS read-ahead hint
/// (<c>PrefetchVirtualMemory</c> on Windows, <c>posix_madvise(WILLNEED)</c> on Linux),
/// which coalesces the I/O; (2) a parallel per-page stride read that <i>guarantees</i>
/// residency and gives the wall-clock measurement. Correctness never depends on the
/// hint succeeding — the stride read is the source of truth.</para>
///
/// <para>Tunable via <c>SHARPI_PREFAULT</c>: <c>0</c> disables all prefaulting,
/// <c>1</c> forces it on (bypassing the RAM-fit heuristic). Unset = auto.</para>
/// </summary>
internal static unsafe partial class MmapPrefault
{
    /// <summary>RAM-fit policy applied in auto mode (env unset).</summary>
    internal enum RamGate
    {
        /// <summary>Skip when the mapped set exceeds ~80% of available RAM — it would
        /// just thrash. Used by the GPU-offload hybrid passes, whose CPU-resident
        /// weights are a subset of the model that should comfortably fit.</summary>
        FitsInRam,

        /// <summary>Always sweep (subject only to the <c>SHARPI_PREFAULT=0</c> kill
        /// switch). Used by the fully-CPU-resident passes, where the user has already
        /// chosen to run the whole model from RAM and prefaulting is the point.</summary>
        Always,
    }

    internal readonly record struct Result(bool Ran, long Bytes, double Seconds, string Reason);

    /// <summary>
    /// Pure gating decision, factored out so it can be unit-tested without touching
    /// memory or the environment. <paramref name="mode"/> is the raw
    /// <c>SHARPI_PREFAULT</c> value (null = unset).
    /// </summary>
    internal static bool ShouldRun(string? mode, long totalBytes, long availRamBytes,
        RamGate gate, out string reason)
    {
        if (totalBytes <= 0) { reason = "no mapped weights"; return false; }
        if (mode == "0") { reason = "disabled (SHARPI_PREFAULT=0)"; return false; }
        if (mode == "1") { reason = "forced (SHARPI_PREFAULT=1)"; return true; }
        if (gate == RamGate.FitsInRam && availRamBytes > 0 && totalBytes > availRamBytes / 10 * 8)
        {
            reason = $"skipped: {totalBytes >> 20} MiB mapped exceeds 80% of "
                   + $"{availRamBytes >> 20} MiB RAM (set SHARPI_PREFAULT=1 to force)";
            return false;
        }
        reason = "auto";
        return true;
    }

    /// <summary>
    /// Pre-fault the given CPU-resident mmap regions. Null/zero-size regions are
    /// ignored, so callers can pass their whole weight set and let the helper filter.
    /// Logs the warmed size and rate (or the skip reason) to <see cref="Console.Error"/>.
    /// </summary>
    internal static Result Run(string label, List<(nint Ptr, long Bytes)> regions,
        RamGate gate = RamGate.FitsInRam)
    {
        long total = 0;
        foreach (var (ptr, bytes) in regions)
            if (ptr != 0 && bytes > 0) total += bytes;

        string? mode = Environment.GetEnvironmentVariable("SHARPI_PREFAULT");
        long avail = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        if (!ShouldRun(mode, total, avail, gate, out string reason))
        {
            // Only announce a deliberate skip — staying silent on "nothing to do".
            if (total > 0) Console.Error.WriteLine($"[{label}] Pre-fault {reason}.");
            return new Result(false, total, 0, reason);
        }

        long t0 = Stopwatch.GetTimestamp();
        TryAdvise(regions);                  // best-effort OS read-ahead hint
        long touchSum = StrideRead(regions); // guaranteed residency + measurement
        double secs = (Stopwatch.GetTimestamp() - t0) / (double)Stopwatch.Frequency;

        double gib = total / (1024.0 * 1024 * 1024);
        double rate = secs > 1e-3 ? gib / secs : 0;
        Console.Error.WriteLine(
            $"[{label}] Pre-faulted {gib:F2} GiB of CPU-resident weights in {secs:F1}s ({rate:F1} GiB/s).");
        // Defeat dead-code elimination of the reads (touchSum is otherwise unused).
        if (touchSum == long.MinValue) Console.Error.Write(touchSum);
        return new Result(true, total, secs, reason);
    }

    /// <summary>Parallel per-page stride read. Large regions are split into fixed-size
    /// chunks so a handful of multi-GiB expert tensors still spread across all cores
    /// rather than bottlenecking one thread per tensor.</summary>
    private static long StrideRead(List<(nint Ptr, long Bytes)> regions)
    {
        int pageSize = Environment.SystemPageSize;
        const long chunk = 32L * 1024 * 1024;
        var jobs = new List<(nint Ptr, long Start, long End)>();
        foreach (var (ptr, bytes) in regions)
        {
            if (ptr == 0 || bytes <= 0) continue;
            for (long s = 0; s < bytes; s += chunk)
                jobs.Add((ptr, s, Math.Min(s + chunk, bytes)));
        }

        long touchSum = 0;
        Parallel.ForEach(jobs, job =>
        {
            byte* p = (byte*)job.Ptr;
            long localSum = 0;
            for (long off = job.Start; off < job.End; off += pageSize)
                localSum += p[off];
            localSum += p[job.End - 1]; // tail page of this chunk
            Interlocked.Add(ref touchSum, localSum);
        });
        return touchSum;
    }

    /// <summary>Best-effort OS read-ahead hint. Wrapped so a P/Invoke failure (missing
    /// symbol, unsupported platform) never derails the guaranteed stride read.</summary>
    private static void TryAdvise(List<(nint Ptr, long Bytes)> regions)
    {
        try
        {
            if (OperatingSystem.IsWindows()) AdviseWindows(regions);
            else if (OperatingSystem.IsLinux()) AdviseLinux(regions);
        }
        catch
        {
            // Ignored: the stride read below still forces residency.
        }
    }

    // ── Windows: PrefetchVirtualMemory ──────────────────────────────────────
    private static void AdviseWindows(List<(nint Ptr, long Bytes)> regions)
    {
        int n = 0;
        foreach (var (ptr, bytes) in regions)
            if (ptr != 0 && bytes > 0) n++;
        if (n == 0) return;

        var entries = new Win32MemoryRangeEntry[n];
        int j = 0;
        foreach (var (ptr, bytes) in regions)
        {
            if (ptr == 0 || bytes <= 0) continue;
            entries[j].VirtualAddress = ptr;
            entries[j].NumberOfBytes = (nuint)bytes;
            j++;
        }
        // Return value is intentionally ignored: this is an asynchronous, best-effort
        // hint; the stride read is the residency guarantee.
        fixed (Win32MemoryRangeEntry* e = entries)
            _ = PrefetchVirtualMemory(GetCurrentProcess(), (nuint)n, e, 0);
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct Win32MemoryRangeEntry
    {
        public nint VirtualAddress;
        public nuint NumberOfBytes;
    }

    [LibraryImport("kernel32.dll", EntryPoint = "PrefetchVirtualMemory")]
    private static partial int PrefetchVirtualMemory(nint hProcess, nuint numberOfEntries,
        Win32MemoryRangeEntry* virtualAddresses, uint flags);

    [LibraryImport("kernel32.dll", EntryPoint = "GetCurrentProcess")]
    private static partial nint GetCurrentProcess();

    // ── Linux: posix_madvise(WILLNEED) ──────────────────────────────────────
    private static void AdviseLinux(List<(nint Ptr, long Bytes)> regions)
    {
        const int posixMadvWillNeed = 3;
        long pageSize = Environment.SystemPageSize;
        foreach (var (ptr, bytes) in regions)
        {
            if (ptr == 0 || bytes <= 0) continue;
            // posix_madvise wants a page-aligned address; round down and extend length.
            long addr = ptr;
            long aligned = addr & ~(pageSize - 1);
            long len = bytes + (addr - aligned);
            _ = posix_madvise((nint)aligned, (nuint)len, posixMadvWillNeed);
        }
    }

    [LibraryImport("libc", EntryPoint = "posix_madvise")]
    private static partial int posix_madvise(nint addr, nuint length, int advice);
}
