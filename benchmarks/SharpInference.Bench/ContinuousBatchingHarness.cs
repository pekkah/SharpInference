using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SharpInference.Bench;

/// <summary>
/// Manual A/B harness for issue #183 (GPU/CPU saturation under concurrent load):
/// measures how much a long-prompt admission stalls already-decoding sequences,
/// with legacy blocking prefill (chunk=0) vs chunked+packed interleaved prefill.
///
/// Scenario: 3 interactive requests decode continuously; once they are all streaming,
/// a long-prompt request is injected. Reported per mode:
///   - per-interactive-sequence max / p95 inter-token gap (the "stall")
///   - long request time-to-first-token and total latency
///   - aggregate generated tokens/s
///
/// Usage: dotnet run --project benchmarks/SharpInference.Bench -c Release -- --cb
///        [--chunk N] (run a single chunk size instead of the 0-vs-256 A/B)
///        [--long-tokens N] (approx. long-prompt token count, default 1024)
///        [--decode N] (interactive MaxNewTokens, default 96)
/// </summary>
public static class ContinuousBatchingHarness
{
    private sealed record SeqResult(string Name, List<double> ChunkTimesMs, double SubmitMs, double DoneMs)
    {
        public int Tokens => ChunkTimesMs.Count;
        public double TtftMs => ChunkTimesMs.Count > 0 ? ChunkTimesMs[0] - SubmitMs : double.NaN;

        public (double Max, double P95) Gaps()
        {
            if (ChunkTimesMs.Count < 2) return (double.NaN, double.NaN);
            var gaps = new List<double>(ChunkTimesMs.Count - 1);
            for (int i = 1; i < ChunkTimesMs.Count; i++)
                gaps.Add(ChunkTimesMs[i] - ChunkTimesMs[i - 1]);
            gaps.Sort();
            return (gaps[^1], gaps[(int)(gaps.Count * 0.95) is var k && k >= gaps.Count ? gaps.Count - 1 : k]);
        }
    }

    public static async Task Run(string[] args)
    {
        int chunkArg = ArgInt(args, "--chunk", -1);
        int longTokens = ArgInt(args, "--long-tokens", 1024);
        int decodeTokens = ArgInt(args, "--decode", 96);

        var path = BenchmarkHelper.FindModelPath("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("SmolLM2-1.7B-Instruct-Q4_K_M.gguf not found");

        Console.WriteLine($"[cb] model: {path}");
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(model, backend, hp);

        // Issue #189: report the dequant-once weight-cache state so the prefill A/B below is
        // interpretable. Force it off for the "before" numbers with SHARPI_PREFILL_DEQUANT_MB=0.
        Console.WriteLine($"[cb] OpenBLAS: {(SimdKernels.BlasAvailable ? "LOADED" : "not found")}");
        string budgetStr = fwd.PrefillDequantCacheBudgetBytes == long.MaxValue
            ? "unlimited"
            : $"{fwd.PrefillDequantCacheBudgetBytes / (1024 * 1024)} MiB";
        Console.WriteLine($"[cb] dequant cache: {(fwd.PrefillDequantCacheActive
            ? $"ACTIVE ({budgetStr} budget, covers model)"
            : fwd.PrefillDequantCacheBudgetBytes > 0 ? "partial (budget < model)" : "off")}");

        // Long prompt: repeat filler text until the tokenizer crosses the target count.
        const string filler = "The quick brown fox jumps over the lazy dog near the quiet river bank. ";
        var sb = new System.Text.StringBuilder("<|im_start|>user\nSummarize the following text:\n");
        while (tokenizer.Encode(sb.ToString()).Count < longTokens)
            sb.Append(filler);
        sb.Append("<|im_end|>\n<|im_start|>assistant\n");
        string longPrompt = sb.ToString();
        int longPromptTokens = tokenizer.Encode(longPrompt).Count;

        const string shortPrompt = "<|im_start|>user\nTell me a short story about a lighthouse keeper.<|im_end|>\n<|im_start|>assistant\n";

        // Warm the weight pages once so the first measured mode isn't penalized by
        // cold mmap reads (see bench-cache-warmth feedback).
        Console.WriteLine("[cb] warmup...");
        {
            using var warmEngine = new ContinuousBatchingEngine(fwd, tokenizer, "warmup", maxBatchSize: 1);
            var wsp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };
            await foreach (var _ in warmEngine.GenerateAsync(shortPrompt, wsp)) { }
        }

        if (args.Contains("--packed"))
        {
            // --chunk N drives the per-prompt chunk size of the chunked modes so the A/B can
            // reproduce the issue #189 chunk sweep (e.g. --packed --chunk 32 / 64). Default 64.
            RunPackedPrefillAb(fwd, tokenizer, longPrompt, chunkArg > 0 ? chunkArg : 64);
            return;
        }

        int[] modes = chunkArg >= 0 ? [chunkArg] : [0, 256];
        foreach (int chunk in modes)
        {
            Console.WriteLine();
            Console.WriteLine($"══ mode: prefillChunkTokens={chunk} ({(chunk == 0 ? "legacy blocking prefill" : "chunked + packed interleaved")}) ══");
            await RunMode(fwd, tokenizer, chunk, shortPrompt, longPrompt, longPromptTokens, decodeTokens);
        }
    }

    /// <summary>
    /// Direct A/B for issue #183 Gap 2: prefill S prompts of T tokens each, serially
    /// (one PrefillWithCache per prompt) vs as ONE packed PrefillPackedMulti call.
    /// Isolates the weight-read amortization across prompts from the scheduling change.
    /// </summary>
    private static void RunPackedPrefillAb(ForwardPass fwd, GgufTokenizer tokenizer, string longText, int perSeqChunk)
    {
        const int S = 4, T = 256;
        var all = tokenizer.Encode(longText).ToArray();
        var prompts = new int[S][];
        for (int s = 0; s < S; s++)
            prompts[s] = all.AsSpan(s * T, T).ToArray();

        int perSeq = Math.Clamp(perSeqChunk, 1, T); // chunk each prompt advances per round in the chunked modes
        Console.WriteLine($"  (per-prompt chunk = {perSeq} tokens)");

        foreach (string mode in new[] { "whole-prompt serial", "chunked serial", "chunked packed" })
        {
            var sw = Stopwatch.StartNew();
            var caches = new PagedKvCache[S];
            for (int s = 0; s < S; s++) caches[s] = fwd.CreateCache();

            switch (mode)
            {
                case "whole-prompt serial":
                    // Pre-#183 admission: each prompt prefills alone, full length.
                    for (int s = 0; s < S; s++)
                        fwd.PrefillWithCache(prompts[s], caches[s]);
                    break;

                case "chunked serial":
                    // Gap 1 without Gap 2: same rounds, but one small call per prompt —
                    // each call re-pays the weight read/dequant for only perSeq tokens.
                    for (int consumed = 0; consumed < T; consumed += perSeq)
                    {
                        int take = Math.Min(perSeq, T - consumed);
                        for (int s = 0; s < S; s++)
                        {
                            var segment = new ArraySegment<int>(prompts[s], consumed, take);
                            fwd.PrefillWithCache(segment, caches[s], startPos: consumed);
                        }
                    }
                    break;

                case "chunked packed":
                    // Gap 1 + Gap 2: the S per-prompt chunks run as ONE packed pass.
                    for (int consumed = 0; consumed < T; consumed += perSeq)
                    {
                        int take = Math.Min(perSeq, T - consumed);
                        var chunks = new ReadOnlyMemory<int>[S];
                        var startPos = new int[S];
                        var want = new bool[S];
                        for (int s = 0; s < S; s++)
                        {
                            chunks[s] = prompts[s].AsMemory(consumed, take);
                            startPos[s] = consumed;
                            want[s] = consumed + take == T;
                        }
                        fwd.PrefillPackedMulti(chunks, startPos, caches, want);
                    }
                    break;
            }

            sw.Stop();
            foreach (var c in caches) c.Dispose();
            double tps = S * T / sw.Elapsed.TotalSeconds;
            Console.WriteLine($"  {mode,-20} ({(mode == "whole-prompt serial" ? $"{T}/call" : $"{perSeq}/prompt/round")}): {S}×{T} tokens in {sw.Elapsed.TotalMilliseconds,7:F0} ms → {tps,7:F1} tok/s");
        }
    }

    private static async Task RunMode(
        ForwardPass fwd, GgufTokenizer tokenizer, int chunk,
        string shortPrompt, string longPrompt, int longPromptTokens, int decodeTokens)
    {
        using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "cb-bench",
            maxBatchSize: 8, prefillChunkTokens: chunk);

        var clock = Stopwatch.StartNew();
        var results = new List<Task<SeqResult>>();
        var progress = new int[3]; // live token counters for the interactive sequences

        Task<SeqResult> Launch(string name, string prompt, int maxNew, int progressSlot)
        {
            double submit = clock.Elapsed.TotalMilliseconds;
            var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = maxNew };
            return Task.Run(async () =>
            {
                var times = new List<double>(maxNew);
                await foreach (var _ in engine.GenerateAsync(prompt, sp))
                {
                    times.Add(clock.Elapsed.TotalMilliseconds);
                    if (progressSlot >= 0) Interlocked.Increment(ref progress[progressSlot]);
                }
                return new SeqResult(name, times, submit, clock.Elapsed.TotalMilliseconds);
            });
        }

        // 3 interactive sequences decoding continuously.
        for (int i = 0; i < 3; i++)
            results.Add(Launch($"interactive-{i}", shortPrompt, decodeTokens, i));

        // Wait until all three are visibly streaming, then inject the long prompt.
        while (!results.All(r => r.IsCompleted)
               && !Enumerable.Range(0, 3).All(i => Volatile.Read(ref progress[i]) >= 4 || results[i].IsCompleted))
            await Task.Delay(25);

        double injectAt = clock.Elapsed.TotalMilliseconds;
        results.Add(Launch($"long-prompt({longPromptTokens}tok)", longPrompt, 8, -1));

        var done = await Task.WhenAll(results);
        double wallMs = clock.Elapsed.TotalMilliseconds;

        int totalTokens = done.Sum(r => r.Tokens);
        Console.WriteLine($"  long prompt injected at {injectAt,8:F0} ms");
        Console.WriteLine($"  {"sequence",-22} {"tokens",6} {"ttft ms",9} {"max gap ms",11} {"p95 gap ms",11} {"done ms",9}");
        foreach (var r in done)
        {
            var (max, p95) = r.Gaps();
            Console.WriteLine($"  {r.Name,-22} {r.Tokens,6} {r.TtftMs,9:F0} {max,11:F0} {p95,11:F0} {r.DoneMs,9:F0}");
        }
        Console.WriteLine($"  wall {wallMs:F0} ms · generated {totalTokens} tokens · aggregate {totalTokens / (wallMs / 1000.0):F1} tok/s");
    }

    private static int ArgInt(string[] args, string name, int fallback)
    {
        int idx = Array.IndexOf(args, name);
        return idx >= 0 && idx + 1 < args.Length && int.TryParse(args[idx + 1], out int v) ? v : fallback;
    }
}
