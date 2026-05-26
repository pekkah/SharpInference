using System.Globalization;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Investigation harness for issue #31's main-forward parity bug on
/// Qwen3.6-27B-MTP. Off by default; runs only when SHARPI_REPRO_POS13=1.
/// Feeds the exact 13-token prompt from tests/fixtures/mtp_parity_27b.json
/// (chat-templated "The capital of France is") and dumps the top-K
/// position-12 logits — which select the position-13 token — plus the
/// full per-layer trace, to a file under tmp/ for offline comparison
/// against llama-eval-callback's output.
/// </summary>
public sealed class Repro_Pos13Parity
{
    private static int[] PromptTokens
    {
        get
        {
            var alt = Environment.GetEnvironmentVariable("SHARPI_REPRO_TOKENS");
            if (!string.IsNullOrEmpty(alt))
            {
                var parts = alt.Split(',');
                var ids = new List<int>(parts.Length);
                foreach (var p in parts)
                    if (int.TryParse(p.Trim(), out var id)) ids.Add(id);
                return ids.ToArray();
            }
            return new[]
            {
                248045, 846, 198, 760, 6511, 314, 9338, 369,
                248046, 198, 248045, 74455, 271,
            };
        }
    }

    private static string? FindMtpModelPath()
    {
        string[] absoluteCandidates =
        {
            @"C:\p\sharpi\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;
        return null;
    }

    [Fact]
    public void DumpPos13Logits()
    {
        if (Environment.GetEnvironmentVariable("SHARPI_REPRO_POS13") != "1") return;

        var modelPath = FindMtpModelPath();
        Assert.NotNull(modelPath);

        using var model = GgufModel.Open(modelPath!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);

        Directory.CreateDirectory(@"C:\p\sharpi\tmp");

        // Redirect stderr to a file so the SHARPI_TRACE_LAYERS=1 trace ends up
        // somewhere we can compare offline. xUnit otherwise captures stderr
        // and only surfaces it on test failure.
        var traceFile = Environment.GetEnvironmentVariable("SHARPI_TRACE_LAYERS") == "1"
            ? new StreamWriter(@"C:\p\sharpi\tmp\sharpi_layer_trace.txt") { AutoFlush = true }
            : null;
        TextWriter? prevErr = null;
        if (traceFile != null)
        {
            prevErr = Console.Error;
            Console.SetError(traceFile);
        }

        float[] logits;
        try
        {
            logits = fwd.Prefill(PromptTokens).ToArray();
        }
        finally
        {
            if (traceFile != null)
            {
                Console.SetError(prevErr!);
                traceFile.Dispose();
            }
        }

        var bypassTag =
            (Environment.GetEnvironmentVariable("SHARPI_BYPASS_GDN")  == "1" ? "_nogdn"  : "") +
            (Environment.GetEnvironmentVariable("SHARPI_BYPASS_ATTN") == "1" ? "_noattn" : "") +
            (Environment.GetEnvironmentVariable("SHARPI_BYPASS_MOE")  == "1" ? "_nomoe"  : "");
        var outPath = Path.Combine(@"C:\p\sharpi\tmp", $"sharpi_pos12_logits{bypassTag}.txt");

        var inv = CultureInfo.InvariantCulture;
        const int K = 30;
        var top = new (int idx, float val)[K];
        for (int i = 0; i < K; i++) top[i] = (-1, float.MinValue);
        for (int i = 0; i < logits.Length; i++)
        {
            float lv = logits[i];
            for (int j = 0; j < K; j++)
            {
                if (lv > top[j].val)
                {
                    for (int s = K - 1; s > j; s--) top[s] = top[s - 1];
                    top[j] = (i, lv);
                    break;
                }
            }
        }

        using var w = new StreamWriter(outPath);
        w.WriteLine($"# sharpi pos-12 logits (predicts pos 13) on Qwen3.6-27B-MTP-Q4_K_M");
        w.WriteLine($"# prompt_tokens = [{string.Join(",", PromptTokens)}]");
        w.WriteLine($"# vocab_size = {logits.Length}");
        w.WriteLine($"# bypass: GDN={Environment.GetEnvironmentVariable("SHARPI_BYPASS_GDN")} " +
                    $"ATTN={Environment.GetEnvironmentVariable("SHARPI_BYPASS_ATTN")} " +
                    $"MOE={Environment.GetEnvironmentVariable("SHARPI_BYPASS_MOE")}");
        w.WriteLine($"# rank token_id  logit");
        for (int j = 0; j < K; j++)
            w.WriteLine($"{j,4}  {top[j].idx,7}  {top[j].val.ToString("G8", inv)}");

        // A handful of specific tokens we care about for the divergence.
        int[] interesting = { 198, 248046, 248045, 271, 1235 /* think */, 151649 };
        w.WriteLine("# specific tokens (id  logit  rank):");
        foreach (var id in interesting)
        {
            if ((uint)id >= (uint)logits.Length) continue;
            int rank = 0;
            float v = logits[id];
            for (int i = 0; i < logits.Length; i++) if (logits[i] > v) rank++;
            w.WriteLine($"#   {id,7}  {v.ToString("G8", inv)}  rank={rank}");
        }

        Console.Error.WriteLine($"[repro] wrote {outPath}");
    }
}
