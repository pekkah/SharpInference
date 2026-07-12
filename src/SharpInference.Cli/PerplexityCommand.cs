using System.ComponentModel;
using System.Diagnostics;
using Spectre.Console;
using Spectre.Console.Cli;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Cli;

/// <summary>
/// Teacher-forced perplexity evaluation over a text file — the llama.cpp
/// <c>llama-perplexity</c> analogue and the KVarN P0 accuracy gate
/// (issue #180, docs/kvarn-feasibility-research.md §6).
///
/// Method: tokenize the file (raw text path, BOS prepended when the model's
/// <c>add_bos_token</c> asks for it), take the first <c>-c</c> tokens, and feed
/// them one at a time through the CPU <see cref="ForwardPass"/> —
/// <b>token-by-token <see cref="ForwardPass.Forward"/>, not batched Prefill</b>,
/// so with <c>--tq</c> every step exercises the TurboQuant/KVarN compressed-read
/// attention path exactly like decode does. After feeding token t we accumulate
/// the negative log-likelihood of token t+1 under the full-vocab log-softmax
/// (natural log, double accumulation). The first position is never scored.
///
/// Reported: token count, mean NLL, perplexity = exp(mean NLL), and mean NLL
/// split by target-position buckets ([1, window) / [window, 1024) / [1024, +))
/// so degradation in the compressed region is visible separately from the
/// always-FP32 recent window.
///
/// Usage: sharpi-cli perplexity -m model.gguf -f corpus.txt -c 2048 [--tq [--tq-mode kvarn]]
/// </summary>
public sealed class PerplexityCommand : Command<PerplexityCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model")]
        [Description("Path to GGUF model file")]
        public string? ModelPath { get; init; }

        [CommandOption("-f|--file")]
        [Description("UTF-8 text file to evaluate (llama.cpp -f/--file). Tokenized raw (no chat template); the first -c tokens are scored.")]
        public string? TextFile { get; init; }

        [CommandOption("-c|--ctx-size")]
        [Description("Number of tokens to evaluate (default: 2048). Clamped to the model context length and the corpus length.")]
        [DefaultValue(2048)]
        public int CtxSize { get; init; }

        [CommandOption("--tq")]
        [Description("Enable TurboQuant KV cache compression (same flag as the run command)")]
        [DefaultValue(false)]
        public bool TurboQuant { get; init; }

        [CommandOption("--tq-mode")]
        [Description("TurboQuant quantizer for --tq: auto (default: kvarn where supported, else lloydmax with a quality warning), kvarn (issue #180: 4-bit K / 2-bit V, 128-token tiles), or lloydmax (3-bit codebooks; severely degrades quality on QK-norm models such as Qwen3 — issue #432).")]
        [DefaultValue("auto")]
        public string TqModeStr { get; init; } = "auto";

        [CommandOption("--tq-window")]
        [Description("FP32 recent-token window before compression kicks in (default: 256; min 128 for kvarn — one full tile). Also sets the first position-bucket edge of the report, so pass the same value to the fp32 baseline for bucket-comparable numbers.")]
        [DefaultValue(256)]
        public int TqWindow { get; init; }

        [CommandOption("--ngl|--n-gpu-layers|--gpu-layers|-g")]
        [Description("Layers on GPU: 0 (default, CPU forward pass) or -1 (full CUDA offload via CudaForwardPass — issue #180 Task 5a). Partial offload is not supported.")]
        [DefaultValue(0)]
        public int NGpuLayers { get; init; }
    }

    /// <summary>
    /// Validates the flag combination (no model needed) and resolves the quantizer.
    /// Mirrors the run command's --tq/--tq-mode rules: kvarn requires --tq, and the
    /// harness runs either the CPU forward pass (-g 0) or the full-CUDA-offload
    /// CudaForwardPass (-g -1, issue #180 Task 5a) — any other -g is rejected
    /// outright. The window floor is one compressed tile (128 for KVarN, 32 for
    /// Lloyd-Max FastScan) so the cache constructor can't throw later with a less
    /// actionable message. When <paramref name="autoMode"/> comes back true the
    /// returned quantizer is tentative — <see cref="ResolveAutoQuantizer"/> picks the
    /// final codec once the model hyperparams are known (issue #432).
    /// </summary>
    internal static bool TryValidateFlags(bool tq, string tqModeStr, int tqWindow, int nGpuLayers,
        out TqQuantizer quantizer, out bool autoMode, out string? error)
    {
        quantizer = TqQuantizer.LloydMax;
        autoMode = false;
        switch (tqModeStr.Trim().ToLowerInvariant())
        {
            case "" or "auto":
                autoMode = true;
                break;
            case "lloydmax" or "lloyd-max":
                quantizer = TqQuantizer.LloydMax;
                break;
            case "kvarn":
                quantizer = TqQuantizer.KVarN;
                break;
            default:
                error = $"Unknown --tq-mode value '{tqModeStr}'. Expected one of: auto, lloydmax, kvarn.";
                return false;
        }

        if (!autoMode && quantizer == TqQuantizer.KVarN && !tq)
        {
            error = "--tq-mode kvarn requires --tq.";
            return false;
        }
        if (nGpuLayers is not (0 or -1))
        {
            error = "the perplexity harness runs either the CPU forward pass (-g 0) or full CUDA offload (-g -1); partial offload is not supported.";
            return false;
        }
        // Auto mode only resolves to KVarN when the window fits a whole tile, so its
        // hard floor is the Lloyd-Max one.
        int minWindow = !autoMode && quantizer == TqQuantizer.KVarN ? 128 : 32;
        if (tq && tqWindow < minWindow)
        {
            error = $"--tq-window must be >= {minWindow} for --tq-mode {(quantizer == TqQuantizer.KVarN ? "kvarn (one full 128-token tile)" : "lloydmax (one FastScan tile)")}; got {tqWindow}.";
            return false;
        }
        if (tqWindow < 1)
        {
            error = $"--tq-window must be >= 1; got {tqWindow}.";
            return false;
        }

        error = null;
        return true;
    }

    /// <summary>
    /// Resolves the auto --tq-mode (issue #432): KVarN wherever it is supported,
    /// otherwise Lloyd-Max with <paramref name="fallbackReason"/> set so the caller
    /// can print the quality warning (Lloyd-Max 3-bit collapses on QK-norm models —
    /// Qwen3-0.6B wikitext-2 PPL: 15.47 fp32 / 15.67 KVarN / 945.6 Lloyd-Max 3-bit).
    /// Takes primitives instead of hyperparams so the matrix is unit-testable.
    /// </summary>
    internal static TqQuantizer ResolveAutoQuantizer(int tqWindow, int nGpuLayers, int headDim,
        bool isMoE, bool snapKvEnabled, out string? fallbackReason)
    {
        // Perplexity offloads only to CUDA (never Vulkan), so isVulkan=false and CUDA is
        // assumed available on the GPU branch — TqSupport is the shared matrix (issue #437).
        fallbackReason = TqSupport.KVarNBlockedReason(
            headDim, snapKvEnabled, onGpu: nGpuLayers != 0,
            isVulkan: false, cudaAvailable: true, isMoE, window: tqWindow);
        return fallbackReason is null ? TqQuantizer.KVarN : TqQuantizer.LloydMax;
    }

    /// <summary>
    /// NLL of <paramref name="target"/> under the full-vocab log-softmax of
    /// <paramref name="logits"/>, in nats. Two passes (max, then sum-exp) with
    /// double accumulation; returns a non-finite value if the logits contain
    /// NaN/Inf so the caller can count anomalies instead of silently averaging them.
    /// </summary>
    internal static double NegativeLogLikelihood(ReadOnlySpan<float> logits, int target)
    {
        if ((uint)target >= (uint)logits.Length) return double.NaN;

        double max = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];   // NaN never compares greater — caught below

        double sumExp = 0.0;
        for (int i = 0; i < logits.Length; i++)
            sumExp += Math.Exp(logits[i] - max);

        // log-softmax: logit[target] - max - log(sum exp(logit - max))
        return -(logits[target] - max - Math.Log(sumExp));
    }

    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellation)
    {
        if (!TryValidateFlags(settings.TurboQuant, settings.TqModeStr, settings.TqWindow,
                settings.NGpuLayers, out TqQuantizer quantizer, out bool tqModeIsAuto, out string? flagError))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(flagError!)}");
            return 1;
        }
        if (!tqModeIsAuto && settings.TurboQuant && quantizer == TqQuantizer.KVarN && SnapKvConfig.FromEnvironment().Enabled)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] --tq-mode kvarn does not compose with SnapKV eviction yet (issue #180 follow-up); unset [yellow]SHARPI_SNAPKV_BUDGET[/].");
            return 1;
        }

        if (settings.TextFile is not { Length: > 0 } textFile)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] No text file given. Use [yellow]-f <corpus.txt>[/]");
            return 1;
        }
        if (!File.Exists(textFile))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] text file not found: {Markup.Escape(textFile)}");
            return 1;
        }
        string text;
        try
        {
            text = File.ReadAllText(textFile);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                      or System.Security.SecurityException or NotSupportedException)
        {
            AnsiConsole.MarkupLine($"[red]Error reading text file:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }

        var modelPath = settings.ModelPath;
        if (modelPath is null || !File.Exists(modelPath))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] No model file found. Use [yellow]-m <path>[/]");
            return 1;
        }
        if (settings.CtxSize < 2)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] -c must be >= 2 (need at least one prediction); got {settings.CtxSize}.");
            return 1;
        }

        AnsiConsole.MarkupLine($"[dim]Loading model:[/] {Markup.Escape(modelPath)}");
        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Auto --tq-mode (issue #432): prefer KVarN, fall back to Lloyd-Max with a
        // loud quality warning where KVarN is unsupported.
        if (tqModeIsAuto && settings.TurboQuant)
        {
            quantizer = ResolveAutoQuantizer(settings.TqWindow, settings.NGpuLayers, hp.HeadDim,
                hp.IsMoE, SnapKvConfig.FromEnvironment().Enabled, out string? fallbackReason);
            if (fallbackReason is not null)
                AnsiConsole.MarkupLine(
                    $"[yellow]Warning:[/] --tq is falling back to the Lloyd-Max 3-bit quantizer ({Markup.Escape(fallbackReason)}). " +
                    $"{Markup.Escape(TqSupport.QualityWarningReason)}; " +
                    "pass [yellow]--tq-mode lloydmax[/] explicitly to silence this warning.");
        }

        // Same head-dim compatibility rules as the run command (issue #180): Lloyd-Max
        // ships hardcoded codebooks for 128/256; KVarN accepts any power-of-2 in
        // [8, 1024] on CPU, [8, 256] on CUDA (the shared-memory WHT cap).
        if (settings.TurboQuant)
        {
            int headDim = hp.HeadDim;
            if (quantizer == TqQuantizer.KVarN)
            {
                if (!TqSupport.IsKVarNHeadDim(headDim))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] --tq-mode kvarn requires a power-of-2 head dimension in [[8, 1024]]; this model has head dim {headDim}.");
                    return 1;
                }
                if (settings.NGpuLayers == -1 && headDim > TqSupport.KVarNCudaMaxHeadDim)
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] --tq-mode kvarn on CUDA requires head dim ≤ 256 (shared-memory WHT cap); this model has head dim {headDim}. Use [yellow]-g 0[/].");
                    return 1;
                }
            }
            else if (!TqSupport.IsLloydMaxHeadDim(headDim))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] TurboQuant requires head dimension 128 or 256; this model has head dim {headDim}.");
                return 1;
            }
        }
        if (settings.NGpuLayers == -1 && settings.TurboQuant && quantizer == TqQuantizer.KVarN && hp.IsMoE)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] --tq-mode kvarn on CUDA supports dense models only (issue #180 Task 5a); use [yellow]-g 0[/] for MoE.");
            return 1;
        }

        // Raw-text tokenization (no chat template), like llama.cpp perplexity. BOS is
        // prepended for models whose metadata asks for it (add_bos_token=true), mirroring
        // the run command's SHARPI_RAW_PROMPT path.
        var tokens = tokenizer.Encode(text);
        if (tokenizer.AddBosToken && tokenizer.BosTokenId >= 0
            && (tokens.Count == 0 || tokens[0] != tokenizer.BosTokenId))
        {
            var withBos = new List<int>(tokens.Count + 1) { tokenizer.BosTokenId };
            withBos.AddRange(tokens);
            tokens = withBos;
        }

        int ctx = Math.Min(settings.CtxSize, hp.ContextLength);
        if (ctx < settings.CtxSize)
            AnsiConsole.MarkupLine($"[yellow]Note:[/] -c {settings.CtxSize} clamped to the model context length ({hp.ContextLength}).");
        int n = Math.Min(tokens.Count, ctx);
        if (n < 2)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] corpus tokenized to {tokens.Count} token(s); need at least 2.");
            return 1;
        }
        if (tokens.Count < ctx)
            AnsiConsole.MarkupLine($"[yellow]Note:[/] corpus has only {tokens.Count} tokens (< -c {ctx}); evaluating all of them.");

        string config = !settings.TurboQuant ? "fp32"
            : quantizer == TqQuantizer.KVarN ? $"tq-kvarn-k4v2 (window={settings.TqWindow})"
            : $"tq-lloydmax-3bit (window={settings.TqWindow})";

        if (settings.TurboQuant && settings.TqWindow >= n)
            AnsiConsole.MarkupLine($"[yellow]Note:[/] --tq-window {settings.TqWindow} >= evaluated tokens ({n}); the cache never compresses, so this measures the FP32 path.");

        // Build the forward pass: CPU ForwardPass (-g 0) or the full-CUDA-offload
        // CudaForwardPass (-g -1, issue #180 Task 5a). Both feed the same
        // teacher-forced token-by-token Forward loop below, so with --tq every
        // step exercises the compressed-read attention path exactly like decode.
        CpuBackend? cpuBackend = null;
        CudaBackend? cudaBackend = null;
        IForwardPass fwd;
        string backendLabel;
        if (settings.NGpuLayers == -1)
        {
            if (!CudaBackend.IsAvailable())
            {
                AnsiConsole.MarkupLine("[red]Error:[/] -g -1 requires a CUDA device; none is available. Use [yellow]-g 0[/] for the CPU path.");
                return 1;
            }
            if (hp.IsHybridSsm)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] -g -1 perplexity uses CudaForwardPass, which does not cover hybrid GDN models; use [yellow]-g 0[/].");
                return 1;
            }
            cudaBackend = CudaBackend.Create();
            try
            {
                fwd = new CudaForwardPass(model, cudaBackend, hp, maxContextLength: ctx,
                    enableTurboQuant: settings.TurboQuant, tqFp32Window: settings.TqWindow,
                    tqQuantizer: quantizer);
            }
            catch
            {
                cudaBackend.Dispose();
                throw;
            }
            backendLabel = $"[green]CUDA[/] ({cudaBackend.Name}, all {hp.NumLayers} layers)";
        }
        else
        {
            cpuBackend = new CpuBackend();
            var cpuFwd = new ForwardPass(model, cpuBackend, hp, maxContextLength: ctx);
            if (settings.TurboQuant)
                cpuFwd.EnableTurboQuant(fp32WindowSize: settings.TqWindow, bits: 3, quantizer: quantizer);
            fwd = cpuFwd;
            backendLabel = "[blue]CPU[/]";
        }
        // Declared backend-first so `using` disposal (reverse order) tears down the
        // forward pass before its backend.
        using var _cpuDispose = cpuBackend;
        using var _cudaDispose = cudaBackend;
        using var _fwdDispose = (IDisposable)fwd;

        AnsiConsole.MarkupLine($"[dim]Backend: {backendLabel} | config: {Markup.Escape(config)} | evaluating {n} tokens[/]");

        // Position buckets over the *target* position: [1, edge0) is context that always
        // fits the FP32 window, [edge0, edge1) early compressed region, [edge1, +) deep.
        int edge0 = settings.TqWindow;
        int edge1 = Math.Max(1024, 2 * edge0);
        Span<double> bucketNll = stackalloc double[3];
        Span<int> bucketCount = stackalloc int[3];
        double totalNll = 0.0;
        int scored = 0, nonFinite = 0;

        var sw = Stopwatch.StartNew();
        for (int pos = 0; pos < n - 1; pos++)
        {
            cancellation.ThrowIfCancellationRequested();
            var logits = fwd.Forward(tokens[pos], pos);
            int targetPos = pos + 1;
            double nll = NegativeLogLikelihood(logits, tokens[targetPos]);
            if (!double.IsFinite(nll))
            {
                nonFinite++;
            }
            else
            {
                int b = targetPos < edge0 ? 0 : targetPos < edge1 ? 1 : 2;
                bucketNll[b] += nll;
                bucketCount[b]++;
                totalNll += nll;
                scored++;
            }

            if (targetPos % 256 == 0 && scored > 0)
            {
                double tps = targetPos / sw.Elapsed.TotalSeconds;
                AnsiConsole.MarkupLine($"[dim]  pos {targetPos}/{n - 1}  running ppl {Math.Exp(totalNll / scored):F4}  ({tps:F1} tok/s)[/]");
            }
        }
        double elapsedS = sw.Elapsed.TotalSeconds;

        if (scored == 0)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] no positions produced finite NLL — the forward pass emitted NaN/Inf logits throughout.");
            return 1;
        }

        double meanNll = totalNll / scored;
        Console.WriteLine();
        Console.WriteLine($"perplexity: model={Path.GetFileName(modelPath)} file={Path.GetFileName(textFile)} config={config}");
        Console.WriteLine($"tokens scored: {scored} (targets at positions 1..{n - 1}; first position skipped)");
        Console.WriteLine($"mean NLL: {meanNll:F6}   perplexity: {Math.Exp(meanNll):F4}");
        WriteBucket($"[1,{edge0})", bucketNll[0], bucketCount[0]);
        WriteBucket($"[{edge0},{edge1})", bucketNll[1], bucketCount[1]);
        WriteBucket($"[{edge1},+)", bucketNll[2], bucketCount[2]);
        if (nonFinite > 0)
            AnsiConsole.MarkupLine($"[red]non-finite NLL steps (excluded from the mean): {nonFinite}[/]");
        Console.WriteLine($"elapsed: {elapsedS:F1}s ({(n - 1) / elapsedS:F2} tok/s)");
        return 0;

        static void WriteBucket(string range, double nllSum, int count)
        {
            Console.WriteLine(count == 0
                ? $"bucket {range}: count=0"
                : $"bucket {range}: count={count}  mean NLL={nllSum / count:F6}  ppl={Math.Exp(nllSum / count):F4}");
        }
    }
}
