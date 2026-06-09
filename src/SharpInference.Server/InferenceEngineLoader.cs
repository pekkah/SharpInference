using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.Vulkan;

namespace SharpInference.Server;

/// <summary>
/// Default <see cref="SharpInferenceServerOptions.EngineFactory"/> implementation: reads
/// <see cref="SharpInferenceServerOptions.ModelPath"/>, opens the GGUF file, picks the
/// fastest forward-pass implementation that satisfies the configured backend + layer
/// budget, and wraps it in either <see cref="InferenceEngine"/> or
/// <see cref="ContinuousBatchingEngine"/>.
///
/// <para>
/// The branching mirrors <c>SharpInference.Cli</c>'s <c>RunCommand</c> backend-selection
/// logic so an operator can express the same tuning in <c>appsettings.json</c> that they
/// would on the command line.
/// </para>
/// </summary>
public static class InferenceEngineLoader
{
    /// <summary>
    /// Opens the GGUF file referenced by <paramref name="opts"/> and constructs the
    /// inference engine. Throws <see cref="InvalidOperationException"/> when the model
    /// file cannot be located or the requested configuration is unsupported.
    /// </summary>
    public static LoadedEngine Load(SharpInferenceServerOptions opts)
    {
        // ── 0. Translate the MoE env-var-backed knobs BEFORE building the forward pass.
        // WarmPinConfig / HybridForwardPass / slot-manager constructors read these once at
        // load time, so they have to be in the environment by the time GgufModel.Open
        // chains into ForwardPass construction below.
        ApplyMoeEnvironment(opts);

        // ── 1. Apply the SIMD/BLAS crossover threshold (overrides SHARPI_MIN_BATCH_BLAS).
        if (opts.MinBatchBlas > 0)
            SimdKernels.MinBatchForBlas = opts.MinBatchBlas;

        // ── 2. Resolve & open the model.
        var modelPath = ResolveModelPath(opts.ModelPath);
        var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var arch = model.Metadata.TryGetValue("general.architecture", out var a)
            ? (string)a
            : opts.Architecture;
        var modelId = Path.GetFileNameWithoutExtension(modelPath);
        var (thinkTokenId, endThinkTokenId) = ResolveReasoningTokens(tokenizer);

        // ── 3. Validate TurboQuant up-front so a mis-shaped request fails fast (and not
        // after the model has already been mmap'd into VRAM).
        bool turboQuant = opts.TurboQuant;
        if (turboQuant && hp.IsHybridSsm)
            throw new InvalidOperationException(
                "TurboQuant is not supported for hybrid GDN models (no KV cache on GDN layers).");
        if (turboQuant && hp.HeadDim is not 128 and not 256)
            throw new InvalidOperationException(
                $"TurboQuant requires head dimension 128 or 256; this model has head dim {hp.HeadDim}.");

        int ctxSize = opts.ContextSize;
        int nGpuLayers = opts.NGpuLayers;

        // ── 4. Build the forward pass. The owned[] list collects everything the engine
        // must dispose on shutdown (backends, the forward pass itself, the GGUF handle).
        var owned = new List<IDisposable>();
        IForwardPass fwd;
        bool batchingSupported;

        try
        {
            (fwd, batchingSupported) = BuildForwardPass(model, hp, arch, ctxSize, nGpuLayers, opts.Backend, turboQuant, owned);
            owned.Add(model);
        }
        catch
        {
            foreach (var d in owned) try { d.Dispose(); } catch { /* fall through to rethrow */ }
            model.Dispose();
            throw;
        }

        // ── 5. Wrap in the right engine. ContinuousBatchingEngine takes the concrete
        // ForwardPass — it isn't built for the GPU / hybrid paths — so we honour
        // MaxBatchSize > 1 only when batching is structurally possible.
        IInferenceEngine engine;
        if (opts.MaxBatchSize > 1 && batchingSupported && fwd is ForwardPass cpuFwd)
        {
            long kvTokenBudget = ResolveKvTokenBudget(cpuFwd);
            engine = new ContinuousBatchingEngine(cpuFwd, tokenizer, modelId, opts.MaxBatchSize,
                thinkTokenId, endThinkTokenId, kvTokenBudget);
            // ContinuousBatchingEngine doesn't accept owned[] disposables; transfer
            // disposal responsibility by wrapping it in a composite disposable.
            engine = new OwnedDisposableEngine(engine, owned);
        }
        else
        {
            engine = new InferenceEngine(fwd, tokenizer, modelId, thinkTokenId, endThinkTokenId,
                owned.ToArray());
        }

        return new LoadedEngine(engine, arch, tokenizer.ChatTemplate);
    }

    /// <summary>
    /// Resolve the KV-cache admission budget (in token positions) for
    /// <see cref="ContinuousBatchingEngine"/>. <c>SHARPI_KV_TOKEN_BUDGET</c> wins when set:
    /// a positive value caps the live KV footprint; <c>0</c> disables the budget (unlimited).
    /// Absent the env var, derive a budget from available RAM and the model's per-token KV
    /// cost so a burst of long-prompt requests is throttled instead of OOM-ing the host.
    /// </summary>
    private static long ResolveKvTokenBudget(ForwardPass cpuFwd)
    {
        if (Environment.GetEnvironmentVariable("SHARPI_KV_TOKEN_BUDGET") is { } raw
            && long.TryParse(raw, out long explicitBudget) && explicitBudget >= 0)
        {
            return explicitBudget;
        }

        var hw = HardwareProfile.Detect();
        return hw.EstimateKvTokenBudget(cpuFwd.KvBytesPerToken);
    }

    // ── Backend dispatch ─────────────────────────────────────────────────────

    private static (IForwardPass Fwd, bool BatchingSupported) BuildForwardPass(
        GgufModel model, ModelHyperparams hp, string arch, int ctxSize, int nGpuLayers,
        ServerBackend backend, bool turboQuant, List<IDisposable> owned)
    {
        // Resolve "auto" first so the rest of the method can treat backend as concrete.
        if (nGpuLayers != 0 && backend == ServerBackend.Auto)
        {
            bool tqOk = !turboQuant || hp.HeadDim is 128 or 256;
            if (tqOk && CudaBackend.IsAvailable())
                backend = ServerBackend.Cuda;
            else
                backend = ServerBackend.Vulkan;
        }
        else if (nGpuLayers == 0 || backend == ServerBackend.Cpu)
        {
            backend = ServerBackend.Cpu;
        }

        var cpuBackend = new CpuBackend();
        owned.Add(cpuBackend);

        // CPU path: covers hybrid GDN (HybridGdnForwardPass) and dense (ForwardPass).
        if (backend == ServerBackend.Cpu)
        {
            if (hp.IsHybridSsm)
            {
                var hybrid = new HybridGdnForwardPass(model, cpuBackend, hp);
                owned.Add(hybrid);
                return (hybrid, BatchingSupported: false);
            }

            var dense = new ForwardPass(model, cpuBackend, hp);
            owned.Add(dense);
            if (turboQuant)
                dense.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
            // ContinuousBatchingEngine doesn't yet support MoE or TurboQuant fan-out.
            bool batchOk = !hp.IsMoE && !turboQuant;
            return (dense, batchOk);
        }

        // GPU paths share a CPU baseline (for the hybrid-CPU half of partial offload).
        // For full GPU paths the dense CPU fwd is unused but still cheap to construct.
        ForwardPass? cpuDense = hp.IsHybridSsm ? null : new ForwardPass(model, cpuBackend, hp);
        if (cpuDense is not null) owned.Add(cpuDense);

        if (backend == ServerBackend.Cuda)
        {
            var cuda = CudaBackend.Create();
            owned.Add(cuda);

            if (hp.IsHybridSsm)
            {
                // Layer placement for hybrid GDN is driven by hp.LayerTypes, not VRAM
                // budget — so TierPlanner is skipped and we always claim "all layers
                // on GPU" (the GDN/MoE routing is implicit per-layer).
                var placement = new LayerPlacement(
                    GpuLayers: hp.NumLayers,
                    CpuLayers: 0,
                    GpuWeightBytes: 0,
                    GpuKvBytes: 0,
                    RecommendedCtxSize: ctxSize > 0 ? ctxSize : Math.Min(hp.ContextLength, 4096));
                var chgdn = new CudaHybridGdnForwardPass(model, cuda, hp, placement);
                owned.Add(chgdn);
                return (chgdn, BatchingSupported: false);
            }

            // Dense + CUDA: ask TierPlanner for a layer count when -1, then route to
            // full-offload / hybrid / CPU based on what we got back.
            var hwProfile = HardwareProfile.Detect(cuda);
            int gpuLayers = nGpuLayers == -1
                ? TierPlanner.Plan(model, hp, hwProfile, turboQuant, requestedCtxSize: ctxSize).GpuLayers
                : nGpuLayers;
            if (nGpuLayers == -1)
                gpuLayers = ClampGemma4KvShareBoundary(hp, gpuLayers);

            if (gpuLayers <= 0)
            {
                // GPU planner says nothing fits — fall back to CPU dense.
                if (turboQuant) cpuDense!.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
                return (cpuDense!, BatchingSupported: !hp.IsMoE && !turboQuant);
            }

            if (gpuLayers >= hp.NumLayers)
            {
                var cfwd = new CudaForwardPass(model, cuda, hp, ctxSize, enableTurboQuant: turboQuant);
                owned.Add(cfwd);
                return (cfwd, BatchingSupported: false);
            }

            var planForHybrid = TierPlanner.Plan(model, hp, hwProfile, turboQuant, requestedCtxSize: ctxSize)
                with { GpuLayers = gpuLayers, CpuLayers = hp.NumLayers - gpuLayers };
            var chfwd = new CudaHybridForwardPass(model, cuda, hp, planForHybrid, turboQuant);
            owned.Add(chfwd);
            return (chfwd, BatchingSupported: false);
        }

        if (backend == ServerBackend.Vulkan)
        {
            if (hp.IsHybridSsm)
                throw new InvalidOperationException(
                    "Hybrid GDN models (qwen35moe) are not supported on the Vulkan backend. " +
                    "Set Backend=Cuda or NGpuLayers=0.");

            var vulkan = new VulkanBackend();
            owned.Add(vulkan);

            var hwProfile = HardwareProfile.Detect(vulkan);
            int gpuLayers = nGpuLayers == -1
                ? TierPlanner.Plan(model, hp, hwProfile, turboQuant, requestedCtxSize: ctxSize).GpuLayers
                : nGpuLayers;

            if (gpuLayers <= 0)
            {
                if (turboQuant) cpuDense!.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
                return (cpuDense!, BatchingSupported: !hp.IsMoE && !turboQuant);
            }

            if (gpuLayers >= hp.NumLayers)
            {
                var gfwd = new GpuForwardPass(model, vulkan, hp, ctxSize, enableTurboQuant: turboQuant);
                owned.Add(gfwd);
                return (gfwd, BatchingSupported: false);
            }

            var planForHybrid = TierPlanner.Plan(model, hp, hwProfile, turboQuant, requestedCtxSize: ctxSize)
                with { GpuLayers = gpuLayers, CpuLayers = hp.NumLayers - gpuLayers };
            var hfwd = new HybridForwardPass(model, vulkan, hp, planForHybrid, turboQuant);
            owned.Add(hfwd);
            return (hfwd, BatchingSupported: false);
        }

        throw new InvalidOperationException($"Unknown backend selection: {backend}");
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Sets the SHARPI_MOE_* environment variables from the options object so the engine's
    /// MoE-cache code picks them up at construction time. The engine code reads these
    /// once per backend instance, so we have to do this BEFORE the backend is built.
    /// </summary>
    private static void ApplyMoeEnvironment(SharpInferenceServerOptions opts)
    {
        if (opts.MoeWarmPin is int wp)
            Environment.SetEnvironmentVariable("SHARPI_MOE_WARMPIN", wp.ToString(System.Globalization.CultureInfo.InvariantCulture));
        if (opts.MoeWarmPinAfter > 0)
            Environment.SetEnvironmentVariable("SHARPI_MOE_WARMPIN_AFTER", opts.MoeWarmPinAfter.ToString(System.Globalization.CultureInfo.InvariantCulture));
        if (!opts.MoePredictPrefetch)
            Environment.SetEnvironmentVariable("SHARPI_MOE_PREDICT_PREFETCH", "0");
        if (!string.IsNullOrEmpty(opts.ExpertStatsPath))
            Environment.SetEnvironmentVariable("SHARPI_EXPERT_STATS", opts.ExpertStatsPath);
    }

    /// <summary>
    /// Resolves the open/close special-token IDs that bracket a model's reasoning stream so
    /// the engine can split it into a separate <c>Thinking</c> channel. Tries the ChatML
    /// <c>&lt;think&gt;</c>/<c>&lt;/think&gt;</c> convention first, then Gemma 4's
    /// <c>&lt;|channel&gt;</c>/<c>&lt;channel|&gt;</c> "thought" channel. Both IDs must be
    /// positive — id 0 is usually <c>&lt;pad&gt;</c>/<c>&lt;unk&gt;</c> and would mis-trigger —
    /// and both must be present. No match leaves reasoning-stream splitting disabled (the
    /// engine emits Text chunks only), which is why a Gemma model would otherwise leak raw
    /// <c>&lt;|channel&gt;thought…&lt;channel|&gt;</c> markers into the assistant content.
    /// </summary>
    private static (int thinkTokenId, int endThinkTokenId) ResolveReasoningTokens(GgufTokenizer tokenizer)
    {
        if (tokenizer.SpecialTokens.TryGetValue("<think>", out int tid)
            && tokenizer.SpecialTokens.TryGetValue("</think>", out int eid)
            && tid > 0 && eid > 0)
            return (tid, eid);
        // Gemma 4 wraps reasoning in <|channel>thought … <channel|> (single special tokens).
        // The template strips every channel block from history, so treating the whole block
        // as reasoning matches the model's own content/thought split.
        if (tokenizer.SpecialTokens.TryGetValue("<|channel>", out int cid)
            && tokenizer.SpecialTokens.TryGetValue("<channel|>", out int ceid)
            && cid > 0 && ceid > 0)
            return (cid, ceid);
        return (-1, -1);
    }

    /// <summary>
    /// Gemma 4 KV-share constraint: the shared-KV source layers must live on the same tier as
    /// their dependent shared-KV tail because cross-tier KV reads aren't wired (see
    /// <c>CudaHybridForwardPass.Gemma4ValidateHybridSplit</c>). <see cref="TierPlanner"/> doesn't
    /// model this and can return a layer count that straddles the boundary (e.g. 30 on E4B Q8 /
    /// 12 GB). When the auto value lands in the forbidden band, promote to full offload — the
    /// planner's per-layer KV budget is pessimistic for shared-KV layers (they alias source
    /// pages instead of growing their own cache), so full offload almost always fits once the
    /// auto value already cleared the safe max. Mirrors the <c>SharpInference.Cli</c> RunCommand
    /// fix (#82); only applied on the auto (<c>-g -1</c>) path so an explicit <c>-g</c> is honoured.
    /// </summary>
    private static int ClampGemma4KvShareBoundary(ModelHyperparams hp, int gpuLayers)
    {
        if (hp.KvSourceLayer is not { } ksl) return gpuLayers;

        int minSrc = int.MaxValue;
        for (int i = 0; i < hp.NumLayers; i++)
            if (ksl[i] >= 0 && ksl[i] < minSrc) minSrc = ksl[i];

        if (minSrc != int.MaxValue && gpuLayers > minSrc && gpuLayers < hp.NumLayers)
        {
            Console.Error.WriteLine(
                $"[SharpInference] TierPlanner returned -g {gpuLayers}, which would cross the " +
                $"Gemma 4 KV-share boundary (sources <= {minSrc}); promoting to full offload " +
                $"(-g {hp.NumLayers}). Set NGpuLayers={minSrc} explicitly if VRAM is tight.");
            return hp.NumLayers;
        }
        return gpuLayers;
    }

    /// <summary>
    /// Resolves a possibly-relative path against the CWD, the entry-assembly directory, and
    /// a handful of parent directories so <c>SHARPI_MODEL=models/foo.gguf</c> works whether
    /// the process was launched from the repo root, the project directory (as
    /// <c>dotnet run --project</c> sets it), or a published-binary directory.
    /// </summary>
    private static string ResolveModelPath(string? modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new InvalidOperationException(
                "SharpInferenceServerOptions.ModelPath is required. " +
                "Set it via Configure(o => o.ModelPath = ...) or the SHARPI_MODEL environment variable.");

        if (Path.IsPathRooted(modelPath) && File.Exists(modelPath))
            return modelPath;

        if (File.Exists(modelPath))
            return Path.GetFullPath(modelPath);

        var candidates = new List<string>
        {
            Path.Combine(Directory.GetCurrentDirectory(), modelPath),
            Path.Combine(AppContext.BaseDirectory, modelPath),
        };
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        for (int i = 0; i < 5 && dir is not null; i++, dir = dir.Parent)
            candidates.Add(Path.Combine(dir.FullName, modelPath));

        var resolved = candidates.FirstOrDefault(File.Exists);
        if (resolved is not null) return resolved;

        throw new InvalidOperationException(
            $"Model file not found: '{modelPath}'. " +
            "Set SharpInferenceServerOptions.ModelPath, the SHARPI_MODEL environment variable, " +
            "or the SharpInference:ModelPath configuration key.");
    }
}

/// <summary>
/// Composite engine wrapper that disposes a list of <see cref="IDisposable"/> resources
/// when the inner engine is itself disposed. Used by the loader to attach ownership of
/// the backend, forward pass, and GGUF handle to a <see cref="ContinuousBatchingEngine"/>,
/// which doesn't accept an <c>owned[]</c> array in its constructor.
/// </summary>
internal sealed class OwnedDisposableEngine(IInferenceEngine inner, IList<IDisposable> owned)
    : IInferenceEngine, IDisposable
{
    public string ModelId             => inner.ModelId;
    public int QueueDepth             => inner.QueueDepth;
    public int ActiveRequests         => inner.ActiveRequests;
    public bool PrefixCacheEnabled    => inner.PrefixCacheEnabled;
    public long PrefillTokensReused   => inner.PrefillTokensReused;

    public IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt, SamplingParams sp, CancellationToken ct = default, string? canonicalHistoryPrefix = null)
        => inner.GenerateChunksAsync(prompt, sp, ct, canonicalHistoryPrefix);

    public void Dispose()
    {
        (inner as IDisposable)?.Dispose();
        for (int i = owned.Count - 1; i >= 0; i--)
        {
            try { owned[i].Dispose(); } catch { /* best-effort teardown */ }
        }
    }
}
