using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.Vision;
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

        // KV-cache dtype (#179): the CUDA dense forward pass reads SHARPI_KV_DTYPE in its
        // constructor, so translate the option into the environment before BuildForwardPass.
        // Only set when explicitly configured — leave an externally-set env var alone so
        // SHARPI_KV_DTYPE-only operation keeps working. CudaForwardPass validates the value.
        if (!string.IsNullOrWhiteSpace(opts.KvType))
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", opts.KvType);

        // ── 2. Resolve & open the model.
        var modelPath = ResolvePath(opts.ModelPath, "model", "SHARPI_MODEL", "ModelPath");
        var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var arch = model.Metadata.TryGetValue("general.architecture", out var a)
            ? (string)a
            : opts.Architecture;
        var modelId = Path.GetFileNameWithoutExtension(modelPath);
        var (thinkTokenId, endThinkTokenId) = tokenizer.ReasoningTokens;

        // Tool-boundary stop tokens for agentic loops (issue #304): resolve the architecture's
        // adapter markers (Gemma 4: <|tool_response>) against the vocab. The chat endpoints add
        // these to the stop set on tool-active requests so the model halts the instant it finishes
        // its tool calls instead of opening a hallucinated trailing turn.
        var toolBoundaryMarkers = ToolCallAdapterRegistry.Get(arch).ToolBoundaryStopMarkers;
        var toolBoundaryStopTokenIds = toolBoundaryMarkers
            .Select(m => tokenizer.SpecialTokens.TryGetValue(m, out int id) ? id : -1)
            .Where(id => id > 0)
            .Distinct()
            .ToArray();
        // The adapter declared it needs a tool-boundary stop but none resolved to a vocab id —
        // agentic generation will run past the tool calls (the #304 symptom) with no other clue.
        // Warn rather than fail silently (mirrors the image-input diagnostic below).
        if (toolBoundaryMarkers.Count > 0 && toolBoundaryStopTokenIds.Length == 0)
            Console.Error.WriteLine(
                $"[SharpInference] {arch} declares tool-boundary stop markers [{string.Join(", ", toolBoundaryMarkers)}] " +
                "that were not found in this model's vocab; agentic tool-call generation may run past " +
                "the tool calls (issue #304).");

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
            (fwd, batchingSupported) = BuildForwardPass(model, hp, arch, ctxSize, nGpuLayers, opts.Backend, turboQuant, owned,
                DequantCacheBytes(opts.PrefillDequantCacheMb), preferBatchingOverAutoSnapKv: opts.MaxBatchSize > 1);
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
        // MaxBatchSize > 1 only when batching is structurally possible. Mirror the
        // BuildForwardPass guard above: if an engine constructor throws, dispose everything
        // in owned[] (the backend, the multi-GB forward pass, the GGUF handle) rather than
        // leaking it — ownership only transfers once construction succeeds.
        IInferenceEngine engine;
        try
        {
            // ── Image input (issue #253): open the mmproj vision projector when configured.
            // Requires an embedding-capable forward pass (CPU / full-CUDA Gemma 4) and the
            // single-user InferenceEngine path; reject other configs with a clear error.
            GemmaUvVisionEmbedder? visionEmbedder = null;
            VisionModel? visionModel = null;
            (int Open, int Close, int Placeholder) imgIds = default;
            if (!string.IsNullOrWhiteSpace(opts.MmprojPath))
            {
                // The gemma4uv splice path is specific to Gemma 4 text models. The CPU forward
                // pass reports SupportsEmbeddingInput=true for every architecture, so without this
                // arch check a non-Gemma CPU model + a valid gemma4uv mmproj would load and either
                // splice foreign soft tokens into the wrong trunk (garbage) or throw an opaque
                // dimension error mid-request. Fail fast at load instead.
                if (!string.Equals(arch, "gemma4", StringComparison.Ordinal))
                    throw new InvalidOperationException(
                        "Image input (MmprojPath / SHARPI_MMPROJ) is only supported for Gemma 4 (gemma4uv) " +
                        $"text models; this model's architecture is '{arch}'.");
                if (!fwd.SupportsEmbeddingInput)
                    throw new InvalidOperationException(
                        "MmprojPath / SHARPI_MMPROJ is set but image input requires a forward pass that accepts " +
                        "precomputed-embedding input: CPU (NGpuLayers=0) or full CUDA offload (NGpuLayers=-1) of a " +
                        $"Gemma 4 model that fits VRAM. The configured pass ({fwd.GetType().Name}) does not support it.");
                if (opts.MaxBatchSize > 1 && batchingSupported && fwd is IBatchedForwardPass)
                    throw new InvalidOperationException(
                        "Image input is not supported with continuous batching (MaxBatchSize > 1). Set MaxBatchSize=1.");

                var mmprojPath = ResolvePath(opts.MmprojPath, "mmproj projector", "SHARPI_MMPROJ", "MmprojPath");
                visionModel = VisionModel.Open(mmprojPath); // validates clip / gemma4uv projector, else throws
                owned.Add(visionModel);
                visionEmbedder = new GemmaUvVisionEmbedder(visionModel);
                imgIds = (
                    tokenizer.SpecialTokens.TryGetValue("<|image>", out var o) ? o : 255999,
                    tokenizer.SpecialTokens.TryGetValue("<image|>", out var c) ? c : 258882,
                    tokenizer.SpecialTokens.TryGetValue("<|image|>", out var p) ? p : 258880);
            }

            string? dsparkPath = !string.IsNullOrWhiteSpace(opts.DSparkModelPath)
                ? opts.DSparkModelPath
                : Environment.GetEnvironmentVariable("SHARPI_DSPARK_MODEL");

            if (opts.MaxBatchSize > 1 && batchingSupported && fwd is IBatchedForwardPass batchFwd)
            {
                if (!string.IsNullOrWhiteSpace(dsparkPath))
                    throw new InvalidOperationException(
                        "DSpark (DSparkModelPath / SHARPI_DSPARK_MODEL) is not supported with " +
                        "continuous batching (MaxBatchSize > 1) — the tap buffer is " +
                        "single-sequence (docs/dspark-plan.md Phase 6). Set MaxBatchSize=1.");
                engine = new ContinuousBatchingEngine(batchFwd, tokenizer, modelId, opts.MaxBatchSize,
                    thinkTokenId, endThinkTokenId,
                    prefillChunkTokens: opts.PrefillChunkTokens,
                    kvBudgetBytes: opts.KvBudgetMb > 0 ? opts.KvBudgetMb * 1024 * 1024 : opts.KvBudgetMb);
                // ContinuousBatchingEngine doesn't accept owned[] disposables; transfer
                // disposal responsibility by wrapping it in a composite disposable.
                engine = new OwnedDisposableEngine(engine, owned);
            }
            else
            {
                var ie = new InferenceEngine(fwd, tokenizer, modelId, thinkTokenId, endThinkTokenId,
                    owned.ToArray());
                if (visionEmbedder is not null)
                    ie.EnableImageInput(visionEmbedder, visionModel!, imgIds.Open, imgIds.Close, imgIds.Placeholder);
                if (!string.IsNullOrWhiteSpace(dsparkPath))
                {
                    try
                    {
                        AttachDSpark(ie, fwd, model, hp, owned, opts, dsparkPath, ctxSize);
                    }
                    catch
                    {
                        // The engine already owns fwd + owned[] and runs a background
                        // worker thread; dispose IT (which tears all of that down) and
                        // clear the list so the outer catch can't double-dispose.
                        ie.Dispose();
                        owned.Clear();
                        throw;
                    }
                }
                engine = ie;
            }
        }
        catch
        {
            foreach (var d in owned) try { d.Dispose(); } catch { /* fall through to rethrow */ }
            throw;
        }

        // Grammar-constrained tool-call decoding (issue #374): expose a vocabulary view built from
        // the model's tokenizer. The full-vocab byte table inside it is materialised lazily on first
        // constrained request, so this costs nothing unless tool-grammar is actually used.
        var grammarVocab = new SharpInference.Core.Grammar.GrammarVocabulary(tokenizer);

        return new LoadedEngine(engine, arch, tokenizer.ChatTemplate, toolBoundaryStopTokenIds, grammarVocab);
    }

    // ── DSpark draft head (docs/dspark-plan.md Phase 6, PR #413) ─────────────

    /// <summary>
    /// Load the configured DSpark draft head and attach it to the single-user engine:
    /// resolve the safetensors + sibling config.json, validate head↔target compatibility
    /// and tap support, run the placement planner (GPU draft on the target's CudaBackend,
    /// CPU draft otherwise), enable hidden taps BEFORE any request runs, and hand
    /// ownership of the draft to the engine. Unlike the CLI (which falls back to normal
    /// generation), an explicitly configured server head that can't be honored throws —
    /// silent degradation at startup would misreport the deployment's capabilities.
    /// </summary>
    private static void AttachDSpark(InferenceEngine ie, IForwardPass fwd, GgufModel model,
        ModelHyperparams hp, List<IDisposable> owned, SharpInferenceServerOptions opts,
        string configuredPath, int ctxSize)
    {
        string stPath = configuredPath;
        if (Directory.Exists(stPath)) stPath = Path.Combine(stPath, "model.safetensors");
        if (!File.Exists(stPath))
            throw new FileNotFoundException($"DSpark model not found: {stPath}");
        string cfgPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(stPath))!, "config.json");
        if (!File.Exists(cfgPath))
            throw new FileNotFoundException($"DSpark config.json not found next to the safetensors: {cfgPath}");

        var cfg = DSparkConfig.FromJsonFile(cfgPath);
        if (cfg.VocabSize != hp.VocabSize || cfg.NumTargetLayers != hp.NumLayers
            || cfg.HiddenSize != hp.EmbeddingDim)
            throw new InvalidOperationException(
                $"DSpark head/target mismatch — head expects vocab {cfg.VocabSize}, " +
                $"{cfg.NumTargetLayers} target layers, hidden {cfg.HiddenSize}; target has " +
                $"vocab {hp.VocabSize}, {hp.NumLayers} layers, hidden {hp.EmbeddingDim}.");
        if (!fwd.SupportsHiddenTaps)
            throw new InvalidOperationException(
                "DSpark requires a tap-capable dense forward pass (CPU, NGpuLayers=0, or full " +
                "CUDA offload, NGpuLayers=-1; no MoE / Gemma-4 / TurboQuant / SnapKV). " +
                $"The configured pass ({fwd.GetType().Name}) can't capture hidden taps.");

        // The GPU draft shares the TARGET's CudaBackend (one stream orders the tap
        // producer and draft consumer); only meaningful for the dense CUDA pass.
        CudaBackend? cuda = null;
        if (fwd is CudaForwardPass)
            foreach (var d in owned)
                if (d is CudaBackend cb) { cuda = cb; break; }

        var userPlace = !string.IsNullOrWhiteSpace(opts.DSparkPlace)
            ? DSparkPlacementPlanner.ParsePlacement(opts.DSparkPlace)
            : DSparkPlacementPlanner.ResolvePlacement(null);
        var hwProfile = cuda is not null ? HardwareProfile.Detect(cuda) : HardwareProfile.Detect();
        var targetPlacement = TierPlanner.Plan(model, hp, hwProfile, requestedCtxSize: ctxSize);
        long headBytesGpu = CudaDSparkDraftModel.EstimateGpuResidentBytes(cfg);
        long headBytesCpu = DSparkDraftModel.EstimateResidentBytes(cfg);
        long tapBytes = (long)targetPlacement.RecommendedCtxSize * cfg.TapDim * sizeof(float);
        var decision = DSparkPlacementPlanner.Plan(
            hwProfile, targetPlacement, headBytesGpu, headBytesCpu, userPlace,
            hostTapBytes: tapBytes);

        if (decision.Placement == DSparkPlacement.Gpu && cuda is null)
        {
            // Gpu → Cpu → Off graceful fallback: re-plan in Auto over a GPU-less
            // profile so the RAM budget is actually checked.
            decision = DSparkPlacementPlanner.Plan(
                hwProfile with { VramBytes = 0 }, targetPlacement,
                headBytesGpu, headBytesCpu, DSparkPlacement.Auto,
                hostTapBytes: tapBytes);
        }
        Console.Error.WriteLine($"[InferenceEngine] DSpark placement: {decision.Placement} — {decision.Reason}");
        if (decision.Placement == DSparkPlacement.Off)
            throw new InvalidOperationException(
                $"DSpark was configured (SHARPI_DSPARK_MODEL / DSparkModelPath) but placement " +
                $"resolved to Off — {decision.Reason}. Free resources, pass DSparkPlace=cpu/gpu " +
                "explicitly, or unset the head.");

        fwd.EnableHiddenTaps(cfg.TargetLayerIds);
        using var st = SafetensorsLoader.Open(stPath);
        IDSparkDraft draft = decision.Placement == DSparkPlacement.Gpu
            ? new CudaDSparkDraftModel(cfg, st, cuda!, fwd.MaxSeqLen)
            : new DSparkDraftModel(cfg, st, fwd.MaxSeqLen);
        try
        {
            ie.AttachDSparkDraft(draft);
        }
        catch
        {
            draft.Dispose();
            throw;
        }
        Console.Error.WriteLine(
            $"[InferenceEngine] DSpark draft attached: {cfg.NumLayers}L block-{cfg.BlockSize} " +
            $"({decision.Placement}) from {stPath}");
    }

    // ── Backend dispatch ─────────────────────────────────────────────────────

    /// <summary>
    /// Translate <see cref="SharpInferenceServerOptions.PrefillDequantCacheMb"/> (MiB, nullable)
    /// into the <see cref="ForwardPass"/> constructor's byte budget: <c>null</c> → defer to the
    /// <c>SHARPI_PREFILL_DEQUANT_MB</c> env / auto-sizing; <c>0</c> → off; negative → unlimited;
    /// positive → that many MiB (saturating, never overflowing).
    /// </summary>
    private static long DequantCacheBytes(long? mb) =>
        mb is null ? long.MinValue : ForwardPass.MbToBudgetBytes(mb.Value);

    private static (IForwardPass Fwd, bool BatchingSupported) BuildForwardPass(
        GgufModel model, ModelHyperparams hp, string arch, int ctxSize, int nGpuLayers,
        ServerBackend backend, bool turboQuant, List<IDisposable> owned, long prefillDequantCacheBytes,
        bool preferBatchingOverAutoSnapKv = false)
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

            var dense = new ForwardPass(model, cpuBackend, hp,
                prefillDequantCacheBytes: prefillDequantCacheBytes);
            owned.Add(dense);
            if (turboQuant)
                dense.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
            // ContinuousBatchingEngine doesn't yet support MoE, TurboQuant fan-out, or
            // gemma4 per-layer head_dim (PrefillWithCache / BatchForwardMulti /
            // PrefillPackedMulti all throw NotSupportedException) — those fall back to
            // the single-user InferenceEngine instead of failing every request.
            bool batchOk = !hp.IsMoE && !turboQuant && hp.LayerHeadDim is null;
            return (dense, batchOk);
        }

        // GPU paths share a CPU baseline (for the hybrid-CPU half of partial offload).
        // For full GPU paths the dense CPU fwd is unused but still cheap to construct.
        // The #189 dequant cache is off here: GPU/hybrid never drive the batched CPU prefill
        // that consults it, so a full F32 model copy would be pure wasted RAM.
        ForwardPass? cpuDense = hp.IsHybridSsm ? null
            : new ForwardPass(model, cpuBackend, hp, prefillDequantCacheBytes: 0);
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
                ? TierPlanner.Plan(model, hp, hwProfile, turboQuant, requestedCtxSize: ctxSize,
                    kvDtype: CudaForwardPass.ResolveConfiguredKvDType()).GpuLayers
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
                // #196 Option 2: when batching is requested, suppress the VRAM-scaled SnapKV
                // auto-enable (prefer pure batching over routing every sequence through the slower
                // per-sequence-eviction decode). An explicit SHARPI_SNAPKV_BUDGET>0 still wins and
                // composes with batching via #196 Option 1.
                var cfwd = new CudaForwardPass(model, cuda, hp, ctxSize, enableTurboQuant: turboQuant,
                    preferBatchingOverAutoSnapKv: preferBatchingOverAutoSnapKv);
                owned.Add(cfwd);
                // Issue #190 (dense) / #195 (Gemma 4): CUDA full-offload supports continuous
                // batching (per-sequence GPU KV caches + true batched decode). SupportsContinuous-
                // Batching is the single source of truth shared with CudaForwardPass's runtime
                // guard, so the loader gate can't diverge from what the batched methods accept — it
                // admits dense AND Gemma-4 models and folds OUT MoE, TurboQuant, a dense final-logit
                // softcap, and any non-GEMM-N-batchable trunk/output weight dtype (Q4_0). A SnapKV
                // budget no longer disqualifies batching (#196 — it composes via per-sequence eviction).
                bool batches = cfwd.SupportsContinuousBatching;
                // #196 footgun guard: we suppressed the SnapKV auto-enable because batching was
                // requested, but this model can't actually batch (e.g. dense softcap / Q4_0) — so it
                // would fall back to single-user with NO auto-SnapKV either. Warn so the operator can
                // restore the memory savings with an explicit budget or a narrowed --kv-type.
                if (preferBatchingOverAutoSnapKv && !batches && !cfwd.SnapKvEnabled)
                    Console.Error.WriteLine(
                        "[InferenceEngineLoader] MaxBatchSize>1 suppressed SnapKV auto-enable, but this " +
                        "model does not support continuous batching (it will run single-user). Set an " +
                        "explicit SHARPI_SNAPKV_BUDGET>0 or a narrowed --kv-type to keep KV memory bounded.");
                return (cfwd, batches);
            }

            // pinGpuLayers (not a `with { GpuLayers = }` override) so the expert-cache budget the
            // MoE CPU-vs-SLRU auto-decision reads is priced for THIS split, not the auto one (#224).
            var planForHybrid = TierPlanner.Plan(model, hp, hwProfile, turboQuant, requestedCtxSize: ctxSize,
                kvDtype: CudaForwardPass.ResolveConfiguredKvDType(), pinGpuLayers: gpuLayers);
            var chfwd = new CudaHybridForwardPass(model, cuda, hp, planForHybrid, turboQuant);
            owned.Add(chfwd);
            return (chfwd, BatchingSupported: false);
        }

        if (backend == ServerBackend.Vulkan)
        {
            var vulkan = new VulkanBackend();
            owned.Add(vulkan);

            if (hp.IsHybridSsm)
            {
                // Layer placement for hybrid GDN is driven by hp.LayerTypes, not VRAM budget —
                // so TierPlanner is skipped and we claim "all layers on GPU" (GDN/attn routing
                // is implicit; FFN is per-layer GPU/CPU). Mirrors the CUDA loader branch.
                // (PR4 Round 1 — dense FFN; Round 2 — MoE FFN via CPU-MoE / GPU-SLRU.)
                var placement = new LayerPlacement(
                    GpuLayers: hp.NumLayers,
                    CpuLayers: 0,
                    GpuWeightBytes: 0,
                    GpuKvBytes: 0,
                    RecommendedCtxSize: ctxSize > 0 ? ctxSize : Math.Min(hp.ContextLength, 4096));
                var vhgdn = new VulkanHybridGdnForwardPass(model, vulkan, hp, placement);
                owned.Add(vhgdn);
                return (vhgdn, BatchingSupported: false);
            }

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
                var gfwd = new GpuForwardPass(model, vulkan, hp, ctxSize, enableTurboQuant: turboQuant,
                    kvDtype: CudaForwardPass.ResolveConfiguredKvDType());
                owned.Add(gfwd);
                return (gfwd, BatchingSupported: false);
            }

            var planForHybrid = TierPlanner.Plan(model, hp, hwProfile, turboQuant, requestedCtxSize: ctxSize,
                pinGpuLayers: gpuLayers);
            var hfwd = new HybridForwardPass(model, vulkan, hp, planForHybrid, turboQuant);
            owned.Add(hfwd);
            return (hfwd, BatchingSupported: false);
        }

        throw new InvalidOperationException($"Unknown backend selection: {backend}");
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Sets the MoE-related environment variables (SHARPI_MOE_* cache knobs plus the
    /// SHARPI_CPU_MOE placement override, issue #93) from the options object so the engine's
    /// MoE code picks them up at construction time. The engine reads these once per backend
    /// instance, so we have to do this BEFORE the backend is built.
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

        // CPU-MoE placement (issue #93, mirrors the CLI's --cpu-moe issue #80). The hybrid
        // forward passes read SHARPI_CPU_MOE once at construction, so write it before load.
        // Nullable: only an explicit option writes — null leaves any externally-set value (and
        // the engine's VRAM-fit auto-select) untouched.
        if (opts.CpuMoe is bool cpuMoe)
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", cpuMoe ? "1" : "0");

        // GPU op-offload of the CPU-MoE routed prefill (mirrors the CLI's --gpu-moe-prefill).
        // Opt-in, default OFF in the engine; nullable here so only an explicit option overrides.
        if (opts.GpuMoePrefill is bool gpuMoePrefill)
            Environment.SetEnvironmentVariable("SHARPI_MOE_GPU_PREFILL", gpuMoePrefill ? "1" : "0");
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
    private static string ResolvePath(string? path, string what, string envVar, string configKey)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new InvalidOperationException(
                $"SharpInferenceServerOptions.{configKey} ({what} path) is required. " +
                $"Set it via Configure(o => o.{configKey} = ...) or the {envVar} environment variable.");

        if (Path.IsPathRooted(path) && File.Exists(path))
            return path;

        if (File.Exists(path))
            return Path.GetFullPath(path);

        var candidates = new List<string>
        {
            Path.Combine(Directory.GetCurrentDirectory(), path),
            Path.Combine(AppContext.BaseDirectory, path),
        };
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        for (int i = 0; i < 5 && dir is not null; i++, dir = dir.Parent)
            candidates.Add(Path.Combine(dir.FullName, path));

        var resolved = candidates.FirstOrDefault(File.Exists);
        if (resolved is not null) return resolved;

        throw new InvalidOperationException(
            $"{char.ToUpperInvariant(what[0])}{what[1..]} file not found: '{path}'. " +
            $"Set SharpInferenceServerOptions.{configKey}, the {envVar} environment variable, " +
            $"or the SharpInference:{configKey} configuration key.");
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
