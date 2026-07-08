using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using SharpInference.Core;
using SharpInference.Vision;

namespace SharpInference.Engine;

/// <summary>
/// Top-level inference engine. Wraps a forward pass + tokenizer, applies pre-formatted
/// prompts (caller applies chat template), and provides serialized async generation.
/// One request runs at a time; concurrent callers block in arrival order.
///
/// Prefix caching: if successive prompts share a page-aligned token prefix, the KV cache
/// for those positions is reused and only the new suffix is prefilled — eliminating
/// redundant computation for repeated system prompts.
///
/// Reasoning support: when constructed with <c>thinkTokenId</c> / <c>endThinkTokenId</c>
/// for a model that emits <c>&lt;think&gt;...&lt;/think&gt;</c>, the engine splits output
/// into <see cref="GenerateChunkKind.Thinking"/> and <see cref="GenerateChunkKind.Text"/>
/// chunks. Boundary tokens themselves are consumed and never appear in chunk text.
/// </summary>
public sealed class InferenceEngine : IInferenceEngine, IDisposable, IAsyncDisposable
{
    private readonly IForwardPass _fwd;
    private readonly ITokenizer _tokenizer;
    private readonly IDisposable[] _owned;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly int _thinkTokenId;
    private readonly int _endThinkTokenId;

    // Cancelled by Dispose/DisposeAsync; linked into every generation's token so a
    // shutdown stops the in-flight background worker promptly rather than letting it
    // run to MaxNewTokens (issue #132).
    private readonly CancellationTokenSource _shutdownCts = new();

    // Issue #302: a single dedicated, long-lived thread that drives every forward pass for this
    // engine. CUDA contexts are thread-affine — running the forward pass on arbitrary Task.Run
    // pool threads (the old model) deadlocks the first CUDA call in a non-interactive session,
    // where the driver does not keep the device's primary context current on freshly-scheduled
    // worker threads. A stable thread binds the backend context once (the first request's
    // _fwd.BindToCurrentThread, idempotent thereafter) and keeps CUDA streams/graphs on one
    // consistent thread for the engine's lifetime. _gate already serializes requests to one at a
    // time, so a single engine thread per device is sufficient. The per-request streaming
    // Channel<GenerateChunk> is unchanged; only the producer's execution context moved off the
    // thread pool. Background so a leaked/undisposed engine never blocks process exit.
    private readonly BlockingCollection<Action> _engineWork = new();
    private readonly Thread _engineThread;

    // Prefix caching state (guarded by _gate — only accessed during generation).
    private int[]? _prevTokens;

    // Multi-slot prefix cache (issue #212), opt-in via SHARPI_PREFIX_SLOTS=2. Non-null only
    // when the dense rewindable pass supports it AND the env requested 2 slots; otherwise the
    // engine stays exactly single-slot (byte-identical default path). Slot 0 = the pass's owned
    // cache (legacy _prevTokens semantics); slot 1 = a bounded scratch cache for short
    // interleaved requests so an aux call doesn't evict the long resident prefix in slot 0.
    // _slotTokens[s] mirrors _prevTokens for slot s. Engine owns/disposes the scratch slot(s).
    private readonly IMultiSlotKvCache? _multiSlot;
    private ISequenceKvCache[]? _slots;
    private int[]?[]? _slotTokens;
    private readonly int _scratchCap;

    // Image input (issue #253). Set once at load via EnableImageInput; the VisionModel is
    // owned/disposed elsewhere (in _owned). Image requests splice projected soft tokens at
    // each placeholder during a fresh (no-prefix-reuse) prefill.
    private GemmaUvVisionEmbedder? _visionEmbedder;
    // DSpark draft head (PR #413 Phase 6) — attached post-construction by the loader;
    // engine-owned (disposed in DisposeCore before the forward pass it drafts against).
    private IDSparkDraft? _dspark;
    private VisionModel? _visionModel;
    private int _imgOpenId, _imgCloseId, _imgPlaceholderId;

    // Observability counters (updated via Interlocked).
    private int _pendingCount;
    private int _activeCount;
    private long _prefillTokensReused;

    // 0 = live, 1 = disposed. Interlocked so Dispose/DisposeAsync are single-shot even
    // if both are called (or called concurrently).
    private int _disposed;

    // Upper bound on how long Dispose waits for an in-flight generation to drain before
    // giving up. The worker is cancelled (via _shutdownCts) and cooperatively checks
    // cancellation between decode tokens and prefill chunks, so the real wait is one
    // forward/prefill-chunk; this is only a backstop against a wedged backend or an
    // abandoned (never-disposed) enumerator stranding the gate. On timeout the forward
    // pass is leaked rather than freed (see DisposeCore). Overridable by tests via
    // InternalsVisibleTo to exercise the timeout path without a 10s wait.
    internal TimeSpan _disposeDrainTimeout = TimeSpan.FromSeconds(10);

    /// <summary>
    /// Optional logger for per-request diagnostics. Set by the DI registration from the host's
    /// <see cref="ILoggerFactory"/>; null in non-hosted use (CLI, tests) — calls are no-ops then.
    /// The per-request perf trace is emitted at <see cref="LogLevel.Debug"/>, so it's off unless
    /// the host enables Debug for this category (e.g. <c>"SharpInference.Engine": "Debug"</c>).
    /// </summary>
    public ILogger? Logger { get; set; }

    public string ModelId { get; }

    /// <summary>Number of callers blocked waiting to acquire the generation gate.</summary>
    public int QueueDepth => _pendingCount;

    /// <summary>1 while a generation is in progress, 0 otherwise.</summary>
    public int ActiveRequests => _activeCount;

    /// <inheritdoc/>
    /// <remarks>
    /// True when the underlying pass supports partial rewind (page-aligned reuse across
    /// any matching prefix) OR a single-slot end-of-prefill snapshot (issue #21 / #102,
    /// used for chat-continuation reuse on GDN-hybrid passes via a caller-supplied
    /// canonical-history hint). A <c>false</c> here is the only configuration where the
    /// engine cannot reuse KV state across turns; reuse on a <c>true</c> backend is still
    /// gated per-request on a matching prefix being available.
    /// </remarks>
    public bool PrefixCacheEnabled => _fwd.SupportsPartialRewind || _fwd.SupportsSnapshot;

    /// <inheritdoc/>
    public long PrefillTokensReused => Interlocked.Read(ref _prefillTokensReused);

    /// <param name="fwd">Forward pass implementation (CPU / GPU / Hybrid). Owned by this engine.</param>
    /// <param name="tokenizer">Tokenizer matching the model vocabulary.</param>
    /// <param name="modelId">Human-readable model identifier returned in API responses.</param>
    /// <param name="thinkTokenId">
    /// Token ID of the model's <c>&lt;think&gt;</c> marker, or <c>-1</c> if the model has no reasoning
    /// stream. When <c>-1</c>, all chunks are emitted as <see cref="GenerateChunkKind.Text"/>.
    /// </param>
    /// <param name="endThinkTokenId">
    /// Token ID of the model's <c>&lt;/think&gt;</c> marker, or <c>-1</c>. Must be paired with
    /// a non-negative <paramref name="thinkTokenId"/> to enable reasoning-stream splitting.
    /// </param>
    /// <param name="owned">Additional disposable resources owned by this engine (backend, model handle, etc.).</param>
    public InferenceEngine(
        IForwardPass fwd,
        ITokenizer tokenizer,
        string modelId,
        int thinkTokenId,
        int endThinkTokenId,
        params IDisposable[] owned)
    {
        _fwd = fwd;
        _tokenizer = tokenizer;
        ModelId = modelId;
        _thinkTokenId = thinkTokenId;
        _endThinkTokenId = endThinkTokenId;
        _owned = owned;

        if (!fwd.SupportsPartialRewind)
        {
            // Issue #102: GDN-hybrid passes report SupportsPartialRewind=false but still
            // expose a single-slot snapshot via CaptureSnapshot/SnapshotLength. The engine
            // uses it for chat-continuation reuse when callers pass a canonicalHistoryPrefix,
            // so flagging the cache as universally "disabled" is misleading. Distinguish
            // the two flavours so server logs match what actually happens at request time.
            if (fwd.SupportsSnapshot)
            {
                Console.Error.WriteLine(
                    $"[InferenceEngine] partial-rewind prefix cache disabled — {fwd.GetType().Name} " +
                    "uses destructive recurrent state. Snapshot-based prefix reuse is available across " +
                    "chat turns when the caller supplies a canonical-history hint (chat completion endpoints do).");
            }
            else
            {
                Console.Error.WriteLine(
                    $"[InferenceEngine] prefix cache disabled — {fwd.GetType().Name} reports SupportsPartialRewind == false " +
                    "and exposes no snapshot facility. Multi-turn requests will re-prefill the full prompt.");
            }
        }

        // Multi-slot prefix cache (issue #212): opt-in via SHARPI_PREFIX_SLOTS=2, dense
        // rewindable path only. A bounded scratch slot (SHARPI_PREFIX_SCRATCH_TOKENS, default
        // 4096) holds short interleaved requests so they don't evict slot 0's long prefix.
        // Default (env unset / != 2) leaves _multiSlot null → exact single-slot behavior.
        _scratchCap = 4096;
        if (int.TryParse(
                Environment.GetEnvironmentVariable("SHARPI_PREFIX_SCRATCH_TOKENS"),
                System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out int sc) && sc > 0)
            _scratchCap = sc;

        if (int.TryParse(
                Environment.GetEnvironmentVariable("SHARPI_PREFIX_SLOTS"),
                System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out int slots)
            && slots == 2
            && fwd.SupportsPartialRewind
            && fwd is IMultiSlotKvCache ms && ms.SupportsMultiSlotPrefix)
        {
            // The scratch tensor is clamped to the model's max context (AllocateScratchSlot caps at
            // _maxSeqLen); clamp the routing cap to match so SelectSlot's footprint gate can't pass
            // a request the (clamped) scratch allocation couldn't actually hold.
            _scratchCap = Math.Min(_scratchCap, fwd.MaxSeqLen);

            // Allocating the extra scratch KV region can OOM: auto-context sizes _maxSeqLen to
            // fit exactly one full KV cache in VRAM and doesn't budget the scratch slot. Since
            // the feature is opt-in, degrade to single-slot on a device-allocation failure rather
            // than aborting engine construction. (Logger is DI-set after construction, so it's
            // unavailable here — emit to stderr like the other SHARPI_* opt-in traces.) A
            // NotSupportedException means the capability checks above disagree with
            // AllocateScratchSlot — a real bug, not an allocation failure — so let it propagate.
            try
            {
                var scratch = ms.AllocateScratchSlot(_scratchCap);
                _multiSlot = ms;
                _slots = [ms.OwnedSlot, scratch];
                _slotTokens = new int[]?[2];
            }
            catch (Exception ex) when (ex is not NotSupportedException and not OutOfMemoryException
                                       and not OperationCanceledException)
            {
                Console.Error.WriteLine(
                    $"[InferenceEngine] SHARPI_PREFIX_SLOTS=2: could not allocate the " +
                    $"{_scratchCap}-token scratch KV slot ({ex.GetType().Name}: {ex.Message}); " +
                    "falling back to single-slot prefix cache.");
            }
        }

        // Issue #302: start the dedicated forward-pass thread. It binds the backend's thread-
        // affine context on the first work item and then serves generation work in arrival order
        // until Dispose completes the queue.
        _engineThread = new Thread(EngineThreadLoop)
        {
            IsBackground = true,
            Name = "sharpi-engine",
        };
        _engineThread.Start();
    }

    /// <summary>
    /// Body of the dedicated forward-pass thread (issue #302). Runs queued generation work items
    /// in arrival order until <see cref="Dispose"/> / <see cref="DisposeAsync"/> completes the
    /// queue. Each work item binds the CUDA context (idempotent; a no-op on CPU/Vulkan) and routes
    /// its own exceptions to its per-request channel, so this loop only needs a last-resort guard
    /// to keep a surprise escape from tearing down the shared thread and stranding later requests.
    /// </summary>
    private void EngineThreadLoop()
    {
        foreach (var work in _engineWork.GetConsumingEnumerable())
        {
            try
            {
                work();
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[InferenceEngine] forward-pass work item threw unexpectedly: {ex}");
            }
        }
    }

    /// <summary>
    /// Issue #302: queue <paramref name="work"/> onto the dedicated forward-pass thread and return
    /// a task that completes when it finishes. Replaces the old <c>Task.Run</c>, which dispatched
    /// the forward pass to an arbitrary pool thread the CUDA context wasn't bound to. The work item
    /// reports its own errors through the request's channel and never throws, so this only needs to
    /// signal completion (so the generation's drain in the <c>finally</c> can release the gate).
    /// </summary>
    private Task RunOnEngineThread(Action work)
    {
        var tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _engineWork.Add(() =>
        {
            try { work(); }
            finally { tcs.TrySetResult(); }
        });
        return tcs.Task;
    }

    /// <summary>
    /// Issue #212: whether the 2-slot prefix cache is active for this engine. False when
    /// <c>SHARPI_PREFIX_SLOTS</c> wasn't 2, the backend can't multi-slot, or the scratch
    /// allocation failed and the engine degraded to single-slot. Lets a host observe the
    /// degraded fallback (which otherwise only writes a one-shot stderr line at construction).
    /// </summary>
    public bool MultiSlotPrefixActive => _slots is not null;

    /// <summary>
    /// Convenience constructor that auto-resolves the reasoning-stream boundary tokens from the
    /// tokenizer (<see cref="ITokenizer.ReasoningTokens"/>) so an in-process consumer gets the
    /// same Thinking/Text split the server and CLI configure — without having to re-derive the
    /// model-family convention. A tokenizer that defines none (the default) yields
    /// <c>(-1, -1)</c>, leaving the engine in plain text-only mode exactly as before. Pass the
    /// explicit five-argument constructor with <c>thinkTokenId = -1</c> to force the split off.
    /// (Fixes issue #304: a Gemma 4 host using this constructor leaked <c>&lt;|channel&gt;</c>
    /// markers because the split was hardcoded off here.)
    /// </summary>
    public InferenceEngine(
        IForwardPass fwd,
        ITokenizer tokenizer,
        string modelId,
        params IDisposable[] owned)
        // Guard tokenizer before the base initializer dereferences ReasoningTokens, so a null
        // surfaces as a clear ArgumentNullException rather than an opaque NullReferenceException.
        : this(fwd ?? throw new ArgumentNullException(nameof(fwd)),
               tokenizer ?? throw new ArgumentNullException(nameof(tokenizer)),
               modelId,
               tokenizer.ReasoningTokens.Open, tokenizer.ReasoningTokens.Close, owned)
    {
    }

    /// <summary>
    /// Enable image (vision) input (issue #253). Call once at load time, before any request.
    /// The <paramref name="model"/> must already be in this engine's <c>owned</c> disposables
    /// (the loader adds it) — this method only borrows references. <paramref name="openId"/>,
    /// <paramref name="closeId"/>, <paramref name="placeholderId"/> are the Gemma 4
    /// <c>&lt;|image&gt;</c> / <c>&lt;image|&gt;</c> / <c>&lt;|image|&gt;</c> token ids.
    /// </summary>
    public void EnableImageInput(GemmaUvVisionEmbedder embedder, VisionModel model,
        int openId, int closeId, int placeholderId)
    {
        _visionEmbedder = embedder;
        _visionModel = model;
        _imgOpenId = openId;
        _imgCloseId = closeId;
        _imgPlaceholderId = placeholderId;
    }

    /// <summary>
    /// Attach a DSpark draft head (docs/dspark-plan.md Phase 6, PR #413) so greedy
    /// no-thinking requests decode through <see cref="DSparkDecoder"/>. The caller
    /// must have called <see cref="IForwardPass.EnableHiddenTaps"/> on the forward
    /// pass with the head's target layer ids BEFORE any request runs (the loader
    /// does this at setup — the tap buffer must cover every prompt position the
    /// decoder later replays). The engine takes ownership and disposes the draft.
    /// </summary>
    public void AttachDSparkDraft(IDSparkDraft draft)
    {
        ArgumentNullException.ThrowIfNull(draft);
        if (_dspark is not null)
            throw new InvalidOperationException("A DSpark draft is already attached.");
        if (!_fwd.SupportsPartialRewind || !_fwd.SupportsHiddenTaps)
            throw new InvalidOperationException(
                $"DSpark requires a partial-rewind, tap-capable forward pass; {_fwd.GetType().Name} " +
                "reports SupportsPartialRewind=" + _fwd.SupportsPartialRewind +
                ", SupportsHiddenTaps=" + _fwd.SupportsHiddenTaps + ".");
        if (_fwd.HiddenTapDim != draft.TapDim)
            throw new InvalidOperationException(
                $"Hidden taps not enabled for this head: target HiddenTapDim={_fwd.HiddenTapDim}, " +
                $"draft TapDim={draft.TapDim}. Call EnableHiddenTaps(head.TargetLayerIds) before attaching.");
        if (_fwd.VocabSize != draft.VocabSize)
            throw new InvalidOperationException(
                $"DSpark head vocab ({draft.VocabSize}) != model vocab ({_fwd.VocabSize}).");
        if (_multiSlot is not null)
            throw new InvalidOperationException(
                "DSpark is incompatible with the multi-slot prefix cache (SHARPI_PREFIX_SLOTS=2): " +
                "the tap buffer is position-indexed against a single KV region, and a scratch-slot " +
                "request would overwrite tap rows the owned slot's cached prefix still needs. " +
                "Unset SHARPI_PREFIX_SLOTS to use DSpark.");
        _dspark = draft;
    }

    /// <inheritdoc/>
    public bool SupportsImageInput => _visionEmbedder is not null && _fwd.SupportsEmbeddingInput;

    /// <summary>
    /// Issue #253 image prefill: project each image to its soft-token block, then walk the
    /// prompt tokens — emitting ordinary tokens via <see cref="IForwardPass.Forward"/> and
    /// expanding each <c>&lt;|image|&gt;</c> placeholder to open-marker + per-soft-token
    /// <see cref="IForwardPass.ForwardEmbedding"/> + close-marker (mirrors the CLI's
    /// <c>RunImagePrompt</c>). Returns the final prefill logits; <paramref name="endPos"/>
    /// receives the post-prefill sequence length. Runs on the generation worker thread.
    /// </summary>
    private ReadOnlySpan<float> SplicePrefillImages(int[] tokens, IReadOnlyList<byte[]> imageBytes,
        CancellationToken ct, out int endPos)
    {
        var vision = _visionModel!;
        var embedder = _visionEmbedder!;
        int embDim = vision.EmbeddingLength;

        // Project every image up front (decode → preprocess → embed), in request order.
        var blocks = new (float[] Soft, int NTok)[imageBytes.Count];
        for (int i = 0; i < imageBytes.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            byte[] rgb;
            int w, h;
            using (var ms = new MemoryStream(imageBytes[i], writable: false))
                rgb = ImageIO.LoadRgb(ms, out w, out h);
            var pre = ImagePreprocessor.Preprocess(rgb, w, h, vision);
            var soft = embedder.Forward(pre.Chw, pre.Height, pre.Width, out int nTok);
            blocks[i] = (soft, nTok);
        }

        int placeholders = 0;
        foreach (int id in tokens) if (id == _imgPlaceholderId) placeholders++;
        if (placeholders != imageBytes.Count)
            throw new InvalidOperationException(
                $"Image count mismatch: the rendered prompt has {placeholders} <|image|> placeholder(s) " +
                $"but {imageBytes.Count} image(s) were supplied.");

        int pos = 0;
        int imgIdx = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in tokens)
        {
            ct.ThrowIfCancellationRequested();
            if (id == _imgPlaceholderId)
            {
                var (soft, nTok) = blocks[imgIdx++];
                logits = _fwd.Forward(_imgOpenId, pos++);
                for (int t = 0; t < nTok; t++)
                    logits = _fwd.ForwardEmbedding(soft.AsSpan(t * embDim, embDim), pos++);
                logits = _fwd.Forward(_imgCloseId, pos++);
            }
            else
            {
                logits = _fwd.Forward(id, pos++);
            }
        }
        endPos = pos;
        return logits;
    }

    /// <summary>
    /// Prompt-prefill batch size, in tokens. A single <see cref="IForwardPass.Prefill"/> call is
    /// opaque to the request's <see cref="CancellationToken"/> — the engine only checks <c>ct</c>
    /// between decode tokens — so a large-prompt prefill would otherwise run to completion even
    /// after the client has disconnected, pinning the engine gate and a CPU/GPU core for the dead
    /// request. Splitting the prompt into chunks and checking <c>ct</c> between them makes prefill
    /// cooperatively cancellable, bounding the post-disconnect compute to (at most) one chunk.
    /// <para>
    /// Chunking is numerically identical to a single prefill: a transformer forward produces one
    /// independent output row per token (GEMM output rows don't mix across the token batch) and the
    /// GDN recurrent state carries across calls via the cache — the same multi-call pattern the
    /// prefix-reuse and canonical-snapshot prefill paths already rely on. Tunable via
    /// <c>SHARPI_PREFILL_CHUNK</c> (set very large to effectively disable chunking, e.g. for
    /// prefill throughput benchmarking).
    /// </para>
    /// </summary>
    private static readonly int PrefillChunkSize =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPI_PREFILL_CHUNK"), out int c) && c > 0
            ? c
            : 512;

    /// <summary>
    /// Prefill the absolute token range <c>[from, to)</c> (where <c>tokens[i]</c> sits at cache
    /// position <c>i</c>) in <see cref="PrefillChunkSize"/>-token chunks, checking
    /// <paramref name="ct"/> before each chunk so a client disconnect aborts prefill promptly.
    /// Returns the logits for the last processed token (the only ones a caller consumes).
    /// </summary>
    private ReadOnlySpan<float> PrefillChunked(int[] tokens, int from, int to, CancellationToken ct)
    {
        ReadOnlySpan<float> logits = default;
        for (int pos = from; pos < to; pos += PrefillChunkSize)
        {
            ct.ThrowIfCancellationRequested();
            int len = Math.Min(PrefillChunkSize, to - pos);
            logits = _fwd.Prefill(new ArraySegment<int>(tokens, pos, len), pos);
        }
        return logits;
    }

    /// <summary>
    /// Cancellation-checkpointed twin of <see cref="PrefillChunked"/> for the MTP attention KV
    /// cache (issue #33). Same chunking rationale; populates one chunk per call so a disconnect
    /// during MTP-prompt population is honored within one chunk.
    /// </summary>
    private void PrefillMtpChunked(int[] tokens, int from, int to, CancellationToken ct)
    {
        for (int pos = from; pos < to; pos += PrefillChunkSize)
        {
            ct.ThrowIfCancellationRequested();
            int len = Math.Min(PrefillChunkSize, to - pos);
            _fwd.PrefillMtp(new ArraySegment<int>(tokens, pos, len), pos);
        }
    }

    /// <summary>
    /// Finds the longest page-aligned prefix shared between the new token array and the cached
    /// previous token array, returning its length (0 if no reusable prefix exists).
    /// </summary>
    private int FindCacheablePrefix(int[] tokens) => PageAlignedSharedPrefix(tokens, _prevTokens, tokens.Length);

    /// <summary>
    /// Longest page-aligned token prefix shared between <paramref name="a"/> and a cached
    /// <paramref name="prev"/> array (0 if none). Factored out of <see cref="FindCacheablePrefix"/>
    /// so the issue-#212 multi-slot selector can score each resident slot with identical logic.
    /// </summary>
    private static int PageAlignedSharedPrefix(int[] a, int[]? prev, int newLen)
    {
        if (prev == null || newLen <= PagedKvCache.PageSize)
            return 0;

        // Compare up to all-but-last-page tokens (need at least one page to bother).
        int maxCompare = Math.Min(newLen - 1, prev.Length);
        int match = 0;
        while (match < maxCompare && a[match] == prev[match])
            match++;

        // Align down to page boundary (must be at least one full page).
        return (match / PagedKvCache.PageSize) * PagedKvCache.PageSize;
    }

    /// <summary>
    /// Issue #212 multi-slot selection: pick the resident slot to serve <paramref name="tokens"/>.
    /// Scores each slot's page-aligned shared prefix; reuses the slot with the largest match.
    /// On no match, picks a victim — the bounded scratch slot when the prompt fits its capacity
    /// (so a short interleaved request leaves slot 0's long prefix intact), otherwise the
    /// full-size owned slot. Returns the chosen slot index and the prefix length to reuse
    /// (0 for the victim). Only called when <see cref="_slots"/> is non-null.
    /// </summary>
    private (int SlotIdx, int PrefixLen) SelectSlot(int[] tokens, int maxNewTokens)
    {
        int[]?[] slotTokens = _slotTokens!;

        // Worst-case absolute positions this request will occupy: prompt + every decoded token
        // (decode appends KV at startPos+i for i in [0, maxNewTokens), so the highest written
        // offset is (footprint-1)*kvDim). A request may bind the BOUNDED scratch slot only if
        // its whole footprint fits the allocation — otherwise decode would append KV past the
        // scratch tensor (out-of-bounds VRAM write). The owned slot 0 is full-size (_maxSeqLen),
        // so it always fits and needs no such gate. Issue #212.
        long footprint = (long)tokens.Length + Math.Max(0, maxNewTokens);
        bool scratchFits = footprint <= _scratchCap;

        int bestSlot = -1, bestPrefix = 0;
        for (int s = 0; s < slotTokens.Length; s++)
        {
            // Skip a prefix match against the scratch slot when the request can't safely live
            // there — fall through to (re-prefilling on) the owned slot instead.
            if (s == ScratchSlotIdx && !scratchFits) continue;
            int p = PageAlignedSharedPrefix(tokens, slotTokens[s], tokens.Length);
            if (p > bestPrefix)
            {
                bestPrefix = p;
                bestSlot = s;
            }
        }
        if (bestSlot >= 0)
            return (bestSlot, bestPrefix);

        // No reusable prefix: pick a victim. Prefer the bounded scratch slot when the footprint
        // fits (so a short interleaved request leaves slot 0's long prefix intact); otherwise the
        // full-size owned slot absorbs it.
        int victim = scratchFits ? ScratchSlotIdx : OwnedSlotIdx;
        return (victim, 0);
    }

    // Issue #212 slot indices: slot 0 is the pass's owned (full-size) cache; slot 1 is the
    // bounded scratch cache. Named for clarity in SelectSlot's footprint gating.
    private const int OwnedSlotIdx = 0;
    private const int ScratchSlotIdx = 1;

    /// <summary>
    /// Back-compat string-stream view of <see cref="GenerateChunksAsync"/>: yields only
    /// user-facing answer text, suppressing reasoning chunks. Equivalent to the default
    /// interface-method implementation on <see cref="IInferenceEngine"/> but callable
    /// directly on the concrete type.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var c in GenerateChunksAsync(prompt, sp, ct).WithCancellation(ct).ConfigureAwait(false))
        {
            if (c.Kind == GenerateChunkKind.Text)
                yield return c.Text;
        }
    }

    /// <inheritdoc/>
    public IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt,
        SamplingParams sp,
        CancellationToken ct = default,
        string? canonicalHistoryPrefix = null)
        => GenerateCore(prompt, sp, ct, canonicalHistoryPrefix, imageBytes: null);

    /// <inheritdoc/>
    public IAsyncEnumerable<GenerateChunk> GenerateImageChunksAsync(
        string prompt,
        IReadOnlyList<byte[]> imageBytes,
        SamplingParams sp,
        CancellationToken ct = default)
    {
        if (!SupportsImageInput)
            throw new NotSupportedException(
                "This engine is not configured for image input. Load an mmproj (SHARPI_MMPROJ / " +
                "SharpInferenceServerOptions.MmprojPath) for a Gemma 4 model on a CPU or full-CUDA backend.");
        ArgumentNullException.ThrowIfNull(imageBytes);
        return GenerateCore(prompt, sp, ct, canonicalHistoryPrefix: null, imageBytes);
    }

    private async IAsyncEnumerable<GenerateChunk> GenerateCore(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct,
        string? canonicalHistoryPrefix,
        IReadOnlyList<byte[]>? imageBytes)
    {
        // Link the caller's token with the engine-shutdown token so Dispose can stop this
        // generation's background worker (issue #132). Everything below uses `ct` — the
        // linked token — so cancellation from either source aborts decode and releases the
        // gate via the finally blocks. Reading _shutdownCts.Token can race a concurrent
        // Dispose that already disposed it (we passed the _disposed check above, then got
        // pre-empted); surface that as the engine's ObjectDisposedException, not the
        // CancellationTokenSource's, so callers see a consistent object name.
        CancellationTokenSource linkedCts;
        try
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(InferenceEngine));
            linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, _shutdownCts.Token);
        }
        catch (ObjectDisposedException)
        {
            throw new ObjectDisposedException(nameof(InferenceEngine));
        }
        using var _linkedCts = linkedCts;
        ct = linkedCts.Token;

        Interlocked.Increment(ref _pendingCount);
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        Interlocked.Decrement(ref _pendingCount);
        Interlocked.Increment(ref _activeCount);
        try
        {
            var channel = Channel.CreateUnbounded<GenerateChunk>(
                new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });

            // Capture reasoning-token IDs into locals to avoid repeated field reads in the hot loop.
            // A request whose template was rendered with enable_thinking=false
            // (sp.ThinkingDisabled) opts out of reasoning entirely: the model was told
            // not to think, so the engine skips the think/text stream split AND the
            // speculative fast paths' no-thinking gates pass (CLI --no-thinking parity).
            int thinkId = _thinkTokenId;
            int endThinkId = _endThinkTokenId;
            bool thinkingEnabled = thinkId >= 0 && endThinkId >= 0 && !sp.ThinkingDisabled;

            // Run the blocking generation on the engine's dedicated forward-pass thread (issue
            // #302), not an arbitrary thread-pool thread — CUDA contexts are thread-affine.
            var genTask = RunOnEngineThread(() =>
            {
                // Per-request perf trace (issue: server feels slow on big agentic prompts).
                // Splits encode / prefill / decode so a huge-prompt request shows where the
                // wall time went. Emitted via ILogger at Debug, so enable
                // "SharpInference.Engine": "Debug" in the host's Logging:LogLevel to see it.
                var swReq = Stopwatch.StartNew();
                // Capture the logger once: Logger has a public setter, so reading it twice
                // (IsEnabled check here, LogDebug in the finally) could race a concurrent set.
                var logger = Logger;
                bool logReq = logger?.IsEnabled(LogLevel.Debug) == true;
                int promptTokenCount = 0, reusedTokens = 0, decodeTokens = 0;
                long encodeMs = 0, prefillMs = 0, ttftMs = -1;
                // Issue #212: index of the multi-slot prefix slot bound for this request, or -1
                // when no non-owned slot is active (single-slot mode, or this request stayed on
                // the owned slot). -1 keeps the finally's DeactivateSlot a no-op.
                int activeSlotIdx = -1;
                try
                {
                    // Issue #302: bind the backend's thread-affine context (CUDA) to this thread
                    // before the first CUDA call. No-op on CPU/Vulkan and free after the first
                    // call on the (stable) engine thread; kept inside the try so a bind failure
                    // surfaces through this request's channel instead of hanging the consumer.
                    _fwd.BindToCurrentThread();
                    var tokens = _tokenizer.Encode(prompt).ToArray();
                    promptTokenCount = tokens.Length;
                    encodeMs = swReq.ElapsedMilliseconds;
                    // Surface the prompt-token count out-of-band so endpoints can report
                    // usage.prompt_tokens / input_tokens without re-tokenizing (issue #150).
                    // Emitted before any text/thinking chunk; consumers that only read text
                    // ignore the Usage kind.
                    channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Usage, "", promptTokenCount));
                    var rng = new Random();
                    // Explicit StopTokenIds REPLACE the EOG set; AdditionalStopTokenIds are unioned
                    // on top so a caller can add a stop (e.g. a tool-boundary token) without dropping
                    // EOG — issue #304.
                    System.Collections.Immutable.ImmutableArray<int> stopIds =
                        sp.ResolveStopSet(_tokenizer.EogTokenIds);

                    // Issue #102: canonical-history prefix resolution. The endpoint passes a
                    // chat-template render of just the message history (add_generation_prompt=false);
                    // tokenize it and verify it's a strict token-level prefix of the generating
                    // prompt. The validated boundary becomes (a) the position at which to capture
                    // the GDN snapshot mid-prefill, and (b) the tokens stored as _prevTokens so the
                    // next turn's generating prompt — which starts with that same canonical history
                    // and then appends scrubbed assistant content + the next user turn — matches.
                    // Empty / mismatched canonical → fall through to legacy behavior (snapshot at
                    // end-of-decode, _prevTokens stores generating + decoded).
                    int canonicalLen = 0;
                    if (!string.IsNullOrEmpty(canonicalHistoryPrefix))
                    {
                        var canonTokens = _tokenizer.Encode(canonicalHistoryPrefix).ToArray();
                        if (canonTokens.Length > 0
                            && canonTokens.Length < tokens.Length
                            && tokens.AsSpan(0, canonTokens.Length).SequenceEqual(canonTokens))
                        {
                            canonicalLen = canonTokens.Length;
                        }
                        else if (Environment.GetEnvironmentVariable("SHARPI_TRACE_SNAPSHOT") == "1")
                        {
                            Console.Error.WriteLine(
                                $"[InferenceEngine] canonical-history prefix is not a strict token prefix of the generating prompt " +
                                $"(canon.Len={canonTokens.Length}, prompt.Len={tokens.Length}); " +
                                "falling back to end-of-decode snapshot.");
                        }
                    }

                    // Track the full generated sequence (prompt + decoded tokens) so the
                    // next turn's prefix match has the right baseline — issue #21. Sized
                    // up front: prompt + MaxNewTokens is a safe upper bound. List<int> is
                    // cheap (vs. native alloc) and just shadows the tokens we'd lose to
                    // the channel writer.
                    var fullSeq = new List<int>(tokens.Length + Math.Max(1, sp.MaxNewTokens));
                    fullSeq.AddRange(tokens);

                    // MTP self-speculative decoding (issue #25): when the forward pass
                    // ships an MTP head AND we're in greedy, non-reasoning mode AND
                    // sp.SpecType doesn't forbid it, drive decode through MtpDecoder
                    // instead of the per-token sampling loop below. The MTP path
                    // emits one extra MTP-drafted token per main forward.
                    //
                    // Decided BEFORE prefix-reuse (issue #33): MTP requires the MTP KV
                    // cache to cover every prompt position via PrefillMtp, which only
                    // works when Prefill starts from position 0. Prefix reuse is
                    // therefore skipped on MTP runs.
                    //
                    // sp.SpecType wires the llama.cpp-compatible --spec-type flag:
                    //   Auto  → enable when HasMtpHead && greedy && !thinking
                    //   None  → off (always)
                    //   Mtp   → on; surface a clear error if any prerequisite is missing
                    //
                    // SHARPI_DISABLE_MTP=1 is a back-compat off-switch that wins over Auto
                    // and Mtp (so existing benchmarking scripts that set it keep working).
                    bool mtpEnvDisabled = Environment.GetEnvironmentVariable("SHARPI_DISABLE_MTP") == "1";
                    bool useMtp;
                    bool useDSpark = false;
                    switch (sp.SpecType)
                    {
                        case SpecType.None:
                            useMtp = false;
                            break;
                        case SpecType.DSpark:
                            if (_dspark is null)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=DSpark but no DSpark draft head is attached. " +
                                    "Configure one at load time (SHARPI_DSPARK_MODEL / DSparkModelPath) or " +
                                    "use SpecType.Auto/None.");
                            if (sp.Temperature > 0f)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=DSpark requires greedy sampling (Temperature=0). " +
                                    "DSpark verification is greedy (argmax match).");
                            if (thinkingEnabled)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=DSpark is incompatible with reasoning mode. " +
                                    "Render the chat template with enable_thinking=false.");
                            useDSpark = true;
                            useMtp = false;
                            break;
                        case SpecType.Mtp:
                            if (mtpEnvDisabled)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=Mtp conflicts with SHARPI_DISABLE_MTP=1. " +
                                    "Unset the env var or use SpecType.None.");
                            if (!_fwd.HasMtpHead)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=Mtp requires a model with an MTP head. " +
                                    $"{_fwd.GetType().Name} reports HasMtpHead=false (no nextn tensors in the GGUF).");
                            if (sp.Temperature > 0f)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=Mtp requires greedy sampling (Temperature=0). " +
                                    "MTP verification is greedy (argmax match); sampling support is not yet implemented.");
                            if (thinkingEnabled)
                                throw new InvalidOperationException(
                                    "SamplingParams.SpecType=Mtp is incompatible with reasoning mode. " +
                                    "Pass --no-thinking (or render the chat template with enable_thinking=false).");
                            useMtp = true;
                            break;
                        default: // Auto
                            // An attached DSpark head is an explicit operator choice, so it
                            // outranks the model's innate MTP head when both are eligible.
                            useDSpark = _dspark is not null
                                && sp.Temperature <= 0f
                                && !thinkingEnabled;
                            useMtp = !useDSpark
                                && _fwd.HasMtpHead
                                && !mtpEnvDisabled
                                && sp.Temperature <= 0f
                                && !thinkingEnabled;
                            break;
                    }
                    // Image input (#253) splices precomputed embeddings during prefill; neither
                    // MTP's PrefillMtp / batched verify nor DSpark's tap replay model that.
                    if (imageBytes is { Count: > 0 }) { useMtp = false; useDSpark = false; }

                    // Grammar-constrained decoding (#374) masks logits per token, which the
                    // batched-verify paths don't model — fall back to per-token decode so the
                    // constraint sees and gates every sampled token.
                    var constraint = sp.Constraint;
                    if (constraint is not null) { useMtp = false; useDSpark = false; constraint.Reset(); }

                    // --spec-draft-n-max parity with llama.cpp (issue #30): the MTP
                    // draft-chain length per step. Unset (0) resolves via
                    // SHARPI_MTP_DRAFT_N → built-in default; MtpDecoder clamps per
                    // step against the pass's snapshot-ring capacity
                    // (MaxBatchVerifyTokens), so over-asking degrades gracefully.
                    int mtpDraftN = MtpDecoder.ResolveDraftN(sp.SpecDraftNMax);

                    // Prefix cache decision: two branches.
                    //   (a) Rewindable attention pass — existing FindCacheablePrefix path,
                    //       page-aligned KV reuse.
                    //   (b) Non-rewindable GDN hybrid with a snapshot — issue #21: try
                    //       exact-match against the most-recent snapshot length so the
                    //       held GDN recurrent state can be restored. Works for MTP
                    //       runs too (issue #106) — PrefillMtp accepts startPos > 0 and
                    //       TruncateTo soft-truncates the MTP KV alongside the trunk.
                    // If neither branch hits, fall back to a full ResetCache + Prefill.
                    // startPos = the sequence length after prefill. For text it equals
                    // tokens.Length; for an image request each <|image|> placeholder expands to
                    // open + soft tokens + close, so startPos > tokens.Length and the decode
                    // below must advance positions from there.
                    int startPos;
                    int prefixLen = 0;      // cached-prefix length; 0 for the image path (no reuse)
                    int[] suffixTokens;     // suffix after any cached prefix; reused by the MTP path
                    bool useCanonicalSnapshot = false;
                    long prefillStartMs = swReq.ElapsedMilliseconds;
                    ReadOnlySpan<float> logits;

                    if (imageBytes is { Count: > 0 })
                    {
                        // Image input (#253): no prefix reuse — project each image and splice its
                        // soft tokens in place of its placeholder during a fresh prefill.
                        // Invalidate _prevTokens BEFORE prefilling: the image cache holds soft-token
                        // KV at expanded positions that don't correspond to plain token positions, so
                        // a later text request must NOT match this sequence in FindCacheablePrefix and
                        // TruncateTo into it. Nulling it up front also covers a mid-prefill throw,
                        // which leaves the cache partially written (the end-of-decode snapshot below
                        // is already skipped for image requests).
                        _prevTokens = null;
                        // Issue #212: the image path runs on the owned cache (slot 0, no slot
                        // bind) and ResetCache wipes its KV — invalidate slot 0's token shadow so
                        // a later text request can't SelectSlot-match the now-overwritten prefix.
                        // (The bounded scratch slot 1 is untouched and stays valid.)
                        if (_slotTokens is not null) _slotTokens[0] = null;
                        _fwd.ResetCache();
                        suffixTokens = tokens;
                        logits = SplicePrefillImages(tokens, imageBytes, ct, out startPos);
                    }
                    else
                    {
                        // Issue #212: multi-slot prefix cache. On the dense rewindable path, pick
                        // the resident slot (owned slot 0 or bounded scratch slot 1) with the best
                        // page-aligned prefix, bind it as the active KV region, and point the
                        // engine's working _prevTokens at that slot's cached tokens so the existing
                        // FindCacheablePrefix/TruncateTo block below operates on it unchanged. The
                        // bind is released in this request's finally (DeactivateSlot).
                        if (_slots is not null && _fwd.SupportsPartialRewind)
                        {
                            if (useMtp)
                            {
                                // MTP stays single-slot (slot 0 / owned cache) — no slot bind. But
                                // _prevTokens may still mirror a prior scratch request, so realign
                                // it to slot 0's shadow so prefix reuse matches the owned KV.
                                _prevTokens = _slotTokens![0];
                            }
                            else
                            {
                                (activeSlotIdx, _) = SelectSlot(tokens, sp.MaxNewTokens);
                                _multiSlot!.ActivateSlot(_slots[activeSlotIdx]);
                                _prevTokens = _slotTokens![activeSlotIdx];
                            }
                        }

                        if (_fwd.SupportsPartialRewind)
                        {
                            int candidate = FindCacheablePrefix(tokens);
                            if (candidate > 0)
                            {
                                _fwd.TruncateTo(candidate);
                                prefixLen = candidate;
                            }
                        }
                        else if (_fwd.SupportsSnapshot && _fwd.SnapshotLength > 0 && _prevTokens != null)
                        {
                            int snapLen = _fwd.SnapshotLength;
                            // The pair (snapshot @ snapLen, _prevTokens) is maintained as an
                            // invariant: on the canonical path both are written together immediately
                            // after CaptureSnapshot, on the legacy path both are written together at
                            // end-of-decode. snapLen must be a strict prefix of the new prompt (need
                            // at least one suffix token to drive the decoder) AND a token-level
                            // prefix of _prevTokens — the latter is the actual reuse precondition.
                            if (snapLen <= tokens.Length - 1 && snapLen <= _prevTokens.Length
                                && tokens.AsSpan(0, snapLen).SequenceEqual(_prevTokens.AsSpan(0, snapLen)))
                            {
                                _fwd.TruncateTo(snapLen);
                                prefixLen = snapLen;
                            }
                        }
                        if (prefixLen == 0)
                        {
                            _fwd.ResetCache();
                        }
                        else
                        {
                            // Issue #22 observability: account reused tokens regardless of
                            // mechanism (attention partial-rewind or GDN snapshot).
                            reusedTokens = prefixLen;
                            Interlocked.Add(ref _prefillTokensReused, prefixLen);
                        }

                        // Issue #212: from here the active slot's KV is mutated (ResetCache wiped
                        // it, or the upcoming prefill overwrites positions >= prefixLen). _prevTokens
                        // was already read above for the prefix match, so null the active slot's
                        // shadow now — a mid-decode throw/cancel skips the end-of-decode shadow
                        // write, and a stale shadow would let the next request SelectSlot-match a
                        // prefix whose KV was destroyed. Re-populated only on a complete decode.
                        if (_slotTokens is not null && activeSlotIdx >= 0)
                            _slotTokens[activeSlotIdx] = null;

                        // Prefill: process all prompt tokens (or just the suffix after the cached prefix).
                        //
                        // Issue #102 split: when a canonical-history boundary lies strictly between
                        // prefixLen and the end of the prompt, run the prefill in two stages with a
                        // CaptureSnapshot in between. That snapshot sits at the canonical boundary so
                        // the next turn — whose generating prompt starts with the same canonical
                        // history plus a scrubbed assistant response and a fresh user turn — can
                        // restore it. Issue #106: also applies on MTP runs; the sticky hidden
                        // history buffer + MTP KV soft-truncate make PrefillMtp(startPos=snapLen)
                        // viable. Skipped for backends that don't expose a snapshot (the snapshot
                        // call is a no-op, but skipping the split keeps the prefill in one shot
                        // for cache efficiency).
                        useCanonicalSnapshot =
                            canonicalLen > prefixLen
                            && canonicalLen < tokens.Length
                            && _fwd.SupportsSnapshot;

                        suffixTokens = prefixLen > 0 ? tokens[prefixLen..] : tokens;

                        if (useCanonicalSnapshot)
                        {
                            PrefillChunked(tokens, prefixLen, canonicalLen, ct);
                            _fwd.CaptureSnapshot();
                            // Pair _prevTokens with the snapshot atomically: stage-2 failure or
                            // mid-decode cancellation now leaves snapshot + _prevTokens consistent
                            // with each other (both describe this turn's canonical state). Without
                            // the immediate write, a stage-2 throw would leave the snapshot pointing
                            // at this turn's canonical while _prevTokens still described the prior
                            // turn — the next request could pass the snapshot-match check and
                            // TruncateTo to state that doesn't correspond to its prefix.
                            _prevTokens = tokens[..canonicalLen];
                            logits = PrefillChunked(tokens, canonicalLen, tokens.Length, ct);
                        }
                        else
                        {
                            if (suffixTokens.Length > 0)
                                logits = PrefillChunked(tokens, prefixLen, tokens.Length, ct);
                            else
                                logits = _fwd.Forward(tokens[^1], tokens.Length - 1);
                        }
                        startPos = tokens.Length;
                    }
                    prefillMs = swReq.ElapsedMilliseconds - prefillStartMs;

                    if (useDSpark)
                    {
                        // DSpark block-speculative decode (PR #413 Phase 6). Initialize
                        // copies the prefill logits and replays the tap buffer's rows
                        // [0, promptLen) into the draft's context — cached-prefix
                        // positions were captured by the earlier request that owns the
                        // matching KV (tap validity tracks KV validity: both describe
                        // the last processed sequence, and prefix reuse already
                        // guarantees the tokens match).
                        var dsDec = new DSparkDecoder(_fwd, _dspark!);
                        dsDec.Initialize(tokens.Length, logits);

                        var textDecDs = new Utf8StreamDecoder();
                        dsDec.Decode(sp.MaxNewTokens, stopIds.AsSpan(), tok =>
                        {
                            if (ttftMs < 0) ttftMs = swReq.ElapsedMilliseconds;
                            decodeTokens++;
                            fullSeq.Add(tok);
                            var bytes = _tokenizer.DecodeBytes(tok);
                            var chunk = textDecDs.Append(bytes);
                            if (chunk.Length > 0)
                                channel.Writer.TryWrite(
                                    new GenerateChunk(GenerateChunkKind.Text, chunk));
                        },
                            minConfidence: DSparkDecoder.ResolveMinConfidence(-1f),
                            verifyLenCap: DSparkDecoder.ResolveVerifyLen(0),
                            ct: ct);

                        var textFlushDs = textDecDs.Flush();
                        if (textFlushDs.Length > 0)
                            channel.Writer.TryWrite(
                                new GenerateChunk(GenerateChunkKind.Text, textFlushDs));

                        if (Environment.GetEnvironmentVariable("SHARPI_TRACE_DSPARK") == "1"
                            && dsDec.TotalDraftsEmitted > 0)
                        {
                            Console.Error.WriteLine(
                                $"[InferenceEngine] DSpark: {dsDec.TotalDraftsAccepted}/{dsDec.TotalDraftsEmitted} " +
                                $"drafts accepted ({dsDec.AcceptanceRate:P1}); " +
                                $"phase ms draft={dsDec.DraftMs:F0} verify={dsDec.VerifyMs:F0} commit={dsDec.CommitMs:F0}");
                        }

                        // End-of-decode snapshot — see the non-MTP twin below for the
                        // !useCanonicalSnapshot rationale. DSpark requires a rewindable
                        // pass, so CaptureSnapshot is a no-op and only the _prevTokens
                        // pairing matters (prefix reuse for the next turn).
                        if (!useCanonicalSnapshot && !ct.IsCancellationRequested)
                        {
                            _fwd.CaptureSnapshot();
                            var snapDs = fullSeq.ToArray();
                            _prevTokens = snapDs;
                            if (_slotTokens is not null) _slotTokens[0] = snapDs;
                        }
                        // DSparkDecoder.Decode never emits the stop token itself, so
                        // decodeTokens reaching MaxNewTokens means budget exhaustion.
                        channel.Writer.TryWrite(new GenerateChunk(
                            GenerateChunkKind.Stop, "", TruncatedByMaxTokens: decodeTokens >= sp.MaxNewTokens));
                        channel.Writer.TryComplete();
                        return;
                    }

                    if (useMtp)
                    {
                        // Initialize captures the main logits + LastHidden BEFORE
                        // PrefillMtp's MtpForward calls overwrite the shared _logits
                        // scratch buffer (issue #33). PrefillMtp does not touch
                        // _lastHidden, so the captured hidden remains h_{N-1}.
                        var mtpDec = new MtpDecoder(_fwd);
                        mtpDec.Initialize(tokens.Length, logits);

                        // Issue #33: populate the MTP KV cache for the prompt so the
                        // first decode-step MTP attention sees the full prompt context.
                        // ~1.6%/token overhead — only paid on MTP-enabled runs.
                        if (suffixTokens.Length > 0)
                            PrefillMtpChunked(tokens, prefixLen, tokens.Length, ct);

                        var textDecMtp = new Utf8StreamDecoder();

                        mtpDec.Decode(sp.MaxNewTokens, stopIds.AsSpan(), tok =>
                        {
                            if (ttftMs < 0) ttftMs = swReq.ElapsedMilliseconds;
                            decodeTokens++;
                            fullSeq.Add(tok);
                            var bytes = _tokenizer.DecodeBytes(tok);
                            var chunk = textDecMtp.Append(bytes);
                            if (chunk.Length > 0)
                                channel.Writer.TryWrite(
                                    new GenerateChunk(GenerateChunkKind.Text, chunk));
                        }, pMin: sp.SpecDraftPMin, draftN: mtpDraftN, ct: ct);

                        var textFlushMtp = textDecMtp.Flush();
                        if (textFlushMtp.Length > 0)
                            channel.Writer.TryWrite(
                                new GenerateChunk(GenerateChunkKind.Text, textFlushMtp));

                        if (Environment.GetEnvironmentVariable("SHARPI_TRACE_MTP") == "1"
                            && mtpDec.TotalDraftsEmitted > 0)
                        {
                            Console.Error.WriteLine(
                                $"[InferenceEngine] MTP: {mtpDec.TotalDraftsAccepted}/{mtpDec.TotalDraftsEmitted} " +
                                $"drafts accepted ({mtpDec.AcceptanceRate:P1}); " +
                                $"phase ms draft={mtpDec.DraftMs:F0} verify={mtpDec.VerifyMs:F0} commit={mtpDec.CommitMs:F0}");
                        }

                        // End-of-decode snapshot — see the non-MTP twin below for the
                        // !useCanonicalSnapshot rationale.
                        if (!useCanonicalSnapshot && !ct.IsCancellationRequested)
                        {
                            _fwd.CaptureSnapshot();
                            var snapMtp = fullSeq.ToArray();
                            _prevTokens = snapMtp;
                            // Issue #212: MTP ran single-slot on slot 0 (owned) — record the
                            // transcript against slot 0 so the next turn's SelectSlot can rematch.
                            if (_slotTokens is not null) _slotTokens[0] = snapMtp;
                        }
                        // MtpDecoder.Decode never invokes emitToken for the stop token itself
                        // (documented contract), so decodeTokens reaching MaxNewTokens here means
                        // the budget was exhausted before a stop token was seen.
                        channel.Writer.TryWrite(new GenerateChunk(
                            GenerateChunkKind.Stop, "", TruncatedByMaxTokens: decodeTokens >= sp.MaxNewTokens));
                        channel.Writer.TryComplete();
                        return;
                    }

                    // Decode loop. Separate stateful UTF-8 decoders for the answer stream and
                    // the thinking stream so multi-byte characters in either stream reassemble
                    // independently — the two streams never share decoder state.
                    var textDec = new Utf8StreamDecoder();
                    var thinkDec = new Utf8StreamDecoder();

                    // Seed inThinking from the prompt itself (issue #92). Qwen3.6 and other
                    // reasoning models append a bare `<think>` token to the generation prompt
                    // via their chat template, so the model is already inside a reasoning
                    // block before the first sampled token. Without this scan the engine
                    // would route the model's "Here's a thinking process:" preamble into
                    // the content stream instead of the reasoning stream.
                    bool inThinking = false;
                    if (thinkingEnabled)
                    {
                        foreach (int tok in tokens)
                        {
                            if (tok == thinkId) inThinking = true;
                            else if (tok == endThinkId) inThinking = false;
                        }
                    }
                    int thinkingCount = 0;

                    // #219: on the pure-greedy path (temp 0, no logit bias) with a pass that can
                    // argmax on-device, carry just the next token id instead of downloading the full
                    // vocab logits every step. gpuNext holds the argmax of the most recent forward;
                    // the first comes from the prefill logits already on the host.
                    // A constraint masks logits before sampling, so the device-side argmax fast path
                    // (which never downloads the full vocab) can't be used while one is attached.
                    bool useGpuArgmax = sp.Temperature <= 0f
                        && _fwd.SupportsGpuArgmax
                        && sp.LogitBias is not { Count: > 0 }
                        && constraint is null;
                    int gpuNext = useGpuArgmax ? Sampler.Greedy(logits) : 0;

                    // Assume the budget was exhausted unless the loop finds an explicit stop
                    // token below; that's the only other way out of the loop (cancellation
                    // throws and skips the Stop-chunk write entirely).
                    bool hitMaxTokens = true;

                    for (int i = 0; i < sp.MaxNewTokens; i++)
                    {
                        ct.ThrowIfCancellationRequested();

                        // Force-inject </think> when the model has burned through its reasoning
                        // budget. Mirrors the CLI's --max-thinking-tokens path: the forced close
                        // token still routes through the boundary branch below and is fed back
                        // into forward(...) so the model continues from its post-think state.
                        int next;
                        if (inThinking && sp.MaxThinkingTokens > 0 && thinkingCount >= sp.MaxThinkingTokens
                            && thinkingEnabled && endThinkId > 0)
                        {
                            next = endThinkId;
                        }
                        else if (useGpuArgmax)
                        {
                            next = gpuNext;
                        }
                        else
                        {
                            // While the constraint is inside a constrained region, sample from the
                            // masked logits (grammar-forbidden tokens set to -inf); otherwise sample
                            // the raw logits exactly as the unconstrained path would.
                            var sampleLogits = constraint is { IsConstraining: true }
                                ? constraint.Filter(logits)
                                : logits;
                            next = sp.Temperature <= 0f
                                ? Sampler.Greedy(sampleLogits)
                                : Sampler.Sample(sampleLogits, sp, rng);
                        }

                        // Advance the grammar constraint with the just-chosen token (every token,
                        // so it can detect a tool-call boundary and begin/end constraining).
                        constraint?.Accept(next);

                        // Record every emitted/consumed token (stop tokens included) so the
                        // post-decode snapshot of _prevTokens reflects the full transcript.
                        // Issue #21: chat-continuation prompts for turn N+1 typically extend
                        // turn N's full transcript, not just turn N's prompt.
                        fullSeq.Add(next);
                        if (ttftMs < 0) ttftMs = swReq.ElapsedMilliseconds;
                        decodeTokens++;

                        if (stopIds.Contains(next)) { hitMaxTokens = false; break; }

                        // Counter update mirrors RunCommand.DecodeLoop: reset on each <think>
                        // open, otherwise increment whenever inThinking was true on entry to
                        // this iteration. That includes the </think> boundary token itself —
                        // so N content tokens of reasoning trips the force-close on iteration N+1.
                        if (thinkingEnabled && next == thinkId) thinkingCount = 0;
                        else if (inThinking) thinkingCount++;

                        // Reasoning boundary tokens are ALWAYS consumed and never emitted (the
                        // documented contract). State flips only on a valid transition, so a
                        // malformed double-open or an orphan close is swallowed instead of leaking
                        // its literal marker as text — e.g. Gemma 4's post-tool generation prompt
                        // primes <|channel>, so the answer pass can emit a bare <channel|> close
                        // with no preceding open (issue #304).
                        if (thinkingEnabled && next == thinkId)
                        {
                            if (!inThinking)
                            {
                                // Flush any pending text bytes before entering thinking mode.
                                var textTail = textDec.Flush();
                                if (textTail.Length > 0)
                                    channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, textTail));
                                inThinking = true;
                            }
                            if (useGpuArgmax) gpuNext = _fwd.ForwardArgmax(next, startPos + i).Token;
                            else logits = _fwd.Forward(next, startPos + i);
                            continue;
                        }
                        if (thinkingEnabled && next == endThinkId)
                        {
                            if (inThinking)
                            {
                                // Flush thinking-decoder tail as a final Thinking chunk.
                                var thinkTail = thinkDec.Flush();
                                if (thinkTail.Length > 0)
                                    channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, thinkTail));
                                inThinking = false;
                            }
                            if (useGpuArgmax) gpuNext = _fwd.ForwardArgmax(next, startPos + i).Token;
                            else logits = _fwd.Forward(next, startPos + i);
                            continue;
                        }

                        var bytes = _tokenizer.DecodeBytes(next);
                        if (inThinking)
                        {
                            var chunk = thinkDec.Append(bytes);
                            if (chunk.Length > 0)
                                channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, chunk));
                        }
                        else
                        {
                            var chunk = textDec.Append(bytes);
                            if (chunk.Length > 0)
                                channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, chunk));
                        }

                        if (useGpuArgmax) gpuNext = _fwd.ForwardArgmax(next, startPos + i).Token;
                        else logits = _fwd.Forward(next, startPos + i);
                    }

                    // End-of-loop: flush both decoders defensively (whichever was active).
                    var textFlush = textDec.Flush();
                    if (textFlush.Length > 0)
                        channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Text, textFlush));
                    var thinkFlush = thinkDec.Flush();
                    if (thinkFlush.Length > 0)
                        channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Thinking, thinkFlush));

                    // End-of-decode snapshot for the legacy path only — the canonical path
                    // already captured + paired _prevTokens during the split prefill above.
                    // ct.ThrowIfCancellationRequested earlier ensures we don't reach here on a
                    // cancelled token; skip capture in that case to avoid stale state.
                    // Image requests run fresh each time (no prefix reuse) and their positions
                    // don't correspond to plain token positions, so don't snapshot them.
                    if (!useCanonicalSnapshot && !ct.IsCancellationRequested && imageBytes is null)
                    {
                        _fwd.CaptureSnapshot();
                        var snap = fullSeq.ToArray();
                        _prevTokens = snap;
                        // Issue #212: record the transcript against the active slot so the next
                        // turn's SelectSlot can rematch it (slot 0 = owned long prefix, slot 1 =
                        // scratch). When no non-owned slot is bound (single-slot, MTP), this is
                        // skipped and _prevTokens carries the state as before.
                        if (activeSlotIdx >= 0) _slotTokens![activeSlotIdx] = snap;
                    }

                    channel.Writer.TryWrite(new GenerateChunk(GenerateChunkKind.Stop, "", TruncatedByMaxTokens: hitMaxTokens));
                    channel.Writer.TryComplete();
                }
                catch (Exception ex)
                {
                    channel.Writer.TryComplete(ex);
                }
                finally
                {
                    // Issue #212: always release a bound non-owned slot, even on exception or
                    // cancellation, so a scratch bind can't leak into the next request's
                    // owned-cache state (mirrors how RestoreOwned is always in a finally).
                    if (activeSlotIdx >= 0) _multiSlot!.DeactivateSlot();
                    if (logReq)
                    {
                        long totalMs = swReq.ElapsedMilliseconds;
                        int newTokens = promptTokenCount - reusedTokens;
                        double prefTps = prefillMs > 0 ? newTokens * 1000.0 / prefillMs : 0;
                        // Decode wall = everything after the first token landed. total-ttft
                        // also folds in the end-of-decode snapshot, a negligible tail.
                        double decTps = (ttftMs >= 0 && totalMs > ttftMs)
                            ? decodeTokens * 1000.0 / (totalMs - ttftMs) : 0;
                        logger!.LogDebug(
                            "request prompt={PromptTokens}tok (reused={ReusedTokens} new={NewTokens}) " +
                            "encode={EncodeMs}ms prefill={PrefillMs}ms ({PrefillTps:F0} tok/s) " +
                            "ttft={TtftMs}ms decode={DecodeTokens}tok ({DecodeTps:F0} tok/s) total={TotalMs}ms",
                            promptTokenCount, reusedTokens, newTokens,
                            encodeMs, prefillMs, prefTps,
                            ttftMs >= 0 ? ttftMs : totalMs, decodeTokens, decTps, totalMs);
                    }
                }
            });

            try
            {
                await foreach (var chunk in channel.Reader.ReadAllAsync(ct).ConfigureAwait(false))
                    yield return chunk;
            }
            finally
            {
                // Issue #109: the background generation task drives the (non-thread-safe)
                // forward pass + shared KV cache. We must wait for it to fully stop before
                // releasing the gate — otherwise a cancelled/abandoned consumer (e.g. an
                // agentic client that disconnects mid-decode and immediately fires the next
                // request) would throw out of the await foreach above and release the gate
                // while genTask is still inside _fwd.Forward(...), letting the next request
                // reset/prefill the same cache concurrently and corrupt it (observed hang).
                // genTask routes all exceptions through the channel (surfaced by the foreach),
                // so awaiting it here only re-throws on the unobserved cancellation path,
                // which we swallow — the original cause already propagates from the foreach.
                try
                {
                    await genTask.ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                }
            }
        }
        finally
        {
            Interlocked.Decrement(ref _activeCount);
            _gate.Release();
        }
    }

    /// <summary>
    /// Synchronous dispose. Prefer <see cref="DisposeAsync"/> on async shutdown paths
    /// (e.g. a Generic Host's <c>StopAsync</c>); this overload blocks the calling thread
    /// while the in-flight generation drains.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // Stop any in-flight worker, then wait for it to release the gate before freeing
        // _fwd — see DrainNote on DisposeAsync.
        _shutdownCts.Cancel();
        // Don't release after acquiring: we're about to dispose the gate, and releasing it
        // would let a queued waiter whose cancellation hasn't yet propagated slip in and
        // touch _fwd as it's freed. Holding the permit guarantees exclusivity through teardown.
        bool drained = _gate.Wait(_disposeDrainTimeout);
        DisposeCore(drained);
    }

    /// <summary>
    /// Asynchronous dispose. Signals shutdown and awaits the in-flight generation worker's
    /// completion before freeing the engine-owned <see cref="IForwardPass"/>, so a host
    /// that cancels and disposes mid-decode gets a clean teardown instead of an access
    /// violation from the worker touching freed KV-cache memory (issue #132).
    /// </summary>
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _shutdownCts.Cancel();
        // DrainNote: the gate is held for the entire lifetime of a GenerateChunksAsync call
        // — across the background worker AND its drain (the finally awaits genTask before
        // releasing) — so re-acquiring it guarantees no worker is still inside _fwd.Forward /
        // the shared PagedKvCache. Cancelling _shutdownCts above makes the active worker exit
        // at its next per-token / per-prefill-chunk cancellation checkpoint, so this normally
        // returns in well under one forward. The timeout is only a backstop against a wedged
        // backend or a consumer that abandoned its enumerator without disposing it (which
        // would otherwise strand the gate forever).
        // See Dispose(): keep the permit held through teardown rather than releasing it.
        bool drained = await _gate.WaitAsync(_disposeDrainTimeout).ConfigureAwait(false);
        DisposeCore(drained);
    }

    /// <summary>
    /// Frees engine-owned resources. <paramref name="drained"/> is <c>false</c> only when the
    /// dispose drain timed out — i.e. a worker may still be live inside <c>_fwd</c> / the KV
    /// cache. In that case we deliberately leak the forward pass and owned handles rather than
    /// free them out from under the worker, which is the very access violation (#132) this fix
    /// exists to prevent: a leaked allocation on a shutdown that is already wedged is strictly
    /// better than a native crash.
    /// </summary>
    private void DisposeCore(bool drained)
    {
        if (!drained)
        {
            Console.Error.WriteLine(
                $"[InferenceEngine] dispose timed out after {_disposeDrainTimeout.TotalSeconds:0}s waiting " +
                "for the in-flight generation to drain; leaking the forward pass instead of freeing it under a " +
                "live worker. Ensure consumers dispose their generation enumerators and that the backend is responsive.");
            return;
        }
        // Issue #302: stop the dedicated forward-pass thread before freeing the pass it drives.
        // Safe here because `drained` means the gate was re-acquired — no work item is running,
        // and request admission is gated by that same gate, so none can be enqueued after this.
        // CompleteAdding ends the thread's consuming loop; Join awaits its exit (it is idle, so
        // this returns at once) before the forward pass is disposed below.
        _engineWork.CompleteAdding();
        _engineThread.Join();
        _engineWork.Dispose();

        // Issue #302: the teardown below frees thread-affine backend resources (CUDA device
        // buffers / streams / modules) on this disposing thread, not the now-exited engine
        // thread. Bind the context here too, so a host that disposes from an unbound pool thread
        // (e.g. StopAsync) frees cleanly instead of failing on an invalid current context. The
        // engine thread has already been joined, so the context is no longer current there.
        // No-op on CPU/Vulkan; idempotent. Best-effort: a bind failure here must not abort
        // teardown (the frees below may then fail, but we still free everything we can).
        try { _fwd.BindToCurrentThread(); }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"[InferenceEngine] could not bind backend context for teardown ({ex.GetType().Name}: {ex.Message}); " +
                "freeing anyway.");
        }

        // DSpark draft (PR #413): a CUDA draft frees device buffers through the shared
        // backend, so dispose it while the backend is still live (before _fwd/_owned).
        _dspark?.Dispose();

        // Issue #212: dispose the scratch slot(s) the engine allocated (slot 0 is the pass's
        // OwnedSlot — owned/freed by _fwd.Dispose, never disposed here). Done before _fwd.Dispose
        // so the scratch tensors free while the backend is still live.
        if (_slots is not null)
            for (int s = 1; s < _slots.Length; s++)
                _slots[s].Dispose();

        _fwd.Dispose();
        foreach (var d in _owned)
            d.Dispose();
        _gate.Dispose();
        _shutdownCts.Dispose();
    }
}
