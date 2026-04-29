using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Pipeline;
using SharpInference.TurboQuant;
using SharpInference.Vulkan;

namespace SharpInference.Engine;

/// <summary>
/// Hybrid GPU/CPU forward pass for models larger than VRAM.
/// First N layers run on GPU (Vulkan compute shaders), remaining layers on CPU (AVX2 SIMD).
/// Hidden state transfers via pinned host memory at GPU↔CPU boundaries.
/// </summary>
public sealed unsafe class HybridForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly VulkanBackend _gpu;
    private readonly ModelHyperparams _hp;
    private readonly LayerPlacement _placement;

    // Dimensions
    private readonly int _embDim, _headDim, _numHeads, _numKvHeads, _headsPerKvGroup, _intermDim, _expertDim;
    private readonly int _nGpuLayers, _nCpuLayers;

    // ── GPU resources (layers 0..nGpuLayers-1) ──
    private readonly Tensor _gpuHidden, _gpuResidual, _gpuNormBuf;
    private readonly Tensor _gpuQ, _gpuK, _gpuV, _gpuAttnOut;
    private readonly Tensor _gpuFfnGate, _gpuFfnUp;
    private readonly Tensor _gpuLogits;
    private readonly Tensor? _gpuRouterLogits, _gpuMoeSharedOut, _gpuMoeExpertOut;
    private readonly Tensor? _gpuEmbedding, _gpuOutputWeight, _gpuOutputNorm;
    private readonly bool _embIsQuantized;
    private readonly Tensor[] _gpuAttnNorm, _gpuWq, _gpuWk, _gpuWv, _gpuWo;
    private readonly Tensor[] _gpuFfnNorm, _gpuWGate, _gpuWUp, _gpuWDown;
    private readonly Tensor[]? _gpuWGateInp, _gpuWGateShexp, _gpuWUpShexp, _gpuWDownShexp;
    private readonly Tensor[]? _gpuBq, _gpuBk, _gpuBv, _gpuBo;
    private readonly Tensor[]? _gpuQNorm, _gpuKNorm;
    private readonly Tensor[] _gpuKCache, _gpuVCache;
    private readonly Tensor[]? _gpuTqKCache, _gpuTqVCache, _gpuSignPatterns;
    private readonly Tensor? _gpuCodebook, _gpuBoundaries, _gpuRotatedQ, _gpuEvictK, _gpuEvictV;
    private readonly Dictionary<nint, DType> _gpuWeightDTypes = new();

    // ── CPU resources (layers nGpuLayers..numLayers-1) ──
    private readonly float* _cpuHidden, _cpuResidual, _cpuNormBuf;
    private readonly float* _cpuQ, _cpuK, _cpuV, _cpuAttnOut;
    private readonly float* _cpuFfnGate, _cpuFfnUp;
    private readonly float* _cpuAttnScores;
    private readonly CpuWeightRef[] _cpuAttnNorm, _cpuWq, _cpuWk, _cpuWv, _cpuWo;
    private readonly CpuWeightRef[] _cpuFfnNorm, _cpuWGate, _cpuWUp, _cpuWDown;
    private readonly float*[] _cpuBq, _cpuBk, _cpuBv, _cpuBo;
    private readonly float*[] _cpuQNorm, _cpuKNorm;
    private readonly CpuWeightRef[]? _cpuWGateInp, _cpuWGateShexp, _cpuWUpShexp, _cpuWDownShexp;
    private readonly CpuWeightRef[]? _cpuWGateExps, _cpuWUpExps, _cpuWDownExps;
    private readonly CpuWeightRef _cpuEmbedding, _cpuOutputWeight, _cpuOutputNorm;
    private readonly float* _cpuRouterLogits;
    private readonly float* _cpuSharedOut;
    private readonly float* _cpuExpertGate;
    private readonly float* _cpuExpertUp;
    private readonly KvCache _cpuKvCache;
    private readonly TurboQuantKvCache? _cpuTqKvCache;
    private readonly float* _cpuRotatedQuery; // scratch for TQ query rotation [headDim]
    private readonly float* _cpuDecompBuf;    // scratch for TQ value decompress [headDim]
    private readonly Dictionary<string, nint> _cpuNormCache = new();

    // ── Shared ──
    private readonly Tensor _pinnedHidden; // host-visible buffer for GPU↔CPU transfer
    private readonly float[] _logitsBuf;
    private readonly float[]? _gpuRouterBuf;
    private readonly bool _hasAttnBias, _hasQkNorm, _isMoE, _hasSharedExpert;
    private readonly bool _cpuEmbeddingOutputOnly;
    private readonly bool _tqEnabled;
    private readonly int _tqFp32Window;
    private readonly int _tqBlockBytes;
    private int _gpuTqCompressedLen;
    private int _gpuFp32WriteIdx;
    private int _gpuFp32Count;
    private int _kvLength;
    private readonly int _maxSeqLen;

    // Precomputed RoPE cos/sin tables for CPU layers [maxSeqLen * halfDim]
    private readonly float* _ropeCosTable;
    private readonly float* _ropeSinTable;
    private readonly int _ropeHalfDim;

    // ── Expert slot cache (for MoE GPU layers with lazy/evictable expert loading) ──
    private ExpertSlotManager? _expertSlotManager;
    private MoEPrefetcher? _prefetcher;
    // Pinned host-visible GPU tensor for uploading CPU fallback contributions to GPU hidden state.
    private Tensor? _gpuFallbackContrib;
    // Pinned host-visible GPU tensor for reading the norm buffer on CPU without a separate Download.
    // The GPU session copies _gpuNormBuf into this buffer before EndRecordAndSubmit, so MapPinned
    // after the submit gives the CPU a zero-copy view of the norm data.
    private Tensor? _gpuPinnedNorm;
    // Lazily-allocated CPU scratch arrays for the expert cache-miss CPU fallback path.
    private float[]? _cpuFallbackBuf;  // [embDim] accumulated contribution from missed experts
    private float[]? _cpuFallbackGate; // [expertDim] scratch for expert gate projection
    private float[]? _cpuFallbackUp;   // [expertDim] scratch for expert up projection

    public int MaxSeqLen => _maxSeqLen;
    public LayerPlacement Placement => _placement;

    /// <summary>Vocabulary size of this model.</summary>
    public int VocabSize => _hp.VocabSize;

    /// <summary>
    /// Truncate the KV cache to the given length, discarding positions >= length.
    /// Used by speculative decoding to rewind rejected draft tokens.
    /// </summary>
    public void TruncateTo(int length)
    {
        _kvLength = length;
        _gpuTqCompressedLen = Math.Min(_gpuTqCompressedLen, length);
        if (_cpuTqKvCache != null)
            _cpuTqKvCache.TruncateTo(length);
        else
            _cpuKvCache.TruncateTo(length);
    }

    public HybridForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp,
        LayerPlacement placement, bool enableTq = false, int tqFp32Window = 256, int tqBits = 3,
        int expertSlotCapacity = -1)
    {
        _model = model;
        _gpu = gpu;
        _hp = hp;
        _placement = placement;
        _nGpuLayers = placement.GpuLayers;
        _nCpuLayers = placement.CpuLayers;
        _maxSeqLen = placement.RecommendedCtxSize;

        _embDim = hp.EmbeddingDim;
        _headDim = hp.HeadDim;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = hp.NumHeads / hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;
        _expertDim = hp.IsMoE ? hp.ExpertIntermediateDim : hp.IntermediateDim;
        _hasAttnBias = hp.HasAttnBias;
        _hasQkNorm = hp.HasQkNorm;
        _isMoE = hp.IsMoE;
        _hasSharedExpert = hp.HasSharedExpert;
        _cpuEmbeddingOutputOnly = ShouldKeepFixedWeightsOnCpu(
            model.FindTensor("token_embd.weight")!.Value,
            model.FindTensor("output.weight"));
        _tqEnabled = enableTq;
        if (_tqEnabled && _headDim is not 128 and not 256)
            throw new NotSupportedException($"TurboQuant currently supports head dimensions 128 and 256; model head dim is {_headDim}.");
        _tqFp32Window = enableTq ? Math.Min(tqFp32Window, _maxSeqLen) : 0;
        _tqBlockBytes = enableTq ? TurboQuantOps.BlockSize(tqBits, _headDim) : 0;
        _gpuRouterBuf = _isMoE && _nGpuLayers > 0 ? new float[hp.NumExperts] : null;

        Console.Error.WriteLine($"[HybridForwardPass] {placement.Summary()}{(enableTq ? $" [TQ{tqBits}]" : "")}");

        // ── Allocate GPU scratch buffers ──
        _gpuHidden = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuResidual = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuNormBuf = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuQ = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
        _gpuK = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        _gpuV = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        _gpuAttnOut = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
        int gpuFfnScratch = Math.Max(_intermDim, _expertDim);
        _gpuFfnGate = gpu.Allocate(TensorShape.D1(gpuFfnScratch));
        _gpuFfnUp = gpu.Allocate(TensorShape.D1(gpuFfnScratch));
        _gpuLogits = gpu.Allocate(TensorShape.D1(hp.VocabSize));
        _gpuRouterLogits = _isMoE && _nGpuLayers > 0 ? gpu.Allocate(TensorShape.D1(hp.NumExperts)) : null;
        _gpuMoeSharedOut = _isMoE && _hasSharedExpert && _nGpuLayers > 0 ? gpu.Allocate(TensorShape.D1(_embDim)) : null;
        _gpuMoeExpertOut = _isMoE && _nGpuLayers > 0 ? gpu.Allocate(TensorShape.D1(_embDim)) : null;
        _logitsBuf = new float[hp.VocabSize];

        // Pinned buffer for hidden state transfer (embDim floats)
        _pinnedHidden = gpu.AllocatePinned(TensorShape.D1(_embDim));

        _cpuEmbedding = ResolveCpuWeight("token_embd.weight");
        _cpuOutputNorm = ResolveCpuWeight("output_norm.weight");
        _cpuOutputWeight = model.FindTensor("output.weight") is not null
            ? ResolveCpuWeight("output.weight")
            : _cpuEmbedding;

        // ── Upload GPU weights (embedding + output + first N layers) ──
        if (_cpuEmbeddingOutputOnly)
        {
            _gpuEmbedding = null;
            _gpuOutputNorm = null;
            _gpuOutputWeight = null;
            _embIsQuantized = false;
        }
        else
        {
            _gpuEmbedding = UploadWeight("token_embd.weight");
            _embIsQuantized = model.FindTensor("token_embd.weight")!.Value.DType == DType.Q4_K;
            _gpuWeightDTypes[_gpuEmbedding.Handle] = _embIsQuantized ? DType.Q4_K : DType.Float32;
            _gpuOutputNorm = UploadWeight("output_norm.weight");
            _gpuOutputWeight = model.FindTensor("output.weight") is not null
                ? UploadWeight("output.weight")
                : _gpuEmbedding;
        }

        _gpuAttnNorm = new Tensor[_nGpuLayers];
        _gpuWq = new Tensor[_nGpuLayers]; _gpuWk = new Tensor[_nGpuLayers];
        _gpuWv = new Tensor[_nGpuLayers]; _gpuWo = new Tensor[_nGpuLayers];
        _gpuFfnNorm = new Tensor[_nGpuLayers];
        _gpuWGate = new Tensor[_nGpuLayers]; _gpuWUp = new Tensor[_nGpuLayers]; _gpuWDown = new Tensor[_nGpuLayers];
        _gpuWGateInp = _isMoE ? new Tensor[_nGpuLayers] : null;
        _gpuWGateShexp = _isMoE && _hasSharedExpert ? new Tensor[_nGpuLayers] : null;
        _gpuWUpShexp = _isMoE && _hasSharedExpert ? new Tensor[_nGpuLayers] : null;
        _gpuWDownShexp = _isMoE && _hasSharedExpert ? new Tensor[_nGpuLayers] : null;

        if (_hasAttnBias) { _gpuBq = new Tensor[_nGpuLayers]; _gpuBk = new Tensor[_nGpuLayers]; _gpuBv = new Tensor[_nGpuLayers]; _gpuBo = new Tensor[_nGpuLayers]; }
        if (_hasQkNorm) { _gpuQNorm = new Tensor[_nGpuLayers]; _gpuKNorm = new Tensor[_nGpuLayers]; }

        int kvDim = _numKvHeads * _headDim;
        _gpuKCache = new Tensor[_nGpuLayers];
        _gpuVCache = new Tensor[_nGpuLayers];
        long tqUintsPerLayer = 0;
        if (_tqEnabled)
        {
            int maxTqPositions = Math.Max(0, _maxSeqLen - _tqFp32Window);
            long tqBytesPerPos = (long)_numKvHeads * _tqBlockBytes;
            tqUintsPerLayer = (maxTqPositions * tqBytesPerPos + 3) / 4;
            _gpuTqKCache = new Tensor[_nGpuLayers];
            _gpuTqVCache = new Tensor[_nGpuLayers];
            _gpuSignPatterns = new Tensor[_nGpuLayers];
            var centroids = TurboQuantCodebooks.GetCentroids(tqBits, _headDim).ToArray();
            _gpuCodebook = gpu.Upload(centroids, TensorShape.D1(centroids.Length));
            var boundaries = TurboQuantCodebooks.GetBoundaries(tqBits, _headDim).ToArray();
            _gpuBoundaries = gpu.Upload(boundaries, TensorShape.D1(boundaries.Length));
            _gpuRotatedQ = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
            _gpuEvictK = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
            _gpuEvictV = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        }

        Console.Error.Write($"[HybridForwardPass] Uploading {_nGpuLayers} GPU layers...");
        for (int i = 0; i < _nGpuLayers; i++)
        {
            _gpuAttnNorm[i] = UploadWeight($"blk.{i}.attn_norm.weight");
            _gpuWq[i] = UploadWeight($"blk.{i}.attn_q.weight");
            _gpuWk[i] = UploadWeight($"blk.{i}.attn_k.weight");
            _gpuWv[i] = UploadWeight($"blk.{i}.attn_v.weight");
            _gpuWo[i] = UploadWeight($"blk.{i}.attn_output.weight");
            _gpuFfnNorm[i] = UploadWeight($"blk.{i}.ffn_norm.weight");
            if (_isMoE)
            {
                _gpuWGateInp![i] = UploadWeight($"blk.{i}.ffn_gate_inp.weight");
                // Expert weights (gate/up/down exps) are loaded lazily by ExpertSlotManager
                if (_hasSharedExpert)
                {
                    _gpuWGateShexp![i] = UploadWeight($"blk.{i}.ffn_gate_shexp.weight");
                    _gpuWUpShexp![i] = UploadWeight($"blk.{i}.ffn_up_shexp.weight");
                    _gpuWDownShexp![i] = UploadWeight($"blk.{i}.ffn_down_shexp.weight");
                }
            }
            else
            {
                _gpuWGate[i] = UploadWeight($"blk.{i}.ffn_gate.weight");
                _gpuWUp[i] = UploadWeight($"blk.{i}.ffn_up.weight");
                _gpuWDown[i] = UploadWeight($"blk.{i}.ffn_down.weight");
            }

            if (_hasAttnBias)
            {
                _gpuBq![i] = UploadWeight($"blk.{i}.attn_q.bias");
                _gpuBk![i] = UploadWeight($"blk.{i}.attn_k.bias");
                _gpuBv![i] = UploadWeight($"blk.{i}.attn_v.bias");
                _gpuBo![i] = UploadWeight($"blk.{i}.attn_output.bias");
            }
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _gpuQNorm![i] = UploadWeight($"blk.{i}.attn_q_norm.weight");
                _gpuKNorm![i] = UploadWeight($"blk.{i}.attn_k_norm.weight");
            }

            if (_tqEnabled)
            {
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_tqFp32Window * kvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_tqFp32Window * kvDim));
                _gpuTqKCache![i] = gpu.Allocate(TensorShape.D1(tqUintsPerLayer));
                _gpuTqVCache![i] = gpu.Allocate(TensorShape.D1(tqUintsPerLayer));
                _gpuSignPatterns![i] = UploadTqSignPatterns(i);
            }
            else
            {
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
            }
            Console.Error.Write(".");
        }
        Console.Error.WriteLine(" done.");

        // ── Create expert slot manager for GPU MoE layers ──
        if (_isMoE && _nGpuLayers > 0)
        {
            int totalExperts = hp.NumExperts * _nGpuLayers;
            int capacity = expertSlotCapacity > 0
                ? Math.Min(expertSlotCapacity, totalExperts)
                : totalExperts;
            _expertSlotManager = new ExpertSlotManager(gpu, model, hp, capacity, _gpuWeightDTypes);
            _prefetcher = new MoEPrefetcher(_expertSlotManager);
            _gpuFallbackContrib = gpu.AllocatePinned(TensorShape.D1(_embDim));
            _gpuPinnedNorm      = gpu.AllocatePinned(TensorShape.D1(_embDim));
            Console.Error.WriteLine($"[HybridForwardPass] MoE expert slot cache: {capacity} slots ({hp.NumExperts} experts × {_nGpuLayers} layers), SLRU lazy-load.");
        }

        // ── Resolve CPU weights (layers nGpuLayers..numLayers-1) ──
        _cpuHidden = Alloc(_embDim);
        _cpuResidual = Alloc(_embDim);
        _cpuNormBuf = Alloc(_embDim);
        _cpuQ = Alloc(_numHeads * _headDim);
        _cpuK = Alloc(_numKvHeads * _headDim);
        _cpuV = Alloc(_numKvHeads * _headDim);
        _cpuAttnOut = Alloc(_numHeads * _headDim);
        _cpuFfnGate = Alloc(_intermDim);
        _cpuFfnUp = Alloc(_intermDim);
        _cpuAttnScores = Alloc(_numHeads * _maxSeqLen);
        _cpuRouterLogits = _isMoE ? Alloc(hp.NumExperts) : null;
        _cpuSharedOut = _isMoE && _hasSharedExpert ? Alloc(_embDim) : null;
        _cpuExpertGate = _isMoE ? Alloc(_expertDim) : null;
        _cpuExpertUp = _isMoE ? Alloc(_expertDim) : null;

        // Precompute RoPE cos/sin tables for CPU layers
        _ropeHalfDim = _headDim / 2;
        _ropeCosTable = (float*)NativeMemory.Alloc((nuint)((long)_maxSeqLen * _ropeHalfDim * sizeof(float)));
        _ropeSinTable = (float*)NativeMemory.Alloc((nuint)((long)_maxSeqLen * _ropeHalfDim * sizeof(float)));
        SimdKernels.BuildRopeTable(_ropeCosTable, _ropeSinTable, _maxSeqLen, _headDim, hp.RopeTheta);

        _cpuAttnNorm = new CpuWeightRef[_nCpuLayers];
        _cpuWq = new CpuWeightRef[_nCpuLayers]; _cpuWk = new CpuWeightRef[_nCpuLayers];
        _cpuWv = new CpuWeightRef[_nCpuLayers]; _cpuWo = new CpuWeightRef[_nCpuLayers];
        _cpuFfnNorm = new CpuWeightRef[_nCpuLayers];
        _cpuWGate = new CpuWeightRef[_nCpuLayers]; _cpuWUp = new CpuWeightRef[_nCpuLayers]; _cpuWDown = new CpuWeightRef[_nCpuLayers];
        _cpuBq = new float*[_nCpuLayers]; _cpuBk = new float*[_nCpuLayers];
        _cpuBv = new float*[_nCpuLayers]; _cpuBo = new float*[_nCpuLayers];
        _cpuQNorm = new float*[_nCpuLayers]; _cpuKNorm = new float*[_nCpuLayers];
        if (_isMoE)
        {
            _cpuWGateInp = new CpuWeightRef[_nCpuLayers];
            _cpuWGateExps = new CpuWeightRef[_nCpuLayers];
            _cpuWUpExps = new CpuWeightRef[_nCpuLayers];
            _cpuWDownExps = new CpuWeightRef[_nCpuLayers];
            if (_hasSharedExpert)
            {
                _cpuWGateShexp = new CpuWeightRef[_nCpuLayers];
                _cpuWUpShexp = new CpuWeightRef[_nCpuLayers];
                _cpuWDownShexp = new CpuWeightRef[_nCpuLayers];
            }
        }

        for (int ci = 0; ci < _nCpuLayers; ci++)
        {
            int li = ci + _nGpuLayers; // actual layer index
            _cpuAttnNorm[ci] = ResolveCpuWeight($"blk.{li}.attn_norm.weight");
            _cpuWq[ci] = ResolveCpuWeight($"blk.{li}.attn_q.weight");
            _cpuWk[ci] = ResolveCpuWeight($"blk.{li}.attn_k.weight");
            _cpuWv[ci] = ResolveCpuWeight($"blk.{li}.attn_v.weight");
            _cpuWo[ci] = ResolveCpuWeight($"blk.{li}.attn_output.weight");
            _cpuFfnNorm[ci] = ResolveCpuWeight($"blk.{li}.ffn_norm.weight");
            if (_isMoE)
            {
                _cpuWGateInp![ci] = ResolveCpuWeight($"blk.{li}.ffn_gate_inp.weight");
                _cpuWGateExps![ci] = ResolveCpuWeight($"blk.{li}.ffn_gate_exps.weight");
                _cpuWUpExps![ci] = ResolveCpuWeight($"blk.{li}.ffn_up_exps.weight");
                _cpuWDownExps![ci] = ResolveCpuWeight($"blk.{li}.ffn_down_exps.weight");
                if (_hasSharedExpert)
                {
                    _cpuWGateShexp![ci] = ResolveCpuWeight($"blk.{li}.ffn_gate_shexp.weight");
                    _cpuWUpShexp![ci] = ResolveCpuWeight($"blk.{li}.ffn_up_shexp.weight");
                    _cpuWDownShexp![ci] = ResolveCpuWeight($"blk.{li}.ffn_down_shexp.weight");
                }
            }
            else
            {
                _cpuWGate[ci] = ResolveCpuWeight($"blk.{li}.ffn_gate.weight");
                _cpuWUp[ci] = ResolveCpuWeight($"blk.{li}.ffn_up.weight");
                _cpuWDown[ci] = ResolveCpuWeight($"blk.{li}.ffn_down.weight");
            }

            if (_hasAttnBias)
            {
                _cpuBq[ci] = LoadCpuBias($"blk.{li}.attn_q.bias", _numHeads * _headDim);
                _cpuBk[ci] = LoadCpuBias($"blk.{li}.attn_k.bias", _numKvHeads * _headDim);
                _cpuBv[ci] = LoadCpuBias($"blk.{li}.attn_v.bias", _numKvHeads * _headDim);
                _cpuBo[ci] = LoadCpuBias($"blk.{li}.attn_output.bias", _embDim);
            }
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _cpuQNorm[ci] = LoadCpuBias($"blk.{li}.attn_q_norm.weight", _headDim);
                _cpuKNorm[ci] = LoadCpuBias($"blk.{li}.attn_k_norm.weight", _headDim);
            }
        }

        _cpuKvCache = new KvCache(_nCpuLayers, _maxSeqLen, _numKvHeads, _headDim);

        // Pre-fault mmap pages for CPU layers: touch the first byte of each weight tensor
        // to ensure OS pages them into RAM before the first forward pass.
        if (_nCpuLayers > 0)
        {
            Console.Error.Write($"[HybridForwardPass] Pre-faulting CPU weight pages...");
            long touchSum = 0;
            IEnumerable<CpuWeightRef> weightsToTouch = _cpuWq.Concat(_cpuWk).Concat(_cpuWv).Concat(_cpuWo);
            if (_isMoE)
            {
                weightsToTouch = weightsToTouch
                    .Concat(_cpuWGateInp!)
                    .Concat(_cpuWGateExps!)
                    .Concat(_cpuWUpExps!)
                    .Concat(_cpuWDownExps!);
                if (_hasSharedExpert)
                {
                    weightsToTouch = weightsToTouch
                        .Concat(_cpuWGateShexp!)
                        .Concat(_cpuWUpShexp!)
                        .Concat(_cpuWDownShexp!);
                }
            }
            else
            {
                weightsToTouch = weightsToTouch
                    .Concat(_cpuWGate)
                    .Concat(_cpuWUp)
                    .Concat(_cpuWDown);
            }

            foreach (var wRef in weightsToTouch)
            {
                // Touch first and last cache line of each weight tensor
                touchSum += wRef.DataPtr[0];
                long size = wRef.Info.ByteSize;
                if (size > 64) touchSum += wRef.DataPtr[size - 1];
            }
            Console.Error.WriteLine($" done. (touch={touchSum})");
        }

        if (_tqEnabled && _nCpuLayers > 0)
        {
            _cpuTqKvCache = new TurboQuantKvCache(
                _nCpuLayers, _maxSeqLen, _numKvHeads, _headDim,
                _tqFp32Window, tqBits,
                layerIndexBase: _nGpuLayers, totalLayerCountForSeeds: _hp.NumLayers);
            _cpuRotatedQuery = Alloc(_numHeads * _headDim);
            _cpuDecompBuf = Alloc(_numHeads * _headDim);
        }
    }

    // ================================================================
    //  Forward Pass
    // ================================================================

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // ── Phase 1: GPU layers ──
        _gpu.BeginRecord();

        // Embed token on GPU when the embedding table fits there, otherwise
        // dequantize on CPU and upload just the hidden state row.
        if (_gpuEmbedding is not null)
        {
            if (_embIsQuantized)
                _gpu.EmbedLookupQ4K(_gpuEmbedding, _gpuHidden, (uint)token, (uint)_embDim);
            else
                _gpu.EmbedLookup(_gpuEmbedding, _gpuHidden, (uint)token, (uint)_embDim);
            _gpu.RecordBarrier();
        }
        else
        {
            float* pinned = _gpu.MapPinned(_pinnedHidden);
            CpuEmbedToken(token, pinned);
            _gpu.UnmapPinned(_pinnedHidden);
            CopyGpuBuffer(_gpuHidden, _pinnedHidden);
            _gpu.RecordBarrier();
        }

        for (int i = 0; i < _nGpuLayers; i++)
        {
            GpuLayer(i, position);
        }

        if (_tqEnabled && _nGpuLayers > 0)
        {
            if (_gpuFp32Count >= _tqFp32Window)
                _gpuTqCompressedLen++;

            _gpuFp32WriteIdx = (_gpuFp32WriteIdx + 1) % _tqFp32Window;
            if (_gpuFp32Count < _tqFp32Window)
                _gpuFp32Count++;
        }

        if (_nCpuLayers > 0)
        {
            // Download hidden state to pinned buffer.
            CopyGpuBuffer(_pinnedHidden, _gpuHidden);
            _gpu.RecordBarrier();
        }

        _gpu.EndRecordAndSubmit();

        if (_nCpuLayers > 0)
        {
            // ── Phase 2: Transfer GPU hidden → CPU ──
            float* pinned = _gpu.MapPinned(_pinnedHidden);
            new Span<float>(pinned, _embDim).CopyTo(new Span<float>(_cpuHidden, _embDim));
            _gpu.UnmapPinned(_pinnedHidden);

            // ── Phase 3: CPU layers ──
            for (int ci = 0; ci < _nCpuLayers; ci++)
            {
                CpuLayer(ci, position);
            }

            // Increment CPU KV cache
            if (_cpuTqKvCache != null)
                _cpuTqKvCache.IncrementPosition();
            else
                _cpuKvCache.IncrementPosition();

            if (_gpuOutputWeight is not null)
            {
                // ── Phase 4: Transfer CPU hidden → GPU ──
                pinned = _gpu.MapPinned(_pinnedHidden);
                new ReadOnlySpan<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(pinned, _embDim));
                _gpu.UnmapPinned(_pinnedHidden);

                // Upload pinned → GPU hidden, then final norm + output
                _gpu.BeginRecord();
                CopyGpuBuffer(_gpuHidden, _pinnedHidden);
                _gpu.RecordBarrier();
            }
            else
            {
                ComputeCpuOutput();
                _kvLength = position + 1;
                return _logitsBuf;
            }
        }
        else
        {
            if (_gpuOutputWeight is null)
            {
                CopyGpuBuffer(_pinnedHidden, _gpuHidden);
                _gpu.RecordBarrier();
                _gpu.EndRecordAndSubmit();

                float* pinned = _gpu.MapPinned(_pinnedHidden);
                new Span<float>(pinned, _embDim).CopyTo(new Span<float>(_cpuHidden, _embDim));
                _gpu.UnmapPinned(_pinnedHidden);

                ComputeCpuOutput();
                _kvLength = position + 1;
                return _logitsBuf;
            }

            _gpu.BeginRecord();
        }

        // ── Phase 5: Final norm + output projection on GPU ──
        _gpu.RecordBarrier();
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm!, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden);

        _gpu.RecordComputeToTransferBarrier();
        _gpu.RecordDownloadToStaging(_gpuLogits, _logitsBuf.Length);
        _gpu.EndRecordAndSubmit();
        _gpu.ReadFromStaging(_logitsBuf);

        _kvLength = position + 1;
        return _logitsBuf;
    }

    public void ResetCache()
    {
        _kvLength = 0;
        _gpuTqCompressedLen = 0;
        _gpuFp32WriteIdx = 0;
        _gpuFp32Count = 0;
        if (_cpuTqKvCache != null)
            _cpuTqKvCache.Reset();
        else
            _cpuKvCache.Reset();
    }

    /// <inheritdoc/>
    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = Forward(tokens[i], startPos + i);
        return logits;
    }

    // ================================================================
    //  GPU Layer (same pattern as GpuForwardPass)
    // ================================================================

    private void GpuLayer(int i, int position)
    {
        CopyGpuBuffer(_gpuResidual, _gpuHidden);
        _gpu.RecordBarrier();

        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuAttnNorm[i], _hp.RmsNormEps);
        _gpu.RecordBarrier();

        GpuMatMul(_gpuQ, _gpuWq[i], _gpuNormBuf);
        GpuMatMul(_gpuK, _gpuWk[i], _gpuNormBuf);
        GpuMatMul(_gpuV, _gpuWv[i], _gpuNormBuf);
        _gpu.RecordBarrier();

        if (_hasAttnBias)
        {
            _gpu.AddInPlace(_gpuQ, _gpuBq![i]);
            _gpu.AddInPlace(_gpuK, _gpuBk![i]);
            _gpu.AddInPlace(_gpuV, _gpuBv![i]);
            _gpu.RecordBarrier();
        }

        {
            // NoPE: skip RoPE for NoPE layers
            bool useRoPE = _hp.NoRopeLayerStep == 0
                || (i + 1) % _hp.NoRopeLayerStep != 0;
            if (useRoPE)
            {
                _gpu.RoPE(_gpuQ, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                _gpu.RoPE(_gpuK, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                _gpu.RecordBarrier();
            }

            // QK-norm: for L2 (Llama-4), only on RoPE layers per llama.cpp
            if (_hasQkNorm && (_hp.UseL2QkNorm ? useRoPE : true))
            {
                if (_hp.UseL2QkNorm)
                {
                    _gpu.HeadNormPure(_gpuQ, (uint)_numHeads, (uint)_headDim, _hp.RmsNormEps);
                    _gpu.HeadNormPure(_gpuK, (uint)_numKvHeads, (uint)_headDim, _hp.RmsNormEps);
                }
                else
                {
                    _gpu.HeadNorm(_gpuQ, _gpuQNorm![i], (uint)_numHeads, (uint)_headDim, _hp.RmsNormEps);
                    _gpu.HeadNorm(_gpuK, _gpuKNorm![i], (uint)_numKvHeads, (uint)_headDim, _hp.RmsNormEps);
                }
                _gpu.RecordBarrier();
            }
        }

        if (_tqEnabled)
        {
            int kvDim = _numKvHeads * _headDim;
            long rowBytes = (long)kvDim * sizeof(float);

            if (_gpuFp32Count >= _tqFp32Window)
            {
                CopyGpuBufferRegion(_gpuEvictK!, 0, _gpuKCache[i], (long)_gpuFp32WriteIdx * rowBytes, rowBytes);
                CopyGpuBufferRegion(_gpuEvictV!, 0, _gpuVCache[i], (long)_gpuFp32WriteIdx * rowBytes, rowBytes);
                _gpu.RecordBarrier();

                _gpu.TqKvAppend(_gpuEvictK!, _gpuEvictV!, _gpuTqKCache![i], _gpuTqVCache![i],
                    _gpuSignPatterns![i], _gpuCodebook!, _gpuBoundaries!,
                    (uint)kvDim, (uint)_headDim, (uint)_gpuTqCompressedLen,
                    (uint)_maxSeqLen, (uint)_numKvHeads, (uint)_tqBlockBytes);
                _gpu.RecordBarrier();
            }

            _gpu.KvAppend(_gpuK, _gpuV, _gpuKCache[i], _gpuVCache[i],
                (uint)kvDim, (uint)_gpuFp32WriteIdx, (uint)_tqFp32Window);
            _gpu.RecordBarrier();

            _gpu.TqRotateQuery(_gpuQ, _gpuRotatedQ!, _gpuSignPatterns![i],
                (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim);
            _gpu.RecordBarrier();

            uint fp32SeqLen = (uint)Math.Min(_gpuFp32Count + 1, _tqFp32Window);
            _gpu.TqAttention(_gpuQ, _gpuRotatedQ!, _gpuTqKCache![i], _gpuTqVCache![i],
                _gpuKCache[i], _gpuVCache[i], _gpuAttnOut, _gpuCodebook!,
                (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                (uint)_gpuTqCompressedLen, fp32SeqLen, (uint)_maxSeqLen, (uint)_tqBlockBytes);
        }
        else
        {
            _gpu.KvAppend(_gpuK, _gpuV, _gpuKCache[i], _gpuVCache[i],
                (uint)(_numKvHeads * _headDim), (uint)position, (uint)_maxSeqLen);
            _gpu.RecordBarrier();

            _gpu.Attention(_gpuQ, _gpuKCache[i], _gpuVCache[i], _gpuAttnOut,
                (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                (uint)(position + 1), (uint)_maxSeqLen);
        }
        _gpu.RecordBarrier();

        GpuMatMul(_gpuHidden, _gpuWo[i], _gpuAttnOut);
        if (_hasAttnBias)
        {
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_gpuHidden, _gpuBo![i]);
        }
        _gpu.RecordBarrier();
        _gpu.AddInPlace(_gpuHidden, _gpuResidual);
        _gpu.RecordBarrier();

        CopyGpuBuffer(_gpuResidual, _gpuHidden);
        _gpu.RecordBarrier();

        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuFfnNorm[i], _hp.RmsNormEps);
        _gpu.RecordBarrier();

        if (_isMoE)
            GpuMoeFfn(i);
        else
            GpuDenseFfn(i);
        _gpu.RecordBarrier();

        _gpu.AddInPlace(_gpuHidden, _gpuResidual);
        _gpu.RecordBarrier();
    }

    // ================================================================
    //  CPU Layer (same pattern as ForwardPass)
    // ================================================================

    private void CpuLayer(int ci, int position)
    {
        // Save residual
        new Span<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(_cpuResidual, _embDim));

        // Pre-attention RMS norm
        var normW = GetCpuNormWeight(_cpuAttnNorm[ci]);
        SimdKernels.RmsNorm(_cpuNormBuf, _cpuHidden, normW, _embDim, _hp.RmsNormEps);

        // Q/K/V projections
        SimdKernels.MatVec(_cpuQ, _cpuWq[ci].DataPtr, _cpuNormBuf, _numHeads * _headDim, _embDim, _cpuWq[ci].DType);
        SimdKernels.MatVec(_cpuK, _cpuWk[ci].DataPtr, _cpuNormBuf, _numKvHeads * _headDim, _embDim, _cpuWk[ci].DType);
        SimdKernels.MatVec(_cpuV, _cpuWv[ci].DataPtr, _cpuNormBuf, _numKvHeads * _headDim, _embDim, _cpuWv[ci].DType);

        if (_hasAttnBias)
        {
            SimdKernels.AddInPlace(_cpuQ, _cpuBq[ci], _numHeads * _headDim);
            SimdKernels.AddInPlace(_cpuK, _cpuBk[ci], _numKvHeads * _headDim);
            SimdKernels.AddInPlace(_cpuV, _cpuBv[ci], _numKvHeads * _headDim);
        }

        // NoPE: actual layer index = ci + _nGpuLayers
        int actualLayer = ci + _nGpuLayers;
        bool useRoPE = _hp.NoRopeLayerStep == 0
            || (actualLayer + 1) % _hp.NoRopeLayerStep != 0;

        if (useRoPE)
        {
            var cos = _ropeCosTable + (long)position * _ropeHalfDim;
            var sin = _ropeSinTable + (long)position * _ropeHalfDim;
            if (_hp.IsNeoxRope)
            {
                SimdKernels.ApplyRoPECachedNeox(_cpuQ, cos, sin, _numHeads, _headDim);
                SimdKernels.ApplyRoPECachedNeox(_cpuK, cos, sin, _numKvHeads, _headDim);
            }
            else
            {
                SimdKernels.ApplyRoPECached(_cpuQ, cos, sin, _numHeads, _headDim);
                SimdKernels.ApplyRoPECached(_cpuK, cos, sin, _numKvHeads, _headDim);
            }
        }

        // QK-norm: for L2 (Llama-4), only on RoPE layers per llama.cpp
        if (_hasQkNorm && (_hp.UseL2QkNorm ? useRoPE : true))
        {
            if (_hp.UseL2QkNorm)
            {
                PerHeadPureRmsNorm(_cpuQ, _numHeads, _headDim, _hp.RmsNormEps);
                PerHeadPureRmsNorm(_cpuK, _numKvHeads, _headDim, _hp.RmsNormEps);
            }
            else
            {
                PerHeadRmsNorm(_cpuQ, _cpuQNorm[ci], _numHeads, _headDim, _hp.RmsNormEps);
                PerHeadRmsNorm(_cpuK, _cpuKNorm[ci], _numKvHeads, _headDim, _hp.RmsNormEps);
            }
        }

        // KV cache append (ci = CPU layer index)
        if (_cpuTqKvCache != null)
        {
            _cpuTqKvCache.Append(ci,
                new ReadOnlySpan<float>(_cpuK, _numKvHeads * _headDim),
                new ReadOnlySpan<float>(_cpuV, _numKvHeads * _headDim));
        }
        else
        {
            _cpuKvCache.Append(ci,
                new ReadOnlySpan<float>(_cpuK, _numKvHeads * _headDim),
                new ReadOnlySpan<float>(_cpuV, _numKvHeads * _headDim));
        }

        // Attention
        if (_cpuTqKvCache != null)
            CpuTqAttention(ci, position);
        else
            CpuAttention(ci, position);

        // Output projection
        SimdKernels.MatVec(_cpuHidden, _cpuWo[ci].DataPtr, _cpuAttnOut, _embDim, _numHeads * _headDim, _cpuWo[ci].DType);
        if (_hasAttnBias)
            SimdKernels.AddInPlace(_cpuHidden, _cpuBo[ci], _embDim);

        // Residual
        SimdKernels.AddInPlace(_cpuHidden, _cpuResidual, _embDim);

        // Save residual for FFN
        new Span<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(_cpuResidual, _embDim));

        // Pre-FFN RMS norm
        var ffnNormW = GetCpuNormWeight(_cpuFfnNorm[ci]);
        SimdKernels.RmsNorm(_cpuNormBuf, _cpuHidden, ffnNormW, _embDim, _hp.RmsNormEps);

        if (_isMoE)
            CpuMoeFfn(ci);
        else
            CpuDenseFfn(ci);

        // Residual
        SimdKernels.AddInPlace(_cpuHidden, _cpuResidual, _embDim);
    }

    private void CpuDenseFfn(int ci)
    {
        SimdKernels.MatVecDual(_cpuFfnGate, _cpuWGate[ci].DataPtr, _cpuFfnUp, _cpuWUp[ci].DataPtr,
            _cpuNormBuf, _intermDim, _embDim, _cpuWGate[ci].DType, _cpuWUp[ci].DType);
        SimdKernels.SiLuMul(_cpuFfnGate, _cpuFfnUp, _intermDim);
        SimdKernels.MatVec(_cpuHidden, _cpuWDown[ci].DataPtr, _cpuFfnGate, _embDim, _intermDim, _cpuWDown[ci].DType);
    }

    private void CpuMoeFfn(int ci)
    {
        int numExperts = _hp.NumExperts;
        int numActive = _hp.NumActiveExperts;

        SimdKernels.MatVec(_cpuRouterLogits, _cpuWGateInp![ci].DataPtr, _cpuNormBuf, numExperts, _embDim, _cpuWGateInp[ci].DType);
        if (_hp.UseSigmoidGating)
            SimdKernels.SigmoidInPlace(_cpuRouterLogits, numExperts);
        else
            SimdKernels.SoftmaxInPlace(_cpuRouterLogits, numExperts);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_cpuRouterLogits, numExperts, numActive, selectedExperts, expertWeights);

        if (_hasSharedExpert)
        {
            SimdKernels.MatVec(_cpuExpertGate, _cpuWGateShexp![ci].DataPtr, _cpuNormBuf, _expertDim, _embDim, _cpuWGateShexp[ci].DType);
            SimdKernels.MatVec(_cpuExpertUp, _cpuWUpShexp![ci].DataPtr, _cpuNormBuf, _expertDim, _embDim, _cpuWUpShexp[ci].DType);
            SimdKernels.SiLuMul(_cpuExpertGate, _cpuExpertUp, _expertDim);
            SimdKernels.MatVec(_cpuSharedOut, _cpuWDownShexp![ci].DataPtr, _cpuExpertGate, _embDim, _expertDim, _cpuWDownShexp[ci].DType);
        }

        new Span<float>(_cpuHidden, _embDim).Clear();

        for (int k = 0; k < numActive; k++)
        {
            int expertIdx = selectedExperts[k];
            float weight = expertWeights[k];

            ExpertMatVec(_cpuExpertGate, _cpuWGateExps![ci], expertIdx, _expertDim, _embDim, _cpuNormBuf);
            ExpertMatVec(_cpuExpertUp, _cpuWUpExps![ci], expertIdx, _expertDim, _embDim, _cpuNormBuf);

            if (_hp.UseSigmoidGating)
            {
                SimdKernels.ScaleInPlace(_cpuExpertGate, weight, _expertDim);
                SimdKernels.ScaleInPlace(_cpuExpertUp, weight, _expertDim);
                weight = 1.0f;
            }

            SimdKernels.SiLuMul(_cpuExpertGate, _cpuExpertUp, _expertDim);
            ExpertMatVecDown(_cpuHidden, _cpuWDownExps![ci], expertIdx, _embDim, _expertDim, _cpuExpertGate, weight);
        }

        if (_hasSharedExpert)
            SimdKernels.AddInPlace(_cpuHidden, _cpuSharedOut, _embDim);
    }

    private void CpuAttention(int ci, int position)
    {
        int seqLen = position + 1;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        int maxSeqLen = _maxSeqLen; int hd = _headDim; int hpkg = _headsPerKvGroup;
        var q = _cpuQ; var attnOut = _cpuAttnOut; var scores = _cpuAttnScores;
        var kvCache = _cpuKvCache;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hd;
            float* outHead = attnOut + h * hd;
            float* headScores = scores + (long)h * maxSeqLen;

            for (int t = 0; t < seqLen; t++)
            {
                float* kVec = kvCache.KeyAt(ci, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;

            for (int t = 0; t < seqLen; t++)
            {
                float* vVec = kvCache.ValueAt(ci, t) + kvHead * hd;
                float w = headScores[t];
                if (Fma.IsSupported && hd >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hd; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
    }

    private void CpuTqAttention(int ci, int position)
    {
        var tq = _cpuTqKvCache!;
        int seqLen = position + 1;
        int tqLen = tq.GetTqLength(ci);
        int fp32Start = tqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        int maxSeqLen = _maxSeqLen; int hd = _headDim; int hpkg = _headsPerKvGroup;
        int tqBlkSz = tq.TqBlockSize;
        var q = _cpuQ; var attnOut = _cpuAttnOut; var scores = _cpuAttnScores;
        var rotated = _cpuRotatedQuery; var decomp = _cpuDecompBuf;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hd;
            float* outHead = attnOut + h * hd;
            float* headScores = scores + (long)h * maxSeqLen;
            float* headRotated = rotated + h * hd;
            float* headDecomp = decomp + h * hd;

            var keyCompressor = tq.GetKeyCompressor(ci, kvHead);
            keyCompressor.RotateQuery(
                new ReadOnlySpan<float>(qHead, hd),
                new Span<float>(headRotated, hd));

            for (int t = 0; t < tqLen; t++)
            {
                byte* tqKey = tq.TqKeyAt(ci, t, kvHead);
                float dot = keyCompressor.DequantDot(
                    new ReadOnlySpan<byte>(tqKey, tqBlkSz),
                    new ReadOnlySpan<float>(headRotated, hd));
                headScores[t] = dot * scale;
            }

            for (int t = fp32Start; t < seqLen; t++)
            {
                float* kVec = tq.Fp32KeyAt(ci, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;

            var valCompressor = tq.GetValueCompressor(ci, kvHead);
            for (int t = 0; t < tqLen; t++)
            {
                byte* tqVal = tq.TqValueAt(ci, t, kvHead);
                valCompressor.Decompress(
                    new ReadOnlySpan<byte>(tqVal, tqBlkSz),
                    new Span<float>(headDecomp, hd));
                float w = headScores[t];
                for (int d = 0; d < hd; d++)
                    outHead[d] += w * headDecomp[d];
            }

            for (int t = fp32Start; t < seqLen; t++)
            {
                float* vVec = tq.Fp32ValueAt(ci, t) + kvHead * hd;
                float w = headScores[t];
                if (Fma.IsSupported && hd >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hd; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
    }

    private void ExpertMatVec(float* output, in CpuWeightRef packedTensor,
        int expertIdx, int rows, int cols, float* input)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;
        SimdKernels.MatVec(output, expertData, input, rows, cols, packedTensor.DType);
    }

    private void ExpertMatVecDown(float* output, in CpuWeightRef packedTensor,
        int expertIdx, int rows, int cols, float* input, float weight)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;

        float* temp = _cpuAttnOut;
        SimdKernels.MatVec(temp, expertData, input, rows, cols, packedTensor.DType);
        for (int i = 0; i < rows; i++)
            output[i] += weight * temp[i];
    }

    private static void SelectTopK(float* logits, int n, int k,
        Span<int> indices, Span<float> weights)
    {
        for (int ki = 0; ki < k; ki++)
        {
            int bestIdx = 0;
            float bestVal = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                bool alreadySelected = false;
                for (int j = 0; j < ki; j++)
                {
                    if (indices[j] != i)
                        continue;

                    alreadySelected = true;
                    break;
                }

                if (!alreadySelected && logits[i] > bestVal)
                {
                    bestVal = logits[i];
                    bestIdx = i;
                }
            }

            indices[ki] = bestIdx;
            weights[ki] = bestVal;
        }

        if (k <= 1)
            return;

        float sum = 0;
        for (int i = 0; i < k; i++)
            sum += weights[i];

        if (sum <= 0)
            return;

        for (int i = 0; i < k; i++)
            weights[i] /= sum;
    }

    // ================================================================
    //  Helpers
    // ================================================================

    private readonly unsafe struct CpuWeightRef
    {
        public readonly string Name;
        public readonly GgufTensorInfo Info;
        public readonly DType DType;
        public readonly byte* DataPtr;

        public CpuWeightRef(string name, GgufTensorInfo info, DType dtype, byte* dataPtr)
        { Name = name; Info = info; DType = dtype; DataPtr = dataPtr; }
    }

    private CpuWeightRef ResolveCpuWeight(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        return new CpuWeightRef(name, info, info.DType, _model.GetTensorDataPtr(info));
    }

    private float* LoadCpuBias(string name, int count)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing bias tensor: {name}");
        var data = _model.GetTensorData(info);
        var buf = Alloc(count);
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), info.DType, count);
        return buf;
    }

    private float* GetCpuNormWeight(in CpuWeightRef tensor)
    {
        if (_cpuNormCache.TryGetValue(tensor.Name, out var cached))
            return (float*)cached;
        var data = _model.GetTensorData(tensor.Info);
        int count = (int)tensor.Info.ElementCount;
        var buf = Alloc(count);
        if (tensor.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), tensor.DType, count);
        _cpuNormCache[tensor.Name] = (nint)buf;
        return buf;
    }

    private void CpuEmbedToken(int token, float* dest)
    {
        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(_cpuEmbedding.DType))
                        * DTypeInfo.BytesPerBlock(_cpuEmbedding.DType);
        byte* rowPtr = _cpuEmbedding.DataPtr + (long)token * bytesPerRow;
        if (_cpuEmbedding.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, _embDim)
                .CopyTo(new Span<float>(dest, _embDim));
        }
        else
        {
            Dequantize.ToFloat32(
                new ReadOnlySpan<byte>(rowPtr, bytesPerRow),
                new Span<float>(dest, _embDim),
                _cpuEmbedding.DType,
                _embDim);
        }
    }

    private void ComputeCpuOutput()
    {
        var outputNorm = GetCpuNormWeight(_cpuOutputNorm);
        SimdKernels.RmsNorm(_cpuNormBuf, _cpuHidden, outputNorm, _embDim, _hp.RmsNormEps);
        fixed (float* logits = _logitsBuf)
            SimdKernels.MatVec(logits, _cpuOutputWeight.DataPtr, _cpuNormBuf, _hp.VocabSize, _embDim, _cpuOutputWeight.DType);
    }

    private static bool ShouldKeepFixedWeightsOnCpu(GgufTensorInfo embedding, GgufTensorInfo? output)
    {
        // Always keep embedding+output on CPU for the hybrid path. The GPU embedding-lookup
        // shader produces garbage when invoked inside HybridForwardPass (root cause unclear —
        // identical shader works in GpuForwardPass), and the embedding is one-row-per-token
        // which is negligible vs. the layer compute, so the perf cost is near zero.
        // TODO: investigate why EmbedLookupQ4K writes wrong values in hybrid context.
        return true;
    }

    private static long EstimateGpuTensorBytes(GgufTensorInfo tensor)
    {
        if (tensor.DType == DType.Float32 || tensor.DType == DType.Q4_K || tensor.DType == DType.Q6_K)
            return (tensor.ByteSize + 3) & ~3L;

        return tensor.ElementCount * sizeof(float);
    }

    private Tensor UploadWeight(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        Tensor result;
        if (info.DType == DType.Float32)
        {
            var floats = MemoryMarshal.Cast<byte, float>(data);
            result = _gpu.Upload(floats, TensorShape.D1(floats.Length));
            _gpuWeightDTypes[result.Handle] = DType.Float32;
        }
        else if (info.DType == DType.Q4_K || info.DType == DType.Q6_K)
        {
            int floatCount = data.Length / 4;
            var rawFloats = new float[floatCount];
            data.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount));
            _gpuWeightDTypes[result.Handle] = info.DType;
        }
        else
        {
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count));
            _gpuWeightDTypes[result.Handle] = DType.Float32;
        }
        return result;
    }

    private void GpuDenseFfn(int layer)
    {
        GpuMatMul(_gpuFfnGate, _gpuWGate[layer], _gpuNormBuf);
        GpuMatMul(_gpuFfnUp, _gpuWUp[layer], _gpuNormBuf);
        _gpu.RecordBarrier();

        _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
        _gpu.RecordBarrier();

        GpuMatMul(_gpuHidden, _gpuWDown[layer], _gpuFfnGate);
    }

    private void GpuMoeFfn(int layer)
    {
        int numActive = _hp.NumActiveExperts;

        GpuMatMul(_gpuRouterLogits!, _gpuWGateInp![layer], _gpuNormBuf);
        _gpu.RecordBarrier();
        if (_hp.UseSigmoidGating)
            _gpu.Sigmoid(_gpuRouterLogits!);
        else
            _gpu.Softmax(_gpuRouterLogits!);
        // Copy norm buf to pinned memory while still recording so the CPU can
        // read it after submit via MapPinned — avoids a second CopyBuffer call.
        _gpu.RecordBarrier();
        CopyGpuBuffer(_gpuPinnedNorm!, _gpuNormBuf);
        _gpu.EndRecordAndSubmit();
        _gpu.Download(_gpuRouterLogits!, _gpuRouterBuf!);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_gpuRouterBuf!, numActive, selectedExperts, expertWeights);

        // ── CPU fallback for cache misses ──
        // Experts not yet in the slot cache are computed on CPU while the GPU
        // is idle (between EndRecordAndSubmit and the next BeginRecord).
        // Their weighted outputs are accumulated in _cpuFallbackBuf and
        // uploaded to the pre-allocated pinned tensor for GPU AddInPlace.
        // Prefetch the same experts for the next token (1-token lookahead).
        _prefetcher?.EnqueuePrefetch(layer, selectedExperts);
        Span<bool> isGpu = stackalloc bool[numActive];
        // ExpertGpuSlot contains Tensor (managed reference type fields) — heap-allocate.
        ExpertGpuSlot[] cachedSlots = new ExpertGpuSlot[numActive];
        bool hasCpuFallback = false;

        for (int i = 0; i < numActive; i++)
        {
            isGpu[i] = _expertSlotManager!.TryGetCached(layer, selectedExperts[i], out cachedSlots[i]);
            if (!isGpu[i]) hasCpuFallback = true;
        }

        if (hasCpuFallback)
        {
            // _gpuPinnedNorm was populated by the GPU session above — map it directly,
            // no extra Download / CopyBuffer call needed.
            unsafe
            {
                float* normPtr = _gpu.MapPinned(_gpuPinnedNorm!);
                GpuMoeFfnCpuFallback(layer, selectedExperts, expertWeights, isGpu, numActive, normPtr);
                _gpu.UnmapPinned(_gpuPinnedNorm!);
            }
        }

        _gpu.BeginRecord();

        if (_hasSharedExpert)
        {
            GpuMatMul(_gpuFfnGate, _gpuWGateShexp![layer], _gpuNormBuf);
            GpuMatMul(_gpuFfnUp, _gpuWUpShexp![layer], _gpuNormBuf);
            _gpu.RecordBarrier();
            _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
            _gpu.RecordBarrier();
            GpuMatMul(_gpuMoeSharedOut!, _gpuWDownShexp![layer], _gpuFfnGate);
            _gpu.RecordBarrier();
        }

        _gpu.Clear(_gpuHidden);
        _gpu.RecordBarrier();

        for (int i = 0; i < numActive; i++)
        {
            if (!isGpu[i]) continue; // handled by CPU fallback

            float expertWeight = expertWeights[i];
            GpuMatMul(_gpuFfnGate, cachedSlots[i].Gate, _gpuNormBuf);
            GpuMatMul(_gpuFfnUp, cachedSlots[i].Up, _gpuNormBuf);
            _gpu.RecordBarrier();

            if (_hp.UseSigmoidGating)
            {
                _gpu.ScaleInPlace(_gpuFfnGate, expertWeight);
                _gpu.ScaleInPlace(_gpuFfnUp, expertWeight);
                _gpu.RecordBarrier();
            }

            _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
            _gpu.RecordBarrier();
            GpuMatMul(_gpuMoeExpertOut!, cachedSlots[i].Down, _gpuFfnGate);
            _gpu.RecordBarrier();

            if (_hp.UseSigmoidGating)
                _gpu.AddInPlace(_gpuHidden, _gpuMoeExpertOut!);
            else
                _gpu.AddScaledInPlace(_gpuHidden, _gpuMoeExpertOut!, expertWeight);
            _gpu.RecordBarrier();
        }

        // Add CPU-computed contributions (if any) via pre-allocated pinned buffer.
        if (hasCpuFallback)
        {
            unsafe
            {
                fixed (float* srcPtr = _cpuFallbackBuf)
                {
                    float* mapped = _gpu.MapPinned(_gpuFallbackContrib!);
                    new ReadOnlySpan<float>(srcPtr, _embDim).CopyTo(new Span<float>(mapped, _embDim));
                    _gpu.UnmapPinned(_gpuFallbackContrib!);
                }
            }
            // _gpuFallbackContrib is HOST_COHERENT — host writes are visible to device.
            // A compute barrier ensures any prior GPU writes complete before we add the fallback.
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_gpuHidden, _gpuFallbackContrib!);
            _gpu.RecordBarrier();
        }

        if (_hasSharedExpert)
            _gpu.AddInPlace(_gpuHidden, _gpuMoeSharedOut!);
    }

    private unsafe void GpuMoeFfnCpuFallback(int layer, ReadOnlySpan<int> selectedExperts,
        ReadOnlySpan<float> expertWeights, ReadOnlySpan<bool> isGpu, int numActive, float* normPtr)
    {
        // Lazily allocate CPU scratch arrays.
        _cpuFallbackBuf ??= new float[_embDim];
        _cpuFallbackGate ??= new float[_expertDim];
        _cpuFallbackUp ??= new float[_expertDim];

        Array.Clear(_cpuFallbackBuf);

        // Resolve mmap weight refs for this layer's expert tensors.
        var wGateExps = ResolveCpuWeight($"blk.{layer}.ffn_gate_exps.weight");
        var wUpExps   = ResolveCpuWeight($"blk.{layer}.ffn_up_exps.weight");
        var wDownExps = ResolveCpuWeight($"blk.{layer}.ffn_down_exps.weight");

        fixed (float* fallbackPtr = _cpuFallbackBuf)
        fixed (float* gatePtr = _cpuFallbackGate)
        fixed (float* upPtr = _cpuFallbackUp)
        {
            for (int i = 0; i < numActive; i++)
            {
                if (isGpu[i]) continue;
                int expertIdx = selectedExperts[i];
                float weight = expertWeights[i];

                ExpertMatVec(gatePtr, wGateExps, expertIdx, _expertDim, _embDim, normPtr);
                ExpertMatVec(upPtr,   wUpExps,   expertIdx, _expertDim, _embDim, normPtr);

                if (_hp.UseSigmoidGating)
                {
                    SimdKernels.ScaleInPlace(gatePtr, weight, _expertDim);
                    SimdKernels.ScaleInPlace(upPtr, weight, _expertDim);
                    weight = 1.0f;
                }

                SimdKernels.SiLuMul(gatePtr, upPtr, _expertDim);
                ExpertMatVecDown(fallbackPtr, wDownExps, expertIdx, _embDim, _expertDim, gatePtr, weight);
            }
        }
    }

    private Tensor[] UploadExpertWeights(string name, int rows, int cols, int expertCount)
    {
        var tensors = new Tensor[expertCount];
        for (int expertIdx = 0; expertIdx < expertCount; expertIdx++)
            tensors[expertIdx] = UploadExpertWeight(name, rows, cols, expertIdx);
        return tensors;
    }

    private Tensor UploadTqSignPatterns(int layerIndex)
    {
        var fullSigns = new float[_numKvHeads * _headDim];
        for (int h = 0; h < _numKvHeads; h++)
        {
            var headSigns = WalshHadamard.GenerateSignPattern(_headDim, layerIndex * _numKvHeads + h);
            headSigns.CopyTo(fullSigns.AsSpan(h * _headDim));
        }

        return _gpu.Upload(fullSigns, TensorShape.D1(fullSigns.Length));
    }

    private Tensor UploadExpertWeight(string name, int rows, int cols, int expertIdx)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        if (info.DType == DType.Float32)
        {
            int elemOffset = expertIdx * rows * cols;
            var floats = MemoryMarshal.Cast<byte, float>(data).Slice(elemOffset, rows * cols);
            var result = _gpu.Upload(floats, TensorShape.D1(floats.Length));
            _gpuWeightDTypes[result.Handle] = DType.Float32;
            return result;
        }

        int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType))
                        * DTypeInfo.BytesPerBlock(info.DType);
        int expertBytes = rows * bytesPerRow;
        int byteOffset = expertIdx * expertBytes;
        var expertData = data.Slice(byteOffset, expertBytes);

        if (info.DType == DType.Q4_K || info.DType == DType.Q6_K)
        {
            int floatCount = expertData.Length / 4;
            var rawFloats = new float[floatCount];
            expertData.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            var result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount));
            _gpuWeightDTypes[result.Handle] = info.DType;
            return result;
        }

        int count = rows * cols;
        var f32 = new float[count];
        Dequantize.ToFloat32(expertData, f32, info.DType, count);
        var tensor = _gpu.Upload(f32, TensorShape.D1(count));
        _gpuWeightDTypes[tensor.Handle] = DType.Float32;
        return tensor;
    }

    private static void SelectTopK(ReadOnlySpan<float> logits, int k,
        Span<int> indices, Span<float> weights)
    {
        for (int ki = 0; ki < k; ki++)
        {
            int bestIdx = 0;
            float bestVal = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
            {
                bool alreadySelected = false;
                for (int j = 0; j < ki; j++)
                {
                    if (indices[j] != i)
                        continue;

                    alreadySelected = true;
                    break;
                }

                if (!alreadySelected && logits[i] > bestVal)
                {
                    bestVal = logits[i];
                    bestIdx = i;
                }
            }

            indices[ki] = bestIdx;
            weights[ki] = bestVal;
        }

        if (k <= 1)
            return;

        float sum = 0;
        for (int i = 0; i < k; i++)
            sum += weights[i];
        if (sum <= 0)
            return;
        for (int i = 0; i < k; i++)
            weights[i] /= sum;
    }

    private void GpuMatMul(Tensor output, Tensor weights, Tensor input)
    {
        _gpu.MatMul(output, weights, input,
            _gpuWeightDTypes.TryGetValue(weights.Handle, out var dt) ? dt : DType.Float32);
    }

    // Compute-shader-based buffer copies. Critical: `vkCmdCopyBuffer` runs in the Transfer
    // pipeline stage, which is NOT synchronized with the rest of this forward pass — every
    // RecordBarrier() in this file is a Compute→Compute memory barrier and does nothing for
    // transfer-stage writes. Using a compute copy keeps everything in the compute stage so
    // those barriers are correct.
    private void CopyGpuBuffer(Tensor dst, Tensor src) => _gpu.RecordComputeCopy(dst, src);

    private void CopyGpuBufferRegion(Tensor dst, long dstOffsetBytes, Tensor src, long srcOffsetBytes, long sizeBytes)
    {
        _gpu.RecordComputeCopyRegion(dst, dstOffsetBytes, src, srcOffsetBytes, sizeBytes);
    }

    private static void PerHeadRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.RmsNorm(data + h * headDim, data + h * headDim, weight, headDim, eps);
    }

    private static void PerHeadPureRmsNorm(float* data, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.PureRmsNorm(data + h * headDim, data + h * headDim, headDim, eps);
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)count, (nuint)sizeof(float));

    // ================================================================
    //  Disposal
    // ================================================================

    public void Dispose()
    {
        _gpu.Free(_gpuHidden); _gpu.Free(_gpuResidual); _gpu.Free(_gpuNormBuf);
        _gpu.Free(_gpuQ); _gpu.Free(_gpuK); _gpu.Free(_gpuV); _gpu.Free(_gpuAttnOut);
        _gpu.Free(_gpuFfnGate); _gpu.Free(_gpuFfnUp); _gpu.Free(_gpuLogits);
        if (_gpuRouterLogits is not null) _gpu.Free(_gpuRouterLogits);
        if (_gpuMoeSharedOut is not null) _gpu.Free(_gpuMoeSharedOut);
        if (_gpuMoeExpertOut is not null) _gpu.Free(_gpuMoeExpertOut);
        _gpu.Free(_pinnedHidden);

        for (int i = 0; i < _nGpuLayers; i++)
        {
            _gpu.Free(_gpuAttnNorm[i]); _gpu.Free(_gpuFfnNorm[i]);
            _gpu.Free(_gpuWq[i]); _gpu.Free(_gpuWk[i]); _gpu.Free(_gpuWv[i]); _gpu.Free(_gpuWo[i]);
            if (_isMoE)
            {
                _gpu.Free(_gpuWGateInp![i]);
                // Expert tensors are managed by ExpertSlotManager — freed in _expertSlotManager.Dispose()
                if (_hasSharedExpert)
                {
                    _gpu.Free(_gpuWGateShexp![i]);
                    _gpu.Free(_gpuWUpShexp![i]);
                    _gpu.Free(_gpuWDownShexp![i]);
                }
            }
            else
            {
                _gpu.Free(_gpuWGate[i]); _gpu.Free(_gpuWUp[i]); _gpu.Free(_gpuWDown[i]);
            }
            _gpu.Free(_gpuKCache[i]); _gpu.Free(_gpuVCache[i]);
            if (_tqEnabled)
            {
                _gpu.Free(_gpuTqKCache![i]);
                _gpu.Free(_gpuTqVCache![i]);
                _gpu.Free(_gpuSignPatterns![i]);
            }

            if (_hasAttnBias)
            { _gpu.Free(_gpuBq![i]); _gpu.Free(_gpuBk![i]); _gpu.Free(_gpuBv![i]); _gpu.Free(_gpuBo![i]); }
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            { _gpu.Free(_gpuQNorm![i]); _gpu.Free(_gpuKNorm![i]); }
        }
        if (_tqEnabled)
        {
            _gpu.Free(_gpuCodebook!);
            _gpu.Free(_gpuBoundaries!);
            _gpu.Free(_gpuRotatedQ!);
            _gpu.Free(_gpuEvictK!);
            _gpu.Free(_gpuEvictV!);
        }
        if (_gpuOutputNorm is not null)
            _gpu.Free(_gpuOutputNorm);
        if (_gpuOutputWeight is not null && _gpuEmbedding is not null && _gpuOutputWeight.Handle != _gpuEmbedding.Handle)
            _gpu.Free(_gpuOutputWeight);
        if (_gpuEmbedding is not null)
            _gpu.Free(_gpuEmbedding);

        NativeMemory.Free(_cpuHidden); NativeMemory.Free(_cpuResidual); NativeMemory.Free(_cpuNormBuf);
        NativeMemory.Free(_cpuQ); NativeMemory.Free(_cpuK); NativeMemory.Free(_cpuV); NativeMemory.Free(_cpuAttnOut);
        NativeMemory.Free(_cpuFfnGate); NativeMemory.Free(_cpuFfnUp); NativeMemory.Free(_cpuAttnScores);
        NativeMemory.Free(_ropeCosTable); NativeMemory.Free(_ropeSinTable);
        if (_cpuRouterLogits != null) NativeMemory.Free(_cpuRouterLogits);
        if (_cpuSharedOut != null) NativeMemory.Free(_cpuSharedOut);
        if (_cpuExpertGate != null) NativeMemory.Free(_cpuExpertGate);
        if (_cpuExpertUp != null) NativeMemory.Free(_cpuExpertUp);
        if (_cpuRotatedQuery != null) NativeMemory.Free(_cpuRotatedQuery);
        if (_cpuDecompBuf != null) NativeMemory.Free(_cpuDecompBuf);
        _cpuKvCache.Dispose();
        _cpuTqKvCache?.Dispose();
        _prefetcher?.Dispose();
        _expertSlotManager?.Dispose();
        if (_gpuFallbackContrib is not null) _gpu.Free(_gpuFallbackContrib);
        if (_gpuPinnedNorm is not null) _gpu.Free(_gpuPinnedNorm);
    }
}
