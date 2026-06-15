using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Vision;

/// <summary>
/// A loaded Gemma 4 "unified" (encoder-free) multimodal projector — the small
/// companion mmproj GGUF (<c>general.architecture=clip</c>,
/// <c>clip.vision.projector_type=gemma4uv</c>) that ships next to the text model.
///
/// Unlike the encoder-full <c>gemma4v</c> (SigLIP-style ViT), this variant has no
/// transformer blocks: it projects raw image patches straight into the LLM
/// embedding space via three LayerNorms + a single linear patch embed + a learned
/// 2D position table + a final RMSNorm and linear projection. See issue #250 and
/// llama.cpp <c>tools/mtmd/models/gemma4uv.cpp</c>.
///
/// This type owns the loaded GGUF and exposes the 11 projector tensors + config;
/// the actual forward lives in <see cref="GemmaUvVisionEmbedder"/>.
/// </summary>
public sealed class VisionModel : IDisposable
{
    /// <summary>The only projector type currently supported.</summary>
    public const string ProjectorTypeGemma4Uv = "gemma4uv";

    private readonly GgufModel _gguf;
    private bool _disposed;

    private VisionModel(GgufModel gguf) => _gguf = gguf;

    /// <summary><c>clip.vision.projector_type</c> — always "gemma4uv" for a loaded model.</summary>
    public required string ProjectorType { get; init; }
    public required bool HasVisionEncoder { get; init; }
    public required bool HasAudioEncoder { get; init; }

    /// <summary>Declared patch size (<c>clip.vision.patch_size</c> = 16).</summary>
    public required int ConfigPatchSize { get; init; }

    /// <summary>
    /// Token-merge factor (gemma4uv hardcodes <c>n_merge=3</c>, then folds it into the
    /// patch size so merging happens in the patch-embed step rather than via pooling).
    /// </summary>
    public int NMerge => 3;

    /// <summary>Effective im2col patch edge = <see cref="ConfigPatchSize"/> × <see cref="NMerge"/> = 48.</summary>
    public int PatchSize => ConfigPatchSize * NMerge;

    public required int ImageSize { get; init; }        // 224
    public required int EmbeddingLength { get; init; }   // 3840
    public required int ProjectionDim { get; init; }     // 3840 (== text model n_embd)

    /// <summary>Soft-token budget per image (gemma4uv hardcodes <c>set_limit_image_tokens(40, 280)</c>).</summary>
    public int MinImageTokens => 40;
    public int MaxImageTokens => 280;

    /// <summary>LayerNorm eps for the three patch_norm layers (gemma4uv hardcodes the PyTorch default 1e-5).</summary>
    public float LayerNormEps => 1e-5f;

    /// <summary>RMSNorm eps for the pre-projection norm (<c>clip.vision.attention.layer_norm_epsilon</c>).</summary>
    public required float RmsNormEps { get; init; }

    internal GgufModel Gguf => _gguf;

    // ── resolved projector tensors (ne-order dims documented inline) ──
    internal GgufTensorInfo PatchEmbdWeight { get; init; }   // [6912, 3840] F32  (in=6912, out=3840)
    internal GgufTensorInfo PatchEmbdBias { get; init; }     // [3840] F32
    internal GgufTensorInfo PatchNorm1W { get; init; }       // [6912] F32
    internal GgufTensorInfo PatchNorm1B { get; init; }       // [6912] F32
    internal GgufTensorInfo PatchNorm2W { get; init; }       // [3840] F32
    internal GgufTensorInfo PatchNorm2B { get; init; }       // [3840] F32
    internal GgufTensorInfo PatchNorm3W { get; init; }       // [3840] F32  (pos_norm)
    internal GgufTensorInfo PatchNorm3B { get; init; }       // [3840] F32
    internal GgufTensorInfo PositionEmbd { get; init; }      // [3840, 1120, 2] F32  (x/y tables)
    internal GgufTensorInfo MmInputProjection { get; init; } // [3840, 3840] BF16 (in=3840, out=3840)

    /// <summary>
    /// Open and validate a Gemma 4 unified (gemma4uv) mmproj GGUF. Throws if the file
    /// is not a clip-arch projector of the supported type, or is missing a tensor.
    /// </summary>
    public static VisionModel Open(string mmprojPath)
    {
        var gguf = GgufModel.Open(mmprojPath);
        try
        {
            string arch = gguf.GetMetadata("general.architecture", "");
            if (arch != "clip")
                throw new NotSupportedException(
                    $"'{mmprojPath}' is not a clip mmproj (general.architecture='{arch}').");

            string proj = gguf.GetMetadata("clip.vision.projector_type", "");
            if (proj != ProjectorTypeGemma4Uv)
                throw new NotSupportedException(
                    $"Unsupported vision projector_type '{proj}'. Only '{ProjectorTypeGemma4Uv}' " +
                    "(Gemma 4 12B encoder-free) is implemented (issue #250).");

            GgufTensorInfo Req(string name) =>
                gguf.FindTensor(name)
                ?? throw new InvalidDataException($"mmproj '{mmprojPath}' is missing tensor '{name}'.");

            return new VisionModel(gguf)
            {
                ProjectorType    = proj,
                HasVisionEncoder = gguf.GetMetadata("clip.has_vision_encoder", true),
                HasAudioEncoder  = gguf.GetMetadata("clip.has_audio_encoder", false),
                ConfigPatchSize  = gguf.GetMetadata("clip.vision.patch_size", 16),
                ImageSize        = gguf.GetMetadata("clip.vision.image_size", 224),
                EmbeddingLength  = gguf.GetMetadata("clip.vision.embedding_length", 3840),
                ProjectionDim    = gguf.GetMetadata("clip.vision.projection_dim", 3840),
                RmsNormEps       = gguf.GetMetadata("clip.vision.attention.layer_norm_epsilon", 1e-6f),
                PatchEmbdWeight  = Req("v.patch_embd.weight"),
                PatchEmbdBias    = Req("v.patch_embd.bias"),
                PatchNorm1W      = Req("v.patch_norm.1.weight"),
                PatchNorm1B      = Req("v.patch_norm.1.bias"),
                PatchNorm2W      = Req("v.patch_norm.2.weight"),
                PatchNorm2B      = Req("v.patch_norm.2.bias"),
                PatchNorm3W      = Req("v.patch_norm.3.weight"),
                PatchNorm3B      = Req("v.patch_norm.3.bias"),
                PositionEmbd     = Req("v.position_embd.weight"),
                MmInputProjection = Req("mm.input_projection.weight"),
            };
        }
        catch
        {
            gguf.Dispose();
            throw;
        }
    }

    /// <summary>Dequantize a (small) F32/BF16 tensor to a managed float array.</summary>
    internal float[] LoadFloats(GgufTensorInfo t)
    {
        var dst = new float[t.ElementCount];
        Dequantize.ToFloat32(_gguf.GetTensorData(t), dst, t.DType, t.ElementCount);
        return dst;
    }

    /// <summary>
    /// Guard for borrowers of this model's memory-mapped tensors (e.g.
    /// <see cref="GemmaUvVisionEmbedder"/>, which caches raw weight pointers): once the
    /// model is disposed those pointers dangle, so any further use must fail loudly.
    /// </summary>
    internal void EnsureNotDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _gguf.Dispose();
    }
}
