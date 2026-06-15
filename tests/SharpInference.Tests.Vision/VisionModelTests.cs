using SharpInference.Core;
using SharpInference.Vision;

namespace SharpInference.Tests.Vision;

public class VisionModelTests
{
    [Fact]
    public void Open_LoadsGemma4UvConfigAndTensors()
    {
        var path = VisionTestPaths.FindMmproj();
        if (path is null) return;   // model-gated test (mmproj only present on dev machines)

        using var m = VisionModel.Open(path);

        // config
        Assert.Equal(VisionModel.ProjectorTypeGemma4Uv, m.ProjectorType);
        Assert.True(m.HasVisionEncoder);
        Assert.Equal(16, m.ConfigPatchSize);
        Assert.Equal(3, m.NMerge);
        Assert.Equal(48, m.PatchSize);          // effective im2col patch (16 * 3)
        Assert.Equal(224, m.ImageSize);
        Assert.Equal(3840, m.EmbeddingLength);
        Assert.Equal(3840, m.ProjectionDim);
        Assert.Equal(40, m.MinImageTokens);
        Assert.Equal(280, m.MaxImageTokens);

        // tensor shapes (ne-order: fastest dim first)
        Assert.Equal(new long[] { 6912, 3840 }, m.PatchEmbdWeight.Dimensions);
        Assert.Equal(new long[] { 3840 }, m.PatchEmbdBias.Dimensions);
        Assert.Equal(new long[] { 6912 }, m.PatchNorm1W.Dimensions);
        Assert.Equal(new long[] { 3840 }, m.PatchNorm2W.Dimensions);
        Assert.Equal(new long[] { 3840 }, m.PatchNorm3W.Dimensions);
        Assert.Equal(new long[] { 3840, 1120, 2 }, m.PositionEmbd.Dimensions);
        Assert.Equal(new long[] { 3840, 3840 }, m.MmInputProjection.Dimensions);

        // dtypes: patch embed is F32, the mm projection is BF16
        Assert.Equal(DType.Float32, m.PatchEmbdWeight.DType);
        Assert.Equal(DType.BFloat16, m.MmInputProjection.DType);
    }

    [Fact]
    public void Open_RejectsTextModelAsMmproj()
    {
        var textPath = VisionTestPaths.FindTextModel();
        if (textPath is null) return;   // model-gated test

        // The text GGUF is arch=gemma4, not a clip mmproj -> must be rejected clearly.
        Assert.ThrowsAny<NotSupportedException>(() => VisionModel.Open(textPath));
    }
}
