namespace SharpInference.Vision;

/// <summary>A preprocessed image ready for <see cref="GemmaUvVisionEmbedder.Forward"/>: planar CHW float in [0,1].</summary>
public sealed record PreprocessedImage(float[] Chw, int Width, int Height);

/// <summary>
/// Image preprocessing for the Gemma 4 unified vision projector, mirroring llama.cpp
/// <c>mtmd-image.cpp</c> (the <c>dyn_size</c> path used by gemma4uv):
///   1. <see cref="CalcSizePreservedRatio"/>: aspect-preserving target, each side rounded to a
///      multiple of the effective patch (48), constrained so the soft-token count lands in [40,280].
///   2. align-corners bilinear resize to that target (RESIZE_ALGO_BILINEAR, PAD_NONE).
///   3. normalize <c>x/255</c> (image_mean=[0,0,0], image_std=[1,1,1]) and pack to planar CHW.
/// </summary>
public static class ImagePreprocessor
{
    /// <summary>
    /// llama.cpp <c>img_tool::calc_size_preserved_ratio</c>: round each side to a multiple of
    /// <paramref name="align"/>, then shrink/grow (preserving ratio) so width*height is within
    /// [<paramref name="minPixels"/>, <paramref name="maxPixels"/>]. Returns (width, height).
    /// </summary>
    public static (int W, int H) CalcSizePreservedRatio(int width, int height, int align, long minPixels, long maxPixels)
    {
        int RoundBy(double x) => (int)Math.Round(x / align, MidpointRounding.AwayFromZero) * align;
        int CeilBy(double x) => (int)Math.Ceiling(x / align) * align;
        int FloorBy(double x) => (int)Math.Floor(x / align) * align;

        int hBar = Math.Max(align, RoundBy(height));
        int wBar = Math.Max(align, RoundBy(width));

        if ((long)hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt((double)height * width / maxPixels);
            hBar = Math.Max(align, FloorBy(height / beta));
            wBar = Math.Max(align, FloorBy(width / beta));
        }
        else if ((long)hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt((double)minPixels / ((double)height * width));
            hBar = CeilBy(height * beta);
            wBar = CeilBy(width * beta);
        }
        return (wBar, hBar);
    }

    /// <summary>
    /// Preprocess an interleaved 8-bit RGB image (HWC, from <see cref="ImageIO.LoadRgb"/>) for
    /// the given <paramref name="model"/>. Returns planar CHW float in [0,1] at a 48-aligned size.
    /// </summary>
    public static PreprocessedImage Preprocess(byte[] rgbHwc, int srcWidth, int srcHeight, VisionModel model)
    {
        int align = model.PatchSize;                 // 48
        long perToken = (long)align * align;         // pixels per soft token
        long minPixels = model.MinImageTokens * perToken;
        long maxPixels = model.MaxImageTokens * perToken;

        var (w, h) = CalcSizePreservedRatio(srcWidth, srcHeight, align, minPixels, maxPixels);

        // align-corners bilinear resize -> planar CHW float, normalized x/255.
        var chw = new float[3 * h * w];
        int plane = h * w;
        float xRatio = w > 1 ? (float)(srcWidth - 1) / (w - 1) : 0f;
        float yRatio = h > 1 ? (float)(srcHeight - 1) / (h - 1) : 0f;

        for (int y = 0; y < h; y++)
        {
            float py = y * yRatio;
            int y0 = Math.Min((int)py, srcHeight - 1);
            int y1 = Math.Min(y0 + 1, srcHeight - 1);
            float yf = py - y0;
            for (int x = 0; x < w; x++)
            {
                float px = x * xRatio;
                int x0 = Math.Min((int)px, srcWidth - 1);
                int x1 = Math.Min(x0 + 1, srcWidth - 1);
                float xf = px - x0;

                int i00 = (y0 * srcWidth + x0) * 3, i10 = (y0 * srcWidth + x1) * 3;
                int i01 = (y1 * srcWidth + x0) * 3, i11 = (y1 * srcWidth + x1) * 3;
                int outIdx = y * w + x;
                for (int c = 0; c < 3; c++)
                {
                    float top = Lerp(rgbHwc[i00 + c], rgbHwc[i10 + c], xf);
                    float bottom = Lerp(rgbHwc[i01 + c], rgbHwc[i11 + c], xf);
                    // llama.cpp truncates the resized u8, then divides by 255 -> match the truncation.
                    byte u8 = (byte)Lerp(top, bottom, yf);
                    chw[c * plane + outIdx] = u8 / 255f;
                }
            }
        }
        return new PreprocessedImage(chw, w, h);
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;
}
