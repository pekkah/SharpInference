namespace SharpInference.Diffusion;

/// <summary>
/// Euler flow-matching scheduler for FLUX.1 and similar rectified flow models.
///
/// Flow matching defines a straight-line path from noise (t=1) to data (t=0):
///   x(t) = (1-t)*x_data + t*x_noise
/// The velocity field v(x,t) ≈ x_data - x_noise points from noise toward data.
///
/// Euler step:  x_{t-dt} = x_t - dt * v_predicted
/// (dt > 0; equivalently, since our scheduler uses dt = tNext-t < 0: x -= dt*v)
///
/// FLUX.1-schnell default: 4 steps, linear schedule from t=1 → t=0.
/// FLUX.1-dev:             28 steps with optional time-shifting.
/// </summary>
public sealed class EulerFlowScheduler
{
    private readonly float[] _timesteps;

    public int NumSteps => _timesteps.Length;

    private EulerFlowScheduler(float[] timesteps) => _timesteps = timesteps;

    /// <summary>
    /// Create a scheduler with linearly spaced timesteps.
    /// </summary>
    /// <param name="numSteps">Number of denoising steps.</param>
    /// <param name="shift">
    /// Time-shift parameter (FLUX.1-dev uses shift≈3 for high resolution;
    /// schnell uses shift=1 = no shift).
    /// </param>
    public static EulerFlowScheduler Linear(int numSteps, float shift = 1f)
    {
        var ts = new float[numSteps];
        for (int i = 0; i < numSteps; i++)
        {
            // Linear from 1 → 0, evaluated at the start of each interval
            float t = 1f - (float)i / numSteps;
            // Apply optional sigma-shift: t' = t / (t + (1-t)/shift)
            if (shift != 1f)
                t = t * shift / (1f + (shift - 1f) * t);
            ts[i] = t;
        }
        return new EulerFlowScheduler(ts);
    }

    /// <summary>
    /// Run the full denoising loop.
    /// </summary>
    /// <param name="noise">Initial noise tensor (will be modified in-place).</param>
    /// <param name="ditForward">
    /// Function (latent, timestep) → velocity prediction.
    /// Receives the current noisy latent and scalar t ∈ [0,1].
    /// </param>
    /// <param name="progress">Optional progress callback (step, totalSteps).</param>
    public float[] Denoise(float[] noise,
                           Func<float[], float, float[]> ditForward,
                           Action<int, int>? progress = null)
    {
        var x = (float[])noise.Clone();
        int n = _timesteps.Length;

        for (int i = 0; i < n; i++)
        {
            float t    = _timesteps[i];
            float tNext = (i + 1 < n) ? _timesteps[i + 1] : 0f;
            float dt   = tNext - t;  // negative (we go from 1→0)

            // Predict velocity
            var v = ditForward(x, t);

            // Euler step (flow matching, backward integration):
            //   x_{t-Δt} = x_t - Δt * v   (Δt > 0, subtracting the velocity)
            // With dt = tNext - t < 0: x_new = x - dt * v = x + |dt| * v
            // This moves x toward the data (clean image) direction.
            for (int j = 0; j < x.Length; j++)
                x[j] -= dt * v[j];

            progress?.Invoke(i + 1, n);
        }
        return x;
    }

    /// <summary>
    /// Pack image latent [1, C, H, W] into sequence of patches [nPatches, patchDim].
    /// FLUX uses 2×2 patches over the latent spatial dims.
    /// patchDim = patchSize * patchSize * latentChannels = 2*2*16 = 64.
    /// Ordering: channel-first (ch, ky, kx) — matches FLUX patchify convention.
    /// </summary>
    public static float[] PackLatent(float[] latent, int c, int h, int w, int patchSize = 2)
    {
        int pH = h / patchSize, pW = w / patchSize;
        int patchDim = patchSize * patchSize * c;
        int nPatches = pH * pW;
        var packed = new float[nPatches * patchDim];

        for (int ph = 0; ph < pH; ph++)
        {
            for (int pw = 0; pw < pW; pw++)
            {
                int patchIdx = ph * pW + pw;
                int packOff  = patchIdx * patchDim;
                int flatIdx  = 0;
                for (int ch = 0; ch < c; ch++)
                {
                    for (int ky = 0; ky < patchSize; ky++)
                    {
                        for (int kx = 0; kx < patchSize; kx++)
                        {
                            int ih = ph * patchSize + ky;
                            int iw = pw * patchSize + kx;
                            packed[packOff + flatIdx++] = latent[ch * h * w + ih * w + iw];
                        }
                    }
                }
            }
        }
        return packed;
    }

    /// <summary>
    /// Unpack sequence [nPatches, patchDim] back to latent [C, H, W].
    /// </summary>
    public static float[] UnpackLatent(float[] packed, int c, int h, int w, int patchSize = 2)
    {
        int pH = h / patchSize, pW = w / patchSize;
        int patchDim = patchSize * patchSize * c;
        var latent = new float[c * h * w];

        for (int ph = 0; ph < pH; ph++)
        {
            for (int pw = 0; pw < pW; pw++)
            {
                int patchIdx = ph * pW + pw;
                int packOff  = patchIdx * patchDim;
                int flatIdx  = 0;
                for (int ch = 0; ch < c; ch++)
                {
                    for (int ky = 0; ky < patchSize; ky++)
                    {
                        for (int kx = 0; kx < patchSize; kx++)
                        {
                            int ih = ph * patchSize + ky;
                            int iw = pw * patchSize + kx;
                            latent[ch * h * w + ih * w + iw] = packed[packOff + flatIdx++];
                        }
                    }
                }
            }
        }
        return latent;
    }

    /// <summary>
    /// Pack image latent [C, H, W] → patches [nPatches, patchDim] with spatial-first ordering.
    /// Z-Image patchify convention: permute(0,2,4,3,5,1) → (B,H//p,W//p,p,p,C) → flatten.
    /// Each 64-dim patch = [pos(0,0)_allC, pos(0,1)_allC, pos(1,0)_allC, pos(1,1)_allC].
    /// </summary>
    public static float[] PackLatentSpatialFirst(float[] latent, int c, int h, int w, int patchSize = 2)
    {
        int pH = h / patchSize, pW = w / patchSize;
        int patchDim = patchSize * patchSize * c;
        int nPatches = pH * pW;
        var packed = new float[nPatches * patchDim];

        for (int ph = 0; ph < pH; ph++)
        {
            for (int pw = 0; pw < pW; pw++)
            {
                int patchIdx = ph * pW + pw;
                int packOff  = patchIdx * patchDim;
                int flatIdx  = 0;
                for (int ky = 0; ky < patchSize; ky++)
                {
                    for (int kx = 0; kx < patchSize; kx++)
                    {
                        int ih = ph * patchSize + ky;
                        int iw = pw * patchSize + kx;
                        for (int ch = 0; ch < c; ch++)
                            packed[packOff + flatIdx++] = latent[ch * h * w + ih * w + iw];
                    }
                }
            }
        }
        return packed;
    }

    /// <summary>
    /// Unpack patches [nPatches, patchDim] → latent [C, H, W] with spatial-first ordering.
    /// Inverse of PackLatentSpatialFirst.
    /// </summary>
    public static float[] UnpackLatentSpatialFirst(float[] packed, int c, int h, int w, int patchSize = 2)
    {
        int pH = h / patchSize, pW = w / patchSize;
        int patchDim = patchSize * patchSize * c;
        var latent = new float[c * h * w];

        for (int ph = 0; ph < pH; ph++)
        {
            for (int pw = 0; pw < pW; pw++)
            {
                int patchIdx = ph * pW + pw;
                int packOff  = patchIdx * patchDim;
                int flatIdx  = 0;
                for (int ky = 0; ky < patchSize; ky++)
                {
                    for (int kx = 0; kx < patchSize; kx++)
                    {
                        int ih = ph * patchSize + ky;
                        int iw = pw * patchSize + kx;
                        for (int ch = 0; ch < c; ch++)
                            latent[ch * h * w + ih * w + iw] = packed[packOff + flatIdx++];
                    }
                }
            }
        }
        return latent;
    }

    /// <summary>Generate random Gaussian noise of given size using a seeded RNG.</summary>
    public static float[] SampleNoise(int size, int seed = -1)
    {
        var rng = seed >= 0 ? new Random(seed) : new Random();
        var noise = new float[size];
        for (int i = 0; i < size - 1; i += 2)
        {
            // Box-Muller transform
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double mag = Math.Sqrt(-2.0 * Math.Log(u1));
            noise[i]     = (float)(mag * Math.Cos(2 * Math.PI * u2));
            noise[i + 1] = (float)(mag * Math.Sin(2 * Math.PI * u2));
        }
        if (size % 2 != 0)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            noise[size - 1] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return noise;
    }
}
