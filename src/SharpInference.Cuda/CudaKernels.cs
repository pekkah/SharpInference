namespace SharpInference.Cuda;

/// <summary>
/// CUDA C kernel source code for image operations, compiled at runtime via NVRTC.
/// All kernels are combined in a single compilation unit to minimise NVRTC overhead.
/// </summary>
internal static class CudaKernels
{
    /// <summary>
    /// Combined CUDA C source for all image-processing kernels.
    /// Compiled once per CudaBackend lifetime via NVRTC.
    /// </summary>
    public const string Source = @"
// ── im2col ────────────────────────────────────────────────────────────────
// Materialise a 3×3 convolution window into a column matrix for GEMM.
//
// Input:  inp[inCh, H, W]            (CHW row-major, full image, N = H*W)
// Output: col[K, N_tile]             (K = inCh*9, N_tile pixels)
//
// Layout col[k, pixel_local] is chosen so GEMM A-matrix columns are contiguous:
//   cuBLAS Sgemm(OpN,OpN) reads column k of col as col[k*N_tile .. k*N_tile+N_tile-1]
//   — all N_tile pixels for kernel-position k — which is one contiguous cache line run.
//
// Block (32=pixel, 8=k): 256 threads.
// Grid (ceil(N_tile/32), ceil(K/8)).
//   • Writes: col[(blockIdx.y*8+ty)*N_tile + blockIdx.x*32+tx]
//             — consecutive tx (pixel varies) ⟹ fully coalesced within warp.
//   • Reads:  inp[ic*N + oy*W + ox] — scattered by 3×3 window, L2-resident.
extern ""C"" __global__ void im2col(
    const float* __restrict__ inp,   // [inCh, H, W]
    float*       __restrict__ col,   // [K, N_tile]
    int H, int W, int N,             // N = H*W (full image)
    int ph_start,                    // first row of this tile
    int N_tile,                      // pixel count in this tile
    int inCh, int K)                 // K = inCh * 9
{
    int pixel = blockIdx.x * 32 + threadIdx.x;
    int k     = blockIdx.y *  8 + threadIdx.y;
    if (pixel >= N_tile || k >= K) return;

    int ic  = k / 9;
    int kp  = k - ic * 9;
    int kh  = kp / 3;
    int kw  = kp - kh * 3;

    int ph_local = pixel / W;
    int pw       = pixel - ph_local * W;
    int oy       = ph_start + ph_local + kh - 1;
    int ox       = pw + kw - 1;

    float v = ((unsigned)oy < (unsigned)H) & ((unsigned)ox < (unsigned)W)
        ? inp[ic * N + oy * W + ox] : 0.f;
    col[(long)k * N_tile + pixel] = v;
}

// ── bias_add ──────────────────────────────────────────────────────────────
// out[oc, pixel] += bias[oc]  (layout: [outCh, N=H*W])
// 1-D grid over pixels; inner loop over outCh — keeps block count at ~1 024
// for 512×512 inputs instead of the 65 536 that a flat outCh×N launch produces.
extern ""C"" __global__ void bias_add(
    float* __restrict__ x, const float* __restrict__ bias,
    int N, int outCh)
{
    int pixel = (int)(blockIdx.x * 256 + threadIdx.x);
    if (pixel >= N) return;
    for (int oc = 0; oc < outCh; oc++)
        x[oc * N + pixel] += bias[oc];
}

// ── leaky_relu_inplace ────────────────────────────────────────────────────
extern ""C"" __global__ void leaky_relu_inplace(float* __restrict__ x, float neg_slope, int n)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    float v = x[idx];
    x[idx] = v >= 0.0f ? v : neg_slope * v;
}

// ── scale_inplace ─────────────────────────────────────────────────────────
extern ""C"" __global__ void scale_inplace(float* __restrict__ x, float scale, int n)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    x[idx] *= scale;
}

// ── add_inplace ───────────────────────────────────────────────────────────
// a[i] += b[i]
extern ""C"" __global__ void add_inplace(float* __restrict__ a, const float* __restrict__ b, int n)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    a[idx] += b[idx];
}

// ── add_scaled_inplace ────────────────────────────────────────────────────
// dst[i] += src[i] * scale
extern ""C"" __global__ void add_scaled_inplace(
    float* __restrict__ dst, const float* __restrict__ src, float scale, int n)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    dst[idx] += src[idx] * scale;
}

// ── scale_rows_inplace ────────────────────────────────────────────────────
// Per-row scalar multiply: buf[i*cols + e] *= scales[i].  One thread per
// (row, col) element; total = rows*cols.  The multiply rounds to float exactly
// like a per-row scale_inplace launch, so the result is bit-identical to
// applying ScaleInPlace(row_i, scales[i]) once per row.
extern ""C"" __global__ void llm_scale_rows_inplace(
    float* __restrict__ buf, const float* __restrict__ scales, int rows, int cols)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * cols;
    if (idx >= total) return;
    int i = (int)(idx / cols);
    buf[idx] *= scales[i];
}

// ── moe_weighted_reduce ───────────────────────────────────────────────────
// Per-token MoE reduce: for each (token i, element e) sum the na unweighted
// down partials in top-k slot order (k = 0..na-1) with their per-(token,slot)
// weights, then add the already-scaled-and-rounded shared-expert value LAST.
//
//   acc = 0
//   for k in 0..na-1:  acc += downPartial[(i*na+k)*embDim + e] * weights[i*na+k]
//   acc += shared[i*embDim + e]
//   shared[i*embDim + e] = acc
//
// One thread per (i, e); total = N*embDim.  `shared` is in/out — the thread
// that owns element (i,e) is the only reader and writer, so the read-modify-
// write is race-free.  `acc` is a single float register: each `acc += p*w`
// contracts to fmaf under NVRTC's default fmad=true (one rounding per term,
// matching add_scaled_inplace), and the shared add is a plain a+b (one
// rounding, matching add_inplace).  Order (routed first, shared last) and per-
// op rounding therefore reproduce the sequential Clear + AddScaledInPlace×na +
// AddInPlace accumulation byte-for-byte.
extern ""C"" __global__ void llm_moe_weighted_reduce(
    const float* __restrict__ downPartial, const float* __restrict__ weights,
    float* __restrict__ shared, int N, int na, int embDim)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * embDim;
    if (idx >= total) return;
    int i = (int)(idx / embDim);
    int e = (int)(idx - (long)i * embDim);
    float acc = 0.0f;
    const float* w = weights + (long)i * na;
    const float* p = downPartial + ((long)i * na) * embDim + e;
    for (int k = 0; k < na; k++)
        acc += p[(long)k * embDim] * w[k];
    acc += shared[(long)i * embDim + e];
    shared[(long)i * embDim + e] = acc;
}

// ── clamp_inplace ─────────────────────────────────────────────────────────
extern ""C"" __global__ void clamp_inplace(float* __restrict__ x, float lo, float hi, int n)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    float v = x[idx];
    x[idx] = v < lo ? lo : (v > hi ? hi : v);
}

// ── pixel_shuffle ─────────────────────────────────────────────────────────
// input[inCh, H, W] → output[outCh, H*r, W*r]  where outCh = inCh / (r*r)
// Inverse of pixel_unshuffle.
extern ""C"" __global__ void pixel_shuffle(
    const float* __restrict__ input, float* __restrict__ output,
    int outCh, int H, int W, int r)
{
    int outH = H * r;
    int outW = W * r;
    int idx  = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= outCh * outH * outW) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int oc = idx / (outW * outH);

    int ic = oc * r * r + (oh % r) * r + (ow % r);
    int ih = oh / r;
    int iw = ow / r;
    output[idx] = input[ic * H * W + ih * W + iw];
}

// ── pixel_unshuffle ───────────────────────────────────────────────────────
// input[inCh, outH*r, outW*r] → output[inCh*r², outH, outW]
// outH and outW are the SMALL (output) spatial dimensions.
extern ""C"" __global__ void pixel_unshuffle(
    const float* __restrict__ input, float* __restrict__ output,
    int inCh, int outH, int outW, int r)
{
    int inH  = outH * r;
    int inW  = outW * r;
    int outCh = inCh * r * r;
    int idx  = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= outCh * outH * outW) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int oc = idx / (outW * outH);

    int ic = oc / (r * r);
    int rk = oc % (r * r);
    int rh = rk / r;
    int rw = rk % r;
    output[idx] = input[ic * inH * inW + (oh * r + rh) * inW + (ow * r + rw)];
}

// ── upsample2x ────────────────────────────────────────────────────────────
// Nearest-neighbour 2× upsample: input[ch, H, W] → output[ch, 2H, 2W]
extern ""C"" __global__ void upsample2x(
    const float* __restrict__ input, float* __restrict__ output,
    int ch, int H, int W)
{
    int outH = H * 2;
    int outW = W * 2;
    int idx  = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= ch * outH * outW) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c  = idx / (outH * outW);
    output[idx] = input[c * H * W + (oh / 2) * W + (ow / 2)];
}
";
}
