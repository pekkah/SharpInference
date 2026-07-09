namespace SharpInference.Cuda;

/// <summary>
/// CUDA C kernel source for LLM transformer operations, compiled at runtime via NVRTC.
/// Mirrors the GLSL compute shaders in <c>SharpInference.Vulkan.Shaders</c>:
/// RmsNorm, HeadNorm, RoPE (interleaved + NEOX), Softmax, Sigmoid, SiLuMul,
/// EmbedLookup (F32 and Q4_K), KvAppend, scaled dot-product Attention with GQA,
/// and Q4_K / Q6_K / F32 matrix-vector multiplies.
///
/// Kernels rely only on FP32 arithmetic + integer bit ops; no cuda_fp16.h header needed.
/// FP16 decoding for Q4_K / Q6_K super-block scales is implemented inline with bit math
/// so NVRTC can compile this in isolation (no header registration).
/// </summary>
internal static class CudaTextKernels
{
    public const string Source = @"
// NVRTC standalone compilation: <math.h> isn't available, so define our own
// floating-point +inf rather than relying on the C99 INFINITY macro.
__device__ __forceinline__ float sharpi_neg_inf() { return __int_as_float(0xff800000); }

// ── FP16 → FP32 decode ────────────────────────────────────────────────────
// Hardware path: single-cycle `cvt.f32.f16` PTX instruction (CUDA core's F2F
// unit). Earlier this code shipped a software branch-on-exp implementation
// that NVCC expanded to ~30 SASS instructions per call — and the matvec_q4k
// kernel called it 4× per super-block per lane (super_d, super_dmin, two Q8_1
// d's). That single change accounted for the bulk of the kernel's instruction
// count and was the dominant reason Q4_K matvec ran 5× slower than Vulkan.
__device__ __forceinline__ float sharpi_fp16_to_fp32(unsigned int h)
{
    float result;
    asm(""cvt.f32.f16 %0, %1;"" : ""=f""(result) : ""h""((unsigned short)h));
    return result;
}

// ── half2 (fp16x2) pack / fma / unpack via inline PTX ──────────────────────
// NVRTC compiles this source without cuda_fp16.h, so the __half2 intrinsics are
// unavailable; these wrap the raw PTX. A packed fp16x2 lives in one unsigned int
// (low half = first element). Used by the flash-attention QK dot to do 2 multiply-
// accumulates per instruction (FP16x2 path, ~2× the fp32 FMA rate) — the scores
// tolerate fp16-rounded inputs (argmax-stable), and the running sum over only a few
// pairs per lane keeps fp16 accumulation error negligible (final reduce is fp32).
__device__ __forceinline__ unsigned int sharpi_f32x2_to_f16x2(float a, float b)
{
    unsigned int r;
    asm(""{ .reg .b16 lo, hi; cvt.rn.f16.f32 lo, %1; cvt.rn.f16.f32 hi, %2; mov.b32 %0, {lo, hi}; }""
        : ""=r""(r) : ""f""(a), ""f""(b));
    return r;
}
__device__ __forceinline__ unsigned int sharpi_hfma2(unsigned int a, unsigned int b, unsigned int acc)
{
    unsigned int r;
    asm(""fma.rn.f16x2 %0, %1, %2, %3;"" : ""=r""(r) : ""r""(a), ""r""(b), ""r""(acc));
    return r;
}
// Sum the low and high fp16 halves of a packed fp16x2 into one fp32.
__device__ __forceinline__ float sharpi_f16x2_sum(unsigned int p)
{
    float lo, hi;
    asm(""{ .reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }""
        : ""=f""(lo), ""=f""(hi) : ""r""(p));
    return lo + hi;
}

// Warp-level sum reduction over 32 lanes (full-warp mask).
__device__ __forceinline__ float sharpi_warp_reduce_sum(float v)
{
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    v += __shfl_xor_sync(0xffffffffu, v,  8);
    v += __shfl_xor_sync(0xffffffffu, v,  4);
    v += __shfl_xor_sync(0xffffffffu, v,  2);
    v += __shfl_xor_sync(0xffffffffu, v,  1);
    return v;
}

// ── FP32 → FP16 encode ────────────────────────────────────────────────────
// Round-to-nearest-even. Returns 16-bit pattern in the low 16 of the result.
// Subnormals are flushed to zero (the Q8_1 path only stores scales d = amax/127
// which are well in the normal range for any non-pathological activation).
__device__ __forceinline__ unsigned int sharpi_fp32_to_fp16(float f)
{
    unsigned short h;
    asm(""cvt.rn.f16.f32 %0, %1;"" : ""=h""(h) : ""f""(f));
    return (unsigned int)h;
}

// ── BF16 ↔ FP32 (no header) ───────────────────────────────────────────────
// bfloat16 is just the high 16 bits of an IEEE-754 fp32. Decode = left shift;
// encode = round-to-nearest-even on bit 15 of the discarded mantissa half.
// We don't bother special-casing NaN — the KV cache is written from RoPE'd
// activations that are never NaN under normal operation, and any NaN that
// does appear will round-trip as a NaN (sign/exponent bits survive) which is
// the same outcome as fp32.
__device__ __forceinline__ float sharpi_bf16_to_fp32(unsigned int bits)
{
    unsigned int f = (bits & 0xFFFFu) << 16;
    return __int_as_float(f);
}

__device__ __forceinline__ unsigned int sharpi_fp32_to_bf16(float f)
{
    unsigned int bits = __float_as_uint(f);
    unsigned int lsb  = (bits >> 16) & 1u;
    unsigned int rb   = 0x7FFFu + lsb;
    bits += rb;
    return (bits >> 16) & 0xFFFFu;
}

// KV-cache element load (issue #179): one float per element when the cache is fp32,
// or a bf16 short decoded to fp32 when it's half-width. Overloaded on the pointer
// type so a single templated kernel body serves both dtypes; the fp32 overload is a
// plain p[i] (byte-identical to the pre-#179 kernels).
__device__ __forceinline__ float sharpi_kvload(const float* __restrict__ p, long i) { return p[i]; }
__device__ __forceinline__ float sharpi_kvload(const unsigned short* __restrict__ p, long i) { return sharpi_bf16_to_fp32((unsigned int)p[i]); }

// Issue #179 (q8_0 KV): block-quantized cache element. One block packs 32 int8
// quants sharing one fp16 scale (ggml block_q8_0 layout; 34 B/block ≈ 1.06 B/elem,
// ~half of bf16, ~quarter of fp32). The store side (llm_kv_append_q8_0 / _batched)
// computes the scale per 32-lane warp; this overload decodes ONE element so the
// SAME templated attention/flash bodies that serve fp32/bf16 also serve q8_0 via
// sharpi_kvload. The load is purely per-element (block `i>>5`, lane `i&31`); the
// store side (llm_kv_append_q8_0) is what relies on kv_dim being a multiple of 32 so
// a KV row's blocks never straddle a row boundary — see that kernel's note.
struct block_q8_0 { unsigned short d; signed char qs[32]; };
__device__ __forceinline__ float sharpi_kvload(const block_q8_0* __restrict__ p, long i)
{
    long b = i >> 5;
    int  j = (int)(i & 31);
    return sharpi_fp16_to_fp32((unsigned int)p[b].d) * (float)p[b].qs[j];
}

// Issue #213: dot of a query vector q[0..n) with a CONTIGUOUS cache row at element
// offset `off`, specialized per cache dtype. fp32/bf16 are the same trivial per-element loop
// the inlined sharpi_kvload produced (no change for those dtypes). The q8_0 overload caches the
// per-32-block fp16 scale — loading + cvt'ing it once per block instead of 32× per element —
// while keeping the SAME per-element accumulation order and the SAME scale value, so it is
// BIT-IDENTICAL to the per-element path. This removes the redundant scale work on the dominant
// contiguous K-row read of decode attention (#213: q8 attention is compute-bound on the dequant,
// not KV bandwidth — it reads ~4× fewer bytes than fp32 yet is ~50% slower).
__device__ __forceinline__ float sharpi_kv_dot(const float* __restrict__ q, const float* __restrict__ k, long off, int n)
{
    float dot = 0.f;
    for (int d = 0; d < n; d++) dot += q[d] * k[off + d];
    return dot;
}
__device__ __forceinline__ float sharpi_kv_dot(const float* __restrict__ q, const unsigned short* __restrict__ k, long off, int n)
{
    float dot = 0.f;
    for (int d = 0; d < n; d++) dot += q[d] * sharpi_bf16_to_fp32((unsigned int)k[off + d]);
    return dot;
}
__device__ __forceinline__ float sharpi_kv_dot(const float* __restrict__ q, const block_q8_0* __restrict__ k, long off, int n)
{
    float dot = 0.f;
    long b = off >> 5;              // block holding element `off`
    int lane = (int)(off & 31);     // starting lane within that block (0 when off is 32-aligned)
    for (int d = 0; d < n; )
    {
        float s = sharpi_fp16_to_fp32((unsigned int)k[b].d);   // convert the fp16 scale ONCE per block
        for (; lane < 32 && d < n; lane++, d++)
            dot += q[d] * (s * (float)k[b].qs[lane]);
        lane = 0;
        b++;
    }
    return dot;
}

// Read one byte from a uint32-stride buffer at absolute byte offset B.
__device__ __forceinline__ unsigned int sharpi_byte_at(const unsigned int* __restrict__ buf, long B)
{
    unsigned int w = buf[(B) >> 2];
    return (w >> (((unsigned int)(B) & 3u) * 8u)) & 0xFFu;
}

// Same as sharpi_byte_at but interprets the byte as int8 (signed).
__device__ __forceinline__ int sharpi_int8_at(const unsigned int* __restrict__ buf, long B)
{
    int v = (int)sharpi_byte_at(buf, B);
    return v >= 128 ? v - 256 : v;
}

// Read a 4-byte little-endian word from a uint32-stride buffer at an arbitrary
// (possibly 2-byte-aligned) byte offset B. Q8_0 qs start at a 2-byte offset inside
// each 34-byte block, so a raw int load would be misaligned; this assembles the
// word from the two covering aligned uints via funnelshift (1 extra load only when
// B is not 4-aligned). Mirrors llama.cpp's get_int_b2 (vecdotq.cuh).
__device__ __forceinline__ unsigned int sharpi_uint_at(const unsigned int* __restrict__ buf, long B)
{
    long aB = B & ~3L;
    unsigned int sh = (unsigned int)(B & 3L) * 8u;
    unsigned int lo = buf[aB >> 2];
    if (sh == 0u) return lo;
    unsigned int hi = buf[(aB >> 2) + 1];
    return __funnelshift_r(lo, hi, sh);
}

// ── RmsNorm ────────────────────────────────────────────────────────────────
// output[i] = input[i] / rms * weight[i], rms = sqrt(mean(input^2) + eps).
// 1 block of 256 threads. Push-equivalent: (n, eps).
extern ""C"" __global__ void llm_rmsnorm(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int n, float eps)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;

    float sum = 0.f;
    for (int i = (int)tid; i < n; i += 256) {
        float v = input[i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = (int)tid; i < n; i += 256)
        output[i] = input[i] * scale * weight[i];
}

// ── HeadNorm (weighted per-head RMS) ───────────────────────────────────────
// One block per head, 256 threads.
// weight_stride controls QK-norm weight layout:
//   0        → shared weight vector of length head_dim (Qwen3 style)
//   head_dim → per-channel weights of length num_heads*head_dim (OLMoE style)
extern ""C"" __global__ void llm_head_norm(
    float* __restrict__ data,
    const float* __restrict__ weight,
    int head_dim, int num_heads, float eps, int weight_stride)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int head = blockIdx.x;
    if ((int)head >= num_heads) return;

    int base_off = (int)head * head_dim;
    int w_off    = (int)head * weight_stride;

    float sum = 0.f;
    for (int i = (int)tid; i < head_dim; i += 256) {
        float v = data[base_off + i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = (int)tid; i < head_dim; i += 256)
        data[base_off + i] = data[base_off + i] * scale * weight[w_off + i];
}

// ── HeadNormPure (L2 normalize per head, no weights) ───────────────────────
extern ""C"" __global__ void llm_head_norm_pure(
    float* __restrict__ data,
    int head_dim, int num_heads, float eps)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int head = blockIdx.x;
    if ((int)head >= num_heads) return;

    int base_off = (int)head * head_dim;

    float sum = 0.f;
    for (int i = (int)tid; i < head_dim; i += 256) {
        float v = data[base_off + i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = (int)tid; i < head_dim; i += 256)
        data[base_off + i] = data[base_off + i] * scale;
}

// Dual Q+K HeadNorm: Gemma 4 applies the same per-head RmsNorm to Q (num_heads)
// and K (num_kv_heads) with separate weights. One launch over num_heads+num_kv_heads
// blocks halves the launch pair; per block this is bit-identical to llm_head_norm.
extern ""C"" __global__ void llm_head_norm_qk(
    float* __restrict__ q_data, const float* __restrict__ q_weight,
    float* __restrict__ k_data, const float* __restrict__ k_weight,
    int head_dim, int num_heads, int num_kv_heads, float eps, int weight_stride)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int blk = blockIdx.x;
    if ((int)blk >= num_heads + num_kv_heads) return;

    bool is_q = (int)blk < num_heads;
    int head = is_q ? (int)blk : (int)blk - num_heads;
    float* data = is_q ? q_data : k_data;
    const float* weight = is_q ? q_weight : k_weight;

    int base_off = head * head_dim;
    int w_off    = head * weight_stride;

    float sum = 0.f;
    for (int i = (int)tid; i < head_dim; i += 256) {
        float v = data[base_off + i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = (int)tid; i < head_dim; i += 256)
        data[base_off + i] = data[base_off + i] * scale * weight[w_off + i];
}

// ── Fused SiLU(gate) * up ──────────────────────────────────────────────────
// gate[i] = gate[i] / (1 + exp(-gate[i])) * up[i]
extern ""C"" __global__ void llm_silu_mul(
    float* __restrict__ gate, const float* __restrict__ up, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float g = gate[i];
    gate[i] = g / (1.0f + __expf(-g)) * up[i];
}

// ── Fused tanh-approximate GELU(gate) * up ─────────────────────────────────
// Gemma 4 FFN activation. Matches the CPU reference SimdKernels.GeluTanhMul:
//   inner = sqrt(2/π) * (g + 0.044715 * g^3)
//   out   = 0.5 * g * (1 + tanh(inner)) * up
// In-place into `gate` so the call-site signature mirrors llm_silu_mul.
extern ""C"" __global__ void llm_gelu_tanh_mul(
    float* __restrict__ gate, const float* __restrict__ up, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float g = gate[i];
    float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
    gate[i] = 0.5f * g * (1.0f + tanhf(inner)) * up[i];
}

// Strided-up variant: gate is [n_tok × width] contiguous; the up operand for
// token t lives at up + t*up_stride + up_offset (width contiguous floats). Lets
// batched PLE inject the per-layer slice of a [n_tok × (L*pleWidth)] projection
// buffer without a gather. Per element bit-identical to llm_gelu_tanh_mul.
extern ""C"" __global__ void llm_gelu_tanh_mul_strided(
    float* __restrict__ gate, const float* __restrict__ up,
    int width, long up_stride, long up_offset, int n_tok)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)n_tok * (long)width;
    if (idx >= total) return;
    long t = idx / width;
    long j = idx % width;
    float g = gate[idx];
    float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
    float u = up[t * up_stride + up_offset + j];
    gate[idx] = 0.5f * g * (1.0f + tanhf(inner)) * u;
}

// ── Final-logit softcap (in place) ─────────────────────────────────────────
// x[i] = tanh(x[i] / cap) * cap. Matches SimdKernels.SoftcapInPlace.
// Used by Gemma 4 for the final logit clipping (cap=30).
extern ""C"" __global__ void llm_softcap_inplace(
    float* __restrict__ x, int n, float cap)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    x[i] = tanhf(x[i] / cap) * cap;
}

// ── Element-wise sigmoid in-place ──────────────────────────────────────────
extern ""C"" __global__ void llm_sigmoid_inplace(float* __restrict__ x, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    x[i] = 1.0f / (1.0f + __expf(-x[i]));
}

// ── Greedy argmax (#219): two-pass block reduction over the vocab logits ─────
// Replaces the per-token full-vocab D2H + host scan when the sampler is greedy: the
// (idx, value) pair is reduced on-device and only 8 bytes are downloaded.
// Tie-break MUST match Sampler.Greedy (CPU): it scans left-to-right with a strict
// `>`, so the FIRST (lowest-index) occurrence of the maximum survives. Every reduction
// step here therefore keeps the LOWER index on an exact value tie. Threads init their
// running value to -FLT_MAX with index 0; for finite logits (the decode contract, which
// the coherence tests guard) every element beats the sentinel, so the result is bit-exact
// with the host scan. (A NaN-at-index-0 buffer is the one degenerate case where the host's
// `logits[0]` seed and this sentinel disagree — not reachable with finite decode logits.)
#define SHARPI_ARGMAX_NEG_INF (-3.402823466e38f)

extern ""C"" __global__ void llm_argmax_partial(
    const float* __restrict__ logits, int n, float* __restrict__ partialVal, int* __restrict__ partialIdx)
{
    __shared__ float sVal[256];
    __shared__ int   sIdx[256];
    int tid = (int)threadIdx.x;
    float best = SHARPI_ARGMAX_NEG_INF;
    int bestIdx = 0;
    for (int i = (int)(blockIdx.x * blockDim.x) + tid; i < n; i += (int)(gridDim.x * blockDim.x))
    {
        float v = logits[i];
        if (v > best) { best = v; bestIdx = i; }
    }
    sVal[tid] = best; sIdx[tid] = bestIdx;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            float ov = sVal[tid + s]; int oi = sIdx[tid + s];
            if (ov > sVal[tid] || (ov == sVal[tid] && oi < sIdx[tid])) { sVal[tid] = ov; sIdx[tid] = oi; }
        }
        __syncthreads();
    }
    if (tid == 0) { partialVal[blockIdx.x] = sVal[0]; partialIdx[blockIdx.x] = sIdx[0]; }
}

// Second pass: one block reduces the per-block partials. out[0] holds the winning index
// (raw int bits), out[1] the winning value (float).
extern ""C"" __global__ void llm_argmax_final(
    const float* __restrict__ partialVal, const int* __restrict__ partialIdx, int numParts, void* out)
{
    __shared__ float sVal[256];
    __shared__ int   sIdx[256];
    int tid = (int)threadIdx.x;
    float best = SHARPI_ARGMAX_NEG_INF;
    int bestIdx = 0;
    for (int i = tid; i < numParts; i += (int)blockDim.x)
    {
        float v = partialVal[i]; int idx = partialIdx[i];
        if (v > best || (v == best && idx < bestIdx)) { best = v; bestIdx = idx; }
    }
    sVal[tid] = best; sIdx[tid] = bestIdx;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            float ov = sVal[tid + s]; int oi = sIdx[tid + s];
            if (ov > sVal[tid] || (ov == sVal[tid] && oi < sIdx[tid])) { sVal[tid] = ov; sIdx[tid] = oi; }
        }
        __syncthreads();
    }
    if (tid == 0) { ((int*)out)[0] = sIdx[0]; ((float*)out)[1] = sVal[0]; }
}

// Batched argmax (#219, MTP/spec verify): one block per row reduces row `blockIdx.x` of a
// [rows × rowStride] buffer (the packed per-position verify logits), writing (idx, value) pairs
// to out[2*row], out[2*row+1]. Same lowest-index tie-break as the single-row kernels. Rows run on
// separate SMs in parallel, so k argmaxes cost ~one row's reduction instead of k full-vocab D2H.
extern ""C"" __global__ void llm_argmax_rows(
    const float* __restrict__ logits, int n, int rowStride, void* out)
{
    __shared__ float sVal[256];
    __shared__ int   sIdx[256];
    int row = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* r = logits + (long long)row * (long long)rowStride;
    float best = SHARPI_ARGMAX_NEG_INF;
    int bestIdx = 0;
    for (int i = tid; i < n; i += (int)blockDim.x)
    {
        float v = r[i];
        if (v > best) { best = v; bestIdx = i; }
    }
    sVal[tid] = best; sIdx[tid] = bestIdx;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            float ov = sVal[tid + s]; int oi = sIdx[tid + s];
            if (ov > sVal[tid] || (ov == sVal[tid] && oi < sIdx[tid])) { sVal[tid] = ov; sIdx[tid] = oi; }
        }
        __syncthreads();
    }
    if (tid == 0) { ((int*)out)[row * 2] = sIdx[0]; ((float*)out)[row * 2 + 1] = sVal[0]; }
}

// ── Softmax in place ───────────────────────────────────────────────────────
// 1 block of 256 threads. 3-pass: max, exp+sum, normalize.
extern ""C"" __global__ void llm_softmax(float* __restrict__ x, int n)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;

    float local_max = sharpi_neg_inf();
    for (int i = (int)tid; i < n; i += 256)
        local_max = fmaxf(local_max, x[i]);
    sdata[tid] = local_max;
    __syncthreads();

    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    float local_sum = 0.f;
    for (int i = (int)tid; i < n; i += 256) {
        float e = __expf(x[i] - max_val);
        x[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int i = (int)tid; i < n; i += 256)
        x[i] *= inv_sum;
}

// ── Buffer clear (zero) ────────────────────────────────────────────────────
extern ""C"" __global__ void llm_clear_f32(float* __restrict__ dst, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    dst[i] = 0.f;
}

// ── Index-based row gather / scatter (GPU op-offload MoE prefill) ───────────
// One launch each replaces the per-row CopyDeviceRegion loops in the CPU-MoE
// GPU-offload routed-prefill path: rowIdx is CSR-ordered (expert-bucketed) so a
// single grid-stride pass moves all nRows·cols floats. Each destination slot is
// written exactly once (the CSR covers every (token,slot) selection once), so no
// atomics are needed in the scatter.
//   gather:  dst[g*cols + c] = src[rowIdx[g]*cols + c]
extern ""C"" __global__ void llm_gather_rows(
    float* __restrict__ dst, const float* __restrict__ src,
    const int* __restrict__ rowIdx, int nRows, int cols)
{
    long total = (long)nRows * cols;
    for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (long)gridDim.x * blockDim.x)
    {
        int g = (int)(t / cols);
        int c = (int)(t - (long)g * cols);
        dst[t] = src[(long)rowIdx[g] * cols + c];
    }
}

//   scatter: dst[dstRowIdx[g]*cols + c] = src[g*cols + c]
extern ""C"" __global__ void llm_scatter_rows(
    float* __restrict__ dst, const float* __restrict__ src,
    const int* __restrict__ dstRowIdx, int nRows, int cols)
{
    long total = (long)nRows * cols;
    for (long t = (long)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (long)gridDim.x * blockDim.x)
    {
        int g = (int)(t / cols);
        int c = (int)(t - (long)g * cols);
        dst[(long)dstRowIdx[g] * cols + c] = src[t];
    }
}

// ── Memory-bound baseline (diagnostic only) ────────────────────────────────
// Each thread reads/writes one uint4 (16 bytes), the access width NVIDIA's
// own bandwidthTest uses. Saturates HBM at ~400 GB/s on RTX 4070 Ti when the
// system is healthy.
extern ""C"" __global__ void llm_bw_baseline(
    const uint4* __restrict__ src,
    uint4* __restrict__ dst,
    int n_uint4)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n_uint4) return;
    dst[idx] = src[idx];   // pure streaming copy — 16 bytes/thread = 1 cache sector/lane
}

// ── RoPE (interleaved pairs (2i, 2i+1)) ────────────────────────────────────
// Each thread handles one pair across all heads. total_pairs = num_heads * head_dim/2.
extern ""C"" __global__ void llm_rope_interleaved(
    float* __restrict__ x,
    int num_heads, int head_dim, int position, float theta)
{
    int pair_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int half_dim = head_dim / 2;
    int total_pairs = num_heads * half_dim;
    if (pair_idx >= total_pairs) return;

    int h = pair_idx / half_dim;
    int i = pair_idx % half_dim;

    float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)head_dim);
    float angle = (float)position * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    int base = h * head_dim + 2 * i;
    float x0 = x[base];
    float x1 = x[base + 1];
    x[base]     = x0 * c - x1 * s;
    x[base + 1] = x0 * s + x1 * c;
}

// ── RoPE NEOX (pairs offset by head_dim/2) ─────────────────────────────────
extern ""C"" __global__ void llm_rope_neox(
    float* __restrict__ x,
    int num_heads, int head_dim, int position, float theta)
{
    int pair_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int half_dim = head_dim / 2;
    int total_pairs = num_heads * half_dim;
    if (pair_idx >= total_pairs) return;

    int h = pair_idx / half_dim;
    int i = pair_idx % half_dim;

    float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)head_dim);
    float angle = (float)position * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    int head_base = h * head_dim;
    int a = head_base + i;
    int b = head_base + i + half_dim;
    float x0 = x[a];
    float x1 = x[b];
    x[a] = x0 * c - x1 * s;
    x[b] = x0 * s + x1 * c;
}

// ── RoPE NEOX with per-half-dim freq_factors (Gemma 4 global layers) ──────
// llama.cpp gemma4.cpp:191 passes `rope_freqs.weight` only for non-SWA layers
// of Gemma 4 / Gemma-3n: the table is size head_dim/2 and divides each pair's
// frequency, masking the high-frequency tail to ~identity for long context.
// Mirrors the CPU `SimdKernels.BuildRopeTable(..., globalFreqFactors)` path.
extern ""C"" __global__ void llm_rope_neox_with_factors(
    float* __restrict__ x,
    int num_heads, int head_dim, int position, float theta,
    const float* __restrict__ freq_factors)
{
    int pair_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int half_dim = head_dim / 2;
    int total_pairs = num_heads * half_dim;
    if (pair_idx >= total_pairs) return;

    int h = pair_idx / half_dim;
    int i = pair_idx % half_dim;

    float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)head_dim);
    freq /= freq_factors[i];
    float angle = (float)position * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    int head_base = h * head_dim;
    int a = head_base + i;
    int b = head_base + i + half_dim;
    float x0 = x[a];
    float x1 = x[b];
    x[a] = x0 * c - x1 * s;
    x[b] = x0 * s + x1 * c;
}

// Batched NEOX-with-factors RoPE over N tokens (Gemma 4 global layers in batched
// prefill). Position for token t is base_position + t; x row stride = num_heads*
// head_dim. Per row this is bit-identical to llm_rope_neox_with_factors.
extern ""C"" __global__ void llm_rope_neox_with_factors_batched(
    float* __restrict__ x,
    int num_heads, int head_dim, int base_position, float theta,
    const float* __restrict__ freq_factors, int n_tok)
{
    int pair_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int half_dim = head_dim / 2;
    int total_pairs = num_heads * half_dim;
    int token = (int)blockIdx.y;
    if (pair_idx >= total_pairs || token >= n_tok) return;

    int h = pair_idx / half_dim;
    int i = pair_idx % half_dim;
    int position = base_position + token;

    float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)head_dim);
    freq /= freq_factors[i];
    float angle = (float)position * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    long head_base = (long)token * (long)num_heads * (long)head_dim + (long)h * head_dim;
    long a = head_base + i;
    long b = head_base + i + half_dim;
    float x0 = x[a];
    float x1 = x[b];
    x[a] = x0 * c - x1 * s;
    x[b] = x0 * s + x1 * c;
}

// ── RoPE NEOX partial (rotate dims [0, rope_dim); pass dims [rope_dim, head_dim)) ──
// qwen35moe rotates only the first 64 of each 256-dim head. The frequency
// exponent uses `rope_dim` (not `head_dim`) — this matches the CPU reference
// `SimdKernels.ApplyRoPECachedNeoxPartial`, which precomputes the table from
// `rope_dim`. Pair layout: (i, i + rope_half_dim) for i ∈ [0, rope_half_dim).
extern ""C"" __global__ void llm_rope_neox_partial(
    float* __restrict__ x,
    int num_heads, int head_dim, int rope_dim, int position, float theta)
{
    int pair_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int rope_half_dim = rope_dim / 2;
    int total_pairs = num_heads * rope_half_dim;
    if (pair_idx >= total_pairs) return;

    int h = pair_idx / rope_half_dim;
    int i = pair_idx % rope_half_dim;

    float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)rope_dim);
    float angle = (float)position * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    int head_base = h * head_dim;
    int a = head_base + i;
    int b = head_base + i + rope_half_dim;
    float x0 = x[a];
    float x1 = x[b];
    x[a] = x0 * c - x1 * s;
    x[b] = x0 * s + x1 * c;
    // Dims [rope_dim, head_dim) pass through untouched.
}

// ── Element-wise multiply (output = a * b) ─────────────────────────────────
extern ""C"" __global__ void llm_mul(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    output[i] = a[i] * b[i];
}

// ── Fused sigmoid * mul in-place (x *= sigmoid(gate)) ──────────────────────
// One launch replaces the previous Sigmoid(gate) + ElementwiseMul(x, x, gate)
// pair for the qwen35moe GLU attention gate. Single elementwise pass.
extern ""C"" __global__ void llm_sigmoid_mul_inplace(
    float* __restrict__ x,
    const float* __restrict__ gate,
    int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float g = gate[i];
    x[i] *= 1.0f / (1.0f + __expf(-g));
}

// ── Strided de-interleave of [Q‖G] → Q, G ─────────────────────────────────
// qwen35moe's attn_q.weight emits per-head pairs `[Q[head_dim], G[head_dim]]`
// concatenated (output stride = 2 * head_dim per head). This kernel splits
// the interleaved buffer into two contiguous per-head outputs.
extern ""C"" __global__ void llm_split_qg(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ g,
    int num_heads, int head_dim)
{
    int total = num_heads * head_dim;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= total) return;

    int h = idx / head_dim;
    int j = idx % head_dim;
    int src_base = h * head_dim * 2;
    q[h * head_dim + j] = qg[src_base + j];
    g[h * head_dim + j] = qg[src_base + head_dim + j];
}

// ── KV cache append ────────────────────────────────────────────────────────
extern ""C"" __global__ void llm_kv_append(
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    int kv_dim, int position, int max_seq_len)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= kv_dim) return;
    // Ring slot: `position % max_seq_len`. `max_seq_len` is the allocated cache size,
    // so for a full-context (dense / global) cache `position < max_seq_len` makes this
    // the identity; for a window-sized SWA ring it wraps the write into the ring.
    long offset = (long)(position % max_seq_len) * (long)kv_dim + (long)i;
    k_cache[offset] = k_in[i];
    v_cache[offset] = v_in[i];
}

// ── KV cache append (bf16 store) ───────────────────────────────────────────
// FP32 K/V activations in → bf16 K/V cache out. Halves the KV-cache footprint
// at the cost of one fp32→bf16 conversion per element on the write. Read-side
// recovery happens in `llm_attention_bf16`. Used by `CudaHybridGdnForwardPass`
// for the hybrid GDN models (qwen35, qwen35moe, qwen35-MTP) — see issue #27.
extern ""C"" __global__ void llm_kv_append_bf16(
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    unsigned short* __restrict__ k_cache,
    unsigned short* __restrict__ v_cache,
    int kv_dim, int position, int max_seq_len)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= kv_dim) return;
    // Ring slot `position % max_seq_len` (identity for a full-context cache; wraps a
    // window-sized ring). Matches the f32 llm_kv_append so the write/read indexing stays
    // uniform if a windowed model ever uses the bf16 KV cache (today only the full-context
    // GDN-hybrid path does, where position < max_seq_len makes this the identity).
    long offset = (long)(position % max_seq_len) * (long)kv_dim + (long)i;
    k_cache[offset] = (unsigned short)sharpi_fp32_to_bf16(k_in[i]);
    v_cache[offset] = (unsigned short)sharpi_fp32_to_bf16(v_in[i]);
}

// ── KV cache append (q8_0 store, issue #179) ───────────────────────────────
// FP32 K/V activations in → block-quantized q8_0 K/V cache out (~quarter of the
// fp32 footprint). Each hardware warp (32 consecutive lanes) owns one q8_0 block:
// it warp-reduces the sub-block amax, derives the fp16 scale d = amax/127, and
// writes 32 int8 quants (rintf, clamped ±127 — the codebase's q8 convention, see
// llm_quantize_q8_1) + the scale. kv_dim is a multiple of 32, so block boundaries
// align to KV-row boundaries; the ring slot `position % max_seq_len` matches the
// f32/bf16 appends. Read-side recovery is sharpi_kvload(const block_q8_0*, ...).
__device__ __forceinline__ void sharpi_q8_append_one(
    float val, bool valid, block_q8_0* __restrict__ cache, long block_idx, int lane)
{
    // amax across the 32-lane sub-block (all lanes participate — no early return).
    float a = fabsf(val);
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 16));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  8));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  4));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  2));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  1));
    float d    = a / 127.f;
    // Threshold (not d==0): if the whole sub-block is subnormal-small, d is a subnormal,
    // 1/d overflows to +inf, and a zero lane's 0*inf = NaN → (int)NaN clamps to -127
    // instead of 0. 1e-30 is far below any real KV scale (d ≈ amax/127 ≈ 1e-3..1e-1) yet
    // above the subnormal danger zone, so real blocks are unaffected; an all-near-zero
    // block quantizes to all-zeros, which is correct to within its negligible magnitude.
    float invd = (d < 1e-30f) ? 0.f : (1.f / d);
    int   q    = (int)rintf(val * invd);
    if (q >  127) q =  127;
    if (q < -127) q = -127;
    if (!valid) return;
    block_q8_0* dst = cache + block_idx;
    dst->qs[lane] = (signed char)q;
    if (lane == 0) dst->d = (unsigned short)sharpi_fp32_to_fp16(d);
}

extern ""C"" __global__ void llm_kv_append_q8_0(
    const float* __restrict__ k_in,
    const float* __restrict__ v_in,
    block_q8_0* __restrict__ k_cache,
    block_q8_0* __restrict__ v_cache,
    int kv_dim, int position, int max_seq_len)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane = (int)(threadIdx.x & 31);
    bool valid = (i < kv_dim);
    // Flat element index in the cache; row = position % max_seq_len. Both kv_dim and
    // the row stride are multiples of 32, so `(row*kv_dim + i) >> 5` is the block.
    long row   = (long)(position % max_seq_len);
    long elem  = row * (long)kv_dim + (long)i;
    long block = elem >> 5;
    sharpi_q8_append_one(valid ? k_in[i] : 0.f, valid, k_cache, block, lane);
    sharpi_q8_append_one(valid ? v_in[i] : 0.f, valid, v_cache, block, lane);
}

// ── Embedding lookup (F32 table) ───────────────────────────────────────────
extern ""C"" __global__ void llm_embed_lookup_f32(
    const float* __restrict__ emb_table,
    float* __restrict__ output,
    int token_id, int emb_dim)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= emb_dim) return;
    long src = (long)token_id * (long)emb_dim + (long)i;
    output[i] = emb_table[src];
}

// ── Embedding lookup from Q4_K table ───────────────────────────────────────
// One block of 256 threads dequantizes one row (= emb_dim elements).
// emb_dim must be a multiple of 256 (Q4_K block size).
//
// Q4_K block (144 bytes, 36 uint32 words):
//   words[0]      : low 16 = d (fp16), high 16 = dmin (fp16)
//   words[1..3]   : 12 bytes of packed 6-bit (scale,min) pairs
//   words[4..35]  : 128 bytes of 4-bit nibbles (256 elements)
extern ""C"" __global__ void llm_embed_lookup_q4k(
    const unsigned int* __restrict__ emb_data,
    float* __restrict__ output,
    int token_id, int emb_dim)
{
    __shared__ unsigned int blk[36];
    unsigned int tid = threadIdx.x;

    int num_blocks = emb_dim >> 8;                  // emb_dim / 256
    int bytes_per_row = num_blocks * 144;
    long row_word_base = (long)token_id * (long)(bytes_per_row >> 2);  // bytes_per_row / 4

    for (int block = 0; block < num_blocks; block++) {
        long blk_word_base = row_word_base + (long)((block * 144) >> 2);   // == block * 36
        if (tid < 36)
            blk[tid] = emb_data[blk_word_base + tid];
        __syncthreads();

        // Block-shared d, dmin
        unsigned int w0 = blk[0];
        float d    = sharpi_fp16_to_fp32(w0 & 0xffffu);
        float dmin = sharpi_fp16_to_fp32(w0 >> 16);

        // Compute scale/min sub-indices the same way the Vulkan shader does.
        unsigned int chunk = tid >> 6;          // 0..3
        unsigned int sub   = tid & 63u;         // 0..63
        unsigned int is_upper = (sub >= 32u) ? 1u : 0u;
        unsigned int byte_pos = sub & 31u;

        // sm0/1/2 from words[1..3]
        unsigned int sm0 = blk[1];
        unsigned int sm1 = blk[2];
        unsigned int sm2 = blk[3];

        unsigned int si = chunk * 2u + is_upper;
        float sc, mn;
        if (si < 4u) {
            sc = (float)((sm0 >> (si * 8u)) & 63u);
            mn = (float)((sm1 >> (si * 8u)) & 63u);
        } else {
            unsigned int j = si - 4u;
            sc = (float)(((sm2 >> (j * 8u)) & 0xFu)
                       | (((sm0 >> (j * 8u + 6u)) & 3u) << 4));
            mn = (float)(((sm2 >> (j * 8u + 4u)) & 0xFu)
                       | (((sm1 >> (j * 8u + 6u)) & 3u) << 4));
        }

        // Read one quant byte at offset 16 + chunk*32 + byte_pos (relative to block start).
        unsigned int qword = blk[4u + chunk * 8u + (byte_pos >> 2)];
        unsigned int qbyte = (qword >> ((byte_pos & 3u) * 8u)) & 0xFFu;
        unsigned int nibble = is_upper ? (qbyte >> 4) : (qbyte & 0xFu);

        output[block * 256 + (int)tid] = d * sc * (float)nibble - dmin * mn;

        __syncthreads();
    }
}

// ── Embedding lookup from Q5_K table (issue #39) ───────────────────────────
// One block of 256 threads dequantizes one row (= emb_dim elements).
// emb_dim must be a multiple of 256 (Q5_K block size).
//
// Q5_K block (176 bytes per 256 elements):
//   [0:2]     fp16 d
//   [2:4]     fp16 dmin
//   [4:16]    scales[12] — packed 6-bit (scale, min) pairs
//   [16:48]   qh[32]     — high bit per element
//   [48:176]  ql[128]    — lower 4 bits, two elements per byte
//
// Element decoding mirrors llm_matvec_q5k's per-(chunk, lane) layout: for each
// 256-element block, tid splits into (chunk in 0..3, sub in 0..63, is_upper,
// byte_pos in 0..31). Lower 4 bits come from ql; high bit from qh; the
// (scale, min) pair is chosen by is_upper from the chunk's two pairs.
extern ""C"" __global__ void llm_embed_lookup_q5k(
    const unsigned char* __restrict__ emb_data,
    float* __restrict__ output,
    int token_id, int emb_dim)
{
    __shared__ unsigned char blk[176];
    unsigned int tid = threadIdx.x;

    int num_blocks = emb_dim >> 8;
    long bytes_per_row = (long)num_blocks * 176L;
    long row_byte_base = (long)token_id * bytes_per_row;

    for (int block = 0; block < num_blocks; block++) {
        long blk_byte_base = row_byte_base + (long)block * 176L;
        if (tid < 176)
            blk[tid] = emb_data[blk_byte_base + tid];
        __syncthreads();

        // d, dmin: first two fp16s (4 bytes).
        unsigned int dword0 =
              ((unsigned int)blk[0])
            | ((unsigned int)blk[1] << 8)
            | ((unsigned int)blk[2] << 16)
            | ((unsigned int)blk[3] << 24);
        float d    = sharpi_fp16_to_fp32(dword0 & 0xffffu);
        float dmin = sharpi_fp16_to_fp32(dword0 >> 16);

        unsigned int chunk    = tid >> 6;          // 0..3
        unsigned int sub      = tid & 63u;         // 0..63
        unsigned int is_upper = (sub >= 32u) ? 1u : 0u;
        unsigned int byte_pos = sub & 31u;         // 0..31

        // Decode the chunk's (scale, min) pair for our half (low if !is_upper,
        // high if is_upper). j = 2*chunk + is_upper. Same packing as Q4_K /
        // ggml's get_scale_min_k4. scales[] starts at blk[4], 12 bytes wide.
        unsigned int j = chunk * 2u + is_upper;
        float sc, mn;
        if (j < 4u) {
            // scales[j] low-6 bits = scale, scales[j+4] low-6 bits = min.
            sc = (float)(blk[4 + j] & 63u);
            mn = (float)(blk[4 + j + 4u] & 63u);
        } else {
            // Pairs 4..7: scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
            //             min   = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
            unsigned int a = blk[4 + j + 4u];       // scales[j+4]
            unsigned int b = blk[4 + (j - 4u)];     // scales[j-4]
            unsigned int c = blk[4 + j];            // scales[j]
            sc = (float)((a & 0xFu) | (((b >> 6) & 3u) << 4));
            mn = (float)(((a >> 4) & 0xFu) | (((c >> 6) & 3u) << 4));
        }

        // qh bit: chunk c uses bit (2c + is_upper) of qh[byte_pos].
        unsigned int qh_byte = blk[16 + byte_pos];
        unsigned int u = 1u << (2u * chunk + is_upper);
        int hbit = (qh_byte & u) != 0u ? 16 : 0;

        // ql byte: low nibble for !is_upper, high nibble for is_upper. Offset
        // within ql is chunk*32 + byte_pos (matvec uses the same address; the
        // (lo, hi) split there feeds two elements per byte from one lane).
        unsigned int ql_byte = blk[48 + chunk * 32u + byte_pos];
        unsigned int nibble = is_upper ? (ql_byte >> 4) : (ql_byte & 0xFu);

        output[block * 256 + (int)tid] = d * sc * (float)((int)nibble + hbit) - dmin * mn;

        __syncthreads();
    }
}

// ── Embedding lookup from Q6_K table (issue #124, Gemma 4 12B tied embedding) ─
// Q6_K block (210 bytes / 256 elems): [ql:128][qh:64][scales:16 int8][d:fp16].
// One CUDA block of 256 threads gathers one row (emb_dim elements) of token_id;
// thread tid emits element tid of each 256-element super-block. Per-element decode
// mirrors llm_matvec_q6k exactly. emb_dim must be a multiple of 256.
extern ""C"" __global__ void llm_embed_lookup_q6k(
    const unsigned char* __restrict__ emb_data,
    float* __restrict__ output,
    int token_id, int emb_dim)
{
    __shared__ unsigned char blk[210];
    unsigned int tid = threadIdx.x;

    int num_blocks = emb_dim >> 8;                  // emb_dim / 256
    long bytes_per_row = (long)num_blocks * 210L;
    long row_byte_base = (long)token_id * bytes_per_row;

    for (int block = 0; block < num_blocks; block++) {
        long base = row_byte_base + (long)block * 210L;
        if (tid < 210) blk[tid] = emb_data[base + tid];
        __syncthreads();

        unsigned int lane = tid & 31u;          // 0..31
        unsigned int g    = tid >> 5;           // group 0..7
        unsigned int isc  = lane >> 4;          // 0 or 1 (scale half)

        float d = sharpi_fp16_to_fp32((unsigned int)blk[208] | ((unsigned int)blk[209] << 8));
        float scale = d * (float)((signed char)blk[192 + 2u * g + isc]);

        // ql byte: groups {0,2}->ql0, {1,3}->ql1, {4,6}->ql2, {5,7}->ql3 (+lane).
        unsigned int ql_index = (g < 4u) ? (g & 1u) : (2u + (g & 1u));
        unsigned int ql_byte  = blk[ql_index * 32u + lane];
        unsigned int high     = (g >> 1) & 1u;  // groups 2,3,6,7 use the high nibble
        unsigned int nib      = high ? (ql_byte >> 4) : (ql_byte & 0xFu);

        // qh: groups 0-3 from qh0 (offset 128), 4-7 from qh1 (160); 2-bit field per group.
        unsigned int qh_byte = (g < 4u) ? blk[128 + lane] : blk[160 + lane];
        unsigned int shift   = 2u * (g & 3u);
        int q = (int)(nib | (((qh_byte >> shift) & 3u) << 4)) - 32;

        output[block * 256 + (int)tid] = scale * (float)q;
        __syncthreads();
    }
}

// ── Embedding lookup from Q8_0 table ───────────────────────────────────────
// Q8_0 block (34 bytes per 32 elements): [d:fp16][qs:32 × int8].
// One CUDA block of 256 threads dequantizes one row (= emb_dim elements).
// emb_dim must be a multiple of 256 (8 Q8_0 blocks per outer iteration); this
// holds for every transformer embedding dim in practice. Each iteration loads
// 8 Q8_0 blocks (= 272 bytes) into shared memory, then every thread emits one
// output element from the (block, lane) it owns.
extern ""C"" __global__ void llm_embed_lookup_q8_0(
    const unsigned char* __restrict__ emb_data,
    float* __restrict__ output,
    int token_id, int emb_dim)
{
    __shared__ unsigned char blk[272];   // 8 Q8_0 blocks × 34 bytes
    unsigned int tid = threadIdx.x;

    int num_blocks = emb_dim >> 5;                   // emb_dim / 32
    long bytes_per_row = (long)num_blocks * 34L;
    long row_byte_base = (long)token_id * bytes_per_row;

    int outer_iters = emb_dim >> 8;                  // emb_dim / 256
    for (int outer = 0; outer < outer_iters; outer++) {
        long base_byte = row_byte_base + (long)(outer * 8) * 34L;
        // Cooperative load of 272 bytes (8 blocks) — 256 threads cover all bytes
        // with one each (272 > 256, last 16 lanes do a second load).
        if (tid < 272) blk[tid] = emb_data[base_byte + tid];
        if (tid < 16)  blk[256 + tid] = emb_data[base_byte + 256 + tid];
        __syncthreads();

        // tid splits into (block_in_outer ∈ 0..7, lane ∈ 0..31).
        unsigned int block_in_outer = tid >> 5;
        unsigned int lane           = tid & 31u;

        // d (fp16) lives in bytes [b*34 .. b*34 + 1]; quants in [b*34 + 2 .. b*34 + 33].
        unsigned int block_off = block_in_outer * 34u;
        unsigned int d_bits = (unsigned int)blk[block_off]
                            | ((unsigned int)blk[block_off + 1u] << 8);
        float d = sharpi_fp16_to_fp32(d_bits);

        int q = (int)(signed char)blk[block_off + 2u + lane];
        output[outer * 256 + (int)tid] = d * (float)q;

        __syncthreads();
    }
}

// Batched Q8_0 embedding lookup: one block per query token (grid.x = n_tok),
// reading token_ids[blockIdx.x] and writing row blockIdx.x of output. Collapses
// the prefill's per-token EmbedLookupQ8_0 + copy (2·N host launches) into one
// launch — the per-token body is identical to llm_embed_lookup_q8_0.
extern ""C"" __global__ void llm_embed_lookup_q8_0_batched(
    const unsigned char* __restrict__ emb_data,
    float* __restrict__ output,           // [n_tok * emb_dim]
    const int* __restrict__ token_ids,    // [n_tok]
    int n_tok, int emb_dim)
{
    __shared__ unsigned char blk[272];
    unsigned int tid = threadIdx.x;
    int i = (int)blockIdx.x;
    if (i >= n_tok) return;

    int token_id = token_ids[i];
    float* out_row = output + (long)i * emb_dim;
    int num_blocks = emb_dim >> 5;
    long bytes_per_row = (long)num_blocks * 34L;
    long row_byte_base = (long)token_id * bytes_per_row;

    int outer_iters = emb_dim >> 8;
    for (int outer = 0; outer < outer_iters; outer++) {
        long base_byte = row_byte_base + (long)(outer * 8) * 34L;
        if (tid < 272) blk[tid] = emb_data[base_byte + tid];
        if (tid < 16)  blk[256 + tid] = emb_data[base_byte + 256 + tid];
        __syncthreads();

        unsigned int block_in_outer = tid >> 5;
        unsigned int lane           = tid & 31u;
        unsigned int block_off = block_in_outer * 34u;
        unsigned int d_bits = (unsigned int)blk[block_off]
                            | ((unsigned int)blk[block_off + 1u] << 8);
        float d = sharpi_fp16_to_fp32(d_bits);
        int q = (int)(signed char)blk[block_off + 2u + lane];
        out_row[outer * 256 + (int)tid] = d * (float)q;
        __syncthreads();
    }
}

// ── Contiguous-row dequant (issue #247: GPU-side Gemma-4 PLE pre-pass) ───────
// Dequantize n_rows CONTIGUOUS packed rows into an f32 [n_rows × row_dim] buffer:
// row i of src → row i of dst (no token-id indirection — the caller has already
// gathered the PLE rows for the prompt's tokens into a packed quant buffer, so the
// expensive per-element dequant runs on the GPU instead of a CPU Parallel.For + a
// 4×-larger float upload). The per-row decode is byte-for-byte identical to
// llm_embed_lookup_q8_0 (same cvt.f32.f16 scale × int8), so the batched PLE row is
// bit-identical to the CPU Dequantize.ToFloat32 the per-token oracle uses. row_dim
// must be a multiple of 256 (8 Q8_0 blocks per outer iteration).
extern ""C"" __global__ void llm_dequant_rows_q8_0(
    const unsigned char* __restrict__ src,    // [n_rows * row_dim] Q8_0 packed
    float* __restrict__ dst,                  // [n_rows * row_dim] f32
    int n_rows, int row_dim)
{
    __shared__ unsigned char blk[272];
    unsigned int tid = threadIdx.x;
    int i = (int)blockIdx.x;
    if (i >= n_rows) return;

    int num_blocks = row_dim >> 5;
    long bytes_per_row = (long)num_blocks * 34L;
    long row_byte_base = (long)i * bytes_per_row;
    float* out_row = dst + (long)i * row_dim;

    int outer_iters = row_dim >> 8;
    for (int outer = 0; outer < outer_iters; outer++) {
        long base_byte = row_byte_base + (long)(outer * 8) * 34L;
        if (tid < 272) blk[tid] = src[base_byte + tid];
        if (tid < 16)  blk[256 + tid] = src[base_byte + 256 + tid];
        __syncthreads();

        unsigned int block_in_outer = tid >> 5;
        unsigned int lane           = tid & 31u;
        unsigned int block_off = block_in_outer * 34u;
        unsigned int d_bits = (unsigned int)blk[block_off]
                            | ((unsigned int)blk[block_off + 1u] << 8);
        float d = sharpi_fp16_to_fp32(d_bits);
        int q = (int)(signed char)blk[block_off + 2u + lane];
        out_row[outer * 256 + (int)tid] = d * (float)q;
        __syncthreads();
    }
}

// Q6_K variant of llm_dequant_rows_q8_0. Per-element decode mirrors
// llm_embed_lookup_q6k / llm_matvec_q6k exactly ((d·scale)·q, same order as the
// CPU DequantQ6K), so the batched PLE row is bit-identical to the per-token CPU
// dequant. row_dim must be a multiple of 256.
extern ""C"" __global__ void llm_dequant_rows_q6k(
    const unsigned char* __restrict__ src,    // [n_rows * row_dim] Q6_K packed
    float* __restrict__ dst,                  // [n_rows * row_dim] f32
    int n_rows, int row_dim)
{
    __shared__ unsigned char blk[210];
    unsigned int tid = threadIdx.x;
    int i = (int)blockIdx.x;
    if (i >= n_rows) return;

    int num_blocks = row_dim >> 8;            // row_dim / 256
    long bytes_per_row = (long)num_blocks * 210L;
    long row_byte_base = (long)i * bytes_per_row;
    float* out_row = dst + (long)i * row_dim;

    for (int block = 0; block < num_blocks; block++) {
        long base = row_byte_base + (long)block * 210L;
        if (tid < 210) blk[tid] = src[base + tid];
        __syncthreads();

        unsigned int lane = tid & 31u;          // 0..31
        unsigned int g    = tid >> 5;           // group 0..7
        unsigned int isc  = lane >> 4;          // 0 or 1 (scale half)

        float d = sharpi_fp16_to_fp32((unsigned int)blk[208] | ((unsigned int)blk[209] << 8));
        float scale = d * (float)((signed char)blk[192 + 2u * g + isc]);

        unsigned int ql_index = (g < 4u) ? (g & 1u) : (2u + (g & 1u));
        unsigned int ql_byte  = blk[ql_index * 32u + lane];
        unsigned int high     = (g >> 1) & 1u;
        unsigned int nib      = high ? (ql_byte >> 4) : (ql_byte & 0xFu);

        unsigned int qh_byte = (g < 4u) ? blk[128 + lane] : blk[160 + lane];
        unsigned int shift   = 2u * (g & 3u);
        int q = (int)(nib | (((qh_byte >> shift) & 3u) << 4)) - 32;

        out_row[block * 256 + (int)tid] = scale * (float)q;
        __syncthreads();
    }
}

// ── MatVec F32 ─────────────────────────────────────────────────────────────
// 256 threads/block, 8 rows/block, 32 threads/row → warp reduce.
// One grid dim x covers ceil(rows/8) blocks.
extern ""C"" __global__ void llm_matvec_f32(
    const float* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    float acc = 0.f;
    long base = (long)row * (long)cols;
    for (int i = lane; i < cols; i += THREADS_PER_ROW)
        acc += weights[base + i] * input[i];

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output[row] = result;
}

// ── Quantize FP32 → Q8_1 (per-32-element sub-block) ───────────────────────
// Output layout (matches ggml block_q8_1): per 32 elements:
//   offset 0..3  : { fp16 d, fp16 _ }    (we leave the second half unused;
//                                         the Q4_K matvec re-derives Σq via
//                                         __dp4a(0x01010101, …))
//   offset 4..35 : 32 × int8 quantized samples
// One CUDA block of 32 threads quantizes one Q8_1 sub-block.
extern ""C"" __global__ void llm_quantize_q8_1(
    const float* __restrict__ x,
    unsigned char* __restrict__ out,
    int n)
{
    int block_id = (int)blockIdx.x;
    int lane     = (int)threadIdx.x;
    int elem_idx = block_id * 32 + lane;

    float val = (elem_idx < n) ? x[elem_idx] : 0.f;

    // amax across the 32-lane sub-block.
    float a = fabsf(val);
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 16));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  8));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  4));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  2));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  1));

    float d    = a / 127.f;
    float invd = (d == 0.f) ? 0.f : (1.f / d);
    int   q    = (int)rintf(val * invd);
    if (q >  127) q =  127;
    if (q < -127) q = -127;

    unsigned char* dst = out + (long)block_id * 36L;
    dst[4 + lane] = (unsigned char)(signed char)q;

    // #156 C2: per-sub-block activation sum Σq, dequantized as d·Σq, packed as the
    // fp16 `s` half at bytes [2:4]. The Q4_K MMQ min-bias term (super_dmin·mn·s) reads
    // it; every other q8_1 reader masks the d-word with 0xffff, so this high half is
    // otherwise inert (mirrors ggml block_q8_1's ds = {d, s}).
    int qsum = q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        qsum += __shfl_xor_sync(0xffffffffu, qsum, off);

    if (lane == 0) {
        // Pack {d, d·Σq} as two fp16 halves into one uint32 at offset 0..3.
        unsigned int d_bits = sharpi_fp32_to_fp16(d);
        unsigned int s_bits = sharpi_fp32_to_fp16(d * (float)qsum);
        *reinterpret_cast<unsigned int*>(dst) = d_bits | (s_bits << 16);
    }
}

// ── Quantize FP32 → Q8_1, SoA activation layout (Track A, #124/#173) ────────
// Identical per-32-element quantization math to llm_quantize_q8_1, but emits the
// struct-of-arrays layout the activation-coalesced MMQ wants:
//   qs_out : contiguous int8 quants, block `b` at [b*32 .. b*32+31]  (32 B/block,
//            no 4-B header gap → a token's nb blocks are 32*nb contiguous bytes)
//   ds_out : one uint32 per block, packed { fp16 d, fp16 d·Σq } — the SAME {d,s}
//            word llm_quantize_q8_1 stores at bytes [0:4] of each AoS block, so
//            every reader extracts d = ds & 0xffff and s = ds >> 16 unchanged.
// Splitting d/s out is what lets the MMQ load a token's qs as aligned, contiguous
// 128-B lines (the AoS 36-B stride wastes 4 B/block + scatters across tokens — the
// uncoalesced bulk ncu flagged). Values are bit-identical to the AoS producer;
// only the byte layout differs. One CUDA block of 32 threads = one sub-block.
extern ""C"" __global__ void llm_quantize_q8_1_soa(
    const float* __restrict__ x,
    unsigned char* __restrict__ qs_out,   // [n/32 blocks × 32 int8], contiguous
    unsigned char* __restrict__ ds_out,   // [n/32 blocks × uint32 {d,s}]
    int n)
{
    int block_id = (int)blockIdx.x;
    int lane     = (int)threadIdx.x;
    int elem_idx = block_id * 32 + lane;

    float val = (elem_idx < n) ? x[elem_idx] : 0.f;

    float a = fabsf(val);
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 16));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  8));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  4));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  2));
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a,  1));

    float d    = a / 127.f;
    float invd = (d == 0.f) ? 0.f : (1.f / d);
    int   q    = (int)rintf(val * invd);
    if (q >  127) q =  127;
    if (q < -127) q = -127;

    // Contiguous quant store: 32 B/block, no header.
    qs_out[(long)block_id * 32L + lane] = (unsigned char)(signed char)q;

    int qsum = q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        qsum += __shfl_xor_sync(0xffffffffu, qsum, off);

    if (lane == 0) {
        unsigned int d_bits = sharpi_fp32_to_fp16(d);
        unsigned int s_bits = sharpi_fp32_to_fp16(d * (float)qsum);
        reinterpret_cast<unsigned int*>(ds_out)[block_id] = d_bits | (s_bits << 16);
    }
}

// ── MatVec Q4_K  —  __dp4a / Q8_1 path ────────────────────────────────────
// Mirrors llama.cpp's mul_mat_vec_q4_K_q8_1:
//   • 1 output row per CUDA block; 4 warps (128 threads) cooperate on it.
//   • Input vector is pre-quantized to Q8_1 (32-element sub-blocks, one fp16 d
//     plus 32 int8 quants per sub-block).
//   • Each thread handles 8 weight nibbles per Q4_K super-block (32-byte chunk
//     split into 4 low nibbles + 4 high nibbles) and the corresponding
//     8 activations via two __dp4a instructions.
//   • Min correction Σ_j q_y[j] is re-derived in the same dp4a pipeline via
//     __dp4a(0x01010101, act, 0) — no second pass over the data.
//
// Lane layout for one super-block:
//   chunk     = lane >> 3            ∈ {0,1,2,3}
//   byte_off  = (lane & 7) * 4       ∈ {0,4,8,…,28}
// Each lane reads ONE uint32 of weight bytes (4 packed bytes = 8 nibbles =
// 4 low nibbles + 4 high nibbles), and ONE uint32 of activations per
// nibble polarity (4 int8 each).
#define MATVEC_Q4K_NWARPS 8
extern ""C"" __global__ void llm_matvec_q4k(
    const unsigned int* __restrict__ weights,
    const unsigned char* __restrict__ y_q81,
    float* __restrict__ output,
    int rows, int cols)
{
    int row     = (int)blockIdx.x;
    int warp_id = (int)threadIdx.y;     // 0..NWARPS-1
    int lane    = (int)threadIdx.x;     // 0..31
    if (row >= rows) return;

    int num_blocks = cols >> 8;                       // cols / 256
    long word_row_base = (long)row * (long)num_blocks * 36L;

    int chunk    = lane >> 3;                         // 0..3
    int byte_off = (lane & 7) * 4;                    // 0,4,8,…,28
    int q4_offset = 4 + chunk * 8 + (lane & 7);       // weight word index within super-block

    // Q8_1 layout: 36 bytes per 32-element sub-block; 8 sub-blocks per Q4_K super-block.
    // Sub-block index 2*chunk feeds low-nibble elems, 2*chunk+1 feeds high-nibble elems.

    float acc = 0.f;

    for (int block = warp_id; block < num_blocks; block += MATVEC_Q4K_NWARPS) {
        long word_base = word_row_base + (long)block * 36L;

        unsigned int w0  = __ldg(&weights[word_base]);
        unsigned int sm0 = __ldg(&weights[word_base + 1]);
        unsigned int sm1 = __ldg(&weights[word_base + 2]);
        unsigned int sm2 = __ldg(&weights[word_base + 3]);
        float super_d    = sharpi_fp16_to_fp32(w0 & 0xffffu);
        float super_dmin = sharpi_fp16_to_fp32(w0 >> 16);

        // Extract this lane's two sub-block (sc, m) pairs: lo = 2*chunk, hi = 2*chunk+1.
        unsigned int sc_lo, mn_lo, sc_hi, mn_hi;
        switch (chunk) {
            case 0:
                sc_lo = (sm0)       & 63u; mn_lo = (sm1)       & 63u;
                sc_hi = (sm0 >>  8) & 63u; mn_hi = (sm1 >>  8) & 63u;
                break;
            case 1:
                sc_lo = (sm0 >> 16) & 63u; mn_lo = (sm1 >> 16) & 63u;
                sc_hi = (sm0 >> 24) & 63u; mn_hi = (sm1 >> 24) & 63u;
                break;
            case 2:
                sc_lo = (sm2        & 0xFu) | (((sm0 >>  6) & 3u) << 4);
                mn_lo = ((sm2 >>  4) & 0xFu) | (((sm1 >>  6) & 3u) << 4);
                sc_hi = ((sm2 >>  8) & 0xFu) | (((sm0 >> 14) & 3u) << 4);
                mn_hi = ((sm2 >> 12) & 0xFu) | (((sm1 >> 14) & 3u) << 4);
                break;
            default: // chunk == 3
                sc_lo = ((sm2 >> 16) & 0xFu) | (((sm0 >> 22) & 3u) << 4);
                mn_lo = ((sm2 >> 20) & 0xFu) | (((sm1 >> 22) & 3u) << 4);
                sc_hi = ((sm2 >> 24) & 0xFu) | (((sm0 >> 30) & 3u) << 4);
                mn_hi = ((sm2 >> 28) & 0xFu) | (((sm1 >> 30) & 3u) << 4);
                break;
        }

        // Load this lane's 4 weight bytes: 4 low nibbles + 4 high nibbles.
        unsigned int wq    = __ldg(&weights[word_base + q4_offset]);
        unsigned int wq_lo = wq & 0x0F0F0F0Fu;
        unsigned int wq_hi = (wq >> 4) & 0x0F0F0F0Fu;

        // Q8_1 base for this super-block.
        long q81_base_lo = (long)(block * 8 + chunk * 2)     * 36L;
        long q81_base_hi = (long)(block * 8 + chunk * 2 + 1) * 36L;

        // Per-sub-block d (fp16 in first 2 bytes of each sub-block).
        unsigned int d_bits_lo = __ldg(reinterpret_cast<const unsigned int*>(y_q81 + q81_base_lo)) & 0xffffu;
        unsigned int d_bits_hi = __ldg(reinterpret_cast<const unsigned int*>(y_q81 + q81_base_hi)) & 0xffffu;
        float d8_lo = sharpi_fp16_to_fp32(d_bits_lo);
        float d8_hi = sharpi_fp16_to_fp32(d_bits_hi);

        // 4 int8 activations per sub-block at byte offset (4 + byte_off).
        int act_lo = *reinterpret_cast<const int*>(y_q81 + q81_base_lo + 4 + byte_off);
        int act_hi = *reinterpret_cast<const int*>(y_q81 + q81_base_hi + 4 + byte_off);

        // dp4a: dot(4 nibbles in 8-bit lanes, 4 int8 activations).
        // Cast nibble-packed words to signed int to disambiguate the dp4a
        // overload — nibbles are in 0..15 so the signed reinterpretation is
        // bitwise identical and the int8×int8 lane semantics are what we want.
        int dot_lo   = __dp4a((int)wq_lo, act_lo, 0);
        int dot_hi   = __dp4a((int)wq_hi, act_hi, 0);
        // Σ_j q_y[j] for the min correction — same dp4a pipeline.
        int sum_lo   = __dp4a((int)0x01010101, act_lo, 0);
        int sum_hi   = __dp4a((int)0x01010101, act_hi, 0);

        // Per-sub-block contribution: super_d * (sc * d8 * dot) − super_dmin * (m * d8 * Σq_y).
        float coef_d_lo = super_d    * (float)sc_lo * d8_lo;
        float coef_m_lo = super_dmin * (float)mn_lo * d8_lo;
        float coef_d_hi = super_d    * (float)sc_hi * d8_hi;
        float coef_m_hi = super_dmin * (float)mn_hi * d8_hi;
        acc += coef_d_lo * (float)dot_lo - coef_m_lo * (float)sum_lo;
        acc += coef_d_hi * (float)dot_hi - coef_m_hi * (float)sum_hi;
    }

    // Intra-warp reduction.
    acc = sharpi_warp_reduce_sum(acc);

    // Inter-warp reduction via shared memory.
    __shared__ float warp_acc[MATVEC_Q4K_NWARPS];
    if (lane == 0) warp_acc[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q4K_NWARPS; w++) s += warp_acc[w];
        output[row] = s;
    }
}

// ── MatVec Q4_K  —  scale-pre-unpacked SoA decode path (issue #156) ─────────
// The AoS llm_matvec_q4k decode matvec is bandwidth-aligned (144-byte super-block
// is 16-byte aligned, every load coalesced) yet hits only ~74% of HBM peak vs the
// Q8_0 dp4a path's ~90% — the gap is per-super-block COMPUTE: the 6-bit (scale,min)
// unpack switch + super-scale float mults form a long dependent chain that starves
// memory-level parallelism. This path repacks each Q4_K super-block ONCE at upload
// into three contiguous regions so the matvec reads plain bytes (no switch):
//   [Q  rows*nb*128 B] 4-bit quants, verbatim (16-byte aligned per super-block)
//   [S  rows*nb*16  B] 8 unpacked scale bytes then 8 unpacked min bytes (0..63)
//   [D  rows*nb*4   B] {fp16 d | fp16 dmin} packed as one uint32
// The scale/min bytes are the IDENTICAL 6-bit integer values the AoS switch derives,
// so the dp4a arithmetic is unchanged → bit-identical to llm_matvec_q4k. Costs +4 B
// per 144-B super-block (+2.8%) of read traffic to delete the unpack chain; net win
// because the kernel was compute-overhang-bound, not bandwidth-bound.
#define MATVEC_Q4K_SOA_NWARPS 8
extern ""C"" __global__ void llm_matvec_q4k_soa(
    const unsigned int* __restrict__ weights,    // SoA: [Q][S][D]
    const unsigned char* __restrict__ y_q81,
    float* __restrict__ output,
    int rows, int cols)
{
    int row     = (int)blockIdx.x;
    int warp_id = (int)threadIdx.y;
    int lane    = (int)threadIdx.x;
    if (row >= rows) return;

    int num_blocks = cols >> 8;                       // cols / 256
    long totalSub  = (long)rows * num_blocks;         // super-blocks in the whole tensor

    // Region bases (byte offsets from `weights`).
    const unsigned char* qReg = (const unsigned char*)weights;
    const unsigned char* sReg = qReg + totalSub * 128L;
    const unsigned int*  dReg = (const unsigned int*)(sReg + totalSub * 16L);

    int chunk    = lane >> 3;                          // 0..3
    int byte_off = (lane & 7) * 4;                     // 0,4,…,28
    // Within this row's quant region, lane's uint32 word for super-block `block`:
    //   word = block*32 + chunk*8 + (lane&7)   (the AoS q4_offset minus the 4-word header).
    int q_word_in_block = chunk * 8 + (lane & 7);

    long row_blk_base = (long)row * num_blocks;        // super-block index of row start

    float acc = 0.f;

    for (int block = warp_id; block < num_blocks; block += MATVEC_Q4K_SOA_NWARPS) {
        long sb = row_blk_base + block;                // global super-block index

        // d / dmin: one aligned uint32.
        unsigned int dd  = __ldg(&dReg[sb]);
        float super_d    = sharpi_fp16_to_fp32(dd & 0xffffu);
        float super_dmin = sharpi_fp16_to_fp32(dd >> 16);

        // Scale/min for this lane's chunk = sub-blocks 2*chunk (lo) and 2*chunk+1 (hi).
        const unsigned char* sblk = sReg + sb * 16L;
        unsigned int sc_lo = sblk[2 * chunk];
        unsigned int sc_hi = sblk[2 * chunk + 1];
        unsigned int mn_lo = sblk[8 + 2 * chunk];
        unsigned int mn_hi = sblk[8 + 2 * chunk + 1];

        // This lane's 4 weight bytes (4 low nibbles + 4 high nibbles).
        unsigned int wq    = __ldg(&weights[sb * 32L + q_word_in_block]);
        unsigned int wq_lo = wq & 0x0F0F0F0Fu;
        unsigned int wq_hi = (wq >> 4) & 0x0F0F0F0Fu;

        // Q8_1 activation sub-blocks: 2*chunk (lo nibbles), 2*chunk+1 (hi nibbles).
        long q81_base_lo = (long)(block * 8 + chunk * 2)     * 36L;
        long q81_base_hi = (long)(block * 8 + chunk * 2 + 1) * 36L;
        unsigned int d_bits_lo = __ldg(reinterpret_cast<const unsigned int*>(y_q81 + q81_base_lo)) & 0xffffu;
        unsigned int d_bits_hi = __ldg(reinterpret_cast<const unsigned int*>(y_q81 + q81_base_hi)) & 0xffffu;
        float d8_lo = sharpi_fp16_to_fp32(d_bits_lo);
        float d8_hi = sharpi_fp16_to_fp32(d_bits_hi);

        int act_lo = *reinterpret_cast<const int*>(y_q81 + q81_base_lo + 4 + byte_off);
        int act_hi = *reinterpret_cast<const int*>(y_q81 + q81_base_hi + 4 + byte_off);

        int dot_lo = __dp4a((int)wq_lo, act_lo, 0);
        int dot_hi = __dp4a((int)wq_hi, act_hi, 0);
        int sum_lo = __dp4a((int)0x01010101, act_lo, 0);
        int sum_hi = __dp4a((int)0x01010101, act_hi, 0);

        float coef_d_lo = super_d    * (float)sc_lo * d8_lo;
        float coef_m_lo = super_dmin * (float)mn_lo * d8_lo;
        float coef_d_hi = super_d    * (float)sc_hi * d8_hi;
        float coef_m_hi = super_dmin * (float)mn_hi * d8_hi;
        acc += coef_d_lo * (float)dot_lo - coef_m_lo * (float)sum_lo;
        acc += coef_d_hi * (float)dot_hi - coef_m_hi * (float)sum_hi;
    }

    acc = sharpi_warp_reduce_sum(acc);
    __shared__ float warp_acc[MATVEC_Q4K_SOA_NWARPS];
    if (lane == 0) warp_acc[warp_id] = acc;
    __syncthreads();
    if (warp_id == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q4K_SOA_NWARPS; w++) s += warp_acc[w];
        output[row] = s;
    }
}

// One-time repack of an interleaved Q4_K weight [rows × nb × 144 B] into the SoA
// [Q rows*nb*128][S rows*nb*16][D rows*nb*4] layout consumed by llm_matvec_q4k_soa
// (issue #156). One thread per 256-element super-block: copies the 128 quant bytes
// verbatim, unpacks the 8 (scale,min) 6-bit pairs to 16 plain bytes, and packs
// {d, dmin} into one uint32. The unpack mirrors llm_matvec_q4k's switch exactly so
// the stored integers are bit-identical.
extern ""C"" __global__ void llm_q4k_repack_soa(
    const unsigned char* __restrict__ src,   // interleaved, 144 B/super-block
    unsigned char* __restrict__ dst,         // SoA [Q][S][D]
    int rows, int cols)
{
    long sb = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int nb = cols >> 8;
    long total = (long)rows * nb;
    if (sb >= total) return;

    long srcOff = sb * 144L;
    unsigned char* qDst = dst + sb * 128L;
    unsigned char* sDst = dst + total * 128L + sb * 16L;
    unsigned char* dDst = dst + total * (128L + 16L) + sb * 4L;

    // d / dmin (bytes 0..3) → D region.
    dDst[0] = src[srcOff];     dDst[1] = src[srcOff + 1];
    dDst[2] = src[srcOff + 2]; dDst[3] = src[srcOff + 3];

    // 12 packed scale bytes (4..15) → three uint32 sm0,sm1,sm2.
    unsigned int sm0 = (unsigned int)src[srcOff + 4]  | ((unsigned int)src[srcOff + 5]  << 8) | ((unsigned int)src[srcOff + 6]  << 16) | ((unsigned int)src[srcOff + 7]  << 24);
    unsigned int sm1 = (unsigned int)src[srcOff + 8]  | ((unsigned int)src[srcOff + 9]  << 8) | ((unsigned int)src[srcOff + 10] << 16) | ((unsigned int)src[srcOff + 11] << 24);
    unsigned int sm2 = (unsigned int)src[srcOff + 12] | ((unsigned int)src[srcOff + 13] << 8) | ((unsigned int)src[srcOff + 14] << 16) | ((unsigned int)src[srcOff + 15] << 24);

    // Unpack 8 (sc, m) pairs — identical to the llm_matvec_q4k switch (sub-block j,
    // chunk c=j>>1, lo=even / hi=odd).
    unsigned char sc[8], mn[8];
    sc[0] = (sm0)       & 63u; mn[0] = (sm1)       & 63u;
    sc[1] = (sm0 >>  8) & 63u; mn[1] = (sm1 >>  8) & 63u;
    sc[2] = (sm0 >> 16) & 63u; mn[2] = (sm1 >> 16) & 63u;
    sc[3] = (sm0 >> 24) & 63u; mn[3] = (sm1 >> 24) & 63u;
    sc[4] = (sm2        & 0xFu) | (((sm0 >>  6) & 3u) << 4); mn[4] = ((sm2 >>  4) & 0xFu) | (((sm1 >>  6) & 3u) << 4);
    sc[5] = ((sm2 >>  8) & 0xFu) | (((sm0 >> 14) & 3u) << 4); mn[5] = ((sm2 >> 12) & 0xFu) | (((sm1 >> 14) & 3u) << 4);
    sc[6] = ((sm2 >> 16) & 0xFu) | (((sm0 >> 22) & 3u) << 4); mn[6] = ((sm2 >> 20) & 0xFu) | (((sm1 >> 22) & 3u) << 4);
    sc[7] = ((sm2 >> 24) & 0xFu) | (((sm0 >> 30) & 3u) << 4); mn[7] = ((sm2 >> 28) & 0xFu) | (((sm1 >> 30) & 3u) << 4);
    #pragma unroll
    for (int j = 0; j < 8; j++) { sDst[j] = sc[j]; sDst[8 + j] = mn[j]; }

    // 128 quant bytes (16..143) → Q region, verbatim.
    #pragma unroll
    for (int i = 0; i < 128; i++) qDst[i] = src[srcOff + 16 + i];
}

// ── #204 Q6_K → SoA repack for the int8 decode-MMQ tile ─────────────────────
// One-time repack of an interleaved Q6_K weight [rows × nb × 210 B] into a SoA
// layout the llm_mmq_q6k_soa_acts_n16 tile consumes:
//   [Q  total*256][S  total*16][D  total*4]   (total = rows*nb super-blocks)
// Q stores, for each natural element e of the super-block, the SIGNED int8 weight
// (q6(e) − 32) at byte position e — i.e. the SAME natural-order layout the shared
// Q8_1 activation uses (sub-block sb = e>>5, word (e&31)>>2, byte (e&31)&3), so the
// kernel's a-fragment load mirrors the Q4_K tile's exactly (4 consecutive int8 per
// word). q6 ∈ [0,63] → q6−32 ∈ [−32,31] fits signed int8. S stores the 16 int8
// scales verbatim (bytes 192..207); D stores {fp16 d (bytes 208..209), 0 pad}. The
// reconstruction mirrors llm_dequant_q6k_to_f16 / DotQ6K exactly, so the int8 weight
// bytes are bit-identical to the matvec's pre-multiply (q − 32). One thread / super-block.
extern ""C"" __global__ void llm_q6k_repack_soa(
    const unsigned char* __restrict__ src,   // interleaved, 210 B/super-block
    unsigned char* __restrict__ dst,         // SoA [Q][S][D]
    int rows, int cols)
{
    long sb = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int nb = cols >> 8;
    long total = (long)rows * nb;
    if (sb >= total) return;

    long srcOff = sb * 210L;
    signed char*   qDst = (signed char*)(dst + sb * 256L);
    unsigned char* sDst = dst + total * 256L + sb * 16L;
    unsigned char* dDst = dst + total * (256L + 16L) + sb * 4L;

    const unsigned char* ql = src + srcOff;          // [0:128]
    const unsigned char* qh = src + srcOff + 128;    // [128:192]

    // 16 int8 scales (192..207) → S region, verbatim.
    #pragma unroll
    for (int i = 0; i < 16; i++) sDst[i] = src[srcOff + 192 + i];
    // {d (208,209), 0, 0} → D region.
    dDst[0] = src[srcOff + 208]; dDst[1] = src[srcOff + 209];
    dDst[2] = 0; dDst[3] = 0;

    // Reconstruct all 256 (q6 − 32) signed int8, element e → byte e (natural order).
    // group = e>>5 (0..7), lane = e&31; the ql/qh switch is identical to
    // llm_dequant_q6k_to_f16 (same byte/shift per group).
    #pragma unroll
    for (int e = 0; e < 256; e++) {
        int lane  = e & 31;
        int group = e >> 5;
        unsigned int qlb, qhb; int q6;
        switch (group) {
            case 0:  qlb = ql[  0 + lane]; qhb = qh[ 0 + lane]; q6 = (int)((qlb & 0xFu)        | (((qhb >> 0) & 3u) << 4)); break;
            case 1:  qlb = ql[ 32 + lane]; qhb = qh[ 0 + lane]; q6 = (int)((qlb & 0xFu)        | (((qhb >> 2) & 3u) << 4)); break;
            case 2:  qlb = ql[  0 + lane]; qhb = qh[ 0 + lane]; q6 = (int)(((qlb >> 4) & 0xFu) | (((qhb >> 4) & 3u) << 4)); break;
            case 3:  qlb = ql[ 32 + lane]; qhb = qh[ 0 + lane]; q6 = (int)(((qlb >> 4) & 0xFu) | (((qhb >> 6) & 3u) << 4)); break;
            case 4:  qlb = ql[ 64 + lane]; qhb = qh[32 + lane]; q6 = (int)((qlb & 0xFu)        | (((qhb >> 0) & 3u) << 4)); break;
            case 5:  qlb = ql[ 96 + lane]; qhb = qh[32 + lane]; q6 = (int)((qlb & 0xFu)        | (((qhb >> 2) & 3u) << 4)); break;
            case 6:  qlb = ql[ 64 + lane]; qhb = qh[32 + lane]; q6 = (int)(((qlb >> 4) & 0xFu) | (((qhb >> 4) & 3u) << 4)); break;
            default: qlb = ql[ 96 + lane]; qhb = qh[32 + lane]; q6 = (int)(((qlb >> 4) & 0xFu) | (((qhb >> 6) & 3u) << 4)); break;
        }
        qDst[e] = (signed char)(q6 - 32);
    }
}

// ── MatVec Q6_K ────────────────────────────────────────────────────────────
// Q6_K block (210 bytes per 256 elements):
//   [0:128]   ql — lower 4-bit nibbles (two 64-byte halves)
//   [128:192] qh — upper 2-bit pairs   (two 32-byte halves)
//   [192:208] 16 int8 scales
//   [208:210] FP16 super-block scale d
// Reads through uint32 words via byte gathers; mirrors Vulkan MatVecQ6K.
extern ""C"" __global__ void llm_matvec_q6k(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long row_base_bytes = (long)row * (long)num_blocks * 210L;

    float acc = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 210L;

        // d: FP16 at offset 208 (two bytes)
        unsigned int dlo = (weights[(b0 + 208) >> 2] >> (((b0 + 208) & 3) * 8)) & 0xFFu;
        unsigned int dhi = (weights[(b0 + 209) >> 2] >> (((b0 + 209) & 3) * 8)) & 0xFFu;
        float d = sharpi_fp16_to_fp32(dlo | (dhi << 8));

        // Eight signed-int8 scales for this lane (isc = lane >> 4 selects 0 or 1).
        long isc = (long)(lane >> 4);

        float sc0 = d * (float)sharpi_int8_at(weights, b0 + 192 + isc);
        float sc1 = d * (float)sharpi_int8_at(weights, b0 + 194 + isc);
        float sc2 = d * (float)sharpi_int8_at(weights, b0 + 196 + isc);
        float sc3 = d * (float)sharpi_int8_at(weights, b0 + 198 + isc);
        float sc4 = d * (float)sharpi_int8_at(weights, b0 + 200 + isc);
        float sc5 = d * (float)sharpi_int8_at(weights, b0 + 202 + isc);
        float sc6 = d * (float)sharpi_int8_at(weights, b0 + 204 + isc);
        float sc7 = d * (float)sharpi_int8_at(weights, b0 + 206 + isc);

        // Six quant bytes per lane.
        unsigned int ql0 = sharpi_byte_at(weights, b0 +   0 + lane);
        unsigned int ql1 = sharpi_byte_at(weights, b0 +  32 + lane);
        unsigned int ql2 = sharpi_byte_at(weights, b0 +  64 + lane);
        unsigned int ql3 = sharpi_byte_at(weights, b0 +  96 + lane);
        unsigned int qh0 = sharpi_byte_at(weights, b0 + 128 + lane);
        unsigned int qh1 = sharpi_byte_at(weights, b0 + 160 + lane);

        int base_elem = block * 256;

        acc += sc0 * (float)((int)((ql0 & 0xFu)        | (((qh0 >> 0) & 3u) << 4)) - 32) * input[base_elem +       lane];
        acc += sc1 * (float)((int)((ql1 & 0xFu)        | (((qh0 >> 2) & 3u) << 4)) - 32) * input[base_elem +  32 + lane];
        acc += sc2 * (float)((int)(((ql0 >> 4) & 0xFu) | (((qh0 >> 4) & 3u) << 4)) - 32) * input[base_elem +  64 + lane];
        acc += sc3 * (float)((int)(((ql1 >> 4) & 0xFu) | (((qh0 >> 6) & 3u) << 4)) - 32) * input[base_elem +  96 + lane];
        acc += sc4 * (float)((int)((ql2 & 0xFu)        | (((qh1 >> 0) & 3u) << 4)) - 32) * input[base_elem + 128 + lane];
        acc += sc5 * (float)((int)((ql3 & 0xFu)        | (((qh1 >> 2) & 3u) << 4)) - 32) * input[base_elem + 160 + lane];
        acc += sc6 * (float)((int)(((ql2 >> 4) & 0xFu) | (((qh1 >> 4) & 3u) << 4)) - 32) * input[base_elem + 192 + lane];
        acc += sc7 * (float)((int)(((ql3 >> 4) & 0xFu) | (((qh1 >> 6) & 3u) << 4)) - 32) * input[base_elem + 224 + lane];
    }

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output[row] = result;
}

// ── MatVec Q6_K — SoA (#204) ────────────────────────────────────────────────
// Bit-identical clone of llm_matvec_q6k over the scale-pre-unpacked SoA layout
// (llm_q6k_repack_soa): [Q total*256 signed-int8 (q6−32)][S total*16 int8 scales]
// [D total*4 {fp16 d, 0, 0}], super-block index g = row*nb + block. Element e of
// the super-block is the signed weight Q[g*256 + e]; its scale is S[g*16 + (e>>4)];
// d is D[g*4]. The matvec reads the SAME element order (group 0..7, lane) and forms
// the SAME sc·(q6−32)·input products in the SAME reduction order as the AoS kernel,
// so the output is bit-identical to llm_matvec_q6k (no bit-unpacking, no scale gather).
extern ""C"" __global__ void llm_matvec_q6k_soa(
    const unsigned char* __restrict__ weights,   // SoA [Q][S][D]
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long total_sb = (long)rows * num_blocks;
    const signed char*   qReg = (const signed char*)weights;                 // [Q] total*256 B
    const signed char*   sReg = (const signed char*)weights + total_sb * 256L; // [S] total*16 B
    const unsigned char*  dReg = (const unsigned char*)weights + total_sb * (256L + 16L); // [D] total*4 B

    long isc = (long)(lane >> 4);   // scale half within the 32-element group

    float acc = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long g = (long)row * num_blocks + block;
        const signed char* q = qReg + g * 256L;       // 256 signed int8 (q6-32)
        const signed char* s = sReg + g * 16L;         // 16 int8 scales
        unsigned int dbits = (unsigned int)(*(const unsigned short*)(dReg + g * 4L));   // #204 review: dReg 16-B aligned, g*4 aligned → single 16-bit d load
        float d = sharpi_fp16_to_fp32(dbits);

        float sc0 = d * (float)s[ 0 + isc];
        float sc1 = d * (float)s[ 2 + isc];
        float sc2 = d * (float)s[ 4 + isc];
        float sc3 = d * (float)s[ 6 + isc];
        float sc4 = d * (float)s[ 8 + isc];
        float sc5 = d * (float)s[10 + isc];
        float sc6 = d * (float)s[12 + isc];
        float sc7 = d * (float)s[14 + isc];

        int base_elem = block * 256;
        acc += sc0 * (float)q[       lane] * input[base_elem +       lane];
        acc += sc1 * (float)q[ 32 + lane] * input[base_elem +  32 + lane];
        acc += sc2 * (float)q[ 64 + lane] * input[base_elem +  64 + lane];
        acc += sc3 * (float)q[ 96 + lane] * input[base_elem +  96 + lane];
        acc += sc4 * (float)q[128 + lane] * input[base_elem + 128 + lane];
        acc += sc5 * (float)q[160 + lane] * input[base_elem + 160 + lane];
        acc += sc6 * (float)q[192 + lane] * input[base_elem + 192 + lane];
        acc += sc7 * (float)q[224 + lane] * input[base_elem + 224 + lane];
    }

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output[row] = result;
}

// ── MatVec Q5_K ────────────────────────────────────────────────────────────
// Q5_K block (176 bytes per 256 elements):
//   [0:2]     fp16 d  — super-block scale
//   [2:4]     fp16 dmin — super-block min
//   [4:16]    scales[12] — packed 6-bit (scale, min) pairs (8 pairs, same packing as Q4_K)
//   [16:48]   qh[32]   — high bit per element (one bit, 8 polarities × 32 elements)
//   [48:176]  ql[128]  — lower 4 bits, two elements per byte (128 bytes × 2 nibbles = 256 elems)
//
// CPU reference: Dequantize.DequantQ5K / SimdKernels.DotQ5K_Scalar. Layout matches
// llama.cpp's block_q5_K in ggml-common.h. We mirror Q6_K's launch geometry
// (8 rows/block × 32 threads/row) and read weight bytes through byte-gather
// helpers so the kernel works against uint32-strided uploads.
extern ""C"" __global__ void llm_matvec_q5k(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long row_base_bytes = (long)row * (long)num_blocks * 176L;

    float acc = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 176L;

        // d, dmin at offsets 0..3 (two fp16 halves packed in the first uint32).
        unsigned int dword0 = __ldg(&weights[b0 >> 2]);
        float d    = sharpi_fp16_to_fp32(dword0 & 0xffffu);
        float dmin = sharpi_fp16_to_fp32(dword0 >> 16);

        // qh byte at offset 16 + lane (lane ∈ 0..31 covers the full 32-byte qh array).
        unsigned int qh_byte = sharpi_byte_at(weights, b0 + 16 + lane);

        int base_elem = block * 256;

        // 4 chunks × 64 elements each. Chunk c uses (sc[2c], m[2c]) for the low
        // 32 elems and (sc[2c+1], m[2c+1]) for the high 32 elems. qh bit shifts
        // for the low/high polarity within each chunk are (2c, 2c+1).
        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            // Decode the two 6-bit (scale, min) pairs for this chunk. Matches
            // get_scale_min_k4 in ggml-quants.c; the 12-byte `scales[]` array
            // starts at b0 + 4. Pairs 0..3 are stored as low-6-bit fields; pairs
            // 4..7 splice 4 bits from the second half with 2 bits from the first
            // half. We unpack both pairs in this chunk inline.
            unsigned int sc_lo_byte, sc_hi_byte;
            unsigned int sc1, m1, sc2, m2;
            int j_lo = chunk * 2;
            int j_hi = j_lo + 1;
            if (j_lo < 4) {
                // Pairs 0..3: q[j] holds scale<<2 in low 6 bits, q[j+4] holds min<<2 in low 6 bits.
                sc_lo_byte = sharpi_byte_at(weights, b0 + 4 + j_lo);
                sc_hi_byte = sharpi_byte_at(weights, b0 + 4 + j_lo + 4);
                sc1 = sc_lo_byte & 63u;
                m1  = sc_hi_byte & 63u;
                // Pair j_hi = j_lo + 1 is also < 4.
                unsigned int sc_lo2 = sharpi_byte_at(weights, b0 + 4 + j_hi);
                unsigned int sc_hi2 = sharpi_byte_at(weights, b0 + 4 + j_hi + 4);
                sc2 = sc_lo2 & 63u;
                m2  = sc_hi2 & 63u;
            } else {
                // Pairs 4..7: scale = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
                //             min   = (q[j+4] >> 4)  | ((q[j]   >> 6) << 4)
                unsigned int a_lo = sharpi_byte_at(weights, b0 + 4 + j_lo + 4); // q[j+4] for j=j_lo
                unsigned int b_lo = sharpi_byte_at(weights, b0 + 4 + j_lo - 4); // q[j-4]
                unsigned int c_lo = sharpi_byte_at(weights, b0 + 4 + j_lo);     // q[j]
                sc1 = (a_lo & 0xFu) | (((b_lo >> 6) & 3u) << 4);
                m1  = ((a_lo >> 4) & 0xFu) | (((c_lo >> 6) & 3u) << 4);

                unsigned int a_hi = sharpi_byte_at(weights, b0 + 4 + j_hi + 4);
                unsigned int b_hi = sharpi_byte_at(weights, b0 + 4 + j_hi - 4);
                unsigned int c_hi = sharpi_byte_at(weights, b0 + 4 + j_hi);
                sc2 = (a_hi & 0xFu) | (((b_hi >> 6) & 3u) << 4);
                m2  = ((a_hi >> 4) & 0xFu) | (((c_hi >> 6) & 3u) << 4);
            }

            float d1  = d * (float)sc1;
            float dm1 = dmin * (float)m1;
            float d2  = d * (float)sc2;
            float dm2 = dmin * (float)m2;

            // qh masks for this chunk's two polarities (u1 = 1<<(2c), u2 = 1<<(2c+1)).
            unsigned int u1 = 1u << (2 * chunk);
            unsigned int u2 = u1 << 1;

            // ql byte for this lane in this chunk: offset (48 + chunk*32 + lane).
            unsigned int ql_byte = sharpi_byte_at(weights, b0 + 48 + chunk * 32 + lane);
            unsigned int low4 = ql_byte & 0xFu;
            unsigned int hi4  = (ql_byte >> 4) & 0xFu;

            int hLo = (qh_byte & u1) != 0u ? 16 : 0;
            int hHi = (qh_byte & u2) != 0u ? 16 : 0;

            int elem_lo = base_elem + chunk * 64 + lane;
            int elem_hi = elem_lo + 32;

            acc += (d1 * (float)((int)low4 + hLo) - dm1) * input[elem_lo];
            acc += (d2 * (float)((int)hi4  + hHi) - dm2) * input[elem_hi];
        }
    }

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output[row] = result;
}

// ── MatVec Q8_0 ────────────────────────────────────────────────────────────
// Q8_0 block (34 bytes per 32 elements): [d:fp16][qs:32 × int8].
// Launch geometry mirrors Q5_K/Q6_K: 8 rows/block × 32 threads/row → warp
// reduce. Each lane (0..31) processes one int8 quant per block, so one full
// pass over 32 lanes covers a whole 32-element Q8_0 block. cols must be a
// multiple of 32; this holds for every projection dim in practice.
//
// Weights are read through uint32 byte-gather helpers so the kernel works
// against the uint32-strided UploadRaw layout used by every other quantized
// matvec in this file. d is decoded inline from two byte gathers to avoid
// any alignment assumptions on the 34-byte block stride.
extern ""C"" __global__ void llm_matvec_q8_0(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 5;                       // cols / 32
    long row_base_bytes = (long)row * (long)num_blocks * 34L;

    float acc = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 34L;

        // d: FP16 at offset 0..1 of the block (two bytes).
        unsigned int dlo = sharpi_byte_at(weights, b0 + 0);
        unsigned int dhi = sharpi_byte_at(weights, b0 + 1);
        float d = sharpi_fp16_to_fp32(dlo | (dhi << 8));

        // One int8 quant per lane; element offset within the block = lane.
        int q = sharpi_int8_at(weights, b0 + 2 + (long)lane);
        float x = input[block * 32 + lane];
        acc += d * (float)q * x;
    }

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output[row] = result;
}

// ── Q4_0 fp32 matvec (Gemma 4 12B QAT primary weights, issue #124) ──────────
// Q4_0 block (18 bytes per 32 elements): [d:fp16][qs:16 × uint8] — two signed
// nibbles per byte. Element j (0..15) = low nibble of qs[j]; element j+16 = high
// nibble. Value = (nibble - 8) * d. Mirrors dequantize_row_q4_0 / the CPU
// DequantQ4_0 path. Same 8-rows/block × 32-threads/row geometry as the Q8_0
// kernel; each lane owns exactly one element of the 32-wide block.
extern ""C"" __global__ void llm_matvec_q4_0(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 5;                       // cols / 32
    long row_base_bytes = (long)row * (long)num_blocks * 18L;

    float acc = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 18L;

        // d: FP16 at offset 0..1 of the block (two bytes).
        unsigned int dlo = sharpi_byte_at(weights, b0 + 0);
        unsigned int dhi = sharpi_byte_at(weights, b0 + 1);
        float d = sharpi_fp16_to_fp32(dlo | (dhi << 8));

        // Two nibbles per byte: lane 0..15 read the low nibble of qs[lane],
        // lanes 16..31 the high nibble of qs[lane-16]. q = nibble - 8.
        unsigned int qbyte = sharpi_byte_at(weights, b0 + 2 + (long)(lane & 15));
        int nib = (lane < 16) ? (int)(qbyte & 0xFu) : (int)(qbyte >> 4);
        int q = nib - 8;

        float x = input[block * 32 + lane];
        acc += d * (float)q * x;
    }

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output[row] = result;
}

// SoA-layout fp32 matvec (issue #149): bit-identical to llm_matvec_q8_0 but reads the
// Q8_0 weight from the SoA buffer [quants rows*cols B][scales rows*nb fp16] — aligned
// quant bytes, one aligned fp16 scale per block, no funnelshift.
extern ""C"" __global__ void llm_matvec_q8_0_soa(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 5;
    long qrow = (long)row * cols;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + (long)rows * cols);
    long srow = (long)row * num_blocks;

    float acc = 0.f;
    for (int block = 0; block < num_blocks; block++) {
        float d = sharpi_fp16_to_fp32(ws[srow + block]);
        int q = sharpi_int8_at(weights, qrow + (long)block * 32 + lane);
        float x = input[block * 32 + lane];
        acc += d * (float)q * x;
    }
    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output[row] = result;
}

// ── MatVec Q8_0 — __dp4a / Q8_1 path (issue #142) ─────────────────────────
// Decode matvec mirroring llama.cpp's mul_mat_vec_q8_0_q8_1. The input vector is
// pre-quantized to Q8_1 (36-byte sub-blocks: fp16 d at [0:2], 32 int8 at [4:36]),
// so each 4-int8 inner product is one __dp4a instruction instead of four
// int8→float converts + fp32 FMAs — far fewer instructions per byte of weight,
// pushing the (already memory-coalesced) Q8_0 matvec closer to the HBM ceiling.
//
// One output row per block; MATVEC_Q80_NWARPS warps cooperate. Within a warp the
// 32 lanes split into 4 groups of 8: group g = lane>>3 owns one Q8_0 block of the
// warp's 4-block stripe; sub-lane s = lane&7 handles int32 word s (4 int8) of that
// block via one __dp4a. The 8 sub-lane partials are summed (shfl_xor over the
// group of 8) and scaled by d_w·d_a, then accumulated across the stripe.
//
// Q8_0 block = 34 bytes (fp16 d + 32 int8); qs is only 2-byte aligned, so the
// 4-int8 weight words are assembled with __funnelshift_r from two aligned uint
// loads (the activation Q8_1 qs at +4 is naturally 4-aligned).
#define MATVEC_Q80_NWARPS 8
extern ""C"" __global__ void llm_matvec_q8_0_dp4a(
    const unsigned int* __restrict__ weights,
    const unsigned char* __restrict__ y_q81,
    float* __restrict__ output,
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int warp_id = (int)threadIdx.y;     // 0..NWARPS-1
    int lane    = (int)threadIdx.x;     // 0..31
    int grp     = lane >> 3;            // 0..3  block within the warp's 4-block stripe
    int sub     = lane & 7;             // 0..7  int32 word within the block

    int num_blocks = cols >> 5;         // cols / 32
    long row_base_bytes = (long)row * (long)num_blocks * 34L;

    float acc = 0.f;

    for (int block0 = warp_id * 4; block0 < num_blocks; block0 += MATVEC_Q80_NWARPS * 4) {
        int block = block0 + grp;
        float part = 0.f;
        if (block < num_blocks) {
            long b0 = row_base_bytes + (long)block * 34L;
            unsigned int dlo = sharpi_byte_at(weights, b0);
            unsigned int dhi = sharpi_byte_at(weights, b0 + 1);
            float dw = sharpi_fp16_to_fp32(dlo | (dhi << 8));

            // This sub-lane's 4 weight int8 = qs[sub*4 .. sub*4+4) at byte b0+2+sub*4.
            long wb        = b0 + 2 + (long)sub * 4;
            long aligned   = wb & ~3L;
            unsigned int shift = (unsigned int)(wb & 3L) * 8u;
            unsigned int w_lo = weights[aligned >> 2];
            int wq;
            if (shift == 0u) wq = (int)w_lo;
            else {
                unsigned int w_hi = weights[(aligned >> 2) + 1];
                wq = (int)__funnelshift_r(w_lo, w_hi, shift);
            }

            // Activation Q8_1 sub-block = block; d at base (low 16), 32 int8 at +4.
            long ab = (long)block * 36L;
            unsigned int d_bits = (*reinterpret_cast<const unsigned int*>(y_q81 + ab)) & 0xffffu;
            float da = sharpi_fp16_to_fp32(d_bits);
            int aq = *reinterpret_cast<const int*>(y_q81 + ab + 4 + (long)sub * 4);

            int dot = __dp4a(wq, aq, 0);
            part = dw * da * (float)dot;
        }
        // Sum the 8 sub-lanes within each aligned group of 8.
        part += __shfl_xor_sync(0xffffffffu, part, 4);
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        if (sub == 0) acc += part;
    }

    // Group leaders (sub==0: lanes 0,8,16,24) hold per-stripe sums; reduce across
    // the 4 groups and all warps via shared memory.
    __shared__ float warp_acc[MATVEC_Q80_NWARPS][4];
    if (sub == 0) warp_acc[warp_id][grp] = acc;
    __syncthreads();
    if (warp_id == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q80_NWARPS; w++)
            for (int g = 0; g < 4; g++) s += warp_acc[w][g];
        output[row] = s;
    }
}

// SoA-layout dp4a decode matvec (issue #149). Same as llm_matvec_q8_0_dp4a but the
// Q8_0 weight is in the SoA buffer [quants rows*cols B][scales rows*nb fp16], so the
// quant words are plain aligned loads (no funnelshift) and the scale is one aligned
// fp16 read. Bit-identical to the AoS dp4a. ws locates the scale region at byte rows*cols.
extern ""C"" __global__ void llm_matvec_q8_0_dp4a_soa(
    const unsigned int* __restrict__ weights,    // SoA: [quants][scales]
    const unsigned char* __restrict__ y_q81,
    float* __restrict__ output,
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int warp_id = (int)threadIdx.y;
    int lane    = (int)threadIdx.x;
    int grp     = lane >> 3;
    int sub     = lane & 7;

    int num_blocks = cols >> 5;
    long qrow = (long)row * (cols >> 2);            // uint index of this row's quants
    const unsigned short* ws = (const unsigned short*)((const char*)weights + (long)rows * cols);
    long srow = (long)row * num_blocks;             // ushort index of this row's scales

    float acc = 0.f;
    for (int block0 = warp_id * 4; block0 < num_blocks; block0 += MATVEC_Q80_NWARPS * 4) {
        int block = block0 + grp;
        float part = 0.f;
        if (block < num_blocks) {
            float dw = sharpi_fp16_to_fp32(ws[srow + block]);
            int wq = (int)weights[qrow + (long)block * 8 + sub];   // aligned, no funnelshift

            long ab = (long)block * 36L;
            unsigned int d_bits = (*reinterpret_cast<const unsigned int*>(y_q81 + ab)) & 0xffffu;
            float da = sharpi_fp16_to_fp32(d_bits);
            int aq = *reinterpret_cast<const int*>(y_q81 + ab + 4 + (long)sub * 4);

            int dot = __dp4a(wq, aq, 0);
            part = dw * da * (float)dot;
        }
        part += __shfl_xor_sync(0xffffffffu, part, 4);
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        if (sub == 0) acc += part;
    }

    __shared__ float warp_acc[MATVEC_Q80_NWARPS][4];
    if (sub == 0) warp_acc[warp_id][grp] = acc;
    __syncthreads();
    if (warp_id == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q80_NWARPS; w++)
            for (int g = 0; g < 4; g++) s += warp_acc[w][g];
        output[row] = s;
    }
}

// ── MatVec Q8_0 — high-MLP mmvq decode (issue #405) ───────────────────────
// A faithful port of llama.cpp's mul_mat_vec_q<Q8_0, ncols_dst=1> for the cold
// single-token (N=1) decode of the Q8_0 trunk. The existing llm_matvec_q8_0_dp4a
// is occupancy- and DRAM-throughput-saturated under ncu's WARM replay, but COLD /
// in-context it only reaches ~215 GB/s (4070 Ti peak ~504): the access pattern
// keeps too few independent weight loads in flight to hide DRAM latency on a cold
// read (each lane assembles ONE int-word via two dependent funnelshift loads, and
// at 256 threads/block the per-block memory-level parallelism is low). llama.cpp's
// mmvq fixes exactly this by:
//   • 128 threads/block (4 warps × 32), ONE output row per block — so each weight
//     load is a fully independent in-flight transaction (no funnelshift dependency);
//   • each thread issuing vdr=2 INDEPENDENT consecutive int-word loads per block
//     (get_int_b2: the AoS qs is 2-byte aligned, so two uint16 reads assemble each
//     int — no cross-word dependency), doubling loads-in-flight per thread;
//   • a grid-stride loop over blocks_per_iter = vdr·nwarps·warp_size/qi = 32 blocks,
//     so all 128 threads have many independent (weight, activation) loads queued —
//     raising MLP enough to saturate cold DRAM;
//   • one __dp4a per int-word against the Q8_1 activation, then a per-warp partial
//     reduce (shared mem across the 4 warps) + a single 32-lane warp_reduce_sum.
// Reads the SAME 34-B/block AoS Q8_0 weight the dp4a kernel reads (no SoA repack),
// and the SAME 36-B/block Q8_1 activation. Argmax-stable, NOT byte-exact (int8 Q8_1
// activation + warp-reduce order) — identical contract to llm_matvec_q8_0_dp4a.
//
// Layout constants (Q8_0): qk=32, qi=8 (int-words/block), vdr=2; activation QK8_1=32.
// Block (32, MATVEC_Q80_MMVQ_NWARPS); grid = rows; tmp_shared = nwarps×warp_size.
#define MATVEC_Q80_MMVQ_NWARPS 4

// Read int-word i32 from a 2-byte-aligned int8 quant region (mirrors get_int_b2).
__device__ __forceinline__ int sharpi_get_int_b2(const unsigned char* __restrict__ p, int i32)
{
    const unsigned short* p16 = reinterpret_cast<const unsigned short*>(p);
    return (int)((unsigned int)p16[2 * i32] | ((unsigned int)p16[2 * i32 + 1] << 16));
}

extern ""C"" __global__ void llm_matvec_q8_0_mmvq(
    const unsigned char* __restrict__ weights,   // AoS: rows × nb × 34 B
    const unsigned char* __restrict__ y_q81,      // Q8_1: nb × 36 B (fp16 d at [0:2], 32 int8 at [4:36])
    float* __restrict__ output,
    int rows, int cols)
{
    const int row = (int)blockIdx.x;              // one output row per block
    if (row >= rows) return;

    const int warp_size = 32;
    const int qi  = 8;                            // int-words per Q8_0 block
    const int vdr = 2;                            // int-words handled per thread per block
    const int tid = warp_size * (int)threadIdx.y + (int)threadIdx.x;   // 0..127
    const int nb  = cols >> 5;                    // Q8_0 blocks per row
    const int blocks_per_iter = vdr * MATVEC_Q80_MMVQ_NWARPS * warp_size / qi;  // = 32

    const long row_base = (long)row * (long)nb * 34L;

    float tmp = 0.f;
    // kbx = block index this thread starts on; kqs = first int-word within the block.
    for (int kbx = tid / (qi / vdr); kbx < nb; kbx += blocks_per_iter)
    {
        const int kqs = vdr * (tid % (qi / vdr));         // 0,2,4,6
        const long b0 = row_base + (long)kbx * 34L;       // block base byte
        const unsigned char* qs = weights + b0 + 2;       // int8 quants (2-byte aligned)
        unsigned int d_bits = (unsigned int)qs[-2] | ((unsigned int)qs[-1] << 8);
        const float dw = sharpi_fp16_to_fp32(d_bits);     // weight block scale

        const long ab = (long)kbx * 36L;                  // activation block base
        unsigned int da_bits = (*reinterpret_cast<const unsigned int*>(y_q81 + ab)) & 0xffffu;
        const float da = sharpi_fp16_to_fp32(da_bits);    // activation block scale
        const int* aq = reinterpret_cast<const int*>(y_q81 + ab + 4);

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < vdr; i++)
            sumi = __dp4a(sharpi_get_int_b2(qs, kqs + i), aq[kqs + i], sumi);
        tmp += dw * da * (float)sumi;
    }

    // Per-warp partials → shared, warp 0 sums across warps then 32-lane warp reduce.
    __shared__ float tmp_shared[MATVEC_Q80_MMVQ_NWARPS][warp_size];
    tmp_shared[threadIdx.y][threadIdx.x] = tmp;
    __syncthreads();
    if (threadIdx.y != 0) return;
    float acc = tmp_shared[0][threadIdx.x];
    #pragma unroll
    for (int w = 1; w < MATVEC_Q80_MMVQ_NWARPS; w++) acc += tmp_shared[w][threadIdx.x];
    acc = sharpi_warp_reduce_sum(acc);
    if (threadIdx.x == 0) output[row] = acc;
}

// ── MatVec Q4_0 — __dp4a / Q8_1 path (issue #124) ─────────────────────────
// Decode matvec mirroring llama.cpp's mul_mat_vec_q4_0_q8_1. The input vector is
// pre-quantized to Q8_1 (36-byte sub-blocks: fp16 d at [0:2], fp16 s=d·Σq at
// [2:4], 32 int8 at [4:36]). Q4_0 weight is symmetric (value = (nibble-8)·d), so
// the asymmetric dp4a trick avoids per-nibble centering: dp4a the RAW nibbles
// (0..15) against the activations, then subtract 8·Σq once per block via the
// stored Q8_1 sum s. Far fewer instructions per byte than the per-element fp32
// matvec (llm_matvec_q4_0), pushing the bandwidth-bound decode toward HBM peak.
//
// One output row per block; MATVEC_Q40_NWARPS warps cooperate. A Q4_0 block is
// 16 qs bytes = 4 uint words; lanes split 8 groups × 4 sub-lanes: group g =
// lane>>2 owns one block of the warp's 8-block stripe, sub-lane s = lane&3 owns
// uint word s. Word s holds 4 low nibbles (elements 4s..4s+3) and 4 high nibbles
// (elements 16+4s..16+4s+3); two __dp4a's cover both halves. The −8·d_w·s
// correction is added once per block (sub==0). qs is only 2-byte aligned, so the
// weight word is assembled with __funnelshift_r from two aligned uint loads; the
// activation int8 at +4+4·s is naturally 4-aligned.
#define MATVEC_Q40_NWARPS 8
extern ""C"" __global__ void llm_matvec_q4_0_dp4a(
    const unsigned int* __restrict__ weights,
    const unsigned char* __restrict__ y_q81,
    float* __restrict__ output,
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int warp_id = (int)threadIdx.y;     // 0..NWARPS-1
    int lane    = (int)threadIdx.x;     // 0..31
    int grp     = lane >> 2;            // 0..7  block within the warp's 8-block stripe
    int sub     = lane & 3;             // 0..3  uint word within the block

    int num_blocks = cols >> 5;         // cols / 32
    long row_base_bytes = (long)row * (long)num_blocks * 18L;

    float acc = 0.f;

    for (int block0 = warp_id * 8; block0 < num_blocks; block0 += MATVEC_Q40_NWARPS * 8) {
        int block = block0 + grp;
        float part = 0.f;
        if (block < num_blocks) {
            long b0 = row_base_bytes + (long)block * 18L;
            unsigned int dlo = sharpi_byte_at(weights, b0);
            unsigned int dhi = sharpi_byte_at(weights, b0 + 1);
            float dw = sharpi_fp16_to_fp32(dlo | (dhi << 8));

            // This sub-lane's 4 weight bytes = qs[sub*4 .. sub*4+4) at byte b0+2+sub*4.
            long wb        = b0 + 2 + (long)sub * 4;
            long aligned   = wb & ~3L;
            unsigned int shift = (unsigned int)(wb & 3L) * 8u;
            unsigned int w_lo = weights[aligned >> 2];
            unsigned int w;
            if (shift == 0u) w = w_lo;
            else {
                unsigned int w_hi = weights[(aligned >> 2) + 1];
                w = __funnelshift_r(w_lo, w_hi, shift);
            }
            int vi0 = (int)(w & 0x0F0F0F0Fu);          // low nibbles  → elements 4s..4s+3
            int vi1 = (int)((w >> 4) & 0x0F0F0F0Fu);   // high nibbles → elements 16+4s..

            long ab = (long)block * 36L;
            unsigned int d_bits = (*reinterpret_cast<const unsigned int*>(y_q81 + ab)) & 0xffffu;
            float da = sharpi_fp16_to_fp32(d_bits);
            // 4 activations for the low half (elements 4s..) and the high half (16+4s..).
            int aq0 = *reinterpret_cast<const int*>(y_q81 + ab + 4 + (long)sub * 4);
            int aq1 = *reinterpret_cast<const int*>(y_q81 + ab + 4 + 16 + (long)sub * 4);

            int dot = __dp4a(vi0, aq0, 0);
            dot = __dp4a(vi1, aq1, dot);
            part = dw * da * (float)dot;

            // −8·Σq·d_x·d_w added once per block (sub==0): Σq·d_x is the stored s.
            if (sub == 0) {
                unsigned int s_bits = (*reinterpret_cast<const unsigned int*>(y_q81 + ab)) >> 16;
                float s = sharpi_fp16_to_fp32(s_bits);
                part -= 8.f * dw * s;
            }
        }
        // Sum the 4 sub-lanes within each aligned group of 4.
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        if (sub == 0) acc += part;
    }

    // Group leaders (sub==0: lanes 0,4,8,…,28) hold per-stripe sums; reduce across
    // the 8 groups and all warps via shared memory.
    __shared__ float warp_acc[MATVEC_Q40_NWARPS][8];
    if (sub == 0) warp_acc[warp_id][grp] = acc;
    __syncthreads();
    if (warp_id == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q40_NWARPS; w++)
            for (int g = 0; g < 8; g++) s += warp_acc[w][g];
        output[row] = s;
    }
}

// One-time repack of an interleaved Q8_0 weight [rows × nb × 34 B] into the SoA
// buffer [quants rows*cols B][scales rows*nb fp16] (issue #149). One thread per
// 32-int8 block: copies the 2-byte fp16 scale to the scale region and the 32 quants
// to the (16-byte-aligned) quant region. Runs once at weight upload.
extern ""C"" __global__ void llm_q8_0_repack_soa(
    const unsigned char* __restrict__ src,   // interleaved, 34 B/block
    unsigned char* __restrict__ dst,         // SoA [quants][scales]
    int rows, int cols)
{
    long blk = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int nb = cols >> 5;
    long total = (long)rows * nb;
    if (blk >= total) return;
    long srcOff = blk * 34L;
    long qDst = blk * 32L;                       // quants region
    long sDst = (long)rows * cols + blk * 2L;    // scales region
    dst[sDst]     = src[srcOff];
    dst[sDst + 1] = src[srcOff + 1];
    #pragma unroll
    for (int i = 0; i < 32; i++) dst[qDst + i] = src[srcOff + 2 + i];
}

// ── Q8_0 × Q8_1 int8 tensor-core MMQ (issue #141 prefill) ──────────────────
// Replaces the dequant→fp16→cuBLAS GEMM round-trip with a direct int8 tensor-core
// multiply: each Q8_0 weight is read once as int8 (no fp16 weight temp written to
// HBM) and fed to the m16n8k32 s8 mma. Activations are pre-quantized to the
// 36-byte/block Q8_1 layout (fp16 d at [0:2], 32 int8 at [4:36]) — the same buffer
// the dp4a decode matvec uses. result[r,t] = Σ_block ( int32(W_blk·Y_blk) · d_w · d_a ).
// Q8_0 is symmetric (scale only, no min), so — like llama.cpp's D4 layout — there
// is NO sum/bias correction term; the activation `s` field is never read.
//
// The mma fragment register layout follows the PTX m16n8k32.row.col spec exactly:
//   groupID = lane>>2 (0..7), tig = lane&3 (0..3)
//   A(16×32 s8): a0=W[grp][tig*4..], a1=W[grp+8][..], a2=W[grp][16+tig*4..], a3=W[grp+8][..]
//   B(32×8  s8): b0=Y[tig*4..][grp], b1=Y[16+tig*4..][grp]    (col=token=grp)
//   C(16×8 s32): c0=[grp][tig*2], c1=[grp][tig*2+1], c2=[grp+8][tig*2], c3=[grp+8][tig*2+1]
//
// Shared-tiled: a 256-thread block computes a 64(row)×128(token) output tile, looping
// K in 32-wide Q8_0 blocks. Each K-block stages the weight + activation sub-tiles into
// shared once; 8 warps (4 row × 2 col) then run 8 m16n8k32 s8 mma's each (one per
// 8-token N-tile), accumulating per-block-scaled products in fp32 registers. Every
// weight row is read once per (row-block, K-block) and reused across all 128 tokens in
// the tile — the weight-read-once property the dequant->fp16->cuBLAS path bought with a
// 2x fp16 HBM temp, here without the temp and on int8 tensor cores (2x fp16 TC peak).
// The wide 128-token tile halves the weight re-read factor (nTok/MMQ_BN) vs a 64-token
// tile, which matters because the Q8_0 qs 2-byte misalignment taxes each weight word.
#define MMQ_BM 64
#define MMQ_BN 128
extern ""C"" __global__ void llm_mmq_q8_0(
    const unsigned int*  __restrict__ weights,   // Q8_0 [rows × cols], 34 B/block
    const unsigned char* __restrict__ y_q81,     // Q8_1 [n_tok × cols], 36 B/block
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    __shared__ int   sW[MMQ_BM * 8];   // 64 weight rows × 8 int32 (32 int8)
    __shared__ float sWd[MMQ_BM];      // 64 weight block-scales
    __shared__ int   sY[MMQ_BN * 8];   // 128 tokens × 8 int32 acts
    __shared__ float sYd[MMQ_BN];      // 128 act block-scales

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb = cols >> 5;

    int tid  = (int)threadIdx.x;       // 0..255
    int warp = tid >> 5;               // 0..7
    int lane = tid & 31;
    int grp  = lane >> 2;              // 0..7
    int tig  = lane & 3;               // 0..3
    int wr   = warp & 3;               // 0..3 row-group → rows [wr*16 : +16]
    int wc   = warp >> 2;              // 0..1 col-group → tokens [wc*64 : +64]
    int mrow0 = wr * 16;

    float acc[8][4];                   // 8 N-tiles × 4 C registers, fp32
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    // Register-prefetch double-buffer: each thread stages 2 weight words (li=tid,
    // tid+256) and 4 act words (li=tid,+256,+512,+768) plus, for tid in range, one
    // weight/act block-scale. The next K-tile's global loads are issued into these
    // registers while the current tile's mma's run, so global latency hides behind
    // compute instead of stalling the per-K-block barrier (cp.async can't be used —
    // the Q8_0 qs is only 2-byte aligned). Macro loads tile `KB` into the rX regs.
    unsigned int rW0, rW1, rY0, rY1, rY2, rY3;
    float rWd, rYd;
    #define MMQ_LOAD_TILE(KB) do { \
        int gw0 = row_block + (tid >> 3); \
        rW0 = (gw0 < rows) ? sharpi_uint_at(weights, ((long)gw0 * nb + (KB)) * 34L + 2 + (long)(tid & 7) * 4) : 0u; \
        int gw1 = row_block + ((tid + 256) >> 3); \
        rW1 = (gw1 < rows) ? sharpi_uint_at(weights, ((long)gw1 * nb + (KB)) * 34L + 2 + (long)((tid + 256) & 7) * 4) : 0u; \
        if (tid < MMQ_BM) { long wb = ((long)(row_block + tid) * nb + (KB)) * 34L; \
            rWd = (row_block + tid < rows) ? sharpi_fp16_to_fp32(sharpi_byte_at(weights, wb) | (sharpi_byte_at(weights, wb + 1) << 8)) : 0.f; } \
        int gy0 = tok_block + (tid >> 3); \
        rY0 = (gy0 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy0 * nb + (KB)) * 36L + 4 + (long)(tid & 7) * 4) : 0u; \
        int gy1 = tok_block + ((tid + 256) >> 3); \
        rY1 = (gy1 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy1 * nb + (KB)) * 36L + 4 + (long)((tid + 256) & 7) * 4) : 0u; \
        int gy2 = tok_block + ((tid + 512) >> 3); \
        rY2 = (gy2 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy2 * nb + (KB)) * 36L + 4 + (long)((tid + 512) & 7) * 4) : 0u; \
        int gy3 = tok_block + ((tid + 768) >> 3); \
        rY3 = (gy3 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy3 * nb + (KB)) * 36L + 4 + (long)((tid + 768) & 7) * 4) : 0u; \
        if (tid < MMQ_BN) { int gt = tok_block + tid; \
            rYd = (gt < n_tok) ? sharpi_fp16_to_fp32((*reinterpret_cast<const unsigned int*>(y_q81 + ((long)gt * nb + (KB)) * 36L)) & 0xffffu) : 0.f; } \
    } while (0)

    MMQ_LOAD_TILE(0);

    for (int kb = 0; kb < nb; kb++) {
        // Publish the prefetched tile to shared.
        sW[tid] = (int)rW0; sW[tid + 256] = (int)rW1;
        if (tid < MMQ_BM) sWd[tid] = rWd;
        sY[tid] = (int)rY0; sY[tid + 256] = (int)rY1; sY[tid + 512] = (int)rY2; sY[tid + 768] = (int)rY3;
        if (tid < MMQ_BN) sYd[tid] = rYd;
        __syncthreads();

        // Issue the next tile's global loads (in flight during the mma's below).
        if (kb + 1 < nb) MMQ_LOAD_TILE(kb + 1);

        // A fragment for this warp's 16-row tile (read once, reused over 8 N-tiles).
        int a0 = sW[(mrow0 + grp) * 8     + tig];
        int a1 = sW[(mrow0 + grp + 8) * 8 + tig];
        int a2 = sW[(mrow0 + grp) * 8     + tig + 4];
        int a3 = sW[(mrow0 + grp + 8) * 8 + tig + 4];
        float dwA = sWd[mrow0 + grp];
        float dwB = sWd[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[(ncol0 + grp) * 8 + tig];
            int b1 = sY[(ncol0 + grp) * 8 + tig + 4];
            float daC0 = sYd[ncol0 + tig * 2];
            float daC1 = sYd[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            acc[nt][0] += (float)c0 * dwA * daC0;
            acc[nt][1] += (float)c1 * dwA * daC1;
            acc[nt][2] += (float)c2 * dwB * daC0;
            acc[nt][3] += (float)c3 * dwB * daC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef MMQ_LOAD_TILE
#undef MMQ_BM
#undef MMQ_BN

// ── SoA-layout int8 MMQ (issue #149) ───────────────────────────────────────
// Same tiling/mma as llm_mmq_q8_0, but the Q8_0 weights are pre-repacked into a
// struct-of-arrays layout: `qw` holds the 32 int8 quants per block contiguous and
// 16-byte aligned (8 uints/block, row-major by (row*nb+kb)), and `ws` holds the
// fp16 block scales separately. This kills the AoS tax: in the interleaved 34-byte
// layout the qs start at a 2-byte offset, so every weight word costs an extra load
// + __funnelshift (sharpi_uint_at); here every weight word is a plain aligned load.
// The roofline probe put the interleaved MMQ at 23-34% of int8 TC peak, inner-loop-
// bound on exactly this. Activations (Q8_1) are already 4-aligned, so only the
// weight-load path changes. Bit-identical to llm_mmq_q8_0 given a faithful repack.
#define MMQ_BM 64
#define MMQ_BN 128
extern ""C"" __global__ void llm_mmq_q8_0_soa(
    const unsigned int*  __restrict__ weights,   // SoA buffer: [quants rows*cols B][scales rows*nb fp16]
    const unsigned char* __restrict__ y_q81,     // Q8_1 [n_tok × cols], 36 B/block
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    // Single SoA buffer; split into the quant and scale views (host stores both
    // contiguous so the kernel signature matches llm_mmq_q8_0 — the dispatch just
    // swaps the kernel function pointer based on whether the weight was repacked).
    const unsigned int*   qw = weights;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + (long)rows * cols);

    __shared__ int   sW[MMQ_BM * 8];
    __shared__ float sWd[MMQ_BM];
    __shared__ int   sY[MMQ_BN * 8];
    __shared__ float sYd[MMQ_BN];

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb = cols >> 5;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp  = lane >> 2;
    int tig  = lane & 3;
    int wr   = warp & 3;
    int wc   = warp >> 2;
    int mrow0 = wr * 16;

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    // Register-prefetch double-buffer, SoA weights: aligned uint loads (no funnelshift).
    unsigned int rW0, rW1, rY0, rY1, rY2, rY3;
    float rWd, rYd;
    #define MMQ_LOAD_TILE_SOA(KB) do { \
        int gw0 = row_block + (tid >> 3); \
        rW0 = (gw0 < rows) ? qw[((long)gw0 * nb + (KB)) * 8L + (tid & 7)] : 0u; \
        int gw1 = row_block + ((tid + 256) >> 3); \
        rW1 = (gw1 < rows) ? qw[((long)gw1 * nb + (KB)) * 8L + ((tid + 256) & 7)] : 0u; \
        if (tid < MMQ_BM) { int gw = row_block + tid; \
            rWd = (gw < rows) ? sharpi_fp16_to_fp32(ws[(long)gw * nb + (KB)]) : 0.f; } \
        int gy0 = tok_block + (tid >> 3); \
        rY0 = (gy0 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy0 * nb + (KB)) * 36L + 4 + (long)(tid & 7) * 4) : 0u; \
        int gy1 = tok_block + ((tid + 256) >> 3); \
        rY1 = (gy1 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy1 * nb + (KB)) * 36L + 4 + (long)((tid + 256) & 7) * 4) : 0u; \
        int gy2 = tok_block + ((tid + 512) >> 3); \
        rY2 = (gy2 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy2 * nb + (KB)) * 36L + 4 + (long)((tid + 512) & 7) * 4) : 0u; \
        int gy3 = tok_block + ((tid + 768) >> 3); \
        rY3 = (gy3 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy3 * nb + (KB)) * 36L + 4 + (long)((tid + 768) & 7) * 4) : 0u; \
        if (tid < MMQ_BN) { int gt = tok_block + tid; \
            rYd = (gt < n_tok) ? sharpi_fp16_to_fp32((*reinterpret_cast<const unsigned int*>(y_q81 + ((long)gt * nb + (KB)) * 36L)) & 0xffffu) : 0.f; } \
    } while (0)

    MMQ_LOAD_TILE_SOA(0);

    for (int kb = 0; kb < nb; kb++) {
        sW[tid] = (int)rW0; sW[tid + 256] = (int)rW1;
        if (tid < MMQ_BM) sWd[tid] = rWd;
        sY[tid] = (int)rY0; sY[tid + 256] = (int)rY1; sY[tid + 512] = (int)rY2; sY[tid + 768] = (int)rY3;
        if (tid < MMQ_BN) sYd[tid] = rYd;
        __syncthreads();

        if (kb + 1 < nb) MMQ_LOAD_TILE_SOA(kb + 1);

        int a0 = sW[(mrow0 + grp) * 8     + tig];
        int a1 = sW[(mrow0 + grp + 8) * 8 + tig];
        int a2 = sW[(mrow0 + grp) * 8     + tig + 4];
        int a3 = sW[(mrow0 + grp + 8) * 8 + tig + 4];
        float dwA = sWd[mrow0 + grp];
        float dwB = sWd[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[(ncol0 + grp) * 8 + tig];
            int b1 = sY[(ncol0 + grp) * 8 + tig + 4];
            float daC0 = sYd[ncol0 + tig * 2];
            float daC1 = sYd[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            acc[nt][0] += (float)c0 * dwA * daC0;
            acc[nt][1] += (float)c1 * dwA * daC1;
            acc[nt][2] += (float)c2 * dwB * daC0;
            acc[nt][3] += (float)c3 * dwB * daC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef MMQ_LOAD_TILE_SOA
#undef MMQ_BM
#undef MMQ_BN

// ── SoA weights + SoA activations int8 MMQ (Track A, #124/#173) ─────────────
// Phase-A substrate: bit-identical to llm_mmq_q8_0_soa, but the activations are read
// from the struct-of-arrays Q8_1 layout (llm_quantize_q8_1_soa) — contiguous int8 qs
// (32 B/block, no header) + a separate {d,s} uint32 array — instead of the interleaved
// 36-B AoS block. SAME load mapping (4 tokens × 8 uints per warp, one K-block), SAME
// fragment map and accumulation order, so maxAbs==0 vs the AoS-activation kernel. This
// only changes the activation *address arithmetic*; it is the layout Phase B's
// coalesced per-token multi-K-block load is built on. nb == cols>>5 is the per-token
// activation block count (== the weight block count for Q8_0).
#define MMQ_BM 64
#define MMQ_BN 128
// __launch_bounds__(256, 4): ncu (2026-06-08) showed the SoA activation layout cut
// uncoalesced global sectors 55%→43% (a token's quants are now contiguous) — but at fixed
// 49% occupancy (3 blocks, register-bound at 80 regs) the kernel is latency-stalled (0.69
// eligible warps/cyc), so the coalescing win didn't show. The lever is OCCUPANCY: this
// variant DROPS the register-prefetch double-buffer (loads each K-block directly into
// shared in-loop) so registers fall enough for __launch_bounds__(256, 4) to pack 4
// resident blocks (≤64 regs → 67% occupancy). More active warps hide the global-load
// latency that the prefetch used to; the coalesced SoA load keeps the traffic low. Same
// int8 mma + SAME K-order accumulation → bit-identical to llm_mmq_q8_0_soa.
extern ""C"" __global__ void __launch_bounds__(256, 4) llm_mmq_q8_0_soa_acts(
    const unsigned int*  __restrict__ weights,   // SoA buffer: [quants rows*cols B][scales rows*nb fp16]
    const unsigned int*  __restrict__ y_qs,      // SoA Q8_1 quants [n_tok × nb × 32 int8], contiguous
    const unsigned int*  __restrict__ y_ds,      // SoA Q8_1 scales [n_tok × nb] {d,s} uint32
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    const unsigned int*   qw = weights;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + (long)rows * cols);

    __shared__ int   sW[MMQ_BM * 8];
    __shared__ float sWd[MMQ_BM];
    __shared__ int   sY[MMQ_BN * 8];
    __shared__ float sYd[MMQ_BN];

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb = cols >> 5;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp  = lane >> 2;
    int tig  = lane & 3;
    int wr   = warp & 3;
    int wc   = warp >> 2;
    int mrow0 = wr * 16;

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    for (int kb = 0; kb < nb; kb++) {
        // No-prefetch in-loop staging (frees the rW/rY prefetch regs for occupancy).
        int gw0 = row_block + (tid >> 3);
        sW[tid] = (gw0 < rows) ? (int)qw[((long)gw0 * nb + kb) * 8L + (tid & 7)] : 0;
        int gw1 = row_block + ((tid + 256) >> 3);
        sW[tid + 256] = (gw1 < rows) ? (int)qw[((long)gw1 * nb + kb) * 8L + ((tid + 256) & 7)] : 0;
        if (tid < MMQ_BM) { int gw = row_block + tid;
            sWd[tid] = (gw < rows) ? sharpi_fp16_to_fp32(ws[(long)gw * nb + kb]) : 0.f; }
        int gy0 = tok_block + (tid >> 3);
        sY[tid] = (gy0 < n_tok) ? (int)y_qs[((long)gy0 * nb + kb) * 8L + (tid & 7)] : 0;
        int gy1 = tok_block + ((tid + 256) >> 3);
        sY[tid + 256] = (gy1 < n_tok) ? (int)y_qs[((long)gy1 * nb + kb) * 8L + ((tid + 256) & 7)] : 0;
        int gy2 = tok_block + ((tid + 512) >> 3);
        sY[tid + 512] = (gy2 < n_tok) ? (int)y_qs[((long)gy2 * nb + kb) * 8L + ((tid + 512) & 7)] : 0;
        int gy3 = tok_block + ((tid + 768) >> 3);
        sY[tid + 768] = (gy3 < n_tok) ? (int)y_qs[((long)gy3 * nb + kb) * 8L + ((tid + 768) & 7)] : 0;
        if (tid < MMQ_BN) { int gt = tok_block + tid;
            sYd[tid] = (gt < n_tok) ? sharpi_fp16_to_fp32(y_ds[(long)gt * nb + kb] & 0xffffu) : 0.f; }
        __syncthreads();

        int a0 = sW[(mrow0 + grp) * 8     + tig];
        int a1 = sW[(mrow0 + grp + 8) * 8 + tig];
        int a2 = sW[(mrow0 + grp) * 8     + tig + 4];
        int a3 = sW[(mrow0 + grp + 8) * 8 + tig + 4];
        float dwA = sWd[mrow0 + grp];
        float dwB = sWd[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[(ncol0 + grp) * 8 + tig];
            int b1 = sY[(ncol0 + grp) * 8 + tig + 4];
            float daC0 = sYd[ncol0 + tig * 2];
            float daC1 = sYd[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            acc[nt][0] += (float)c0 * dwA * daC0;
            acc[nt][1] += (float)c1 * dwA * daC1;
            acc[nt][2] += (float)c2 * dwB * daC0;
            acc[nt][3] += (float)c3 * dwB * daC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef MMQ_LOAD_TILE_SOA_ACTS
#undef MMQ_BM
#undef MMQ_BN

// ── cp.async 16-B helpers (Track B) — arch-guarded so the module still compiles on
//    pre-Ampere (sm_75 Turing, where the int8 mma works but cp.async does not). On
//    sm_80+ they issue real async copies; on older arch they fall back to a synchronous
//    16-B scalar shared store (the commit/wait become no-ops), keeping the kernel correct
//    everywhere — just without the async-pipeline speedup. Bit-identical either way.
__device__ __forceinline__ void sharpi_cp_async16(int* smem, const void* gmem)
{
#if __CUDA_ARCH__ >= 800
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem);
    asm volatile(""cp.async.cg.shared.global [%0], [%1], 16;"" :: ""r""(s), ""l""(gmem));
#else
    const int* g = (const int*)gmem;
    smem[0] = g[0]; smem[1] = g[1]; smem[2] = g[2]; smem[3] = g[3];
#endif
}
__device__ __forceinline__ void sharpi_cp_commit()
{
#if __CUDA_ARCH__ >= 800
    asm volatile(""cp.async.commit_group;"");
#endif
}
__device__ __forceinline__ void sharpi_cp_wait_keep1()
{
#if __CUDA_ARCH__ >= 800
    asm volatile(""cp.async.wait_group 1;"");
#endif
}
__device__ __forceinline__ void sharpi_cp_wait_all()
{
#if __CUDA_ARCH__ >= 800
    asm volatile(""cp.async.wait_group 0;"");
#endif
}

// ── Q8_0 SoA-acts MMQ with cp.async pipelined global→shared (Track B port) ──
// The lever ncu identified: the kernel is L1TEX-bound (78.6%), and occupancy/coalescing
// were both proven not to move the 1.14 ms floor. cp.async copies global→shared WITHOUT
// going through the LSU/register path (L2→shared direct), taking the bulk weight+act
// streaming OFF the L1TEX pipe — and double-buffering overlaps the next K-tile's copy with
// the current tile's mma in hardware (no register-prefetch cost → 4 blocks fit). This is
// what llama.cpp's mul_mat_q actually uses for int8 (NOT ldmatrix). It is ONLY possible on
// Phase A's SoA activation layout: the contiguous 16-B-aligned qs are cp.async-eligible,
// where the AoS 36-B block's 2-byte-misaligned qs were not. Same int8 values into the same
// shared tiles + SAME K-order accumulation → bit-identical to llm_mmq_q8_0_soa.
// Requires sm_80+ (Ada sm_89 here). 16-B cp.async.cg per chunk; the small per-block scales
// stay on the scalar path (a negligible fraction of the traffic).
#define MMQ_BM 64
#define MMQ_BN 128
extern ""C"" __global__ void __launch_bounds__(256, 4) llm_mmq_q8_0_soa_acts_cpa(
    const unsigned int*  __restrict__ weights,   // SoA: [quants rows*cols B][scales rows*nb fp16]
    const unsigned int*  __restrict__ y_qs,      // SoA Q8_1 quants [n_tok × nb × 32 int8], contiguous
    const unsigned int*  __restrict__ y_ds,      // SoA Q8_1 scales [n_tok × nb] {d,s} uint32
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    const unsigned int*   qw = weights;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + (long)rows * cols);

    __shared__ int   sW[2][MMQ_BM * 8];   // double-buffered weight quants
    __shared__ int   sY[2][MMQ_BN * 8];   // double-buffered act quants
    __shared__ float sWd[MMQ_BM];         // per-K-block scales (scalar path, single-buffered)
    __shared__ float sYd[MMQ_BN];

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb = cols >> 5;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp  = lane >> 2;
    int tig  = lane & 3;
    int wr   = warp & 3;
    int wc   = warp >> 2;
    int mrow0 = wr * 16;

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    // Issue this block's weight+act 16-B chunks into shared stage `st` via cp.async.
    //   weights: 64 rows × 2 chunks = 128 chunks → threads 0..127, chunk c=tid: row=c>>1, half=c&1.
    //   acts:   128 tokens × 2 chunks = 256 chunks → threads 0..255, chunk c=tid: token=c>>1, half=c&1.
    // Out-of-range rows/tokens are zero-filled scalar (cp.async can't read invalid global).
    #define CPA_ISSUE(KB, ST) do { \
        if (tid < MMQ_BM * 2) { int row = tid >> 1, half = tid & 1; int gw = row_block + row; \
            int* dstp = &sW[ST][row * 8 + half * 4]; \
            if (gw < rows) sharpi_cp_async16(dstp, &qw[((long)gw * nb + (KB)) * 8L + half * 4]); \
            else { dstp[0] = dstp[1] = dstp[2] = dstp[3] = 0; } } \
        { int token = tid >> 1, half = tid & 1; int gy = tok_block + token; \
            int* dstp = &sY[ST][token * 8 + half * 4]; \
            if (gy < n_tok) sharpi_cp_async16(dstp, &y_qs[((long)gy * nb + (KB)) * 8L + half * 4]); \
            else { dstp[0] = dstp[1] = dstp[2] = dstp[3] = 0; } } \
    } while (0)

    CPA_ISSUE(0, 0);
    sharpi_cp_commit();

    for (int kb = 0; kb < nb; kb++) {
        int cur = kb & 1;
        if (kb + 1 < nb) {
            CPA_ISSUE(kb + 1, (kb + 1) & 1);
            sharpi_cp_commit();
            sharpi_cp_wait_keep1();   // keep the just-issued next tile in flight
        } else {
            sharpi_cp_wait_all();
        }
        // Per-block scales for the CURRENT kb (scalar path — tiny vs the quant stream).
        if (tid < MMQ_BM) { int gw = row_block + tid;
            sWd[tid] = (gw < rows) ? sharpi_fp16_to_fp32(ws[(long)gw * nb + kb]) : 0.f; }
        if (tid < MMQ_BN) { int gt = tok_block + tid;
            sYd[tid] = (gt < n_tok) ? sharpi_fp16_to_fp32(y_ds[(long)gt * nb + kb] & 0xffffu) : 0.f; }
        __syncthreads();

        int a0 = sW[cur][(mrow0 + grp) * 8     + tig];
        int a1 = sW[cur][(mrow0 + grp + 8) * 8 + tig];
        int a2 = sW[cur][(mrow0 + grp) * 8     + tig + 4];
        int a3 = sW[cur][(mrow0 + grp + 8) * 8 + tig + 4];
        float dwA = sWd[mrow0 + grp];
        float dwB = sWd[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[cur][(ncol0 + grp) * 8 + tig];
            int b1 = sY[cur][(ncol0 + grp) * 8 + tig + 4];
            float daC0 = sYd[ncol0 + tig * 2];
            float daC1 = sYd[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            acc[nt][0] += (float)c0 * dwA * daC0;
            acc[nt][1] += (float)c1 * dwA * daC1;
            acc[nt][2] += (float)c2 * dwB * daC0;
            acc[nt][3] += (float)c3 * dwB * daC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef CPA_ISSUE
#undef MMQ_BM
#undef MMQ_BN

// ── Q4_0 SoA-acts MMQ with cp.async pipelined global→shared (Track B port) ──
// Q4_0 analogue of llm_mmq_q8_0_soa_acts_cpa. The SoA Q4_0 weight block is 4 uints (16 B,
// one cp.async chunk) holding 32 packed nibbles; we cp.async the RAW nibbles into shared
// (half the staging of the expanded form) and nibble-expand in the mma fragment read
// (low nibbles → weight-words 0..3, high nibbles → 4..7, the same split llm_mmq_q4_0_soa
// does at load). Activations are the same SoA Q8_1 as Q8_0. Q4_0 symmetric → the
// −8·d_w·(d_a·Σq_a) centering term per block (the act `s` field). Bit-identical to
// llm_mmq_q4_0_soa_acts (same nibbles/scales/accumulation order). Requires sm_80+.
#define MMQ_BM 64
#define MMQ_BN 128
extern ""C"" __global__ void __launch_bounds__(256, 4) llm_mmq_q4_0_soa_acts_cpa(
    const unsigned int*  __restrict__ weights,   // SoA: [quants rows*cols/2 B][scales rows*nb fp16]
    const unsigned int*  __restrict__ y_qs,      // SoA Q8_1 quants [n_tok × nb × 32 int8]
    const unsigned int*  __restrict__ y_ds,      // SoA Q8_1 scales [n_tok × nb] {d,s} uint32
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    const unsigned int*   qw = weights;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + ((long)rows * cols) / 2);

    __shared__ int   sWraw[2][MMQ_BM * 4];   // raw Q4_0 nibbles, 4 uints/block (16 B)
    __shared__ int   sY[2][MMQ_BN * 8];      // act quants
    __shared__ float sWd[MMQ_BM];
    __shared__ float sYd[MMQ_BN];
    __shared__ float sYs[MMQ_BN];

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb = cols >> 5;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp  = lane >> 2;
    int tig  = lane & 3;
    int wr   = warp & 3;
    int wc   = warp >> 2;
    int mrow0 = wr * 16;

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    // weights: 64 rows × 1 16-B chunk → threads 0..63 (row=tid).
    // acts:    128 tokens × 2 16-B chunks → threads 0..255 (token=tid>>1, half=tid&1).
    #define CPA_ISSUE_Q40(KB, ST) do { \
        if (tid < MMQ_BM) { int gw = row_block + tid; \
            int* dstp = &sWraw[ST][tid * 4]; \
            if (gw < rows) sharpi_cp_async16(dstp, &qw[((long)gw * nb + (KB)) * 4L]); \
            else { dstp[0] = dstp[1] = dstp[2] = dstp[3] = 0; } } \
        { int token = tid >> 1, half = tid & 1; int gy = tok_block + token; \
            int* dstp = &sY[ST][token * 8 + half * 4]; \
            if (gy < n_tok) sharpi_cp_async16(dstp, &y_qs[((long)gy * nb + (KB)) * 8L + half * 4]); \
            else { dstp[0] = dstp[1] = dstp[2] = dstp[3] = 0; } } \
    } while (0)

    CPA_ISSUE_Q40(0, 0);
    sharpi_cp_commit();

    for (int kb = 0; kb < nb; kb++) {
        int cur = kb & 1;
        if (kb + 1 < nb) {
            CPA_ISSUE_Q40(kb + 1, (kb + 1) & 1);
            sharpi_cp_commit();
            sharpi_cp_wait_keep1();
        } else {
            sharpi_cp_wait_all();
        }
        if (tid < MMQ_BM) { int gw = row_block + tid;
            sWd[tid] = (gw < rows) ? sharpi_fp16_to_fp32(ws[(long)gw * nb + kb]) : 0.f; }
        if (tid < MMQ_BN) { int gt = tok_block + tid;
            if (gt < n_tok) { unsigned int dw = y_ds[(long)gt * nb + kb];
                sYd[tid] = sharpi_fp16_to_fp32(dw & 0xffffu); sYs[tid] = sharpi_fp16_to_fp32(dw >> 16); }
            else { sYd[tid] = 0.f; sYs[tid] = 0.f; } }
        __syncthreads();

        // Nibble-expand the A fragment from raw shared: a0/a1 = low nibbles (weight-words
        // tig), a2/a3 = high nibbles (weight-words tig+4) — matches llm_mmq_q4_0_soa's split.
        unsigned int wlo0 = (unsigned int)sWraw[cur][(mrow0 + grp) * 4     + tig];
        unsigned int wlo1 = (unsigned int)sWraw[cur][(mrow0 + grp + 8) * 4 + tig];
        int a0 = (int)(wlo0 & 0x0F0F0F0Fu);
        int a1 = (int)(wlo1 & 0x0F0F0F0Fu);
        int a2 = (int)((wlo0 >> 4) & 0x0F0F0F0Fu);
        int a3 = (int)((wlo1 >> 4) & 0x0F0F0F0Fu);
        float dwA = sWd[mrow0 + grp];
        float dwB = sWd[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[cur][(ncol0 + grp) * 8 + tig];
            int b1 = sY[cur][(ncol0 + grp) * 8 + tig + 4];
            float dC0 = sYd[ncol0 + tig * 2], dC1 = sYd[ncol0 + tig * 2 + 1];
            float sC0 = sYs[ncol0 + tig * 2], sC1 = sYs[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            acc[nt][0] += (float)c0 * dwA * dC0 - 8.f * dwA * sC0;
            acc[nt][1] += (float)c1 * dwA * dC1 - 8.f * dwA * sC1;
            acc[nt][2] += (float)c2 * dwB * dC0 - 8.f * dwB * sC0;
            acc[nt][3] += (float)c3 * dwB * dC1 - 8.f * dwB * sC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef CPA_ISSUE_Q40
#undef MMQ_BM
#undef MMQ_BN

// get_scale_min_k4 (ggml-quants.c) for one Q4_K sub-block `sb` (0..7). The 12-byte
// scales[] array lives in the three uints sm0/sm1/sm2 (block bytes [4:16]); each
// sub-block has a 6-bit scale `sc` and 6-bit min `mn`. Matches the lo/hi switch in
// llm_matvec_q4k_gemm_n (chunk = sb>>1, polarity = sb&1).
__device__ __forceinline__ void sharpi_q4k_scale_min(
    unsigned int sm0, unsigned int sm1, unsigned int sm2, int sb,
    unsigned int* sc, unsigned int* mn)
{
    int chunk = sb >> 1;
    if (sb & 1) {  // odd sub-block (high half of the chunk)
        switch (chunk) {
            case 0:  *sc = (sm0 >>  8) & 63u; *mn = (sm1 >>  8) & 63u; break;
            case 1:  *sc = (sm0 >> 24) & 63u; *mn = (sm1 >> 24) & 63u; break;
            case 2:  *sc = ((sm2 >>  8) & 0xFu) | (((sm0 >> 14) & 3u) << 4);
                     *mn = ((sm2 >> 12) & 0xFu) | (((sm1 >> 14) & 3u) << 4); break;
            default: *sc = ((sm2 >> 24) & 0xFu) | (((sm0 >> 30) & 3u) << 4);
                     *mn = ((sm2 >> 28) & 0xFu) | (((sm1 >> 30) & 3u) << 4); break;
        }
    } else {       // even sub-block (low half of the chunk)
        switch (chunk) {
            case 0:  *sc = (sm0)       & 63u; *mn = (sm1)       & 63u; break;
            case 1:  *sc = (sm0 >> 16) & 63u; *mn = (sm1 >> 16) & 63u; break;
            case 2:  *sc = (sm2        & 0xFu) | (((sm0 >>  6) & 3u) << 4);
                     *mn = ((sm2 >>  4) & 0xFu) | (((sm1 >>  6) & 3u) << 4); break;
            default: *sc = ((sm2 >> 16) & 0xFu) | (((sm0 >> 22) & 3u) << 4);
                     *mn = ((sm2 >> 20) & 0xFu) | (((sm1 >> 22) & 3u) << 4); break;
        }
    }
}

// ── int8 MMQ Q4_K (issue #156 Item C2) ─────────────────────────────────────
// Maximal Item C: reads each Q4_K weight once as int8 (no fp16 dequant temp to HBM —
// the cost that capped the C1 dequant→fp16→cuBLAS GEMM) and feeds the m16n8k32 s8
// mma, mirroring llm_mmq_q8_0's tiling/fragment map exactly. The only Q4_K-specific
// work vs Q8_0: (a) the 4-bit weight nibbles are expanded to int8 (values 0..15) on
// load via the same lo/hi split as llm_matvec_q4k_gemm_n, (b) the per-(row,sub-block)
// (scale,min) are unpacked with get_scale_min_k4, and (c) Q4_K is asymmetric
// (w = super_d·sc·q − super_dmin·mn), so each sub-block adds a min-bias term
// −super_dmin·mn·(d_a·Σq_a). The activation sum d_a·Σq_a is the fp16 `s` half that
// llm_quantize_q8_1 packs at bytes [2:4] of each q8_1 block.
//
// One K-tile = one 32-element Q4_K sub-block (8 per 256-elem super-block). result
//   [r,t] = Σ_sb ( super_d·sc[r,sb]·d_a[t,sb]·⟨q_w[r,sb],q_a[t,sb]⟩
//                  − super_dmin·mn[r,sb]·(d_a·Σq_a)[t,sb] ).
// The mma fragment register layout is byte-identical to llm_mmq_q8_0 (validated by
// #141): only the int8 values staged into sW and the per-row/-token scale
// coefficients differ. Argmax-stable, not bit-exact (both operands int8-quantized).
#define MMQ_BM 64
#define MMQ_BN 128
extern ""C"" __global__ void llm_mmq_q4k(
    const unsigned int*  __restrict__ weights,   // Q4_K [rows × cols], 144 B/super-block
    const unsigned char* __restrict__ y_q81,     // Q8_1 [n_tok × cols], 36 B/block (d at [0:2], s at [2:4])
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    __shared__ int   sW[MMQ_BM * 8];   // 64 weight rows × 8 int32 (32 nibbles→int8)
    __shared__ float sWdd[MMQ_BM];     // 64 × super_d·sc (per row, this sub-block)
    __shared__ float sWdm[MMQ_BM];     // 64 × super_dmin·mn (per row, this sub-block)
    __shared__ int   sY[MMQ_BN * 8];   // 128 tokens × 8 int32 acts
    __shared__ float sYd[MMQ_BN];      // 128 × activation block-scale d_a
    __shared__ float sYs[MMQ_BN];      // 128 × activation sum d_a·Σq_a

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb_super  = cols >> 8;          // super-blocks per row
    int sub_total = cols >> 5;          // 32-element sub-blocks per row (= K-tiles)

    int tid  = (int)threadIdx.x;        // 0..255
    int warp = tid >> 5;                // 0..7
    int lane = tid & 31;
    int grp  = lane >> 2;               // 0..7
    int tig  = lane & 3;                // 0..3
    int wr   = warp & 3;                // 0..3 row-group → rows [wr*16 : +16]
    int wc   = warp >> 2;               // 0..1 col-group → tokens [wc*64 : +64]
    int mrow0 = wr * 16;

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    // Register-prefetch double-buffer, mirroring llm_mmq_q8_0: each thread stages 2
    // weight words (nibble-expanded) and 4 act words, plus — for the in-range threads —
    // one weight (super_d·sc, super_dmin·mn) coefficient pair and one act (d_a, s) pair.
    unsigned int rWq0, rWq1, rY0, rY1, rY2, rY3;
    float rWdd, rWdm, rYd, rYs;
    #define MMQ_LOAD_TILE_Q4K(KB) do { \
        int kbs = (KB) >> 3; int sb = (KB) & 7; int chk = sb >> 1; int pol = sb & 1; \
        int gw0 = row_block + (tid >> 3); \
        { unsigned int w = (gw0 < rows) ? weights[((long)gw0 * nb_super + kbs) * 36L + 4 + chk * 8 + (tid & 7)] : 0u; \
          rWq0 = pol ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu); } \
        int gw1 = row_block + ((tid + 256) >> 3); \
        { unsigned int w = (gw1 < rows) ? weights[((long)gw1 * nb_super + kbs) * 36L + 4 + chk * 8 + ((tid + 256) & 7)] : 0u; \
          rWq1 = pol ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu); } \
        if (tid < MMQ_BM) { int gw = row_block + tid; \
          if (gw < rows) { long hb = (long)(gw * nb_super + kbs) * 36L; \
            unsigned int w0 = weights[hb]; \
            unsigned int sm0 = weights[hb + 1], sm1 = weights[hb + 2], sm2 = weights[hb + 3]; \
            unsigned int sc, mn; sharpi_q4k_scale_min(sm0, sm1, sm2, sb, &sc, &mn); \
            rWdd = sharpi_fp16_to_fp32(w0 & 0xffffu) * (float)sc; \
            rWdm = sharpi_fp16_to_fp32(w0 >> 16)     * (float)mn; \
          } else { rWdd = 0.f; rWdm = 0.f; } } \
        int gy0 = tok_block + (tid >> 3); \
        rY0 = (gy0 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy0 * sub_total + (KB)) * 36L + 4 + (long)(tid & 7) * 4) : 0u; \
        int gy1 = tok_block + ((tid + 256) >> 3); \
        rY1 = (gy1 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy1 * sub_total + (KB)) * 36L + 4 + (long)((tid + 256) & 7) * 4) : 0u; \
        int gy2 = tok_block + ((tid + 512) >> 3); \
        rY2 = (gy2 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy2 * sub_total + (KB)) * 36L + 4 + (long)((tid + 512) & 7) * 4) : 0u; \
        int gy3 = tok_block + ((tid + 768) >> 3); \
        rY3 = (gy3 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy3 * sub_total + (KB)) * 36L + 4 + (long)((tid + 768) & 7) * 4) : 0u; \
        if (tid < MMQ_BN) { int gt = tok_block + tid; \
          if (gt < n_tok) { unsigned int dw = *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gt * sub_total + (KB)) * 36L); \
            rYd = sharpi_fp16_to_fp32(dw & 0xffffu); rYs = sharpi_fp16_to_fp32(dw >> 16); } \
          else { rYd = 0.f; rYs = 0.f; } } \
    } while (0)

    MMQ_LOAD_TILE_Q4K(0);

    for (int kb = 0; kb < sub_total; kb++) {
        // Publish the prefetched sub-block tile to shared.
        sW[tid] = (int)rWq0; sW[tid + 256] = (int)rWq1;
        if (tid < MMQ_BM) { sWdd[tid] = rWdd; sWdm[tid] = rWdm; }
        sY[tid] = (int)rY0; sY[tid + 256] = (int)rY1; sY[tid + 512] = (int)rY2; sY[tid + 768] = (int)rY3;
        if (tid < MMQ_BN) { sYd[tid] = rYd; sYs[tid] = rYs; }
        __syncthreads();

        // Issue the next sub-block's global loads (in flight during the mma's below).
        if (kb + 1 < sub_total) MMQ_LOAD_TILE_Q4K(kb + 1);

        // A fragment for this warp's 16-row tile (read once, reused over 8 N-tiles).
        int a0 = sW[(mrow0 + grp) * 8     + tig];
        int a1 = sW[(mrow0 + grp + 8) * 8 + tig];
        int a2 = sW[(mrow0 + grp) * 8     + tig + 4];
        int a3 = sW[(mrow0 + grp + 8) * 8 + tig + 4];
        float ddA = sWdd[mrow0 + grp],     dmA = sWdm[mrow0 + grp];
        float ddB = sWdd[mrow0 + grp + 8], dmB = sWdm[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[(ncol0 + grp) * 8 + tig];
            int b1 = sY[(ncol0 + grp) * 8 + tig + 4];
            float dC0 = sYd[ncol0 + tig * 2], dC1 = sYd[ncol0 + tig * 2 + 1];
            float sC0 = sYs[ncol0 + tig * 2], sC1 = sYs[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            // dot term super_d·sc·d_a·⟨q_w,q_a⟩ minus the asymmetric min-bias
            // super_dmin·mn·(d_a·Σq_a); both accumulate over all sub-blocks.
            acc[nt][0] += (float)c0 * ddA * dC0 - dmA * sC0;
            acc[nt][1] += (float)c1 * ddA * dC1 - dmA * sC1;
            acc[nt][2] += (float)c2 * ddB * dC0 - dmB * sC0;
            acc[nt][3] += (float)c3 * ddB * dC1 - dmB * sC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef MMQ_LOAD_TILE_Q4K
#undef MMQ_BM
#undef MMQ_BN

// ── int8 MMQ Q4_K over the scale-pre-unpacked SoA weight (issue #156) ───────
// Identical mma tiling / fragment map / accumulation to llm_mmq_q4k — only the three
// weight loads change to read the [Q quants][S sc|mn bytes][D d|dmin] regions instead
// of the interleaved 144-B super-block. The staged int8 values and scale coefficients
// are bit-identical, so this is bit-identical to llm_mmq_q4k (which is itself only
// argmax-stable vs fp, both operands int8-quantized). Lets a Q4_K weight be repacked
// to SoA ONCE at upload and serve BOTH the decode matvec and the prefill GEMM.
#define MMQ_BM 64
#define MMQ_BN 128
extern ""C"" __global__ void llm_mmq_q4k_soa(
    const unsigned int*  __restrict__ weights,   // SoA [Q][S][D]
    const unsigned char* __restrict__ y_q81,     // Q8_1 [n_tok × cols], 36 B/block
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    __shared__ int   sW[MMQ_BM * 8];
    __shared__ float sWdd[MMQ_BM];
    __shared__ float sWdm[MMQ_BM];
    __shared__ int   sY[MMQ_BN * 8];
    __shared__ float sYd[MMQ_BN];
    __shared__ float sYs[MMQ_BN];

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb_super  = cols >> 8;
    int sub_total = cols >> 5;
    long total_sb = (long)rows * nb_super;          // super-blocks in the whole tensor

    // SoA region bases.
    const unsigned int*  qReg = weights;                                       // [total_sb*32 uint]
    const unsigned char* sReg = (const unsigned char*)weights + total_sb * 128L; // [total_sb*16 B]
    const unsigned int*  dReg = (const unsigned int*)(sReg + total_sb * 16L);    // [total_sb uint]

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp  = lane >> 2;
    int tig  = lane & 3;
    int wr   = warp & 3;
    int wc   = warp >> 2;
    int mrow0 = wr * 16;

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    unsigned int rWq0, rWq1, rY0, rY1, rY2, rY3;
    float rWdd, rWdm, rYd, rYs;
    #define MMQ_LOAD_TILE_Q4K_SOA(KB) do { \
        int kbs = (KB) >> 3; int sb = (KB) & 7; int chk = sb >> 1; int pol = sb & 1; \
        int gw0 = row_block + (tid >> 3); \
        { unsigned int w = (gw0 < rows) ? qReg[((long)gw0 * nb_super + kbs) * 32L + chk * 8 + (tid & 7)] : 0u; \
          rWq0 = pol ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu); } \
        int gw1 = row_block + ((tid + 256) >> 3); \
        { unsigned int w = (gw1 < rows) ? qReg[((long)gw1 * nb_super + kbs) * 32L + chk * 8 + ((tid + 256) & 7)] : 0u; \
          rWq1 = pol ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu); } \
        if (tid < MMQ_BM) { int gw = row_block + tid; \
          if (gw < rows) { long sbi = (long)gw * nb_super + kbs; \
            unsigned int dd = dReg[sbi]; \
            unsigned int sc = sReg[sbi * 16L + sb]; \
            unsigned int mn = sReg[sbi * 16L + 8 + sb]; \
            rWdd = sharpi_fp16_to_fp32(dd & 0xffffu) * (float)sc; \
            rWdm = sharpi_fp16_to_fp32(dd >> 16)     * (float)mn; \
          } else { rWdd = 0.f; rWdm = 0.f; } } \
        int gy0 = tok_block + (tid >> 3); \
        rY0 = (gy0 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy0 * sub_total + (KB)) * 36L + 4 + (long)(tid & 7) * 4) : 0u; \
        int gy1 = tok_block + ((tid + 256) >> 3); \
        rY1 = (gy1 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy1 * sub_total + (KB)) * 36L + 4 + (long)((tid + 256) & 7) * 4) : 0u; \
        int gy2 = tok_block + ((tid + 512) >> 3); \
        rY2 = (gy2 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy2 * sub_total + (KB)) * 36L + 4 + (long)((tid + 512) & 7) * 4) : 0u; \
        int gy3 = tok_block + ((tid + 768) >> 3); \
        rY3 = (gy3 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy3 * sub_total + (KB)) * 36L + 4 + (long)((tid + 768) & 7) * 4) : 0u; \
        if (tid < MMQ_BN) { int gt = tok_block + tid; \
          if (gt < n_tok) { unsigned int dw = *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gt * sub_total + (KB)) * 36L); \
            rYd = sharpi_fp16_to_fp32(dw & 0xffffu); rYs = sharpi_fp16_to_fp32(dw >> 16); } \
          else { rYd = 0.f; rYs = 0.f; } } \
    } while (0)

    MMQ_LOAD_TILE_Q4K_SOA(0);

    for (int kb = 0; kb < sub_total; kb++) {
        sW[tid] = (int)rWq0; sW[tid + 256] = (int)rWq1;
        if (tid < MMQ_BM) { sWdd[tid] = rWdd; sWdm[tid] = rWdm; }
        sY[tid] = (int)rY0; sY[tid + 256] = (int)rY1; sY[tid + 512] = (int)rY2; sY[tid + 768] = (int)rY3;
        if (tid < MMQ_BN) { sYd[tid] = rYd; sYs[tid] = rYs; }
        __syncthreads();

        if (kb + 1 < sub_total) MMQ_LOAD_TILE_Q4K_SOA(kb + 1);

        int a0 = sW[(mrow0 + grp) * 8     + tig];
        int a1 = sW[(mrow0 + grp + 8) * 8 + tig];
        int a2 = sW[(mrow0 + grp) * 8     + tig + 4];
        int a3 = sW[(mrow0 + grp + 8) * 8 + tig + 4];
        float ddA = sWdd[mrow0 + grp],     dmA = sWdm[mrow0 + grp];
        float ddB = sWdd[mrow0 + grp + 8], dmB = sWdm[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[(ncol0 + grp) * 8 + tig];
            int b1 = sY[(ncol0 + grp) * 8 + tig + 4];
            float dC0 = sYd[ncol0 + tig * 2], dC1 = sYd[ncol0 + tig * 2 + 1];
            float sC0 = sYs[ncol0 + tig * 2], sC1 = sYs[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            acc[nt][0] += (float)c0 * ddA * dC0 - dmA * sC0;
            acc[nt][1] += (float)c1 * ddA * dC1 - dmA * sC1;
            acc[nt][2] += (float)c2 * ddB * dC0 - dmB * sC0;
            acc[nt][3] += (float)c3 * ddB * dC1 - dmB * sC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef MMQ_LOAD_TILE_Q4K_SOA
#undef MMQ_BM
#undef MMQ_BN

// ── Q4_K SoA weights + SoA activations int8 MMQ (Track A, #124/#173) ────────
// Bit-identical to llm_mmq_q4k_soa; only the activation reads switch to the SoA
// Q8_1 layout (contiguous qs + separate {d,s}). SAME load mapping / fragment map /
// accumulation order. sub_total == cols>>5 is the per-token activation block count.
#define MMQ_BM 64
#define MMQ_BN 128
extern ""C"" __global__ void llm_mmq_q4k_soa_acts(
    const unsigned int*  __restrict__ weights,   // SoA [Q][S][D]
    const unsigned int*  __restrict__ y_qs,      // SoA Q8_1 quants [n_tok × sub_total × 32 int8]
    const unsigned int*  __restrict__ y_ds,      // SoA Q8_1 scales [n_tok × sub_total] {d,s} uint32
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    __shared__ int   sW[MMQ_BM * 8];
    __shared__ float sWdd[MMQ_BM];
    __shared__ float sWdm[MMQ_BM];
    __shared__ int   sY[MMQ_BN * 8];
    __shared__ float sYd[MMQ_BN];
    __shared__ float sYs[MMQ_BN];

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb_super  = cols >> 8;
    int sub_total = cols >> 5;
    long total_sb = (long)rows * nb_super;

    const unsigned int*  qReg = weights;
    const unsigned char* sReg = (const unsigned char*)weights + total_sb * 128L;
    const unsigned int*  dReg = (const unsigned int*)(sReg + total_sb * 16L);

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp  = lane >> 2;
    int tig  = lane & 3;
    int wr   = warp & 3;
    int wc   = warp >> 2;
    int mrow0 = wr * 16;

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    unsigned int rWq0, rWq1, rY0, rY1, rY2, rY3;
    float rWdd, rWdm, rYd, rYs;
    #define MMQ_LOAD_TILE_Q4K_SOA_ACTS(KB) do { \
        int kbs = (KB) >> 3; int sb = (KB) & 7; int chk = sb >> 1; int pol = sb & 1; \
        int gw0 = row_block + (tid >> 3); \
        { unsigned int w = (gw0 < rows) ? qReg[((long)gw0 * nb_super + kbs) * 32L + chk * 8 + (tid & 7)] : 0u; \
          rWq0 = pol ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu); } \
        int gw1 = row_block + ((tid + 256) >> 3); \
        { unsigned int w = (gw1 < rows) ? qReg[((long)gw1 * nb_super + kbs) * 32L + chk * 8 + ((tid + 256) & 7)] : 0u; \
          rWq1 = pol ? ((w >> 4) & 0x0F0F0F0Fu) : (w & 0x0F0F0F0Fu); } \
        if (tid < MMQ_BM) { int gw = row_block + tid; \
          if (gw < rows) { long sbi = (long)gw * nb_super + kbs; \
            unsigned int dd = dReg[sbi]; \
            unsigned int sc = sReg[sbi * 16L + sb]; \
            unsigned int mn = sReg[sbi * 16L + 8 + sb]; \
            rWdd = sharpi_fp16_to_fp32(dd & 0xffffu) * (float)sc; \
            rWdm = sharpi_fp16_to_fp32(dd >> 16)     * (float)mn; \
          } else { rWdd = 0.f; rWdm = 0.f; } } \
        int gy0 = tok_block + (tid >> 3); \
        rY0 = (gy0 < n_tok) ? y_qs[((long)gy0 * sub_total + (KB)) * 8L + (tid & 7)] : 0u; \
        int gy1 = tok_block + ((tid + 256) >> 3); \
        rY1 = (gy1 < n_tok) ? y_qs[((long)gy1 * sub_total + (KB)) * 8L + ((tid + 256) & 7)] : 0u; \
        int gy2 = tok_block + ((tid + 512) >> 3); \
        rY2 = (gy2 < n_tok) ? y_qs[((long)gy2 * sub_total + (KB)) * 8L + ((tid + 512) & 7)] : 0u; \
        int gy3 = tok_block + ((tid + 768) >> 3); \
        rY3 = (gy3 < n_tok) ? y_qs[((long)gy3 * sub_total + (KB)) * 8L + ((tid + 768) & 7)] : 0u; \
        if (tid < MMQ_BN) { int gt = tok_block + tid; \
          if (gt < n_tok) { unsigned int dw = y_ds[(long)gt * sub_total + (KB)]; \
            rYd = sharpi_fp16_to_fp32(dw & 0xffffu); rYs = sharpi_fp16_to_fp32(dw >> 16); } \
          else { rYd = 0.f; rYs = 0.f; } } \
    } while (0)

    MMQ_LOAD_TILE_Q4K_SOA_ACTS(0);

    for (int kb = 0; kb < sub_total; kb++) {
        sW[tid] = (int)rWq0; sW[tid + 256] = (int)rWq1;
        if (tid < MMQ_BM) { sWdd[tid] = rWdd; sWdm[tid] = rWdm; }
        sY[tid] = (int)rY0; sY[tid + 256] = (int)rY1; sY[tid + 512] = (int)rY2; sY[tid + 768] = (int)rY3;
        if (tid < MMQ_BN) { sYd[tid] = rYd; sYs[tid] = rYs; }
        __syncthreads();

        if (kb + 1 < sub_total) MMQ_LOAD_TILE_Q4K_SOA_ACTS(kb + 1);

        int a0 = sW[(mrow0 + grp) * 8     + tig];
        int a1 = sW[(mrow0 + grp + 8) * 8 + tig];
        int a2 = sW[(mrow0 + grp) * 8     + tig + 4];
        int a3 = sW[(mrow0 + grp + 8) * 8 + tig + 4];
        float ddA = sWdd[mrow0 + grp],     dmA = sWdm[mrow0 + grp];
        float ddB = sWdd[mrow0 + grp + 8], dmB = sWdm[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[(ncol0 + grp) * 8 + tig];
            int b1 = sY[(ncol0 + grp) * 8 + tig + 4];
            float dC0 = sYd[ncol0 + tig * 2], dC1 = sYd[ncol0 + tig * 2 + 1];
            float sC0 = sYs[ncol0 + tig * 2], sC1 = sYs[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            acc[nt][0] += (float)c0 * ddA * dC0 - dmA * sC0;
            acc[nt][1] += (float)c1 * ddA * dC1 - dmA * sC1;
            acc[nt][2] += (float)c2 * ddB * dC0 - dmB * sC0;
            acc[nt][3] += (float)c3 * ddB * dC1 - dmB * sC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef MMQ_LOAD_TILE_Q4K_SOA_ACTS
#undef MMQ_BM
#undef MMQ_BN

// ── int8 MMQ Q4_0 × Q8_1 (issue #124/#173 prefill) ─────────────────────────
// Q4_0 analogue of llm_mmq_q8_0 / llm_mmq_q4k: reads each Q4_0 weight once as int8
// (nibble-expanded to the raw 0..15 value, NO fp16 dequant temp written to HBM —
// the cost that capped the dequant→fp16→cuBLAS GEMM) and feeds the m16n8k32 s8 mma.
// Q4_0 is symmetric (w = d·(q−8)), so — exactly like the dp4a decode matvec
// (llm_matvec_q4_0_dp4a) — the int8 dot runs on the RAW nibbles (0..15) and the
// −8·d_w·(d_a·Σq_a) centering term is added once per block. Σq_a·d_a is the fp16 `s`
// field llm_quantize_q8_1 packs at bytes [2:4] of each Q8_1 block. There is NO
// per-sub-block (scale,min) unpack (unlike Q4_K): one fp16 block scale d_w and the
// constant offset 8.
//   result[r,t] = Σ_block ( d_w·d_a·⟨q_w_raw, q_a⟩ − 8·d_w·(d_a·Σq_a) ).
// The mma fragment register layout / tiling is byte-identical to llm_mmq_q8_0
// (validated by #141); the activation staging is identical to llm_mmq_q8_0 (Q8_1
// acts), and the per-block bias mirrors llm_mmq_q4k's −super_dmin·mn·(d_a·Σq_a) with
// (super_dmin·mn) collapsed to the constant 8·d_w. The Q4_0 block is 18 B
// ([d:fp16][qs:16]); qs starts at a 2-byte offset so the weight word is assembled via
// sharpi_uint_at (funnelshift). Argmax-stable, not bit-exact (both operands int8).
//
// Weight word ww (0..7) holds K-elements 4·ww..4·ww+3: ww<4 → low nibbles of qs-word
// ww, ww≥4 → high nibbles of qs-word (ww−4) — the same low/high split the dp4a matvec
// uses (low nibbles are elements 0..15, high nibbles 16..31). The Q8_1 activation word
// ww holds act elements 4·ww..4·ww+3, so weight element k multiplies activation k.
#define MMQ_BM 64
#define MMQ_BN 128
extern ""C"" __global__ void llm_mmq_q4_0(
    const unsigned int*  __restrict__ weights,   // Q4_0 [rows × cols], 18 B/block
    const unsigned char* __restrict__ y_q81,     // Q8_1 [n_tok × cols], 36 B/block (d at [0:2], s at [2:4])
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    __shared__ int   sW[MMQ_BM * 8];   // 64 weight rows × 8 int32 (32 nibbles→int8)
    __shared__ float sWd[MMQ_BM];      // 64 × d_w (per row, this block)
    __shared__ int   sY[MMQ_BN * 8];   // 128 tokens × 8 int32 acts
    __shared__ float sYd[MMQ_BN];      // 128 × activation block-scale d_a
    __shared__ float sYs[MMQ_BN];      // 128 × activation sum d_a·Σq_a

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb = cols >> 5;                 // 32-element blocks per row = K-tiles

    int tid  = (int)threadIdx.x;        // 0..255
    int warp = tid >> 5;                // 0..7
    int lane = tid & 31;
    int grp  = lane >> 2;               // 0..7
    int tig  = lane & 3;                // 0..3
    int wr   = warp & 3;                // 0..3 row-group → rows [wr*16 : +16]
    int wc   = warp >> 2;               // 0..1 col-group → tokens [wc*64 : +64]
    int mrow0 = wr * 16;

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    // Register-prefetch double-buffer, mirroring llm_mmq_q8_0: each thread stages 2
    // weight words (nibble-expanded) and 4 act words, plus — for the in-range threads —
    // one weight scale d_w and one act (d_a, s) pair.
    unsigned int rWq0, rWq1, rY0, rY1, rY2, rY3;
    float rWd, rYd, rYs;
    #define MMQ_LOAD_TILE_Q40(KB) do { \
        int ww = tid & 7; int qword = ww & 3; \
        int gw0 = row_block + (tid >> 3); \
        { unsigned int w = (gw0 < rows) ? sharpi_uint_at(weights, ((long)gw0 * nb + (KB)) * 18L + 2 + (long)qword * 4) : 0u; \
          rWq0 = (ww < 4) ? (w & 0x0F0F0F0Fu) : ((w >> 4) & 0x0F0F0F0Fu); } \
        int gw1 = row_block + ((tid + 256) >> 3); \
        { unsigned int w = (gw1 < rows) ? sharpi_uint_at(weights, ((long)gw1 * nb + (KB)) * 18L + 2 + (long)qword * 4) : 0u; \
          rWq1 = (ww < 4) ? (w & 0x0F0F0F0Fu) : ((w >> 4) & 0x0F0F0F0Fu); } \
        if (tid < MMQ_BM) { int gw = row_block + tid; \
          if (gw < rows) { long bb = ((long)gw * nb + (KB)) * 18L; \
            rWd = sharpi_fp16_to_fp32(sharpi_byte_at(weights, bb) | (sharpi_byte_at(weights, bb + 1) << 8)); } \
          else rWd = 0.f; } \
        int gy0 = tok_block + (tid >> 3); \
        rY0 = (gy0 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy0 * nb + (KB)) * 36L + 4 + (long)(tid & 7) * 4) : 0u; \
        int gy1 = tok_block + ((tid + 256) >> 3); \
        rY1 = (gy1 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy1 * nb + (KB)) * 36L + 4 + (long)((tid + 256) & 7) * 4) : 0u; \
        int gy2 = tok_block + ((tid + 512) >> 3); \
        rY2 = (gy2 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy2 * nb + (KB)) * 36L + 4 + (long)((tid + 512) & 7) * 4) : 0u; \
        int gy3 = tok_block + ((tid + 768) >> 3); \
        rY3 = (gy3 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy3 * nb + (KB)) * 36L + 4 + (long)((tid + 768) & 7) * 4) : 0u; \
        if (tid < MMQ_BN) { int gt = tok_block + tid; \
          if (gt < n_tok) { unsigned int dw = *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gt * nb + (KB)) * 36L); \
            rYd = sharpi_fp16_to_fp32(dw & 0xffffu); rYs = sharpi_fp16_to_fp32(dw >> 16); } \
          else { rYd = 0.f; rYs = 0.f; } } \
    } while (0)

    MMQ_LOAD_TILE_Q40(0);

    for (int kb = 0; kb < nb; kb++) {
        // Publish the prefetched block tile to shared.
        sW[tid] = (int)rWq0; sW[tid + 256] = (int)rWq1;
        if (tid < MMQ_BM) sWd[tid] = rWd;
        sY[tid] = (int)rY0; sY[tid + 256] = (int)rY1; sY[tid + 512] = (int)rY2; sY[tid + 768] = (int)rY3;
        if (tid < MMQ_BN) { sYd[tid] = rYd; sYs[tid] = rYs; }
        __syncthreads();

        // Issue the next block's global loads (in flight during the mma's below).
        if (kb + 1 < nb) MMQ_LOAD_TILE_Q40(kb + 1);

        // A fragment for this warp's 16-row tile (read once, reused over 8 N-tiles).
        int a0 = sW[(mrow0 + grp) * 8     + tig];
        int a1 = sW[(mrow0 + grp + 8) * 8 + tig];
        int a2 = sW[(mrow0 + grp) * 8     + tig + 4];
        int a3 = sW[(mrow0 + grp + 8) * 8 + tig + 4];
        float dwA = sWd[mrow0 + grp];
        float dwB = sWd[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[(ncol0 + grp) * 8 + tig];
            int b1 = sY[(ncol0 + grp) * 8 + tig + 4];
            float dC0 = sYd[ncol0 + tig * 2], dC1 = sYd[ncol0 + tig * 2 + 1];
            float sC0 = sYs[ncol0 + tig * 2], sC1 = sYs[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            // dot term d_w·d_a·⟨q_w_raw,q_a⟩ minus the symmetric centering
            // 8·d_w·(d_a·Σq_a); both accumulate over all blocks.
            acc[nt][0] += (float)c0 * dwA * dC0 - 8.f * dwA * sC0;
            acc[nt][1] += (float)c1 * dwA * dC1 - 8.f * dwA * sC1;
            acc[nt][2] += (float)c2 * dwB * dC0 - 8.f * dwB * sC0;
            acc[nt][3] += (float)c3 * dwB * dC1 - 8.f * dwB * sC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef MMQ_LOAD_TILE_Q40
#undef MMQ_BM
#undef MMQ_BN

// Q8_0 GEMM-N: N tokens through one weight matrix. grid=( (rows+7)/8, n_tok ).
// Input [n_tok, cols] and output [n_tok, rows] are offset by token; the per-row
// accumulation + warp reduce is identical to llm_matvec_q8_0, so this is
// bit-identical to n_tok sequential matvecs. (Weights are re-read per token —
// the launch-count collapse is the win; weight-reuse tiling is a follow-up.)
extern ""C"" __global__ void llm_matvec_q8_0_gemm_n(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input,   // [n_tok, cols]
    float* __restrict__ output,        // [n_tok, rows]
    int rows, int cols, int n_tok)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    int token = (int)blockIdx.y;
    if (row >= rows || token >= n_tok) return;

    int num_blocks = cols >> 5;
    long row_base_bytes = (long)row * (long)num_blocks * 34L;
    const float* in = input + (long)token * (long)cols;

    float acc = 0.f;
    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 34L;
        unsigned int dlo = sharpi_byte_at(weights, b0 + 0);
        unsigned int dhi = sharpi_byte_at(weights, b0 + 1);
        float d = sharpi_fp16_to_fp32(dlo | (dhi << 8));
        int q = sharpi_int8_at(weights, b0 + 2 + (long)lane);
        float x = in[block * 32 + lane];
        acc += d * (float)q * x;
    }
    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output[(long)token * (long)rows + row] = result;
}

// SoA-layout GEMM-N (issue #149): bit-identical to llm_matvec_q8_0_gemm_n, aligned
// SoA weight reads.
extern ""C"" __global__ void llm_matvec_q8_0_gemm_n_soa(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols, int n_tok)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    int token = (int)blockIdx.y;
    if (row >= rows || token >= n_tok) return;

    int num_blocks = cols >> 5;
    long qrow = (long)row * cols;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + (long)rows * cols);
    long srow = (long)row * num_blocks;
    const float* in = input + (long)token * (long)cols;

    float acc = 0.f;
    for (int block = 0; block < num_blocks; block++) {
        float d = sharpi_fp16_to_fp32(ws[srow + block]);
        int q = sharpi_int8_at(weights, qrow + (long)block * 32 + lane);
        float x = in[block * 32 + lane];
        acc += d * (float)q * x;
    }
    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output[(long)token * (long)rows + row] = result;
}

// ── Q8_0 → FP16 dequant for cuBLAS prefill GEMM (issue #141) ───────────────
// Dequantizes a Q8_0-packed weight matrix [rows × cols] into a row-major fp16
// matrix [rows × cols]. One block per row (256 threads), each thread strides
// over the row's columns. Element (row, c): super-block b = c >> 5, lane =
// c & 31; the per-block fp16 scale d lives at byte (row*nb + b)*34, the int8
// quant at +2+lane. Stored value d*q is rounded to fp16 — this is the only
// lossy step vs the fp32 matvec (which keeps d*q*x in fp32), and it's what lets
// the prefill GEMM read each weight once per batch instead of once per token.
extern ""C"" __global__ void llm_dequant_q8_0_to_f16(
    const unsigned int* __restrict__ weights,
    unsigned short* __restrict__ out,    // [rows * cols] fp16
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int num_blocks = cols >> 5;
    long row_base_bytes = (long)row * (long)num_blocks * 34L;
    long out_row = (long)row * (long)cols;
    for (int c = (int)threadIdx.x; c < cols; c += (int)blockDim.x) {
        int block = c >> 5;
        int lane  = c & 31;
        long b0 = row_base_bytes + (long)block * 34L;
        unsigned int dlo = sharpi_byte_at(weights, b0 + 0);
        unsigned int dhi = sharpi_byte_at(weights, b0 + 1);
        float d = sharpi_fp16_to_fp32(dlo | (dhi << 8));
        int q = sharpi_int8_at(weights, b0 + 2 + (long)lane);
        out[out_row + c] = (unsigned short)sharpi_fp32_to_fp16(d * (float)q);
    }
}

// SoA-layout dequant (issue #149): bit-identical to llm_dequant_q8_0_to_f16, aligned
// SoA weight reads.
extern ""C"" __global__ void llm_dequant_q8_0_to_f16_soa(
    const unsigned int* __restrict__ weights,
    unsigned short* __restrict__ out,
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int num_blocks = cols >> 5;
    long qrow = (long)row * cols;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + (long)rows * cols);
    long srow = (long)row * num_blocks;
    long out_row = (long)row * (long)cols;
    for (int c = (int)threadIdx.x; c < cols; c += (int)blockDim.x) {
        int block = c >> 5;
        int lane  = c & 31;
        float d = sharpi_fp16_to_fp32(ws[srow + block]);
        int q = sharpi_int8_at(weights, qrow + (long)block * 32 + lane);
        out[out_row + c] = (unsigned short)sharpi_fp32_to_fp16(d * (float)q);
    }
}

// ── Q4_0 → FP16 dequant for cuBLAS prefill GEMM (issue #124) ───────────────
// Dequantizes a Q4_0-packed weight matrix [rows × cols] into a row-major fp16
// matrix [rows × cols]. One block per row (256 threads), each thread strides over
// the row's columns. Element (row, c): block b = c >> 5, within = c & 31; the
// block layout is [d:fp16][qs:16 × uint8] (18 bytes). within<16 reads the low
// nibble of qs[within], within>=16 the high nibble of qs[within-16]; value =
// (nibble - 8) * d. Mirrors llm_matvec_q4_0; d*q rounded to fp16 is the only lossy
// step, letting the prefill GEMM read each weight once per batch (vs per token).
extern ""C"" __global__ void llm_dequant_q4_0_to_f16(
    const unsigned int* __restrict__ weights,
    unsigned short* __restrict__ out,    // [rows * cols] fp16
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int num_blocks = cols >> 5;
    long row_base_bytes = (long)row * (long)num_blocks * 18L;
    long out_row = (long)row * (long)cols;
    for (int c = (int)threadIdx.x; c < cols; c += (int)blockDim.x) {
        int block = c >> 5;
        int within = c & 31;
        long b0 = row_base_bytes + (long)block * 18L;
        unsigned int dlo = sharpi_byte_at(weights, b0 + 0);
        unsigned int dhi = sharpi_byte_at(weights, b0 + 1);
        float d = sharpi_fp16_to_fp32(dlo | (dhi << 8));
        unsigned int qbyte = sharpi_byte_at(weights, b0 + 2 + (long)(within & 15));
        int nib = (within < 16) ? (int)(qbyte & 0xFu) : (int)(qbyte >> 4);
        int q = nib - 8;
        out[out_row + c] = (unsigned short)sharpi_fp32_to_fp16(d * (float)q);
    }
}

// ── Q4_0 SoA repack + SoA readers (issue #124/#173, mirrors #149) ───────────
// The Q4_0 block is 18 B ([d:fp16][qs:16]) and 18 is not a multiple of 4, so half
// the blocks' qs start at a 2-byte offset — every weight uint costs a funnelshift
// (sharpi_uint_at) in the MMQ / dp4a readers, which the roofline puts at ~52 TFLOPS
// (~32% of fp16 TC peak), the same class as the pre-#149 Q8_0 ceiling. A one-time
// upload repack into struct-of-arrays — [quants rows*cols/2 B (16 B/block, 16-byte
// aligned)][scales rows*nb fp16] — lets every reader use plain aligned uint loads.
// Same total bytes as AoS (16+2 = 18). Quant byte values and scales are bit-identical
// to the interleaved layout, so each SoA reader is bit-identical to its AoS twin
// (and the SoA MMQ is bit-identical to llm_mmq_q4_0, itself argmax-stable vs fp).

// One thread per 32-element block: copy the 16 qs bytes to the (16-byte-aligned)
// quants region and the 2-byte fp16 scale to the scales region.
extern ""C"" __global__ void llm_q4_0_repack_soa(
    const unsigned char* __restrict__ src,   // interleaved, 18 B/block
    unsigned char* __restrict__ dst,         // SoA [quants rows*cols/2 B][scales rows*nb fp16]
    int rows, int cols)
{
    long blk = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int nb = cols >> 5;
    long total = (long)rows * nb;
    if (blk >= total) return;
    long srcOff = blk * 18L;
    long qDst = blk * 16L;                            // quants region (16 B/block)
    long sDst = ((long)rows * cols) / 2 + blk * 2L;   // scales region (fp16/block)
    dst[sDst]     = src[srcOff];
    dst[sDst + 1] = src[srcOff + 1];
    #pragma unroll
    for (int i = 0; i < 16; i++) dst[qDst + i] = src[srcOff + 2 + i];
}

// SoA fp32 matvec — bit-identical to llm_matvec_q4_0 (same byte value / fp16 d /
// per-lane accumulation order), reading the SoA layout instead of the 18-B block.
extern ""C"" __global__ void llm_matvec_q4_0_soa(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 5;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + ((long)rows * cols) / 2);
    long srow = (long)row * num_blocks;

    float acc = 0.f;
    for (int block = 0; block < num_blocks; block++) {
        long blk_idx = srow + block;
        float d = sharpi_fp16_to_fp32(ws[blk_idx]);
        unsigned int qbyte = sharpi_byte_at(weights, blk_idx * 16L + (long)(lane & 15));
        int nib = (lane < 16) ? (int)(qbyte & 0xFu) : (int)(qbyte >> 4);
        int q = nib - 8;
        float x = input[block * 32 + lane];
        acc += d * (float)q * x;
    }
    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output[row] = result;
}

// SoA dp4a decode matvec — bit-identical to llm_matvec_q4_0_dp4a (same nibble words /
// fp16 d / −8·d_w·s correction / 8-warp reduction), aligned weight uint loads.
extern ""C"" __global__ void llm_matvec_q4_0_dp4a_soa(
    const unsigned int* __restrict__ weights,    // SoA [quants][scales]
    const unsigned char* __restrict__ y_q81,
    float* __restrict__ output,
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int warp_id = (int)threadIdx.y;
    int lane    = (int)threadIdx.x;
    int grp     = lane >> 2;
    int sub     = lane & 3;

    int num_blocks = cols >> 5;
    long qrow = (long)row * num_blocks;             // block-index base for this row
    const unsigned short* ws = (const unsigned short*)((const char*)weights + ((long)rows * cols) / 2);

    float acc = 0.f;
    for (int block0 = warp_id * 8; block0 < num_blocks; block0 += MATVEC_Q40_NWARPS * 8) {
        int block = block0 + grp;
        float part = 0.f;
        if (block < num_blocks) {
            long blk_idx = qrow + block;
            float dw = sharpi_fp16_to_fp32(ws[blk_idx]);
            unsigned int w = weights[blk_idx * 4L + sub];      // aligned, no funnelshift
            int vi0 = (int)(w & 0x0F0F0F0Fu);
            int vi1 = (int)((w >> 4) & 0x0F0F0F0Fu);

            long ab = (long)block * 36L;
            unsigned int d_bits = (*reinterpret_cast<const unsigned int*>(y_q81 + ab)) & 0xffffu;
            float da = sharpi_fp16_to_fp32(d_bits);
            int aq0 = *reinterpret_cast<const int*>(y_q81 + ab + 4 + (long)sub * 4);
            int aq1 = *reinterpret_cast<const int*>(y_q81 + ab + 4 + 16 + (long)sub * 4);

            int dot = __dp4a(vi0, aq0, 0);
            dot = __dp4a(vi1, aq1, dot);
            part = dw * da * (float)dot;

            if (sub == 0) {
                unsigned int s_bits = (*reinterpret_cast<const unsigned int*>(y_q81 + ab)) >> 16;
                float s = sharpi_fp16_to_fp32(s_bits);
                part -= 8.f * dw * s;
            }
        }
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        if (sub == 0) acc += part;
    }

    __shared__ float warp_acc[MATVEC_Q40_NWARPS][8];
    if (sub == 0) warp_acc[warp_id][grp] = acc;
    __syncthreads();
    if (warp_id == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q40_NWARPS; w++)
            for (int g = 0; g < 8; g++) s += warp_acc[w][g];
        output[row] = s;
    }
}

// SoA Q4_0 → fp16 dequant (cuBLAS prefill GEMM fallback, SHARPI_PREFILL_MMQ=0) —
// bit-identical to llm_dequant_q4_0_to_f16, reading the SoA layout.
extern ""C"" __global__ void llm_dequant_q4_0_to_f16_soa(
    const unsigned int* __restrict__ weights,   // SoA [quants][scales]
    unsigned short* __restrict__ out,           // [rows * cols] fp16
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int num_blocks = cols >> 5;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + ((long)rows * cols) / 2);
    long srow = (long)row * num_blocks;
    long out_row = (long)row * (long)cols;
    for (int c = (int)threadIdx.x; c < cols; c += (int)blockDim.x) {
        int block = c >> 5;
        int within = c & 31;
        long blk_idx = srow + block;
        float d = sharpi_fp16_to_fp32(ws[blk_idx]);
        unsigned int qbyte = sharpi_byte_at(weights, blk_idx * 16L + (long)(within & 15));
        int nib = (within < 16) ? (int)(qbyte & 0xFu) : (int)(qbyte >> 4);
        int q = nib - 8;
        out[out_row + c] = (unsigned short)sharpi_fp32_to_fp16(d * (float)q);
    }
}

// SoA int8 MMQ — bit-identical to llm_mmq_q4_0 (same int8 nibbles / fp16 scales /
// 8-warp mma accumulation), aligned weight uint loads instead of sharpi_uint_at.
#define MMQ_BM 64
#define MMQ_BN 128
extern ""C"" __global__ void llm_mmq_q4_0_soa(
    const unsigned int*  __restrict__ weights,   // SoA [quants rows*cols/2 B][scales rows*nb fp16]
    const unsigned char* __restrict__ y_q81,     // Q8_1 [n_tok × cols], 36 B/block
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    const unsigned int*   qw = weights;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + ((long)rows * cols) / 2);

    __shared__ int   sW[MMQ_BM * 8];
    __shared__ float sWd[MMQ_BM];
    __shared__ int   sY[MMQ_BN * 8];
    __shared__ float sYd[MMQ_BN];
    __shared__ float sYs[MMQ_BN];

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb = cols >> 5;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp  = lane >> 2;
    int tig  = lane & 3;
    int wr   = warp & 3;
    int wc   = warp >> 2;
    int mrow0 = wr * 16;

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    unsigned int rWq0, rWq1, rY0, rY1, rY2, rY3;
    float rWd, rYd, rYs;
    #define MMQ_LOAD_TILE_Q40_SOA(KB) do { \
        int ww = tid & 7; int qword = ww & 3; \
        int gw0 = row_block + (tid >> 3); \
        { unsigned int w = (gw0 < rows) ? qw[((long)gw0 * nb + (KB)) * 4L + qword] : 0u; \
          rWq0 = (ww < 4) ? (w & 0x0F0F0F0Fu) : ((w >> 4) & 0x0F0F0F0Fu); } \
        int gw1 = row_block + ((tid + 256) >> 3); \
        { unsigned int w = (gw1 < rows) ? qw[((long)gw1 * nb + (KB)) * 4L + qword] : 0u; \
          rWq1 = (ww < 4) ? (w & 0x0F0F0F0Fu) : ((w >> 4) & 0x0F0F0F0Fu); } \
        if (tid < MMQ_BM) { int gw = row_block + tid; \
          rWd = (gw < rows) ? sharpi_fp16_to_fp32(ws[(long)gw * nb + (KB)]) : 0.f; } \
        int gy0 = tok_block + (tid >> 3); \
        rY0 = (gy0 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy0 * nb + (KB)) * 36L + 4 + (long)(tid & 7) * 4) : 0u; \
        int gy1 = tok_block + ((tid + 256) >> 3); \
        rY1 = (gy1 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy1 * nb + (KB)) * 36L + 4 + (long)((tid + 256) & 7) * 4) : 0u; \
        int gy2 = tok_block + ((tid + 512) >> 3); \
        rY2 = (gy2 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy2 * nb + (KB)) * 36L + 4 + (long)((tid + 512) & 7) * 4) : 0u; \
        int gy3 = tok_block + ((tid + 768) >> 3); \
        rY3 = (gy3 < n_tok) ? *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gy3 * nb + (KB)) * 36L + 4 + (long)((tid + 768) & 7) * 4) : 0u; \
        if (tid < MMQ_BN) { int gt = tok_block + tid; \
          if (gt < n_tok) { unsigned int dw = *reinterpret_cast<const unsigned int*>(y_q81 + ((long)gt * nb + (KB)) * 36L); \
            rYd = sharpi_fp16_to_fp32(dw & 0xffffu); rYs = sharpi_fp16_to_fp32(dw >> 16); } \
          else { rYd = 0.f; rYs = 0.f; } } \
    } while (0)

    MMQ_LOAD_TILE_Q40_SOA(0);

    for (int kb = 0; kb < nb; kb++) {
        sW[tid] = (int)rWq0; sW[tid + 256] = (int)rWq1;
        if (tid < MMQ_BM) sWd[tid] = rWd;
        sY[tid] = (int)rY0; sY[tid + 256] = (int)rY1; sY[tid + 512] = (int)rY2; sY[tid + 768] = (int)rY3;
        if (tid < MMQ_BN) { sYd[tid] = rYd; sYs[tid] = rYs; }
        __syncthreads();

        if (kb + 1 < nb) MMQ_LOAD_TILE_Q40_SOA(kb + 1);

        int a0 = sW[(mrow0 + grp) * 8     + tig];
        int a1 = sW[(mrow0 + grp + 8) * 8 + tig];
        int a2 = sW[(mrow0 + grp) * 8     + tig + 4];
        int a3 = sW[(mrow0 + grp + 8) * 8 + tig + 4];
        float dwA = sWd[mrow0 + grp];
        float dwB = sWd[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[(ncol0 + grp) * 8 + tig];
            int b1 = sY[(ncol0 + grp) * 8 + tig + 4];
            float dC0 = sYd[ncol0 + tig * 2], dC1 = sYd[ncol0 + tig * 2 + 1];
            float sC0 = sYs[ncol0 + tig * 2], sC1 = sYs[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            acc[nt][0] += (float)c0 * dwA * dC0 - 8.f * dwA * sC0;
            acc[nt][1] += (float)c1 * dwA * dC1 - 8.f * dwA * sC1;
            acc[nt][2] += (float)c2 * dwB * dC0 - 8.f * dwB * sC0;
            acc[nt][3] += (float)c3 * dwB * dC1 - 8.f * dwB * sC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef MMQ_LOAD_TILE_Q40_SOA
#undef MMQ_BM
#undef MMQ_BN

// ── Q4_0 SoA weights + SoA activations int8 MMQ (Track A, #124/#173) ────────
// Bit-identical to llm_mmq_q4_0_soa; only the activation reads switch to the SoA
// Q8_1 layout (contiguous qs + separate {d,s}). SAME load mapping / fragment map /
// accumulation order. nb == cols>>5 is the per-token activation block count.
#define MMQ_BM 64
#define MMQ_BN 128
extern ""C"" __global__ void llm_mmq_q4_0_soa_acts(
    const unsigned int*  __restrict__ weights,   // SoA [quants rows*cols/2 B][scales rows*nb fp16]
    const unsigned int*  __restrict__ y_qs,      // SoA Q8_1 quants [n_tok × nb × 32 int8]
    const unsigned int*  __restrict__ y_ds,      // SoA Q8_1 scales [n_tok × nb] {d,s} uint32
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    const unsigned int*   qw = weights;
    const unsigned short* ws = (const unsigned short*)((const char*)weights + ((long)rows * cols) / 2);

    __shared__ int   sW[MMQ_BM * 8];
    __shared__ float sWd[MMQ_BM];
    __shared__ int   sY[MMQ_BN * 8];
    __shared__ float sYd[MMQ_BN];
    __shared__ float sYs[MMQ_BN];

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb = cols >> 5;

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp  = lane >> 2;
    int tig  = lane & 3;
    int wr   = warp & 3;
    int wc   = warp >> 2;
    int mrow0 = wr * 16;

    float acc[8][4];
    #pragma unroll
    for (int n = 0; n < 8; n++) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f; }

    unsigned int rWq0, rWq1, rY0, rY1, rY2, rY3;
    float rWd, rYd, rYs;
    #define MMQ_LOAD_TILE_Q40_SOA_ACTS(KB) do { \
        int ww = tid & 7; int qword = ww & 3; \
        int gw0 = row_block + (tid >> 3); \
        { unsigned int w = (gw0 < rows) ? qw[((long)gw0 * nb + (KB)) * 4L + qword] : 0u; \
          rWq0 = (ww < 4) ? (w & 0x0F0F0F0Fu) : ((w >> 4) & 0x0F0F0F0Fu); } \
        int gw1 = row_block + ((tid + 256) >> 3); \
        { unsigned int w = (gw1 < rows) ? qw[((long)gw1 * nb + (KB)) * 4L + qword] : 0u; \
          rWq1 = (ww < 4) ? (w & 0x0F0F0F0Fu) : ((w >> 4) & 0x0F0F0F0Fu); } \
        if (tid < MMQ_BM) { int gw = row_block + tid; \
          rWd = (gw < rows) ? sharpi_fp16_to_fp32(ws[(long)gw * nb + (KB)]) : 0.f; } \
        int gy0 = tok_block + (tid >> 3); \
        rY0 = (gy0 < n_tok) ? y_qs[((long)gy0 * nb + (KB)) * 8L + (tid & 7)] : 0u; \
        int gy1 = tok_block + ((tid + 256) >> 3); \
        rY1 = (gy1 < n_tok) ? y_qs[((long)gy1 * nb + (KB)) * 8L + ((tid + 256) & 7)] : 0u; \
        int gy2 = tok_block + ((tid + 512) >> 3); \
        rY2 = (gy2 < n_tok) ? y_qs[((long)gy2 * nb + (KB)) * 8L + ((tid + 512) & 7)] : 0u; \
        int gy3 = tok_block + ((tid + 768) >> 3); \
        rY3 = (gy3 < n_tok) ? y_qs[((long)gy3 * nb + (KB)) * 8L + ((tid + 768) & 7)] : 0u; \
        if (tid < MMQ_BN) { int gt = tok_block + tid; \
          if (gt < n_tok) { unsigned int dw = y_ds[(long)gt * nb + (KB)]; \
            rYd = sharpi_fp16_to_fp32(dw & 0xffffu); rYs = sharpi_fp16_to_fp32(dw >> 16); } \
          else { rYd = 0.f; rYs = 0.f; } } \
    } while (0)

    MMQ_LOAD_TILE_Q40_SOA_ACTS(0);

    for (int kb = 0; kb < nb; kb++) {
        sW[tid] = (int)rWq0; sW[tid + 256] = (int)rWq1;
        if (tid < MMQ_BM) sWd[tid] = rWd;
        sY[tid] = (int)rY0; sY[tid + 256] = (int)rY1; sY[tid + 512] = (int)rY2; sY[tid + 768] = (int)rY3;
        if (tid < MMQ_BN) { sYd[tid] = rYd; sYs[tid] = rYs; }
        __syncthreads();

        if (kb + 1 < nb) MMQ_LOAD_TILE_Q40_SOA_ACTS(kb + 1);

        int a0 = sW[(mrow0 + grp) * 8     + tig];
        int a1 = sW[(mrow0 + grp + 8) * 8 + tig];
        int a2 = sW[(mrow0 + grp) * 8     + tig + 4];
        int a3 = sW[(mrow0 + grp + 8) * 8 + tig + 4];
        float dwA = sWd[mrow0 + grp];
        float dwB = sWd[mrow0 + grp + 8];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int ncol0 = wc * 64 + nt * 8;
            int b0 = sY[(ncol0 + grp) * 8 + tig];
            int b1 = sY[(ncol0 + grp) * 8 + tig + 4];
            float dC0 = sYd[ncol0 + tig * 2], dC1 = sYd[ncol0 + tig * 2 + 1];
            float sC0 = sYs[ncol0 + tig * 2], sC1 = sYs[ncol0 + tig * 2 + 1];
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            acc[nt][0] += (float)c0 * dwA * dC0 - 8.f * dwA * sC0;
            acc[nt][1] += (float)c1 * dwA * dC1 - 8.f * dwA * sC1;
            acc[nt][2] += (float)c2 * dwB * dC0 - 8.f * dwB * sC0;
            acc[nt][3] += (float)c3 * dwB * dC1 - 8.f * dwB * sC1;
        }
        __syncthreads();
    }

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        int ncol0 = tok_block + wc * 64 + nt * 8;
        int tokC0 = ncol0 + tig * 2;
        int tokC1 = ncol0 + tig * 2 + 1;
        if (rowA < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc[nt][0];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc[nt][1];
        }
        if (rowB < rows) {
            if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc[nt][2];
            if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc[nt][3];
        }
    }
}
#undef MMQ_LOAD_TILE_Q40_SOA_ACTS
#undef MMQ_BM
#undef MMQ_BN

// Batched per-head pure RmsNorm (no learned weight) over N tokens. grid =
// (num_heads, n_tok); data is token-major [n_tok × num_heads × head_dim].
// Per (head, token) bit-identical to llm_head_norm_pure. Used for the Gemma 4
// 12B k_eq_v V-norm in the batched-trunk prefill (issue #124).
extern ""C"" __global__ void llm_head_norm_pure_batched(
    float* __restrict__ data,
    int head_dim, int num_heads, float eps, int n_tok)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int head = blockIdx.x;
    int token = (int)blockIdx.y;
    if ((int)head >= num_heads || token >= n_tok) return;

    long base_off = ((long)token * (long)num_heads + (long)head) * (long)head_dim;

    float sum = 0.f;
    for (int i = (int)tid; i < head_dim; i += 256) {
        float v = data[base_off + i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = (int)tid; i < head_dim; i += 256)
        data[base_off + i] = data[base_off + i] * scale;
}

// ── Dequant Q4_K weight [rows × cols] → fp16 (issue #156 Item C / C1) ───────
// One block of 256 threads dequantizes one weight row (cols elements). cols must
// be a multiple of 256 (Q4_K super-block = 256 elements / 144 bytes). The per-
// element decode is identical to llm_embed_lookup_q4k / llm_matvec_q4k (same
// d*sc*nibble - dmin*mn), but parameterized by row and written as fp16. The fp16
// rounding is the only lossy step vs the fp32 dp4a matvec — it lets the prefill
// GEMM read each weight once per batch instead of re-streaming it once per token.
extern ""C"" __global__ void llm_dequant_q4k_to_f16(
    const unsigned int* __restrict__ weights,
    unsigned short* __restrict__ out,    // [rows * cols] fp16
    int rows, int cols)
{
    __shared__ unsigned int blk[36];
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    unsigned int tid = threadIdx.x;

    int num_blocks = cols >> 8;                              // cols / 256
    long row_word_base = (long)row * (long)num_blocks * 36L; // 144 bytes = 36 words per super-block
    long out_row = (long)row * (long)cols;

    for (int block = 0; block < num_blocks; block++) {
        long blk_word_base = row_word_base + (long)block * 36L;
        if (tid < 36)
            blk[tid] = weights[blk_word_base + tid];
        __syncthreads();

        unsigned int w0 = blk[0];
        float d    = sharpi_fp16_to_fp32(w0 & 0xffffu);
        float dmin = sharpi_fp16_to_fp32(w0 >> 16);

        unsigned int chunk = tid >> 6;          // 0..3
        unsigned int sub   = tid & 63u;         // 0..63
        unsigned int is_upper = (sub >= 32u) ? 1u : 0u;
        unsigned int byte_pos = sub & 31u;

        unsigned int sm0 = blk[1];
        unsigned int sm1 = blk[2];
        unsigned int sm2 = blk[3];

        unsigned int si = chunk * 2u + is_upper;
        float sc, mn;
        if (si < 4u) {
            sc = (float)((sm0 >> (si * 8u)) & 63u);
            mn = (float)((sm1 >> (si * 8u)) & 63u);
        } else {
            unsigned int j = si - 4u;
            sc = (float)(((sm2 >> (j * 8u)) & 0xFu)
                       | (((sm0 >> (j * 8u + 6u)) & 3u) << 4));
            mn = (float)(((sm2 >> (j * 8u + 4u)) & 0xFu)
                       | (((sm1 >> (j * 8u + 6u)) & 3u) << 4));
        }

        unsigned int qword = blk[4u + chunk * 8u + (byte_pos >> 2)];
        unsigned int qbyte = (qword >> ((byte_pos & 3u) * 8u)) & 0xFFu;
        unsigned int nibble = is_upper ? (qbyte >> 4) : (qbyte & 0xFu);

        float val = d * sc * (float)nibble - dmin * mn;
        out[out_row + (long)block * 256 + (int)tid] = (unsigned short)sharpi_fp32_to_fp16(val);
        __syncthreads();
    }
}

// ── Dequant Q3_K → FP16 (issue #388) ───────────────────────────────────────
// Standalone dequant of the 110-byte Q3_K super-block to a row-major [rows×cols]
// fp16 temp that cuBLAS GEMMs, mirroring llm_dequant_q4k_to_f16's output layout
// and llm_matvec_q3k_gemm_n's decode (same 110-byte AoS format + 6-bit scale aux
// unpack). One block of 256 threads per row; thread tid is element e_local in the
// 256-elem super-block, decoded as s=tid>>5, lane=tid&31 — the SAME (s,lane)→
// element mapping the matvec uses (input[base + s*32 + lane]), so f16(weight)×act
// via cuBLAS matches the in-kernel F32 GEMM-N dot (argmax-stable, fp16-rounded).
extern ""C"" __global__ void llm_dequant_q3k_to_f16(
    const unsigned int* __restrict__ weights,
    unsigned short* __restrict__ out,    // [rows * cols] fp16
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    unsigned int tid = threadIdx.x;          // 0..255 = element within super-block
    int num_blocks = cols >> 8;              // cols / 256
    long row_base_bytes = (long)row * (long)num_blocks * 110L;
    long out_row = (long)row * (long)cols;

    int s     = (int)(tid >> 5);             // 0..7 sub-block
    int lane  = (int)(tid & 31u);            // 0..31
    int group = lane >> 4;                    // 0/1
    int si    = 2 * s + group;               // scale index 0..15
    int half  = s >> 2;
    int shift = (s & 3) * 2;
    unsigned int m = 1u << s;

    const unsigned int kmask1 = 0x03030303u;
    const unsigned int kmask2 = 0x0f0f0f0fu;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 110L;

        unsigned int dlo = sharpi_byte_at(weights, b0 + 108);
        unsigned int dhi = sharpi_byte_at(weights, b0 + 109);
        float dAll = sharpi_fp16_to_fp32(dlo | (dhi << 8));

        unsigned int a0 = sharpi_byte_at(weights,b0+96)|(sharpi_byte_at(weights,b0+97)<<8)|(sharpi_byte_at(weights,b0+98)<<16)|(sharpi_byte_at(weights,b0+99)<<24);
        unsigned int a1 = sharpi_byte_at(weights,b0+100)|(sharpi_byte_at(weights,b0+101)<<8)|(sharpi_byte_at(weights,b0+102)<<16)|(sharpi_byte_at(weights,b0+103)<<24);
        unsigned int tmp= sharpi_byte_at(weights,b0+104)|(sharpi_byte_at(weights,b0+105)<<8)|(sharpi_byte_at(weights,b0+106)<<16)|(sharpi_byte_at(weights,b0+107)<<24);
        unsigned int aux[4];
        aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int sc = (int)((aux[si >> 2] >> ((si & 3) * 8)) & 0xFFu) - 32;   // signed 6-bit − 32
        unsigned int qsb = sharpi_byte_at(weights, b0 + 32 + half * 32 + lane);
        unsigned int hmb = sharpi_byte_at(weights, b0 + lane);
        int qval = (int)((qsb >> shift) & 3u) - ((hmb & m) != 0u ? 0 : 4);

        float val = dAll * (float)sc * (float)qval;
        out[out_row + (long)block * 256 + (int)tid] = (unsigned short)sharpi_fp32_to_fp16(val);
    }
}

// ── Dequant Q4_K → FP16 over the scale-pre-unpacked SoA weight (issue #156) ──
// SoA twin of llm_dequant_q4k_to_f16: same d*sc*nibble − dmin*mn decode, only the
// (scale, min) come from the pre-unpacked SoA bytes (sblk[si] / sblk[8+si]) and
// d/dmin from the D region. The fp16 output is bit-identical to the AoS dequant.
// Used only on the SHARPI_PREFILL_MMQ=0 dequant→fp16→GEMM fallback so a SoA
// weight never throws there.
extern ""C"" __global__ void llm_dequant_q4k_to_f16_soa(
    const unsigned int* __restrict__ weights,   // SoA: [Q][S][D]
    unsigned short* __restrict__ out,           // [rows * cols] fp16
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    unsigned int tid = threadIdx.x;

    int num_blocks = cols >> 8;
    long totalSub  = (long)rows * num_blocks;
    const unsigned char* qReg = (const unsigned char*)weights;
    const unsigned char* sReg = qReg + totalSub * 128L;
    const unsigned int*  dReg = (const unsigned int*)(sReg + totalSub * 16L);

    long out_row = (long)row * (long)cols;
    long row_blk_base = (long)row * num_blocks;

    for (int block = 0; block < num_blocks; block++) {
        long sb = row_blk_base + block;

        unsigned int dd = __ldg(&dReg[sb]);
        float d    = sharpi_fp16_to_fp32(dd & 0xffffu);
        float dmin = sharpi_fp16_to_fp32(dd >> 16);

        unsigned int chunk    = tid >> 6;          // 0..3
        unsigned int sub      = tid & 63u;         // 0..63
        unsigned int is_upper = (sub >= 32u) ? 1u : 0u;
        unsigned int byte_pos = sub & 31u;
        unsigned int si       = chunk * 2u + is_upper;

        const unsigned char* sblk = sReg + sb * 16L;
        float sc = (float)sblk[si];
        float mn = (float)sblk[8u + si];

        unsigned int qword  = __ldg(&weights[sb * 32L + (long)(chunk * 8u + (byte_pos >> 2))]);
        unsigned int qbyte  = (qword >> ((byte_pos & 3u) * 8u)) & 0xFFu;
        unsigned int nibble = is_upper ? (qbyte >> 4) : (qbyte & 0xFu);

        float val = d * sc * (float)nibble - dmin * mn;
        out[out_row + (long)block * 256 + (int)tid] = (unsigned short)sharpi_fp32_to_fp16(val);
    }
}

// ── Dequant Q6_K → FP16 for cuBLAS prefill GEMM (issue #162) ────────────────
// Qwen3-8B-Q4_K_M (and other _M mixes) keep ~half of ffn_down + attn_v in Q6_K. Those
// trunk matmuls had no compute-bound prefill path, so they fell back to the GEMM-N
// matvec (llm_matvec_q6k_gemm_n) which re-streams the whole weight ONCE PER TOKEN —
// memory-bound, and the dominant prefill cost at large N (94 GB of weight reads for one
// N=2290 ffn_down). This dequant lets the Q6_K weight be read once per batch into an
// fp16 temp that cuBLAS GEMMs, mirroring llm_dequant_q4k_to_f16.
//
// Element decode mirrors llm_matvec_q6k exactly. Thread e (0..255) owns weight column e
// of the super-block: with lane=e&31, group=e>>5, the matvec multiplies input[group*32
// + lane] == input[e], so out column == e. The 16 int8 scales cover 16 elements each
// (scale index 2*group + (lane>>4)); value = d·scale·(q − 32), fp16-rounded (the only
// lossy step vs the fp32 matvec). One block of 256 threads per row, looping super-blocks.
extern ""C"" __global__ void llm_dequant_q6k_to_f16(
    const unsigned int* __restrict__ weights,
    unsigned short* __restrict__ out,    // [rows * cols] fp16
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int num_blocks = cols >> 8;                              // cols / 256
    long row_base_bytes = (long)row * (long)num_blocks * 210L;
    long out_row = (long)row * (long)cols;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 210L;
        unsigned int dlo = sharpi_byte_at(weights, b0 + 208);
        unsigned int dhi = sharpi_byte_at(weights, b0 + 209);
        float d = sharpi_fp16_to_fp32(dlo | (dhi << 8));

        for (int e = (int)threadIdx.x; e < 256; e += (int)blockDim.x) {
            int lane  = e & 31;
            int group = e >> 5;        // 0..7
            int isc   = lane >> 4;     // 0 or 1
            float sc  = d * (float)sharpi_int8_at(weights, b0 + 192 + group * 2 + isc);

            unsigned int ql, qh; int q;
            switch (group) {
                case 0:  ql = sharpi_byte_at(weights, b0 +   0 + lane); qh = sharpi_byte_at(weights, b0 + 128 + lane);
                         q = (int)((ql & 0xFu)        | (((qh >> 0) & 3u) << 4)); break;
                case 1:  ql = sharpi_byte_at(weights, b0 +  32 + lane); qh = sharpi_byte_at(weights, b0 + 128 + lane);
                         q = (int)((ql & 0xFu)        | (((qh >> 2) & 3u) << 4)); break;
                case 2:  ql = sharpi_byte_at(weights, b0 +   0 + lane); qh = sharpi_byte_at(weights, b0 + 128 + lane);
                         q = (int)(((ql >> 4) & 0xFu) | (((qh >> 4) & 3u) << 4)); break;
                case 3:  ql = sharpi_byte_at(weights, b0 +  32 + lane); qh = sharpi_byte_at(weights, b0 + 128 + lane);
                         q = (int)(((ql >> 4) & 0xFu) | (((qh >> 6) & 3u) << 4)); break;
                case 4:  ql = sharpi_byte_at(weights, b0 +  64 + lane); qh = sharpi_byte_at(weights, b0 + 160 + lane);
                         q = (int)((ql & 0xFu)        | (((qh >> 0) & 3u) << 4)); break;
                case 5:  ql = sharpi_byte_at(weights, b0 +  96 + lane); qh = sharpi_byte_at(weights, b0 + 160 + lane);
                         q = (int)((ql & 0xFu)        | (((qh >> 2) & 3u) << 4)); break;
                case 6:  ql = sharpi_byte_at(weights, b0 +  64 + lane); qh = sharpi_byte_at(weights, b0 + 160 + lane);
                         q = (int)(((ql >> 4) & 0xFu) | (((qh >> 4) & 3u) << 4)); break;
                default: ql = sharpi_byte_at(weights, b0 +  96 + lane); qh = sharpi_byte_at(weights, b0 + 160 + lane);
                         q = (int)(((ql >> 4) & 0xFu) | (((qh >> 6) & 3u) << 4)); break;
            }
            float val = sc * (float)(q - 32);
            out[out_row + (long)block * 256 + e] = (unsigned short)sharpi_fp32_to_fp16(val);
        }
    }
}

// ── Dequant Q6_K → FP16 SoA (#204) ──────────────────────────────────────────
// Bit-identical clone of llm_dequant_q6k_to_f16 over the SoA layout (see
// llm_matvec_q6k_soa). Thread e owns weight column e of the super-block: scale
// S[g*16 + group*2 + (lane>>4)], weight (q6-32) = Q[g*256 + e], d = D[g*4].
// val = d·scale·(q6-32), fp16-rounded — same lossy step + same out index as the AoS kernel.
extern ""C"" __global__ void llm_dequant_q6k_to_f16_soa(
    const unsigned char* __restrict__ weights,   // SoA [Q][S][D]
    unsigned short* __restrict__ out,    // [rows * cols] fp16
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int num_blocks = cols >> 8;                              // cols / 256
    long out_row = (long)row * (long)cols;
    long total_sb = (long)rows * num_blocks;
    const signed char*   qReg = (const signed char*)weights;
    const signed char*   sReg = (const signed char*)weights + total_sb * 256L;
    const unsigned char*  dReg = (const unsigned char*)weights + total_sb * (256L + 16L);

    for (int block = 0; block < num_blocks; block++) {
        long g = (long)row * num_blocks + block;
        const signed char* q = qReg + g * 256L;
        const signed char* s = sReg + g * 16L;
        unsigned int dbits = (unsigned int)(*(const unsigned short*)(dReg + g * 4L));   // #204 review: dReg 16-B aligned, g*4 aligned → single 16-bit d load
        float d = sharpi_fp16_to_fp32(dbits);

        for (int e = (int)threadIdx.x; e < 256; e += (int)blockDim.x) {
            int lane  = e & 31;
            int group = e >> 5;        // 0..7
            int isc   = lane >> 4;     // 0 or 1
            float sc  = d * (float)s[group * 2 + isc];
            float val = sc * (float)q[e];
            out[out_row + (long)block * 256 + e] = (unsigned short)sharpi_fp32_to_fp16(val);
        }
    }
}

// ── Dequant Q5_K → FP16 for cuBLAS prefill GEMM (issue #162) ────────────────
// Same motivation as the Q6_K dequant: Q5_K_M mixes keep q/k/o/gate/up in Q5_K, which
// otherwise fell to the per-token GEMM-N matvec. Element decode mirrors llm_matvec_q5k:
// thread e owns weight column e (lane=e&31, chunk=e>>6, isHigh=(e>>5)&1); the matvec
// multiplies that value by input[e], so out column == e. The 12-byte scales[] use the
// SAME 6-bit packing as Q4_K (sharpi_q4k_scale_min over sub-block e>>5); the value is
// d·sc·(low4|hi4 + 16·qh_bit) − dmin·mn, fp16-rounded. One block of 256 threads per row.
extern ""C"" __global__ void llm_dequant_q5k_to_f16(
    const unsigned int* __restrict__ weights,
    unsigned short* __restrict__ out,    // [rows * cols] fp16
    int rows, int cols)
{
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    int num_blocks = cols >> 8;                              // cols / 256
    long row_base_bytes = (long)row * (long)num_blocks * 176L;
    long out_row = (long)row * (long)cols;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 176L;       // 176 B / super-block, 4-aligned
        unsigned int dword0 = weights[b0 >> 2];
        float d    = sharpi_fp16_to_fp32(dword0 & 0xffffu);
        float dmin = sharpi_fp16_to_fp32(dword0 >> 16);
        unsigned int sm0 = weights[(b0 >> 2) + 1];           // scales[0:4]
        unsigned int sm1 = weights[(b0 >> 2) + 2];           // scales[4:8]
        unsigned int sm2 = weights[(b0 >> 2) + 3];           // scales[8:12]

        for (int e = (int)threadIdx.x; e < 256; e += (int)blockDim.x) {
            int chunk  = e >> 6;          // 0..3
            int lane   = e & 31;
            int isHigh = (e >> 5) & 1;    // low (0..31) vs high (32..63) half of the chunk
            unsigned int sc, mn;
            sharpi_q4k_scale_min(sm0, sm1, sm2, (e >> 5), &sc, &mn);   // sub-block = e/32

            unsigned int ql_byte = sharpi_byte_at(weights, b0 + 48 + chunk * 32 + lane);
            unsigned int nibble  = isHigh ? ((ql_byte >> 4) & 0xFu) : (ql_byte & 0xFu);
            unsigned int qh_byte = sharpi_byte_at(weights, b0 + 16 + lane);
            unsigned int u = 1u << (2 * chunk + isHigh);
            int q5 = (int)nibble + ((qh_byte & u) != 0u ? 16 : 0);

            float val = d * (float)sc * (float)q5 - dmin * (float)mn;
            out[out_row + (long)block * 256 + e] = (unsigned short)sharpi_fp32_to_fp16(val);
        }
    }
}

// ── FP32 → FP16 elementwise convert (issue #141) ──────────────────────────
// Converts the prefill activation batch [n] fp32 → fp16 so it can feed the
// cuBLAS fp16 GEMM alongside the dequantized weights.
extern ""C"" __global__ void llm_f32_to_f16(
    const float* __restrict__ in,
    unsigned short* __restrict__ out,
    int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = (unsigned short)sharpi_fp32_to_fp16(in[i]);
}

// ── MatVec F32 — N=2 variant (issue #43) ──────────────────────────────────
// Two input vectors, two output vectors, single weight read. Mirrors the
// llm_matvec_f32 launch geometry (8 rows × 32 threads/row). Wins over two
// sequential single-input matvecs by issuing one global weight load per
// (row, lane) and folding it into two MACs.
extern ""C"" __global__ void llm_matvec_f32_n2(
    const float* __restrict__ weights,
    const float* __restrict__ input_a,
    const float* __restrict__ input_b,
    float* __restrict__ output_a,
    float* __restrict__ output_b,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    float acc_a = 0.f, acc_b = 0.f;
    long base = (long)row * (long)cols;
    for (int i = lane; i < cols; i += THREADS_PER_ROW) {
        float w = weights[base + i];
        acc_a += w * input_a[i];
        acc_b += w * input_b[i];
    }

    float ra = sharpi_warp_reduce_sum(acc_a);
    float rb = sharpi_warp_reduce_sum(acc_b);
    if (lane == 0) {
        output_a[row] = ra;
        output_b[row] = rb;
    }
}

// ── MatVec Q4_K — N=2 variant (issue #43) ─────────────────────────────────
// Reads two pre-quantized Q8_1 input buffers (one per draft token), produces
// two output rows. Each lane decodes the weight super-block once and runs
// two __dp4a chains — one against y_q81_a, one against y_q81_b. The min
// correction is computed independently per input via the 0x01010101 dp4a.
extern ""C"" __global__ void llm_matvec_q4k_n2(
    const unsigned int* __restrict__ weights,
    const unsigned char* __restrict__ y_q81_a,
    const unsigned char* __restrict__ y_q81_b,
    float* __restrict__ output_a,
    float* __restrict__ output_b,
    int rows, int cols)
{
    int row     = (int)blockIdx.x;
    int warp_id = (int)threadIdx.y;
    int lane    = (int)threadIdx.x;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long word_row_base = (long)row * (long)num_blocks * 36L;

    int chunk    = lane >> 3;
    int byte_off = (lane & 7) * 4;
    int q4_offset = 4 + chunk * 8 + (lane & 7);

    float acc_a = 0.f, acc_b = 0.f;

    for (int block = warp_id; block < num_blocks; block += MATVEC_Q4K_NWARPS) {
        long word_base = word_row_base + (long)block * 36L;

        unsigned int w0  = __ldg(&weights[word_base]);
        unsigned int sm0 = __ldg(&weights[word_base + 1]);
        unsigned int sm1 = __ldg(&weights[word_base + 2]);
        unsigned int sm2 = __ldg(&weights[word_base + 3]);
        float super_d    = sharpi_fp16_to_fp32(w0 & 0xffffu);
        float super_dmin = sharpi_fp16_to_fp32(w0 >> 16);

        unsigned int sc_lo, mn_lo, sc_hi, mn_hi;
        switch (chunk) {
            case 0:
                sc_lo = (sm0)       & 63u; mn_lo = (sm1)       & 63u;
                sc_hi = (sm0 >>  8) & 63u; mn_hi = (sm1 >>  8) & 63u;
                break;
            case 1:
                sc_lo = (sm0 >> 16) & 63u; mn_lo = (sm1 >> 16) & 63u;
                sc_hi = (sm0 >> 24) & 63u; mn_hi = (sm1 >> 24) & 63u;
                break;
            case 2:
                sc_lo = (sm2        & 0xFu) | (((sm0 >>  6) & 3u) << 4);
                mn_lo = ((sm2 >>  4) & 0xFu) | (((sm1 >>  6) & 3u) << 4);
                sc_hi = ((sm2 >>  8) & 0xFu) | (((sm0 >> 14) & 3u) << 4);
                mn_hi = ((sm2 >> 12) & 0xFu) | (((sm1 >> 14) & 3u) << 4);
                break;
            default:
                sc_lo = ((sm2 >> 16) & 0xFu) | (((sm0 >> 22) & 3u) << 4);
                mn_lo = ((sm2 >> 20) & 0xFu) | (((sm1 >> 22) & 3u) << 4);
                sc_hi = ((sm2 >> 24) & 0xFu) | (((sm0 >> 30) & 3u) << 4);
                mn_hi = ((sm2 >> 28) & 0xFu) | (((sm1 >> 30) & 3u) << 4);
                break;
        }

        unsigned int wq    = __ldg(&weights[word_base + q4_offset]);
        unsigned int wq_lo = wq & 0x0F0F0F0Fu;
        unsigned int wq_hi = (wq >> 4) & 0x0F0F0F0Fu;

        long q81_base_lo = (long)(block * 8 + chunk * 2)     * 36L;
        long q81_base_hi = (long)(block * 8 + chunk * 2 + 1) * 36L;

        // Per-input d and activation pair — loaded for A and B independently.
        unsigned int d_bits_lo_a = __ldg(reinterpret_cast<const unsigned int*>(y_q81_a + q81_base_lo)) & 0xffffu;
        unsigned int d_bits_hi_a = __ldg(reinterpret_cast<const unsigned int*>(y_q81_a + q81_base_hi)) & 0xffffu;
        unsigned int d_bits_lo_b = __ldg(reinterpret_cast<const unsigned int*>(y_q81_b + q81_base_lo)) & 0xffffu;
        unsigned int d_bits_hi_b = __ldg(reinterpret_cast<const unsigned int*>(y_q81_b + q81_base_hi)) & 0xffffu;
        float d8_lo_a = sharpi_fp16_to_fp32(d_bits_lo_a);
        float d8_hi_a = sharpi_fp16_to_fp32(d_bits_hi_a);
        float d8_lo_b = sharpi_fp16_to_fp32(d_bits_lo_b);
        float d8_hi_b = sharpi_fp16_to_fp32(d_bits_hi_b);

        int act_lo_a = *reinterpret_cast<const int*>(y_q81_a + q81_base_lo + 4 + byte_off);
        int act_hi_a = *reinterpret_cast<const int*>(y_q81_a + q81_base_hi + 4 + byte_off);
        int act_lo_b = *reinterpret_cast<const int*>(y_q81_b + q81_base_lo + 4 + byte_off);
        int act_hi_b = *reinterpret_cast<const int*>(y_q81_b + q81_base_hi + 4 + byte_off);

        int dot_lo_a = __dp4a((int)wq_lo, act_lo_a, 0);
        int dot_hi_a = __dp4a((int)wq_hi, act_hi_a, 0);
        int sum_lo_a = __dp4a((int)0x01010101, act_lo_a, 0);
        int sum_hi_a = __dp4a((int)0x01010101, act_hi_a, 0);
        int dot_lo_b = __dp4a((int)wq_lo, act_lo_b, 0);
        int dot_hi_b = __dp4a((int)wq_hi, act_hi_b, 0);
        int sum_lo_b = __dp4a((int)0x01010101, act_lo_b, 0);
        int sum_hi_b = __dp4a((int)0x01010101, act_hi_b, 0);

        float sd_sc_lo = super_d * (float)sc_lo;
        float sm_mn_lo = super_dmin * (float)mn_lo;
        float sd_sc_hi = super_d * (float)sc_hi;
        float sm_mn_hi = super_dmin * (float)mn_hi;

        acc_a += sd_sc_lo * d8_lo_a * (float)dot_lo_a - sm_mn_lo * d8_lo_a * (float)sum_lo_a;
        acc_a += sd_sc_hi * d8_hi_a * (float)dot_hi_a - sm_mn_hi * d8_hi_a * (float)sum_hi_a;
        acc_b += sd_sc_lo * d8_lo_b * (float)dot_lo_b - sm_mn_lo * d8_lo_b * (float)sum_lo_b;
        acc_b += sd_sc_hi * d8_hi_b * (float)dot_hi_b - sm_mn_hi * d8_hi_b * (float)sum_hi_b;
    }

    acc_a = sharpi_warp_reduce_sum(acc_a);
    acc_b = sharpi_warp_reduce_sum(acc_b);

    __shared__ float warp_acc_a[MATVEC_Q4K_NWARPS];
    __shared__ float warp_acc_b[MATVEC_Q4K_NWARPS];
    if (lane == 0) { warp_acc_a[warp_id] = acc_a; warp_acc_b[warp_id] = acc_b; }
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        float sa = 0.f, sb = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q4K_NWARPS; w++) { sa += warp_acc_a[w]; sb += warp_acc_b[w]; }
        output_a[row] = sa;
        output_b[row] = sb;
    }
}

// ── MatVec Q4_K — N=2 over the scale-pre-unpacked SoA weight (issue #156) ──
// SoA twin of llm_matvec_q4k_n2: identical two-input dp4a + accumulation order,
// only the (scale, min) and d/dmin come from the pre-unpacked SoA regions
// (plain bytes, no 6-bit switch) instead of the interleaved 144-B block. The
// stored integers are bit-identical to the switch output, so this is
// bit-identical to llm_matvec_q4k_n2 (and thus to N sequential AoS matvecs).
// Same NWARPS=8 and per-warp reduction tree → FP-order-preserving (the MTP
// byte-parity oracle is sensitive to cumulative trunk drift). Lets the dense
// MTP batched-verify path consume the SoA weight that the decode matvec uses.
extern ""C"" __global__ void llm_matvec_q4k_n2_soa(
    const unsigned int* __restrict__ weights,    // SoA: [Q][S][D]
    const unsigned char* __restrict__ y_q81_a,
    const unsigned char* __restrict__ y_q81_b,
    float* __restrict__ output_a,
    float* __restrict__ output_b,
    int rows, int cols)
{
    int row     = (int)blockIdx.x;
    int warp_id = (int)threadIdx.y;
    int lane    = (int)threadIdx.x;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long totalSub  = (long)rows * num_blocks;

    const unsigned char* qReg = (const unsigned char*)weights;
    const unsigned char* sReg = qReg + totalSub * 128L;
    const unsigned int*  dReg = (const unsigned int*)(sReg + totalSub * 16L);

    int chunk    = lane >> 3;
    int byte_off = (lane & 7) * 4;
    int q_word_in_block = chunk * 8 + (lane & 7);

    long row_blk_base = (long)row * num_blocks;

    float acc_a = 0.f, acc_b = 0.f;

    for (int block = warp_id; block < num_blocks; block += MATVEC_Q4K_NWARPS) {
        long sb = row_blk_base + block;

        unsigned int dd  = __ldg(&dReg[sb]);
        float super_d    = sharpi_fp16_to_fp32(dd & 0xffffu);
        float super_dmin = sharpi_fp16_to_fp32(dd >> 16);

        const unsigned char* sblk = sReg + sb * 16L;
        unsigned int sc_lo = sblk[2 * chunk];
        unsigned int sc_hi = sblk[2 * chunk + 1];
        unsigned int mn_lo = sblk[8 + 2 * chunk];
        unsigned int mn_hi = sblk[8 + 2 * chunk + 1];

        unsigned int wq    = __ldg(&weights[sb * 32L + q_word_in_block]);
        unsigned int wq_lo = wq & 0x0F0F0F0Fu;
        unsigned int wq_hi = (wq >> 4) & 0x0F0F0F0Fu;

        long q81_base_lo = (long)(block * 8 + chunk * 2)     * 36L;
        long q81_base_hi = (long)(block * 8 + chunk * 2 + 1) * 36L;

        unsigned int d_bits_lo_a = __ldg(reinterpret_cast<const unsigned int*>(y_q81_a + q81_base_lo)) & 0xffffu;
        unsigned int d_bits_hi_a = __ldg(reinterpret_cast<const unsigned int*>(y_q81_a + q81_base_hi)) & 0xffffu;
        unsigned int d_bits_lo_b = __ldg(reinterpret_cast<const unsigned int*>(y_q81_b + q81_base_lo)) & 0xffffu;
        unsigned int d_bits_hi_b = __ldg(reinterpret_cast<const unsigned int*>(y_q81_b + q81_base_hi)) & 0xffffu;
        float d8_lo_a = sharpi_fp16_to_fp32(d_bits_lo_a);
        float d8_hi_a = sharpi_fp16_to_fp32(d_bits_hi_a);
        float d8_lo_b = sharpi_fp16_to_fp32(d_bits_lo_b);
        float d8_hi_b = sharpi_fp16_to_fp32(d_bits_hi_b);

        int act_lo_a = *reinterpret_cast<const int*>(y_q81_a + q81_base_lo + 4 + byte_off);
        int act_hi_a = *reinterpret_cast<const int*>(y_q81_a + q81_base_hi + 4 + byte_off);
        int act_lo_b = *reinterpret_cast<const int*>(y_q81_b + q81_base_lo + 4 + byte_off);
        int act_hi_b = *reinterpret_cast<const int*>(y_q81_b + q81_base_hi + 4 + byte_off);

        int dot_lo_a = __dp4a((int)wq_lo, act_lo_a, 0);
        int dot_hi_a = __dp4a((int)wq_hi, act_hi_a, 0);
        int sum_lo_a = __dp4a((int)0x01010101, act_lo_a, 0);
        int sum_hi_a = __dp4a((int)0x01010101, act_hi_a, 0);
        int dot_lo_b = __dp4a((int)wq_lo, act_lo_b, 0);
        int dot_hi_b = __dp4a((int)wq_hi, act_hi_b, 0);
        int sum_lo_b = __dp4a((int)0x01010101, act_lo_b, 0);
        int sum_hi_b = __dp4a((int)0x01010101, act_hi_b, 0);

        float sd_sc_lo = super_d * (float)sc_lo;
        float sm_mn_lo = super_dmin * (float)mn_lo;
        float sd_sc_hi = super_d * (float)sc_hi;
        float sm_mn_hi = super_dmin * (float)mn_hi;

        acc_a += sd_sc_lo * d8_lo_a * (float)dot_lo_a - sm_mn_lo * d8_lo_a * (float)sum_lo_a;
        acc_a += sd_sc_hi * d8_hi_a * (float)dot_hi_a - sm_mn_hi * d8_hi_a * (float)sum_hi_a;
        acc_b += sd_sc_lo * d8_lo_b * (float)dot_lo_b - sm_mn_lo * d8_lo_b * (float)sum_lo_b;
        acc_b += sd_sc_hi * d8_hi_b * (float)dot_hi_b - sm_mn_hi * d8_hi_b * (float)sum_hi_b;
    }

    acc_a = sharpi_warp_reduce_sum(acc_a);
    acc_b = sharpi_warp_reduce_sum(acc_b);

    __shared__ float warp_acc_a[MATVEC_Q4K_NWARPS];
    __shared__ float warp_acc_b[MATVEC_Q4K_NWARPS];
    if (lane == 0) { warp_acc_a[warp_id] = acc_a; warp_acc_b[warp_id] = acc_b; }
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        float sa = 0.f, sb = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q4K_NWARPS; w++) { sa += warp_acc_a[w]; sb += warp_acc_b[w]; }
        output_a[row] = sa;
        output_b[row] = sb;
    }
}

// ── MatVec Q6_K — N=2 variant (issue #43) ─────────────────────────────────
extern ""C"" __global__ void llm_matvec_q6k_n2(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input_a,
    const float* __restrict__ input_b,
    float* __restrict__ output_a,
    float* __restrict__ output_b,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long row_base_bytes = (long)row * (long)num_blocks * 210L;

    float acc_a = 0.f, acc_b = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 210L;

        unsigned int dlo = (weights[(b0 + 208) >> 2] >> (((b0 + 208) & 3) * 8)) & 0xFFu;
        unsigned int dhi = (weights[(b0 + 209) >> 2] >> (((b0 + 209) & 3) * 8)) & 0xFFu;
        float d = sharpi_fp16_to_fp32(dlo | (dhi << 8));

        long isc = (long)(lane >> 4);

        float sc0 = d * (float)sharpi_int8_at(weights, b0 + 192 + isc);
        float sc1 = d * (float)sharpi_int8_at(weights, b0 + 194 + isc);
        float sc2 = d * (float)sharpi_int8_at(weights, b0 + 196 + isc);
        float sc3 = d * (float)sharpi_int8_at(weights, b0 + 198 + isc);
        float sc4 = d * (float)sharpi_int8_at(weights, b0 + 200 + isc);
        float sc5 = d * (float)sharpi_int8_at(weights, b0 + 202 + isc);
        float sc6 = d * (float)sharpi_int8_at(weights, b0 + 204 + isc);
        float sc7 = d * (float)sharpi_int8_at(weights, b0 + 206 + isc);

        unsigned int ql0 = sharpi_byte_at(weights, b0 +   0 + lane);
        unsigned int ql1 = sharpi_byte_at(weights, b0 +  32 + lane);
        unsigned int ql2 = sharpi_byte_at(weights, b0 +  64 + lane);
        unsigned int ql3 = sharpi_byte_at(weights, b0 +  96 + lane);
        unsigned int qh0 = sharpi_byte_at(weights, b0 + 128 + lane);
        unsigned int qh1 = sharpi_byte_at(weights, b0 + 160 + lane);

        int base_elem = block * 256;

        float q0 = sc0 * (float)((int)((ql0 & 0xFu)        | (((qh0 >> 0) & 3u) << 4)) - 32);
        float q1 = sc1 * (float)((int)((ql1 & 0xFu)        | (((qh0 >> 2) & 3u) << 4)) - 32);
        float q2 = sc2 * (float)((int)(((ql0 >> 4) & 0xFu) | (((qh0 >> 4) & 3u) << 4)) - 32);
        float q3 = sc3 * (float)((int)(((ql1 >> 4) & 0xFu) | (((qh0 >> 6) & 3u) << 4)) - 32);
        float q4 = sc4 * (float)((int)((ql2 & 0xFu)        | (((qh1 >> 0) & 3u) << 4)) - 32);
        float q5 = sc5 * (float)((int)((ql3 & 0xFu)        | (((qh1 >> 2) & 3u) << 4)) - 32);
        float q6 = sc6 * (float)((int)(((ql2 >> 4) & 0xFu) | (((qh1 >> 4) & 3u) << 4)) - 32);
        float q7 = sc7 * (float)((int)(((ql3 >> 4) & 0xFu) | (((qh1 >> 6) & 3u) << 4)) - 32);

        acc_a += q0 * input_a[base_elem +       lane];
        acc_a += q1 * input_a[base_elem +  32 + lane];
        acc_a += q2 * input_a[base_elem +  64 + lane];
        acc_a += q3 * input_a[base_elem +  96 + lane];
        acc_a += q4 * input_a[base_elem + 128 + lane];
        acc_a += q5 * input_a[base_elem + 160 + lane];
        acc_a += q6 * input_a[base_elem + 192 + lane];
        acc_a += q7 * input_a[base_elem + 224 + lane];

        acc_b += q0 * input_b[base_elem +       lane];
        acc_b += q1 * input_b[base_elem +  32 + lane];
        acc_b += q2 * input_b[base_elem +  64 + lane];
        acc_b += q3 * input_b[base_elem +  96 + lane];
        acc_b += q4 * input_b[base_elem + 128 + lane];
        acc_b += q5 * input_b[base_elem + 160 + lane];
        acc_b += q6 * input_b[base_elem + 192 + lane];
        acc_b += q7 * input_b[base_elem + 224 + lane];
    }

    float ra = sharpi_warp_reduce_sum(acc_a);
    float rb = sharpi_warp_reduce_sum(acc_b);
    if (lane == 0) {
        output_a[row] = ra;
        output_b[row] = rb;
    }
}

// ── MatVec Q6_K — N=2 SoA (#204) ────────────────────────────────────────────
// Bit-identical clone of llm_matvec_q6k_n2 over the SoA layout (see
// llm_matvec_q6k_soa). Same per-element dequant + reduction order for both inputs.
extern ""C"" __global__ void llm_matvec_q6k_n2_soa(
    const unsigned char* __restrict__ weights,   // SoA [Q][S][D]
    const float* __restrict__ input_a,
    const float* __restrict__ input_b,
    float* __restrict__ output_a,
    float* __restrict__ output_b,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long total_sb = (long)rows * num_blocks;
    const signed char*   qReg = (const signed char*)weights;
    const signed char*   sReg = (const signed char*)weights + total_sb * 256L;
    const unsigned char*  dReg = (const unsigned char*)weights + total_sb * (256L + 16L);

    long isc = (long)(lane >> 4);

    float acc_a = 0.f, acc_b = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long g = (long)row * num_blocks + block;
        const signed char* q = qReg + g * 256L;
        const signed char* s = sReg + g * 16L;
        unsigned int dbits = (unsigned int)(*(const unsigned short*)(dReg + g * 4L));   // #204 review: dReg 16-B aligned, g*4 aligned → single 16-bit d load
        float d = sharpi_fp16_to_fp32(dbits);

        float sc0 = d * (float)s[ 0 + isc];
        float sc1 = d * (float)s[ 2 + isc];
        float sc2 = d * (float)s[ 4 + isc];
        float sc3 = d * (float)s[ 6 + isc];
        float sc4 = d * (float)s[ 8 + isc];
        float sc5 = d * (float)s[10 + isc];
        float sc6 = d * (float)s[12 + isc];
        float sc7 = d * (float)s[14 + isc];

        int base_elem = block * 256;

        float q0 = sc0 * (float)q[       lane];
        float q1 = sc1 * (float)q[ 32 + lane];
        float q2 = sc2 * (float)q[ 64 + lane];
        float q3 = sc3 * (float)q[ 96 + lane];
        float q4 = sc4 * (float)q[128 + lane];
        float q5 = sc5 * (float)q[160 + lane];
        float q6 = sc6 * (float)q[192 + lane];
        float q7 = sc7 * (float)q[224 + lane];

        acc_a += q0 * input_a[base_elem +       lane];
        acc_a += q1 * input_a[base_elem +  32 + lane];
        acc_a += q2 * input_a[base_elem +  64 + lane];
        acc_a += q3 * input_a[base_elem +  96 + lane];
        acc_a += q4 * input_a[base_elem + 128 + lane];
        acc_a += q5 * input_a[base_elem + 160 + lane];
        acc_a += q6 * input_a[base_elem + 192 + lane];
        acc_a += q7 * input_a[base_elem + 224 + lane];

        acc_b += q0 * input_b[base_elem +       lane];
        acc_b += q1 * input_b[base_elem +  32 + lane];
        acc_b += q2 * input_b[base_elem +  64 + lane];
        acc_b += q3 * input_b[base_elem +  96 + lane];
        acc_b += q4 * input_b[base_elem + 128 + lane];
        acc_b += q5 * input_b[base_elem + 160 + lane];
        acc_b += q6 * input_b[base_elem + 192 + lane];
        acc_b += q7 * input_b[base_elem + 224 + lane];
    }

    float ra = sharpi_warp_reduce_sum(acc_a);
    float rb = sharpi_warp_reduce_sum(acc_b);
    if (lane == 0) {
        output_a[row] = ra;
        output_b[row] = rb;
    }
}

// ── MatVec Q5_K — N=2 variant (issue #43) ─────────────────────────────────
extern ""C"" __global__ void llm_matvec_q5k_n2(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input_a,
    const float* __restrict__ input_b,
    float* __restrict__ output_a,
    float* __restrict__ output_b,
    int rows, int cols)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long row_base_bytes = (long)row * (long)num_blocks * 176L;

    float acc_a = 0.f, acc_b = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 176L;

        unsigned int dword0 = __ldg(&weights[b0 >> 2]);
        float d    = sharpi_fp16_to_fp32(dword0 & 0xffffu);
        float dmin = sharpi_fp16_to_fp32(dword0 >> 16);

        unsigned int qh_byte = sharpi_byte_at(weights, b0 + 16 + lane);

        int base_elem = block * 256;

        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned int sc_lo_byte, sc_hi_byte;
            unsigned int sc1, m1, sc2, m2;
            int j_lo = chunk * 2;
            int j_hi = j_lo + 1;
            if (j_lo < 4) {
                sc_lo_byte = sharpi_byte_at(weights, b0 + 4 + j_lo);
                sc_hi_byte = sharpi_byte_at(weights, b0 + 4 + j_lo + 4);
                sc1 = sc_lo_byte & 63u;
                m1  = sc_hi_byte & 63u;
                unsigned int sc_lo2 = sharpi_byte_at(weights, b0 + 4 + j_hi);
                unsigned int sc_hi2 = sharpi_byte_at(weights, b0 + 4 + j_hi + 4);
                sc2 = sc_lo2 & 63u;
                m2  = sc_hi2 & 63u;
            } else {
                unsigned int a_lo = sharpi_byte_at(weights, b0 + 4 + j_lo + 4);
                unsigned int b_lo = sharpi_byte_at(weights, b0 + 4 + j_lo - 4);
                unsigned int c_lo = sharpi_byte_at(weights, b0 + 4 + j_lo);
                sc1 = (a_lo & 0xFu) | (((b_lo >> 6) & 3u) << 4);
                m1  = ((a_lo >> 4) & 0xFu) | (((c_lo >> 6) & 3u) << 4);

                unsigned int a_hi = sharpi_byte_at(weights, b0 + 4 + j_hi + 4);
                unsigned int b_hi = sharpi_byte_at(weights, b0 + 4 + j_hi - 4);
                unsigned int c_hi = sharpi_byte_at(weights, b0 + 4 + j_hi);
                sc2 = (a_hi & 0xFu) | (((b_hi >> 6) & 3u) << 4);
                m2  = ((a_hi >> 4) & 0xFu) | (((c_hi >> 6) & 3u) << 4);
            }

            float d1  = d * (float)sc1;
            float dm1 = dmin * (float)m1;
            float d2  = d * (float)sc2;
            float dm2 = dmin * (float)m2;

            unsigned int u1 = 1u << (2 * chunk);
            unsigned int u2 = u1 << 1;

            unsigned int ql_byte = sharpi_byte_at(weights, b0 + 48 + chunk * 32 + lane);
            unsigned int low4 = ql_byte & 0xFu;
            unsigned int hi4  = (ql_byte >> 4) & 0xFu;

            int hLo = (qh_byte & u1) != 0u ? 16 : 0;
            int hHi = (qh_byte & u2) != 0u ? 16 : 0;

            int elem_lo = base_elem + chunk * 64 + lane;
            int elem_hi = elem_lo + 32;

            float qlo = d1 * (float)((int)low4 + hLo) - dm1;
            float qhi = d2 * (float)((int)hi4  + hHi) - dm2;

            acc_a += qlo * input_a[elem_lo];
            acc_a += qhi * input_a[elem_hi];
            acc_b += qlo * input_b[elem_lo];
            acc_b += qhi * input_b[elem_hi];
        }
    }

    float ra = sharpi_warp_reduce_sum(acc_a);
    float rb = sharpi_warp_reduce_sum(acc_b);
    if (lane == 0) {
        output_a[row] = ra;
        output_b[row] = rb;
    }
}

// ── MatVec F32 — batched GEMM-N variant (issue #111) ──────────────────────
// One weight matrix, N input vectors, N output rows. grid = (ceil(rows/8), N);
// block = 256 threads (8 rows × 32 lanes). Block (rx, t) computes 8 output rows
// for token t. Each (row, token) runs the IDENTICAL per-row reduction as
// llm_matvec_f32, so output_all is bit-identical to N sequential llm_matvec_f32
// launches — only the input/output base pointers shift by token. Collapses the
// N per-token launches into one, killing the host launch overhead that dominates
// GDN-hybrid prefill (#111).
extern ""C"" __global__ void llm_matvec_f32_gemm_n(
    const float* __restrict__ weights,
    const float* __restrict__ input_all,   // [n_tok][cols] contiguous
    float* __restrict__ output_all,        // [n_tok][rows] contiguous
    int rows, int cols, int n_tok)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    int token = (int)blockIdx.y;
    if (row >= rows || token >= n_tok) return;

    const float* input = input_all + (long)token * (long)cols;

    float acc = 0.f;
    long base = (long)row * (long)cols;
    for (int i = lane; i < cols; i += THREADS_PER_ROW)
        acc += weights[base + i] * input[i];

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output_all[(long)token * (long)rows + row] = result;
}

// ── MatVec Q4_K — batched GEMM-N variant (issue #111) ─────────────────────
// One weight matrix, N input vectors (pre-quantized to Q8_1 as N contiguous
// q81 rows), N output rows. grid = (rows, N); block = (32, NWARPS). Block
// (row, token) runs the IDENTICAL per-row reduction as llm_matvec_q4k — same
// weight decode, same dp4a chain, same warp + shared reduce — so output_all is
// bit-identical to N sequential llm_matvec_q4k launches. Weight is reread per
// token (the trunk bottleneck is launch latency, not weight bandwidth — #111).
extern ""C"" __global__ void llm_matvec_q4k_gemm_n(
    const unsigned int* __restrict__ weights,
    const unsigned char* __restrict__ y_q81_all,  // [n_tok][num_blocks*8*36] bytes
    float* __restrict__ output_all,               // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    int row     = (int)blockIdx.x;
    int token   = (int)blockIdx.y;
    int warp_id = (int)threadIdx.y;
    int lane    = (int)threadIdx.x;
    if (row >= rows || token >= n_tok) return;

    int num_blocks = cols >> 8;
    long word_row_base = (long)row * (long)num_blocks * 36L;
    // q81 row stride = (cols/32) sub-blocks × 36 bytes = num_blocks*8*36.
    const unsigned char* y_q81 = y_q81_all + (long)token * (long)num_blocks * 8L * 36L;

    int chunk    = lane >> 3;
    int byte_off = (lane & 7) * 4;
    int q4_offset = 4 + chunk * 8 + (lane & 7);

    float acc = 0.f;

    for (int block = warp_id; block < num_blocks; block += MATVEC_Q4K_NWARPS) {
        long word_base = word_row_base + (long)block * 36L;

        unsigned int w0  = __ldg(&weights[word_base]);
        unsigned int sm0 = __ldg(&weights[word_base + 1]);
        unsigned int sm1 = __ldg(&weights[word_base + 2]);
        unsigned int sm2 = __ldg(&weights[word_base + 3]);
        float super_d    = sharpi_fp16_to_fp32(w0 & 0xffffu);
        float super_dmin = sharpi_fp16_to_fp32(w0 >> 16);

        unsigned int sc_lo, mn_lo, sc_hi, mn_hi;
        switch (chunk) {
            case 0:
                sc_lo = (sm0)       & 63u; mn_lo = (sm1)       & 63u;
                sc_hi = (sm0 >>  8) & 63u; mn_hi = (sm1 >>  8) & 63u;
                break;
            case 1:
                sc_lo = (sm0 >> 16) & 63u; mn_lo = (sm1 >> 16) & 63u;
                sc_hi = (sm0 >> 24) & 63u; mn_hi = (sm1 >> 24) & 63u;
                break;
            case 2:
                sc_lo = (sm2        & 0xFu) | (((sm0 >>  6) & 3u) << 4);
                mn_lo = ((sm2 >>  4) & 0xFu) | (((sm1 >>  6) & 3u) << 4);
                sc_hi = ((sm2 >>  8) & 0xFu) | (((sm0 >> 14) & 3u) << 4);
                mn_hi = ((sm2 >> 12) & 0xFu) | (((sm1 >> 14) & 3u) << 4);
                break;
            default:
                sc_lo = ((sm2 >> 16) & 0xFu) | (((sm0 >> 22) & 3u) << 4);
                mn_lo = ((sm2 >> 20) & 0xFu) | (((sm1 >> 22) & 3u) << 4);
                sc_hi = ((sm2 >> 24) & 0xFu) | (((sm0 >> 30) & 3u) << 4);
                mn_hi = ((sm2 >> 28) & 0xFu) | (((sm1 >> 30) & 3u) << 4);
                break;
        }

        unsigned int wq    = __ldg(&weights[word_base + q4_offset]);
        unsigned int wq_lo = wq & 0x0F0F0F0Fu;
        unsigned int wq_hi = (wq >> 4) & 0x0F0F0F0Fu;

        long q81_base_lo = (long)(block * 8 + chunk * 2)     * 36L;
        long q81_base_hi = (long)(block * 8 + chunk * 2 + 1) * 36L;

        unsigned int d_bits_lo = __ldg(reinterpret_cast<const unsigned int*>(y_q81 + q81_base_lo)) & 0xffffu;
        unsigned int d_bits_hi = __ldg(reinterpret_cast<const unsigned int*>(y_q81 + q81_base_hi)) & 0xffffu;
        float d8_lo = sharpi_fp16_to_fp32(d_bits_lo);
        float d8_hi = sharpi_fp16_to_fp32(d_bits_hi);

        int act_lo = *reinterpret_cast<const int*>(y_q81 + q81_base_lo + 4 + byte_off);
        int act_hi = *reinterpret_cast<const int*>(y_q81 + q81_base_hi + 4 + byte_off);

        int dot_lo   = __dp4a((int)wq_lo, act_lo, 0);
        int dot_hi   = __dp4a((int)wq_hi, act_hi, 0);
        int sum_lo   = __dp4a((int)0x01010101, act_lo, 0);
        int sum_hi   = __dp4a((int)0x01010101, act_hi, 0);

        float coef_d_lo = super_d    * (float)sc_lo * d8_lo;
        float coef_m_lo = super_dmin * (float)mn_lo * d8_lo;
        float coef_d_hi = super_d    * (float)sc_hi * d8_hi;
        float coef_m_hi = super_dmin * (float)mn_hi * d8_hi;
        acc += coef_d_lo * (float)dot_lo - coef_m_lo * (float)sum_lo;
        acc += coef_d_hi * (float)dot_hi - coef_m_hi * (float)sum_hi;
    }

    acc = sharpi_warp_reduce_sum(acc);

    __shared__ float warp_acc[MATVEC_Q4K_NWARPS];
    if (lane == 0) warp_acc[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q4K_NWARPS; w++) s += warp_acc[w];
        output_all[(long)token * (long)rows + row] = s;
    }
}

// ── MatVec Q4_K — batched GEMM-N over the scale-pre-unpacked SoA weight (#156) ─
// SoA twin of llm_matvec_q4k_gemm_n: identical per-(row,token) reduction, only
// the weight decode reads the pre-unpacked SoA regions. Bit-identical to the AoS
// GEMM-N (and to N sequential llm_matvec_q4k_soa launches). Used only on the
// SHARPI_PREFILL_MMQ=0 fallback prefill path so a SoA weight never throws there.
extern ""C"" __global__ void llm_matvec_q4k_gemm_n_soa(
    const unsigned int* __restrict__ weights,     // SoA: [Q][S][D]
    const unsigned char* __restrict__ y_q81_all,  // [n_tok][num_blocks*8*36] bytes
    float* __restrict__ output_all,               // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    int row     = (int)blockIdx.x;
    int token   = (int)blockIdx.y;
    int warp_id = (int)threadIdx.y;
    int lane    = (int)threadIdx.x;
    if (row >= rows || token >= n_tok) return;

    int num_blocks = cols >> 8;
    long totalSub  = (long)rows * num_blocks;
    const unsigned char* qReg = (const unsigned char*)weights;
    const unsigned char* sReg = qReg + totalSub * 128L;
    const unsigned int*  dReg = (const unsigned int*)(sReg + totalSub * 16L);

    const unsigned char* y_q81 = y_q81_all + (long)token * (long)num_blocks * 8L * 36L;

    int chunk    = lane >> 3;
    int byte_off = (lane & 7) * 4;
    int q_word_in_block = chunk * 8 + (lane & 7);

    long row_blk_base = (long)row * num_blocks;

    float acc = 0.f;

    for (int block = warp_id; block < num_blocks; block += MATVEC_Q4K_NWARPS) {
        long sb = row_blk_base + block;

        unsigned int dd  = __ldg(&dReg[sb]);
        float super_d    = sharpi_fp16_to_fp32(dd & 0xffffu);
        float super_dmin = sharpi_fp16_to_fp32(dd >> 16);

        const unsigned char* sblk = sReg + sb * 16L;
        unsigned int sc_lo = sblk[2 * chunk];
        unsigned int sc_hi = sblk[2 * chunk + 1];
        unsigned int mn_lo = sblk[8 + 2 * chunk];
        unsigned int mn_hi = sblk[8 + 2 * chunk + 1];

        unsigned int wq    = __ldg(&weights[sb * 32L + q_word_in_block]);
        unsigned int wq_lo = wq & 0x0F0F0F0Fu;
        unsigned int wq_hi = (wq >> 4) & 0x0F0F0F0Fu;

        long q81_base_lo = (long)(block * 8 + chunk * 2)     * 36L;
        long q81_base_hi = (long)(block * 8 + chunk * 2 + 1) * 36L;

        unsigned int d_bits_lo = __ldg(reinterpret_cast<const unsigned int*>(y_q81 + q81_base_lo)) & 0xffffu;
        unsigned int d_bits_hi = __ldg(reinterpret_cast<const unsigned int*>(y_q81 + q81_base_hi)) & 0xffffu;
        float d8_lo = sharpi_fp16_to_fp32(d_bits_lo);
        float d8_hi = sharpi_fp16_to_fp32(d_bits_hi);

        int act_lo = *reinterpret_cast<const int*>(y_q81 + q81_base_lo + 4 + byte_off);
        int act_hi = *reinterpret_cast<const int*>(y_q81 + q81_base_hi + 4 + byte_off);

        int dot_lo   = __dp4a((int)wq_lo, act_lo, 0);
        int dot_hi   = __dp4a((int)wq_hi, act_hi, 0);
        int sum_lo   = __dp4a((int)0x01010101, act_lo, 0);
        int sum_hi   = __dp4a((int)0x01010101, act_hi, 0);

        float coef_d_lo = super_d    * (float)sc_lo * d8_lo;
        float coef_m_lo = super_dmin * (float)mn_lo * d8_lo;
        float coef_d_hi = super_d    * (float)sc_hi * d8_hi;
        float coef_m_hi = super_dmin * (float)mn_hi * d8_hi;
        acc += coef_d_lo * (float)dot_lo - coef_m_lo * (float)sum_lo;
        acc += coef_d_hi * (float)dot_hi - coef_m_hi * (float)sum_hi;
    }

    acc = sharpi_warp_reduce_sum(acc);

    __shared__ float warp_acc[MATVEC_Q4K_NWARPS];
    if (lane == 0) warp_acc[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q4K_NWARPS; w++) s += warp_acc[w];
        output_all[(long)token * (long)rows + row] = s;
    }
}

// ── MatVec Q6_K — batched GEMM-N variant (issue #111) ─────────────────────
// One weight matrix, N F32 input vectors, N output rows. grid = (ceil(rows/8), N);
// block = 256 (8 rows × 32 lanes). Block (rx, token) computes 8 output rows for
// token. Identical per-row reduction to llm_matvec_q6k; only input/output base
// pointers shift by token. Used for the shared-expert down projection (Q6_K).
extern ""C"" __global__ void llm_matvec_q6k_gemm_n(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input_all,   // [n_tok][cols]
    float* __restrict__ output_all,        // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    int token = (int)blockIdx.y;
    if (row >= rows || token >= n_tok) return;

    const float* input = input_all + (long)token * (long)cols;
    int num_blocks = cols >> 8;
    long row_base_bytes = (long)row * (long)num_blocks * 210L;

    float acc = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 210L;

        unsigned int dlo = (weights[(b0 + 208) >> 2] >> (((b0 + 208) & 3) * 8)) & 0xFFu;
        unsigned int dhi = (weights[(b0 + 209) >> 2] >> (((b0 + 209) & 3) * 8)) & 0xFFu;
        float d = sharpi_fp16_to_fp32(dlo | (dhi << 8));

        long isc = (long)(lane >> 4);

        float sc0 = d * (float)sharpi_int8_at(weights, b0 + 192 + isc);
        float sc1 = d * (float)sharpi_int8_at(weights, b0 + 194 + isc);
        float sc2 = d * (float)sharpi_int8_at(weights, b0 + 196 + isc);
        float sc3 = d * (float)sharpi_int8_at(weights, b0 + 198 + isc);
        float sc4 = d * (float)sharpi_int8_at(weights, b0 + 200 + isc);
        float sc5 = d * (float)sharpi_int8_at(weights, b0 + 202 + isc);
        float sc6 = d * (float)sharpi_int8_at(weights, b0 + 204 + isc);
        float sc7 = d * (float)sharpi_int8_at(weights, b0 + 206 + isc);

        unsigned int ql0 = sharpi_byte_at(weights, b0 +   0 + lane);
        unsigned int ql1 = sharpi_byte_at(weights, b0 +  32 + lane);
        unsigned int ql2 = sharpi_byte_at(weights, b0 +  64 + lane);
        unsigned int ql3 = sharpi_byte_at(weights, b0 +  96 + lane);
        unsigned int qh0 = sharpi_byte_at(weights, b0 + 128 + lane);
        unsigned int qh1 = sharpi_byte_at(weights, b0 + 160 + lane);

        int base_elem = block * 256;

        acc += sc0 * (float)((int)((ql0 & 0xFu)        | (((qh0 >> 0) & 3u) << 4)) - 32) * input[base_elem +       lane];
        acc += sc1 * (float)((int)((ql1 & 0xFu)        | (((qh0 >> 2) & 3u) << 4)) - 32) * input[base_elem +  32 + lane];
        acc += sc2 * (float)((int)(((ql0 >> 4) & 0xFu) | (((qh0 >> 4) & 3u) << 4)) - 32) * input[base_elem +  64 + lane];
        acc += sc3 * (float)((int)(((ql1 >> 4) & 0xFu) | (((qh0 >> 6) & 3u) << 4)) - 32) * input[base_elem +  96 + lane];
        acc += sc4 * (float)((int)((ql2 & 0xFu)        | (((qh1 >> 0) & 3u) << 4)) - 32) * input[base_elem + 128 + lane];
        acc += sc5 * (float)((int)((ql3 & 0xFu)        | (((qh1 >> 2) & 3u) << 4)) - 32) * input[base_elem + 160 + lane];
        acc += sc6 * (float)((int)(((ql2 >> 4) & 0xFu) | (((qh1 >> 4) & 3u) << 4)) - 32) * input[base_elem + 192 + lane];
        acc += sc7 * (float)((int)(((ql3 >> 4) & 0xFu) | (((qh1 >> 6) & 3u) << 4)) - 32) * input[base_elem + 224 + lane];
    }

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output_all[(long)token * (long)rows + row] = result;
}

// ── MatVec Q6_K GEMM-N SoA (#204) ───────────────────────────────────────────
// Bit-identical clone of llm_matvec_q6k_gemm_n over the SoA layout (see
// llm_matvec_q6k_soa). Same per-(row,token) reduction order, so a (rows, n_tok)
// launch is bit-identical to n_tok sequential llm_matvec_q6k_soa calls.
extern ""C"" __global__ void llm_matvec_q6k_gemm_n_soa(
    const unsigned char* __restrict__ weights,   // SoA [Q][S][D]
    const float* __restrict__ input_all,   // [n_tok][cols]
    float* __restrict__ output_all,        // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    int token = (int)blockIdx.y;
    if (row >= rows || token >= n_tok) return;

    const float* input = input_all + (long)token * (long)cols;
    int num_blocks = cols >> 8;
    long total_sb = (long)rows * num_blocks;
    const signed char*   qReg = (const signed char*)weights;
    const signed char*   sReg = (const signed char*)weights + total_sb * 256L;
    const unsigned char*  dReg = (const unsigned char*)weights + total_sb * (256L + 16L);

    long isc = (long)(lane >> 4);

    float acc = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long g = (long)row * num_blocks + block;
        const signed char* q = qReg + g * 256L;
        const signed char* s = sReg + g * 16L;
        unsigned int dbits = (unsigned int)(*(const unsigned short*)(dReg + g * 4L));   // #204 review: dReg 16-B aligned, g*4 aligned → single 16-bit d load
        float d = sharpi_fp16_to_fp32(dbits);

        float sc0 = d * (float)s[ 0 + isc];
        float sc1 = d * (float)s[ 2 + isc];
        float sc2 = d * (float)s[ 4 + isc];
        float sc3 = d * (float)s[ 6 + isc];
        float sc4 = d * (float)s[ 8 + isc];
        float sc5 = d * (float)s[10 + isc];
        float sc6 = d * (float)s[12 + isc];
        float sc7 = d * (float)s[14 + isc];

        int base_elem = block * 256;
        acc += sc0 * (float)q[       lane] * input[base_elem +       lane];
        acc += sc1 * (float)q[ 32 + lane] * input[base_elem +  32 + lane];
        acc += sc2 * (float)q[ 64 + lane] * input[base_elem +  64 + lane];
        acc += sc3 * (float)q[ 96 + lane] * input[base_elem +  96 + lane];
        acc += sc4 * (float)q[128 + lane] * input[base_elem + 128 + lane];
        acc += sc5 * (float)q[160 + lane] * input[base_elem + 160 + lane];
        acc += sc6 * (float)q[192 + lane] * input[base_elem + 192 + lane];
        acc += sc7 * (float)q[224 + lane] * input[base_elem + 224 + lane];
    }

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output_all[(long)token * (long)rows + row] = result;
}

// ── MatVec Q3_K GEMM-N (issue #100) ────────────────────────────────────────
// Raw in-kernel-dequant Q3_K GEMM-N so the op-offload MoE prefill can upload the
// compact Q3_K bytes (110 B / 256-elem super-block) and matmul on GPU directly —
// no host dequant→F32 (the 24.6s / 79% prefill cost this kernel removes). F32
// input, per-element Q3_K weight decode, 8 rows/block × 32 thr/row warp reduce —
// same geometry + output layout output_all[token*rows + row] as the Q5_K/Q6_K/Q8_0
// GEMM-N siblings, so it is a drop-in MatMulBatched dispatch target.
//
// Q3_K super-block (110 bytes, mirrors SimdKernels.DotQ3K / ggml block_q3_K):
//   [0:32]    hmask — one high bit per element (bit s of byte e for sub-block s)
//   [32:96]   qs    — 2 low bits per element (64 bytes; half h at qs[h*32 + e])
//   [96:108]  12 scale bytes → 16 signed 6-bit scales via the kmask1/kmask2 aux
//             unpack (identical to DotQ3K_Scalar / ggml get_scale)
//   [108:110] dAll  — FP16 super-block scale
// Element value qu = ((qs[..]>>shift)&3) + (hmask bit ? 4 : 0) ∈ [0,7]; the weight
// contribution is dAll·(scale_g−32)·(qu−4)·act, summed over all 256 elements.
//
// Lane layout (lane 0..31, one warp/row): the 8 sub-blocks s=0..7 are walked in
// order; lane handles element e = s*32 + lane. shift = (s&3)*2, mask bit m = 1<<s,
// qs byte = qs[(s>>2)*32 + lane], hmask byte = hmask[lane], scale group = lane>>4
// (so scale index si = 2*s + (lane>>4)). This reproduces the SAME per-element
// products as DotQ3K_Scalar (only the warp lanes parallelize the inner l-loop;
// each lane's term is identical), giving an exact F32 dequant-and-dot — argmax-
// stable vs the host DotQ3K F32 reference (no activation quantization here).
//
// 110 is not 4-aligned per super-block, so every weight byte is read via
// sharpi_byte_at / sharpi_int8_at byte gathers (cf. llm_matvec_q6k), never a
// uint-indexed load that would assume 4-alignment. The three scale uint32 words
// are likewise assembled from byte gathers before the aux unpack.
//
// Reference: llama.cpp ggml-cuda dequantize_block_q3_K (convert.cu) /
// vec_dot_q3_K_q8_1 (vecdotq.cuh) for the 3-bit unpack + 6-bit scale decode (MIT);
// adapted to our 110-byte AoS layout and the GEMM-N output convention.
extern ""C"" __global__ void llm_matvec_q3k_gemm_n(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input_all,   // [n_tok][cols]
    float* __restrict__ output_all,        // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    int token = (int)blockIdx.y;
    if (row >= rows || token >= n_tok) return;

    const float* input = input_all + (long)token * (long)cols;
    int num_blocks = cols >> 8;
    long row_base_bytes = (long)row * (long)num_blocks * 110L;

    int group = lane >> 4;          // 0 → first-16 scale, 1 → second-16 scale

    float acc = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 110L;

        // dAll: FP16 at bytes [108:110].
        unsigned int dlo = sharpi_byte_at(weights, b0 + 108);
        unsigned int dhi = sharpi_byte_at(weights, b0 + 109);
        float dAll = sharpi_fp16_to_fp32(dlo | (dhi << 8));

        // The 12 scale bytes [96:108) → three uint32 words (byte gathers, no
        // 4-alignment assumed), then the kmask1/kmask2 aux unpack producing 16
        // signed 6-bit scales (identical to DotQ3K_Scalar / ggml get_scale).
        const unsigned int kmask1 = 0x03030303u;
        const unsigned int kmask2 = 0x0f0f0f0fu;
        unsigned int a0 = sharpi_byte_at(weights, b0 + 96)
                        | (sharpi_byte_at(weights, b0 + 97) << 8)
                        | (sharpi_byte_at(weights, b0 + 98) << 16)
                        | (sharpi_byte_at(weights, b0 + 99) << 24);
        unsigned int a1 = sharpi_byte_at(weights, b0 + 100)
                        | (sharpi_byte_at(weights, b0 + 101) << 8)
                        | (sharpi_byte_at(weights, b0 + 102) << 16)
                        | (sharpi_byte_at(weights, b0 + 103) << 24);
        unsigned int tmp = sharpi_byte_at(weights, b0 + 104)
                        | (sharpi_byte_at(weights, b0 + 105) << 8)
                        | (sharpi_byte_at(weights, b0 + 106) << 16)
                        | (sharpi_byte_at(weights, b0 + 107) << 24);
        unsigned int aux[4];
        aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int base_elem = block * 256;

        // Walk the 8 sub-blocks; lane handles element s*32 + lane each.
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            int half  = s >> 2;
            int shift = (s & 3) * 2;
            unsigned int m = 1u << s;

            int si = 2 * s + group;                       // scale index 0..15
            int sc = (int)((aux[si >> 2] >> ((si & 3) * 8)) & 0xFFu);
            sc -= 32;                                     // signed 6-bit scale − 32

            unsigned int qsb = sharpi_byte_at(weights, b0 + 32 + half * 32 + lane);
            unsigned int hmb = sharpi_byte_at(weights, b0 + lane);
            int qval = (int)((qsb >> shift) & 3u) - ((hmb & m) != 0u ? 0 : 4);

            acc += dAll * (float)sc * (float)qval * input[base_elem + s * 32 + lane];
        }
    }

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output_all[(long)token * (long)rows + row] = result;
}

// ── MatVec Q5_K GEMM-N (issue #119) ────────────────────────────────────────
// Batched (token-dimension) clone of `llm_matvec_q5k`: F32 input, per-element
// Q5_K weight decode, 8 rows/block × 32 threads/row warp reduce. Per-lane
// accumulation order is byte-for-byte identical to the single-token kernel, so a
// (rows, n_tok) launch is bit-identical to n_tok sequential `llm_matvec_q5k`
// calls. Lets the batched trunk (TrunkBlockBatched) drive Q5_K projection weights,
// which Q4_K_M-quantized GDN-hybrid models (e.g. Qwen3.6-27B-MTP) carry.
extern ""C"" __global__ void llm_matvec_q5k_gemm_n(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input_all,   // [n_tok][cols]
    float* __restrict__ output_all,        // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    int token = (int)blockIdx.y;
    if (row >= rows || token >= n_tok) return;

    const float* input = input_all + (long)token * (long)cols;
    int num_blocks = cols >> 8;
    long row_base_bytes = (long)row * (long)num_blocks * 176L;

    float acc = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 176L;

        unsigned int dword0 = __ldg(&weights[b0 >> 2]);
        float d    = sharpi_fp16_to_fp32(dword0 & 0xffffu);
        float dmin = sharpi_fp16_to_fp32(dword0 >> 16);

        unsigned int qh_byte = sharpi_byte_at(weights, b0 + 16 + lane);

        int base_elem = block * 256;

        #pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            unsigned int sc_lo_byte, sc_hi_byte;
            unsigned int sc1, m1, sc2, m2;
            int j_lo = chunk * 2;
            int j_hi = j_lo + 1;
            if (j_lo < 4) {
                sc_lo_byte = sharpi_byte_at(weights, b0 + 4 + j_lo);
                sc_hi_byte = sharpi_byte_at(weights, b0 + 4 + j_lo + 4);
                sc1 = sc_lo_byte & 63u;
                m1  = sc_hi_byte & 63u;
                unsigned int sc_lo2 = sharpi_byte_at(weights, b0 + 4 + j_hi);
                unsigned int sc_hi2 = sharpi_byte_at(weights, b0 + 4 + j_hi + 4);
                sc2 = sc_lo2 & 63u;
                m2  = sc_hi2 & 63u;
            } else {
                unsigned int a_lo = sharpi_byte_at(weights, b0 + 4 + j_lo + 4);
                unsigned int b_lo = sharpi_byte_at(weights, b0 + 4 + j_lo - 4);
                unsigned int c_lo = sharpi_byte_at(weights, b0 + 4 + j_lo);
                sc1 = (a_lo & 0xFu) | (((b_lo >> 6) & 3u) << 4);
                m1  = ((a_lo >> 4) & 0xFu) | (((c_lo >> 6) & 3u) << 4);

                unsigned int a_hi = sharpi_byte_at(weights, b0 + 4 + j_hi + 4);
                unsigned int b_hi = sharpi_byte_at(weights, b0 + 4 + j_hi - 4);
                unsigned int c_hi = sharpi_byte_at(weights, b0 + 4 + j_hi);
                sc2 = (a_hi & 0xFu) | (((b_hi >> 6) & 3u) << 4);
                m2  = ((a_hi >> 4) & 0xFu) | (((c_hi >> 6) & 3u) << 4);
            }

            float d1  = d * (float)sc1;
            float dm1 = dmin * (float)m1;
            float d2  = d * (float)sc2;
            float dm2 = dmin * (float)m2;

            unsigned int u1 = 1u << (2 * chunk);
            unsigned int u2 = u1 << 1;

            unsigned int ql_byte = sharpi_byte_at(weights, b0 + 48 + chunk * 32 + lane);
            unsigned int low4 = ql_byte & 0xFu;
            unsigned int hi4  = (ql_byte >> 4) & 0xFu;

            int hLo = (qh_byte & u1) != 0u ? 16 : 0;
            int hHi = (qh_byte & u2) != 0u ? 16 : 0;

            int elem_lo = base_elem + chunk * 64 + lane;
            int elem_hi = elem_lo + 32;

            acc += (d1 * (float)((int)low4 + hLo) - dm1) * input[elem_lo];
            acc += (d2 * (float)((int)hi4  + hHi) - dm2) * input[elem_hi];
        }
    }

    float result = sharpi_warp_reduce_sum(acc);
    if (lane == 0) output_all[(long)token * (long)rows + row] = result;
}

// ── Batched trunk elementwise/norm kernels (issue #111) ───────────────────
// Each adds a token dimension (grid.y or a token-major offset) over the
// single-token kernel above; the per-token math is byte-for-byte identical, so
// batched output matches N sequential single-token launches exactly. Used by
// CudaHybridGdnForwardPass batched prefill to collapse the per-token trunk
// launches into one launch per op.

// RmsNorm over N rows: one block per (token), 256 threads. Identical reduction
// to llm_rmsnorm; input/output offset by token*n; weight shared across tokens.
extern ""C"" __global__ void llm_rmsnorm_batched(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int n, float eps, int n_tok)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    int token = (int)blockIdx.x;
    if (token >= n_tok) return;

    const float* in = input + (long)token * (long)n;
    float* out = output + (long)token * (long)n;

    float sum = 0.f;
    for (int i = (int)tid; i < n; i += 256) {
        float v = in[i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = (int)tid; i < n; i += 256)
        out[i] = in[i] * scale * weight[i];
}

// HeadNorm over N rows: grid = (num_heads, n_tok); block 256. data offset by
// token*(num_heads*head_dim). Identical reduction to llm_head_norm.
extern ""C"" __global__ void llm_head_norm_batched(
    float* __restrict__ data,
    const float* __restrict__ weight,
    int head_dim, int num_heads, float eps, int weight_stride, int n_tok)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int head = blockIdx.x;
    int token = (int)blockIdx.y;
    if ((int)head >= num_heads || token >= n_tok) return;

    long token_off = (long)token * (long)num_heads * (long)head_dim;
    long base_off = token_off + (long)head * head_dim;
    int  w_off    = (int)head * weight_stride;

    float sum = 0.f;
    for (int i = (int)tid; i < head_dim; i += 256) {
        float v = data[base_off + i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = (int)tid; i < head_dim; i += 256)
        data[base_off + i] = data[base_off + i] * scale * weight[w_off + i];
}

// Batched dual Q+K HeadNorm over N tokens. grid = (num_heads+num_kv_heads, n_tok);
// Q blocks stride by num_heads*head_dim, K blocks by num_kv_heads*head_dim. Per
// (block, token) bit-identical to llm_head_norm_batched.
extern ""C"" __global__ void llm_head_norm_qk_batched(
    float* __restrict__ q_data, const float* __restrict__ q_weight,
    float* __restrict__ k_data, const float* __restrict__ k_weight,
    int head_dim, int num_heads, int num_kv_heads, float eps, int weight_stride, int n_tok)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int blk = blockIdx.x;
    int token = (int)blockIdx.y;
    if ((int)blk >= num_heads + num_kv_heads || token >= n_tok) return;

    bool is_q = (int)blk < num_heads;
    int head = is_q ? (int)blk : (int)blk - num_heads;
    float* data = is_q ? q_data : k_data;
    const float* weight = is_q ? q_weight : k_weight;
    int heads_here = is_q ? num_heads : num_kv_heads;

    long token_off = (long)token * (long)heads_here * (long)head_dim;
    long base_off = token_off + (long)head * head_dim;
    int  w_off    = head * weight_stride;

    float sum = 0.f;
    for (int i = (int)tid; i < head_dim; i += 256) {
        float v = data[base_off + i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(sdata[0] / (float)head_dim + eps);
    for (int i = (int)tid; i < head_dim; i += 256)
        data[base_off + i] = data[base_off + i] * scale * weight[w_off + i];
}

// Strided de-interleave [Q‖G] → Q, G over N tokens. qg row stride =
// num_heads*head_dim*2; q/g row stride = num_heads*head_dim.
extern ""C"" __global__ void llm_split_qg_batched(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ g,
    int num_heads, int head_dim, int n_tok)
{
    int total = num_heads * head_dim;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int token = (int)blockIdx.y;
    if (idx >= total || token >= n_tok) return;

    int h = idx / head_dim;
    int j = idx % head_dim;
    long qg_base = (long)token * (long)num_heads * (long)head_dim * 2L + (long)h * head_dim * 2;
    long out_base = (long)token * (long)total + (long)h * head_dim;
    q[out_base + j] = qg[qg_base + j];
    g[out_base + j] = qg[qg_base + head_dim + j];
}

// Partial NEOX RoPE over N tokens. Position for token t is base_position + t
// (prompt prefill assigns contiguous positions). x row stride = num_heads*head_dim.
extern ""C"" __global__ void llm_rope_neox_partial_batched(
    float* __restrict__ x,
    int num_heads, int head_dim, int rope_dim, int base_position, float theta, int n_tok)
{
    int pair_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int rope_half_dim = rope_dim / 2;
    int total_pairs = num_heads * rope_half_dim;
    int token = (int)blockIdx.y;
    if (pair_idx >= total_pairs || token >= n_tok) return;

    int h = pair_idx / rope_half_dim;
    int i = pair_idx % rope_half_dim;
    int position = base_position + token;

    float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)rope_dim);
    float angle = (float)position * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    long head_base = (long)token * (long)num_heads * (long)head_dim + (long)h * head_dim;
    long a = head_base + i;
    long b = head_base + i + rope_half_dim;
    float x0 = x[a];
    float x1 = x[b];
    x[a] = x0 * c - x1 * s;
    x[b] = x0 * s + x1 * c;
}

// Fused q+k partial NEOX RoPE over N tokens (DSpark GPU draft, issue #428):
// one launch rotates both the query rows (q, [n_tok × num_q_heads*head_dim])
// and the key rows (k, [n_tok × num_kv_heads*head_dim]) at position
// base_position + t. The pair space is the union of both buffers' pairs:
// [0, q_pairs) hits q, the rest hits k. The per-pair rotation math is the
// same as llm_rope_neox_partial_batched, so per buffer the result is
// bit-identical to two separate launches.
extern ""C"" __global__ void llm_rope_neox_partial_batched_qk(
    float* __restrict__ q,
    float* __restrict__ k,
    int num_q_heads, int num_kv_heads, int head_dim, int rope_dim,
    int base_position, float theta, int n_tok)
{
    int pair_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int rope_half_dim = rope_dim / 2;
    int q_pairs = num_q_heads * rope_half_dim;
    int total_pairs = (num_q_heads + num_kv_heads) * rope_half_dim;
    int token = (int)blockIdx.y;
    if (pair_idx >= total_pairs || token >= n_tok) return;

    float* x;
    int h, i, num_heads;
    if (pair_idx < q_pairs) {
        x = q; h = pair_idx / rope_half_dim; i = pair_idx % rope_half_dim;
        num_heads = num_q_heads;
    } else {
        int p = pair_idx - q_pairs;
        x = k; h = p / rope_half_dim; i = p % rope_half_dim;
        num_heads = num_kv_heads;
    }
    int position = base_position + token;

    float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)rope_dim);
    float angle = (float)position * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    long head_base = (long)token * (long)num_heads * (long)head_dim + (long)h * head_dim;
    long a = head_base + i;
    long b = head_base + i + rope_half_dim;
    float x0 = x[a];
    float x1 = x[b];
    x[a] = x0 * c - x1 * s;
    x[b] = x0 * s + x1 * c;
}

// ── TurboQuant rotate_query ────────────────────────────────────────────────
// Applies the Walsh-Hadamard transform + per-layer sign flip to each query
// head. One block per query head, head_dim threads per block.
//
// head_dim must be a power of two ≤ 256 (the WHT butterfly uses head_dim/2
// active threads per stage and `shared float sdata[256]` storage).
extern ""C"" __global__ void llm_tq_rotate_query(
    const float* __restrict__ q_input,
    float* __restrict__ rotated_q,
    const float* __restrict__ sign_patterns,
    int num_heads, int num_kv_heads, int head_dim)
{
    __shared__ float sdata[256];

    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    if (h >= num_heads || tid >= head_dim) return;

    int kv_head = h / (num_heads / num_kv_heads);
    int q_off = h * head_dim;
    int sign_off = kv_head * head_dim;

    sdata[tid] = q_input[q_off + tid];
    __syncthreads();

    // WHT butterfly: dim/2 active threads per stage.
    for (int s = head_dim >> 1; s >= 1; s >>= 1) {
        if (tid < (head_dim >> 1)) {
            int pos_lo = (tid / s) * (s << 1) + (tid % s);
            int pos_hi = pos_lo + s;
            float a = sdata[pos_lo];
            float b = sdata[pos_hi];
            sdata[pos_lo] = a + b;
            sdata[pos_hi] = a - b;
        }
        __syncthreads();
    }

    float scale = rsqrtf((float)head_dim);
    rotated_q[q_off + tid] = sdata[tid] * scale * sign_patterns[sign_off + tid];
}

// ── TurboQuant KV append ───────────────────────────────────────────────────
// Compresses one K and one V vector for a single token-position into the
// per-layer TQ caches. One block per KV head, head_dim threads per block.
//
// Block layout (3-bit, head_dim=128): 2-byte FP16 norm + 48 bytes packed
// indices + 2 bytes padding = 52 bytes, stored as 13 uint32 words. The output
// buffers are typed as uint[] (matches the Vulkan binding); the host computes
// the byte offset as `position * num_kv_heads * block_bytes + kv_head *
// block_bytes`, and this kernel divides by 4 to address the uint[] view.
extern ""C"" __global__ void llm_tq_kv_append(
    const float* __restrict__ k_input,
    const float* __restrict__ v_input,
    unsigned int* __restrict__ k_cache_tq,
    unsigned int* __restrict__ v_cache_tq,
    const float* __restrict__ sign_patterns,
    const float* __restrict__ codebook,    // unused inside the kernel; kept for binding-layout parity with Vulkan
    const float* __restrict__ boundaries,  // 7 boundaries for 3-bit quantization
    int kv_dim, int head_dim, int position,
    int max_seq_len, int num_kv_heads, int block_bytes)
{
    __shared__ float sdata[256];
    __shared__ int   sidx[256];
    __shared__ float warp_sums[8];
    __shared__ float snorm;

    int kv_head = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    if (kv_head >= num_kv_heads || tid >= head_dim) return;

    int head_offset = kv_head * head_dim;
    long byte_offset = (long)position * (long)num_kv_heads * (long)block_bytes
                     + (long)kv_head * (long)block_bytes;
    long base_uint = byte_offset >> 2;
    int half_dim = head_dim >> 1;

    float bnd0 = boundaries[0];
    float bnd1 = boundaries[1];
    float bnd2 = boundaries[2];
    float bnd3 = boundaries[3];
    float bnd4 = boundaries[4];
    float bnd5 = boundaries[5];
    float bnd6 = boundaries[6];

    // Process K then V using the same shared-memory scratch.
    for (int iter = 0; iter < 2; iter++) {
        const float* input_buf = (iter == 0) ? k_input : v_input;
        unsigned int* cache_buf = (iter == 0) ? k_cache_tq : v_cache_tq;

        sdata[tid] = input_buf[head_offset + tid];
        __syncthreads();

        // In-place WHT butterfly.
        for (int s = half_dim; s >= 1; s >>= 1) {
            if (tid < half_dim) {
                int pos_lo = (tid / s) * (s << 1) + (tid % s);
                int pos_hi = pos_lo + s;
                float a = sdata[pos_lo];
                float b = sdata[pos_hi];
                sdata[pos_lo] = a + b;
                sdata[pos_hi] = a - b;
            }
            __syncthreads();
        }

        // Normalize (1/sqrt(dim)) and apply per-head sign flip.
        float scale = rsqrtf((float)head_dim);
        float v = sdata[tid] * scale * sign_patterns[head_offset + tid];
        sdata[tid] = v;

        // Parallel L2 norm reduction (squared sum), warp then inter-warp.
        float sq = v * v;
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        sq += __shfl_xor_sync(0xffffffffu, sq,  8);
        sq += __shfl_xor_sync(0xffffffffu, sq,  4);
        sq += __shfl_xor_sync(0xffffffffu, sq,  2);
        sq += __shfl_xor_sync(0xffffffffu, sq,  1);

        int warp_id = tid >> 5;
        int lane    = tid & 31;
        if (lane == 0) warp_sums[warp_id] = sq;
        __syncthreads();

        if (tid == 0) {
            int n_warps = head_dim >> 5;
            float total = 0.f;
            for (int w = 0; w < n_warps; w++) total += warp_sums[w];
            snorm = sqrtf(total);
        }
        __syncthreads();

        float norm = snorm;
        float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;
        float normalized = v * inv_norm;

        // 7 boundaries → 8 bins.
        int bin = 0;
        if (normalized >= bnd0) bin = 1;
        if (normalized >= bnd1) bin = 2;
        if (normalized >= bnd2) bin = 3;
        if (normalized >= bnd3) bin = 4;
        if (normalized >= bnd4) bin = 5;
        if (normalized >= bnd5) bin = 6;
        if (normalized >= bnd6) bin = 7;
        sidx[tid] = bin;
        __syncthreads();

        // Thread 0 packs FP16 norm + head_dim × 3-bit indices into block_bytes.
        // For head_dim=128 the block fits in 13 uint32 words (52 bytes).
        if (tid == 0) {
            unsigned int num_uints = (unsigned int)((block_bytes + 3) >> 2);
            unsigned int packed[20];
            for (unsigned int w = 0; w < num_uints; w++) packed[w] = 0;

            unsigned int norm_bits = sharpi_fp32_to_fp16(norm) & 0xFFFFu;
            packed[0] = norm_bits;   // bytes 0..1 of the block

            unsigned int bit_pos = 16;   // skip the 2-byte FP16 norm
            for (int i = 0; i < head_dim; i++) {
                unsigned int index3 = ((unsigned int)sidx[i]) & 0x7u;
                unsigned int word_idx = bit_pos >> 5;
                unsigned int bit_off  = bit_pos & 31u;
                packed[word_idx] |= (index3 << bit_off);
                if (bit_off > 29u) {
                    packed[word_idx + 1u] |= (index3 >> (32u - bit_off));
                }
                bit_pos += 3;
            }

            for (unsigned int w = 0; w < num_uints; w++)
                cache_buf[base_uint + (long)w] = packed[w];
        }
        __syncthreads();
    }
}

// ── TurboQuant attention (hybrid TQ + FP32) ───────────────────────────────
// One workgroup per query head, 256 threads. Computes scaled dot-product
// attention across the TQ-compressed history plus the FP32 ring-buffer window.
//
// Score storage strategy:
//   • Fast path  (total_seq ≤ MAX_SHARED_SCORES = 4096): use __shared__ scores[].
//     Hot data stays in shared memory; per-position K is dequantized once.
//   • Slow path  (total_seq > 4096): use a caller-provided global VRAM scratch
//     buffer `scores_scratch[h * max_seq_len .. h*max_seq_len + total_seq)`.
//     Same single-decompress-per-position semantics — global scores reads after
//     phase 1 land in L1, so the perf hit relative to shared mem is small.
//     A previous design tried a triple-pass recompute (re-derive K dot for every
//     output dim) which gave O(N²·head_dim) per token and was hours slow at 8K;
//     never again.
//
// scores_scratch may be a null pointer when the caller knows total_seq ≤
// MAX_SHARED_SCORES — the fast path won't touch it.
extern ""C"" __global__ void llm_tq_attention(
    const float* __restrict__ q,
    const float* __restrict__ rotated_q,
    const unsigned int* __restrict__ k_cache_tq,
    const unsigned int* __restrict__ v_cache_tq,
    const float* __restrict__ k_cache_fp32,
    const float* __restrict__ v_cache_fp32,
    float* __restrict__ out,
    const float* __restrict__ codebook,    // 8 centroids for 3-bit
    float* __restrict__ scores_scratch,    // [num_heads * max_seq_len], null when total_seq ≤ MAX_SHARED_SCORES
    int num_heads, int num_kv_heads, int head_dim,
    int tq_seq_len, int fp32_seq_len, int max_seq_len, int block_bytes)
{
    const int MAX_SHARED_SCORES = 4096;
    // Max block size: 2 (norm) + ceil(256*3/8) = 98 bytes → 25 uints. Round up to 26.
    const int MAX_BLOCK_UINTS = 26;
    __shared__ float shared_scores[MAX_SHARED_SCORES];
    __shared__ float sdata[256];
    __shared__ float cbook[8];
    // Per-warp staging for the (K, t) block. Each warp owns one t at a time
    // (8 warps × 256 lanes = 8 t's per round), reading 13-25 uints once
    // cooperatively then driving its 32-lane dot-product from shared memory.
    // Replaces the previous per-thread layout where each lane read the K block
    // from scratch for ITS own t — that pattern fully serialized bit-unpack
    // work behind uncoalesced global loads.
    __shared__ unsigned int k_block_uints[8][MAX_BLOCK_UINTS];

    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;          // 0..7
    int lane = tid & 31;          // 0..31
    if (h >= num_heads) return;

    if (tid < 8) cbook[tid] = codebook[tid];

    int kv_head = h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = rsqrtf((float)head_dim);
    long q_off  = (long)h * (long)head_dim;
    long rq_off = q_off;
    long out_off = q_off;
    int total_seq = tq_seq_len + fp32_seq_len;

    long block_uints      = (long)block_bytes >> 2;
    long row_stride_uints = (long)num_kv_heads * block_uints;
    long head_uint_offset = (long)kv_head * block_uints;

    // scores points at the right storage for this block. `head_scratch` is the
    // global slot for this query head; only used when we don't fit in shared.
    bool use_shared = (total_seq <= MAX_SHARED_SCORES);
    float* head_scratch = scores_scratch + (long)h * (long)max_seq_len;

    __syncthreads();   // wait for cbook[] population

    // ────────────────────────────────────────────────────────────────────
    //  Phase 1: write per-position scores to scores buffer (shared or global)
    // ────────────────────────────────────────────────────────────────────

    // 1a — TQ-compressed positions, warp-cooperative.
    //
    // Each round processes 8 positions in parallel (one per warp). Within a warp:
    //   • the 32 lanes load the t-block's 13-25 uints once into shared memory,
    //   • each lane then decodes head_dim/32 indices from that shared block and
    //     contributes a partial dot,
    //   • a 5-stage __shfl_xor warp reduction collapses the 32 partials, and
    //   • lane 0 writes the score.
    // No __syncthreads is needed — every cross-thread interaction stays inside
    // the warp. The previous per-thread structure was bottlenecked by
    // uncoalesced k_cache_tq reads (256 threads, 256 unrelated blocks); the
    // cooperative load amortizes those reads across the warp.
    for (int t_base = 0; t_base < tq_seq_len; t_base += 8) {
        int t = t_base + warp;
        if (t < tq_seq_len) {
            long base_uint = (long)t * row_stride_uints + head_uint_offset;

            for (int w = lane; w < (int)block_uints; w += 32)
                k_block_uints[warp][w] = k_cache_tq[base_uint + (long)w];
            __syncwarp();

            float norm = sharpi_fp16_to_fp32(k_block_uints[warp][0] & 0xFFFFu);

            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += 32) {
                unsigned int bit_pos = 16u + (unsigned int)d * 3u;
                unsigned int word_idx = bit_pos >> 5;
                unsigned int bit_off  = bit_pos & 31u;
                unsigned int raw = k_block_uints[warp][word_idx] >> bit_off;
                if (bit_off > 29u)
                    raw |= k_block_uints[warp][word_idx + 1u] << (32u - bit_off);
                int idx = (int)(raw & 0x7u);
                partial += cbook[idx] * rotated_q[rq_off + d];
            }

            unsigned int mask = 0xFFFFFFFFu;
            partial += __shfl_xor_sync(mask, partial, 16);
            partial += __shfl_xor_sync(mask, partial, 8);
            partial += __shfl_xor_sync(mask, partial, 4);
            partial += __shfl_xor_sync(mask, partial, 2);
            partial += __shfl_xor_sync(mask, partial, 1);

            if (lane == 0) {
                float score = partial * norm * scale;
                if (use_shared) shared_scores[t] = score;
                else            head_scratch[t]  = score;
            }
        }
    }

    // 1b — FP32 recent window.
    for (int t = tid; t < fp32_seq_len; t += 256) {
        float dot = 0.0f;
        long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int d = 0; d < head_dim; d++)
            dot += q[q_off + d] * k_cache_fp32[k_off + d];
        float score = dot * scale;
        if (use_shared) shared_scores[tq_seq_len + t] = score;
        else            head_scratch[tq_seq_len + t]  = score;
    }

    // Pad shared scores with -inf so the max scan doesn't pick up garbage.
    // Global scratch doesn't need padding — the scans iterate only [0, total_seq).
    if (use_shared) {
        for (int t = total_seq + tid; t < MAX_SHARED_SCORES; t += 256)
            shared_scores[t] = sharpi_neg_inf();
    }
    __syncthreads();

    // ────────────────────────────────────────────────────────────────────
    //  Phase 2: in-place softmax over [0, total_seq)
    // ────────────────────────────────────────────────────────────────────

    // Max.
    float local_max = sharpi_neg_inf();
    for (int t = tid; t < total_seq; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        local_max = fmaxf(local_max, s);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if ((unsigned int)tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Exp + sum.
    float local_sum = 0.0f;
    for (int t = tid; t < total_seq; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        float e = __expf(s - max_val);
        if (use_shared) shared_scores[t] = e;
        else            head_scratch[t]  = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if ((unsigned int)tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    // Normalize → softmax weight per position.
    for (int t = tid; t < total_seq; t += 256) {
        if (use_shared) shared_scores[t] *= inv_sum;
        else            head_scratch[t]  *= inv_sum;
    }
    __syncthreads();

    // ────────────────────────────────────────────────────────────────────
    //  Phase 3: weighted V sum into output[head, :]
    //  Each thread owns output dim `d`, iterates positions, reads weight[t]
    //  and the V value at (t, d). K is NOT re-decompressed here.
    // ────────────────────────────────────────────────────────────────────
    for (int d = tid; d < head_dim; d += 256) {
        float acc = 0.0f;

        // TQ-compressed positions.
        for (int t = 0; t < tq_seq_len; t++) {
            float weight = use_shared ? shared_scores[t] : head_scratch[t];

            long base_uint = (long)t * row_stride_uints + head_uint_offset;
            float norm = sharpi_fp16_to_fp32(v_cache_tq[base_uint] & 0xFFFFu);

            unsigned int bit_pos = 16u + (unsigned int)d * 3u;
            unsigned int word_idx = bit_pos >> 5;
            unsigned int bit_off  = bit_pos & 31u;
            unsigned int raw = v_cache_tq[base_uint + (long)word_idx] >> bit_off;
            if (bit_off > 29u)
                raw |= v_cache_tq[base_uint + (long)word_idx + 1L] << (32u - bit_off);
            int idx = (int)(raw & 0x7u);

            acc += weight * cbook[idx] * norm;
        }

        // FP32 recent-window positions.
        for (int t = 0; t < fp32_seq_len; t++) {
            float weight = use_shared
                ? shared_scores[tq_seq_len + t]
                : head_scratch[tq_seq_len + t];
            long v_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += weight * v_cache_fp32[v_off + d];
        }

        out[out_off + d] = acc;
    }
}

// ── Scaled dot-product attention with GQA ─────────────────────────────────
// One block per query head, 256 threads.
//
// Score storage strategy (mirrors `llm_tq_attention`):
//   • Fast path (seq_len ≤ MAX_STORED_SCORES = 4096): keep stored scores in
//     `__shared__ float scores[]`.
//   • Slow path (seq_len > MAX_STORED_SCORES): spill per-position scores to a
//     caller-provided global VRAM scratch buffer
//     `scores_scratch[h * max_seq_len .. h*max_seq_len + seq_len)`. Each
//     position is still scored exactly once — global loads in phase 3 hit L2
//     after phase 1, so the slowdown vs shared memory is modest.
//     The previous triple-pass recompute path re-derived every Q·K dot in
//     phase 3, costing O(seq_len × head_dim²) per token; never again.
//
// `scores_scratch` may be a null pointer when the caller knows
// `seq_len ≤ MAX_STORED_SCORES`.
//
// Q layout:        [num_heads * head_dim]                — single token query
// K_cache layout:  [max_seq_len * (num_kv_heads * head_dim)]
// V_cache layout:  same as K_cache
// Output layout:   [num_heads * head_dim]
extern ""C"" __global__ void llm_attention(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    float* __restrict__ scores_scratch,    // [num_heads * max_seq_len], null when seq_len ≤ MAX_STORED_SCORES
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int max_seq_len, float attn_scale)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    if ((int)h >= num_heads) return;

    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    long q_off  = (long)h * (long)head_dim;
    long out_off = q_off;

    bool use_shared = (seq_len <= MAX_STORED_SCORES);
    float* head_scratch = scores_scratch + (long)h * (long)max_seq_len;

    // ────────────────────────────────────────────────────────────────────
    //  Phase 1: write per-position scores to scores buffer (shared or global)
    // ────────────────────────────────────────────────────────────────────
    for (int t = (int)tid; t < seq_len; t += 256) {
        float dot = 0.f;
        long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int d = 0; d < head_dim; d++)
            dot += q[q_off + d] * k_cache[k_off + d];
        float score = dot * scale;
        if (use_shared) shared_scores[t] = score;
        else            head_scratch[t]  = score;
    }
    // Pad shared scores tail with -inf so the max scan skips stale slots.
    // Global scratch doesn't need padding — scans iterate only [0, seq_len).
    if (use_shared) {
        for (int t = seq_len + (int)tid; t < MAX_STORED_SCORES; t += 256)
            shared_scores[t] = sharpi_neg_inf();
    }
    __syncthreads();

    // ────────────────────────────────────────────────────────────────────
    //  Phase 2: in-place softmax over [0, seq_len)
    // ────────────────────────────────────────────────────────────────────
    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < seq_len; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        local_max = fmaxf(local_max, s);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < seq_len; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        float e = __expf(s - max_val);
        if (use_shared) shared_scores[t] = e;
        else            head_scratch[t]  = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < seq_len; t += 256) {
        if (use_shared) shared_scores[t] *= inv_sum;
        else            head_scratch[t]  *= inv_sum;
    }
    __syncthreads();

    // ────────────────────────────────────────────────────────────────────
    //  Phase 3: weighted V sum. K is NOT re-derived here.
    // ────────────────────────────────────────────────────────────────────
    for (int d = (int)tid; d < head_dim; d += 256) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++) {
            float weight = use_shared ? shared_scores[t] : head_scratch[t];
            long v_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += weight * v_cache[v_off + d];
        }
        out[out_off + d] = acc;
    }
}

// ── Scaled dot-product attention with GQA + sliding window ────────────────
// Bit-for-bit clone of `llm_attention` except positions iterated are
// [window_start, window_end) instead of [0, seq_len). Used by Gemma 4 SWA
// layers (window=512 over a possibly much longer context). All three phases
// (Q·K, softmax, V-weighted sum) operate over `eff_seq = window_end -
// window_start` positions; the shared-scores fast path still applies when
// eff_seq ≤ MAX_STORED_SCORES, indexing scores by `t - window_start`.
extern ""C"" __global__ void llm_attention_swa(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    float* __restrict__ scores_scratch,    // [num_heads * max_seq_len], null when eff_seq ≤ MAX_STORED_SCORES
    int num_heads, int num_kv_heads, int head_dim,
    int window_start, int window_end, int max_seq_len, float attn_scale)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    if ((int)h >= num_heads) return;

    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    long q_off  = (long)h * (long)head_dim;
    long out_off = q_off;

    int eff_seq = window_end - window_start;
    if (eff_seq <= 0) return;
    bool use_shared = (eff_seq <= MAX_STORED_SCORES);
    float* head_scratch = scores_scratch + (long)h * (long)max_seq_len;

    // ────────────────────────────────────────────────────────────────────
    //  Phase 1: per-position scores over [window_start, window_end)
    // ────────────────────────────────────────────────────────────────────
    for (int t = (int)tid; t < eff_seq; t += 256) {
        int abs_t = t + window_start;
        float dot = 0.f;
        // Ring slot `abs_t % max_seq_len` (max_seq_len = allocated cache size): identity
        // for a full cache, wraps a window-sized SWA ring. abs_t itself stays logical.
        long k_off = (long)(abs_t % max_seq_len) * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int d = 0; d < head_dim; d++)
            dot += q[q_off + d] * k_cache[k_off + d];
        float score = dot * scale;
        if (use_shared) shared_scores[t] = score;
        else            head_scratch[t]  = score;
    }
    if (use_shared) {
        for (int t = eff_seq + (int)tid; t < MAX_STORED_SCORES; t += 256)
            shared_scores[t] = sharpi_neg_inf();
    }
    __syncthreads();

    // ────────────────────────────────────────────────────────────────────
    //  Phase 2: softmax over [0, eff_seq)
    // ────────────────────────────────────────────────────────────────────
    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < eff_seq; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        local_max = fmaxf(local_max, s);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < eff_seq; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        float e = __expf(s - max_val);
        if (use_shared) shared_scores[t] = e;
        else            head_scratch[t]  = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < eff_seq; t += 256) {
        if (use_shared) shared_scores[t] *= inv_sum;
        else            head_scratch[t]  *= inv_sum;
    }
    __syncthreads();

    // ────────────────────────────────────────────────────────────────────
    //  Phase 3: weighted V sum over windowed positions
    // ────────────────────────────────────────────────────────────────────
    for (int d = (int)tid; d < head_dim; d += 256) {
        float acc = 0.f;
        for (int t = 0; t < eff_seq; t++) {
            int abs_t = t + window_start;
            float weight = use_shared ? shared_scores[t] : head_scratch[t];
            long v_off = (long)(abs_t % max_seq_len) * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += weight * v_cache[v_off + d];
        }
        out[out_off + d] = acc;
    }
}

// ── mma.sync m16n8k16 fragment-layout validation (issue #146) ──────────────
// Single-warp test harness for the int8/fp16 tensor-core building block used by
// the TC flash-attention prefill. One block / 32 threads computes C[16x8] =
// A[16x16] · B[16x8] on the tensor cores (fp16 multiplicands, fp32 accumulate)
// and writes C back in (row,col) order so a host unit test can compare against a
// CPU fp32 matmul. Validating the A/B/C fragment→lane→register maps in isolation
// (PTX ISA m16n8k16 .f16/.f32 tables) de-risks the full kernel — a wrong mapping
// silently produces garbage. a_in is row-major [16*16], b_in is row-major K-major
// [16*8] (b_in[k*8+n] = B[k][n]), c_out is row-major [16*8].
extern ""C"" __global__ void llm_mma_test_m16n8k16_f32(
    const float* __restrict__ a_in,   // [16*16] row-major A
    const float* __restrict__ b_in,   // [16*8]  K-major  B (b_in[k*8+n])
    float* __restrict__ c_out)        // [16*8]  row-major C = A·B
{
    int lane = (int)(threadIdx.x & 31);
    int gid  = lane >> 2;     // groupID 0..7
    int tig  = lane & 3;      // threadID_in_group 0..3

    // A fragment (16x16 fp16, row-major): 4 regs, each a column-pair {2c,2c+1}.
    unsigned int a0 = sharpi_f32x2_to_f16x2(a_in[(gid    ) * 16 + (2 * tig    )], a_in[(gid    ) * 16 + (2 * tig + 1)]);
    unsigned int a1 = sharpi_f32x2_to_f16x2(a_in[(gid + 8) * 16 + (2 * tig    )], a_in[(gid + 8) * 16 + (2 * tig + 1)]);
    unsigned int a2 = sharpi_f32x2_to_f16x2(a_in[(gid    ) * 16 + (2 * tig + 8)], a_in[(gid    ) * 16 + (2 * tig + 9)]);
    unsigned int a3 = sharpi_f32x2_to_f16x2(a_in[(gid + 8) * 16 + (2 * tig + 8)], a_in[(gid + 8) * 16 + (2 * tig + 9)]);

    // B fragment (16x8 fp16, col-major): 2 regs, each a K-row-pair of column `gid`.
    unsigned int b0 = sharpi_f32x2_to_f16x2(b_in[(2 * tig    ) * 8 + gid], b_in[(2 * tig + 1) * 8 + gid]);
    unsigned int b1 = sharpi_f32x2_to_f16x2(b_in[(2 * tig + 8) * 8 + gid], b_in[(2 * tig + 9) * 8 + gid]);

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    asm volatile(
        ""mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ""
        ""{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};""
        : ""+f""(c0), ""+f""(c1), ""+f""(c2), ""+f""(c3)
        : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));

    // C fragment (16x8 fp32): c0=(gid,2tig) c1=(gid,2tig+1) c2=(gid+8,2tig) c3=(gid+8,2tig+1).
    c_out[(gid    ) * 8 + (2 * tig    )] = c0;
    c_out[(gid    ) * 8 + (2 * tig + 1)] = c1;
    c_out[(gid + 8) * 8 + (2 * tig    )] = c2;
    c_out[(gid + 8) * 8 + (2 * tig + 1)] = c3;
}

// ── Tensor-core flash-attention prefill (issue #146) ───────────────────────
// Full TC version of llm_flash_attn_prefill_f32: both QK^T and P·V run on the
// tensor cores (mma.sync m16n8k16, fp16 multiplicands / fp32 accumulate). One
// warp owns a 16-query tile and streams the keys in 16-key tiles with an online
// softmax. The elegant part: the QK^T score C-fragment is reused *directly* as
// the P·V A-fragment — the key index is QK^T's N-column AND P·V's contraction
// dim, and the m16n8k16 C and A fragment layouts coincide on (row, 2·tig), so no
// transpose / shared round-trip is needed for P. O[16×head_dim] is too large for
// registers (head_dim/8 × 4 = 256 regs/lane at d=512), so it lives in shared fp32
// and is rescaled in place per key-tile. K and V time-share one 16×head_dim fp16
// shared buffer (K is fully consumed by QK^T before V is staged for P·V), so the
// whole kernel fits 16·head_dim·(4+2) = 48 KB at d=512. Requires head_dim%16==0.
// Matches the scalar kernels to fp tolerance (online softmax + fp16 Q/K/V/P), not
// bit-exact. GQA, causal, optional sliding window, per-layer head_dim.
__device__ __forceinline__ float fatc_mask(float s, int qpos, int abs_k, int window_size)
{
    bool ok = (abs_k <= qpos) && (window_size <= 0 || abs_k >= qpos + 1 - window_size);
    return ok ? s : sharpi_neg_inf();
}
#define FATC_KT 16
extern ""C"" __global__ void llm_flash_attn_prefill_tc(
    const float* __restrict__ q_all,      // [n_tok, num_heads*head_dim]
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out_all,          // [n_tok, num_heads*head_dim]
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int window_size, int max_seq_len, int n_tok, float attn_scale)
{
    extern __shared__ unsigned int fatc_smem[];
    float*          sO  = (float*)fatc_smem;                       // [16 * head_dim] fp32
    unsigned short* sKV = (unsigned short*)(sO + 16 * head_dim);   // [16 * head_dim] fp16

    int lane = (int)(threadIdx.x & 31);
    int gid  = lane >> 2;     // 0..7  (query-row base; also the N-col 0..7)
    int tig  = lane & 3;      // 0..3
    int h    = (int)blockIdx.x;
    int qtile0 = (int)blockIdx.y * 16;

    int kv_head = h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    int q_dim   = num_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    int nchunk  = head_dim >> 4;   // d / 16  (QK^T contract chunks)
    int ndtile  = head_dim >> 3;   // d / 8   (P·V head-dim N-tiles)

    int q0 = qtile0 + gid;          // this lane's two query rows
    int q1 = qtile0 + gid + 8;
    int qpos0 = start_pos + q0;
    int qpos1 = start_pos + q1;

    for (int idx = lane; idx < 16 * head_dim; idx += 32) sO[idx] = 0.f;

    float m0 = sharpi_neg_inf(), l0 = 0.f;   // running max/sum, query row gid
    float m1 = sharpi_neg_inf(), l1 = 0.f;   // query row gid+8

    // Union key range over the 16 queries in this tile (causal + optional window).
    int last_q = qtile0 + 15; if (last_q > n_tok - 1) last_q = n_tok - 1;
    int key_end = start_pos + last_q + 1;
    int first_qpos = start_pos + qtile0;
    int key_start = (window_size > 0) ? (first_qpos + 1 - window_size) : 0;
    if (key_start < 0) key_start = 0;
    key_start = (key_start / FATC_KT) * FATC_KT;   // align down to a key tile
    __syncthreads();

    for (int kt0 = key_start; kt0 < key_end; kt0 += FATC_KT) {
        // Stage K-tile [16 keys × head_dim] → sKV (fp16). Out-of-range keys load 0
        // (later masked to -inf by the causal bound).
        for (int idx = lane; idx < FATC_KT * head_dim; idx += 32) {
            int kk = idx / head_dim, d = idx - kk * head_dim;
            int abs_k = kt0 + kk;
            // Cache read at ring slot `abs_k % max_seq_len` (identity for a full cache,
            // wraps a window-sized SWA ring); abs_k stays logical for the causal bound.
            float kv = (abs_k < key_end)
                ? k_cache[(long)(abs_k % max_seq_len) * kv_dim + (long)kv_head * head_dim + d] : 0.f;
            sKV[idx] = (unsigned short)sharpi_fp32_to_fp16(kv);
        }
        __syncthreads();

        // ── QK^T → S[16q × 16k], two N=8 key sub-tiles (nt0: keys 0-7, nt1: 8-15) ──
        float s0[4] = {0.f,0.f,0.f,0.f};   // nt0 C-frag c0..c3
        float s1[4] = {0.f,0.f,0.f,0.f};   // nt1 C-frag
        for (int dc = 0; dc < nchunk; dc++) {
            int d0 = dc * 16;
            long qb = (long)h * head_dim + d0;
            // A = Q frag: rows {gid→q0, gid+8→q1}, cols {2tig,2tig+1,2tig+8,2tig+9}.
            float qa0l = (q0 < n_tok) ? q_all[(long)q0 * q_dim + qb + 2*tig    ] : 0.f;
            float qa0h = (q0 < n_tok) ? q_all[(long)q0 * q_dim + qb + 2*tig + 1] : 0.f;
            float qa1l = (q1 < n_tok) ? q_all[(long)q1 * q_dim + qb + 2*tig    ] : 0.f;
            float qa1h = (q1 < n_tok) ? q_all[(long)q1 * q_dim + qb + 2*tig + 1] : 0.f;
            float qa2l = (q0 < n_tok) ? q_all[(long)q0 * q_dim + qb + 2*tig + 8] : 0.f;
            float qa2h = (q0 < n_tok) ? q_all[(long)q0 * q_dim + qb + 2*tig + 9] : 0.f;
            float qa3l = (q1 < n_tok) ? q_all[(long)q1 * q_dim + qb + 2*tig + 8] : 0.f;
            float qa3h = (q1 < n_tok) ? q_all[(long)q1 * q_dim + qb + 2*tig + 9] : 0.f;
            unsigned int a0 = sharpi_f32x2_to_f16x2(qa0l, qa0h);
            unsigned int a1 = sharpi_f32x2_to_f16x2(qa1l, qa1h);
            unsigned int a2 = sharpi_f32x2_to_f16x2(qa2l, qa2h);
            unsigned int a3 = sharpi_f32x2_to_f16x2(qa3l, qa3h);
            // B = K frag (col-major; col = key within sub-tile, rows = contract d).
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                int kbase = (nt * 8 + gid) * head_dim + d0;   // key (nt*8+gid)
                unsigned int b0 = ((unsigned int)sKV[kbase + 2*tig + 1] << 16) | (unsigned int)sKV[kbase + 2*tig    ];
                unsigned int b1 = ((unsigned int)sKV[kbase + 2*tig + 9] << 16) | (unsigned int)sKV[kbase + 2*tig + 8];
                float* s = nt ? s1 : s0;
                asm volatile(
                    ""mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ""
                    ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
                    : ""+f""(s[0]), ""+f""(s[1]), ""+f""(s[2]), ""+f""(s[3])
                    : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            }
        }

        // Scale + causal/window mask. C-frag: s0[c0]=S[q0,kt0+2tig], s0[c1]=S[q0,kt0+2tig+1],
        // s0[c2]=S[q1,kt0+2tig], s0[c3]=S[q1,kt0+2tig+1]; s1 keys shifted by +8.
        s0[0] = fatc_mask(s0[0]*scale, qpos0, kt0 + 2*tig,     window_size);
        s0[1] = fatc_mask(s0[1]*scale, qpos0, kt0 + 2*tig + 1, window_size);
        s0[2] = fatc_mask(s0[2]*scale, qpos1, kt0 + 2*tig,     window_size);
        s0[3] = fatc_mask(s0[3]*scale, qpos1, kt0 + 2*tig + 1, window_size);
        s1[0] = fatc_mask(s1[0]*scale, qpos0, kt0 + 2*tig + 8, window_size);
        s1[1] = fatc_mask(s1[1]*scale, qpos0, kt0 + 2*tig + 9, window_size);
        s1[2] = fatc_mask(s1[2]*scale, qpos1, kt0 + 2*tig + 8, window_size);
        s1[3] = fatc_mask(s1[3]*scale, qpos1, kt0 + 2*tig + 9, window_size);

        // Online softmax, per query row. Row gid uses {s0[0],s0[1],s1[0],s1[1]} (this
        // lane's 4 keys) reduced across the 4 lanes of the group (tig). Row gid+8 uses
        // {s0[2],s0[3],s1[2],s1[3]}.
        float tmax0 = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1]));
        float tmax1 = fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3]));
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 1));
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 2));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 1));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 2));
        float mnew0 = fmaxf(m0, tmax0);
        float mnew1 = fmaxf(m1, tmax1);
        // A query row whose every key in this tile is masked (sliding window: a later
        // query's early tiles fall entirely outside its window) keeps tmax = -inf; if
        // no valid key has been seen yet mnew is also -inf, so exp(m-mnew)=exp(-inf+inf)
        // is NaN. Guard: when mnew is -inf, skip the update (alpha=1 leaves O=0 intact,
        // probabilities are 0). Once any valid key exists, mnew is finite and a masked
        // score (-inf) safely yields exp(-inf - finite) = 0.
        bool ok0 = mnew0 > sharpi_neg_inf();
        bool ok1 = mnew1 > sharpi_neg_inf();
        float alpha0 = ok0 ? __expf(m0 - mnew0) : 1.f;
        float alpha1 = ok1 ? __expf(m1 - mnew1) : 1.f;
        // Probabilities (unnormalized). Masked (-inf) or no-valid-key → 0.
        float p0[4], p1[4];
        p0[0] = ok0 ? __expf(s0[0] - mnew0) : 0.f; p0[1] = ok0 ? __expf(s0[1] - mnew0) : 0.f;
        p0[2] = ok1 ? __expf(s0[2] - mnew1) : 0.f; p0[3] = ok1 ? __expf(s0[3] - mnew1) : 0.f;
        p1[0] = ok0 ? __expf(s1[0] - mnew0) : 0.f; p1[1] = ok0 ? __expf(s1[1] - mnew0) : 0.f;
        p1[2] = ok1 ? __expf(s1[2] - mnew1) : 0.f; p1[3] = ok1 ? __expf(s1[3] - mnew1) : 0.f;
        float lt0 = p0[0] + p0[1] + p1[0] + p1[1];
        float lt1 = p0[2] + p0[3] + p1[2] + p1[3];
        lt0 += __shfl_xor_sync(0xffffffffu, lt0, 1); lt0 += __shfl_xor_sync(0xffffffffu, lt0, 2);
        lt1 += __shfl_xor_sync(0xffffffffu, lt1, 1); lt1 += __shfl_xor_sync(0xffffffffu, lt1, 2);
        l0 = l0 * alpha0 + lt0;
        l1 = l1 * alpha1 + lt1;
        m0 = mnew0; m1 = mnew1;

        // P A-fragment (fp16), reusing the score layout directly (no transpose):
        // a0=(gid; keys 2tig,2tig+1) a1=(gid+8; …) a2=(gid; keys 8+2tig,…) a3=(gid+8; …).
        unsigned int pa0 = sharpi_f32x2_to_f16x2(p0[0], p0[1]);
        unsigned int pa1 = sharpi_f32x2_to_f16x2(p0[2], p0[3]);
        unsigned int pa2 = sharpi_f32x2_to_f16x2(p1[0], p1[1]);
        unsigned int pa3 = sharpi_f32x2_to_f16x2(p1[2], p1[3]);

        // Stage V over the (now-consumed) K buffer.
        __syncthreads();
        for (int idx = lane; idx < FATC_KT * head_dim; idx += 32) {
            int kk = idx / head_dim, d = idx - kk * head_dim;
            int abs_k = kt0 + kk;
            float vv = (abs_k < key_end)
                ? v_cache[(long)(abs_k % max_seq_len) * kv_dim + (long)kv_head * head_dim + d] : 0.f;
            sKV[idx] = (unsigned short)sharpi_fp32_to_fp16(vv);
        }
        __syncthreads();

        // ── P·V → O[16q × head_dim], rescaling shared O in place per N=8 d-tile ──
        for (int dt = 0; dt < ndtile; dt++) {
            int dbase = dt * 8 + gid;   // V col = head-dim index dt*8+gid
            unsigned int b0 = ((unsigned int)sKV[(2*tig + 1) * head_dim + dbase] << 16) | (unsigned int)sKV[(2*tig    ) * head_dim + dbase];
            unsigned int b1 = ((unsigned int)sKV[(2*tig + 9) * head_dim + dbase] << 16) | (unsigned int)sKV[(2*tig + 8) * head_dim + dbase];
            float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
            asm volatile(
                ""mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ""
                ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
                : ""+f""(o0), ""+f""(o1), ""+f""(o2), ""+f""(o3)
                : ""r""(pa0), ""r""(pa1), ""r""(pa2), ""r""(pa3), ""r""(b0), ""r""(b1));
            // o0=O[gid][dt*8+2tig] o1=O[gid][dt*8+2tig+1] o2=O[gid+8][…] o3=O[gid+8][…+1].
            int c0 = gid * head_dim + dt*8 + 2*tig;
            int c1 = c0 + 1;
            int c2 = (gid + 8) * head_dim + dt*8 + 2*tig;
            int c3 = c2 + 1;
            sO[c0] = sO[c0] * alpha0 + o0;
            sO[c1] = sO[c1] * alpha0 + o1;
            sO[c2] = sO[c2] * alpha1 + o2;
            sO[c3] = sO[c3] * alpha1 + o3;
        }
        __syncthreads();
    }

    // Normalize and write out. The 4 lanes of a group share (gid, l0, l1) so they
    // cooperatively stride the head dim by 4 (tig).
    float inv0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    float inv1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    for (int d = tig; d < head_dim; d += 4) {
        if (q0 < n_tok) out_all[(long)q0 * q_dim + (long)h * head_dim + d] = sO[gid * head_dim + d] * inv0;
        if (q1 < n_tok) out_all[(long)q1 * q_dim + (long)h * head_dim + d] = sO[(gid + 8) * head_dim + d] * inv1;
    }
}
#undef FATC_KT

// ── Multi-warp / d-split tensor-core flash-attention prefill (issue #147) ───
// Fixes the single-warp llm_flash_attn_prefill_tc occupancy limit (1 warp/block +
// 48 KB shared → ~2 warps/SM, measured on RTX 4070 Ti / Ada). Here a block is W
// warps that cooperate on ONE 16-query tile, splitting the head dim: warp w owns
// output columns [w·dW, …) with dW = head_dim/W, so O[16×dW] stays REGISTER-resident
// (16×128 = 64 regs/lane at d=512,W=4) instead of in shared — no per-key-tile
// shared-O rescale traffic, and the freed shared lets occupancy rise to ~16-20
// warps/SM (RTX 4070 Ti / Ada). Each warp computes a
// PARTIAL QK^T over its d-slice; the partials are summed across warps through a
// small shared S buffer ([W×16×16] fp32), after which every warp holds the full
// reduced score tile in its C-fragment and proceeds exactly like the single-warp
// kernel (in-warp softmax, no-transpose score→P, P·V for its d-slice). Requires
// head_dim % (W·16) == 0. Shared = 16·head_dim·2 (K/V fp16) + W·256·4 (S) B.
#define FATC2_W 4
#define FATC2_KT 16
// Issue #179: templated K/V dtype (KV = float for the fp32 cache, unsigned short for
// the bf16 cache, block_q8_0 for the q8_0 cache). The body is unchanged bar the two
// cache-load sites, which go through sharpi_kvload — so the float instantiation is
// byte-identical to the pre-#179 kernel. Three extern ""C"" thunks below give NVRTC
// stable entry points (fp32 / bf16 / q8_0).
template<typename KV>
__device__ __forceinline__ void sharpi_flash_attn_prefill_tc2_impl(
    const float* __restrict__ q_all,
    const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache,
    float* __restrict__ out_all,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int window_size, int max_seq_len, int n_tok, float attn_scale)
{
    extern __shared__ unsigned int fatc2_smem[];
    unsigned short* sKV = (unsigned short*)fatc2_smem;          // [16 * head_dim] fp16
    float*          sS  = (float*)(sKV + 16 * head_dim);        // [W * 16 * 16] fp32

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;          // 0..W-1  (d-slice owner)
    int lane = tid & 31;
    int gid  = lane >> 2;         // 0..7
    int tig  = lane & 3;          // 0..3
    int h    = (int)blockIdx.x;
    int qtile0 = (int)blockIdx.y * 16;

    int kv_head = h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    int q_dim   = num_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    int dW      = head_dim / FATC2_W;     // this warp's head-dim slice width
    int d_off   = warp * dW;              // first dim this warp owns
    int nchunk  = dW >> 4;                // dW / 16  (QK^T contract chunks for the slice)
    int ndt     = dW >> 3;                // dW / 8   (P·V N-tiles for the slice)

    int q0 = qtile0 + gid;
    int q1 = qtile0 + gid + 8;
    int qpos0 = start_pos + q0;
    int qpos1 = start_pos + q1;

    // Register-resident O for this warp's d-slice: ndt N-tiles × 4 (c0..c3) fp32.
    float oacc[64];
    #pragma unroll
    for (int i = 0; i < 64; i++) oacc[i] = 0.f;

    float m0 = sharpi_neg_inf(), l0 = 0.f;
    float m1 = sharpi_neg_inf(), l1 = 0.f;

    int last_q = qtile0 + 15; if (last_q > n_tok - 1) last_q = n_tok - 1;
    int key_end = start_pos + last_q + 1;
    int first_qpos = start_pos + qtile0;
    int key_start = (window_size > 0) ? (first_qpos + 1 - window_size) : 0;
    if (key_start < 0) key_start = 0;
    key_start = (key_start / FATC2_KT) * FATC2_KT;
    __syncthreads();

    for (int kt0 = key_start; kt0 < key_end; kt0 += FATC2_KT) {
        // Stage K-tile [16 × head_dim] → sKV (all warps cooperate over the full tile).
        for (int idx = tid; idx < FATC2_KT * head_dim; idx += FATC2_W * 32) {
            int kk = idx / head_dim, d = idx - kk * head_dim;
            int abs_k = kt0 + kk;
            // Cache read at ring slot `abs_k % max_seq_len` (identity for a full cache,
            // wraps a window-sized SWA ring); abs_k stays logical for the causal bound.
            float kv = (abs_k < key_end)
                ? sharpi_kvload(k_cache, (long)(abs_k % max_seq_len) * kv_dim + (long)kv_head * head_dim + d) : 0.f;
            sKV[idx] = (unsigned short)sharpi_fp32_to_fp16(kv);
        }
        __syncthreads();

        // ── Partial QK^T over this warp's d-slice → s0/s1 (C-frag, 2 key sub-tiles) ──
        float s0[4] = {0.f,0.f,0.f,0.f};
        float s1[4] = {0.f,0.f,0.f,0.f};
        for (int dc = 0; dc < nchunk; dc++) {
            int d0 = d_off + dc * 16;           // absolute head dim of this chunk
            long qb = (long)h * head_dim + d0;
            float qa0l = (q0 < n_tok) ? q_all[(long)q0 * q_dim + qb + 2*tig    ] : 0.f;
            float qa0h = (q0 < n_tok) ? q_all[(long)q0 * q_dim + qb + 2*tig + 1] : 0.f;
            float qa1l = (q1 < n_tok) ? q_all[(long)q1 * q_dim + qb + 2*tig    ] : 0.f;
            float qa1h = (q1 < n_tok) ? q_all[(long)q1 * q_dim + qb + 2*tig + 1] : 0.f;
            float qa2l = (q0 < n_tok) ? q_all[(long)q0 * q_dim + qb + 2*tig + 8] : 0.f;
            float qa2h = (q0 < n_tok) ? q_all[(long)q0 * q_dim + qb + 2*tig + 9] : 0.f;
            float qa3l = (q1 < n_tok) ? q_all[(long)q1 * q_dim + qb + 2*tig + 8] : 0.f;
            float qa3h = (q1 < n_tok) ? q_all[(long)q1 * q_dim + qb + 2*tig + 9] : 0.f;
            unsigned int a0 = sharpi_f32x2_to_f16x2(qa0l, qa0h);
            unsigned int a1 = sharpi_f32x2_to_f16x2(qa1l, qa1h);
            unsigned int a2 = sharpi_f32x2_to_f16x2(qa2l, qa2h);
            unsigned int a3 = sharpi_f32x2_to_f16x2(qa3l, qa3h);
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                int kbase = (nt * 8 + gid) * head_dim + d0;
                unsigned int b0 = ((unsigned int)sKV[kbase + 2*tig + 1] << 16) | (unsigned int)sKV[kbase + 2*tig    ];
                unsigned int b1 = ((unsigned int)sKV[kbase + 2*tig + 9] << 16) | (unsigned int)sKV[kbase + 2*tig + 8];
                float* s = nt ? s1 : s0;
                asm volatile(
                    ""mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ""
                    ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
                    : ""+f""(s[0]), ""+f""(s[1]), ""+f""(s[2]), ""+f""(s[3])
                    : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            }
        }

        // ── Reduce partial scores across warps via shared, then read the full S back ──
        // Each warp writes its C-frag to sS[warp][q][k]; lane (gid,tig) of every warp
        // owns the same (q,k) cells, so after the barrier each lane sums over warps.
        float* sw = sS + warp * 256;
        sw[(gid    ) * 16 + (2*tig    )] = s0[0];
        sw[(gid    ) * 16 + (2*tig + 1)] = s0[1];
        sw[(gid + 8) * 16 + (2*tig    )] = s0[2];
        sw[(gid + 8) * 16 + (2*tig + 1)] = s0[3];
        sw[(gid    ) * 16 + (8 + 2*tig    )] = s1[0];
        sw[(gid    ) * 16 + (8 + 2*tig + 1)] = s1[1];
        sw[(gid + 8) * 16 + (8 + 2*tig    )] = s1[2];
        sw[(gid + 8) * 16 + (8 + 2*tig + 1)] = s1[3];
        __syncthreads();
        // Sum the W partials (start from this warp's own value, add the others).
        #pragma unroll
        for (int ww = 1; ww < FATC2_W; ww++) {
            int o = ((warp + ww) % FATC2_W) * 256;
            s0[0] += sS[o + (gid    )*16 + 2*tig    ];
            s0[1] += sS[o + (gid    )*16 + 2*tig + 1];
            s0[2] += sS[o + (gid + 8)*16 + 2*tig    ];
            s0[3] += sS[o + (gid + 8)*16 + 2*tig + 1];
            s1[0] += sS[o + (gid    )*16 + 8 + 2*tig    ];
            s1[1] += sS[o + (gid    )*16 + 8 + 2*tig + 1];
            s1[2] += sS[o + (gid + 8)*16 + 8 + 2*tig    ];
            s1[3] += sS[o + (gid + 8)*16 + 8 + 2*tig + 1];
        }

        // Scale + causal/window mask (full reduced scores now).
        s0[0] = fatc_mask(s0[0]*scale, qpos0, kt0 + 2*tig,     window_size);
        s0[1] = fatc_mask(s0[1]*scale, qpos0, kt0 + 2*tig + 1, window_size);
        s0[2] = fatc_mask(s0[2]*scale, qpos1, kt0 + 2*tig,     window_size);
        s0[3] = fatc_mask(s0[3]*scale, qpos1, kt0 + 2*tig + 1, window_size);
        s1[0] = fatc_mask(s1[0]*scale, qpos0, kt0 + 2*tig + 8, window_size);
        s1[1] = fatc_mask(s1[1]*scale, qpos0, kt0 + 2*tig + 9, window_size);
        s1[2] = fatc_mask(s1[2]*scale, qpos1, kt0 + 2*tig + 8, window_size);
        s1[3] = fatc_mask(s1[3]*scale, qpos1, kt0 + 2*tig + 9, window_size);

        // Online softmax (per query row; reduce the 4 keys across the group's 4 lanes).
        float tmax0 = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1]));
        float tmax1 = fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3]));
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 1));
        tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 2));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 1));
        tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 2));
        float mnew0 = fmaxf(m0, tmax0);
        float mnew1 = fmaxf(m1, tmax1);
        bool ok0 = mnew0 > sharpi_neg_inf();
        bool ok1 = mnew1 > sharpi_neg_inf();
        float alpha0 = ok0 ? __expf(m0 - mnew0) : 1.f;
        float alpha1 = ok1 ? __expf(m1 - mnew1) : 1.f;
        float p0[4], p1[4];
        p0[0] = ok0 ? __expf(s0[0] - mnew0) : 0.f; p0[1] = ok0 ? __expf(s0[1] - mnew0) : 0.f;
        p0[2] = ok1 ? __expf(s0[2] - mnew1) : 0.f; p0[3] = ok1 ? __expf(s0[3] - mnew1) : 0.f;
        p1[0] = ok0 ? __expf(s1[0] - mnew0) : 0.f; p1[1] = ok0 ? __expf(s1[1] - mnew0) : 0.f;
        p1[2] = ok1 ? __expf(s1[2] - mnew1) : 0.f; p1[3] = ok1 ? __expf(s1[3] - mnew1) : 0.f;
        float lt0 = p0[0] + p0[1] + p1[0] + p1[1];
        float lt1 = p0[2] + p0[3] + p1[2] + p1[3];
        lt0 += __shfl_xor_sync(0xffffffffu, lt0, 1); lt0 += __shfl_xor_sync(0xffffffffu, lt0, 2);
        lt1 += __shfl_xor_sync(0xffffffffu, lt1, 1); lt1 += __shfl_xor_sync(0xffffffffu, lt1, 2);
        l0 = l0 * alpha0 + lt0;
        l1 = l1 * alpha1 + lt1;
        m0 = mnew0; m1 = mnew1;

        unsigned int pa0 = sharpi_f32x2_to_f16x2(p0[0], p0[1]);
        unsigned int pa1 = sharpi_f32x2_to_f16x2(p0[2], p0[3]);
        unsigned int pa2 = sharpi_f32x2_to_f16x2(p1[0], p1[1]);
        unsigned int pa3 = sharpi_f32x2_to_f16x2(p1[2], p1[3]);

        // Stage V over the consumed K buffer.
        __syncthreads();
        for (int idx = tid; idx < FATC2_KT * head_dim; idx += FATC2_W * 32) {
            int kk = idx / head_dim, d = idx - kk * head_dim;
            int abs_k = kt0 + kk;
            float vv = (abs_k < key_end)
                ? sharpi_kvload(v_cache, (long)(abs_k % max_seq_len) * kv_dim + (long)kv_head * head_dim + d) : 0.f;
            sKV[idx] = (unsigned short)sharpi_fp32_to_fp16(vv);
        }
        __syncthreads();

        // ── P·V over this warp's d-slice → register O, rescaled in place ──
        for (int dt = 0; dt < ndt; dt++) {
            int dcol = d_off + dt * 8 + gid;     // absolute head dim (V col)
            unsigned int b0 = ((unsigned int)sKV[(2*tig + 1) * head_dim + dcol] << 16) | (unsigned int)sKV[(2*tig    ) * head_dim + dcol];
            unsigned int b1 = ((unsigned int)sKV[(2*tig + 9) * head_dim + dcol] << 16) | (unsigned int)sKV[(2*tig + 8) * head_dim + dcol];
            float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
            asm volatile(
                ""mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ""
                ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
                : ""+f""(o0), ""+f""(o1), ""+f""(o2), ""+f""(o3)
                : ""r""(pa0), ""r""(pa1), ""r""(pa2), ""r""(pa3), ""r""(b0), ""r""(b1));
            int base = dt * 4;
            oacc[base + 0] = oacc[base + 0] * alpha0 + o0;   // O[gid][dcol_2tig]
            oacc[base + 1] = oacc[base + 1] * alpha0 + o1;
            oacc[base + 2] = oacc[base + 2] * alpha1 + o2;   // O[gid+8]
            oacc[base + 3] = oacc[base + 3] * alpha1 + o3;
        }
        __syncthreads();
    }

    // Normalize and write this warp's d-slice from registers. oacc[dt] cells map to
    // O[q=gid/gid+8][d_off + dt*8 + {2tig,2tig+1}].
    float inv0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    float inv1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    for (int dt = 0; dt < ndt; dt++) {
        int d = d_off + dt * 8 + 2*tig;
        int base = dt * 4;
        if (q0 < n_tok) {
            out_all[(long)q0 * q_dim + (long)h * head_dim + d    ] = oacc[base + 0] * inv0;
            out_all[(long)q0 * q_dim + (long)h * head_dim + d + 1] = oacc[base + 1] * inv0;
        }
        if (q1 < n_tok) {
            out_all[(long)q1 * q_dim + (long)h * head_dim + d    ] = oacc[base + 2] * inv1;
            out_all[(long)q1 * q_dim + (long)h * head_dim + d + 1] = oacc[base + 3] * inv1;
        }
    }
}
#undef FATC2_W
#undef FATC2_KT

// fp32 / bf16 / q8_0 cache entry points (issue #179). The fp32 thunk is the original
// #147 kernel; the bf16 thunk reads a half-width K/V cache; the q8_0 thunk reads a
// block-quantized (~quarter) K/V cache. All decode each element to fp32 on load.
extern ""C"" __global__ void llm_flash_attn_prefill_tc2(
    const float* __restrict__ q_all,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out_all,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int window_size, int max_seq_len, int n_tok, float attn_scale)
{
    sharpi_flash_attn_prefill_tc2_impl<float>(q_all, k_cache, v_cache, out_all,
        num_heads, num_kv_heads, head_dim, start_pos, window_size, max_seq_len, n_tok, attn_scale);
}

extern ""C"" __global__ void llm_flash_attn_prefill_tc2_bf16(
    const float* __restrict__ q_all,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ out_all,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int window_size, int max_seq_len, int n_tok, float attn_scale)
{
    sharpi_flash_attn_prefill_tc2_impl<unsigned short>(q_all, k_cache, v_cache, out_all,
        num_heads, num_kv_heads, head_dim, start_pos, window_size, max_seq_len, n_tok, attn_scale);
}

extern ""C"" __global__ void llm_flash_attn_prefill_tc2_q8_0(
    const float* __restrict__ q_all,
    const block_q8_0* __restrict__ k_cache,
    const block_q8_0* __restrict__ v_cache,
    float* __restrict__ out_all,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int window_size, int max_seq_len, int n_tok, float attn_scale)
{
    sharpi_flash_attn_prefill_tc2_impl<block_q8_0>(q_all, k_cache, v_cache, out_all,
        num_heads, num_kv_heads, head_dim, start_pos, window_size, max_seq_len, n_tok, attn_scale);
}

// ── Flash-attention prefill (issue #141 attention) ─────────────────────────
// Memory-efficient batched SDPA replacing the scalar llm_full_seq_attention /
// llm_attention_swa_batched (one 256-thread block PER query, each re-reading its
// whole K/V range from global → O(n²) global traffic; for SWA, adjacent queries'
// 512-wide windows overlap ~99%, so K/V is re-streamed up to ~512×). Here a block
// handles a TILE of FA_QT queries of one head and streams K/V in shared-memory
// tiles of `kt_tile` keys, so each key is read from global once per FA_QT queries
// (FA_QT× less traffic) and the softmax runs online (running max/sum + rescaled
// output accumulator, FlashAttention-2 style) — no n²-sized score buffer.
//
// One warp = one query. Lane L owns head dims {L, L+32, …}; qreg/oreg hold up to
// 512/32 = 16 dims/lane. The QK dot is a warp reduce (score is then warp-uniform,
// so the causal/window mask `continue` and the online-softmax update never diverge
// within a warp). GQA via kv_head = h/(num_heads/num_kv_heads); per-layer head_dim;
// causal (key ≤ start_pos+qi) + optional sliding window (window_size>0). Matches the
// scalar kernels to fp tolerance (online softmax reassociates the same sum), not
// bit-exact. Dynamic shared = 2*kt_tile*head_dim floats (sized by the host to fit).
// FA_QT warps/block = queries sharing each K/V tile load (the reuse factor).
#define FA_QT 16
extern ""C"" __global__ void llm_flash_attn_prefill_f32(
    const float* __restrict__ q_all,      // [n_tok, num_heads*head_dim]
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out_all,          // [n_tok, num_heads*head_dim]
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int window_size, int max_seq_len, int n_tok,
    int kt_tile, float attn_scale)
{
    // Dynamic shared: K as fp16 (half2-packed, hd/2 uints/key) then V as fp32. K is
    // read with the half2 QK dot (fp16-rounded inputs — argmax-stable); V stays fp32
    // for the exact scalar PV. Each lane owns dim-PAIRS pi = lane+32*p (dims 2·pi,
    // 2·pi+1) so the shared half2 loads stay coalesced. host sizes 6·kt_tile·head_dim B.
    extern __shared__ unsigned int fa_smem[];
    int hd2 = head_dim >> 1;                       // pairs per key
    unsigned int* sKh = fa_smem;                   // [kt_tile * hd2] fp16x2
    float* sV = (float*)(fa_smem + kt_tile * hd2); // [kt_tile * head_dim] fp32

    int h     = (int)blockIdx.x;
    int warp  = (int)(threadIdx.x >> 5);  // 0..FA_QT-1  (query within the tile)
    int lane  = (int)(threadIdx.x & 31);
    int tid   = (int)threadIdx.x;
    int qi    = (int)blockIdx.y * FA_QT + warp;

    int kv_head = h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    int q_dim   = num_heads * head_dim;
    int ndj2    = (hd2 + 31) >> 5;        // dim-pairs per lane (≤8)

    bool active = (qi < n_tok);
    int qpos    = start_pos + qi;
    int win_end = qpos + 1;                                  // keys [.., qpos]
    int win_start = (window_size > 0) ? (win_end - window_size) : 0;
    if (win_start < 0) win_start = 0;

    unsigned int qh2[8];                 // this query's dim-pairs, fp16x2-packed
    float oreg[16];                      // PV accumulator, fp32, 2 per pair
    #pragma unroll
    for (int p = 0; p < 8; p++) { qh2[p] = 0u; oreg[2 * p] = 0.f; oreg[2 * p + 1] = 0.f; }
    if (active) {
        long q_base = (long)qi * q_dim + (long)h * head_dim;
        for (int p = 0; p < ndj2; p++) {
            int pi = lane + 32 * p;
            if (pi < hd2) qh2[p] = sharpi_f32x2_to_f16x2(q_all[q_base + 2 * pi], q_all[q_base + 2 * pi + 1]);
        }
    }
    float m_run = sharpi_neg_inf();
    float l_run = 0.f;

    // Union key range over the FA_QT queries in this block.
    int last_qi = (int)blockIdx.y * FA_QT + (FA_QT - 1);
    if (last_qi > n_tok - 1) last_qi = n_tok - 1;
    int blk_key_end = (start_pos + last_qi) + 1;
    int first_qpos  = start_pos + (int)blockIdx.y * FA_QT;
    int blk_key_start = (window_size > 0) ? (first_qpos + 1 - window_size) : 0;
    if (blk_key_start < 0) blk_key_start = 0;

    for (int kt0 = blk_key_start; kt0 < blk_key_end; kt0 += kt_tile) {
        int tile_keys = blk_key_end - kt0;
        if (tile_keys > kt_tile) tile_keys = kt_tile;

        // Stage K (fp32 global → fp16x2 shared, one pair/thread-step).
        for (int idx = tid; idx < kt_tile * hd2; idx += (int)blockDim.x) {
            int kk = idx / hd2, pr = idx - kk * hd2;
            unsigned int kh = 0u;
            if (kk < tile_keys) {
                // Ring slot `(kt0+kk) % max_seq_len`: identity for a full cache, wraps a
                // window-sized SWA ring. The kt0+kk index stays logical for tile bounds.
                long off = (long)((kt0 + kk) % max_seq_len) * kv_dim + (long)kv_head * head_dim + 2 * pr;
                kh = sharpi_f32x2_to_f16x2(k_cache[off], k_cache[off + 1]);
            }
            sKh[idx] = kh;
        }
        // Stage V (fp32 → fp32 shared).
        for (int idx = tid; idx < kt_tile * head_dim; idx += (int)blockDim.x) {
            int kk = idx / head_dim, d = idx - kk * head_dim;
            sV[idx] = (kk < tile_keys)
                ? v_cache[(long)((kt0 + kk) % max_seq_len) * kv_dim + (long)kv_head * head_dim + d]
                : 0.f;
        }
        __syncthreads();

        if (active) {
            for (int kk = 0; kk < tile_keys; kk++) {
                int abs_t = kt0 + kk;
                if (abs_t >= win_end || abs_t < win_start) continue;  // warp-uniform
                unsigned int acc = 0u;
                for (int p = 0; p < ndj2; p++) {
                    int pi = lane + 32 * p;
                    if (pi < hd2) acc = sharpi_hfma2(qh2[p], sKh[kk * hd2 + pi], acc);
                }
                float part = sharpi_f16x2_sum(acc);
                part += __shfl_xor_sync(0xffffffffu, part, 16);
                part += __shfl_xor_sync(0xffffffffu, part, 8);
                part += __shfl_xor_sync(0xffffffffu, part, 4);
                part += __shfl_xor_sync(0xffffffffu, part, 2);
                part += __shfl_xor_sync(0xffffffffu, part, 1);
                float score = part * scale;
                float m_new = fmaxf(m_run, score);
                float alpha = __expf(m_run - m_new);
                float pw    = __expf(score - m_new);
                l_run = l_run * alpha + pw;
                for (int p = 0; p < ndj2; p++) {
                    int pi = lane + 32 * p;
                    float v0 = (pi < hd2) ? sV[kk * head_dim + 2 * pi]     : 0.f;
                    float v1 = (pi < hd2) ? sV[kk * head_dim + 2 * pi + 1] : 0.f;
                    oreg[2 * p]     = oreg[2 * p]     * alpha + pw * v0;
                    oreg[2 * p + 1] = oreg[2 * p + 1] * alpha + pw * v1;
                }
                m_run = m_new;
            }
        }
        __syncthreads();
    }

    if (active) {
        float inv = (l_run > 0.f) ? (1.f / l_run) : 0.f;
        long o_base = (long)qi * q_dim + (long)h * head_dim;
        for (int p = 0; p < ndj2; p++) {
            int pi = lane + 32 * p;
            if (pi < hd2) {
                out_all[o_base + 2 * pi]     = oreg[2 * p]     * inv;
                out_all[o_base + 2 * pi + 1] = oreg[2 * p + 1] * inv;
            }
        }
    }
}
#undef FA_QT

// Batched sliding-window attention over N query tokens (Gemma 4 SWA layers in
// batched-trunk prefill). Grid = (num_heads, n_tok); query token i sits at
// absolute position start_pos+i and attends [max(0,pos+1-window), pos+1). The
// window bounds eff_seq ≤ window_size, so the shared-scores path always suffices
// (window_size ≤ MAX_STORED_SCORES required by the dispatch). Per (head, token)
// this is bit-identical to the per-token llm_attention_swa.
extern ""C"" __global__ void llm_attention_swa_batched(
    const float* __restrict__ q_all,      // [n_tok, num_heads*head_dim]
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out_all,          // [n_tok, num_heads*head_dim]
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int window_size, int max_seq_len, int n_tok, float attn_scale)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    int i = (int)blockIdx.y;
    if ((int)h >= num_heads || i >= n_tok) return;

    int window_end = start_pos + i + 1;
    int window_start = window_end - window_size;
    if (window_start < 0) window_start = 0;
    int eff_seq = window_end - window_start;
    if (eff_seq <= 0) return;

    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    int q_dim = num_heads * head_dim;
    const float* q = q_all + (long)i * q_dim;
    float* out = out_all + (long)i * q_dim;
    long q_off = (long)h * (long)head_dim;
    long out_off = q_off;

    for (int t = (int)tid; t < eff_seq; t += 256) {
        int abs_t = t + window_start;
        float dot = 0.f;
        // Ring slot `abs_t % max_seq_len` (max_seq_len = allocated cache size): identity
        // for a full cache, wraps a window-sized SWA ring. abs_t itself stays logical.
        long k_off = (long)(abs_t % max_seq_len) * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int dd = 0; dd < head_dim; dd++)
            dot += q[q_off + dd] * k_cache[k_off + dd];
        shared_scores[t] = dot * scale;
    }
    __syncthreads();

    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < eff_seq; t += 256)
        local_max = fmaxf(local_max, shared_scores[t]);
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < eff_seq; t += 256) {
        float ev = __expf(shared_scores[t] - max_val);
        shared_scores[t] = ev;
        local_sum += ev;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < eff_seq; t += 256)
        shared_scores[t] *= inv_sum;
    __syncthreads();

    for (int dd = (int)tid; dd < head_dim; dd += 256) {
        float acc = 0.f;
        for (int t = 0; t < eff_seq; t++) {
            int abs_t = t + window_start;
            long v_off = (long)(abs_t % max_seq_len) * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += shared_scores[t] * v_cache[v_off + dd];
        }
        out[out_off + dd] = acc;
    }
}

// ── Sliding-window attention, narrowed K/V cache (issues #179 + #27) ─────────
// Narrowed-store variant of `llm_attention_swa`: q, out, score scratch, and all
// softmax arithmetic stay fp32, so precision matches the fp32 SWA kernel and only
// the cache footprint shrinks. Preserves the SWA ring (`abs_t % max_seq_len`) and
// Gemma 4's attn_scale=1.0.
// Issue #179: templated K/V dtype (KV = unsigned short for bf16, block_q8_0 for
// q8_0). Body unchanged bar the two cache loads, which go through sharpi_kvload;
// the bf16 thunk is byte-identical to the pre-q8_0 kernel. extern ""C"" thunks below.
template<typename KV>
__device__ void llm_attention_swa_kv_impl(
    const float* __restrict__ q,
    const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache,
    float* __restrict__ out,
    float* __restrict__ scores_scratch,    // [num_heads * max_seq_len], null when eff_seq ≤ MAX_STORED_SCORES
    int num_heads, int num_kv_heads, int head_dim,
    int window_start, int window_end, int max_seq_len, float attn_scale)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    if ((int)h >= num_heads) return;

    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    long q_off  = (long)h * (long)head_dim;
    long out_off = q_off;

    int eff_seq = window_end - window_start;
    if (eff_seq <= 0) return;
    bool use_shared = (eff_seq <= MAX_STORED_SCORES);
    float* head_scratch = scores_scratch + (long)h * (long)max_seq_len;

    for (int t = (int)tid; t < eff_seq; t += 256) {
        int abs_t = t + window_start;
        long k_off = (long)(abs_t % max_seq_len) * (long)kv_dim + (long)kv_head * (long)head_dim;
        float dot = sharpi_kv_dot(q + q_off, k_cache, k_off, head_dim);
        float score = dot * scale;
        if (use_shared) shared_scores[t] = score;
        else            head_scratch[t]  = score;
    }
    if (use_shared) {
        for (int t = eff_seq + (int)tid; t < MAX_STORED_SCORES; t += 256)
            shared_scores[t] = sharpi_neg_inf();
    }
    __syncthreads();

    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < eff_seq; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        local_max = fmaxf(local_max, s);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < eff_seq; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        float e = __expf(s - max_val);
        if (use_shared) shared_scores[t] = e;
        else            head_scratch[t]  = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < eff_seq; t += 256) {
        if (use_shared) shared_scores[t] *= inv_sum;
        else            head_scratch[t]  *= inv_sum;
    }
    __syncthreads();

    for (int d = (int)tid; d < head_dim; d += 256) {
        float acc = 0.f;
        for (int t = 0; t < eff_seq; t++) {
            int abs_t = t + window_start;
            float weight = use_shared ? shared_scores[t] : head_scratch[t];
            long v_off = (long)(abs_t % max_seq_len) * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += weight * sharpi_kvload(v_cache, v_off + d);
        }
        out[out_off + d] = acc;
    }
}

extern ""C"" __global__ void llm_attention_swa_bf16(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ out,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int window_start, int window_end, int max_seq_len, float attn_scale)
{
    llm_attention_swa_kv_impl<unsigned short>(q, k_cache, v_cache, out, scores_scratch,
        num_heads, num_kv_heads, head_dim, window_start, window_end, max_seq_len, attn_scale);
}

extern ""C"" __global__ void llm_attention_swa_q8_0(
    const float* __restrict__ q,
    const block_q8_0* __restrict__ k_cache,
    const block_q8_0* __restrict__ v_cache,
    float* __restrict__ out,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int window_start, int window_end, int max_seq_len, float attn_scale)
{
    llm_attention_swa_kv_impl<block_q8_0>(q, k_cache, v_cache, out, scores_scratch,
        num_heads, num_kv_heads, head_dim, window_start, window_end, max_seq_len, attn_scale);
}

// ── Batched sliding-window attention, narrowed K/V cache (issues #179 + #27) ─
// Narrowed-store variant of `llm_attention_swa_batched`. One 256-thread block per
// (head, query); K/V decoded to fp32 on load (sharpi_kvload), all arithmetic fp32.
// Per (head, token) the bf16 instantiation matches the per-token llm_attention_swa
// bf16 thunk (modulo store rounding); the q8_0 instantiation matches its q8_0 thunk.
template<typename KV>
__device__ void llm_attention_swa_batched_kv_impl(
    const float* __restrict__ q_all,      // [n_tok, num_heads*head_dim]
    const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache,
    float* __restrict__ out_all,          // [n_tok, num_heads*head_dim]
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int window_size, int max_seq_len, int n_tok, float attn_scale)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    int i = (int)blockIdx.y;
    if ((int)h >= num_heads || i >= n_tok) return;

    int window_end = start_pos + i + 1;
    int window_start = window_end - window_size;
    if (window_start < 0) window_start = 0;
    int eff_seq = window_end - window_start;
    if (eff_seq <= 0) return;

    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    int q_dim = num_heads * head_dim;
    const float* q = q_all + (long)i * q_dim;
    float* out = out_all + (long)i * q_dim;
    long q_off = (long)h * (long)head_dim;
    long out_off = q_off;

    for (int t = (int)tid; t < eff_seq; t += 256) {
        int abs_t = t + window_start;
        float dot = 0.f;
        long k_off = (long)(abs_t % max_seq_len) * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int dd = 0; dd < head_dim; dd++)
            dot += q[q_off + dd] * sharpi_kvload(k_cache, k_off + dd);
        shared_scores[t] = dot * scale;
    }
    __syncthreads();

    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < eff_seq; t += 256)
        local_max = fmaxf(local_max, shared_scores[t]);
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < eff_seq; t += 256) {
        float ev = __expf(shared_scores[t] - max_val);
        shared_scores[t] = ev;
        local_sum += ev;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < eff_seq; t += 256)
        shared_scores[t] *= inv_sum;
    __syncthreads();

    for (int dd = (int)tid; dd < head_dim; dd += 256) {
        float acc = 0.f;
        for (int t = 0; t < eff_seq; t++) {
            int abs_t = t + window_start;
            long v_off = (long)(abs_t % max_seq_len) * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += shared_scores[t] * sharpi_kvload(v_cache, v_off + dd);
        }
        out[out_off + dd] = acc;
    }
}

extern ""C"" __global__ void llm_attention_swa_batched_bf16(
    const float* __restrict__ q_all,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ out_all,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int window_size, int max_seq_len, int n_tok, float attn_scale)
{
    llm_attention_swa_batched_kv_impl<unsigned short>(q_all, k_cache, v_cache, out_all,
        num_heads, num_kv_heads, head_dim, start_pos, window_size, max_seq_len, n_tok, attn_scale);
}

extern ""C"" __global__ void llm_attention_swa_batched_q8_0(
    const float* __restrict__ q_all,
    const block_q8_0* __restrict__ k_cache,
    const block_q8_0* __restrict__ v_cache,
    float* __restrict__ out_all,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int window_size, int max_seq_len, int n_tok, float attn_scale)
{
    llm_attention_swa_batched_kv_impl<block_q8_0>(q_all, k_cache, v_cache, out_all,
        num_heads, num_kv_heads, head_dim, start_pos, window_size, max_seq_len, n_tok, attn_scale);
}

// ── Scaled dot-product attention with GQA (bf16 K/V cache) ─────────────────
// Bit-for-bit copy of `llm_attention` except K/V cache is read as bfloat16
// (stored as raw unsigned short, decoded via sharpi_bf16_to_fp32). Score
// scratch, query, and output stay fp32; softmax accumulates in fp32 too.
// Bf16 → fp32 promotion happens at the dot/weighted-sum read points, so all
// arithmetic precision (and overflow head-room) matches the fp32 kernel —
// only the cache footprint is halved. See issue #27.
template<typename KV>
__device__ void llm_attention_kv_impl(
    const float* __restrict__ q,
    const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache,
    float* __restrict__ out,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int max_seq_len, float attn_scale)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    if ((int)h >= num_heads) return;

    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    // attn_scale > 0 overrides (Gemma 4 passes 1.0); ≤0 uses 1/sqrt(head_dim).
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    long q_off  = (long)h * (long)head_dim;
    long out_off = q_off;

    bool use_shared = (seq_len <= MAX_STORED_SCORES);
    float* head_scratch = scores_scratch + (long)h * (long)max_seq_len;

    for (int t = (int)tid; t < seq_len; t += 256) {
        long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
        float dot = sharpi_kv_dot(q + q_off, k_cache, k_off, head_dim);
        float score = dot * scale;
        if (use_shared) shared_scores[t] = score;
        else            head_scratch[t]  = score;
    }
    if (use_shared) {
        for (int t = seq_len + (int)tid; t < MAX_STORED_SCORES; t += 256)
            shared_scores[t] = sharpi_neg_inf();
    }
    __syncthreads();

    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < seq_len; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        local_max = fmaxf(local_max, s);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < seq_len; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        float e = __expf(s - max_val);
        if (use_shared) shared_scores[t] = e;
        else            head_scratch[t]  = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < seq_len; t += 256) {
        if (use_shared) shared_scores[t] *= inv_sum;
        else            head_scratch[t]  *= inv_sum;
    }
    __syncthreads();

    for (int d = (int)tid; d < head_dim; d += 256) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++) {
            float weight = use_shared ? shared_scores[t] : head_scratch[t];
            long v_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += weight * sharpi_kvload(v_cache, v_off + d);
        }
        out[out_off + d] = acc;
    }
}

extern ""C"" __global__ void llm_attention_bf16(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ out,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int max_seq_len, float attn_scale)
{
    llm_attention_kv_impl<unsigned short>(q, k_cache, v_cache, out, scores_scratch,
        num_heads, num_kv_heads, head_dim, seq_len, max_seq_len, attn_scale);
}

extern ""C"" __global__ void llm_attention_q8_0(
    const float* __restrict__ q,
    const block_q8_0* __restrict__ k_cache,
    const block_q8_0* __restrict__ v_cache,
    float* __restrict__ out,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int max_seq_len, float attn_scale)
{
    llm_attention_kv_impl<block_q8_0>(q, k_cache, v_cache, out, scores_scratch,
        num_heads, num_kv_heads, head_dim, seq_len, max_seq_len, attn_scale);
}

// ── Flash-decoding split-KV: partial attention over a KV chunk (issue #235) ──
// Fixes the decode-attention occupancy collapse: the single-block kernels above
// launch only num_heads blocks (~13% of the SMs at decode), serializing the whole
// O(ctx)/token KV scan. This splits each head's causal sequence [0, seq_len) into
// fixed SPLITKV_CHUNK-sized slices across `num_heads × n_splits` blocks, so the KV
// read parallelizes across the SMs (flash-decoding; PyTorch/Tri-Dao 2023). Scalar
// (not tensor-core) per the GQA-ratio-4 analysis in #235.
//
// One block = (query head h = blockIdx.x, KV split s = blockIdx.y). It handles slice
// [s*CHUNK, min((s+1)*CHUNK, seq_len)) and emits the UN-normalized online-softmax
// partial for that slice: local max m_i, local denom l_i = Σ exp(score−m_i), and the
// numerator Õ_i[d] = Σ exp(score−m_i)·v[d]. `llm_attention_combine` then LSE-merges
// the n_splits partials per head. The grid is FIXED at num_heads × n_splits
// (n_splits = ceil(max_seq_len/CHUNK)) so it is CUDA-graph-capturable with seq_len as
// the only per-replay-updated param; out-of-range splits (s*CHUNK ≥ seq_len) write
// (m=−inf, l=0) and return so a stale partial from a prior replay is never merged.
// K/V are decoded via sharpi_kv_dot / sharpi_kvload → fp32/bf16/q8_0 thunks.
#define SPLITKV_CHUNK 512
template<typename KV>
__device__ void llm_attention_splitkv_impl(
    const float* __restrict__ q,
    const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache,
    float* __restrict__ partial_o,     // [num_heads * n_splits * head_dim]
    float* __restrict__ partial_meta,  // [num_heads * n_splits * 2] : (m_i, l_i)
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int n_splits, float attn_scale)
{
    __shared__ float sk_scores[SPLITKV_CHUNK];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    int h = (int)blockIdx.x;
    int s = (int)blockIdx.y;
    if (h >= num_heads || s >= n_splits) return;

    long meta_off = ((long)h * (long)n_splits + (long)s) * 2;
    int t0 = s * SPLITKV_CHUNK;
    // Out-of-range split (fixed grid, short seq_len): mark empty and bail so the
    // combine skips it (scale = exp(−inf − gmax) = 0) and never reads a stale Õ.
    if (t0 >= seq_len) {
        if (tid == 0) { partial_meta[meta_off] = sharpi_neg_inf(); partial_meta[meta_off + 1] = 0.f; }
        return;
    }
    int t1 = t0 + SPLITKV_CHUNK; if (t1 > seq_len) t1 = seq_len;
    int n = t1 - t0;

    int kv_head = h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    long q_off  = (long)h * (long)head_dim;
    // Per-position row stride and this head's invariant base offset, hoisted out of the loops.
    long kv_dim_l = (long)kv_dim;
    long kv_base  = (long)t0 * kv_dim_l + (long)kv_head * (long)head_dim;

    // Phase 1: scores for the slice → shared (indexed t − t0).
    for (int t = (int)tid; t < n; t += 256) {
        long k_off = kv_base + (long)t * kv_dim_l;
        sk_scores[t] = sharpi_kv_dot(q + q_off, k_cache, k_off, head_dim) * scale;
    }
    __syncthreads();

    // Phase 2: local max over the slice.
    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < n; t += 256) local_max = fmaxf(local_max, sk_scores[t]);
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int r = 128; r > 0; r >>= 1) {
        if (tid < r) sdata[tid] = fmaxf(sdata[tid], sdata[tid + r]);
        __syncthreads();
    }
    float m_i = sdata[0];
    __syncthreads();

    // exp(score − m_i) in place + local denom.
    float local_sum = 0.f;
    for (int t = (int)tid; t < n; t += 256) {
        float e = __expf(sk_scores[t] - m_i);
        sk_scores[t] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int r = 128; r > 0; r >>= 1) {
        if (tid < r) sdata[tid] += sdata[tid + r];
        __syncthreads();
    }
    float l_i = sdata[0];
    __syncthreads();

    if (tid == 0) { partial_meta[meta_off] = m_i; partial_meta[meta_off + 1] = l_i; }

    // Phase 3: UN-normalized weighted-V numerator for this slice (combine divides by Σ).
    long o_off = ((long)h * (long)n_splits + (long)s) * (long)head_dim;
    for (int d = (int)tid; d < head_dim; d += 256) {
        float acc = 0.f;
        for (int t = 0; t < n; t++) {
            long v_off = kv_base + (long)t * kv_dim_l;   // same hoisted base as Phase 1
            acc += sk_scores[t] * sharpi_kvload(v_cache, v_off + d);
        }
        partial_o[o_off + d] = acc;
    }
}

extern ""C"" __global__ void llm_attention_splitkv(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ partial_o,
    float* __restrict__ partial_meta,
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int n_splits, float attn_scale)
{
    llm_attention_splitkv_impl<float>(q, k_cache, v_cache, partial_o, partial_meta,
        num_heads, num_kv_heads, head_dim, seq_len, n_splits, attn_scale);
}

extern ""C"" __global__ void llm_attention_splitkv_bf16(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ partial_o,
    float* __restrict__ partial_meta,
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int n_splits, float attn_scale)
{
    llm_attention_splitkv_impl<unsigned short>(q, k_cache, v_cache, partial_o, partial_meta,
        num_heads, num_kv_heads, head_dim, seq_len, n_splits, attn_scale);
}

extern ""C"" __global__ void llm_attention_splitkv_q8_0(
    const float* __restrict__ q,
    const block_q8_0* __restrict__ k_cache,
    const block_q8_0* __restrict__ v_cache,
    float* __restrict__ partial_o,
    float* __restrict__ partial_meta,
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int n_splits, float attn_scale)
{
    llm_attention_splitkv_impl<block_q8_0>(q, k_cache, v_cache, partial_o, partial_meta,
        num_heads, num_kv_heads, head_dim, seq_len, n_splits, attn_scale);
}

// ── Flash-decoding combine: LSE-merge the n_splits partials per head (#235) ──
// One block per query head; merges the per-slice (m_i, l_i, Õ_i) partials into the
// final attention output with the standard online-softmax rescale:
//   m = max_i m_i ; l = Σ_i exp(m_i−m)·l_i ; O[d] = (Σ_i exp(m_i−m)·Õ_i[d]) / l
// Exact (modulo FP reduction order). Empty splits carry m_i=−inf → scale 0 → skipped.
// SPLITKV_MAX_SPLITS bounds the per-head split count (ceil(131072/512)=256).
#define SPLITKV_MAX_SPLITS 256
extern ""C"" __global__ void llm_attention_combine(
    const float* __restrict__ partial_o,     // [num_heads * n_splits * head_dim]
    const float* __restrict__ partial_meta,  // [num_heads * n_splits * 2]
    float* __restrict__ out,                  // [num_heads * head_dim]
    int num_heads, int head_dim, int n_splits)
{
    __shared__ float sh_scale[SPLITKV_MAX_SPLITS];
    __shared__ float red[256];
    __shared__ float sh_gmax;
    __shared__ float sh_denom;

    unsigned int tid = threadIdx.x;
    int h = (int)blockIdx.x;
    if (h >= num_heads) return;
    long base = (long)h * (long)n_splits;

    // Global max over the splits' local maxima.
    float lmax = sharpi_neg_inf();
    for (int s = (int)tid; s < n_splits; s += 256)
        lmax = fmaxf(lmax, partial_meta[(base + s) * 2]);
    red[tid] = lmax;
    __syncthreads();
    for (unsigned int r = 128; r > 0; r >>= 1) {
        if (tid < r) red[tid] = fmaxf(red[tid], red[tid + r]);
        __syncthreads();
    }
    if (tid == 0) sh_gmax = red[0];
    __syncthreads();
    float gmax = sh_gmax;

    // Per-split rescale factor exp(m_i − m) + global denominator Σ exp(m_i−m)·l_i.
    float ldenom = 0.f;
    for (int s = (int)tid; s < n_splits; s += 256) {
        float m = partial_meta[(base + s) * 2];
        float l = partial_meta[(base + s) * 2 + 1];
        float sc = __expf(m - gmax);
        sh_scale[s] = sc;
        ldenom += sc * l;
    }
    red[tid] = ldenom;
    __syncthreads();
    for (unsigned int r = 128; r > 0; r >>= 1) {
        if (tid < r) red[tid] += red[tid + r];
        __syncthreads();
    }
    if (tid == 0) sh_denom = red[0];
    __syncthreads();
    float inv = 1.0f / sh_denom;

    // Weighted sum of the per-split numerators across head_dim. Base offsets hoisted.
    long head_dim_l = (long)head_dim;
    long po_base = base * head_dim_l;          // first split's row for this head
    long out_base = (long)h * head_dim_l;
    for (int d = (int)tid; d < head_dim; d += 256) {
        float acc = 0.f;
        for (int s = 0; s < n_splits; s++) {
            float sc = sh_scale[s];
            if (sc != 0.f) acc += sc * partial_o[po_base + (long)s * head_dim_l + d];
        }
        out[out_base + d] = acc * inv;
    }
}

// ── Multi-query dot for GQA head-sharing (#237): one K read, GF FMAs ─────────
// GF dots of GF query rows (q_base[g*head_dim + .], g=0..GF-1) against one contiguous
// K row at element offset `off`. Each K element is read ONCE and fused into all GF
// accumulators, so the K HBM read is amortized across the query group (vs GF separate
// sharpi_kv_dot calls each re-reading the row). q8_0 keeps the per-32-block scale cache
// (block-walk identical to sharpi_kv_dot). dots[] must hold ≥ GF (GF ≤ 8, host-enforced).
__device__ __forceinline__ void sharpi_kv_dot_multi(const float* __restrict__ q_base, int head_dim, int gf,
                                                     const float* __restrict__ k, long off, float* dots)
{
    for (int g = 0; g < gf; g++) dots[g] = 0.f;
    for (int d = 0; d < head_dim; d++) {
        float kd = k[off + d];
        for (int g = 0; g < gf; g++) dots[g] += q_base[(long)g * (long)head_dim + d] * kd;
    }
}
__device__ __forceinline__ void sharpi_kv_dot_multi(const float* __restrict__ q_base, int head_dim, int gf,
                                                     const unsigned short* __restrict__ k, long off, float* dots)
{
    for (int g = 0; g < gf; g++) dots[g] = 0.f;
    for (int d = 0; d < head_dim; d++) {
        float kd = sharpi_bf16_to_fp32((unsigned int)k[off + d]);
        for (int g = 0; g < gf; g++) dots[g] += q_base[(long)g * (long)head_dim + d] * kd;
    }
}
__device__ __forceinline__ void sharpi_kv_dot_multi(const float* __restrict__ q_base, int head_dim, int gf,
                                                     const block_q8_0* __restrict__ k, long off, float* dots)
{
    for (int g = 0; g < gf; g++) dots[g] = 0.f;
    long b = off >> 5;
    int lane = (int)(off & 31);
    for (int d = 0; d < head_dim; ) {
        float s = sharpi_fp16_to_fp32((unsigned int)k[b].d);
        for (; lane < 32 && d < head_dim; lane++, d++) {
            float kd = s * (float)k[b].qs[lane];
            for (int g = 0; g < gf; g++) dots[g] += q_base[(long)g * (long)head_dim + d] * kd;
        }
        lane = 0; b++;
    }
}

// ── GQA head-sharing split-KV (#237): one block per (KV head, KV split) ──────
// Variant of llm_attention_splitkv that loads each K/V slice element ONCE and reuses it
// across the G = num_heads/num_kv_heads query heads sharing that KV head, instead of G
// separate blocks each re-reading the slice (the per-head split's G× redundant HBM read,
// which dominates the now-bandwidth-bound long-ctx decode — #235/#237). grid =
// (num_kv_heads, n_splits); each block emits the SAME per-query-head partials the per-head
// kernel does, so llm_attention_combine + the partials layout are unchanged. Dynamic shared
// = G*SPLITKV_CHUNK floats (per-head exp-weights). G ≤ 8 (dots/acc/po_base arrays).
template<typename KV>
__device__ void llm_attention_splitkv_grouped_impl(
    const float* __restrict__ q,
    const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache,
    float* __restrict__ partial_o,     // [num_heads * n_splits * head_dim]
    float* __restrict__ partial_meta,  // [num_heads * n_splits * 2] : (m_i, l_i)
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int n_splits, float attn_scale)
{
    extern __shared__ float g_scores[];   // [G * SPLITKV_CHUNK]
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    int kv = (int)blockIdx.x;
    int s  = (int)blockIdx.y;
    if (kv >= num_kv_heads || s >= n_splits) return;

    int G  = num_heads / num_kv_heads;    // query heads per KV head (group size)
    int h0 = kv * G;                       // first query head of this group
    int t0 = s * SPLITKV_CHUNK;
    // Out-of-range split (fixed grid, short seq_len): mark all G heads' partials empty so
    // the combine skips them (scale = exp(−inf − gmax) = 0) and never reads a stale Õ.
    if (t0 >= seq_len) {
        if ((int)tid < G) {
            long mo = ((long)(h0 + (int)tid) * (long)n_splits + (long)s) * 2;
            partial_meta[mo] = sharpi_neg_inf(); partial_meta[mo + 1] = 0.f;
        }
        return;
    }
    int t1 = t0 + SPLITKV_CHUNK; if (t1 > seq_len) t1 = seq_len;
    int n = t1 - t0;

    int kv_dim = num_kv_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    long kv_dim_l = (long)kv_dim;
    long kv_base  = (long)t0 * kv_dim_l + (long)kv * (long)head_dim;
    const float* q_base = q + (long)h0 * (long)head_dim;
    // Per-head partial-O row base, hoisted out of the Phase-3 loops.
    long po_base[8];
    for (int g = 0; g < G; g++)
        po_base[g] = ((long)(h0 + g) * (long)n_splits + (long)s) * (long)head_dim;

    // Phase 1: G dots per slice position — K row read once, GF FMAs.
    for (int t = (int)tid; t < n; t += 256) {
        float dots[8];
        sharpi_kv_dot_multi(q_base, head_dim, G, k_cache, kv_base + (long)t * kv_dim_l, dots);
        for (int g = 0; g < G; g++) g_scores[g * SPLITKV_CHUNK + t] = dots[g] * scale;
    }
    __syncthreads();

    // Phase 2: per query head — max, exp in place, sum → (m_i, l_i). sdata reused per g.
    for (int g = 0; g < G; g++) {
        float* sc = g_scores + g * SPLITKV_CHUNK;
        float lmax = sharpi_neg_inf();
        for (int t = (int)tid; t < n; t += 256) lmax = fmaxf(lmax, sc[t]);
        sdata[tid] = lmax;
        __syncthreads();
        for (unsigned int r = 128; r > 0; r >>= 1) { if (tid < r) sdata[tid] = fmaxf(sdata[tid], sdata[tid + r]); __syncthreads(); }
        float m_i = sdata[0];
        __syncthreads();
        float lsum = 0.f;
        for (int t = (int)tid; t < n; t += 256) { float e = __expf(sc[t] - m_i); sc[t] = e; lsum += e; }
        sdata[tid] = lsum;
        __syncthreads();
        for (unsigned int r = 128; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid + r]; __syncthreads(); }
        float l_i = sdata[0];
        if (tid == 0) {
            long mo = ((long)(h0 + g) * (long)n_splits + (long)s) * 2;
            partial_meta[mo] = m_i; partial_meta[mo + 1] = l_i;
        }
        __syncthreads();
    }

    // Phase 3: UN-normalized weighted-V numerator — V row read once, GF FMAs.
    for (int d = (int)tid; d < head_dim; d += 256) {
        float acc[8];
        for (int g = 0; g < G; g++) acc[g] = 0.f;
        for (int t = 0; t < n; t++) {
            float vd = sharpi_kvload(v_cache, kv_base + (long)t * kv_dim_l + d);
            for (int g = 0; g < G; g++) acc[g] += g_scores[g * SPLITKV_CHUNK + t] * vd;
        }
        for (int g = 0; g < G; g++) partial_o[po_base[g] + d] = acc[g];
    }
}

extern ""C"" __global__ void llm_attention_splitkv_grouped(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ partial_o,
    float* __restrict__ partial_meta,
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int n_splits, float attn_scale)
{
    llm_attention_splitkv_grouped_impl<float>(q, k_cache, v_cache, partial_o, partial_meta,
        num_heads, num_kv_heads, head_dim, seq_len, n_splits, attn_scale);
}

extern ""C"" __global__ void llm_attention_splitkv_grouped_bf16(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ partial_o,
    float* __restrict__ partial_meta,
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int n_splits, float attn_scale)
{
    llm_attention_splitkv_grouped_impl<unsigned short>(q, k_cache, v_cache, partial_o, partial_meta,
        num_heads, num_kv_heads, head_dim, seq_len, n_splits, attn_scale);
}

extern ""C"" __global__ void llm_attention_splitkv_grouped_q8_0(
    const float* __restrict__ q,
    const block_q8_0* __restrict__ k_cache,
    const block_q8_0* __restrict__ v_cache,
    float* __restrict__ partial_o,
    float* __restrict__ partial_meta,
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int n_splits, float attn_scale)
{
    llm_attention_splitkv_grouped_impl<block_q8_0>(q, k_cache, v_cache, partial_o, partial_meta,
        num_heads, num_kv_heads, head_dim, seq_len, n_splits, attn_scale);
}

// ── SnapKV: per-(query, head) attention scoring against the K cache ────────
// Issue #58. Computes the SnapKV importance signal for ONE captured query
// vector against the layer's K cache. For each head h, masks positions
// p > q_abs_pos, scales dots by rsqrt(head_dim), softmaxes across the valid
// prefix, and atomicAdd's the resulting weights into a global per-position
// score accumulator. Mirrors `llm_attention`'s scratch convention: shared
// memory below 4096 positions, scores_scratch[h * max_seq_len + t] above.
//
// The host loops over the captured W queries × num attention layers,
// accumulating into a single shared float accumulator that's downloaded
// post-pass and fed to SnapKvSelector.SelectKeepSet.
extern ""C"" __global__ void llm_snapkv_score(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    float* __restrict__ score_accum,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int prompt_len, int q_abs_pos, int max_seq_len)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    if ((int)h >= num_heads) return;

    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = rsqrtf((float)head_dim);
    long q_off  = (long)h * (long)head_dim;

    bool use_shared = (prompt_len <= MAX_STORED_SCORES);
    float* head_scratch = scores_scratch + (long)h * (long)max_seq_len;

    for (int t = (int)tid; t < prompt_len; t += 256) {
        float score;
        if (t > q_abs_pos) {
            score = sharpi_neg_inf();
        } else {
            float dot = 0.f;
            long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
            for (int d = 0; d < head_dim; d++)
                dot += q[q_off + d] * k_cache[k_off + d];
            score = dot * scale;
        }
        if (use_shared) shared_scores[t] = score;
        else            head_scratch[t]  = score;
    }
    if (use_shared) {
        for (int t = prompt_len + (int)tid; t < MAX_STORED_SCORES; t += 256)
            shared_scores[t] = sharpi_neg_inf();
    }
    __syncthreads();

    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < prompt_len; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        local_max = fmaxf(local_max, s);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < prompt_len; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        float e = (s == sharpi_neg_inf()) ? 0.f : __expf(s - max_val);
        if (use_shared) shared_scores[t] = e;
        else            head_scratch[t]  = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < prompt_len; t += 256) {
        if (t > q_abs_pos) continue;
        float w = (use_shared ? shared_scores[t] : head_scratch[t]) * inv_sum;
        atomicAdd(&score_accum[t], w);
    }
}

// Bf16-K-cache variant of `llm_snapkv_score`. Reads bf16 K via
// sharpi_bf16_to_fp32; arithmetic stays in fp32. Used when SHARPI bf16-KV is
// enabled (issue #27 / PR #56) — same call site as the fp32 variant.
extern ""C"" __global__ void llm_snapkv_score_bf16(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    float* __restrict__ score_accum,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int prompt_len, int q_abs_pos, int max_seq_len)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    if ((int)h >= num_heads) return;

    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = rsqrtf((float)head_dim);
    long q_off  = (long)h * (long)head_dim;

    bool use_shared = (prompt_len <= MAX_STORED_SCORES);
    float* head_scratch = scores_scratch + (long)h * (long)max_seq_len;

    for (int t = (int)tid; t < prompt_len; t += 256) {
        float score;
        if (t > q_abs_pos) {
            score = sharpi_neg_inf();
        } else {
            float dot = 0.f;
            long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
            for (int d = 0; d < head_dim; d++)
                dot += q[q_off + d] * sharpi_bf16_to_fp32((unsigned int)k_cache[k_off + d]);
            score = dot * scale;
        }
        if (use_shared) shared_scores[t] = score;
        else            head_scratch[t]  = score;
    }
    if (use_shared) {
        for (int t = prompt_len + (int)tid; t < MAX_STORED_SCORES; t += 256)
            shared_scores[t] = sharpi_neg_inf();
    }
    __syncthreads();

    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < prompt_len; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        local_max = fmaxf(local_max, s);
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < prompt_len; t += 256) {
        float s = use_shared ? shared_scores[t] : head_scratch[t];
        float e = (s == sharpi_neg_inf()) ? 0.f : __expf(s - max_val);
        if (use_shared) shared_scores[t] = e;
        else            head_scratch[t]  = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < prompt_len; t += 256) {
        if (t > q_abs_pos) continue;
        float w = (use_shared ? shared_scores[t] : head_scratch[t]) * inv_sum;
        atomicAdd(&score_accum[t], w);
    }
}

// ── SnapKV: gather kept positions into a dense [K * kv_dim] prefix ─────────
// Issue #58. Reads a sorted-ascending keep-position list and copies
// src[keep[i] * kv_dim + d] → dst[i * kv_dim + d]. src and dst MUST point at
// different buffers (a separate stage tensor) — the destination is later
// written back over the ring's [0, K * kv_dim) prefix via CopyDeviceRegion.
// Grid: (ceil(kv_dim / 256), K, 1); Block: (256, 1, 1).
extern ""C"" __global__ void llm_kv_compact(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int* __restrict__ keep_positions,
    int K, int kv_dim)
{
    int i = (int)blockIdx.y;
    if (i >= K) return;
    int d = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= kv_dim) return;
    int src_pos = keep_positions[i];
    long src_off = (long)src_pos * (long)kv_dim + (long)d;
    long dst_off = (long)i       * (long)kv_dim + (long)d;
    dst[dst_off] = src[src_off];
}

// Bf16-store variant of `llm_kv_compact`. Operates on raw unsigned short
// elements — no fp32 round-trip — so the bf16 KV ring's bits survive
// untouched through compaction.
extern ""C"" __global__ void llm_kv_compact_bf16(
    const unsigned short* __restrict__ src,
    unsigned short* __restrict__ dst,
    const int* __restrict__ keep_positions,
    int K, int kv_dim)
{
    int i = (int)blockIdx.y;
    if (i >= K) return;
    int d = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= kv_dim) return;
    int src_pos = keep_positions[i];
    long src_off = (long)src_pos * (long)kv_dim + (long)d;
    long dst_off = (long)i       * (long)kv_dim + (long)d;
    dst[dst_off] = src[src_off];
}

// ── Element-wise SiLU in place ─────────────────────────────────────────────
// x[i] = x[i] / (1 + exp(-x[i])).  One thread per element.
extern ""C"" __global__ void llm_silu_inplace(float* __restrict__ x, int n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float v = x[i];
    x[i] = v / (1.0f + __expf(-v));
}

// ── GDN: causal depthwise conv1d decode (single token) ─────────────────────
// state layout: [(K-1), C] row-major, oldest first.
// Operation:
//   output[c] = weight[K-1,c]*x[c] + Σ_{k=0..K-2} weight[k,c]*state[k,c]
//   shift state: state[0..K-3] = state[1..K-2]; state[K-2] = x[c]
// One thread per channel. Kernel size K ≤ 4 in current models.
extern ""C"" __global__ void llm_gdn_conv1d_decode(
    const float* __restrict__ x,
    float* __restrict__ state,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int channels, int kernel_size)
{
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= channels) return;

    int retained = kernel_size - 1;

    // Read old state values into registers (small: ≤3 for our models).
    float s_old[4];   // sized for K up to 5
    #pragma unroll 4
    for (int k = 0; k < retained; k++)
        s_old[k] = state[(long)k * channels + c];

    float x_c = x[c];
    float sum = weight[(long)retained * channels + c] * x_c;
    #pragma unroll 4
    for (int k = 0; k < retained; k++)
        sum += weight[(long)k * channels + c] * s_old[k];
    output[c] = sum;

    // Shift state forward in time (drop oldest, append x).
    #pragma unroll 4
    for (int k = 0; k < retained - 1; k++)
        state[(long)k * channels + c] = s_old[k + 1];
    if (retained >= 1)
        state[(long)(retained - 1) * channels + c] = x_c;
}

// ── GDN: L2-norm per head (no learned weights) ─────────────────────────────
// Matches GdnKernels.L2NormPerHead: scale = 1 / max(sqrt(Σ x²), eps).
// Differs from llm_head_norm_pure (which divides by sqrt(mean+eps)).
// One block per head; 256 threads.
extern ""C"" __global__ void llm_gdn_l2_norm_per_head(
    float* __restrict__ data,
    int head_dim, int num_heads, float eps)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int head = blockIdx.x;
    if ((int)head >= num_heads) return;

    int base_off = (int)head * head_dim;

    float sum = 0.f;
    for (int i = (int)tid; i < head_dim; i += 256) {
        float v = data[base_off + i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float norm = sqrtf(sdata[0]);
    float divisor = norm > eps ? norm : eps;
    float inv_div = 1.0f / divisor;
    for (int i = (int)tid; i < head_dim; i += 256)
        data[base_off + i] *= inv_div;
}

// ── GDN: tile heads (GQA-style broadcast) ──────────────────────────────────
// Tile pattern: dst[h_dst, j] = src[h_dst % src_heads, j] for h_dst ∈ [0, src_heads*repeat).
// Matches GdnKernels.TileHeads exactly (NOT torch repeat_interleave).
extern ""C"" __global__ void llm_gdn_tile_heads(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int src_heads, int repeat, int head_dim)
{
    int dst_total = src_heads * repeat * head_dim;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= dst_total) return;
    int j = idx % head_dim;
    int h_dst = idx / head_dim;
    int h_src = h_dst % src_heads;
    dst[idx] = src[(long)h_src * head_dim + j];
}

// ── GDN: recurrence decode (single token) ──────────────────────────────────
// Implements one autoregressive step of Gated DeltaNet. Mirrors the CPU
// reference GdnKernels.GdnRecurrenceDecode line-for-line:
//   1. decay  = exp(softplus(alphaIn[h] + dtBias[h]) · ssmA[h])
//   2. b      = sigmoid(beta[h])
//   3. S     *= decay                                              (pass A)
//   4. p[j]   = Σ_i k[i] · S[i,j]                                  (pass A, fused)
//   5. d[j]   = b · (v[j] − p[j])
//   6. S[i,j] += k[i] · d[j]                                       (pass B)
//   7. o[j]   = (1/√d) · Σ_i q[i] · S[i,j]                         (pass B, fused)
//   8. o     = RMSNorm(o) · normWeight                              (with eps floor)
//   9. o    *= SiLU(z)
//
// Layout:
//   • One thread block per v-head h ∈ [0, hv).
//   • blockDim.x = head_dim (one thread per output column j).
//   • Shared memory: 8 × head_dim floats (≈4 KiB for d=128).
//   • State stored row-major: S[h*d*d + i*d + j]; coalesced over j.
extern ""C"" __global__ void llm_gdn_recurrence_decode(
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha_in,
    const float* __restrict__ beta,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    const float* __restrict__ norm_weight,
    const float* __restrict__ z,
    float* __restrict__ output,
    int hv, int d, float norm_eps)
{
    int h = (int)blockIdx.x;
    int j = (int)threadIdx.x;
    if (h >= hv || j >= d) return;

    extern __shared__ float smem[];
    float* sK     = smem;
    float* sQ     = sK + d;
    float* sV     = sQ + d;
    float* sZ     = sV + d;
    float* sNormW = sZ + d;
    float* sP     = sNormW + d;   // shared p[j]
    float* sD     = sP + d;       // shared d[j]
    float* sRed   = sD + d;       // RMSNorm reduction scratch

    // Load per-head Q, K, V, Z and per-dim norm weight into shared memory.
    long hd_off = (long)h * d;
    sK[j]     = k[hd_off + j];
    sQ[j]     = q[hd_off + j];
    sV[j]     = v[hd_off + j];
    sZ[j]     = z[hd_off + j];
    sNormW[j] = norm_weight[j];
    __syncthreads();

    // Per-head scalar gates.
    float alpha_x = alpha_in[h] + dt_bias[h];
    float dt      = alpha_x >= 20.0f ? alpha_x : __logf(1.0f + __expf(alpha_x));
    float decay   = __expf(dt * ssm_a[h]);
    float b_sc    = 1.0f / (1.0f + __expf(-beta[h]));

    long state_base = (long)h * (long)d * (long)d;

    // Pass A: decay S, then accumulate p[j] = Σ_i k[i] · S[i,j].
    float p_local = 0.f;
    for (int i = 0; i < d; i++) {
        long off = state_base + (long)i * d + j;
        float sij = state[off] * decay;
        state[off] = sij;
        p_local += sK[i] * sij;
    }
    sP[j] = p_local;
    __syncthreads();

    // Compute d[j].
    float d_j = b_sc * (sV[j] - sP[j]);
    sD[j] = d_j;
    __syncthreads();

    // Pass B: rank-1 update S[i,j] += k[i] · d[j], fused with readout o[j].
    float o_local = 0.f;
    for (int i = 0; i < d; i++) {
        long off = state_base + (long)i * d + j;
        float sij = state[off] + sK[i] * d_j;
        state[off] = sij;
        o_local += sQ[i] * sij;
    }

    // Scale by 1/sqrt(d).
    o_local *= rsqrtf((float)d);

    // RMSNorm: scale = rsqrt(sumSq/d + eps), then o = o * scale * normWeight.
    sRed[j] = o_local * o_local;
    __syncthreads();
    for (int s = d / 2; s > 0; s >>= 1) {
        if (j < s) sRed[j] += sRed[j + s];
        __syncthreads();
    }
    float scale = rsqrtf(sRed[0] / (float)d + norm_eps);

    float o_normed = o_local * scale * sNormW[j];

    // SiLU(z) gate.
    float zv = sZ[j];
    float silu = zv / (1.0f + __expf(-zv));

    output[hd_off + j] = o_normed * silu;
}

// ── GDN: high-occupancy warp-per-column recurrence decode (single token) ─────
// Issue #404 / Option B: the occupancy-optimized sibling of
// llm_gdn_recurrence_decode, mirroring llama.cpp's gated_delta_net_cuda. The
// original launches only `hv` blocks (one per v-head) — leaving most SMs idle — and
// re-reads the [d×d] state from global memory twice. This launches one WARP PER
// OUTPUT COLUMN (grid (hv, 1, ceil(d/blockDim.y)), block (32, blockDim.y)), so the d
// columns of every head spread across many blocks → full occupancy. Within a token
// each warp holds its column's d state rows in registers (sharded ~d/32 per lane),
// reads the state ONCE, and warp-reduces the two key/query dot products. The decay is
// folded into the kv dot (kv from the OLD state, like llama.cpp) so the rank-1 update
// + readout is a single fused pass with one state write. Because columns now span
// multiple blocks, the per-head ssm_norm (RMSNorm) + SiLU(z) gate the original did
// in-block CANNOT reduce here — it is a SEPARATE launch (llm_gdn_decode_norm_gate)
// over the raw readout. The warp-reduce changes the FP reduction order vs the original
// → argmax-stable, NOT byte-exact (same numeric class as the chunked GDN prefill).
//
// State layout: TRANSPOSED (column-major within head) — state[h*d*d + col*d + i] holds
// S[i][col]. This is what makes the per-lane shard read COALESCED: lane reads rows
// i = r*32+lane of column `col` at consecutive addresses (col*d + r*32 + lane). The
// canonical state is row-major S[i*d+col]; the host transposes a layer in-place
// (llm_gdn_state_transpose) on entry to the fast-decode regime and back before any
// row-major consumer (scan / snapshot). GDN_DECODE_MAX_RPL = 4 supports head_dim ≤ 128
// (the qwen35moe value head_dim); the host wrapper requires headDim % 32 == 0 and ≤ 128.
#define GDN_DECODE_MAX_RPL 4
extern ""C"" __global__ void __launch_bounds__(128, 2) llm_gdn_recurrence_decode_fast(
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ alpha_in,
    const float* __restrict__ beta,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    float* __restrict__ readout,
    int hv, int d)
{
    int h    = (int)blockIdx.x;
    int col  = (int)(blockIdx.z * blockDim.y + threadIdx.y);
    int lane = (int)threadIdx.x;              // 0..31
    if (h >= hv || col >= d) return;          // whole warp shares col ⇒ no intra-warp divergence

    int rpl = (d + 31) / 32;                  // rows per lane (= d/32 since d%32==0; ≤ GDN_DECODE_MAX_RPL)

    // Per-head scalar gates: the softplus/exp/sigmoid chain is identical for every
    // column of a head, so compute it once on lane 0 and broadcast across the warp —
    // doing it per-lane would be 32× redundant SFU work in an otherwise tiny kernel.
    float decay = 0.f, b_sc = 0.f;
    if (lane == 0) {
        float alpha_x = alpha_in[h] + dt_bias[h];
        float dt      = alpha_x >= 20.0f ? alpha_x : __logf(1.0f + __expf(alpha_x));
        decay = __expf(dt * ssm_a[h]);
        b_sc  = 1.0f / (1.0f + __expf(-beta[h]));
    }
    decay = __shfl_sync(0xffffffffu, decay, 0);
    b_sc  = __shfl_sync(0xffffffffu, b_sc, 0);

    long state_base = (long)h * (long)d * (long)d;
    long hd_off     = (long)h * d;

    // Load this warp's column shard of the state + the row-sharded k/q into registers.
    // Transposed state: column `col` is contiguous (col*d + i) ⇒ coalesced across lanes.
    // d%32==0 ⇒ i = r*32+lane < d whenever r < rpl, so no per-element bounds test needed.
    long col_base = state_base + (long)col * d;
    float s_shard[GDN_DECODE_MAX_RPL];
    float k_reg[GDN_DECODE_MAX_RPL];
    float q_reg[GDN_DECODE_MAX_RPL];
    #pragma unroll
    for (int r = 0; r < GDN_DECODE_MAX_RPL; r++) {
        bool ok = (r < rpl);
        // Clamp the index to 0 for inactive register slots: a predicated/select form
        // (`ok ? load : 0`) can still let the compiler emit the load to an out-of-range
        // address for r ≥ rpl (e.g. d=96 ⇒ rpl=3, slot 3 would read past the head). For
        // d=128 (rpl=4) every slot is active, so this is a no-op (i == r*32+lane).
        int i = ok ? (r * 32 + lane) : 0;
        s_shard[r] = ok ? state[col_base + i] : 0.f;
        k_reg[r]   = ok ? k[hd_off + i] : 0.f;
        q_reg[r]   = ok ? q[hd_off + i] : 0.f;
    }

    // Pass A: kv[col] = Σ_i k[i]·S[i,col] over the OLD state; p[col] = decay·kv[col].
    float kv_shard = 0.f;
    #pragma unroll
    for (int r = 0; r < GDN_DECODE_MAX_RPL; r++)
        kv_shard += s_shard[r] * k_reg[r];
    float kv_col = sharpi_warp_reduce_sum(kv_shard);
    float p_col  = decay * kv_col;

    float delta_col = b_sc * (v[hd_off + col] - p_col);

    // Pass B: fused rank-1 update S[i,col] = decay·S[i,col] + k[i]·delta, readout
    // o[col] = (1/√d)·Σ_i q[i]·S'[i,col].
    float attn_shard = 0.f;
    #pragma unroll
    for (int r = 0; r < GDN_DECODE_MAX_RPL; r++) {
        s_shard[r]  = decay * s_shard[r] + k_reg[r] * delta_col;
        attn_shard += s_shard[r] * q_reg[r];
    }
    float o_col = sharpi_warp_reduce_sum(attn_shard) * rsqrtf((float)d);

    // Write the updated state shard back (transposed, coalesced) + raw readout.
    #pragma unroll
    for (int r = 0; r < GDN_DECODE_MAX_RPL; r++) {
        int i = r * 32 + lane;
        if (r < rpl)
            state[col_base + i] = s_shard[r];
    }
    if (lane == 0)
        readout[hd_off + col] = o_col;
}

// ── GDN: per-head RMSNorm(readout)·norm_weight · SiLU(z) gate ─────────────────
// Issue #404 / Option B: the split tail of llm_gdn_recurrence_decode_fast. The fast
// recurrence kernel spreads each head's d columns across blocks, so it cannot do the
// per-head RMSNorm reduction in-kernel; this kernel does it as a separate launch. One
// block per v-head, blockDim = d (thread j owns column j). Reads the raw readout
// o[h*d+j] (the 1/√d-scaled Σ q·S'), applies RMSNorm with the same tree reduction +
// eps floor as the original kernel's tail, scales by norm_weight[j], then by
// SiLU(z[h*d+j]), and writes the final gated output back in place. Byte-identical to
// the original kernel's tail given the same readout input.
extern ""C"" __global__ void llm_gdn_decode_norm_gate(
    float* __restrict__ output,
    const float* __restrict__ norm_weight,
    const float* __restrict__ z,
    int hv, int d, float norm_eps)
{
    int h = (int)blockIdx.x;
    int j = (int)threadIdx.x;
    if (h >= hv || j >= d) return;

    extern __shared__ float sRed[];
    long hd_off = (long)h * d;

    float o_local = output[hd_off + j];
    sRed[j] = o_local * o_local;
    __syncthreads();
    // Tree reduction robust to non-power-of-two d (≤ 128): start at 64 and mask the
    // out-of-range partner with `j + s < d`. For d=128 (power of two) `j + s < d`
    // holds exactly when `j < s`, so this is byte-identical to the plain `s = d/2`
    // halving; for d=96 it correctly folds the upper tail instead of dropping it.
    for (int s = 64; s > 0; s >>= 1) {
        if (j < s && j + s < d) sRed[j] += sRed[j + s];
        __syncthreads();
    }
    float scale = rsqrtf(sRed[0] / (float)d + norm_eps);
    float o_normed = o_local * scale * norm_weight[j];

    float zv = z[hd_off + j];
    float silu = zv / (1.0f + __expf(-zv));
    output[hd_off + j] = o_normed * silu;
}

// ── GDN: in-place per-head square transpose of the [d×d] state ────────────────
// Issue #404 / Option B: converts the canonical row-major state S[h*d*d + i*d + j]
// to the transposed (column-major) layout the fast decode kernel reads coalesced —
// and back again (the transpose is its own inverse). One thread per (i,j) upper-
// triangle pair swaps state[i*d+j] ↔ state[j*d+i]; the diagonal is left untouched.
// Used only at layout transitions (fast-decode entry / before any row-major consumer
// such as the batched scan or the end-of-decode snapshot), so it is not perf-critical.
extern ""C"" __global__ void llm_gdn_state_transpose(
    float* __restrict__ state, int hv, int d)
{
    int h = (int)blockIdx.z;
    int i = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int j = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (h >= hv || i >= d || j >= d || i >= j) return;   // upper triangle only (i < j)
    long base = (long)h * (long)d * (long)d;
    long a = base + (long)i * d + j;
    long b = base + (long)j * d + i;
    float tmp = state[a];
    state[a]  = state[b];
    state[b]  = tmp;
}

// ════════════════════════════════════════════════════════════════════════════
//  Issue #114-B: batched GDN trunk kernels — collapse the per-position decode
//  launches into one launch each over all N prompt tokens. Every kernel runs the
//  IDENTICAL per-row / per-position arithmetic (and reduction order) as its
//  single-token counterpart above, so the batched trunk is BIT-IDENTICAL to the
//  per-token TrunkLayerSequential path. The recurrence/conv state evolves exactly
//  as the sequential loop; only the host launch overhead is removed.
// ════════════════════════════════════════════════════════════════════════════

// ── GDN: depthwise causal conv1d over a chunk (read-only state) ─────────────
// Bit-identical to N sequential `llm_gdn_conv1d_decode` calls. Each (channel,
// token) thread computes output[i,c] reading the chunk inputs + the carried
// pre-chunk state (oldest-first [(K-1), channels]). Does NOT mutate state — the
// state advance is a separate launch (`llm_gdn_conv1d_state_update_batched`) so
// concurrent token blocks all read the same old state. Sum order matches the
// single-token kernel exactly: current tap (weight[K-1]) first, then taps
// k=0..K-2 oldest→newest.
extern ""C"" __global__ void llm_gdn_conv1d_decode_batched(
    const float* __restrict__ x,        // [n_tok, channels]
    const float* __restrict__ state,    // [(K-1), channels] oldest-first, pre-chunk
    const float* __restrict__ weight,   // [K, channels]
    float* __restrict__ output,         // [n_tok, channels]
    int channels, int kernel_size, int n_tok)
{
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)blockIdx.y;
    if (c >= channels || i >= n_tok) return;

    int retained = kernel_size - 1;
    float x_c = x[(long)i * channels + c];
    float sum = weight[(long)retained * channels + c] * x_c;
    #pragma unroll 4
    for (int k = 0; k < retained; k++) {
        int p = i - retained + k;       // chunk-relative position of tap k
        float val = (p >= 0)
            ? x[(long)p * channels + c]
            : state[(long)(p + retained) * channels + c];
        sum += weight[(long)k * channels + c] * val;
    }
    output[(long)i * channels + c] = sum;
}

// ── GDN: advance conv1d state past a chunk ──────────────────────────────────
// After the sequential loop processes all n_tok tokens, the retained state holds
// the last (K-1) inputs oldest-first. Reproduces that exactly: new_state[r] is the
// chunk input at position (n_tok-(K-1)+r), or the carried old state when that index
// is still before the chunk (n_tok < K-1). All sources are read into registers
// before any write to tolerate the in-place aliasing of the small-N case.
extern ""C"" __global__ void llm_gdn_conv1d_state_update_batched(
    const float* __restrict__ x,        // [n_tok, channels]
    float* __restrict__ state,          // [(K-1), channels] in/out
    int channels, int kernel_size, int n_tok)
{
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= channels) return;

    int retained = kernel_size - 1;
    float tmp[4];                        // K-1 <= 4 for our models
    #pragma unroll 4
    for (int r = 0; r < retained; r++) {
        int p = n_tok - retained + r;
        tmp[r] = (p >= 0)
            ? x[(long)p * channels + c]
            : state[(long)(p + retained) * channels + c];
    }
    #pragma unroll 4
    for (int r = 0; r < retained; r++)
        state[(long)r * channels + c] = tmp[r];
}

// ── GDN #290: capture intermediate conv1d states into the verify ring ───────
// One launch writes every batched-verify ring slot's conv state. grid =
// (ceil(channels/256), n_capture); block (c, slot) writes the conv state AFTER
// token `slot` (i.e. the state the per-token loop held once it had advanced over
// the first slot+1 tokens) into ring slot `slot`. Byte-identical to invoking
// `llm_gdn_conv1d_state_update_batched` with n_tok = slot+1: the per-slot window
// is [slot+1-retained .. slot], drawing from the carried pre-chunk `state` for the
// early-token (p < 0) padding. Reads the PRE-update state (the caller must run
// this BEFORE the in-place state advance). `ring` points to this layer's region
// in slot 0; `ring_slot_stride` is the float stride between consecutive slots.
extern ""C"" __global__ void llm_gdn_conv1d_state_capture_ring(
    const float* __restrict__ x,        // [n_tok, channels] chunk inputs
    const float* __restrict__ state,    // [(K-1), channels] oldest-first, pre-chunk
    float* __restrict__ ring,           // layer region in slot 0
    int channels, int kernel_size, int ring_slot_stride, int n_capture)
{
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int slot = (int)blockIdx.y;
    if (c >= channels || slot >= n_capture) return;

    int retained = kernel_size - 1;
    int n_eff = slot + 1;               // state after processing tokens [0, n_eff)
    float* dst = ring + (long)slot * ring_slot_stride;
    #pragma unroll 4
    for (int r = 0; r < retained; r++) {
        int p = n_eff - retained + r;
        float v = (p >= 0)
            ? x[(long)p * channels + c]
            : state[(long)(p + retained) * channels + c];
        dst[(long)r * channels + c] = v;
    }
}

// ── GDN: L2-norm per head, batched over n_tok rows ──────────────────────────
// Bit-identical to n_tok sequential `llm_gdn_l2_norm_per_head` calls. grid =
// (num_heads, n_tok); `data` is the region base (already offset to the Q or K
// region by the host), `row_stride` is the per-token element stride (= conv
// channels). One block per (head, token); same 256-thread tree reduction.
extern ""C"" __global__ void llm_gdn_l2_norm_per_head_batched(
    float* __restrict__ data,
    int head_dim, int num_heads, float eps, int row_stride, int n_tok)
{
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int head = blockIdx.x;
    int i = (int)blockIdx.y;
    if ((int)head >= num_heads || i >= n_tok) return;

    float* d = data + (long)i * row_stride + (long)head * head_dim;

    float sum = 0.f;
    for (int e = (int)tid; e < head_dim; e += 256) {
        float v = d[e];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float norm = sqrtf(sdata[0]);
    float divisor = norm > eps ? norm : eps;
    float inv_div = 1.0f / divisor;
    for (int e = (int)tid; e < head_dim; e += 256)
        d[e] *= inv_div;
}

// ── GDN: tile heads (GQA broadcast), batched over n_tok rows ────────────────
// Bit-identical to n_tok sequential `llm_gdn_tile_heads` calls. `src` is the
// region base (host-offset to Q or K region), `src_stride` the per-token source
// stride (= conv channels), `dst_stride` the per-token dst stride (= value_dim).
extern ""C"" __global__ void llm_gdn_tile_heads_batched(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int src_heads, int repeat, int head_dim,
    int src_stride, int dst_stride, int n_tok)
{
    int dst_total = src_heads * repeat * head_dim;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)blockIdx.y;
    if (idx >= dst_total || i >= n_tok) return;
    int j = idx % head_dim;
    int h_dst = idx / head_dim;
    int h_src = h_dst % src_heads;
    dst[(long)i * dst_stride + idx] = src[(long)i * src_stride + (long)h_src * head_dim + j];
}

// ── GDN: fused sequential recurrence scan over a chunk ───────────────────────
// One block per v-head, blockDim = head_dim. Loops the n_tok positions INTERNALLY,
// running the exact passes of `llm_gdn_recurrence_decode` at each step and carrying
// the per-head state matrix in global memory between steps. This is the bit-identical
// fused form of N sequential `llm_gdn_recurrence_decode` launches — NOT the parallel
// chunked-scan (which reorders the FP reductions). Each thread owns output column j
// of its head's state, so the only cross-thread sharing is via shared memory with
// the same __syncthreads barriers as the single-token kernel; the trailing barrier
// makes the position boundary clean before the next step reloads shared inputs.
//
// Per-head input strides let q/k come from the tiled [n_tok, value_dim] buffers,
// v straight from the silu'd conv output (v_head_off into the [n_tok, conv_ch]
// buffer), z from the [n_tok, value_dim] gate, alpha/beta from [n_tok, num_v_heads].
extern ""C"" __global__ void llm_gdn_recurrence_scan(
    float* __restrict__ state,            // [hv, d, d]
    const float* __restrict__ q_all,      // [n_tok, q_stride]; head h at h*d
    const float* __restrict__ k_all,      // [n_tok, k_stride]
    const float* __restrict__ v_all,      // [n_tok, v_stride]; head h at v_head_off + h*d
    const float* __restrict__ alpha_all,  // [n_tok, hv]
    const float* __restrict__ beta_all,   // [n_tok, hv]
    const float* __restrict__ ssm_a,      // [hv]
    const float* __restrict__ dt_bias,    // [hv]
    const float* __restrict__ norm_weight,// [d]
    const float* __restrict__ z_all,      // [n_tok, z_stride]; head h at h*d
    float* __restrict__ output_all,       // [n_tok, o_stride]; head h at h*d
    int hv, int d, float norm_eps,
    int q_stride, int k_stride, int v_stride, int v_head_off,
    int z_stride, int o_stride, int n_tok,
    // ── #290 batched-verify ring capture (nullable) ──────────────────────────
    // When ring_scan != null, the post-token-i state (after Pass B, i.e. exactly
    // what the live `state` holds at that boundary) is also written into ring
    // slot i for i ∈ [0, n_capture). ring_scan points to this layer's region in
    // slot 0; ring_slot_stride is the float stride between consecutive slots
    // (= numGdnLayers * scanStateFloatsPerLayer). Disjoint from `state`, so the
    // scan arithmetic and the live `state` evolution stay byte-unchanged — the
    // capture is purely additional stores to a separate buffer.
    float* __restrict__ ring_scan,
    int ring_slot_stride, int n_capture)
{
    int h = (int)blockIdx.x;
    int j = (int)threadIdx.x;
    if (h >= hv || j >= d) return;

    extern __shared__ float smem[];
    float* sK     = smem;
    float* sQ     = sK + d;
    float* sV     = sQ + d;
    float* sZ     = sV + d;
    float* sNormW = sZ + d;
    float* sP     = sNormW + d;
    float* sD     = sP + d;
    float* sRed   = sD + d;

    sNormW[j] = norm_weight[j];           // layer-constant; each thread reads own j
    long state_base = (long)h * (long)d * (long)d;

    for (int i = 0; i < n_tok; i++) {
        long qoff = (long)i * q_stride + (long)h * d;
        long koff = (long)i * k_stride + (long)h * d;
        long voff = (long)i * v_stride + v_head_off + (long)h * d;
        long zoff = (long)i * z_stride + (long)h * d;
        sK[j] = k_all[koff + j];
        sQ[j] = q_all[qoff + j];
        sV[j] = v_all[voff + j];
        sZ[j] = z_all[zoff + j];
        __syncthreads();

        float alpha_x = alpha_all[(long)i * hv + h] + dt_bias[h];
        float dt      = alpha_x >= 20.0f ? alpha_x : __logf(1.0f + __expf(alpha_x));
        float decay   = __expf(dt * ssm_a[h]);
        float b_sc    = 1.0f / (1.0f + __expf(-beta_all[(long)i * hv + h]));

        // Pass A: decay S, accumulate p[j] = Σ_i k[i]·S[i,j].
        float p_local = 0.f;
        for (int ii = 0; ii < d; ii++) {
            long off = state_base + (long)ii * d + j;
            float sij = state[off] * decay;
            state[off] = sij;
            p_local += sK[ii] * sij;
        }
        sP[j] = p_local;
        __syncthreads();

        float d_j = b_sc * (sV[j] - sP[j]);
        sD[j] = d_j;
        __syncthreads();

        // Pass B: rank-1 update S[i,j] += k[i]·d[j], fused with readout o[j].
        // #290: when capturing, mirror each post-update element into ring slot i
        // (same value the live `state` now holds → byte-identical to the device
        // CopyDeviceRegion the per-position loop used to issue).
        bool capture = ring_scan != nullptr && i < n_capture;
        float* ring_i = capture ? ring_scan + (long)i * ring_slot_stride + state_base : nullptr;
        float o_local = 0.f;
        for (int ii = 0; ii < d; ii++) {
            long off = state_base + (long)ii * d + j;
            float sij = state[off] + sK[ii] * d_j;
            state[off] = sij;
            if (capture) ring_i[(long)ii * d + j] = sij;
            o_local += sQ[ii] * sij;
        }
        o_local *= rsqrtf((float)d);

        sRed[j] = o_local * o_local;
        __syncthreads();
        for (int s = d / 2; s > 0; s >>= 1) {
            if (j < s) sRed[j] += sRed[j + s];
            __syncthreads();
        }
        float scale = rsqrtf(sRed[0] / (float)d + norm_eps);
        float o_normed = o_local * scale * sNormW[j];

        float zv = sZ[j];
        float silu = zv / (1.0f + __expf(-zv));
        output_all[(long)i * o_stride + (long)h * d + j] = o_normed * silu;
        __syncthreads();                  // position boundary: next step reloads shared
    }
}

// ── GDN: chunk-parallel prefill (FlashQLA-style chunk_gated_delta_rule) ───────
// Mirrors the CPU reference GdnKernels.GdnRecurrenceChunkedPrefill. One block per
// v-head, blockDim = d; thread j owns value-column j of everything (state column j,
// proj/u/output element j), so the forward substitution, S0-projections and state
// carry are FULLY per-thread (no cross-thread sync). Only the per-chunk K·K / K·Q
// dot matrices (over the key axis, shared by all j) and the per-token RMSNorm
// reduction need cooperation. Same input strides/layout as llm_gdn_recurrence_scan.
//
// Numerically equal to the sequential scan up to FP reduction order (the chunked
// form resolves the intra-chunk delta-rule coupling via forward substitution over a
// fixed-size GDN_CHUNK tile). GDN_CHUNK must match the C# wrapper's GdnChunkSize.
#define GDN_CHUNK 64
extern ""C"" __global__ void llm_gdn_chunked_prefill(
    float* __restrict__ state,            // [hv, d, d]
    const float* __restrict__ q_all,      // [n_tok, q_stride]; head h at h*d
    const float* __restrict__ k_all,      // [n_tok, k_stride]
    const float* __restrict__ v_all,      // [n_tok, v_stride]; head h at v_head_off + h*d
    const float* __restrict__ alpha_all,  // [n_tok, hv]
    const float* __restrict__ beta_all,   // [n_tok, hv]
    const float* __restrict__ ssm_a,      // [hv]
    const float* __restrict__ dt_bias,    // [hv]
    const float* __restrict__ norm_weight,// [d]
    const float* __restrict__ z_all,      // [n_tok, z_stride]; head h at h*d
    float* __restrict__ output_all,       // [n_tok, o_stride]; head h at h*d
    int hv, int d, float norm_eps,
    int q_stride, int k_stride, int v_stride, int v_head_off,
    int z_stride, int o_stride, int n_tok)
{
    int h = (int)blockIdx.x;
    int j = (int)threadIdx.x;
    if (h >= hv || j >= d) return;

    extern __shared__ float smem[];
    float* sNormW = smem;                              // [d]
    float* sCum   = sNormW + d;                        // [GDN_CHUNK]
    float* sG     = sCum + GDN_CHUNK;                  // [GDN_CHUNK]  exp(cum_t)
    float* sB     = sG + GDN_CHUNK;                    // [GDN_CHUNK]  sigmoid(beta_t)
    float* sKK    = sB + GDN_CHUNK;                    // [GDN_CHUNK*GDN_CHUNK]  K_s·K_t
    float* sKQ    = sKK + GDN_CHUNK * GDN_CHUNK;       // [GDN_CHUNK*GDN_CHUNK]  K_s·Q_t
    float* sRed   = sKQ + GDN_CHUNK * GDN_CHUNK;       // [d]  RMSNorm reduction

    sNormW[j] = norm_weight[j];
    long state_base = (long)h * (long)d * (long)d;
    float inv_sqrt_d = rsqrtf((float)d);

    float projK[GDN_CHUNK];
    float projQ[GDN_CHUNK];
    float u[GDN_CHUNK];

    for (int c0 = 0; c0 < n_tok; c0 += GDN_CHUNK) {
        int cN = n_tok - c0; if (cN > GDN_CHUNK) cN = GDN_CHUNK;

        // Per-token scalars: cumulative log-decay is sequential → thread 0 fills shared.
        if (j == 0) {
            float run = 0.f;
            for (int t = 0; t < cN; t++) {
                float ax = alpha_all[(long)(c0 + t) * hv + h] + dt_bias[h];
                float dt = ax >= 20.0f ? ax : __logf(1.0f + __expf(ax));
                run += dt * ssm_a[h];
                sCum[t] = run;
                sG[t]   = __expf(run);
                sB[t]   = 1.0f / (1.0f + __expf(-beta_all[(long)(c0 + t) * hv + h]));
            }
        }
        __syncthreads();

        // K·K and K·Q dot matrices (lower triangle s<=t); shared across all columns j.
        for (int idx = j; idx < cN * cN; idx += d) {
            int t = idx / cN;
            int s = idx - t * cN;
            if (s <= t) {
                long ks = (long)(c0 + s) * k_stride + (long)h * d;
                long kt = (long)(c0 + t) * k_stride + (long)h * d;
                long qt = (long)(c0 + t) * q_stride + (long)h * d;
                float kk = 0.f, kq = 0.f;
                for (int i = 0; i < d; i++) {
                    float ksi = k_all[ks + i];
                    kk += ksi * k_all[kt + i];
                    kq += ksi * q_all[qt + i];
                }
                sKK[t * GDN_CHUNK + s] = kk;
                sKQ[t * GDN_CHUNK + s] = kq;
            }
        }
        __syncthreads();

        // S0 projections (column j): projK[t]=Σ_i K_t[i]·S0[i,j], projQ[t]=Σ_i Q_t[i]·S0[i,j].
        for (int t = 0; t < cN; t++) {
            long kt = (long)(c0 + t) * k_stride + (long)h * d;
            long qt = (long)(c0 + t) * q_stride + (long)h * d;
            float pk = 0.f, pq = 0.f;
            for (int i = 0; i < d; i++) {
                float sij = state[state_base + (long)i * d + j];
                pk += k_all[kt + i] * sij;
                pq += q_all[qt + i] * sij;
            }
            projK[t] = pk;
            projQ[t] = pq;
        }

        // Forward substitution: u_t = b_t(v_t − g_t·projK_t) − Σ_{s<t} A[t,s] u_s.
        for (int t = 0; t < cN; t++) {
            long vt = (long)(c0 + t) * v_stride + v_head_off + (long)h * d;
            float bt = sB[t];
            float uj = bt * (v_all[vt + j] - sG[t] * projK[t]);
            for (int s = 0; s < t; s++) {
                float a = bt * __expf(sCum[t] - sCum[s]) * sKK[t * GDN_CHUNK + s];
                uj -= a * u[s];
            }
            u[t] = uj;
        }

        // Output + per-head RMSNorm + SiLU(z) gate.
        for (int t = 0; t < cN; t++) {
            float o = sG[t] * projQ[t];
            for (int s = 0; s <= t; s++)
                o += __expf(sCum[t] - sCum[s]) * sKQ[t * GDN_CHUNK + s] * u[s];
            o *= inv_sqrt_d;

            sRed[j] = o * o;
            __syncthreads();
            for (int red = d / 2; red > 0; red >>= 1) {
                if (j < red) sRed[j] += sRed[j + red];
                __syncthreads();
            }
            float scale = rsqrtf(sRed[0] / (float)d + norm_eps);
            float on = o * scale * sNormW[j];
            float zv = z_all[(long)(c0 + t) * z_stride + (long)h * d + j];
            float silu = zv / (1.0f + __expf(-zv));
            output_all[(long)(c0 + t) * o_stride + (long)h * d + j] = on * silu;
            __syncthreads();
        }

        // State carry: S[i,j] = g_{cN-1}·S[i,j] + Σ_s exp(cum_{cN-1}−cum_s)·K_s[i]·u_s.
        float gLast = sG[cN - 1];
        float cumLast = sCum[cN - 1];
        for (int i = 0; i < d; i++) {
            long off = state_base + (long)i * d + j;
            float acc = gLast * state[off];
            for (int s = 0; s < cN; s++) {
                long ks = (long)(c0 + s) * k_stride + (long)h * d;
                acc += __expf(cumLast - sCum[s]) * k_all[ks + i] * u[s];
            }
            state[off] = acc;
        }
        __syncthreads();   // chunk boundary: next chunk overwrites shared
    }
}
#undef GDN_CHUNK

// ════════════════════════════════════════════════════════════════════════════
//  Issue #114-B: batched-query SDPA for prompt prefill.
// ════════════════════════════════════════════════════════════════════════════

// ── KV cache append, batched over n_tok tokens (fp32 store) ─────────────────
// Bit-identical to n_tok sequential `llm_kv_append` calls at positions
// start_pos+i. grid = ((kv_dim+255)/256, n_tok).
extern ""C"" __global__ void llm_kv_append_batched(
    const float* __restrict__ k_all, const float* __restrict__ v_all,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    int kv_dim, int start_pos, int max_seq_len, int n_tok)
{
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)blockIdx.y;
    if (e >= kv_dim || i >= n_tok) return;
    // Ring slot `(start_pos+i) % max_seq_len`: identity for a full-context cache
    // (position < max_seq_len), wraps into a window-sized SWA ring otherwise.
    long off = (long)((start_pos + i) % max_seq_len) * (long)kv_dim + (long)e;
    k_cache[off] = k_all[(long)i * kv_dim + e];
    v_cache[off] = v_all[(long)i * kv_dim + e];
}

// bf16-store variant (default KV dtype). Matches `llm_kv_append_bf16`.
extern ""C"" __global__ void llm_kv_append_batched_bf16(
    const float* __restrict__ k_all, const float* __restrict__ v_all,
    unsigned short* __restrict__ k_cache, unsigned short* __restrict__ v_cache,
    int kv_dim, int start_pos, int max_seq_len, int n_tok)
{
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)blockIdx.y;
    if (e >= kv_dim || i >= n_tok) return;
    // Ring slot `(start_pos+i) % max_seq_len` (identity for a full-context cache; wraps a
    // window-sized ring) — kept in lockstep with the f32 llm_kv_append_batched.
    long off = (long)((start_pos + i) % max_seq_len) * (long)kv_dim + (long)e;
    k_cache[off] = (unsigned short)sharpi_fp32_to_bf16(k_all[(long)i * kv_dim + e]);
    v_cache[off] = (unsigned short)sharpi_fp32_to_bf16(v_all[(long)i * kv_dim + e]);
}

// q8_0-store variant (issue #179). Matches `llm_kv_append_q8_0`; one hardware warp
// per q8_0 block, grid = (ceil(kv_dim/256), n_tok). The token's source row is
// dense fp32 [i*kv_dim + e]; the cache block index uses the ring slot.
extern ""C"" __global__ void llm_kv_append_batched_q8_0(
    const float* __restrict__ k_all, const float* __restrict__ v_all,
    block_q8_0* __restrict__ k_cache, block_q8_0* __restrict__ v_cache,
    int kv_dim, int start_pos, int max_seq_len, int n_tok)
{
    int e = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int i = (int)blockIdx.y;
    int lane = (int)(threadIdx.x & 31);
    bool valid = (e < kv_dim && i < n_tok);
    long row   = (long)((start_pos + i) % max_seq_len);
    long block = (row * (long)kv_dim + (long)e) >> 5;
    sharpi_q8_append_one(valid ? k_all[(long)i * kv_dim + e] : 0.f, valid, k_cache, block, lane);
    sharpi_q8_append_one(valid ? v_all[(long)i * kv_dim + e] : 0.f, valid, v_cache, block, lane);
}

// ── Full-sequence (batched-query) scaled dot-product attention ──────────────
// Implements CudaBackend.FullSeqAttention for prompt prefill. grid = (num_heads,
// n_tok): block (h, i) computes the attention output for query i (absolute
// position start_pos+i) over the causal prefix [0, start_pos+i+1). Bit-identical
// to n_tok sequential `llm_attention` calls (use_shared path) — same per-position
// dot, same 256-thread tree softmax, same V-weighted sum. The host only dispatches
// this when start_pos+n_tok ≤ MAX_STORED_SCORES, so every block stays on the
// shared-scores fast path (no global scratch, no cross-block aliasing).
extern ""C"" __global__ void llm_full_seq_attention(
    const float* __restrict__ q_all,      // [n_tok, num_heads*head_dim]
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out_all,          // [n_tok, num_heads*head_dim]
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int max_seq_len, int n_tok, float attn_scale)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    int i = (int)blockIdx.y;
    if ((int)h >= num_heads || i >= n_tok) return;

    int seq_len = start_pos + i + 1;
    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    int q_dim = num_heads * head_dim;
    const float* q = q_all + (long)i * q_dim;
    float* out = out_all + (long)i * q_dim;
    long q_off = (long)h * (long)head_dim;
    long out_off = q_off;

    for (int t = (int)tid; t < seq_len; t += 256) {
        float dot = 0.f;
        long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int dd = 0; dd < head_dim; dd++)
            dot += q[q_off + dd] * k_cache[k_off + dd];
        shared_scores[t] = dot * scale;
    }
    __syncthreads();

    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < seq_len; t += 256)
        local_max = fmaxf(local_max, shared_scores[t]);
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < seq_len; t += 256) {
        float ev = __expf(shared_scores[t] - max_val);
        shared_scores[t] = ev;
        local_sum += ev;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < seq_len; t += 256)
        shared_scores[t] *= inv_sum;
    __syncthreads();

    for (int dd = (int)tid; dd < head_dim; dd += 256) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++) {
            long v_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += shared_scores[t] * v_cache[v_off + dd];
        }
        out[out_off + dd] = acc;
    }
}

// Templated K/V dtype (bf16 / q8_0; #179). Matches `llm_attention`'s use_shared path,
// decoding each cache element to fp32 on load via sharpi_kvload.
template<typename KV>
__device__ void llm_full_seq_attention_kv_impl(
    const float* __restrict__ q_all,
    const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache,
    float* __restrict__ out_all,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int max_seq_len, int n_tok, float attn_scale)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    int i = (int)blockIdx.y;
    if ((int)h >= num_heads || i >= n_tok) return;

    int seq_len = start_pos + i + 1;
    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    // attn_scale > 0 overrides (Gemma 4 passes 1.0); ≤0 uses 1/sqrt(head_dim).
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    int q_dim = num_heads * head_dim;
    const float* q = q_all + (long)i * q_dim;
    float* out = out_all + (long)i * q_dim;
    long q_off = (long)h * (long)head_dim;
    long out_off = q_off;

    for (int t = (int)tid; t < seq_len; t += 256) {
        float dot = 0.f;
        long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int dd = 0; dd < head_dim; dd++)
            dot += q[q_off + dd] * sharpi_kvload(k_cache, k_off + dd);
        shared_scores[t] = dot * scale;
    }
    __syncthreads();

    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < seq_len; t += 256)
        local_max = fmaxf(local_max, shared_scores[t]);
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < seq_len; t += 256) {
        float ev = __expf(shared_scores[t] - max_val);
        shared_scores[t] = ev;
        local_sum += ev;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < seq_len; t += 256)
        shared_scores[t] *= inv_sum;
    __syncthreads();

    for (int dd = (int)tid; dd < head_dim; dd += 256) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++) {
            long v_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += shared_scores[t] * sharpi_kvload(v_cache, v_off + dd);
        }
        out[out_off + dd] = acc;
    }
}

extern ""C"" __global__ void llm_full_seq_attention_bf16(
    const float* __restrict__ q_all,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ out_all,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int max_seq_len, int n_tok, float attn_scale)
{
    llm_full_seq_attention_kv_impl<unsigned short>(q_all, k_cache, v_cache, out_all,
        num_heads, num_kv_heads, head_dim, start_pos, max_seq_len, n_tok, attn_scale);
}

extern ""C"" __global__ void llm_full_seq_attention_q8_0(
    const float* __restrict__ q_all,
    const block_q8_0* __restrict__ k_cache,
    const block_q8_0* __restrict__ v_cache,
    float* __restrict__ out_all,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int max_seq_len, int n_tok, float attn_scale)
{
    llm_full_seq_attention_kv_impl<block_q8_0>(q_all, k_cache, v_cache, out_all,
        num_heads, num_kv_heads, head_dim, start_pos, max_seq_len, n_tok, attn_scale);
}

// ── Full-sequence (batched-query) SDPA, global-scratch (issue #118) ─────────
// Bit-for-bit clone of `llm_full_seq_attention` except per-position scores spill
// to a caller-provided global slice instead of the 16 KB `__shared__` buffer, so
// it works past the 4096-position shared-scores window. Each block (h, i)
// (grid = num_heads × n_tok) owns the private slice
// `scores_scratch + (i*num_heads + h) * score_stride`, scanned only over
// [0, seq_len) — so the per-(head,query) math (dot, 256-thread tree softmax,
// V-weighted sum) is identical to `llm_attention`'s global-scratch path (the
// per-position >4096 fallback this replaces). `score_stride` must be ≥ the
// largest seq_len any query in this launch reaches (start_pos + n_tok). The host
// drives this in waves of W ≤ n_tok queries (W chosen so the scratch fits a
// bounded budget; each launch's n_tok == that wave's width) — see
// CudaBackend.AttentionBatchedWave.
extern ""C"" __global__ void llm_full_seq_attention_global(
    const float* __restrict__ q_all,      // [n_tok, num_heads*head_dim]
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out_all,          // [n_tok, num_heads*head_dim]
    float* __restrict__ scores_scratch,   // [n_tok * num_heads * score_stride]
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int max_seq_len, int n_tok, int score_stride)
{
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    int i = (int)blockIdx.y;
    if ((int)h >= num_heads || i >= n_tok) return;

    int seq_len = start_pos + i + 1;
    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    float scale = rsqrtf((float)head_dim);
    int q_dim = num_heads * head_dim;
    const float* q = q_all + (long)i * q_dim;
    float* out = out_all + (long)i * q_dim;
    long q_off = (long)h * (long)head_dim;
    long out_off = q_off;
    float* sc = scores_scratch + ((long)i * num_heads + h) * (long)score_stride;

    for (int t = (int)tid; t < seq_len; t += 256) {
        float dot = 0.f;
        long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int dd = 0; dd < head_dim; dd++)
            dot += q[q_off + dd] * k_cache[k_off + dd];
        sc[t] = dot * scale;
    }
    __syncthreads();

    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < seq_len; t += 256)
        local_max = fmaxf(local_max, sc[t]);
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < seq_len; t += 256) {
        float ev = __expf(sc[t] - max_val);
        sc[t] = ev;
        local_sum += ev;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < seq_len; t += 256)
        sc[t] *= inv_sum;
    __syncthreads();

    for (int dd = (int)tid; dd < head_dim; dd += 256) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++) {
            long v_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += sc[t] * v_cache[v_off + dd];
        }
        out[out_off + dd] = acc;
    }
}

// Templated K/V dtype (bf16 / q8_0; #179). Matches `llm_attention`'s global-scratch
// path, decoding each cache element to fp32 on load via sharpi_kvload.
template<typename KV>
__device__ void llm_full_seq_attention_global_kv_impl(
    const float* __restrict__ q_all,
    const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache,
    float* __restrict__ out_all,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int max_seq_len, int n_tok, int score_stride, float attn_scale)
{
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    int i = (int)blockIdx.y;
    if ((int)h >= num_heads || i >= n_tok) return;

    int seq_len = start_pos + i + 1;
    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    // attn_scale > 0 overrides (Gemma 4 passes 1.0); ≤0 uses 1/sqrt(head_dim).
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    int q_dim = num_heads * head_dim;
    const float* q = q_all + (long)i * q_dim;
    float* out = out_all + (long)i * q_dim;
    long q_off = (long)h * (long)head_dim;
    long out_off = q_off;
    float* sc = scores_scratch + ((long)i * num_heads + h) * (long)score_stride;

    for (int t = (int)tid; t < seq_len; t += 256) {
        float dot = 0.f;
        long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int dd = 0; dd < head_dim; dd++)
            dot += q[q_off + dd] * sharpi_kvload(k_cache, k_off + dd);
        sc[t] = dot * scale;
    }
    __syncthreads();

    float local_max = sharpi_neg_inf();
    for (int t = (int)tid; t < seq_len; t += 256)
        local_max = fmaxf(local_max, sc[t]);
    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int t = (int)tid; t < seq_len; t += 256) {
        float ev = __expf(sc[t] - max_val);
        sc[t] = ev;
        local_sum += ev;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    __syncthreads();

    for (int t = (int)tid; t < seq_len; t += 256)
        sc[t] *= inv_sum;
    __syncthreads();

    for (int dd = (int)tid; dd < head_dim; dd += 256) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++) {
            long v_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += sc[t] * sharpi_kvload(v_cache, v_off + dd);
        }
        out[out_off + dd] = acc;
    }
}

extern ""C"" __global__ void llm_full_seq_attention_global_bf16(
    const float* __restrict__ q_all,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ out_all,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int max_seq_len, int n_tok, int score_stride, float attn_scale)
{
    llm_full_seq_attention_global_kv_impl<unsigned short>(q_all, k_cache, v_cache, out_all,
        scores_scratch, num_heads, num_kv_heads, head_dim, start_pos, max_seq_len, n_tok, score_stride, attn_scale);
}

extern ""C"" __global__ void llm_full_seq_attention_global_q8_0(
    const float* __restrict__ q_all,
    const block_q8_0* __restrict__ k_cache,
    const block_q8_0* __restrict__ v_cache,
    float* __restrict__ out_all,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int start_pos, int max_seq_len, int n_tok, int score_stride, float attn_scale)
{
    llm_full_seq_attention_global_kv_impl<block_q8_0>(q_all, k_cache, v_cache, out_all,
        scores_scratch, num_heads, num_kv_heads, head_dim, start_pos, max_seq_len, n_tok, score_stride, attn_scale);
}
";
}
