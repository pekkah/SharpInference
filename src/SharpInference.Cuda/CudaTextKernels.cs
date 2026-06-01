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
    long offset = (long)position * (long)kv_dim + (long)i;
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
    long offset = (long)position * (long)kv_dim + (long)i;
    k_cache[offset] = (unsigned short)sharpi_fp32_to_bf16(k_in[i]);
    v_cache[offset] = (unsigned short)sharpi_fp32_to_bf16(v_in[i]);
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

    if (lane == 0) {
        // Pack {d, 0} as two fp16 halves into one uint32 at offset 0..3.
        unsigned int d_bits = sharpi_fp32_to_fp16(d);
        *reinterpret_cast<unsigned int*>(dst) = d_bits;
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
    int seq_len, int max_seq_len)
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
    int window_start, int window_end, int max_seq_len)
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
        long k_off = (long)abs_t * (long)kv_dim + (long)kv_head * (long)head_dim;
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
            long v_off = (long)abs_t * (long)kv_dim + (long)kv_head * (long)head_dim;
            acc += weight * v_cache[v_off + d];
        }
        out[out_off + d] = acc;
    }
}

// ── Scaled dot-product attention with GQA (bf16 K/V cache) ─────────────────
// Bit-for-bit copy of `llm_attention` except K/V cache is read as bfloat16
// (stored as raw unsigned short, decoded via sharpi_bf16_to_fp32). Score
// scratch, query, and output stay fp32; softmax accumulates in fp32 too.
// Bf16 → fp32 promotion happens at the dot/weighted-sum read points, so all
// arithmetic precision (and overflow head-room) matches the fp32 kernel —
// only the cache footprint is halved. See issue #27.
extern ""C"" __global__ void llm_attention_bf16(
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    float* __restrict__ out,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    int seq_len, int max_seq_len)
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
    long out_off = q_off;

    bool use_shared = (seq_len <= MAX_STORED_SCORES);
    float* head_scratch = scores_scratch + (long)h * (long)max_seq_len;

    for (int t = (int)tid; t < seq_len; t += 256) {
        float dot = 0.f;
        long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int d = 0; d < head_dim; d++)
            dot += q[q_off + d] * sharpi_bf16_to_fp32((unsigned int)k_cache[k_off + d]);
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
            acc += weight * sharpi_bf16_to_fp32((unsigned int)v_cache[v_off + d]);
        }
        out[out_off + d] = acc;
    }
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
";
}
