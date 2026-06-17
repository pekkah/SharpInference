namespace SharpInference.Cuda;

/// <summary>
/// Ragged-batched decode kernels (issue #197), appended to the NVRTC compilation after
/// <see cref="CudaTextKernels"/> / <see cref="CudaWsKernels"/> (whose <c>sharpi_*</c>
/// helpers and <c>block_q8_0</c> they reuse).
///
/// <para>Batched decode (#190/#194) runs one token per sequence, each at its own position
/// against its own per-sequence KV cache. After #194 made the matmuls weight-stationary,
/// the residual per-step cost was the serial per-sequence block in
/// <c>CudaForwardPass.BatchForwardMulti</c>: ~6 low-occupancy single-token launches per
/// sequence per layer (QK-norm ×2, RoPE ×2, KV-append, single-query attention), the
/// attention kernels running back-to-back instead of concurrently. These kernels take the
/// whole batch in ONE launch: the sequence index rides on <c>blockIdx.y</c>, per-sequence
/// positions arrive as a by-value array parameter, and the N distinct cache base pointers
/// arrive as a by-value pointer-table parameter.</para>
///
/// <para><b>Parameter passing:</b> the positions array and the K/V cache pointer tables are
/// plain struct kernel parameters (<c>sharpi_seq_pos</c> / <c>sharpi_seq_ptrs</c>, capacity
/// <see cref="ChunkCapacity"/>). The driver copies kernel parameters with the launch
/// command itself, so there is NO device-side table buffer, NO host→device upload, and NO
/// synchronization on the hot path — the constraint #197 puts on the pointer table. Batches
/// larger than the capacity are chunked into multiple launches by the
/// <c>CudaBackend.*BatchedRagged</c> wrappers (launch count O(N/16), still ~O(1) at decode
/// batch sizes). These kernels are never captured into CUDA graphs (batched decode issues
/// direct launches only), so baking pointers into parameters is safe.</para>
///
/// <para><b>Bit-exact:</b> every kernel keeps the per-element / per-(head, position)
/// computation chain of its single-sequence counterpart (<c>llm_rope_neox</c> /
/// <c>llm_rope_interleaved</c> / <c>llm_kv_append*</c> / <c>llm_attention*</c>) — only the
/// row/cache indirection differs — so each sequence's output is bit-identical to the
/// matching sequential per-token call (CudaRaggedDecodeKernelTests enforce this against
/// the independent per-token kernels).</para>
/// </summary>
internal static class CudaRaggedKernels
{
    /// <summary>Max sequences per launch — the capacity of the by-value struct parameters.
    /// Matches the largest #194 weight-stationary capacity; the backend wrappers chunk
    /// larger batches into ceil(N/16) launches.</summary>
    internal const int ChunkCapacity = 16;

    public static string Source { get; } = @"
// ── Ragged-batch by-value parameter structs (issue #197) ───────────────────
// Passed as plain kernel parameters: the driver snapshots them into the launch
// command, so per-sequence positions / cache pointers need no device buffer and
// no host->device copy. Capacity 16 per launch; the host chunks larger batches.
typedef struct { const void* p[16]; } sharpi_seq_ptrs;
typedef struct { int v[16]; } sharpi_seq_pos;

// ── Ragged NEOX RoPE over n_tok rows at per-sequence positions ─────────────
// Row t of x rotates at positions.v[t] (batched decode: every sequence sits at
// its own position — unlike llm_rope_neox_*_batched's base_position + t prefill
// contract). Per (pair, row) bit-identical to llm_rope_neox at that position.
extern ""C"" __global__ void llm_rope_neox_ragged(
    float* __restrict__ x,
    int num_heads, int head_dim, sharpi_seq_pos positions, float theta, int n_tok)
{
    int pair_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int half_dim = head_dim / 2;
    int total_pairs = num_heads * half_dim;
    int token = (int)blockIdx.y;
    if (pair_idx >= total_pairs || token >= n_tok) return;

    int h = pair_idx / half_dim;
    int i = pair_idx % half_dim;
    int position = positions.v[token];

    float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)head_dim);
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

// ── Ragged interleaved RoPE (GPT-NeoX-style pairs (2i, 2i+1)) ──────────────
// Per (pair, row) bit-identical to llm_rope_interleaved at positions.v[t].
extern ""C"" __global__ void llm_rope_interleaved_ragged(
    float* __restrict__ x,
    int num_heads, int head_dim, sharpi_seq_pos positions, float theta, int n_tok)
{
    int pair_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int half_dim = head_dim / 2;
    int total_pairs = num_heads * half_dim;
    int token = (int)blockIdx.y;
    if (pair_idx >= total_pairs || token >= n_tok) return;

    int h = pair_idx / half_dim;
    int i = pair_idx % half_dim;
    int position = positions.v[token];

    float freq = 1.0f / powf(theta, 2.0f * (float)i / (float)head_dim);
    float angle = (float)position * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    long base = (long)token * (long)num_heads * (long)head_dim + (long)h * head_dim + 2L * i;
    float x0 = x[base];
    float x1 = x[base + 1];
    x[base]     = x0 * c - x1 * s;
    x[base + 1] = x0 * s + x1 * c;
}

// ── Ragged KV append: row t of k/v_in_all → cache t at positions.v[t] ──────
// N distinct caches per launch via the pointer table. Same ring-slot convention
// as llm_kv_append (`position % max_seq_len`; identity for the full-context
// dense caches batched decode uses). Pure copy — bit-identical trivially.
// `positions.v[t]` is the PHYSICAL slot the caller chose: position for an
// uncompacted cache, position - EvictedCount for a SnapKV-compacted one (#277).
extern ""C"" __global__ void llm_kv_append_ragged(
    const float* __restrict__ k_in_all,
    const float* __restrict__ v_in_all,
    sharpi_seq_ptrs k_caches, sharpi_seq_ptrs v_caches,
    sharpi_seq_pos positions,
    int kv_dim, int max_seq_len, int n_tok)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int token = (int)blockIdx.y;
    if (i >= kv_dim || token >= n_tok) return;

    float* k_cache = (float*)k_caches.p[token];
    float* v_cache = (float*)v_caches.p[token];
    long offset = (long)(positions.v[token] % max_seq_len) * (long)kv_dim + (long)i;
    long in_off = (long)token * (long)kv_dim + (long)i;
    k_cache[offset] = k_in_all[in_off];
    v_cache[offset] = v_in_all[in_off];
}

// bf16-store variant: same fp32→bf16 conversion as llm_kv_append_bf16.
extern ""C"" __global__ void llm_kv_append_ragged_bf16(
    const float* __restrict__ k_in_all,
    const float* __restrict__ v_in_all,
    sharpi_seq_ptrs k_caches, sharpi_seq_ptrs v_caches,
    sharpi_seq_pos positions,
    int kv_dim, int max_seq_len, int n_tok)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int token = (int)blockIdx.y;
    if (i >= kv_dim || token >= n_tok) return;

    unsigned short* k_cache = (unsigned short*)k_caches.p[token];
    unsigned short* v_cache = (unsigned short*)v_caches.p[token];
    long offset = (long)(positions.v[token] % max_seq_len) * (long)kv_dim + (long)i;
    long in_off = (long)token * (long)kv_dim + (long)i;
    k_cache[offset] = (unsigned short)sharpi_fp32_to_bf16(k_in_all[in_off]);
    v_cache[offset] = (unsigned short)sharpi_fp32_to_bf16(v_in_all[in_off]);
}

// q8_0-store variant: identical warp-collaborative block quantization to
// llm_kv_append_q8_0 (sharpi_q8_append_one — amax shuffle reduce, fp16 scale,
// rintf clamp ±127). The token-uniform early return keeps every surviving
// warp's 32 lanes intact for the shuffles, exactly as the per-token kernel.
extern ""C"" __global__ void llm_kv_append_ragged_q8_0(
    const float* __restrict__ k_in_all,
    const float* __restrict__ v_in_all,
    sharpi_seq_ptrs k_caches, sharpi_seq_ptrs v_caches,
    sharpi_seq_pos positions,
    int kv_dim, int max_seq_len, int n_tok)
{
    int token = (int)blockIdx.y;
    if (token >= n_tok) return;   // block-uniform: no divergent lanes before shuffles

    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane = (int)(threadIdx.x & 31);
    bool valid = (i < kv_dim);

    block_q8_0* k_cache = (block_q8_0*)k_caches.p[token];
    block_q8_0* v_cache = (block_q8_0*)v_caches.p[token];
    long row   = (long)(positions.v[token] % max_seq_len);
    long elem  = row * (long)kv_dim + (long)i;
    long block = elem >> 5;
    long in_off = (long)token * (long)kv_dim + (long)i;
    sharpi_q8_append_one(valid ? k_in_all[in_off] : 0.f, valid, k_cache, block, lane);
    sharpi_q8_append_one(valid ? v_in_all[in_off] : 0.f, valid, v_cache, block, lane);
}

// ── Ragged single-query attention over N distinct caches ───────────────────
// grid = (num_heads, n_tok): all N sequences' single-query attentions run
// concurrently in one launch instead of N back-to-back kernel calls. Each
// (head, sequence) block runs the EXACT llm_attention_kv_impl computation —
// same per-position score chain, same 256-thread shared-memory softmax
// reductions, same sequential phase-3 V sum — against that sequence's own
// cache over its own ragged length seq_len = positions.v[t] + 1, so per
// sequence the output is bit-identical to the per-token llm_attention call.
// positions.v[t] is the PHYSICAL last slot (position - EvictedCount for a
// SnapKV-compacted cache, #277), so seq_len is the compacted length.
// Spill scratch (seq_len > 4096) is per-(sequence, head):
// scores_scratch[((t*num_heads)+h) * max_seq_len].
template<typename KV>
__device__ void llm_attention_ragged_impl(
    const float* __restrict__ q_all,
    sharpi_seq_ptrs k_caches, sharpi_seq_ptrs v_caches,
    float* __restrict__ out_all,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    sharpi_seq_pos positions, int max_seq_len, float attn_scale, int n_tok)
{
    const int MAX_STORED_SCORES = 4096;
    __shared__ float shared_scores[MAX_STORED_SCORES];
    __shared__ float sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int h = blockIdx.x;
    int token = (int)blockIdx.y;
    if ((int)h >= num_heads || token >= n_tok) return;

    const KV* k_cache = (const KV*)k_caches.p[token];
    const KV* v_cache = (const KV*)v_caches.p[token];
    int seq_len = positions.v[token] + 1;

    int kv_head = (int)h / (num_heads / num_kv_heads);
    int kv_dim  = num_kv_heads * head_dim;
    // attn_scale > 0 overrides (Gemma 4 passes 1.0); ≤0 uses 1/sqrt(head_dim).
    float scale = (attn_scale > 0.f) ? attn_scale : rsqrtf((float)head_dim);
    long q_off  = (long)token * (long)num_heads * (long)head_dim + (long)h * (long)head_dim;
    long out_off = q_off;

    bool use_shared = (seq_len <= MAX_STORED_SCORES);
    float* head_scratch = scores_scratch
        + ((long)token * (long)num_heads + (long)h) * (long)max_seq_len;

    for (int t = (int)tid; t < seq_len; t += 256) {
        float dot = 0.f;
        long k_off = (long)t * (long)kv_dim + (long)kv_head * (long)head_dim;
        for (int d = 0; d < head_dim; d++)
            dot += q_all[q_off + d] * sharpi_kvload(k_cache, k_off + d);
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
        out_all[out_off + d] = acc;
    }
}

extern ""C"" __global__ void llm_attention_ragged(
    const float* __restrict__ q_all,
    sharpi_seq_ptrs k_caches, sharpi_seq_ptrs v_caches,
    float* __restrict__ out_all,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    sharpi_seq_pos positions, int max_seq_len, float attn_scale, int n_tok)
{
    llm_attention_ragged_impl<float>(q_all, k_caches, v_caches, out_all, scores_scratch,
        num_heads, num_kv_heads, head_dim, positions, max_seq_len, attn_scale, n_tok);
}

extern ""C"" __global__ void llm_attention_ragged_bf16(
    const float* __restrict__ q_all,
    sharpi_seq_ptrs k_caches, sharpi_seq_ptrs v_caches,
    float* __restrict__ out_all,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    sharpi_seq_pos positions, int max_seq_len, float attn_scale, int n_tok)
{
    llm_attention_ragged_impl<unsigned short>(q_all, k_caches, v_caches, out_all, scores_scratch,
        num_heads, num_kv_heads, head_dim, positions, max_seq_len, attn_scale, n_tok);
}

extern ""C"" __global__ void llm_attention_ragged_q8_0(
    const float* __restrict__ q_all,
    sharpi_seq_ptrs k_caches, sharpi_seq_ptrs v_caches,
    float* __restrict__ out_all,
    float* __restrict__ scores_scratch,
    int num_heads, int num_kv_heads, int head_dim,
    sharpi_seq_pos positions, int max_seq_len, float attn_scale, int n_tok)
{
    llm_attention_ragged_impl<block_q8_0>(q_all, k_caches, v_caches, out_all, scores_scratch,
        num_heads, num_kv_heads, head_dim, positions, max_seq_len, attn_scale, n_tok);
}

// ── Broadcast bias add over N rows: x_all[t][i] += bias[i] ─────────────────
// Replaces N per-row llm_add launches in the attn-bias / O-bias decode branch.
// One fp32 add per element — bit-identical to the per-row kernel trivially.
extern ""C"" __global__ void llm_add_bias_rows(
    float* __restrict__ x_all,
    const float* __restrict__ bias,
    int dim, int n_tok)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int token = (int)blockIdx.y;
    if (i >= dim || token >= n_tok) return;
    x_all[(long)token * (long)dim + i] += bias[i];
}
";
}
