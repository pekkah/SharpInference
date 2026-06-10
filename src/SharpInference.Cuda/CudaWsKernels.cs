namespace SharpInference.Cuda;

/// <summary>
/// Weight-stationary batched-decode matvec kernels (issue #194), appended to the
/// NVRTC compilation after <see cref="CudaTextKernels"/> (whose <c>sharpi_*</c>
/// helpers and <c>MATVEC_Q4K_NWARPS</c> they reuse).
///
/// <para>The GEMM-N matvecs (<c>llm_matvec_*_gemm_n</c>) put the token on the grid
/// (one block-row group per token), so each weight tile is re-streamed from HBM once
/// per token — at decode batch sizes the win over sequential matvecs is launch-count
/// and L2 reuse only (#190 measured ~1.4× at N=8). These variants drop the token grid
/// dimension instead: each thread block loads a weight element once and applies it to
/// all <c>n_tok</c> activation rows, amortizing the weight HBM read N× — the dominant
/// cost of small-N batched decode.</para>
///
/// <para>Each kernel body is stamped once per compile-time batch capacity
/// (<see cref="Variants"/>; dispatch picks the smallest ≥ <c>n_tok</c>) so the
/// per-token accumulator array indexes resolve at compile time and stays
/// register-resident. Tokens beyond <c>n_tok</c> are guarded off — the guard is
/// block-uniform, so there is no intra-warp divergence.</para>
///
/// <para><b>Bit-exact:</b> each surviving (row, token) pair runs the identical
/// per-element reduction chain as the matching <c>llm_matvec_*_gemm_n</c> kernel —
/// same loads, same product association (token-invariant factors are hoisted only at
/// existing left-fold boundaries), same warp/shared reduce order — so the output is
/// bit-identical to the GEMM-N path and to <c>n_tok</c> sequential matvecs
/// (CudaMatMulBatchedWsTests enforce this). Only the loop nest changes: weight outer,
/// token inner.</para>
/// </summary>
internal static class CudaWsKernels
{
    /// <summary>Compile-time batch capacities stamped for each kernel body. Order
    /// matters: dispatch indexes kernel-handle arrays by position.</summary>
    internal static readonly int[] Variants = [2, 4, 8, 16];

    /// <summary>All weight-stationary kernels, one instantiation per variant.</summary>
    public static string Source { get; } = Build();

    private static string Build()
    {
        var sb = new System.Text.StringBuilder(Template.Length * Variants.Length);
        foreach (int nt in Variants)
            sb.Append(Template.Replace("__NT__", nt.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        return sb.ToString();
    }

    private const string Template = @"
// ── Weight-stationary batched matvecs, batch capacity __NT__ (issue #194) ──

// F32: grid = ceil(rows/8), block = 256 (8 rows × 32 lanes). Same geometry as
// llm_matvec_f32_gemm_n minus the token grid dimension; each weight element is
// loaded once and applied to all n_tok activations.
extern ""C"" __global__ void llm_matvec_f32_ws_n__NT__(
    const float* __restrict__ weights,
    const float* __restrict__ input_all,   // [n_tok][cols]
    float* __restrict__ output_all,        // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    const int NT = __NT__;
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;

    long base = (long)row * (long)cols;
    for (int i = lane; i < cols; i += THREADS_PER_ROW) {
        float w = weights[base + i];
        #pragma unroll
        for (int t = 0; t < NT; t++)
            if (t < n_tok)
                acc[t] += w * input_all[(long)t * (long)cols + i];
    }

    #pragma unroll
    for (int t = 0; t < NT; t++)
        if (t < n_tok) {
            float result = sharpi_warp_reduce_sum(acc[t]);
            if (lane == 0) output_all[(long)t * (long)rows + row] = result;
        }
}

// Q8_0 (interleaved AoS blocks): hoists the token-invariant d*q — the same
// left-fold llm_matvec_q8_0_gemm_n's  d * (float)q * x  contracts to.
extern ""C"" __global__ void llm_matvec_q8_0_ws_n__NT__(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input_all,   // [n_tok][cols]
    float* __restrict__ output_all,        // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    const int NT = __NT__;
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 5;
    long row_base_bytes = (long)row * (long)num_blocks * 34L;

    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 34L;
        unsigned int dlo = sharpi_byte_at(weights, b0 + 0);
        unsigned int dhi = sharpi_byte_at(weights, b0 + 1);
        float d = sharpi_fp16_to_fp32(dlo | (dhi << 8));
        int q = sharpi_int8_at(weights, b0 + 2 + (long)lane);
        float dq = d * (float)q;
        int elem = block * 32 + lane;
        #pragma unroll
        for (int t = 0; t < NT; t++)
            if (t < n_tok)
                acc[t] += dq * input_all[(long)t * (long)cols + elem];
    }

    #pragma unroll
    for (int t = 0; t < NT; t++)
        if (t < n_tok) {
            float result = sharpi_warp_reduce_sum(acc[t]);
            if (lane == 0) output_all[(long)t * (long)rows + row] = result;
        }
}

// Q8_0 SoA (#149 layout): identical reduction, aligned SoA weight reads.
extern ""C"" __global__ void llm_matvec_q8_0_ws_soa_n__NT__(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input_all,
    float* __restrict__ output_all,
    int rows, int cols, int n_tok)
{
    const int NT = __NT__;
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 5;
    long qrow = (long)row * cols;
    const unsigned short* scales = (const unsigned short*)((const char*)weights + (long)rows * cols);
    long srow = (long)row * num_blocks;

    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;

    for (int block = 0; block < num_blocks; block++) {
        float d = sharpi_fp16_to_fp32(scales[srow + block]);
        int q = sharpi_int8_at(weights, qrow + (long)block * 32 + lane);
        float dq = d * (float)q;
        int elem = block * 32 + lane;
        #pragma unroll
        for (int t = 0; t < NT; t++)
            if (t < n_tok)
                acc[t] += dq * input_all[(long)t * (long)cols + elem];
    }

    #pragma unroll
    for (int t = 0; t < NT; t++)
        if (t < n_tok) {
            float result = sharpi_warp_reduce_sum(acc[t]);
            if (lane == 0) output_all[(long)t * (long)rows + row] = result;
        }
}

// Q6_K: hoists the 8 token-invariant dequantized weight values per 256-block —
// the same (scX * decode) left-fold llm_matvec_q6k_gemm_n contracts to before
// the activation multiply.
extern ""C"" __global__ void llm_matvec_q6k_ws_n__NT__(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input_all,   // [n_tok][cols]
    float* __restrict__ output_all,        // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    const int NT = __NT__;
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long row_base_bytes = (long)row * (long)num_blocks * 210L;

    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;

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

        float w0 = sc0 * (float)((int)((ql0 & 0xFu)        | (((qh0 >> 0) & 3u) << 4)) - 32);
        float w1 = sc1 * (float)((int)((ql1 & 0xFu)        | (((qh0 >> 2) & 3u) << 4)) - 32);
        float w2 = sc2 * (float)((int)(((ql0 >> 4) & 0xFu) | (((qh0 >> 4) & 3u) << 4)) - 32);
        float w3 = sc3 * (float)((int)(((ql1 >> 4) & 0xFu) | (((qh0 >> 6) & 3u) << 4)) - 32);
        float w4 = sc4 * (float)((int)((ql2 & 0xFu)        | (((qh1 >> 0) & 3u) << 4)) - 32);
        float w5 = sc5 * (float)((int)((ql3 & 0xFu)        | (((qh1 >> 2) & 3u) << 4)) - 32);
        float w6 = sc6 * (float)((int)(((ql2 >> 4) & 0xFu) | (((qh1 >> 4) & 3u) << 4)) - 32);
        float w7 = sc7 * (float)((int)(((ql3 >> 4) & 0xFu) | (((qh1 >> 6) & 3u) << 4)) - 32);

        int base_elem = block * 256;
        #pragma unroll
        for (int t = 0; t < NT; t++)
            if (t < n_tok) {
                const float* input = input_all + (long)t * (long)cols;
                acc[t] += w0 * input[base_elem +       lane];
                acc[t] += w1 * input[base_elem +  32 + lane];
                acc[t] += w2 * input[base_elem +  64 + lane];
                acc[t] += w3 * input[base_elem +  96 + lane];
                acc[t] += w4 * input[base_elem + 128 + lane];
                acc[t] += w5 * input[base_elem + 160 + lane];
                acc[t] += w6 * input[base_elem + 192 + lane];
                acc[t] += w7 * input[base_elem + 224 + lane];
            }
    }

    #pragma unroll
    for (int t = 0; t < NT; t++)
        if (t < n_tok) {
            float result = sharpi_warp_reduce_sum(acc[t]);
            if (lane == 0) output_all[(long)t * (long)rows + row] = result;
        }
}

// Q5_K: hoists the token-invariant (d1*q - dm1) per element pair — the same
// fma llm_matvec_q5k_gemm_n contracts to before the activation multiply.
extern ""C"" __global__ void llm_matvec_q5k_ws_n__NT__(
    const unsigned int* __restrict__ weights,
    const float* __restrict__ input_all,   // [n_tok][cols]
    float* __restrict__ output_all,        // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    const int NT = __NT__;
    const int N_ROWS = 8;
    const int THREADS_PER_ROW = 32;
    unsigned int tid = threadIdx.x;
    int row_in_wg = (int)tid / THREADS_PER_ROW;
    int lane = (int)tid & (THREADS_PER_ROW - 1);
    int row = (int)blockIdx.x * N_ROWS + row_in_wg;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long row_base_bytes = (long)row * (long)num_blocks * 176L;

    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;

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

            float w_lo = d1 * (float)((int)low4 + hLo) - dm1;
            float w_hi = d2 * (float)((int)hi4  + hHi) - dm2;

            #pragma unroll
            for (int t = 0; t < NT; t++)
                if (t < n_tok) {
                    const float* input = input_all + (long)t * (long)cols;
                    acc[t] += w_lo * input[elem_lo];
                    acc[t] += w_hi * input[elem_hi];
                }
        }
    }

    #pragma unroll
    for (int t = 0; t < NT; t++)
        if (t < n_tok) {
            float result = sharpi_warp_reduce_sum(acc[t]);
            if (lane == 0) output_all[(long)t * (long)rows + row] = result;
        }
}

// Q4_K (interleaved AoS, dp4a over Q8_1 activations): grid = rows, block =
// 32 × MATVEC_Q4K_NWARPS — llm_matvec_q4k_gemm_n minus the token grid dimension.
// The weight words and the hoisted scale products (super_d*sc, super_dmin*mn —
// the same left-fold the GEMM-N coef_* computes) are loaded once per block
// iteration and applied to all n_tok Q8_1 activation rows.
extern ""C"" __global__ void llm_matvec_q4k_ws_n__NT__(
    const unsigned int* __restrict__ weights,
    const unsigned char* __restrict__ y_q81_all,  // [n_tok][num_blocks*8*36] bytes
    float* __restrict__ output_all,               // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    const int NT = __NT__;
    int row     = (int)blockIdx.x;
    int warp_id = (int)threadIdx.y;
    int lane    = (int)threadIdx.x;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long word_row_base = (long)row * (long)num_blocks * 36L;
    // q81 row stride = (cols/32) sub-blocks × 36 bytes = num_blocks*8*36.
    long tok_stride = (long)num_blocks * 8L * 36L;

    int chunk    = lane >> 3;
    int byte_off = (lane & 7) * 4;
    int q4_offset = 4 + chunk * 8 + (lane & 7);

    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;

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

        float sd_sc_lo = super_d    * (float)sc_lo;
        float sm_mn_lo = super_dmin * (float)mn_lo;
        float sd_sc_hi = super_d    * (float)sc_hi;
        float sm_mn_hi = super_dmin * (float)mn_hi;

        long q81_base_lo = (long)(block * 8 + chunk * 2)     * 36L;
        long q81_base_hi = (long)(block * 8 + chunk * 2 + 1) * 36L;

        #pragma unroll
        for (int t = 0; t < NT; t++)
            if (t < n_tok) {
                const unsigned char* y_q81 = y_q81_all + (long)t * tok_stride;

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

                float coef_d_lo = sd_sc_lo * d8_lo;
                float coef_m_lo = sm_mn_lo * d8_lo;
                float coef_d_hi = sd_sc_hi * d8_hi;
                float coef_m_hi = sm_mn_hi * d8_hi;
                acc[t] += coef_d_lo * (float)dot_lo - coef_m_lo * (float)sum_lo;
                acc[t] += coef_d_hi * (float)dot_hi - coef_m_hi * (float)sum_hi;
            }
    }

    #pragma unroll
    for (int t = 0; t < NT; t++)
        if (t < n_tok)
            acc[t] = sharpi_warp_reduce_sum(acc[t]);

    __shared__ float warp_acc[NT * MATVEC_Q4K_NWARPS];
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < NT; t++)
            if (t < n_tok)
                warp_acc[t * MATVEC_Q4K_NWARPS + warp_id] = acc[t];
    }
    __syncthreads();

    // Warp 0, lane t finishes token t (n_tok <= 16 < 32): the w=0..7 partial-sum
    // order matches the GEMM-N kernel's warp_id loop exactly; tokens are independent.
    if (warp_id == 0 && lane < n_tok) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q4K_NWARPS; w++)
            s += warp_acc[lane * MATVEC_Q4K_NWARPS + w];
        output_all[(long)lane * (long)rows + row] = s;
    }
}

// Q4_K over the scale-pre-unpacked SoA weight (#156): identical reduction to
// llm_matvec_q4k_ws_n__NT__, only the weight decode reads the SoA regions.
extern ""C"" __global__ void llm_matvec_q4k_ws_soa_n__NT__(
    const unsigned int* __restrict__ weights,     // SoA: [Q][S][D]
    const unsigned char* __restrict__ y_q81_all,  // [n_tok][num_blocks*8*36] bytes
    float* __restrict__ output_all,               // [n_tok][rows]
    int rows, int cols, int n_tok)
{
    const int NT = __NT__;
    int row     = (int)blockIdx.x;
    int warp_id = (int)threadIdx.y;
    int lane    = (int)threadIdx.x;
    if (row >= rows) return;

    int num_blocks = cols >> 8;
    long totalSub  = (long)rows * num_blocks;
    const unsigned char* qReg = (const unsigned char*)weights;
    const unsigned char* sReg = qReg + totalSub * 128L;
    const unsigned int*  dReg = (const unsigned int*)(sReg + totalSub * 16L);

    long tok_stride = (long)num_blocks * 8L * 36L;

    int chunk    = lane >> 3;
    int byte_off = (lane & 7) * 4;
    int q_word_in_block = chunk * 8 + (lane & 7);

    long row_blk_base = (long)row * num_blocks;

    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;

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

        float sd_sc_lo = super_d    * (float)sc_lo;
        float sm_mn_lo = super_dmin * (float)mn_lo;
        float sd_sc_hi = super_d    * (float)sc_hi;
        float sm_mn_hi = super_dmin * (float)mn_hi;

        long q81_base_lo = (long)(block * 8 + chunk * 2)     * 36L;
        long q81_base_hi = (long)(block * 8 + chunk * 2 + 1) * 36L;

        #pragma unroll
        for (int t = 0; t < NT; t++)
            if (t < n_tok) {
                const unsigned char* y_q81 = y_q81_all + (long)t * tok_stride;

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

                float coef_d_lo = sd_sc_lo * d8_lo;
                float coef_m_lo = sm_mn_lo * d8_lo;
                float coef_d_hi = sd_sc_hi * d8_hi;
                float coef_m_hi = sm_mn_hi * d8_hi;
                acc[t] += coef_d_lo * (float)dot_lo - coef_m_lo * (float)sum_lo;
                acc[t] += coef_d_hi * (float)dot_hi - coef_m_hi * (float)sum_hi;
            }
    }

    #pragma unroll
    for (int t = 0; t < NT; t++)
        if (t < n_tok)
            acc[t] = sharpi_warp_reduce_sum(acc[t]);

    __shared__ float warp_acc[NT * MATVEC_Q4K_NWARPS];
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < NT; t++)
            if (t < n_tok)
                warp_acc[t * MATVEC_Q4K_NWARPS + warp_id] = acc[t];
    }
    __syncthreads();

    if (warp_id == 0 && lane < n_tok) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < MATVEC_Q4K_NWARPS; w++)
            s += warp_acc[lane * MATVEC_Q4K_NWARPS + w];
        output_all[(long)lane * (long)rows + row] = s;
    }
}
";
}
