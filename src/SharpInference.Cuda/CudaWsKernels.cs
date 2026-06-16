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
///
/// <para><b>#201:</b> <c>llm_matvec_q6k_ws_sw_n*</c> (scale-word) replaces the Q6_K
/// kernel's 10 dependent per-super-block scale/d byte-gather loads with five aligned
/// word loads + funnel-shift extracts — same bytes, same chain, bit-identical; it
/// attacks the gather latency the serial walk stalls on. Two sibling #201 attempts
/// were measured and REMOVED — re-read the issue before re-trying them: token-warp
/// (one warp per (row, token), bit-exact by construction) loses everywhere because
/// the NT× weight-decode replication outweighs the occupancy win (N=8 aggregate
/// 243→239 even routed only to the grid-starved attn_v shape, 243→182 routed to
/// ffn_down too); SoA-Q8_1-activation reads (xs) genuinely cut the LSU/L1TEX load
/// (94.7% → 82%) but the second activation-region pointer costs 7 registers in the
/// NT-unrolled token loop (48 → 55) and the occupancy loss (80% → 63%) nets −5 t/s
/// (forcing 48 via launch bounds spills: −36 t/s). The bit-identity contract freezes
/// the lane geometry, so the remaining q4k headroom belongs to the argmax-stable
/// decode MMQ (<c>SHARPI_BATCH_DECODE_MMQ=1</c>), not to load-shape tweaks.</para>
/// </summary>
internal static class CudaWsKernels
{
    /// <summary>Compile-time batch capacities stamped for each kernel body. Order
    /// matters: dispatch indexes kernel-handle arrays by position.</summary>
    internal static readonly int[] Variants = [2, 4, 8, 16];

    /// <summary>All weight-stationary kernels (one instantiation per variant) plus the
    /// #201/#205 decode-MMQ kernel at both row-tile sizes (BM=64 default, BM=32 for
    /// grid-starved low-row shapes).</summary>
    public static string Source { get; } = Build();

    private static string Build()
    {
        var sb = new System.Text.StringBuilder(
            Template.Length * Variants.Length
            + (DecodeMmqTemplate.Length + DecodeMmqQ6KTemplate.Length) * 2);
        foreach (int nt in Variants)
            sb.Append(Template.Replace("__NT__", nt.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        // Decode MMQ, two row-tile sizes (#205). BM/16 row-strips: BM=64 → wr = warp & 3,
        // wc = warp >> 2; BM=32 → wr = warp & 1, wc = warp >> 1. Quant staging is 8 iters
        // either way (MMQ_BM*32 == 8*threads); act staging is (16*64)/threads = 4 / 8 iters.
        sb.Append(EmitDecodeMmq(DecodeMmqTemplate, bm: 64, threads: 256, actJ: 4, wrMask: 3, wcShift: 2, suffix: ""));
        sb.Append(EmitDecodeMmq(DecodeMmqTemplate, bm: 32, threads: 128, actJ: 8, wrMask: 1, wcShift: 1, suffix: "_bm32"));
        // #204 Q6_K decode MMQ: same two row-tiles. The Q region is 256 int8 / super-block
        // (64 words/row) — quant staging is MMQ_BM*64/threads = 16 / 16 iters. The SoA weight
        // is now the ONLY copy of the Q6_K weight (RepackQ6KSoa frees the AoS), so there is no
        // AoS-direct decode-MMQ variant — every Q6_K reader is SoA-aware.
        sb.Append(EmitDecodeMmq(DecodeMmqQ6KTemplate, bm: 64, threads: 256, actJ: 4, wrMask: 3, wcShift: 2, suffix: "",
                                wQuantJ: 16));
        sb.Append(EmitDecodeMmq(DecodeMmqQ6KTemplate, bm: 32, threads: 128, actJ: 8, wrMask: 1, wcShift: 1, suffix: "_bm32",
                                wQuantJ: 16));
        return sb.ToString();
    }

    private static string EmitDecodeMmq(string template, int bm, int threads, int actJ, int wrMask, int wcShift,
                                        string suffix, int wQuantJ = 0)
    {
        static string S(int v) => v.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return template
            .Replace("__BM__", S(bm))
            .Replace("__NTHREADS__", S(threads))
            .Replace("__ACTJ__", S(actJ))
            .Replace("__WQUANTJ__", S(wQuantJ))
            .Replace("__WRMASK__", S(wrMask))
            .Replace("__WCSHIFT__", S(wcShift))
            .Replace("__SUF__", suffix);
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

// Q6_K SoA (#204): bit-identical to llm_matvec_q6k_ws_n* (and to its scale-word
// twin llm_matvec_q6k_ws_sw_n*) over the scale-pre-unpacked SoA layout (see
// llm_matvec_q6k_soa). The 16 int8 scales are already separate, so the scale-word
// trick is moot — read S[g*16 + 2k + (lane>>4)] directly; the (q6−32) int8 weight is
// Q[g*256 + e], d is D[g*4]. Same 8-element-group reduction order + same token loop,
// so the output is bit-identical to both AoS Q6_K WS kernels.
extern ""C"" __global__ void llm_matvec_q6k_ws_soa_n__NT__(
    const unsigned char* __restrict__ weights,   // SoA [Q][S][D]
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
    long total_sb = (long)rows * num_blocks;
    const signed char*   qReg = (const signed char*)weights;
    const signed char*   sReg = (const signed char*)weights + total_sb * 256L;
    const unsigned char*  dReg = (const unsigned char*)weights + total_sb * (256L + 16L);

    long isc = (long)(lane >> 4);

    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;

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

        float w0 = sc0 * (float)q[       lane];
        float w1 = sc1 * (float)q[ 32 + lane];
        float w2 = sc2 * (float)q[ 64 + lane];
        float w3 = sc3 * (float)q[ 96 + lane];
        float w4 = sc4 * (float)q[128 + lane];
        float w5 = sc5 * (float)q[160 + lane];
        float w6 = sc6 * (float)q[192 + lane];
        float w7 = sc7 * (float)q[224 + lane];

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

// ── #201 Q6_K scale-word (sw) variant ──────────────────────────────────────
// Identical geometry (8 rows/block × 32 lanes, token loop inside) and identical
// reduction chain to llm_matvec_q6k_ws_n*, but the per-super-block scale/d
// decode loads the 18-byte tail [scales[16] ‖ fp16 d] (bytes 192..209) as five
// aligned words + funnel-shift extracts instead of 10 separate byte-gather
// loads (8 × sharpi_int8_at + 2 d bytes). Same bytes → same int8 / fp16 values
// → bit-identical; the serial super-block walk #201 profiled as latency-bound
// (ffn_down 48% pipe / 19% DRAM) gets 5 independent LDGs instead of a 10-deep
// gather burst, and the LSU issues ~30% fewer weight-load instructions.
extern ""C"" __global__ void llm_matvec_q6k_ws_sw_n__NT__(
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

    // Scale-byte select: original lanes read byte 192 + 2k + isc, isc = lane>>4.
    unsigned int isc8 = ((unsigned int)lane >> 4) * 8u;

    for (int block = 0; block < num_blocks; block++) {
        long b0 = row_base_bytes + (long)block * 210L;

        // b0 is always even (210-byte stride from an even base), so the tail
        // starts at word offset 0 or 2 and five aligned words cover all 18 bytes.
        // The word covering byte 209 spills ≤2 bytes into the next block — the
        // same word sharpi_byte_at(b0+209) already reads in the #194 kernel.
        long sc_word = (b0 + 192) >> 2;
        unsigned int sw0 = weights[sc_word];
        unsigned int sw1 = weights[sc_word + 1];
        unsigned int sw2 = weights[sc_word + 2];
        unsigned int sw3 = weights[sc_word + 3];
        unsigned int sw4 = weights[sc_word + 4];
        unsigned int s01, s23, s45, s67, d_bits;
        if (((unsigned int)b0 & 3u) != 0u) {   // warp-uniform (b0 is per-block)
            s01 = __funnelshift_r(sw0, sw1, 16);
            s23 = __funnelshift_r(sw1, sw2, 16);
            s45 = __funnelshift_r(sw2, sw3, 16);
            s67 = __funnelshift_r(sw3, sw4, 16);
            d_bits = sw4 >> 16;
        } else {
            s01 = sw0; s23 = sw1; s45 = sw2; s67 = sw3;
            d_bits = sw4 & 0xffffu;
        }
        float d = sharpi_fp16_to_fp32(d_bits);

        // Same byte, same sign-widening as sharpi_int8_at(weights, b0+192+2k+isc).
        int v0 = (int)((s01 >> (isc8      )) & 0xFFu); v0 = v0 >= 128 ? v0 - 256 : v0;
        int v1 = (int)((s01 >> (isc8 + 16u)) & 0xFFu); v1 = v1 >= 128 ? v1 - 256 : v1;
        int v2 = (int)((s23 >> (isc8      )) & 0xFFu); v2 = v2 >= 128 ? v2 - 256 : v2;
        int v3 = (int)((s23 >> (isc8 + 16u)) & 0xFFu); v3 = v3 >= 128 ? v3 - 256 : v3;
        int v4 = (int)((s45 >> (isc8      )) & 0xFFu); v4 = v4 >= 128 ? v4 - 256 : v4;
        int v5 = (int)((s45 >> (isc8 + 16u)) & 0xFFu); v5 = v5 >= 128 ? v5 - 256 : v5;
        int v6 = (int)((s67 >> (isc8      )) & 0xFFu); v6 = v6 >= 128 ? v6 - 256 : v6;
        int v7 = (int)((s67 >> (isc8 + 16u)) & 0xFFu); v7 = v7 >= 128 ? v7 - 256 : v7;

        float sc0 = d * (float)v0;
        float sc1 = d * (float)v1;
        float sc2 = d * (float)v2;
        float sc3 = d * (float)v3;
        float sc4 = d * (float)v4;
        float sc5 = d * (float)v5;
        float sc6 = d * (float)v6;
        float sc7 = d * (float)v7;

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
";

    /// <summary>
    /// #201 decode MMQ (<c>SHARPI_BATCH_DECODE_MMQ=1</c>): the bit-exact WS kernels
    /// are frozen into a lane geometry whose token loop costs ~30 L1TEX wavefront-heavy
    /// load instructions per 36-word super-block at NT=8 (~3× the weight-streaming
    /// floor, #201). This kernel relaxes the contract to argmax-stable — the same
    /// contract the prefill MMQ holds — and re-tiles <c>llm_mmq_q4k_soa_acts</c> for
    /// decode batch sizes: BN drops 128 → 16, so grid.y == 1 for N ≤ 16 and each
    /// weight byte is read from HBM exactly once per step (true weight-stationary)
    /// while the m16n8k32 int8 mma replaces the per-token dp4a chains. Identical
    /// K-order accumulation and {d,s} fixup math to the prefill kernel — only the
    /// tile shape changes (one n-tile per warp, scalar acc0..3 instead of acc[8][4]).
    ///
    /// <para><b>#205:</b> emitted at two row-tile sizes from one template. BM=64 (256
    /// threads, 8 warps) is the default. BM=32 (<c>_bm32</c>, 128 threads, 4 warps)
    /// doubles the grid to ceil(rows/32) blocks — the dispatcher routes the grid-starved
    /// low-row shapes (Q/O proj, the Q4_K ffn_down half: rows≈4096 → only 64 blocks on a
    /// 60-SM card) there so they fill ≥2 waves instead of stalling at ~1. Output is
    /// bit-identical to BM=64 (same fragments / mma / accumulation order; only the
    /// block→row mapping changes), so both share the argmax-stable oracle.</para>
    /// </summary>
    private const string DecodeMmqTemplate = @"
// ── #201/#205 decode MMQ: Q4_K SoA weights × SoA Q8_1 activations, BN=16 ─────
// grid = (ceil(rows/MMQ_BM), ceil(n_tok/16)), block = __NTHREADS__ ((MMQ_BM/16)
// row-strips × 2 token-strips). The K-step is one 256-element SUPER-block: the
// block stages the raw tile (MMQ_BM×32 quant words + scale tails + 16×64 act
// words + 16×8 act {d,s}) with linear fully-coalesced copies, then runs 8 m16n8k32 s8 mma
// per warp between one barrier pair — nibble/scale decode happens at
// fragment-read time from shared. A sub-block-sized K-step (one mma per
// barrier pair) was measured 1.4-1.7× SLOWER than the WS matvec it replaces:
// ~50 cycles of mma cannot hide the next tile's global-load latency, so the
// loop serializes at one latency epoch per 32 elements. The super-block step
// issues ~14 independent loads per thread per epoch and amortizes it 8×.
// Argmax-stable, not bit-exact: both operands int8-quantized, min-bias via the
// fp16 s = d·Σq field (same contract and K-order as the prefill MMQ).
#define MMQ_BM __BM__
#define MMQ_BN 16
extern ""C"" __global__ void __launch_bounds__(__NTHREADS__) llm_mmq_q4k_soa_acts_n16__SUF__(
    const unsigned int*  __restrict__ weights,   // SoA [Q][S][D]
    const unsigned int*  __restrict__ y_qs,      // SoA Q8_1 quants [n_tok × sub_total × 32 B]
    const unsigned int*  __restrict__ y_ds,      // SoA Q8_1 scales [n_tok × sub_total] {d,s}
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    // Row strides padded +1 word: an unpadded 32/64-word stride would land all 8
    // grp-lanes of a fragment read on one shared bank (8-way conflict).
    __shared__ unsigned int sWq[MMQ_BM * 33];    // raw quant words   [row][32+1]
    __shared__ unsigned int sWs[MMQ_BM * 4];     // raw 16-B sc|mn    [row][4]
    __shared__ unsigned int sWd[MMQ_BM];         // raw {d,dmin}      [row]
    __shared__ unsigned int sYq[MMQ_BN * 65];    // raw act words     [tok][64+1]
    __shared__ unsigned int sYd[MMQ_BN * 8];     // raw act {d,s}     [tok][8]

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
    int wr   = warp & __WRMASK__;
    int wc   = warp >> __WCSHIFT__;
    int mrow0 = wr * 16;
    int ncol0 = wc * 8;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    int tokC0 = tok_block + ncol0 + tig * 2;
    int tokC1 = tokC0 + 1;

    for (int ksb = 0; ksb < nb_super; ksb++) {
        // Stage the raw super-block tile. All copies are linear in shared and
        // 128-B-coalesced in global (one row / token segment per warp-instruction).
        #pragma unroll
        for (int j = 0; j < 8; j++) {            // 64 rows × 32 quant words
            int k = tid + j * __NTHREADS__;
            int r = row_block + (k >> 5);
            sWq[(k >> 5) * 33 + (k & 31)] =
                (r < rows) ? qReg[((long)r * nb_super + ksb) * 32L + (k & 31)] : 0u;
        }
        {                                        // 64 rows × 4 scale words (16 B sc|mn)
            int r = row_block + (tid >> 2);
            sWs[tid] = (r < rows)
                ? reinterpret_cast<const unsigned int*>(sReg)[((long)r * nb_super + ksb) * 4L + (tid & 3)]
                : 0u;
        }
        if (tid < MMQ_BM) {                      // 64 rows × {d,dmin}
            int r = row_block + tid;
            sWd[tid] = (r < rows) ? dReg[(long)r * nb_super + ksb] : 0u;
        }
        #pragma unroll
        for (int j = 0; j < __ACTJ__; j++) {     // 16 tokens × 64 act words (256 B)
            int k = tid + j * __NTHREADS__;
            int t = tok_block + (k >> 6);
            sYq[(k >> 6) * 65 + (k & 63)] =
                (t < n_tok) ? y_qs[((long)t * sub_total + ksb * 8) * 8L + (k & 63)] : 0u;
        }
        if (tid < MMQ_BN * 8) {                  // 16 tokens × 8 act {d,s}
            int t = tok_block + (tid >> 3);
            sYd[tid] = (t < n_tok) ? y_ds[(long)t * sub_total + ksb * 8 + (tid & 7)] : 0u;
        }
        __syncthreads();

        // 8 sub-blocks per super-block: same per-sub-block fixup math and K-order
        // as llm_mmq_q4k_soa_acts; nibble polarity and scale bytes decode here.
        const unsigned char* sWsB = reinterpret_cast<const unsigned char*>(sWs);
        #pragma unroll
        for (int sb = 0; sb < 8; sb++) {
            int chk = sb >> 1, pol = sb & 1;
            unsigned int wa0 = sWq[(mrow0 + grp) * 33     + chk * 8 + tig];
            unsigned int wa1 = sWq[(mrow0 + grp + 8) * 33 + chk * 8 + tig];
            unsigned int wa2 = sWq[(mrow0 + grp) * 33     + chk * 8 + tig + 4];
            unsigned int wa3 = sWq[(mrow0 + grp + 8) * 33 + chk * 8 + tig + 4];
            int a0 = (int)(pol ? ((wa0 >> 4) & 0x0F0F0F0Fu) : (wa0 & 0x0F0F0F0Fu));
            int a1 = (int)(pol ? ((wa1 >> 4) & 0x0F0F0F0Fu) : (wa1 & 0x0F0F0F0Fu));
            int a2 = (int)(pol ? ((wa2 >> 4) & 0x0F0F0F0Fu) : (wa2 & 0x0F0F0F0Fu));
            int a3 = (int)(pol ? ((wa3 >> 4) & 0x0F0F0F0Fu) : (wa3 & 0x0F0F0F0Fu));

            unsigned int ddA = sWd[mrow0 + grp];
            unsigned int ddB = sWd[mrow0 + grp + 8];
            float dwA = sharpi_fp16_to_fp32(ddA & 0xffffu) * (float)sWsB[(mrow0 + grp) * 16 + sb];
            float dmA = sharpi_fp16_to_fp32(ddA >> 16)     * (float)sWsB[(mrow0 + grp) * 16 + 8 + sb];
            float dwB = sharpi_fp16_to_fp32(ddB & 0xffffu) * (float)sWsB[(mrow0 + grp + 8) * 16 + sb];
            float dmB = sharpi_fp16_to_fp32(ddB >> 16)     * (float)sWsB[(mrow0 + grp + 8) * 16 + 8 + sb];

            int b0 = (int)sYq[(ncol0 + grp) * 65 + sb * 8 + tig];
            int b1 = (int)sYq[(ncol0 + grp) * 65 + sb * 8 + tig + 4];
            unsigned int dy0 = sYd[(ncol0 + tig * 2) * 8 + sb];
            unsigned int dy1 = sYd[(ncol0 + tig * 2 + 1) * 8 + sb];
            float dC0 = sharpi_fp16_to_fp32(dy0 & 0xffffu), sC0 = sharpi_fp16_to_fp32(dy0 >> 16);
            float dC1 = sharpi_fp16_to_fp32(dy1 & 0xffffu), sC1 = sharpi_fp16_to_fp32(dy1 >> 16);

            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(c0), ""+r""(c1), ""+r""(c2), ""+r""(c3)
              : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));
            acc0 += (float)c0 * dwA * dC0 - dmA * sC0;
            acc1 += (float)c1 * dwA * dC1 - dmA * sC1;
            acc2 += (float)c2 * dwB * dC0 - dmB * sC0;
            acc3 += (float)c3 * dwB * dC1 - dmB * sC1;
        }
        __syncthreads();
    }

    if (rowA < rows) {
        if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc0;
        if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc1;
    }
    if (rowB < rows) {
        if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc2;
        if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc3;
    }
}
#undef MMQ_BM
#undef MMQ_BN
";

    /// <summary>
    /// #204 decode MMQ for Q6_K weights: the int8 tensor-core analogue of
    /// <see cref="DecodeMmqTemplate"/> for the Q6_K trunk shapes (Qwen3-8B Q4_K_M keeps
    /// half of ffn_down + attn_v + lm-head in Q6_K), which otherwise fall to the bit-exact
    /// <c>llm_matvec_q6k_ws_sw</c> matvec (~2.7-3× the weight-streaming floor at N=8).
    ///
    /// <para>The SoA weight (<c>llm_q6k_repack_soa</c>) stores the signed int8
    /// <c>(q6 − 32)</c> for natural element e at byte e — the SAME natural order the
    /// shared Q8_1 activation uses — so the a-fragment load and the b-fragment (activation)
    /// load mirror the Q4_K tile exactly. Q6_K is SYMMETRIC: no min term, the −32 offset
    /// is baked into the int8 weight. The only structural change vs the Q4_K tile is the
    /// per-16-element scale boundary inside each 32-element sub-block: registers a0/a1 of
    /// the m16n8k32 fragment hold k∈[0,16), a2/a3 hold k∈[16,32) — a clean split at that
    /// boundary — so each sub-block runs TWO masked mmas (mma1 = {a0,a1,0,0}, mma2 =
    /// {0,0,a2,a3}) and accumulates <c>c1·dwG0·dC + c2·dwG1·dC</c> with dwG0 = d·scales[2·sb],
    /// dwG1 = d·scales[2·sb+1]. Argmax-stable (both operands int8-quantized), not bit-exact —
    /// same contract as the Q4_K decode/prefill MMQ.</para>
    /// </summary>
    private const string DecodeMmqQ6KTemplate = @"
// ── #204 decode MMQ: Q6_K SoA weights × SoA Q8_1 activations, BN=16 ──────────
// grid = (ceil(rows/MMQ_BM), ceil(n_tok/16)), block = __NTHREADS__. K-step is one
// 256-element super-block: stage the raw tile (MMQ_BM×256 int8 quant words + 16-B
// scales + {d} + 16×64 act words + 16×8 act {d,s}) with coalesced copies, then 8
// sub-blocks × TWO m16n8k32 s8 mma per warp between one barrier pair. The two
// half-fragment mmas split the per-16-element Q6_K scale boundary; the −32 offset is
// in the int8 weight so there is no min term. Argmax-stable (both operands int8).
#define MMQ_BM __BM__
#define MMQ_BN 16
extern ""C"" __global__ void __launch_bounds__(__NTHREADS__) llm_mmq_q6k_soa_acts_n16__SUF__(
    const unsigned int*  __restrict__ weights,   // SoA [Q total*256][S total*16][D total*4]
    const unsigned int*  __restrict__ y_qs,      // SoA Q8_1 quants [n_tok × sub_total × 32 B]
    const unsigned int*  __restrict__ y_ds,      // SoA Q8_1 scales [n_tok × sub_total] {d,s}
    float* __restrict__ output,                  // [n_tok × rows] fp32
    int rows, int cols, int n_tok)
{
    // 64 quant words/row (256 int8). +1 word stride avoids the 8-way bank conflict an
    // unpadded 64-word stride would land on the 8 grp-lanes of a fragment read.
    __shared__ unsigned int sWq[MMQ_BM * 65];    // raw int8 quant words [row][64+1]
    __shared__ unsigned int sWs[MMQ_BM * 4];     // 16 int8 scales       [row][4]
    __shared__ unsigned int sWd[MMQ_BM];         // {d, 0}               [row]
    __shared__ unsigned int sYq[MMQ_BN * 65];    // raw act words        [tok][64+1]
    __shared__ unsigned int sYd[MMQ_BN * 8];     // raw act {d,s}        [tok][8]

    int row_block = (int)blockIdx.x * MMQ_BM;
    int tok_block = (int)blockIdx.y * MMQ_BN;
    int nb_super  = cols >> 8;
    int sub_total = cols >> 5;
    long total_sb = (long)rows * nb_super;

    const unsigned int*  qReg = weights;                                          // [Q] total*256 B
    const unsigned char* sReg = (const unsigned char*)weights + total_sb * 256L;  // [S] total*16 B
    const unsigned int*  dReg = (const unsigned int*)(sReg + total_sb * 16L);     // [D] total*4 B

    int tid  = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp  = lane >> 2;
    int tig  = lane & 3;
    int wr   = warp & __WRMASK__;
    int wc   = warp >> __WCSHIFT__;
    int mrow0 = wr * 16;
    int ncol0 = wc * 8;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    int rowA = row_block + mrow0 + grp;
    int rowB = rowA + 8;
    int tokC0 = tok_block + ncol0 + tig * 2;
    int tokC1 = tokC0 + 1;

    for (int ksb = 0; ksb < nb_super; ksb++) {
        // Stage the raw super-block tile, fully coalesced.
        #pragma unroll
        for (int j = 0; j < __WQUANTJ__; j++) {  // MMQ_BM rows × 64 quant words
            int k = tid + j * __NTHREADS__;
            int r = row_block + (k >> 6);
            sWq[(k >> 6) * 65 + (k & 63)] =
                (r < rows) ? qReg[((long)r * nb_super + ksb) * 64L + (k & 63)] : 0u;
        }
        {                                        // MMQ_BM rows × 4 scale words (16 int8)
            int r = row_block + (tid >> 2);
            sWs[tid] = (r < rows)
                ? reinterpret_cast<const unsigned int*>(sReg)[((long)r * nb_super + ksb) * 4L + (tid & 3)]
                : 0u;
        }
        if (tid < MMQ_BM) {                      // MMQ_BM rows × {d,0}
            int r = row_block + tid;
            sWd[tid] = (r < rows) ? dReg[(long)r * nb_super + ksb] : 0u;
        }
        #pragma unroll
        for (int j = 0; j < __ACTJ__; j++) {     // 16 tokens × 64 act words (256 B)
            int k = tid + j * __NTHREADS__;
            int t = tok_block + (k >> 6);
            sYq[(k >> 6) * 65 + (k & 63)] =
                (t < n_tok) ? y_qs[((long)t * sub_total + ksb * 8) * 8L + (k & 63)] : 0u;
        }
        if (tid < MMQ_BN * 8) {                  // 16 tokens × 8 act {d,s}
            int t = tok_block + (tid >> 3);
            sYd[tid] = (t < n_tok) ? y_ds[(long)t * sub_total + ksb * 8 + (tid & 7)] : 0u;
        }
        __syncthreads();

        const signed char* sWsB = reinterpret_cast<const signed char*>(sWs);
        #pragma unroll
        for (int sb = 0; sb < 8; sb++) {
            // a-fragment: 4 consecutive int8 per word, natural-order (mirrors the Q4_K
            // tile's word load but reads int8 directly). a0/a1 = k∈[0,16) (words tig of
            // the lower 16 lanes), a2/a3 = k∈[16,32) (words tig+4). The 65-word row stride
            // is uint; sub-block sb occupies int8 bytes [sb*32 .. sb*32+31] = uint words
            // [sb*8 .. sb*8+7] within the row.
            int wbaseA = (mrow0 + grp) * 65     + sb * 8;
            int wbaseB = (mrow0 + grp + 8) * 65 + sb * 8;
            int a0 = (int)sWq[wbaseA + tig];
            int a1 = (int)sWq[wbaseB + tig];
            int a2 = (int)sWq[wbaseA + tig + 4];
            int a3 = (int)sWq[wbaseB + tig + 4];

            // Q6_K symmetric scales: sub-block sb's first 16 elems use scales[2*sb],
            // second 16 use scales[2*sb+1]. d in low 16 bits of sWd.
            float dA = sharpi_fp16_to_fp32(sWd[mrow0 + grp]     & 0xffffu);
            float dB = sharpi_fp16_to_fp32(sWd[mrow0 + grp + 8] & 0xffffu);
            float dwA0 = dA * (float)sWsB[(mrow0 + grp) * 16 + 2 * sb];
            float dwA1 = dA * (float)sWsB[(mrow0 + grp) * 16 + 2 * sb + 1];
            float dwB0 = dB * (float)sWsB[(mrow0 + grp + 8) * 16 + 2 * sb];
            float dwB1 = dB * (float)sWsB[(mrow0 + grp + 8) * 16 + 2 * sb + 1];

            int b0 = (int)sYq[(ncol0 + grp) * 65 + sb * 8 + tig];
            int b1 = (int)sYq[(ncol0 + grp) * 65 + sb * 8 + tig + 4];
            unsigned int dy0 = sYd[(ncol0 + tig * 2) * 8 + sb];
            unsigned int dy1 = sYd[(ncol0 + tig * 2 + 1) * 8 + sb];
            float dC0 = sharpi_fp16_to_fp32(dy0 & 0xffffu);
            float dC1 = sharpi_fp16_to_fp32(dy1 & 0xffffu);

            // mma1 = {a0,a1,0,0} × {b0,b1}: Σ_{k=0..15} (q6−32)·q8 (scale group 2·sb).
            int zero = 0;
            int p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(p0), ""+r""(p1), ""+r""(p2), ""+r""(p3)
              : ""r""(a0), ""r""(a1), ""r""(zero), ""r""(zero), ""r""(b0), ""r""(b1));
            // mma2 = {0,0,a2,a3} × {b0,b1}: Σ_{k=16..31} (q6−32)·q8 (scale group 2·sb+1).
            int q0 = 0, q1 = 0, q2 = 0, q3 = 0;
            asm(
              ""mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ""
              ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};""
              : ""+r""(q0), ""+r""(q1), ""+r""(q2), ""+r""(q3)
              : ""r""(zero), ""r""(zero), ""r""(a2), ""r""(a3), ""r""(b0), ""r""(b1));

            acc0 += ((float)p0 * dwA0 + (float)q0 * dwA1) * dC0;
            acc1 += ((float)p1 * dwA0 + (float)q1 * dwA1) * dC1;
            acc2 += ((float)p2 * dwB0 + (float)q2 * dwB1) * dC0;
            acc3 += ((float)p3 * dwB0 + (float)q3 * dwB1) * dC1;
        }
        __syncthreads();
    }

    if (rowA < rows) {
        if (tokC0 < n_tok) output[(long)tokC0 * rows + rowA] = acc0;
        if (tokC1 < n_tok) output[(long)tokC1 * rows + rowA] = acc1;
    }
    if (rowB < rows) {
        if (tokC0 < n_tok) output[(long)tokC0 * rows + rowB] = acc2;
        if (tokC1 < n_tok) output[(long)tokC1 * rows + rowB] = acc3;
    }
}
#undef MMQ_BM
#undef MMQ_BN
";
}
