// SharpInference native AVX-512-VNNI CPU kernels.
//
// First vertical slice (perf/carnice-vnni-moe): a single-input Q3_K * Q8_KS
// dot product using vpdpbusd (AVX512_VNNI), to match llama.cpp's ggml-cpu-zen4
// speed on Carnice (Qwen3.6-35B-A3B), whose routed experts are ~75% Q3_K.
//
// The integer result is bit-identical to the C# scalar reference
// (SimdKernels.DotQ3K_Q8KS_Scalar); only the final per-sub-block float scale
// multiply/accumulate is FP, and it is accumulated in the same sub-block order
// as the scalar so the result is at worst argmax-stable (in practice it matches
// the scalar to FP-noise).
//
// Build: see scripts/build-vnni.ps1 (clang-cl /O2 -mavx512f -mavx512bw
// -mavx512vl -mavx512dq -mavx512vnni /LD).

#include <immintrin.h>
#include <stdint.h>
#include <cpuid.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// CPUID.(EAX=7,ECX=0):ECX bit 11 = AVX512_VNNI (EVEX). Probe WITHOUT executing
// any AVX-512 instruction, so this is safe to call on any x86-64 CPU.
EXPORT int sharpi_has_avx512vnni(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ecx >> 11) & 1; // AVX512VNNI
}

// vpdpbusd self-test: a=u8(1), b=s8(2), 64 lanes -> 16 int32 lanes each = 4*(1*2)=8
// -> horizontal sum = 16*8 = 128. Proves the EVEX VNNI path compiles & runs.
EXPORT int32_t sharpi_vnni_selftest(void) {
    __m512i a   = _mm512_set1_epi8(1);
    __m512i b   = _mm512_set1_epi8(2);
    __m512i acc = _mm512_setzero_si512();
    acc = _mm512_dpbusd_epi32(acc, a, b);
    return _mm512_reduce_add_epi32(acc);
}

// fp16 (IEEE half, little-endian 2 bytes) -> float.
static inline float half2_to_float(const unsigned char* p) {
    uint16_t h = (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
    // _cvtsh_ss requires F16C; clang lowers __fp16 / _Float16 without it, but
    // using the intrinsic keeps us aligned with the -mavx512* target set
    // (AVX-512 implies F16C on every shipping VNNI part). This matches the C#
    // BitConverter.UInt16BitsToHalf round-trip exactly.
    return _cvtsh_ss(h);
}

// Horizontal sum of the low 4 int32 lanes of a 512-bit vector (vpdpbusd over
// 16 input bytes lands in int32 lanes [0,4)).
static inline int32_t reduce_lo4_epi32(__m512i v) {
    __m128i lo = _mm512_castsi512_si128(v);
    __m128i sh = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
    sh = _mm_add_epi32(sh, _mm_shuffle_epi32(sh, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(sh);
}

// Horizontal sum of int32 lanes [4,8) of a 512-bit vector (the second 16 input
// bytes of a 32-byte vpdpbusd land in int32 lanes [4,8)).
static inline int32_t reduce_hi4_epi32(__m512i v) {
    __m128i hi = _mm256_extracti128_si256(_mm512_castsi512_si256(v), 1);
    __m128i sh = _mm_add_epi32(hi, _mm_shuffle_epi32(hi, _MM_SHUFFLE(2, 3, 0, 1)));
    sh = _mm_add_epi32(sh, _mm_shuffle_epi32(sh, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(sh);
}

// Q3_K weight row dot Q8_KS prequantized activation.
//   row     : Q3_K weights, 110 bytes per super-block of 256 elements.
//   scratch : Q8_KS activation (numBlocks super-blocks):
//               dArr     = (float*)scratch                              [numBlocks*8 floats]
//               qsArr    = (sbyte*)(scratch + numBlocks*32)             [numBlocks*256 sbytes]
//               bsumsArr = (short*)(scratch + numBlocks*32 + numBlocks*256) [numBlocks*16 shorts]
//   numBlocks: number of 256-element super-blocks (= cols / 256).
//
// Matches SimdKernels.DotQ3K_Q8KS_Scalar exactly in the integer domain.
EXPORT float sharpi_dot_q3k_q8ks(const unsigned char* row,
                                 const unsigned char* scratch,
                                 int numBlocks) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;

    const float* dArr = (const float*)scratch;
    const signed char* qsArr =
        (const signed char*)(scratch + (size_t)numBlocks * 32);
    const int16_t* bsumsArr =
        (const int16_t*)(scratch + (size_t)numBlocks * 32 + (size_t)numBlocks * 256);

    float acc = 0.0f;

    for (int b = 0; b < numBlocks; b++) {
        const unsigned char* x = row + (size_t)b * 110;
        float dAll = half2_to_float(x + 108);
        const float* dSub = dArr + (size_t)b * 8;

        // Decode the 16 6-bit Q3_K sub-block scales exactly as the scalar does.
        uint32_t aux[4];
        uint32_t a0, a1, tmp;
        a0 = (uint32_t)x[96] | ((uint32_t)x[97] << 8) |
             ((uint32_t)x[98] << 16) | ((uint32_t)x[99] << 24);
        a1 = (uint32_t)x[100] | ((uint32_t)x[101] << 8) |
             ((uint32_t)x[102] << 16) | ((uint32_t)x[103] << 24);
        tmp = (uint32_t)x[104] | ((uint32_t)x[105] << 8) |
              ((uint32_t)x[106] << 16) | ((uint32_t)x[107] << 24);
        aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);

        signed char scales[16];
        for (int i = 0; i < 4; i++) {
            scales[i * 4 + 0] = (signed char)(unsigned char)(aux[i] >> 0);
            scales[i * 4 + 1] = (signed char)(unsigned char)(aux[i] >> 8);
            scales[i * 4 + 2] = (signed char)(unsigned char)(aux[i] >> 16);
            scales[i * 4 + 3] = (signed char)(unsigned char)(aux[i] >> 24);
        }

        const unsigned char* qs = x + 32; // low 2 bits, 64 bytes
        const unsigned char* hm = x;      // high bit mask, 32 bytes
        const signed char* q8 = qsArr + (size_t)b * 256;
        const int16_t* bsums = bsumsArr + (size_t)b * 16;

        // Load the 32-byte high-bit mask once per super-block.
        __m256i hm_v = _mm256_loadu_si256((const __m256i*)hm);

        int isIdx = 0;
        int qOut = 0;
        unsigned char m = 1;
        for (int half = 0; half < 2; half++) {
            // 32 low-2-bit bytes for this half (sub-blocks share these 32 bytes
            // across the 4 j values via the shift; here we load the 32 bytes and
            // shift per j). The high-bit mask m advances per sub-block (1,2,4,...
            // across all 8), so hmHigh is recomputed inside the j loop.
            __m256i qs_v = _mm256_loadu_si256((const __m256i*)(qs + half * 32));

            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int sub = half * 4 + j;

                // hmBits[l] = (hm[l] & m) != 0 ? 4 : 0, via compare-not-zero.
                __m256i m_v = _mm256_set1_epi8((char)m);
                __m256i hmMasked = _mm256_and_si256(hm_v, m_v);
                __m256i hmNZ = _mm256_cmpeq_epi8(hmMasked, m_v); // 0xFF where set
                __m256i hmHigh = _mm256_and_si256(hmNZ, _mm256_set1_epi8(4));

                // qu[l] = ((qs[l] >> shift) & 3) + (high ? 4 : 0), l in [0,32).
                // The 16-bit logical shift is safe: we mask &3 afterward and
                // shift <= 6, so the kept low 2 bits of each byte never pull in
                // bits from the adjacent byte (the standard ggml AVX2 trick).
                __m256i lo = _mm256_srli_epi16(qs_v, shift);
                lo = _mm256_and_si256(lo, _mm256_set1_epi8(3));
                __m256i qu = _mm256_add_epi8(lo, hmHigh); // u8 weights in [0,7]

                // Load the 32 s8 activations for this sub-block contiguously.
                __m256i a8 = _mm256_loadu_si256((const __m256i*)(q8 + qOut));

                // vpdpbusd: u8 weights * s8 activations, 4 bytes -> 1 int32 lane.
                // Bytes [0,16) accumulate into int32 lanes [0,4) (= sub0), bytes
                // [16,32) into lanes [4,8) (= sub1). Zero-extend each ymm into the
                // low 256 bits of a zmm (upper lanes are zero, so they add 0).
                __m512i wu = _mm512_zextsi256_si512(qu);
                __m512i sa = _mm512_zextsi256_si512(a8);
                __m512i prod = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wu, sa);

                int sub0 = reduce_lo4_epi32(prod); // sum qu*q8 over [0,16)
                int sub1 = reduce_hi4_epi32(prod); // sum qu*q8 over [16,32)

                int sc0 = (int)scales[isIdx++] - 32;
                int sc1 = (int)scales[isIdx++] - 32;

                int subInt = sc0 * sub0 + sc1 * sub1
                           - 4 * (sc0 * (int)bsums[isIdx - 2]
                                + sc1 * (int)bsums[isIdx - 1]);
                acc += (dAll * dSub[sub]) * (float)subInt;

                qOut += 32;
                shift += 2;
                m <<= 1;
            }
        }
    }

    return acc;
}

// ---------------------------------------------------------------------------
// Batched Q3_K * Q8_KS: decode the weight row ONCE, dot it against 2 / 4
// Q8_KS activations. The expensive part — the 3-bit unpack (qu) and the 6-bit
// sub-block scale decode — is hoisted out and shared across all inputs, while
// each input keeps a SEPARATE float accumulator. Each input's accumulation is
// byte-for-byte identical (in the integer domain) to a single
// sharpi_dot_q3k_q8ks call, so the result is bit-identical to N single dots.
// This mirrors the C# DotQ3K_Q8KS_{2,4}In_Avx2 reference, which likewise decode
// the weight once and keep N float accumulators. Used by the batched routed-MoE
// path (phaseA gate+up, phaseC down) to amortize the unpack across token
// pairs/quads routing to the same expert.
// ---------------------------------------------------------------------------

// One input's contribution for the current sub-block, identical to the
// single-input kernel's inner body: integer sub0/sub1 via vpdpbusd, the
// sc*sub - 4*sc*bsums correction, and the per-sub-block float FMA.
static inline void q3k_q8ks_accum_one(
    __m256i qu,                  // shared unpacked weights (u8 in [0,7])
    const signed char* q8,       // this input's s8 activations base
    const int16_t* bsums,        // this input's per-sub-block bsums base
    int qOut, int isIdx,         // byte offset into q8 / scale index for this sub
    int sc0, int sc1,            // shared sub-block scales (already -32)
    float scaleSub,              // dAll * dSub[sub] for THIS input
    float* acc)
{
    __m256i a8 = _mm256_loadu_si256((const __m256i*)(q8 + qOut));
    __m512i wu = _mm512_zextsi256_si512(qu);
    __m512i sa = _mm512_zextsi256_si512(a8);
    __m512i prod = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wu, sa);

    int sub0 = reduce_lo4_epi32(prod);
    int sub1 = reduce_hi4_epi32(prod);

    int subInt = sc0 * sub0 + sc1 * sub1
               - 4 * (sc0 * (int)bsums[isIdx]
                    + sc1 * (int)bsums[isIdx + 1]);
    *acc += scaleSub * (float)subInt;
}

// Decodes the shared per-super-block scales (16 sub-block scales, NOT yet
// offset by -32) and returns dAll, mirroring the single-input kernel's decode.
static inline float q3k_decode_scales(const unsigned char* x, signed char scales[16]) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    uint32_t a0, a1, tmp;
    a0 = (uint32_t)x[96] | ((uint32_t)x[97] << 8) |
         ((uint32_t)x[98] << 16) | ((uint32_t)x[99] << 24);
    a1 = (uint32_t)x[100] | ((uint32_t)x[101] << 8) |
         ((uint32_t)x[102] << 16) | ((uint32_t)x[103] << 24);
    tmp = (uint32_t)x[104] | ((uint32_t)x[105] << 8) |
          ((uint32_t)x[106] << 16) | ((uint32_t)x[107] << 24);
    aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
    for (int i = 0; i < 4; i++) {
        scales[i * 4 + 0] = (signed char)(unsigned char)(aux[i] >> 0);
        scales[i * 4 + 1] = (signed char)(unsigned char)(aux[i] >> 8);
        scales[i * 4 + 2] = (signed char)(unsigned char)(aux[i] >> 16);
        scales[i * 4 + 3] = (signed char)(unsigned char)(aux[i] >> 24);
    }
    return half2_to_float(x + 108);
}

// Reconstructs the 32 shared unpacked weights (u8 in [0,7]) for one sub-block,
// identical to the single-input kernel's qu computation.
static inline __m256i q3k_unpack_qu(__m256i qs_v, __m256i hm_v, int shift, unsigned char m) {
    __m256i m_v = _mm256_set1_epi8((char)m);
    __m256i hmMasked = _mm256_and_si256(hm_v, m_v);
    __m256i hmNZ = _mm256_cmpeq_epi8(hmMasked, m_v);
    __m256i hmHigh = _mm256_and_si256(hmNZ, _mm256_set1_epi8(4));
    __m256i lo = _mm256_srli_epi16(qs_v, shift);
    lo = _mm256_and_si256(lo, _mm256_set1_epi8(3));
    return _mm256_add_epi8(lo, hmHigh);
}

EXPORT void sharpi_dot_q3k_q8ks_2in(const unsigned char* row,
                                    const unsigned char* s0, const unsigned char* s1,
                                    int numBlocks,
                                    float* out0, float* out1) {
    const float* dArr0 = (const float*)s0;
    const signed char* qsArr0 = (const signed char*)(s0 + (size_t)numBlocks * 32);
    const int16_t* bsumsArr0 =
        (const int16_t*)(s0 + (size_t)numBlocks * 32 + (size_t)numBlocks * 256);
    const float* dArr1 = (const float*)s1;
    const signed char* qsArr1 = (const signed char*)(s1 + (size_t)numBlocks * 32);
    const int16_t* bsumsArr1 =
        (const int16_t*)(s1 + (size_t)numBlocks * 32 + (size_t)numBlocks * 256);

    float acc0 = 0.0f, acc1 = 0.0f;

    for (int b = 0; b < numBlocks; b++) {
        const unsigned char* x = row + (size_t)b * 110;
        signed char scales[16];
        float dAll = q3k_decode_scales(x, scales);
        const float* dSub0 = dArr0 + (size_t)b * 8;
        const float* dSub1 = dArr1 + (size_t)b * 8;

        const unsigned char* qs = x + 32;
        __m256i hm_v = _mm256_loadu_si256((const __m256i*)x);
        const signed char* q80 = qsArr0 + (size_t)b * 256;
        const signed char* q81 = qsArr1 + (size_t)b * 256;
        const int16_t* bsums0 = bsumsArr0 + (size_t)b * 16;
        const int16_t* bsums1 = bsumsArr1 + (size_t)b * 16;

        int isIdx = 0;
        int qOut = 0;
        unsigned char m = 1;
        for (int half = 0; half < 2; half++) {
            __m256i qs_v = _mm256_loadu_si256((const __m256i*)(qs + half * 32));
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int sub = half * 4 + j;
                __m256i qu = q3k_unpack_qu(qs_v, hm_v, shift, m);

                int sc0 = (int)scales[isIdx] - 32;
                int sc1 = (int)scales[isIdx + 1] - 32;

                q3k_q8ks_accum_one(qu, q80, bsums0, qOut, isIdx, sc0, sc1,
                                   dAll * dSub0[sub], &acc0);
                q3k_q8ks_accum_one(qu, q81, bsums1, qOut, isIdx, sc0, sc1,
                                   dAll * dSub1[sub], &acc1);

                isIdx += 2;
                qOut += 32;
                shift += 2;
                m <<= 1;
            }
        }
    }

    *out0 = acc0;
    *out1 = acc1;
}

EXPORT void sharpi_dot_q3k_q8ks_4in(const unsigned char* row,
                                    const unsigned char* s0, const unsigned char* s1,
                                    const unsigned char* s2, const unsigned char* s3,
                                    int numBlocks,
                                    float* out0, float* out1, float* out2, float* out3) {
    const float* dArr0 = (const float*)s0;
    const signed char* qsArr0 = (const signed char*)(s0 + (size_t)numBlocks * 32);
    const int16_t* bsumsArr0 =
        (const int16_t*)(s0 + (size_t)numBlocks * 32 + (size_t)numBlocks * 256);
    const float* dArr1 = (const float*)s1;
    const signed char* qsArr1 = (const signed char*)(s1 + (size_t)numBlocks * 32);
    const int16_t* bsumsArr1 =
        (const int16_t*)(s1 + (size_t)numBlocks * 32 + (size_t)numBlocks * 256);
    const float* dArr2 = (const float*)s2;
    const signed char* qsArr2 = (const signed char*)(s2 + (size_t)numBlocks * 32);
    const int16_t* bsumsArr2 =
        (const int16_t*)(s2 + (size_t)numBlocks * 32 + (size_t)numBlocks * 256);
    const float* dArr3 = (const float*)s3;
    const signed char* qsArr3 = (const signed char*)(s3 + (size_t)numBlocks * 32);
    const int16_t* bsumsArr3 =
        (const int16_t*)(s3 + (size_t)numBlocks * 32 + (size_t)numBlocks * 256);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int b = 0; b < numBlocks; b++) {
        const unsigned char* x = row + (size_t)b * 110;
        signed char scales[16];
        float dAll = q3k_decode_scales(x, scales);
        const float* dSub0 = dArr0 + (size_t)b * 8;
        const float* dSub1 = dArr1 + (size_t)b * 8;
        const float* dSub2 = dArr2 + (size_t)b * 8;
        const float* dSub3 = dArr3 + (size_t)b * 8;

        const unsigned char* qs = x + 32;
        __m256i hm_v = _mm256_loadu_si256((const __m256i*)x);
        const signed char* q80 = qsArr0 + (size_t)b * 256;
        const signed char* q81 = qsArr1 + (size_t)b * 256;
        const signed char* q82 = qsArr2 + (size_t)b * 256;
        const signed char* q83 = qsArr3 + (size_t)b * 256;
        const int16_t* bsums0 = bsumsArr0 + (size_t)b * 16;
        const int16_t* bsums1 = bsumsArr1 + (size_t)b * 16;
        const int16_t* bsums2 = bsumsArr2 + (size_t)b * 16;
        const int16_t* bsums3 = bsumsArr3 + (size_t)b * 16;

        int isIdx = 0;
        int qOut = 0;
        unsigned char m = 1;
        for (int half = 0; half < 2; half++) {
            __m256i qs_v = _mm256_loadu_si256((const __m256i*)(qs + half * 32));
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int sub = half * 4 + j;
                __m256i qu = q3k_unpack_qu(qs_v, hm_v, shift, m);

                int sc0 = (int)scales[isIdx] - 32;
                int sc1 = (int)scales[isIdx + 1] - 32;

                q3k_q8ks_accum_one(qu, q80, bsums0, qOut, isIdx, sc0, sc1,
                                   dAll * dSub0[sub], &acc0);
                q3k_q8ks_accum_one(qu, q81, bsums1, qOut, isIdx, sc0, sc1,
                                   dAll * dSub1[sub], &acc1);
                q3k_q8ks_accum_one(qu, q82, bsums2, qOut, isIdx, sc0, sc1,
                                   dAll * dSub2[sub], &acc2);
                q3k_q8ks_accum_one(qu, q83, bsums3, qOut, isIdx, sc0, sc1,
                                   dAll * dSub3[sub], &acc3);

                isIdx += 2;
                qOut += 32;
                shift += 2;
                m <<= 1;
            }
        }
    }

    *out0 = acc0;
    *out1 = acc1;
    *out2 = acc2;
    *out3 = acc3;
}
