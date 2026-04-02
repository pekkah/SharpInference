namespace SharpInference.Vulkan;

/// <summary>
/// GLSL compute shader source code for all inference operations.
/// Compiled to SPIR-V at runtime via ShaderCompiler.
/// </summary>
internal static class Shaders
{
    /// <summary>
    /// RMS Normalization: output[i] = input[i] / rms * weight[i]
    /// where rms = sqrt(mean(input^2) + eps).
    ///
    /// Uses workgroup shared memory for parallel reduction of sum-of-squares.
    /// Push constants: { uint n, float eps }.
    /// Bindings: 0=input, 1=weight, 2=output.
    /// Dispatch: 1 workgroup of 256 threads.
    /// </summary>
    internal const string RmsNorm = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Input  { float input_data[]; };
        layout(binding = 1) readonly buffer Weight { float weight_data[]; };
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint n;
            float eps;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint gid = gl_GlobalInvocationID.x;

            // Phase 1: each thread accumulates sum of squares for its stride
            float sum = 0.0;
            for (uint i = tid; i < n; i += 256) {
                float v = input_data[i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            // Phase 2: parallel reduction in shared memory
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            // Phase 3: compute scale factor
            float scale = inversesqrt(sdata[0] / float(n) + eps);

            // Phase 4: apply normalization and weight
            for (uint i = tid; i < n; i += 256) {
                output_data[i] = input_data[i] * scale * weight_data[i];
            }
        }
        """;

    /// <summary>
    /// Fused SiLU(gate) * up: gate[i] = gate[i] * sigmoid(gate[i]) * up[i]
    /// Push constants: { uint n }.
    /// Bindings: 0=gate (in/out), 1=up (in).
    /// </summary>
    internal const string SiLuMul = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Gate { float gate_data[]; };
        layout(binding = 1) readonly buffer Up { float up_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            float g = gate_data[i];
            gate_data[i] = g / (1.0 + exp(-g)) * up_data[i];
        }
        """;

    /// <summary>
    /// Vector add in-place: dst[i] += src[i]
    /// Push constants: { uint n }.
    /// Bindings: 0=dst (in/out), 1=src (in).
    /// </summary>
    internal const string AddInPlace = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Dst { float dst_data[]; };
        layout(binding = 1) readonly buffer Src { float src_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            dst_data[i] += src_data[i];
        }
        """;

    /// <summary>
    /// Element-wise multiply: output[i] = a[i] * b[i]
    /// Push constants: { uint n }.
    /// Bindings: 0=a, 1=b, 2=output.
    /// </summary>
    internal const string ElementwiseMul = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer BufA { float a_data[]; };
        layout(binding = 1) readonly buffer BufB { float b_data[]; };
        layout(binding = 2) writeonly buffer BufC { float c_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            c_data[i] = a_data[i] * b_data[i];
        }
        """;

    /// <summary>
    /// RoPE: interleaved pair rotation.
    /// Push constants: { uint num_heads, uint head_dim, int position, float theta }.
    /// Bindings: 0=x (in/out).
    /// Each thread handles one pair (2 elements).
    /// </summary>
    internal const string RoPE = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            int position;
            float theta;
        };

        void main() {
            uint pair_idx = gl_GlobalInvocationID.x;
            uint half_dim = head_dim / 2;
            uint total_pairs = num_heads * half_dim;
            if (pair_idx >= total_pairs) return;

            uint h = pair_idx / half_dim;
            uint i = pair_idx % half_dim;

            float freq = 1.0 / pow(theta, 2.0 * float(i) / float(head_dim));
            float angle = float(position) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint base_idx = h * head_dim + 2 * i;
            float x0 = x_data[base_idx];
            float x1 = x_data[base_idx + 1];
            x_data[base_idx]     = x0 * cos_a - x1 * sin_a;
            x_data[base_idx + 1] = x0 * sin_a + x1 * cos_a;
        }
        """;

    /// <summary>
    /// Softmax in-place (3-pass: max, exp+sum, normalize).
    /// Uses workgroup shared memory for reductions.
    /// Push constants: { uint n }.
    /// Bindings: 0=x (in/out).
    /// Dispatch: 1 workgroup of 256 threads.
    /// </summary>
    internal const string Softmax = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params { uint n; };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;

            // Pass 1: find max
            float local_max = -1.0/0.0; // -inf
            for (uint i = tid; i < n; i += 256)
                local_max = max(local_max, x_data[i]);
            sdata[tid] = local_max;
            barrier();

            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            // Pass 2: exp(x - max) and sum
            float local_sum = 0.0;
            for (uint i = tid; i < n; i += 256) {
                float e = exp(x_data[i] - max_val);
                x_data[i] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();

            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float sum_val = sdata[0];
            barrier();

            // Pass 3: normalize
            float inv_sum = 1.0 / sum_val;
            for (uint i = tid; i < n; i += 256)
                x_data[i] *= inv_sum;
        }
        """;

    /// <summary>
    /// Embedding lookup: copy one row from embedding table to output.
    /// Push constants: { uint token_id, uint emb_dim }.
    /// Bindings: 0=embedding_table[vocab_size*emb_dim], 1=output[emb_dim].
    /// </summary>
    internal const string EmbedLookup = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer EmbTable { float emb_table[]; };
        layout(binding = 1) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint token_id;
            uint emb_dim;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= emb_dim) return;
            output_data[i] = emb_table[token_id * emb_dim + i];
        }
        """;

    /// <summary>
    /// Copy K and V vectors into the KV cache at the given position.
    /// Push constants: { uint kv_dim, uint position, uint max_seq_len }.
    /// Bindings: 0=k_input[kv_dim], 1=v_input[kv_dim], 2=k_cache[max_seq_len*kv_dim], 3=v_cache[max_seq_len*kv_dim].
    /// </summary>
    internal const string KvAppend = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { float k_cache[]; };
        layout(binding = 3) buffer VCache { float v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;
            uint max_seq_len;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= kv_dim) return;
            uint offset = position * kv_dim + i;
            k_cache[offset] = k_input[i];
            v_cache[offset] = v_input[i];
        }
        """;

    /// <summary>
    /// Scaled dot-product attention with GQA support.
    /// One workgroup per query head. Each workgroup computes:
    ///   scores[t] = Q_h · K[t, kvHead] / sqrt(headDim) for t=0..seqLen
    ///   softmax(scores)
    ///   output[h] = sum(scores[t] * V[t, kvHead])
    ///
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim, uint seq_len, uint max_seq_len }.
    /// Bindings: 0=Q[num_heads*head_dim], 1=K_cache[max_seq_len*kv_dim], 2=V_cache[max_seq_len*kv_dim], 3=output[num_heads*head_dim].
    /// </summary>
    internal const string Attention = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable
        #extension GL_EXT_shader_atomic_float : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { float k_cache[]; };
        layout(binding = 2) readonly buffer VCache { float v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint max_seq_len;
        };

        shared float sdata[256];

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint q_off = h * head_dim;
            uint out_off = h * head_dim;

            // Zero output first (threads cooperate)
            for (uint d = tid; d < head_dim; d += 256)
                out_data[out_off + d] = 0.0;
            barrier();

            // Phase 1: each thread computes score for position tid
            float score = -1.0/0.0;
            if (tid < seq_len) {
                float dot = 0.0;
                uint k_off = tid * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache[k_off + d];
                score = dot * scale;
            }

            // Phase 2: softmax — max reduction
            sdata[tid] = score;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            // Exp + sum
            float exp_val = (tid < seq_len) ? exp(score - max_val) : 0.0;
            sdata[tid] = exp_val;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float weight = exp_val / sdata[0];
            barrier();

            // Phase 3: weighted value sum — each thread atomically adds weight*V[tid]
            // For short seqLen (decode), this is efficient: few threads, no contention
            if (tid < seq_len) {
                uint v_off = tid * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    atomicAdd(out_data[out_off + d], weight * v_cache[v_off + d]);
            }
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with Q6_K dequantization.
    /// Same pattern as Q4_K but different block layout.
    /// Q6_K block (210 bytes per 256 elements):
    ///   [0:128]   ql — lower 4 bits
    ///   [128:192] qh — upper 2 bits
    ///   [192:208] 16 int8 scales
    ///   [208:210] FP16 d (super-block scale)
    /// </summary>
    internal const string MatVecQ6K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        shared float sdata[256];

        uint readByte(uint byteOffset) {
            return (weights_data[byteOffset >> 2] >> ((byteOffset & 3) * 8)) & 0xFF;
        }

        float readHalf(uint byteOffset) {
            uint lo = readByte(byteOffset);
            uint hi = readByte(byteOffset + 1);
            return unpackHalf2x16(lo | (hi << 8)).x;
        }

        int readInt8(uint byteOffset) {
            int val = int(readByte(byteOffset));
            return val >= 128 ? val - 256 : val; // sign extend
        }

        void main() {
            uint row = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (row >= rows) return;

            uint bytes_per_row = (cols / 256) * 210;
            uint boff_base = row * bytes_per_row;

            float acc = 0.0;

            for (uint elem = tid; elem < cols; elem += 256) {
                uint block = elem >> 8;
                uint within = elem & 255;

                uint boff = boff_base + block * 210;
                float d = readHalf(boff + 208);

                // Determine which half-block (0-127 or 128-255)
                uint half_idx = within >> 7; // 0 or 1
                uint in_half = within & 127;

                // 4 groups of 32 within each 128-element half:
                //   group 0: elements 0-31   — ql lower, qh bits 0-1, scale[+0/+1]
                //   group 1: elements 32-63  — qlB lower, qh bits 2-3, scale[+2/+3]
                //   group 2: elements 64-95  — ql upper,  qh bits 4-5, scale[+4/+5]
                //   group 3: elements 96-127 — qlB upper, qh bits 6-7, scale[+6/+7]
                uint group = in_half >> 5; // 0-3
                uint l = in_half & 31;

                uint ql_base = boff + half_idx * 64;
                uint qh_base = boff + 128 + half_idx * 32;
                uint sc_base = boff + 192 + half_idx * 8;

                uint isc = l >> 4; // 0 for l<16, 1 for l>=16
                int scale_val = readInt8(sc_base + isc + group * 2);

                uint ql_byte, qh_byte;
                int quant;

                if (group < 2) {
                    // groups 0,1: lower nibble
                    uint ql_off = (group == 0) ? l : (32 + l);
                    ql_byte = readByte(ql_base + ql_off);
                    qh_byte = readByte(qh_base + l);
                    uint shift = group * 2;
                    quant = int(((ql_byte & 0xF) | (((qh_byte >> shift) & 3) << 4))) - 32;
                } else {
                    // groups 2,3: upper nibble
                    uint ql_off = (group == 2) ? l : (32 + l);
                    ql_byte = readByte(ql_base + ql_off);
                    qh_byte = readByte(qh_base + l);
                    uint shift = group * 2;
                    quant = int((((ql_byte >> 4) & 0xF) | (((qh_byte >> shift) & 3) << 4))) - 32;
                }

                acc += d * float(scale_val) * float(quant) * input_data[elem];
            }

            sdata[tid] = acc;
            barrier();

            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            if (tid == 0) output_data[row] = sdata[0];
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with F32 weights.
    /// Each workgroup computes one output row.
    /// Push constants: { uint rows, uint cols }.
    /// Bindings: 0=weights (float), 1=input (float), 2=output (float).
    /// </summary>
    internal const string MatVecF32 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { float weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        shared float sdata[256];

        void main() {
            uint row = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (row >= rows) return;

            float acc = 0.0;
            uint base_off = row * cols;
            for (uint i = tid; i < cols; i += 256)
                acc += weights_data[base_off + i] * input_data[i];

            sdata[tid] = acc;
            barrier();

            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            if (tid == 0) output_data[row] = sdata[0];
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with Q4_K dequantization.
    /// Each workgroup computes one output row.
    /// Push constants: { uint rows, uint cols }.
    /// Bindings: 0=quantized weights (uint8), 1=input vector (float), 2=output (float).
    ///
    /// Q4_K block layout (144 bytes per 256 elements):
    ///   [0:2]   FP16 d (super-block scale)
    ///   [2:4]   FP16 dmin (super-block minimum)
    ///   [4:16]  12 bytes packed 6-bit scales/mins
    ///   [16:144] 128 bytes 4-bit quantized values
    /// </summary>
    internal const string MatVecQ4K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        shared float sdata[256];

        uint readByte(uint byteOffset) {
            return (weights_data[byteOffset >> 2] >> ((byteOffset & 3) * 8)) & 0xFF;
        }

        float readHalf(uint byteOffset) {
            uint lo = readByte(byteOffset);
            uint hi = readByte(byteOffset + 1);
            return unpackHalf2x16(lo | (hi << 8)).x;
        }

        void getScaleMin(uint scale_off, uint j, out float sc, out float m) {
            if (j < 4) {
                sc = float(readByte(scale_off + j) & 63);
                m  = float(readByte(scale_off + j + 4) & 63);
            } else {
                sc = float((readByte(scale_off + j + 4) & 0xF) | ((readByte(scale_off + j - 4) >> 6) << 4));
                m  = float((readByte(scale_off + j + 4) >> 4) | ((readByte(scale_off + j) >> 6) << 4));
            }
        }

        void main() {
            uint row = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (row >= rows) return;

            uint bytes_per_row = (cols / 256) * 144;
            uint boff_base = row * bytes_per_row;

            float acc = 0.0;

            // ALL 256 threads process elements with stride
            // Each thread handles cols/256 elements total
            for (uint elem = tid; elem < cols; elem += 256) {
                uint block = elem >> 8;  // elem / 256
                uint within = elem & 255; // elem % 256
                uint chunk = within >> 6; // within / 64
                uint sub = within & 63;   // within % 64
                bool is_upper = sub >= 32;
                uint byte_pos = sub & 31;

                uint boff = boff_base + block * 144;
                float d = readHalf(boff);
                float dmin = readHalf(boff + 2);

                uint si = chunk * 2 + (is_upper ? 1u : 0u);
                float sc, mn;
                getScaleMin(boff + 4, si, sc, mn);

                uint qbyte = readByte(boff + 16 + chunk * 32 + byte_pos);
                uint nibble = is_upper ? (qbyte >> 4) : (qbyte & 0xF);

                acc += (d * sc * float(nibble) - dmin * mn) * input_data[elem];
            }

            // Parallel reduction
            sdata[tid] = acc;
            barrier();

            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            if (tid == 0) output_data[row] = sdata[0];
        }
        """;
}
