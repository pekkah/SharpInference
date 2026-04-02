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

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { float k_cache[]; };
        layout(binding = 2) readonly buffer VCache { float v_cache[]; };
        layout(binding = 3) writeonly buffer Out   { float out_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint max_seq_len;
        };

        shared float sdata[256];

        void main() {
            uint h = gl_WorkGroupID.x;      // query head index
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));

            // Phase 1: compute attention scores Q·K^T / sqrt(d)
            // Each thread computes scores for a subset of positions
            // Store scores in shared memory (limited to 256 positions at a time)
            // For simplicity, process seqLen <= 256 in one pass

            // First, compute all scores (tid maps to position t)
            float score = -1.0/0.0; // -inf for positions beyond seqLen
            if (tid < seq_len) {
                float dot = 0.0;
                uint q_off = h * head_dim;
                uint k_off = tid * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache[k_off + d];
                score = dot * scale;
            }
            sdata[tid] = score;
            barrier();

            // Phase 2: softmax over scores[0..seqLen)
            // Find max
            float local_max = sdata[tid];
            barrier();
            // Parallel reduction for max
            for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s && tid + s < 256)
                    sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            // Exp and sum
            float exp_val = 0.0;
            if (tid < seq_len) {
                exp_val = exp(local_max * scale / scale - max_val); // using original score
                // Recompute: we lost the original score during max reduction
            }

            // Actually, let's recompute the score since sdata was overwritten
            if (tid < seq_len) {
                float dot = 0.0;
                uint q_off = h * head_dim;
                uint k_off = tid * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache[k_off + d];
                exp_val = exp(dot * scale - max_val);
            }
            sdata[tid] = exp_val;
            barrier();

            // Sum reduction
            for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float sum_val = sdata[0];
            barrier();

            // Normalize
            float weight = (tid < seq_len) ? exp_val / sum_val : 0.0;

            // Phase 3: weighted sum of values
            // Each thread holds weight for position tid
            // Need to compute output[d] = sum_t(weight_t * V[t,d]) for d=0..headDim
            // Use shared memory: for each d, accumulate weight * V[tid, d]
            uint v_off = tid * kv_dim + kv_head * head_dim;
            uint out_off = h * head_dim;

            // Process headDim elements sequentially (headDim typically 64-128)
            for (uint d = 0; d < head_dim; d++) {
                float val = (tid < seq_len) ? weight * v_cache[v_off + d] : 0.0;
                sdata[tid] = val;
                barrier();

                // Reduce
                for (uint s = 128; s > 0; s >>= 1) {
                    if (tid < s) sdata[tid] += sdata[tid + s];
                    barrier();
                }

                if (tid == 0) out_data[out_off + d] = sdata[0];
                barrier();
            }
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
        #extension GL_EXT_shader_16bit_storage : enable
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

        // Read a byte from the weights buffer (packed as uint32)
        uint readByte(uint byteOffset) {
            uint wordIdx = byteOffset >> 2;
            uint byteIdx = byteOffset & 3;
            return (weights_data[wordIdx] >> (byteIdx * 8)) & 0xFF;
        }

        // Read FP16 from two bytes, convert to float
        float readHalf(uint byteOffset) {
            uint lo = readByte(byteOffset);
            uint hi = readByte(byteOffset + 1);
            return unpackHalf2x16(lo | (hi << 8)).x;
        }

        void main() {
            uint row = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (row >= rows) return;

            uint bytes_per_row = (cols / 256) * 144;
            uint row_offset = row * bytes_per_row;
            uint num_blocks = cols / 256;

            float acc = 0.0;

            // Each thread processes a stride of elements across all blocks
            for (uint block = 0; block < num_blocks; block++) {
                uint boff = row_offset + block * 144;
                float d = readHalf(boff);
                float dmin = readHalf(boff + 2);

                uint elem_base = block * 256;

                // Process 4 chunks of 64 elements per block
                for (uint chunk = 0; chunk < 4; chunk++) {
                    uint scale_off = boff + 4;
                    uint qs_off = boff + 16 + chunk * 32;
                    uint j = chunk * 2;

                    // Decode 6-bit scales and mins
                    float sc1, m1, sc2, m2;
                    if (j < 4) {
                        sc1 = float(readByte(scale_off + j) & 63);
                        m1  = float(readByte(scale_off + j + 4) & 63);
                        sc2 = float(readByte(scale_off + j + 1) & 63);
                        m2  = float(readByte(scale_off + j + 5) & 63);
                    } else {
                        uint j4 = j - 4;
                        sc1 = float((readByte(scale_off + j + 4) & 0xF) | ((readByte(scale_off + j4) >> 6) << 4));
                        m1  = float((readByte(scale_off + j + 4) >> 4) | ((readByte(scale_off + j) >> 6) << 4));
                        uint j41 = j - 3;
                        sc2 = float((readByte(scale_off + j + 5) & 0xF) | ((readByte(scale_off + j41) >> 6) << 4));
                        m2  = float((readByte(scale_off + j + 5) >> 4) | ((readByte(scale_off + j + 1) >> 6) << 4));
                    }

                    float d1 = d * sc1;
                    float dm1 = dmin * m1;
                    float d2 = d * sc2;
                    float dm2 = dmin * m2;

                    uint co = elem_base + chunk * 64;

                    // Lower nibbles: elements [co .. co+31]
                    for (uint l = tid & 31; l < 32; l += 32) {
                        if (tid < 32 || (tid >= 32 && l + (tid/32)*32 < 32)) {
                            // Simplified: each thread handles elements based on tid
                        }
                    }

                    // Process elements assigned to this thread
                    for (uint l = tid; l < 32; l += 256) {
                        uint qbyte = readByte(qs_off + l);
                        float lo_val = d1 * float(qbyte & 0xF) - dm1;
                        float hi_val = d2 * float(qbyte >> 4) - dm2;
                        acc += lo_val * input_data[co + l];
                        acc += hi_val * input_data[co + 32 + l];
                    }
                }
            }

            // Reduce across workgroup
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
