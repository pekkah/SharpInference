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
    /// Vector add in-place with scalar weight: dst[i] += scale * src[i]
    /// Push constants: { uint n, float scale }.
    /// Bindings: 0=dst (in/out), 1=src (in).
    /// </summary>
    internal const string AddScaledInPlace = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Dst { float dst_data[]; };
        layout(binding = 1) readonly buffer Src { float src_data[]; };

        layout(push_constant) uniform Params {
            uint n;
            float scale;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            dst_data[i] += scale * src_data[i];
        }
        """;

    /// <summary>
    /// In-place scalar multiply: data[i] *= scale for i in [0, n).
    /// Push constants: { uint n, float scale }.
    /// Bindings: 0=data (in/out).
    /// </summary>
    internal const string ScaleInPlace = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint n;
            float scale;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            data[i] *= scale;
        }
        """;

    /// <summary>
    /// Raw buffer copy: dst_data[dst_offset + i] = src_data[src_offset + i] for i in [0, count).
    /// Operates on uint32 words (4-byte aligned). All offsets are in uint32 units.
    /// Push constants: { uint count, uint src_offset, uint dst_offset }.
    /// Bindings: 0=src (readonly), 1=dst (writeonly).
    /// Dispatch: ceil(count / 256) workgroups of 256 threads.
    /// </summary>
    internal const string BufferCopy = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Src { uint src_data[]; };
        layout(binding = 1) writeonly buffer Dst { uint dst_data[]; };

        layout(push_constant) uniform Params {
            uint count;
            uint src_offset;
            uint dst_offset;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= count) return;
            dst_data[dst_offset + i] = src_data[src_offset + i];
        }
        """;

    /// <summary>
    /// Fill a buffer with zeros.
    /// Push constants: { uint n }.
    /// Bindings: 0=dst (in/out).
    /// </summary>
    internal const string Clear = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Dst { float dst_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            dst_data[i] = 0.0;
        }
        """;

    /// <summary>
    /// Per-head RMSNorm: applies RMSNorm independently to each head-sized chunk.
    /// data[h*head_dim + i] = data[h*head_dim + i] / rms_h * weight[i]
    /// where rms_h = sqrt(mean(data[h*head_dim .. (h+1)*head_dim]^2) + eps).
    ///
    /// One workgroup per head. Weight is [head_dim] shared across all heads.
    /// Push constants: { uint head_dim, uint num_heads, float eps }.
    /// Bindings: 0=data (in/out), 1=weight (in).
    /// Dispatch: num_heads workgroups of 256 threads.
    /// </summary>
    internal const string HeadNorm = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data   { float data_buf[]; };
        layout(binding = 1) readonly buffer Weight { float weight_data[]; };

        layout(push_constant) uniform Params {
            uint head_dim;
            uint num_heads;
            float eps;
            // 0 = weight is shared across heads (Qwen3 style, len = head_dim).
            // head_dim = per-channel weight (OLMoE style, len = num_heads*head_dim).
            uint weight_stride;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint head = gl_WorkGroupID.x;
            if (head >= num_heads) return;

            uint base_off = head * head_dim;
            uint w_off    = head * weight_stride;

            // Phase 1: accumulate sum of squares
            float sum = 0.0;
            for (uint i = tid; i < head_dim; i += 256) {
                float v = data_buf[base_off + i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            // Phase 2: parallel reduction
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            // Phase 3: normalize in-place with weight
            float scale = inversesqrt(sdata[0] / float(head_dim) + eps);
            for (uint i = tid; i < head_dim; i += 256) {
                data_buf[base_off + i] = data_buf[base_off + i] * scale * weight_data[w_off + i];
            }
        }
        """;

    /// <summary>
    /// Per-head RMS normalization without learned weights (L2 normalize).
    /// Used for Llama4TextL2Norm in QK-norm.
    /// Push constants: { uint head_dim, uint num_heads, float eps }.
    /// </summary>
    internal const string HeadNormPure = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data_buf[]; };

        layout(push_constant) uniform Params {
            uint head_dim;
            uint num_heads;
            float eps;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint head = gl_WorkGroupID.x;
            if (head >= num_heads) return;

            uint base_off = head * head_dim;

            float sum = 0.0;
            for (uint i = tid; i < head_dim; i += 256) {
                float v = data_buf[base_off + i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            float scale = inversesqrt(sdata[0] / float(head_dim) + eps);
            for (uint i = tid; i < head_dim; i += 256) {
                data_buf[base_off + i] = data_buf[base_off + i] * scale;
            }
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
    /// RoPE: interleaved pair rotation. Used by LLaMA, Mistral, SmolLM, etc.
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
    /// RoPE: NEOX/half rotation (pairs offset by head_dim/2). Used by Qwen, Phi, Gemma, Falcon, etc.
    /// </summary>
    internal const string RoPENeox = """
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

            uint head_base = h * head_dim;
            uint a_idx = head_base + i;
            uint b_idx = head_base + i + half_dim;
            float x0 = x_data[a_idx];
            float x1 = x_data[b_idx];
            x_data[a_idx] = x0 * cos_a - x1 * sin_a;
            x_data[b_idx] = x0 * sin_a + x1 * cos_a;
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
            // No extra barrier needed — last reduction iteration's barrier
            // guarantees sdata[0] is visible to all threads

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

            // Pass 3: normalize
            float inv_sum = 1.0 / sum_val;
            for (uint i = tid; i < n; i += 256)
                x_data[i] *= inv_sum;
        }
        """;

    /// <summary>
    /// Element-wise sigmoid in-place: x[i] = 1 / (1 + exp(-x[i])).
    /// Used for Llama-4 MoE router gating.
    /// Push constants: { uint n }.
    /// Bindings: 0=x (in/out).
    /// </summary>
    internal const string Sigmoid = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            x_data[i] = 1.0 / (1.0 + exp(-x_data[i]));
        }
        """;

    /// <summary>
    /// Embedding lookup: copy one row from F32 embedding table to output.
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
    /// Embedding lookup from Q4_K quantized table: dequantize one row to F32 output.
    /// 256 threads cooperate: each processes one block (256 elements) sequentially,
    /// with each thread handling one element per block.
    ///
    /// Push constants: { uint token_id, uint emb_dim }.
    /// Bindings: 0=quantized_table (uint8 via uint32[]), 1=output[emb_dim].
    /// Dispatch: 1 workgroup.
    /// </summary>
    internal const string EmbedLookupQ4K = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer EmbTable { uint emb_data[]; };
        layout(binding = 1) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint token_id;
            uint emb_dim;
        };

        shared uint blk[36]; // 144 bytes = one Q4_K block in shared memory

        uint sReadByte(uint byteOffset) {
            return (blk[byteOffset >> 2] >> ((byteOffset & 3) * 8)) & 0xFF;
        }

        float sReadHalf(uint byteOffset) {
            uint lo = sReadByte(byteOffset);
            uint hi = sReadByte(byteOffset + 1);
            return unpackHalf2x16(lo | (hi << 8)).x;
        }

        void sGetScaleMin(uint j, out float sc, out float m) {
            if (j < 4) {
                sc = float(sReadByte(4 + j) & 63);
                m  = float(sReadByte(4 + j + 4) & 63);
            } else {
                sc = float((sReadByte(4 + j + 4) & 0xF) | ((sReadByte(4 + j - 4) >> 6) << 4));
                m  = float((sReadByte(4 + j + 4) >> 4) | ((sReadByte(4 + j) >> 6) << 4));
            }
        }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint num_blocks = emb_dim >> 8; // emb_dim / 256

            // Byte offset to the start of this token's row
            uint bytes_per_row = num_blocks * 144;
            uint row_base = token_id * (bytes_per_row >> 2); // in uint32 units

            for (uint block = 0; block < num_blocks; block++) {
                // Cooperatively load 36 uint32s (144 bytes) into shared memory
                uint blk_word_base = row_base + (block * 144 >> 2);
                if (tid < 36)
                    blk[tid] = emb_data[blk_word_base + tid];
                barrier();

                // Each thread dequantizes its element
                uint chunk = tid >> 6;
                uint sub = tid & 63;
                bool is_upper = sub >= 32;
                uint byte_pos = sub & 31;

                float d = sReadHalf(0);
                float dmin = sReadHalf(2);

                uint si = chunk * 2 + (is_upper ? 1u : 0u);
                float sc, mn;
                sGetScaleMin(si, sc, mn);

                uint qbyte = sReadByte(16 + chunk * 32 + byte_pos);
                uint nibble = is_upper ? (qbyte >> 4) : (qbyte & 0xF);

                output_data[block * 256 + tid] = d * sc * float(nibble) - dmin * mn;

                barrier();
            }
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
    /// For seq_len &lt;= 4096: stores all scores in shared memory — single Q·K pass,
    /// then softmax, then value accumulation. Matches the TqAttention approach.
    /// For seq_len &gt; 4096: triple-pass with Q·K recomputation (correctness over performance).
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
        layout(binding = 3) buffer Out             { float out_data[]; };
        layout(binding = 4) buffer ScoresScratch   { float scores_scratch[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint max_seq_len;
        };

        // Score-storage strategy mirrors the CUDA `llm_attention` kernel:
        //   • seq_len ≤ MAX_SHARED_SCORES (4096): fast path uses shared memory.
        //   • seq_len > 4096: spills to scores_scratch[h*max_seq_len .. +seq_len).
        // The fast path does not touch the scratch buffer, but Vulkan descriptors
        // require it to be bound — callers pass a 1-float placeholder when the
        // whole context is guaranteed to fit in shared memory.
        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint q_off = h * head_dim;
            uint out_off = h * head_dim;

            bool use_shared = (seq_len <= MAX_SHARED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1: per-position Q·K scores ───
            for (uint t = tid; t < seq_len; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache[k_off + d];
                float score = dot * scale;
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }
            // Pad shared tail so the max scan ignores stale slots. The scratch
            // scans iterate only [0, seq_len), so no padding needed.
            if (use_shared) {
                for (uint t = seq_len + tid; t < MAX_SHARED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }
            barrier();

            // ─── Phase 2: in-place softmax over [0, seq_len) ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < seq_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                local_max = max(local_max, s);
            }
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            float local_sum = 0.0;
            for (uint t = tid; t < seq_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                float e = exp(s - max_val);
                if (use_shared) scores[t] = e;
                else            scores_scratch[scratch_base + t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            for (uint t = tid; t < seq_len; t += 256) {
                if (use_shared) scores[t] *= inv_sum;
                else            scores_scratch[scratch_base + t] *= inv_sum;
            }
            barrier();

            // ─── Phase 3: weighted V sum. K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                float sum = 0.0;
                for (uint t = 0; t < seq_len; t++) {
                    float weight = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum += weight * v_cache[v_off + d];
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// SnapKV (issue #59) — per-head attention scoring across a layer's K cache.
    /// Mirrors the CUDA `llm_snapkv_score` kernel: one workgroup per query head,
    /// 256 threads. Phase 1 computes causal-masked dot(q_head, k_cache[t, kvHead, :]) * scale
    /// for every t in [0, prompt_len); Phase 2 runs an in-place softmax over the
    /// valid prefix; Phase 3 atomicAdds the post-softmax weights into a global
    /// per-position accumulator.
    ///
    /// Vulkan core has no native float atomicAdd, so binding 2 is bound twice —
    /// once as f32 ScoreAccum for readers, once as u32 ScoreAccumAtomic for the
    /// compare-and-swap loop. The two views share the same VkBuffer (same bit
    /// pattern; only the binding type differs).
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, prompt_len, q_abs_pos, max_seq_len }.
    /// Bindings:
    ///   0 = Q (readonly)
    ///   1 = K cache (readonly)
    ///   2 = score_accum, f32 view (coherent, atomic CAS via the u32 alias on the same buffer)
    ///   3 = scores_scratch (writeonly, only used when prompt_len &gt; 4096)
    /// </summary>
    internal const string SnapKvScore = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q      { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { float k_cache[]; };
        // The CUDA reference does a direct float atomicAdd here; Vulkan core
        // exposes only integer atomics, so we keep the storage f32 (callers
        // download it as floats) but mutate it through a uint alias via
        // atomicCompSwap. Same buffer bound twice — the bit pattern of one
        // view IS the bit pattern of the other.
        layout(binding = 2) coherent buffer ScoreAccumAtomic { uint accum_uint[]; };
        // Spill buffer for the > 4096 path: written in Phase 1, re-read in
        // Phase 2 (max-reduce + softmax) and Phase 3 (atomicAdd), so no
        // writeonly qualifier here.
        layout(binding = 3) buffer ScoresScratch   { float scores_scratch[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint prompt_len;
            uint q_abs_pos;
            uint max_seq_len;
        };

        const uint MAX_STORED_SCORES = 4096u;
        shared float scores[MAX_STORED_SCORES];
        shared float sdata[256];

        void atomicAddFloat(uint idx, float value) {
            // Compare-and-swap loop on the uint reinterpretation of the f32 word.
            // The CUDA path uses native float atomicAdd; this matches its semantics
            // (last-writer-wins associative accumulate) on Vulkan core.
            uint oldBits = accum_uint[idx];
            while (true) {
                float oldVal = uintBitsToFloat(oldBits);
                float newVal = oldVal + value;
                uint newBits = floatBitsToUint(newVal);
                uint prev = atomicCompSwap(accum_uint[idx], oldBits, newBits);
                if (prev == oldBits) return;
                oldBits = prev;
            }
        }

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim  = num_kv_heads * head_dim;
            float scale  = inversesqrt(float(head_dim));
            uint q_off   = h * head_dim;

            bool use_shared = (prompt_len <= MAX_STORED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1: per-position causal-masked Q·K dot ───
            for (uint t = tid; t < prompt_len; t += 256) {
                float score;
                if (t > q_abs_pos) {
                    score = -1.0/0.0;
                } else {
                    float dot = 0.0;
                    uint k_off = t * kv_dim + kv_head * head_dim;
                    for (uint d = 0; d < head_dim; d++)
                        dot += q_data[q_off + d] * k_cache[k_off + d];
                    score = dot * scale;
                }
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }
            // Pad shared tail so max-reduce ignores stale slots. Scratch reads
            // iterate only [0, prompt_len), so no padding needed there.
            if (use_shared) {
                for (uint t = prompt_len + tid; t < MAX_STORED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }
            barrier();

            // ─── Phase 2a: max over [0, prompt_len) ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < prompt_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                local_max = max(local_max, s);
            }
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            // ─── Phase 2b: exp(s - max), sum, normalize ───
            float local_sum = 0.0;
            for (uint t = tid; t < prompt_len; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                float e = (s == -1.0/0.0) ? 0.0 : exp(s - max_val);
                if (use_shared) scores[t] = e;
                else            scores_scratch[scratch_base + t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            // ─── Phase 3: atomicAdd softmax weight into global accumulator ───
            for (uint t = tid; t < prompt_len; t += 256) {
                if (t > q_abs_pos) continue;
                float w = (use_shared ? scores[t] : scores_scratch[scratch_base + t]) * inv_sum;
                atomicAddFloat(t, w);
            }
        }
        """;

    /// <summary>
    /// SnapKV (issue #59) — gather kept positions of one KV ring (K or V) into a
    /// dense <c>[K * kv_dim]</c> prefix of <c>dst</c>. <c>src</c> and <c>dst</c> MUST be
    /// different buffers; the destination is later copied back over the ring's
    /// <c>[0, K * kv_dim)</c> region by the caller.
    ///
    /// Each thread copies one float from src[keep[blockIdx.y] * kv_dim + d] to
    /// dst[blockIdx.y * kv_dim + d]. Grid = (ceil(kv_dim/256), K, 1), block 256.
    ///
    /// Push constants: { uint K, kv_dim }.
    /// Bindings: 0=src (readonly), 1=dst (writeonly), 2=keep_positions (readonly int32).
    /// </summary>
    internal const string KvCompact = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Src  { float src_data[]; };
        layout(binding = 1) writeonly buffer Dst  { float dst_data[]; };
        layout(binding = 2) readonly  buffer Keep { int   keep_positions[]; };

        layout(push_constant) uniform Params {
            uint K;
            uint kv_dim;
        };

        void main() {
            uint i = gl_WorkGroupID.y;
            if (i >= K) return;
            uint d = gl_GlobalInvocationID.x;
            if (d >= kv_dim) return;

            uint src_pos = uint(keep_positions[i]);
            uint src_off = src_pos * kv_dim + d;
            uint dst_off = i       * kv_dim + d;
            dst_data[dst_off] = src_data[src_off];
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
        #extension GL_KHR_shader_subgroup_arithmetic : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        // Q6_K block layout (210 bytes per 256 elements):
        //   [0:128]   ql — lower 4-bit nibbles (two 64-byte halves)
        //   [128:192] qh — upper 2-bit pairs (two 32-byte halves)
        //   [192:208] 16 int8 scale values
        //   [208:210] FP16 super-block scale d
        // Thread layout: each lane handles 8 elements (lane, lane+32, ..., lane+224)
        // which all share l = lane within their respective groups — no shared memory needed.
        #define N_ROWS 8
        #define THREADS_PER_ROW 32

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        uint gByte(uint b) { return (weights_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }
        int  gInt8(uint b) { int v = int(gByte(b)); return v >= 128 ? v - 256 : v; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;
            uint boff_base = row * num_blocks * 210;

            float acc = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = boff_base + block * 210;

                float d = unpackHalf2x16(gByte(b0 + 208) | (gByte(b0 + 209) << 8)).x;

                // Precompute the 8 scale floats needed by this lane.
                // isc = lane>>4 selects lower (0) or upper (1) sub-scale row per group.
                uint isc = lane >> 4;
                float sc0 = d * float(gInt8(b0 + 192 + isc));
                float sc1 = d * float(gInt8(b0 + 194 + isc));
                float sc2 = d * float(gInt8(b0 + 196 + isc));
                float sc3 = d * float(gInt8(b0 + 198 + isc));
                float sc4 = d * float(gInt8(b0 + 200 + isc));
                float sc5 = d * float(gInt8(b0 + 202 + isc));
                float sc6 = d * float(gInt8(b0 + 204 + isc));
                float sc7 = d * float(gInt8(b0 + 206 + isc));

                // Load the 6 quantized bytes needed by this lane.
                // Byte layout: groups 0,1 share nibbles from the same byte; 2,3 use upper nibble.
                uint ql0 = gByte(b0 + lane);          // half=0, ql[lane]
                uint ql1 = gByte(b0 + 32 + lane);     // half=0, ql[32+lane]
                uint ql2 = gByte(b0 + 64 + lane);     // half=1, ql[64+lane]
                uint ql3 = gByte(b0 + 96 + lane);     // half=1, ql[96+lane]
                uint qh0 = gByte(b0 + 128 + lane);    // half=0, qh[lane]
                uint qh1 = gByte(b0 + 160 + lane);    // half=1, qh[32+lane]

                uint base_elem = block * 256;

                acc += sc0 * float(int((ql0 & 0xF)        | (((qh0 >> 0) & 3) << 4)) - 32) * input_data[base_elem +       lane];
                acc += sc1 * float(int((ql1 & 0xF)        | (((qh0 >> 2) & 3) << 4)) - 32) * input_data[base_elem +  32 + lane];
                acc += sc2 * float(int(((ql0 >> 4) & 0xF) | (((qh0 >> 4) & 3) << 4)) - 32) * input_data[base_elem +  64 + lane];
                acc += sc3 * float(int(((ql1 >> 4) & 0xF) | (((qh0 >> 6) & 3) << 4)) - 32) * input_data[base_elem +  96 + lane];
                acc += sc4 * float(int((ql2 & 0xF)        | (((qh1 >> 0) & 3) << 4)) - 32) * input_data[base_elem + 128 + lane];
                acc += sc5 * float(int((ql3 & 0xF)        | (((qh1 >> 2) & 3) << 4)) - 32) * input_data[base_elem + 160 + lane];
                acc += sc6 * float(int(((ql2 >> 4) & 0xF) | (((qh1 >> 4) & 3) << 4)) - 32) * input_data[base_elem + 192 + lane];
                acc += sc7 * float(int(((ql3 >> 4) & 0xF) | (((qh1 >> 6) & 3) << 4)) - 32) * input_data[base_elem + 224 + lane];
            }

            float result = subgroupAdd(acc);
            if (subgroupElect())
                output_data[row] = result;
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
        #extension GL_KHR_shader_subgroup_arithmetic : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        #define N_ROWS 8
        #define THREADS_PER_ROW 32

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { float weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            float acc = 0.0;
            uint base_off = row * cols;
            for (uint i = lane; i < cols; i += THREADS_PER_ROW)
                acc += weights_data[base_off + i] * input_data[i];

            float result = subgroupAdd(acc);
            if (subgroupElect())
                output_data[row] = result;
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
        #extension GL_KHR_shader_subgroup_arithmetic : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        // Register-based scale precomputation. subgroupAdd for reduction.
        #define N_ROWS 8
        #define THREADS_PER_ROW 32

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
        };

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;
            uint word_row_base = row * num_blocks * 36;

            float acc = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint word_base = word_row_base + block * 36;

                vec2 dm = unpackHalf2x16(weights_data[word_base]);
                float d = dm.x;
                float dmin = dm.y;

                // Preload scale/min into registers (3 global reads instead of ~32)
                uint sm0 = weights_data[word_base + 1];
                uint sm1 = weights_data[word_base + 2];
                uint sm2 = weights_data[word_base + 3];

                float dsc[8], dmn[8];
                dsc[0] = d * float((sm0) & 63);         dmn[0] = dmin * float((sm1) & 63);
                dsc[1] = d * float((sm0 >> 8) & 63);    dmn[1] = dmin * float((sm1 >> 8) & 63);
                dsc[2] = d * float((sm0 >> 16) & 63);   dmn[2] = dmin * float((sm1 >> 16) & 63);
                dsc[3] = d * float((sm0 >> 24) & 63);   dmn[3] = dmin * float((sm1 >> 24) & 63);
                dsc[4] = d * float((sm2 & 0xF) | (((sm0 >> 6) & 3) << 4));
                dmn[4] = dmin * float(((sm2 >> 4) & 0xF) | (((sm1 >> 6) & 3) << 4));
                dsc[5] = d * float(((sm2 >> 8) & 0xF) | (((sm0 >> 14) & 3) << 4));
                dmn[5] = dmin * float(((sm2 >> 12) & 0xF) | (((sm1 >> 14) & 3) << 4));
                dsc[6] = d * float(((sm2 >> 16) & 0xF) | (((sm0 >> 22) & 3) << 4));
                dmn[6] = dmin * float(((sm2 >> 20) & 0xF) | (((sm1 >> 22) & 3) << 4));
                dsc[7] = d * float(((sm2 >> 24) & 0xF) | (((sm0 >> 30) & 3) << 4));
                dmn[7] = dmin * float(((sm2 >> 28) & 0xF) | (((sm1 >> 30) & 3) << 4));

                // Each of 32 threads handles 8 elements: lane, lane+32, ..., lane+224
                [[unroll]] for (uint e = 0; e < 8; e++) {
                    uint elem_idx = lane + e * 32;
                    uint chunk = elem_idx >> 6;
                    uint sub = elem_idx & 63;
                    bool is_upper = sub >= 32;
                    uint byte_pos = sub & 31;

                    uint qs_off = word_base + 4 + (chunk * 8 + (byte_pos >> 2));
                    uint qbyte = (weights_data[qs_off] >> ((byte_pos & 3) * 8)) & 0xFF;
                    uint nibble = is_upper ? (qbyte >> 4) : (qbyte & 0xF);

                    uint si = chunk * 2 + (is_upper ? 1u : 0u);
                    acc += (dsc[si] * float(nibble) - dmn[si]) * input_data[block * 256 + elem_idx];
                }
            }

            float result = subgroupAdd(acc);
            if (subgroupElect())
                output_data[row] = result;
        }
        """;

    // ================================================================
    //  TurboQuant KV Cache Compression Shaders
    // ================================================================

    /// <summary>
    /// Rotate query vectors for TurboQuant: WHT + sign flip per KV head.
    /// One workgroup per query head. 128 threads per workgroup.
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim }.
    /// Bindings: 0=q_input[num_heads*head_dim], 1=rotated_q[num_heads*head_dim], 2=sign_patterns[num_kv_heads*head_dim].
    /// </summary>
    internal const string TqRotateQuery = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 128) in;

        layout(binding = 0) readonly buffer QIn      { float q_input[]; };
        layout(binding = 1) buffer QOut              { float rotated_q[]; };
        layout(binding = 2) readonly buffer Signs    { float sign_patterns[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
        };

        shared float sdata[128];

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads || tid >= head_dim) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint q_off = h * head_dim;
            uint sign_off = kv_head * head_dim;

            // Load query into shared memory
            sdata[tid] = q_input[q_off + tid];
            barrier();

            // In-place WHT butterfly
            [[unroll]] for (uint stride = 64; stride >= 1; stride >>= 1) {
                barrier();
                uint pair = (tid / stride) * (stride * 2) + (tid % stride);
                float a = sdata[pair];
                float b = sdata[pair + stride];
                sdata[pair] = a + b;
                sdata[pair + stride] = a - b;
            }
            barrier();

            // Normalize and apply sign flip
            float scale = 1.0 / sqrt(float(head_dim));
            rotated_q[q_off + tid] = sdata[tid] * scale * sign_patterns[sign_off + tid];
        }
        """;

    /// <summary>
    /// TurboQuant KV cache append: applies WHT + sign flip + quantization,
    /// then packs into 3-bit compressed format.
    /// Workgroup of 128 threads (one per dimension).
    /// Push constants: { uint kv_dim, uint head_dim, uint position, uint max_seq_len, uint num_kv_heads }.
    /// Bindings: 0=k_input[kv_dim], 1=v_input[kv_dim], 2=k_cache_tq[...], 3=v_cache_tq[...],
    ///           4=sign_patterns[num_kv_heads*head_dim], 5=codebook[8], 6=boundaries[7].
    /// Each workgroup handles one KV head.
    /// </summary>
    internal const string TqKvAppend = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 128) in;

        layout(binding = 0) readonly buffer KIn      { float k_input[]; };
        layout(binding = 1) readonly buffer VIn      { float v_input[]; };
        layout(binding = 2) buffer KCacheTQ          { uint k_cache_tq[]; };
        layout(binding = 3) buffer VCacheTQ          { uint v_cache_tq[]; };
        layout(binding = 4) readonly buffer Signs    { float sign_patterns[]; };
        layout(binding = 5) readonly buffer Codebook { float codebook[8]; };
        layout(binding = 6) readonly buffer Bounds   { float boundaries[7]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint head_dim;
            uint position;
            uint max_seq_len;
            uint num_kv_heads;
            uint block_bytes;    // bytes per compressed block (52 for 3-bit d=128)
        };

        shared float sdata[128];  // shared memory for WHT butterfly

        // Walsh-Hadamard transform (in-place butterfly, 128 elements)
        void wht_128() {
            uint tid = gl_LocalInvocationID.x;
            [[unroll]] for (uint stride = 64; stride >= 1; stride >>= 1) {
                barrier();
                uint pair = (tid / stride) * (stride * 2) + (tid % stride);
                float a = sdata[pair];
                float b = sdata[pair + stride];
                sdata[pair] = a + b;
                sdata[pair + stride] = a - b;
            }
            barrier();
            float scale = 1.0 / sqrt(float(head_dim));
            sdata[tid] *= scale;
            barrier();
        }

        // Find quantization bin for a normalized value
        int find_bin(float val) {
            int bin = 0;
            [[unroll]] for (int i = 0; i < 7; i++) {
                if (val >= boundaries[i]) bin = i + 1;
                else break;
            }
            return bin;
        }

        // Quantize shared memory data and write a compressed block to the key cache.
        void quantize_and_pack_k(uint cache_offset) {
            uint tid = gl_LocalInvocationID.x;

            // Compute L2 norm via parallel reduction
            float val = sdata[tid];
            sdata[tid] = val * val;
            barrier();
            [[unroll]] for (uint s = 64; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float norm = sqrt(sdata[0]);

            // Restore and normalize
            barrier();
            sdata[tid] = val;
            barrier();
            float inv_norm = (norm > 0.0) ? (1.0 / norm) : 0.0;
            float normalized = sdata[tid] * inv_norm;

            // Quantize to 3-bit index
            int idx = find_bin(normalized);

            // Thread 0 writes the FP16 norm
            // Pack 3-bit indices into uint array (each uint holds ~10 indices)
            barrier();

            // We store as: [FP16 norm as uint16 in first 2 bytes][48 bytes of packed 3-bit indices]
            // Using uint buffer: first uint has norm in lower 16 bits + first ~10 indices
            // Simpler approach: pack indices into shared memory, then write cooperatively

            // Each thread contributes its 3-bit index. We pack 10 indices per uint (30 bits).
            // 128 indices / 10 = 13 uints (last has 8 indices).
            // But for simplicity and correctness, pack bit-by-bit.

            // Store indices to shared memory
            sdata[tid] = float(idx);
            barrier();

            // Thread 0 writes the entire block
            if (tid == 0) {
                // Write FP16 norm as the first 2 bytes (stored in first uint, low 16 bits)
                uint norm_bits = packHalf2x16(vec2(norm, 0.0));

                // Pack 128 3-bit indices into 48 bytes = 12 uints
                uint packed[13]; // 13 uints = 52 bytes = our block
                packed[0] = norm_bits & 0xFFFFu; // first 2 bytes are norm

                // Pack bits starting at byte offset 2 (bit offset 16 within packed[0])
                uint bit_pos = 16; // start after norm
                for (uint i = 0; i < 128; i++) {
                    uint index3 = uint(sdata[i]) & 0x7u;
                    uint word_idx = bit_pos / 32;
                    uint bit_off = bit_pos % 32;
                    if (i == 0 && word_idx == 0) {
                        packed[word_idx] |= (index3 << bit_off);
                    } else {
                        if (bit_off == 0 && (i == 0 || (bit_pos % 32) == 0))
                            packed[word_idx] = 0;
                        packed[word_idx] |= (index3 << bit_off);
                    }
                    if (bit_off > 29) { // overflow into next uint
                        uint next_word = word_idx + 1;
                        if (bit_off > 29) packed[next_word] |= (index3 >> (32 - bit_off));
                    }
                    bit_pos += 3;
                }

                // Write packed block to cache buffer
                uint base_idx = cache_offset / 4; // uint offset
                uint num_uints = (block_bytes + 3) / 4;
                for (uint w = 0; w < num_uints; w++)
                    k_cache_tq[base_idx + w] = (w < 13) ? packed[w] : 0u;
            }
        }

        // Quantize shared memory data and write a compressed block to the value cache.
        void quantize_and_pack_v(uint cache_offset) {
            uint tid = gl_LocalInvocationID.x;

            float val = sdata[tid];
            sdata[tid] = val * val;
            barrier();
            [[unroll]] for (uint s = 64; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float norm = sqrt(sdata[0]);

            barrier();
            sdata[tid] = val;
            barrier();
            float inv_norm = (norm > 0.0) ? (1.0 / norm) : 0.0;
            float normalized = sdata[tid] * inv_norm;

            int idx = find_bin(normalized);

            barrier();
            sdata[tid] = float(idx);
            barrier();

            if (tid == 0) {
                uint norm_bits = packHalf2x16(vec2(norm, 0.0));
                uint packed[13];
                packed[0] = norm_bits & 0xFFFFu;

                uint bit_pos = 16;
                for (uint i = 0; i < 128; i++) {
                    uint index3 = uint(sdata[i]) & 0x7u;
                    uint word_idx = bit_pos / 32;
                    uint bit_off = bit_pos % 32;
                    if (i == 0 && word_idx == 0) {
                        packed[word_idx] |= (index3 << bit_off);
                    } else {
                        if (bit_off == 0 && (i == 0 || (bit_pos % 32) == 0))
                            packed[word_idx] = 0;
                        packed[word_idx] |= (index3 << bit_off);
                    }
                    if (bit_off > 29) {
                        uint next_word = word_idx + 1;
                        if (bit_off > 29) packed[next_word] |= (index3 >> (32 - bit_off));
                    }
                    bit_pos += 3;
                }

                uint base_idx = cache_offset / 4;
                uint num_uints = (block_bytes + 3) / 4;
                for (uint w = 0; w < num_uints; w++)
                    v_cache_tq[base_idx + w] = (w < 13) ? packed[w] : 0u;
            }
        }

        void main() {
            uint kv_head = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (kv_head >= num_kv_heads || tid >= head_dim) return;

            uint head_offset = kv_head * head_dim;
            uint byte_offset = position * num_kv_heads * block_bytes + kv_head * block_bytes;

            // --- Compress Key ---
            sdata[tid] = k_input[head_offset + tid];
            barrier();
            wht_128();
            // Apply sign flip
            sdata[tid] *= sign_patterns[head_offset + tid];
            barrier();
            quantize_and_pack_k(byte_offset);

            barrier();

            // --- Compress Value ---
            sdata[tid] = v_input[head_offset + tid];
            barrier();
            wht_128();
            sdata[tid] *= sign_patterns[head_offset + tid];
            barrier();
            quantize_and_pack_v(byte_offset);
        }
        """;

    /// <summary>
    /// TurboQuant attention: fused dequant-dot for compressed KV cache.
    /// One workgroup per query head. Tiles over sequence positions.
    /// Handles both compressed (TQ) positions and FP16 recent window.
    ///
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim,
    ///                    uint tq_seq_len, uint fp16_seq_len, uint max_seq_len,
    ///                    uint block_bytes }.
    /// Bindings: 0=Q[num_heads*head_dim], 1=rotated_Q[num_heads*head_dim],
    ///           2=k_cache_tq[...], 3=v_cache_tq[...],
    ///           4=k_cache_fp16[...], 5=v_cache_fp16[...],
    ///           6=output[num_heads*head_dim], 7=codebook[8],
    ///           8=scores_scratch[num_heads * max_seq_len]  (long-context spill).
    ///
    /// Score-storage strategy mirrors the CUDA kernel `llm_tq_attention`:
    ///   • total_seq ≤ MAX_SHARED_SCORES (4096): hot path uses shared memory.
    ///   • total_seq > 4096: spills to `scores_scratch[h*max_seq_len .. +total_seq)`.
    /// The fast path does not touch the scratch buffer, but Vulkan descriptor sets
    /// require it to be bound regardless — the caller passes a 1-float placeholder
    /// when max_seq_len ≤ 4096.
    /// </summary>
    internal const string TqAttention = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q             { float q_data[]; };
        layout(binding = 1) readonly buffer RotQ          { float rotated_q[]; };
        layout(binding = 2) readonly buffer KCacheTQ      { uint k_cache_tq[]; };
        layout(binding = 3) readonly buffer VCacheTQ      { uint v_cache_tq[]; };
        layout(binding = 4) readonly buffer KCacheFP16    { float k_cache_fp16[]; };
        layout(binding = 5) readonly buffer VCacheFP16    { float v_cache_fp16[]; };
        layout(binding = 6) buffer Out                    { float out_data[]; };
        layout(binding = 7) readonly buffer Codebook      { float codebook[8]; };
        layout(binding = 8) buffer ScoresScratch          { float scores_scratch[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint tq_seq_len;      // number of TQ-compressed positions
            uint fp16_seq_len;    // number of FP16 recent positions
            uint max_seq_len;
            uint block_bytes;
        };

        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];    // reduction scratch

        float tq_dequant_dot_k(uint block_base_uint, uint kv_head) {
            float dot = 0.0;
            uint q_off = gl_WorkGroupID.x * head_dim;
            for (uint d = 0; d < head_dim; d++) {
                uint bit_pos = 16u + d * 3u;
                uint word_idx = block_base_uint + bit_pos / 32u;
                uint bit_off = bit_pos & 31u;
                uint raw = k_cache_tq[word_idx] >> bit_off;
                if (bit_off > 29u) raw |= k_cache_tq[word_idx + 1u] << (32u - bit_off);
                int idx = int(raw & 0x7u);
                dot += codebook[idx] * rotated_q[q_off + d];
            }
            return dot;
        }

        void main() {
            uint h = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint q_off = h * head_dim;
            uint out_off = h * head_dim;
            uint total_seq = tq_seq_len + fp16_seq_len;

            bool use_shared = (total_seq <= MAX_SHARED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1a: per-position scores for TQ-compressed positions ───
            for (uint t = tid; t < tq_seq_len; t += 256) {
                uint block_byte_off = t * num_kv_heads * block_bytes + kv_head * block_bytes;
                uint block_base_uint = block_byte_off / 4u;

                // FP16 per-block norm packed in first 2 bytes of the block.
                uint norm_word = k_cache_tq[block_base_uint];
                float norm = unpackHalf2x16(norm_word).x;

                float dot = tq_dequant_dot_k(block_base_uint, kv_head);
                float score = dot * norm * scale;
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }

            // ─── Phase 1b: FP16 recent-window positions ───
            for (uint t = tid; t < fp16_seq_len; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache_fp16[k_off + d];
                float score = dot * scale;
                if (use_shared) scores[tq_seq_len + t] = score;
                else            scores_scratch[scratch_base + tq_seq_len + t] = score;
            }

            // Pad the shared tail with -inf so the max scan ignores stale slots.
            // The scratch path's scans iterate only [0, total_seq), so no padding needed.
            if (use_shared) {
                for (uint t = total_seq + tid; t < MAX_SHARED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }

            barrier();

            // ─── Phase 2: in-place softmax over [0, total_seq) ───
            // Max.
            float local_max = -1.0/0.0;
            for (uint t = tid; t < total_seq; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                local_max = max(local_max, s);
            }
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            // Exp + sum.
            float local_sum = 0.0;
            for (uint t = tid; t < total_seq; t += 256) {
                float s = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                float e = exp(s - max_val);
                if (use_shared) scores[t] = e;
                else            scores_scratch[scratch_base + t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }
            float inv_sum = 1.0 / sdata[0];
            barrier();

            // Normalize → softmax weight per position.
            for (uint t = tid; t < total_seq; t += 256) {
                if (use_shared) scores[t] *= inv_sum;
                else            scores_scratch[scratch_base + t] *= inv_sum;
            }
            barrier();

            // ─── Phase 3: weighted V sum into output[head, :] ───
            for (uint d = tid; d < head_dim; d += 256) {
                float sum_val = 0.0;

                // TQ-compressed positions.
                for (uint t = 0; t < tq_seq_len; t++) {
                    float weight = use_shared ? scores[t] : scores_scratch[scratch_base + t];

                    uint block_byte_off = t * num_kv_heads * block_bytes + kv_head * block_bytes;
                    uint block_base_uint = block_byte_off / 4u;
                    uint norm_word = v_cache_tq[block_base_uint];
                    float norm = unpackHalf2x16(norm_word).x;

                    uint bit_pos = 16u + d * 3u;
                    uint word_idx = block_base_uint + bit_pos / 32u;
                    uint bit_off = bit_pos & 31u;
                    uint raw = v_cache_tq[word_idx] >> bit_off;
                    if (bit_off > 29u) raw |= v_cache_tq[word_idx + 1u] << (32u - bit_off);
                    int idx = int(raw & 0x7u);

                    sum_val += weight * codebook[idx] * norm;
                }

                // FP16 recent-window positions.
                for (uint t = 0; t < fp16_seq_len; t++) {
                    float weight = use_shared
                        ? scores[tq_seq_len + t]
                        : scores_scratch[scratch_base + tq_seq_len + t];
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum_val += weight * v_cache_fp16[v_off + d];
                }

                out_data[out_off + d] = sum_val;
            }
        }
        """;

    // ================================================================
    //  DiT / Diffusion Shaders
    // ================================================================

    /// <summary>
    /// Tiled SGEMM: C[M,N] = A[M,K] × B[N,K]^T
    /// A is activations [M rows, K cols], B is weights [N rows, K cols] (row = one output neuron's weights).
    /// Uses 16×16 shared-memory tiles with +1 column padding to avoid bank conflicts.
    ///
    /// Push constants: { uint M, uint N, uint K }.
    /// Bindings: 0=A (readonly), 1=B (readonly), 2=C (writeonly).
    /// Dispatch: (ceil(M/16), ceil(N/16), 1) with local_size=(16,16,1).
    /// </summary>
    internal const string SgemmF32 = """
        #version 450

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        layout(push_constant) uniform PC {
            uint M;
            uint N;
            uint K;
        } pc;

        layout(binding = 0) readonly  buffer BufA { float a_data[]; };   // [M, K] activations
        layout(binding = 1) readonly  buffer BufB { float b_data[]; };   // [N, K] weights
        layout(binding = 2) writeonly buffer BufC { float c_data[]; };   // [M, N] output

        shared float tileA[16][17]; // +1 column to avoid bank conflicts
        shared float tileB[16][17];

        void main() {
            uint row = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
            uint col = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;

            float acc = 0.0;
            uint numTiles = (pc.K + 15u) / 16u;

            for (uint t = 0u; t < numTiles; t++) {
                uint aCol = t * 16u + gl_LocalInvocationID.y;
                uint bCol = t * 16u + gl_LocalInvocationID.x;

                tileA[gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
                    (row < pc.M && aCol < pc.K) ? a_data[row * pc.K + aCol] : 0.0;

                tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
                    (col < pc.N && bCol < pc.K) ? b_data[col * pc.K + bCol] : 0.0;

                barrier();

                for (uint k = 0u; k < 16u; k++)
                    acc += tileA[gl_LocalInvocationID.x][k] * tileB[gl_LocalInvocationID.y][k];

                barrier();
            }

            if (row < pc.M && col < pc.N)
                c_data[row * pc.N + col] = acc;
        }
        """;

    /// <summary>
    /// Mixed-precision SGEMM: C[M,N] = A[M,K] × B[N,K]^T
    /// A (activations) is fp32 — avoids activation overflow (e.g. SiLU*gate &gt; 65504).
    /// B (weights) is fp16 — bandwidth savings on large weight matrices.
    /// Accumulation and output C are fp32 — full range, no overflow.
    ///
    /// Requires: VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage
    ///
    /// Push constants: { uint M, uint N, uint K }.
    /// Bindings: 0=A (readonly fp32 activations), 1=B (readonly fp16 weights), 2=C (writeonly fp32).
    /// Dispatch: (ceil(M/16), ceil(N/16), 1) with local_size=(16,16,1).
    /// </summary>
    internal const string SgemmF16 = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_16bit_storage : require

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        layout(push_constant) uniform PC {
            uint M;
            uint N;
            uint K;
        } pc;

        layout(binding = 0) readonly  buffer BufA { float    a_data[]; };
        layout(binding = 1) readonly  buffer BufB { float16_t b_data[]; };
        layout(binding = 2) writeonly buffer BufC { float    c_data[]; };

        // fp32 shared tiles. A reads fp32, B reads fp16 (converted on load).
        shared float tileA[16][17];
        shared float tileB[16][17];

        void main() {
            uint row = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
            uint col = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;

            float acc = 0.0;
            uint numTiles = (pc.K + 15u) / 16u;

            for (uint t = 0u; t < numTiles; t++) {
                uint aCol = t * 16u + gl_LocalInvocationID.y;
                uint bCol = t * 16u + gl_LocalInvocationID.x;

                tileA[gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
                    (row < pc.M && aCol < pc.K) ? a_data[row * pc.K + aCol] : 0.0;

                tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
                    (col < pc.N && bCol < pc.K) ? float(b_data[col * pc.K + bCol]) : 0.0;

                barrier();

                for (uint k = 0u; k < 16u; k++)
                    acc += tileA[gl_LocalInvocationID.x][k] * tileB[gl_LocalInvocationID.y][k];

                barrier();
            }

            if (row < pc.M && col < pc.N)
                c_data[row * pc.N + col] = acc;
        }
        """;

    /// <summary>
    /// Tiled int8-weight × fp16-activation SGEMM: C[M,N] = A[M,K] × (scale * B)[N,K]^T
    /// A is fp16 activations, B is int8 weights (per-row quantized with fp16 scales).
    /// Accumulation is done in fp32.
    ///
    /// Requires: VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage + VK_KHR_8bit_storage
    ///
    /// Push constants: { uint M, uint N, uint K }.
    /// Bindings: 0=A (fp16 activations [M,K]), 1=B (int8 weights [N,K]),
    ///           2=scale (fp16 per-row scales [N]), 3=C (fp16 output [M,N]).
    /// Dispatch: (ceil(M/16), ceil(N/16), 1) with local_size=(16,16,1).
    /// </summary>
    internal const string SgemmInt8Fp16 = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_explicit_arithmetic_types_int8    : require
        #extension GL_EXT_shader_16bit_storage : require
        #extension GL_EXT_shader_8bit_storage  : require

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

        layout(push_constant) uniform PC {
            uint M;
            uint N;
            uint K;
        } pc;

        layout(binding = 0) readonly  buffer BufA  { float16_t a_data[]; };
        layout(binding = 1) readonly  buffer BufB  { int8_t    b_data[]; };
        layout(binding = 2) readonly  buffer BufS  { float16_t b_scale[]; };
        layout(binding = 3) writeonly buffer BufC  { float16_t c_data[]; };

        shared float16_t tileA[16][17];
        shared int8_t    tileB[16][17];

        void main() {
            uint row = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
            uint col = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;

            float acc = 0.0;
            uint numTiles = (pc.K + 15u) / 16u;

            for (uint t = 0u; t < numTiles; t++) {
                uint aCol = t * 16u + gl_LocalInvocationID.y;
                uint bCol = t * 16u + gl_LocalInvocationID.x;

                tileA[gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
                    (row < pc.M && aCol < pc.K) ? a_data[row * pc.K + aCol] : float16_t(0.0);

                tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
                    (col < pc.N && bCol < pc.K) ? b_data[col * pc.K + bCol] : int8_t(0);

                barrier();

                for (uint k = 0u; k < 16u; k++)
                    acc += float(tileA[gl_LocalInvocationID.x][k]) *
                           float(tileB[gl_LocalInvocationID.y][k]);

                barrier();
            }

            if (row < pc.M && col < pc.N) {
                float scale = float(b_scale[col]);
                c_data[row * pc.N + col] = float16_t(acc * scale);
            }
        }
        """;

    /// <summary>
    /// Tiled bf16 SGEMM: C[M,N] = A[M,K] × B[N,K]^T
    /// All inputs and output are bfloat16_t. Accumulation in fp32.
    /// Requires: VK_KHR_shader_bfloat16 + VK_KHR_16bit_storage
    /// Push constants: { uint M, uint N, uint K }.
    /// Bindings: 0=A (readonly bf16), 1=B (readonly bf16), 2=C (writeonly bf16).
    /// </summary>
    internal const string SgemmBf16 = """
        #version 450
        #extension GL_KHR_shader_bfloat16 : require
        #extension GL_EXT_shader_16bit_storage : require

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
        layout(push_constant) uniform PC { uint M; uint N; uint K; } pc;
        layout(binding = 0) readonly  buffer BufA { bfloat16_t a_data[]; };
        layout(binding = 1) readonly  buffer BufB { bfloat16_t b_data[]; };
        layout(binding = 2) writeonly buffer BufC { bfloat16_t c_data[]; };
        // fp32 shared tiles to avoid driver issues with bf16 shared memory;
        // global loads/stores remain bf16 so VRAM bandwidth is fully saved.
        shared float tileA[16][17];
        shared float tileB[16][17];
        void main() {
            uint row = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
            uint col = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;
            float acc = 0.0;
            uint numTiles = (pc.K + 15u) / 16u;
            for (uint t = 0u; t < numTiles; t++) {
                uint aCol = t * 16u + gl_LocalInvocationID.y;
                uint bCol = t * 16u + gl_LocalInvocationID.x;
                tileA[gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
                    (row < pc.M && aCol < pc.K) ? float(a_data[row * pc.K + aCol]) : 0.0;
                tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
                    (col < pc.N && bCol < pc.K) ? float(b_data[col * pc.K + bCol]) : 0.0;
                barrier();
                for (uint k = 0u; k < 16u; k++)
                    acc += tileA[gl_LocalInvocationID.x][k] * tileB[gl_LocalInvocationID.y][k];
                barrier();
            }
            if (row < pc.M && col < pc.N)
                c_data[row * pc.N + col] = bfloat16_t(acc);
        }
        """;

    /// <summary>
    /// Tiled fp8 × fp16 SGEMM: C[M,N] = A[M,K] × B[N,K]^T
    /// A is fp16 activations, B is fp8 E4M3 weights, C is fp16 output.
    /// Requires: VK_EXT_shader_float8 + VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage
    /// Push constants: { uint M, uint N, uint K }.
    /// Bindings: 0=A (fp16), 1=B (fp8 e4m3), 2=C (fp16).
    /// </summary>
    internal const string SgemmFp8 = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float8_e4m3 : require

        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
        layout(push_constant) uniform PC { uint M; uint N; uint K; } pc;
        layout(binding = 0) readonly  buffer BufA { float8_e4m3_t a_data[]; };
        layout(binding = 1) readonly  buffer BufB { float8_e4m3_t b_data[]; };
        layout(binding = 2) writeonly buffer BufC { float c_data[]; };
        // fp32 shared tiles: avoids driver issues with fp16/fp8 shared memory
        shared float tileA[16][17];
        shared float tileB[16][17];
        void main() {
            uint row = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.x;
            uint col = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.y;
            float acc = 0.0;
            uint numTiles = (pc.K + 15u) / 16u;
            for (uint t = 0u; t < numTiles; t++) {
                uint aCol = t * 16u + gl_LocalInvocationID.y;
                uint bCol = t * 16u + gl_LocalInvocationID.x;
                tileA[gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
                    (row < pc.M && aCol < pc.K) ? float(a_data[row * pc.K + aCol]) : 0.0;
                tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =
                    (col < pc.N && bCol < pc.K) ? float(b_data[col * pc.K + bCol]) : 0.0;
                barrier();
                for (uint k = 0u; k < 16u; k++)
                    acc += tileA[gl_LocalInvocationID.x][k] * tileB[gl_LocalInvocationID.y][k];
                barrier();
            }
            if (row < pc.M && col < pc.N)
                c_data[row * pc.N + col] = acc;
        }
        """;

    /// <summary>
    /// GPU-side Q5_K_M dequantization: one workgroup per block, 256 threads per workgroup.
    /// Q5_K block layout (176 bytes / 256 elements):
    ///   [0:2]   FP16 d (super-block scale)
    ///   [2:4]   FP16 dmin
    ///   [4:16]  12 bytes packed 6-bit scales/mins
    ///   [16:48] 32 bytes qh (1 high bit per element)
    ///   [48:176] 128 bytes ql (4-bit nibbles, 2 per byte)
    /// Requires: VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage
    /// Push constants: { uint numBlocks }.
    /// Bindings: 0=src (raw uint32 array), 1=dst (fp16 array).
    /// Dispatch: (numBlocks, 1, 1).
    /// </summary>
    internal const string DequantQ5KM = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_16bit_storage : require

        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
        layout(push_constant) uniform PC { uint numBlocks; } pc;
        layout(binding = 0) readonly  buffer SrcBuf { uint src[]; };
        layout(binding = 1) writeonly buffer DstBuf { float16_t dst[]; };

        uint byteAt(uint bi) { return (src[bi >> 2u] >> ((bi & 3u) << 3u)) & 0xFFu; }

        void getScaleMinK4(uint j, uint scBase, out uint sc, out uint mn) {
            if (j < 4u) { sc = byteAt(scBase + j) & 63u; mn = byteAt(scBase + j + 4u) & 63u; }
            else {
                sc = (byteAt(scBase + j + 4u) & 0xFu) | ((byteAt(scBase + j - 4u) >> 6u) << 4u);
                mn = (byteAt(scBase + j + 4u) >> 4u)  | ((byteAt(scBase + j)       >> 6u) << 4u);
            }
        }

        void main() {
            uint blockIdx = gl_WorkGroupID.x;
            if (blockIdx >= pc.numBlocks) return;

            uint elem  = gl_LocalInvocationID.x;
            uint bBase = blockIdx * 176u;

            uint dBits    = byteAt(bBase + 0u) | (byteAt(bBase + 1u) << 8u);
            uint dminBits = byteAt(bBase + 2u) | (byteAt(bBase + 3u) << 8u);
            float d    = unpackHalf2x16(dBits).x;
            float dmin = unpackHalf2x16(dminBits).x;

            uint scBase = bBase + 4u;
            uint qhBase = bBase + 16u;
            uint qlBase = bBase + 48u;

            uint grp   = elem / 64u;
            uint loc   = elem % 64u;
            uint lo_hi = loc  / 32u;
            uint l     = loc  % 32u;

            uint scaleIdx = grp * 2u + lo_hi;
            uint sc, mn;
            getScaleMinK4(scaleIdx, scBase, sc, mn);
            float df  = d    * float(sc);
            float dmf = dmin * float(mn);

            uint u      = 1u << (grp * 2u + lo_hi);
            uint hBit   = ((byteAt(qhBase + l) & u) != 0u) ? 16u : 0u;
            uint qlByte = byteAt(qlBase + grp * 32u + l);
            uint q5     = (lo_hi == 0u ? (qlByte & 0xFu) : (qlByte >> 4u)) + hBit;

            dst[blockIdx * 256u + elem] = float16_t(df * float(q5) - dmf);
        }
        """;

    /// <summary>
    /// GPU-side Q4_K_M dequantization: one workgroup per block, 256 threads per workgroup.
    /// Q4_K block layout (144 bytes / 256 elements):
    ///   [0:2]   FP16 d
    ///   [2:4]   FP16 dmin
    ///   [4:16]  12 bytes packed 6-bit scales/mins
    ///   [16:144] 128 bytes ql (4-bit nibbles, 2 per byte)
    /// Requires: VK_KHR_shader_float16_int8 + VK_KHR_16bit_storage
    /// Push constants: { uint numBlocks }.
    /// Bindings: 0=src (raw uint32 array), 1=dst (fp16 array).
    /// Dispatch: (numBlocks, 1, 1).
    /// </summary>
    internal const string DequantQ4KM = """
        #version 450
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
        #extension GL_EXT_shader_16bit_storage : require

        layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
        layout(push_constant) uniform PC { uint numBlocks; } pc;
        layout(binding = 0) readonly  buffer SrcBuf { uint src[]; };
        layout(binding = 1) writeonly buffer DstBuf { float16_t dst[]; };

        uint byteAt(uint bi) { return (src[bi >> 2u] >> ((bi & 3u) << 3u)) & 0xFFu; }

        void getScaleMinK4(uint j, uint scBase, out uint sc, out uint mn) {
            if (j < 4u) { sc = byteAt(scBase + j) & 63u; mn = byteAt(scBase + j + 4u) & 63u; }
            else {
                sc = (byteAt(scBase + j + 4u) & 0xFu) | ((byteAt(scBase + j - 4u) >> 6u) << 4u);
                mn = (byteAt(scBase + j + 4u) >> 4u)  | ((byteAt(scBase + j)       >> 6u) << 4u);
            }
        }

        void main() {
            uint blockIdx = gl_WorkGroupID.x;
            if (blockIdx >= pc.numBlocks) return;

            uint elem  = gl_LocalInvocationID.x;
            uint bBase = blockIdx * 144u;

            uint dBits    = byteAt(bBase + 0u) | (byteAt(bBase + 1u) << 8u);
            uint dminBits = byteAt(bBase + 2u) | (byteAt(bBase + 3u) << 8u);
            float d    = unpackHalf2x16(dBits).x;
            float dmin = unpackHalf2x16(dminBits).x;

            uint scBase = bBase + 4u;
            uint qlBase = bBase + 16u;

            uint grp   = elem / 64u;
            uint loc   = elem % 64u;
            uint lo_hi = loc  / 32u;
            uint l     = loc  % 32u;

            uint scaleIdx = grp * 2u + lo_hi;
            uint sc, mn;
            getScaleMinK4(scaleIdx, scBase, sc, mn);
            float df  = d    * float(sc);
            float dmf = dmin * float(mn);

            uint qlByte = byteAt(qlBase + grp * 32u + l);
            uint q4     = (lo_hi == 0u) ? (qlByte & 0xFu) : (qlByte >> 4u);

            dst[blockIdx * 256u + elem] = float16_t(df * float(q4) - dmf);
        }
        """;

    // ── Image upscaler ops (RRDBNet) ──────────────────────────────────────

    /// <summary>
    /// 2D convolution: output[outCh, H, W] = conv(input[inCh, H, W], weight[outCh, inCh, k, k]) + bias[outCh]
    /// stride=1, configurable padding (default same).
    /// Each thread computes one output element (oc, oh, ow).
    /// Push constants: { inCh, outCh, height, width, ksize, padding }.
    /// Bindings: 0=input, 1=weight, 2=bias, 3=output.
    /// Dispatch: ceil(outCh * H * W / 256).
    /// </summary>
    /// <summary>
    /// Conv2d shader using a 2D workgroup dispatch: X=outCh, Y=ceil(H*W/256).
    /// All 256 threads in a workgroup share the same output channel, so they
    /// cooperatively load that channel's weight vector into shared memory once —
    /// reducing weight reads from global memory by 256×.
    ///
    /// Dispatch: (outCh, ceil(H*W / 256), 1) — matches VulkanBackend.Conv2d.
    /// Push constants unchanged: { inCh, outCh, height, width, ksize, padding }.
    /// </summary>
    internal const string Conv2d = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[];  };
        layout(binding = 1) readonly  buffer Weight { float weight_data[]; };
        layout(binding = 2) readonly  buffer Bias   { float bias_data[];   };
        layout(binding = 3) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint inCh;
            uint outCh;
            uint height;
            uint width;
            uint ksize;
            uint padding;
        };

        // Shared memory for one output-channel's weight vector.
        // Max weight per channel: 192 inCh × 3×3 kernel = 1728 floats = 6.75 KB.
        // 2048 slots provides safe alignment margin.
        shared float sWeights[2048];

        void main() {
            uint oc      = gl_WorkGroupID.x;           // output channel index
            uint tileIdx = gl_WorkGroupID.y;           // spatial tile within channel
            uint lid     = gl_LocalInvocationID.x;     // thread within tile (0..255)

            uint hw  = height * width;
            uint pos = tileIdx * 256u + lid;           // output pixel index

            // Cooperatively load all weights for this output channel into shared memory.
            // wLen ≤ 2048 for all configs in RRDBNet; each thread loads ceil(wLen/256) slots.
            uint wLen  = inCh * ksize * ksize;
            uint wBase = oc * wLen;
            for (uint i = lid; i < wLen; i += 256u)
                sWeights[i] = weight_data[wBase + i];

            // Ensure all threads see the fully loaded weights before computing.
            barrier();
            memoryBarrierShared();

            if (oc >= outCh || pos >= hw) return;

            uint oh = pos / width;
            uint ow = pos % width;

            float acc = bias_data[oc];
            for (uint ic = 0u; ic < inCh; ic++) {
                uint iBase   = ic * hw;
                uint wIcBase = ic * ksize * ksize;
                for (uint kh = 0u; kh < ksize; kh++) {
                    for (uint kw = 0u; kw < ksize; kw++) {
                        int ih = int(oh + kh) - int(padding);
                        int iw = int(ow + kw) - int(padding);
                        if (uint(ih) < height && uint(iw) < width)
                            acc += input_data[iBase + uint(ih) * width + uint(iw)]
                                 * sWeights[wIcBase + kh * ksize + kw];
                    }
                }
            }
            output_data[oc * hw + pos] = acc;
        }
        """;

    /// <summary>
    /// LeakyReLU in-place: data[i] = data[i] >= 0 ? data[i] : negSlope * data[i]
    /// Push constants: { n, negSlope }.
    /// Bindings: 0=data (in/out).
    /// </summary>
    internal const string LeakyRelu = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint  n;
            float negSlope;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            float x = data[i];
            data[i] = x >= 0.0 ? x : negSlope * x;
        }
        """;

    /// <summary>
    /// Clamp in-place: data[i] = clamp(data[i], minVal, maxVal)
    /// Push constants: { n, minVal, maxVal }.
    /// Bindings: 0=data (in/out).
    /// </summary>
    internal const string ClampInPlace = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint  n;
            float minVal;
            float maxVal;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            data[i] = clamp(data[i], minVal, maxVal);
        }
        """;

    /// <summary>
    /// Channel concatenation: output[(aCh+bCh), hw] from a[aCh, hw] and b[bCh, hw].
    /// Push constants: { aCh, bCh, hw }.
    /// Bindings: 0=a, 1=b, 2=output.
    /// Dispatch: ceil((aCh+bCh)*hw / 256).
    /// </summary>
    internal const string CatChannels = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer A      { float a_data[];      };
        layout(binding = 1) readonly  buffer B      { float b_data[];      };
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint aCh;
            uint bCh;
            uint hw;
        };

        void main() {
            uint idx   = gl_GlobalInvocationID.x;
            uint outCh = aCh + bCh;
            if (idx >= outCh * hw) return;

            uint c   = idx / hw;
            uint pos = idx % hw;
            output_data[idx] = (c < aCh)
                ? a_data[c * hw + pos]
                : b_data[(c - aCh) * hw + pos];
        }
        """;

    /// <summary>
    /// Pixel shuffle: [inCh, H, W] → [inCh/r², H*r, W*r]  (r = upscale)
    /// Push constants: { inCh, h, w, upscale }.
    /// Bindings: 0=input, 1=output.
    /// Dispatch: ceil(outCh*outH*outW / 256).
    /// </summary>
    internal const string PixelShuffle = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[];  };
        layout(binding = 1) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint inCh;
            uint h;
            uint w;
            uint upscale;
        };

        void main() {
            uint r2    = upscale * upscale;
            uint outCh = inCh / r2;
            uint outH  = h * upscale;
            uint outW  = w * upscale;

            uint idx = gl_GlobalInvocationID.x;
            if (idx >= outCh * outH * outW) return;

            uint outHW = outH * outW;
            uint oc    = idx / outHW;
            uint pos   = idx % outHW;
            uint oh    = pos / outW;
            uint ow    = pos % outW;

            uint ih = oh / upscale;
            uint iw = ow / upscale;
            uint rh = oh % upscale;
            uint rw = ow % upscale;

            // Input channel: oc * r² + rh * upscale + rw
            uint ic = oc * r2 + rh * upscale + rw;
            output_data[idx] = input_data[ic * h * w + ih * w + iw];
        }
        """;

    /// <summary>
    /// Pixel unshuffle (inverse): [inCh, H*r, W*r] → [inCh*r², H, W]  (r = downscale)
    /// Push constants: { inCh, h (output), w (output), downscale }.
    /// Bindings: 0=input, 1=output.
    /// Dispatch: ceil(inCh*r²*h*w / 256).
    /// </summary>
    internal const string PixelUnshuffle = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[];  };
        layout(binding = 1) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint inCh;
            uint h;         // output height = inputH / downscale
            uint w;         // output width  = inputW / downscale
            uint downscale;
        };

        void main() {
            uint d2    = downscale * downscale;
            uint outCh = inCh * d2;
            uint inH   = h * downscale;
            uint inW   = w * downscale;

            uint idx = gl_GlobalInvocationID.x;
            if (idx >= outCh * h * w) return;

            uint hw  = h * w;
            uint oc  = idx / hw;
            uint pos = idx % hw;
            uint oh  = pos / w;
            uint ow  = pos % w;

            uint ic  = oc / d2;
            uint rem = oc % d2;
            uint rh  = rem / downscale;
            uint rw  = rem % downscale;

            uint ih = oh * downscale + rh;
            uint iw = ow * downscale + rw;
            output_data[idx] = input_data[ic * inH * inW + ih * inW + iw];
        }
        """;

    /// <summary>
    /// Nearest-neighbour 2× upsample: [ch, H, W] → [ch, 2H, 2W]
    /// Push constants: { ch, h, w }.
    /// Bindings: 0=input, 1=output.
    /// Dispatch: ceil(ch*2H*2W / 256).
    /// </summary>
    internal const string Upsample2xNearest = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[];  };
        layout(binding = 1) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint ch;
            uint h;
            uint w;
        };

        void main() {
            uint idx   = gl_GlobalInvocationID.x;
            uint outHW = 4u * h * w;   // (2h)*(2w)
            if (idx >= ch * outHW) return;

            uint c   = idx / outHW;
            uint pos = idx % outHW;
            uint oh  = pos / (2u * w);
            uint ow  = pos % (2u * w);

            output_data[idx] = input_data[c * h * w + (oh / 2u) * w + (ow / 2u)];
        }
        """;
}
