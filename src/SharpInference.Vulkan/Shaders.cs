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
    /// Batched RMS Normalization: normalizes each of <c>num_tokens</c> independent rows of a
    /// <c>[num_tokens][n]</c> buffer in a single dispatch. Row r (token r) is normalized EXACTLY
    /// as the single-row <see cref="RmsNorm"/> — its own sum-of-squares reduction over its n
    /// elements, then scale + the shared <c>[n]</c> weight. Bit-identical to <c>num_tokens</c>
    /// separate <see cref="RmsNorm"/> calls (the per-row math is independent; floating-point
    /// reduction order within a row matches the single-row shader's 256-stride + tree reduction).
    ///
    /// One workgroup per row: row index r = <c>gl_WorkGroupID.x</c> (dispatch num_tokens groups).
    /// Push constants: { uint n, float eps, uint num_tokens }.
    /// Bindings: 0=input ([num_tokens][n]), 1=weight ([n], shared), 2=output ([num_tokens][n]).
    /// </summary>
    internal const string RmsNormBatched = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Input  { float input_data[]; };
        layout(binding = 1) readonly buffer Weight { float weight_data[]; };
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint n;
            float eps;
            uint num_tokens;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row = gl_WorkGroupID.x;
            if (row >= num_tokens) return;

            uint base_off = row * n;

            // Phase 1: each thread accumulates sum of squares for its stride within this row.
            float sum = 0.0;
            for (uint i = tid; i < n; i += 256) {
                float v = input_data[base_off + i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            // Phase 2: parallel reduction in shared memory.
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            // Phase 3: compute scale factor.
            float scale = inversesqrt(sdata[0] / float(n) + eps);

            // Phase 4: apply normalization and the shared weight.
            for (uint i = tid; i < n; i += 256) {
                output_data[base_off + i] = input_data[base_off + i] * scale * weight_data[i];
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
    /// Fused tanh-approximate GELU(gate) * up (Gemma FFN activation):
    /// gate[i] = 0.5 * g * (1 + tanh(0.7978845608028654 * (g + 0.044715 * g^3))) * up[i]
    /// where g = gate[i]. Clone of <see cref="SiLuMul"/> with SiLU swapped for GELU-tanh.
    /// Push constants: { uint n }.
    /// Bindings: 0=gate (in/out), 1=up (in).
    /// Matches the CPU reference SimdKernels.GeluTanhMul / CUDA llm_gelu_tanh_mul.
    /// </summary>
    internal const string GeluTanhMul = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Gate { float gate_data[]; };
        layout(binding = 1) readonly buffer Up { float up_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            float g = gate_data[i];
            float inner = 0.7978845608028654 * (g + 0.044715 * g * g * g);
            gate_data[i] = 0.5 * g * (1.0 + tanh(inner)) * up_data[i];
        }
        """;

    /// <summary>
    /// SiLU (Swish) activation in-place: x[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i])).
    /// Push constants: { uint n }.
    /// Bindings: 0=x (in/out).
    /// Standalone (unfused) counterpart to <see cref="SiLuMul"/>; matches the CPU
    /// GdnKernels.SiLu / CUDA SiLUInPlace formula.
    /// </summary>
    internal const string SiLU = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params { uint n; };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            float x = x_data[i];
            x_data[i] = x / (1.0 + exp(-x));
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
    /// In-place final-logit softcap: x[i] = tanh(x[i] / cap) * cap for i in [0, n).
    /// Used by Gemma to clip extreme logits before sampling (cap=30).
    /// Push constants: { uint n, float cap } (reuses the ScaleParams layout, scale=cap).
    /// Bindings: 0=data (in/out).
    /// Matches the CPU reference SimdKernels.SoftcapInPlace / CUDA llm_softcap_inplace.
    /// </summary>
    internal const string Softcap = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data[]; };

        layout(push_constant) uniform Params {
            uint n;
            float cap;
        };

        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= n) return;
            data[i] = tanh(data[i] / cap) * cap;
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
    /// Gated-DeltaNet depthwise causal conv1d for a single decode token. One thread per
    /// channel. State layout <c>[(kernel-1), channels]</c> row-major, oldest first; updated
    /// in place. Weight layout <c>[kernel, channels]</c>.
    ///   output[c] = weight[K-1,c]*x[c] + Σ_{k=0..K-2} weight[k,c]*state[k,c]
    ///   shift state: state[0..K-3] = state[1..K-2]; state[K-2] = x[c]
    /// Mirrors CUDA llm_gdn_conv1d_decode / CPU GdnKernels.CausalDepthwiseConv1dDecode.
    /// Push constants: { uint channels, uint kernel_size }.
    /// Bindings: 0=x (in), 1=state (in/out), 2=weight (in), 3=output (out).
    /// Dispatch: ceil(channels / 256) workgroups of 256 threads.
    /// </summary>
    internal const string GdnConv1dDecode = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer X     { float x_data[]; };
        layout(binding = 1)           buffer State { float state_data[]; };
        layout(binding = 2) readonly  buffer W     { float w_data[]; };
        layout(binding = 3) writeonly buffer O     { float o_data[]; };

        layout(push_constant) uniform Params {
            uint channels;
            uint kernel_size;
        };

        void main() {
            uint c = gl_GlobalInvocationID.x;
            if (c >= channels) return;

            uint retained = kernel_size - 1u;

            // Read old state values into registers (kernel_size <= 4 in our models).
            float s_old[4];
            for (uint k = 0u; k < retained; k++)
                s_old[k] = state_data[k * channels + c];

            float x_c = x_data[c];
            float sum = w_data[retained * channels + c] * x_c;
            for (uint k = 0u; k < retained; k++)
                sum += w_data[k * channels + c] * s_old[k];
            o_data[c] = sum;

            // Shift state forward in time (drop oldest, append x).
            for (uint k = 0u; k + 1u < retained; k++)
                state_data[k * channels + c] = s_old[k + 1u];
            if (retained >= 1u)
                state_data[(retained - 1u) * channels + c] = x_c;
        }
        """;

    /// <summary>
    /// Gated-DeltaNet L2 normalization per head (no learned weights). One workgroup per head,
    /// 256-thread tree reduction. Matches GdnKernels.L2NormPerHead / CUDA llm_gdn_l2_norm_per_head:
    ///   scale = 1 / max(sqrt(Σ x²), eps).
    /// This differs from <see cref="HeadNormPure"/> which divides by sqrt(mean + eps). Operates on
    /// the sub-region of the bound buffer starting at <c>offset</c> float elements.
    /// Push constants: { uint head_dim, uint num_heads, float eps, uint offset }.
    /// Bindings: 0=data (in/out).
    /// Dispatch: num_heads workgroups of 256 threads.
    /// </summary>
    internal const string GdnL2NormPerHead = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data { float data_buf[]; };

        layout(push_constant) uniform Params {
            uint head_dim;
            uint num_heads;
            float eps;
            uint offset;
        };

        shared float sdata[256];

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint head = gl_WorkGroupID.x;
            if (head >= num_heads) return;

            uint base_off = offset + head * head_dim;

            float sum = 0.0;
            for (uint i = tid; i < head_dim; i += 256u) {
                float v = data_buf[base_off + i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            [[unroll]] for (uint s = 128u; s > 0u; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            float norm = sqrt(sdata[0]);
            float divisor = norm > eps ? norm : eps;
            float inv = 1.0 / divisor;
            for (uint i = tid; i < head_dim; i += 256u) {
                data_buf[base_off + i] = data_buf[base_off + i] * inv;
            }
        }
        """;

    /// <summary>
    /// Gated-DeltaNet tile-heads (GQA-style broadcast). One thread per dst element.
    ///   dst[h_dst, j] = src[h_dst % src_heads, j] for h_dst in [0, src_heads*repeat).
    /// Matches GdnKernels.TileHeads / CUDA llm_gdn_tile_heads (tile, NOT torch repeat_interleave).
    /// <c>src_offset</c>/<c>dst_offset</c> are float-element offsets into the bound buffers.
    /// Push constants: { uint src_heads, uint repeat, uint head_dim, uint src_offset, uint dst_offset }.
    /// Bindings: 0=src (in), 1=dst (out).
    /// Dispatch: ceil(src_heads*repeat*head_dim / 256) workgroups of 256 threads.
    /// </summary>
    internal const string GdnTileHeads = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Src { float src_data[]; };
        layout(binding = 1) writeonly buffer Dst { float dst_data[]; };

        layout(push_constant) uniform Params {
            uint src_heads;
            uint repeat;
            uint head_dim;
            uint src_offset;
            uint dst_offset;
        };

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            uint total = src_heads * repeat * head_dim;
            if (idx >= total) return;
            uint j = idx % head_dim;
            uint h_dst = idx / head_dim;
            uint h_src = h_dst % src_heads;
            dst_data[dst_offset + idx] = src_data[src_offset + h_src * head_dim + j];
        }
        """;

    /// <summary>
    /// Gated-DeltaNet recurrence delta-rule scan for a single decode token (issue #356).
    /// One workgroup per v-head; <c>local_size_x = headDim</c> (HARDCODED 128 — both target
    /// models qwen36-35b-a3b / qwen36-27b-mtp have headDim=128, so every one of the 128
    /// invocations is active and reaches every <c>barrier()</c>). Each thread owns output
    /// column <c>j</c>. State layout <c>S[h*d*d + i*d + j]</c> (i=key axis, j=value/output
    /// axis), updated in place. Per head:
    ///   decay = exp(softplus(alpha_in[h]+dt_bias[h]) · ssm_a[h]); b = sigmoid(beta[h])
    ///   pass A: S *= decay; p[j] = Σ_i k[i]·S[i,j]
    ///   d[j]   = b·(v[j] − p[j])
    ///   pass B: S[i,j] += k[i]·d[j]; o[j] = (1/√d)·Σ_i q[i]·S[i,j]
    ///   o = RMSNorm(o)·norm_weight; o *= SiLU(z)
    /// Mirrors CUDA <c>llm_gdn_recurrence_decode</c> / CPU <c>GdnKernels.GdnRecurrenceDecode</c>
    /// op-for-op (full-precision exp/log/inversesqrt to track the CPU oracle tightly).
    /// Push constants: { uint hv, uint d, float norm_eps }.
    /// Bindings: 0=state (in/out), 1=q, 2=k, 3=v, 4=alpha_in, 5=beta, 6=ssm_a, 7=dt_bias,
    ///           8=norm_weight, 9=z (all readonly), 10=output (writeonly).
    /// Dispatch: hv workgroups of 128 threads.
    /// </summary>
    internal const string GdnRecurrenceDecode = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 128) in;

        layout(binding = 0)           buffer State  { float state_data[]; };
        layout(binding = 1) readonly  buffer Q      { float q_data[]; };
        layout(binding = 2) readonly  buffer K      { float k_data[]; };
        layout(binding = 3) readonly  buffer V      { float v_data[]; };
        layout(binding = 4) readonly  buffer AlphaIn{ float alpha_data[]; };
        layout(binding = 5) readonly  buffer Beta   { float beta_data[]; };
        layout(binding = 6) readonly  buffer SsmA   { float ssma_data[]; };
        layout(binding = 7) readonly  buffer DtBias { float dtbias_data[]; };
        layout(binding = 8) readonly  buffer NormW  { float normw_data[]; };
        layout(binding = 9) readonly  buffer Z      { float z_data[]; };
        layout(binding = 10) writeonly buffer O     { float o_data[]; };

        layout(push_constant) uniform Params {
            uint hv;
            uint d;
            float norm_eps;
        };

        shared float sK[128];
        shared float sQ[128];
        shared float sV[128];
        shared float sZ[128];
        shared float sNormW[128];
        shared float sP[128];
        shared float sD[128];
        shared float sRed[128];

        void main() {
            uint h = gl_WorkGroupID.x;
            uint j = gl_LocalInvocationID.x;

            // Load per-head Q, K, V, Z and per-dim norm weight into shared memory.
            uint hd_off = h * d;
            sK[j]     = k_data[hd_off + j];
            sQ[j]     = q_data[hd_off + j];
            sV[j]     = v_data[hd_off + j];
            sZ[j]     = z_data[hd_off + j];
            sNormW[j] = normw_data[j];
            barrier();

            // Per-head scalar gates.
            float alpha_x = alpha_data[h] + dtbias_data[h];
            float dt      = alpha_x >= 20.0 ? alpha_x : log(1.0 + exp(alpha_x));   // softplus
            float decay   = exp(dt * ssma_data[h]);
            float b_sc    = 1.0 / (1.0 + exp(-beta_data[h]));

            uint state_base = h * d * d;

            // Pass A: decay S, then accumulate p[j] = Σ_i k[i] · S[i,j].
            float p_local = 0.0;
            for (uint i = 0u; i < d; i++) {
                uint off = state_base + i * d + j;
                float sij = state_data[off] * decay;
                state_data[off] = sij;
                p_local += sK[i] * sij;
            }
            sP[j] = p_local;
            barrier();

            // Compute d[j].
            float d_j = b_sc * (sV[j] - sP[j]);
            sD[j] = d_j;
            barrier();

            // Pass B: rank-1 update S[i,j] += k[i] · d[j], fused with readout o[j].
            float o_local = 0.0;
            for (uint i = 0u; i < d; i++) {
                uint off = state_base + i * d + j;
                float sij = state_data[off] + sK[i] * d_j;
                state_data[off] = sij;
                o_local += sQ[i] * sij;
            }

            // Scale by 1/sqrt(d).
            o_local *= inversesqrt(float(d));

            // RMSNorm: scale = rsqrt(sumSq/d + eps), then o = o * scale * normWeight.
            sRed[j] = o_local * o_local;
            barrier();
            [[unroll]] for (uint s = 64u; s > 0u; s >>= 1) {
                if (j < s) sRed[j] += sRed[j + s];
                barrier();
            }
            float scale = inversesqrt(sRed[0] / float(d) + norm_eps);

            float o_normed = o_local * scale * sNormW[j];

            // SiLU(z) gate.
            float zv = sZ[j];
            float silu = zv / (1.0 + exp(-zv));

            o_data[hd_off + j] = o_normed * silu;
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
    /// Batched per-head RMSNorm: applies <see cref="HeadNorm"/> independently to each head of
    /// each of <c>num_tokens</c> rows in a <c>[num_tokens][num_heads*head_dim]</c> buffer, in a
    /// single dispatch. Processes <c>num_tokens * num_heads</c> head-groups: head index
    /// h = <c>gl_WorkGroupID.x</c>, token row r = <c>gl_WorkGroupID.y</c> (dispatch
    /// num_heads × num_tokens groups). The weight (shared <c>[head_dim]</c> for Qwen3, or
    /// per-channel <c>[num_heads*head_dim]</c> for OLMoE via weight_stride) is shared across rows.
    /// Bit-identical to <c>num_tokens</c> separate <see cref="HeadNorm"/> calls.
    /// Push constants: { uint head_dim, uint num_heads, float eps, uint weight_stride, uint num_tokens }.
    /// Bindings: 0=data ([num_tokens][num_heads*head_dim], in/out), 1=weight (in).
    /// </summary>
    internal const string HeadNormBatched = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) buffer Data   { float data_buf[]; };
        layout(binding = 1) readonly buffer Weight { float weight_data[]; };

        layout(push_constant) uniform Params {
            uint head_dim;
            uint num_heads;
            float eps;
            // 0 = weight shared across heads (Qwen3, len = head_dim).
            // head_dim = per-channel weight (OLMoE, len = num_heads*head_dim).
            uint weight_stride;
            uint num_tokens;
        };

        shared float sdata[256];

        void main() {
            uint tid  = gl_LocalInvocationID.x;
            uint head = gl_WorkGroupID.x;
            uint row  = gl_WorkGroupID.y;
            if (head >= num_heads || row >= num_tokens) return;

            uint row_off  = row * num_heads * head_dim;
            uint base_off = row_off + head * head_dim;
            uint w_off    = head * weight_stride;

            // Phase 1: accumulate sum of squares for this token's head.
            float sum = 0.0;
            for (uint i = tid; i < head_dim; i += 256) {
                float v = data_buf[base_off + i];
                sum += v * v;
            }
            sdata[tid] = sum;
            barrier();

            // Phase 2: parallel reduction.
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                barrier();
            }

            // Phase 3: normalize in-place with weight.
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
    /// Batched interleaved-pair RoPE: applies <see cref="RoPE"/> to each of <c>num_tokens</c>
    /// independent rows of a <c>[num_tokens][num_heads*head_dim]</c> buffer in one dispatch, where
    /// row r uses position = <c>base_pos + r</c> (per-token absolute position). Pair index in
    /// <c>gl_GlobalInvocationID.x</c>, token row r = <c>gl_WorkGroupID.y</c> (dispatch
    /// ceil(total_pairs/256) × num_tokens groups). Each row computes its own cos/sin from
    /// base_pos+r, so it is bit-identical to <c>num_tokens</c> separate <see cref="RoPE"/> calls
    /// with positions base_pos, base_pos+1, ….
    /// Push constants: { uint num_heads, uint head_dim, int base_pos, float theta }.
    /// Bindings: 0=x ([num_tokens][num_heads*head_dim], in/out).
    /// (num_tokens comes from the dispatched Y group count; no separate push-constant needed.)
    /// </summary>
    internal const string RoPEBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            int base_pos;
            float theta;
        };

        void main() {
            uint pair_idx = gl_GlobalInvocationID.x;
            uint row      = gl_WorkGroupID.y;
            uint half_dim = head_dim / 2;
            uint total_pairs = num_heads * half_dim;
            if (pair_idx >= total_pairs) return;

            uint h = pair_idx / half_dim;
            uint i = pair_idx % half_dim;

            int position = base_pos + int(row);
            float freq = 1.0 / pow(theta, 2.0 * float(i) / float(head_dim));
            float angle = float(position) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint row_off = row * num_heads * head_dim;
            uint base_idx = row_off + h * head_dim + 2 * i;
            float x0 = x_data[base_idx];
            float x1 = x_data[base_idx + 1];
            x_data[base_idx]     = x0 * cos_a - x1 * sin_a;
            x_data[base_idx + 1] = x0 * sin_a + x1 * cos_a;
        }
        """;

    /// <summary>
    /// Batched NEOX/half-rotation RoPE: the <see cref="RoPENeox"/> sibling of
    /// <see cref="RoPEBatched"/>. Applies NEOX RoPE to each of <c>num_tokens</c> rows of a
    /// <c>[num_tokens][num_heads*head_dim]</c> buffer in one dispatch; row r uses position
    /// = <c>base_pos + r</c>. Bit-identical to <c>num_tokens</c> separate <see cref="RoPENeox"/>
    /// calls. Push constants: { uint num_heads, uint head_dim, int base_pos, float theta }.
    /// Bindings: 0=x (in/out). Pair index = <c>gl_GlobalInvocationID.x</c>, token row =
    /// <c>gl_WorkGroupID.y</c>.
    /// </summary>
    internal const string RoPENeoxBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            int base_pos;
            float theta;
        };

        void main() {
            uint pair_idx = gl_GlobalInvocationID.x;
            uint row      = gl_WorkGroupID.y;
            uint half_dim = head_dim / 2;
            uint total_pairs = num_heads * half_dim;
            if (pair_idx >= total_pairs) return;

            uint h = pair_idx / half_dim;
            uint i = pair_idx % half_dim;

            int position = base_pos + int(row);
            float freq = 1.0 / pow(theta, 2.0 * float(i) / float(head_dim));
            float angle = float(position) * freq;
            float cos_a = cos(angle);
            float sin_a = sin(angle);

            uint row_off = row * num_heads * head_dim;
            uint head_base = row_off + h * head_dim;
            uint a_idx = head_base + i;
            uint b_idx = head_base + i + half_dim;
            float x0 = x_data[a_idx];
            float x1 = x_data[b_idx];
            x_data[a_idx] = x0 * cos_a - x1 * sin_a;
            x_data[b_idx] = x0 * sin_a + x1 * cos_a;
        }
        """;

    /// <summary>
    /// RoPE NEOX with per-half-dim freq_factors (Gemma 4 global / non-SWA layers). Identical to
    /// <see cref="RoPENeox"/> except each pair's frequency is divided by <c>freq_factors[i]</c>
    /// (binding 1, size head_dim/2), masking the high-frequency tail to ~identity for long
    /// context. Mirrors the CUDA <c>llm_rope_neox_with_factors</c> kernel and the CPU
    /// <c>SimdKernels.BuildRopeTable(..., globalFreqFactors)</c> path. llama.cpp gemma4.cpp:191
    /// applies this only to non-SWA layers; SWA layers use plain <see cref="RoPENeox"/>.
    /// Push constants: { uint num_heads, uint head_dim, int position, float theta }.
    /// </summary>
    internal const string RoPENeoxWithFactors = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) buffer X { float x_data[]; };
        layout(binding = 1) readonly buffer FreqFactors { float freq_factors[]; };

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
            freq /= freq_factors[i];
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
    /// Embedding lookup from a Q6_K quantized table: dequantize one row to F32 output.
    /// Mirrors the CUDA <c>llm_embed_lookup_q6k</c> kernel (and thus <see cref="MatVecQ6K"/>
    /// and the CPU <c>DequantQ6K</c>) — keeps a large Q6_K tied embedding (e.g. Gemma 4 12B,
    /// [3840, 262144] ≈ 787 MiB raw) off the F32 dequant path that would burn ~4 GB of VRAM.
    ///
    /// 256 threads cooperate: each processes one block (256 elements) sequentially, thread
    /// <c>tid</c> emitting element <c>tid</c> of each 256-element super-block.
    /// Q6_K block (210 bytes per 256 elements):
    ///   [0:128]   ql — lower 4 bits
    ///   [128:192] qh — upper 2 bits
    ///   [192:208] 16 int8 scales
    ///   [208:210] FP16 d (super-block scale)
    ///
    /// Push constants: { uint token_id, uint emb_dim }.
    /// Bindings: 0=quantized_table (uint8 via uint32[]), 1=output[emb_dim].
    /// Dispatch: 1 workgroup.
    /// </summary>
    internal const string EmbedLookupQ6K = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer EmbTable { uint emb_data[]; };
        layout(binding = 1) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint token_id;
            uint emb_dim;
        };

        // Read directly from global memory with absolute byte offsets (no shared
        // memory): Q6_K blocks are 210 bytes, so a block's start is not necessarily
        // 4-byte aligned, which would break a uint32-indexed shared-memory copy.
        // Same byte-addressing approach as MatVecQ6K's gByte.
        uint gByte(uint b) { return (emb_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }
        int  gInt8(uint b) { int v = int(gByte(b)); return v >= 128 ? v - 256 : v; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint num_blocks = emb_dim >> 8; // emb_dim / 256

            // Byte offset to the start of this token's row (210 bytes/block).
            uint bytes_per_row = num_blocks * 210;
            uint row_byte_base = token_id * bytes_per_row;

            uint lane = tid & 31u;          // 0..31
            uint g    = tid >> 5;           // group 0..7
            uint isc  = lane >> 4;          // 0 or 1 (scale half)

            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = row_byte_base + block * 210;

                float d = unpackHalf2x16(gByte(b0 + 208) | (gByte(b0 + 209) << 8)).x;
                float scale = d * float(gInt8(b0 + 192 + 2u * g + isc));

                // ql byte: groups {0,2}->ql0, {1,3}->ql1, {4,6}->ql2, {5,7}->ql3 (+lane).
                uint ql_index = (g < 4u) ? (g & 1u) : (2u + (g & 1u));
                uint ql_byte  = gByte(b0 + ql_index * 32u + lane);
                uint high     = (g >> 1) & 1u;  // groups 2,3,6,7 use the high nibble
                uint nib      = (high != 0u) ? (ql_byte >> 4) : (ql_byte & 0xFu);

                // qh: groups 0-3 from qh0 (offset 128), 4-7 from qh1 (160); 2-bit field per group.
                uint qh_byte = (g < 4u) ? gByte(b0 + 128 + lane) : gByte(b0 + 160 + lane);
                uint shift   = 2u * (g & 3u);
                int q = int(nib | (((qh_byte >> shift) & 3u) << 4)) - 32;

                output_data[block * 256 + tid] = scale * float(q);
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
    /// Batched fp32 <see cref="KvAppend"/> (issue #308): appends K rows of K/V into the cache in ONE
    /// dispatch. 2D grid <c>(ceil(kv_dim/256), K)</c>: column = <c>gl_GlobalInvocationID.x</c> (guarded
    /// against <c>kv_dim</c>), token row = <c>gl_WorkGroupID.y</c>. Row r is written at cache slot
    /// <c>base_pos + r</c>, reading input row r at <c>r * kv_dim</c>. Bit-identical to K separate
    /// <see cref="KvAppend"/> calls at positions base_pos, base_pos+1, … (same element addressing,
    /// no ring modulo). Push constants reuse the <see cref="KvAppend"/> layout (<c>position</c>
    /// carries base_pos). Bindings: 0=k_input[K*kv_dim], 1=v_input[K*kv_dim], 2=k_cache, 3=v_cache.
    /// </summary>
    internal const string KvAppendBatched = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { float k_cache[]; };
        layout(binding = 3) buffer VCache { float v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;     // base_pos; row r writes slot base_pos + r
            uint max_seq_len;
        };

        void main() {
            uint col = gl_GlobalInvocationID.x;
            uint row = gl_WorkGroupID.y;
            if (col >= kv_dim) return;
            // Same element address as the single KvAppend: (position + row) * kv_dim + col.
            uint cache_off = (position + row) * kv_dim + col;
            uint in_off    = row * kv_dim + col;
            k_cache[cache_off] = k_input[in_off];
            v_cache[cache_off] = v_input[in_off];
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
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
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

            // Sliding-window bound (Gemma SWA layers): mirror the CPU ForwardPass.Attention
            // start_seq = window > 0 ? max(0, seq_len - window) : 0. Computed with the uint
            // underflow guard (window < seq_len) so window==0 OR window>=seq_len ⇒ full attention.
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;

            bool use_shared = (seq_len <= MAX_SHARED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1: per-position Q·K scores over [start_seq, seq_len) ───
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache[k_off + d];
                float score = dot * scale;
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }
            // Pad the shared tail so the max scan ignores stale slots. The masked-off head
            // ([0, start_seq)) is never read because every scan below starts at start_seq.
            if (use_shared) {
                for (uint t = seq_len + tid; t < MAX_SHARED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }
            barrier();

            // ─── Phase 2: in-place softmax over [start_seq, seq_len) ───
            float local_max = -1.0/0.0;
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
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
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
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

            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                if (use_shared) scores[t] *= inv_sum;
                else            scores_scratch[scratch_base + t] *= inv_sum;
            }
            barrier();

            // ─── Phase 3: weighted V sum over [start_seq, seq_len). K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                float sum = 0.0;
                for (uint t = start_seq; t < seq_len; t++) {
                    float weight = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum += weight * v_cache[v_off + d];
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// Batched fp32 attention (issue #308): the spec-decode batched-verify crux. Runs K queries in
    /// ONE dispatch over a 2D grid <c>num_heads × num_queries</c> workgroups, where query qi (at
    /// absolute position <c>base_pos + qi</c>) attends causally over <c>[0, base_pos + qi]</c> — i.e.
    /// query qi's <c>seq_len_i = base_pos + qi + 1</c>. This reproduces the causal-among-K behavior of
    /// K separate single-query <see cref="Attention"/> calls at seqLens base_pos+1 … base_pos+K
    /// WITHOUT the per-token gather/scatter.
    ///
    /// CRITICAL — bit-exactness: each <c>(h, qi)</c> workgroup is an INDEPENDENT copy of the
    /// single-query <see cref="Attention"/> ≤4096 shared-memory fast path with <c>seq_len = seq_len_i</c>
    /// and <c>window = 0</c> (no SWA — spec verify never windows). Score iteration order, the
    /// <c>sdata[256]</c> tree reduce, the <c>exp</c>/<c>inv_sum</c> softmax, and the Phase-3 V-sum
    /// order are kept VERBATIM, so the result is bit-identical to the single-query shader. The shared
    /// <c>scores[]</c> tail is padded with -inf up to <c>seq_len_i</c> (per-row bound, NOT a fixed
    /// seqLen) so the max scan ignores stale slots. There is no split-KV / scratch fallback here:
    /// the caller restricts the batched attention to <c>base_pos + K ≤ 4096</c>.
    ///
    /// Q is read from <c>q_data</c> at <c>qi*(num_heads*head_dim) + h*head_dim</c> and output written
    /// to <c>out_data</c> at the same offset (no gather/scatter). K/V are read from the cache exactly
    /// like the single-query shader (<c>t*kv_dim + kv_head*head_dim + d</c>, GQA
    /// <c>kv_head = h/(num_heads/num_kv_heads)</c>, scale <c>inversesqrt(head_dim)</c>).
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, base_pos, max_seq_len, num_queries }.
    /// Bindings: 0=q_data[K*num_heads*head_dim], 1=K_cache, 2=V_cache, 3=out_data[K*num_heads*head_dim].
    /// </summary>
    internal const string AttentionBatched = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { float k_cache[]; };
        layout(binding = 2) readonly buffer VCache { float v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint base_pos;      // query qi is at absolute position base_pos + qi
            uint max_seq_len;
            uint num_queries;   // K
        };

        // seq_len_i = base_pos + qi + 1 ≤ base_pos + K, and the caller guarantees base_pos + K ≤ 4096,
        // so the whole causal range always fits in shared memory — no scratch-spill path here.
        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        void main() {
            uint h   = gl_WorkGroupID.x;
            uint qi  = gl_WorkGroupID.y;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads || qi >= num_queries) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint row_stride = num_heads * head_dim;
            uint q_off   = qi * row_stride + h * head_dim;
            uint out_off = qi * row_stride + h * head_dim;

            // Per-query causal length: query qi (abs pos base_pos+qi) attends [0, base_pos+qi].
            // window = 0 (no SWA), start_seq = 0.
            uint seq_len_i = base_pos + qi + 1u;

            // ─── Phase 1: per-position Q·K scores over [0, seq_len_i) ───
            for (uint t = tid; t < seq_len_i; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache[k_off + d];
                scores[t] = dot * scale;
            }
            // No tail padding needed: every later phase (max scan, exp/sum, V-aggregate) is
            // strictly bounded by seq_len_i, so scores[t >= seq_len_i] is never read.
            barrier();

            // ─── Phase 2: in-place softmax over [0, seq_len_i) ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < seq_len_i; t += 256)
                local_max = max(local_max, scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            float local_sum = 0.0;
            for (uint t = tid; t < seq_len_i; t += 256) {
                float e = exp(scores[t] - max_val);
                scores[t] = e;
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

            for (uint t = tid; t < seq_len_i; t += 256)
                scores[t] *= inv_sum;
            barrier();

            // ─── Phase 3: weighted V sum over [0, seq_len_i). ───
            for (uint d = tid; d < head_dim; d += 256) {
                float sum = 0.0;
                for (uint t = 0; t < seq_len_i; t++) {
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum += scores[t] * v_cache[v_off + d];
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// bf16 (issue #308 follow-up / #332) variant of <see cref="AttentionBatched"/>: control flow,
    /// per-query causal range (<c>seq_len_i = base_pos + qi + 1</c>), no-tail-pad, and 2D grid
    /// (<c>num_heads × num_queries</c>) are IDENTICAL to the fp32 <see cref="AttentionBatched"/>; the
    /// only difference is the K/V cache buffers (bindings 1, 2) are <c>uint[]</c> holding IEEE fp16
    /// packed two-per-uint (<c>unpackHalf2x16</c> on read), using the SAME read idiom as the
    /// single-query <see cref="AttentionBf16"/>. All scores / softmax / value accumulation stay fp32,
    /// so each <c>(h, qi)</c> workgroup is bit-identical to a single-query <see cref="AttentionBf16"/>
    /// call at <c>seq_len = base_pos + qi + 1</c>. No scratch-spill (caller restricts base_pos+K ≤ 4096).
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, base_pos, max_seq_len, num_queries }.
    /// Bindings: 0=q_data (float), 1=K_cache (uint, packed fp16×2), 2=V_cache (uint, packed fp16×2),
    ///           3=out_data (float).
    /// </summary>
    internal const string AttentionBatchedBf16 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache { uint v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint base_pos;      // query qi is at absolute position base_pos + qi
            uint max_seq_len;
            uint num_queries;   // K
        };

        // seq_len_i = base_pos + qi + 1 ≤ base_pos + K ≤ 4096 ⇒ the whole causal range fits in
        // shared memory; no scratch-spill path (matches the fp32 AttentionBatched).
        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        void main() {
            uint h   = gl_WorkGroupID.x;
            uint qi  = gl_WorkGroupID.y;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads || qi >= num_queries) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint row_stride = num_heads * head_dim;
            uint q_off   = qi * row_stride + h * head_dim;
            uint out_off = qi * row_stride + h * head_dim;

            // Per-query causal length: query qi (abs pos base_pos+qi) attends [0, base_pos+qi].
            uint seq_len_i = base_pos + qi + 1u;

            // ─── Phase 1: per-position Q·K scores over [0, seq_len_i) ───
            for (uint t = tid; t < seq_len_i; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                // Read each packed fp16 word once (two K elements at a time) — same idiom as the
                // single-query AttentionBf16. k_off is even (head_dim even, see GpuForwardPass guard).
                uint k_off_half = k_off >> 1;
                for (uint dh = 0; dh < (head_dim >> 1); dh++) {
                    uint d = dh << 1;
                    vec2 kv = unpackHalf2x16(k_cache[k_off_half + dh]);
                    dot += q_data[q_off + d] * kv.x + q_data[q_off + d + 1u] * kv.y;
                }
                scores[t] = dot * scale;
            }
            // No tail padding needed: every later phase is strictly bounded by seq_len_i.
            barrier();

            // ─── Phase 2: in-place softmax over [0, seq_len_i) ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < seq_len_i; t += 256)
                local_max = max(local_max, scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            float local_sum = 0.0;
            for (uint t = tid; t < seq_len_i; t += 256) {
                float e = exp(scores[t] - max_val);
                scores[t] = e;
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

            for (uint t = tid; t < seq_len_i; t += 256)
                scores[t] *= inv_sum;
            barrier();

            // ─── Phase 3: weighted V sum over [0, seq_len_i). K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                // Each thread owns ONE output dim d (threads 256 apart, so adjacent d can't be
                // paired). Hoist the per-d word/component selection out of the t-loop and walk the
                // V row word base incrementally — same idiom as the single-query AttentionBf16.
                uint d_half = d >> 1;
                uint component = d & 1u;
                uint v_off_half = (kv_head * head_dim) >> 1;   // t = 0 row word base
                uint kv_dim_half = kv_dim >> 1;
                float sum = 0.0;
                for (uint t = 0; t < seq_len_i; t++) {
                    float vv = unpackHalf2x16(v_cache[v_off_half + d_half])[component];
                    sum += scores[t] * vv;
                    v_off_half += kv_dim_half;
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// q8_0 (issue #308 follow-up / #332) variant of <see cref="AttentionBatched"/>: control flow,
    /// per-query causal range (<c>seq_len_i = base_pos + qi + 1</c>), no-tail-pad, and 2D grid
    /// (<c>num_heads × num_queries</c>) are IDENTICAL to the fp32 <see cref="AttentionBatched"/>; the
    /// only difference is the K/V cache buffers (bindings 1, 2) are <c>uint[]</c> holding ggml
    /// <c>block_q8_0</c> (34 bytes/block: fp16 scale + 32 int8), read via the SAME byte-gather +
    /// dequant idiom as the single-query <see cref="AttentionQ8_0"/>. All scores / softmax / value
    /// accumulation stay fp32, so each <c>(h, qi)</c> workgroup is bit-identical to a single-query
    /// <see cref="AttentionQ8_0"/> call at <c>seq_len = base_pos + qi + 1</c>. No scratch-spill (caller
    /// restricts base_pos+K ≤ 4096). kv_dim%32==0, so blocks never straddle a KV row.
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, base_pos, max_seq_len, num_queries }.
    /// Bindings: 0=q_data (float), 1=K_cache (uint, block_q8_0), 2=V_cache (uint, block_q8_0),
    ///           3=out_data (float).
    /// </summary>
    internal const string AttentionBatchedQ8_0 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache { uint v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint base_pos;      // query qi is at absolute position base_pos + qi
            uint max_seq_len;
            uint num_queries;   // K
        };

        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        // Sign-extend a single int8 byte in one bitfieldExtract (no ternary branch) — same as
        // the single-query AttentionQ8_0.
        int gInt8K(uint b) { return bitfieldExtract(int(k_cache[b >> 2]), int((b & 3u) * 8u), 8); }
        int gInt8V(uint b) { return bitfieldExtract(int(v_cache[b >> 2]), int((b & 3u) * 8u), 8); }

        float loadK(uint e) {
            uint blk = e >> 5; uint lane = e & 31u; uint b0 = blk * 34u;
            uint w = k_cache[b0 >> 2];
            float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
            return dsc * float(gInt8K(b0 + 2u + lane));
        }
        float loadV(uint e) {
            uint blk = e >> 5; uint lane = e & 31u; uint b0 = blk * 34u;
            uint w = v_cache[b0 >> 2];
            float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
            return dsc * float(gInt8V(b0 + 2u + lane));
        }

        void main() {
            uint h   = gl_WorkGroupID.x;
            uint qi  = gl_WorkGroupID.y;
            uint tid = gl_LocalInvocationID.x;
            if (h >= num_heads || qi >= num_queries) return;

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim = num_kv_heads * head_dim;
            float scale = inversesqrt(float(head_dim));
            uint row_stride = num_heads * head_dim;
            uint q_off   = qi * row_stride + h * head_dim;
            uint out_off = qi * row_stride + h * head_dim;

            uint seq_len_i = base_pos + qi + 1u;

            // ─── Phase 1: per-position Q·K scores over [0, seq_len_i) ───
            for (uint t = tid; t < seq_len_i; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * loadK(k_off + d);
                scores[t] = dot * scale;
            }
            barrier();

            // ─── Phase 2: in-place softmax over [0, seq_len_i) ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < seq_len_i; t += 256)
                local_max = max(local_max, scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint s = 128; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
                barrier();
            }
            float max_val = sdata[0];
            barrier();

            float local_sum = 0.0;
            for (uint t = tid; t < seq_len_i; t += 256) {
                float e = exp(scores[t] - max_val);
                scores[t] = e;
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

            for (uint t = tid; t < seq_len_i; t += 256)
                scores[t] *= inv_sum;
            barrier();

            // ─── Phase 3: weighted V sum over [0, seq_len_i). K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                float sum = 0.0;
                for (uint t = 0; t < seq_len_i; t++) {
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum += scores[t] * loadV(v_off + d);
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// bf16 (issue #308 follow-up / #332) variant of <see cref="KvAppendBatched"/>: appends K rows of
    /// K/V from the packed <c>[K][kvDim]</c> inputs into the cache in ONE dispatch (row r at slot
    /// base_pos + r), storing IEEE fp16 packed two-per-uint (<c>packHalf2x16</c>) — the SAME write
    /// idiom as the single-token <see cref="KvAppendBf16"/>. 2D grid (<c>ceil((kvDim/2)/256), K</c>),
    /// one thread per 2 elements (kv_dim even). Bit-identical to K separate <see cref="KvAppendBf16"/>
    /// calls. Indexes the cache identically to fp32 (<c>(base_pos + row) * kv_dim + i</c>, word-granular).
    ///
    /// Push constants: { uint kv_dim, position (base_pos), max_seq_len }.
    /// Bindings: 0=k_input[K*kv_dim] (float), 1=v_input[K*kv_dim] (float),
    ///           2=k_cache (uint, packed fp16×2), 3=v_cache (uint, packed fp16×2).
    /// </summary>
    internal const string KvAppendBatchedBf16 = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { uint k_cache[]; };
        layout(binding = 3) buffer VCache { uint v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;     // base_pos; row r writes slot base_pos + r
            uint max_seq_len;
        };

        void main() {
            uint w = gl_GlobalInvocationID.x;
            uint row = gl_WorkGroupID.y;
            uint half_dim = kv_dim >> 1;   // kv_dim is even (numKvHeads*headDim)
            if (w >= half_dim) return;
            uint i = w << 1;
            // Same element address as fp32 ((position + row) * kv_dim + i), expressed in words.
            uint row_word = (position + row) * half_dim;
            uint in_elem  = row * kv_dim + i;   // first of the 2 source float elements (element-granular)
            k_cache[row_word + w] = packHalf2x16(vec2(k_input[in_elem], k_input[in_elem + 1u]));
            v_cache[row_word + w] = packHalf2x16(vec2(v_input[in_elem], v_input[in_elem + 1u]));
        }
        """;

    /// <summary>
    /// q8_0 (issue #308 follow-up / #332) variant of <see cref="KvAppendBatched"/>: appends K rows of
    /// K/V from the packed <c>[K][kvDim]</c> inputs into the cache in ONE dispatch (row r at slot
    /// base_pos + r), block-quantizing into ggml <c>block_q8_0</c> (34 bytes/block) with the SAME
    /// amax→quant + masked-atomic-byte-store idiom as the single-token <see cref="KvAppendQ8_0"/>. 2D
    /// grid (<c>ceil((kvDim/32)/256), K</c>), one thread per 32-element block. Bit-identical to K
    /// separate <see cref="KvAppendQ8_0"/> calls: every thread (across ALL blocks AND rows) owns a
    /// DISJOINT set of destination bytes; the only sharing is at seam uint words (between adjacent
    /// blocks within a row and, when blocks_per_row is odd, between the last block of one row and the
    /// first of the next), which the masked atomicAnd+atomicOr byte writer makes correct under any
    /// interleaving. So the result is independent of dispatch order. Indexes the cache identically to fp32
    /// (<c>(base_pos + row) * kv_dim + i</c>, expressed in blocks). kv_dim%32==0.
    ///
    /// Push constants: { uint kv_dim, position (base_pos), max_seq_len }.
    /// Bindings: 0=k_input[K*kv_dim] (float), 1=v_input[K*kv_dim] (float),
    ///           2=k_cache (uint, block_q8_0), 3=v_cache (uint, block_q8_0).
    /// </summary>
    internal const string KvAppendBatchedQ8_0 = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { uint k_cache[]; };
        layout(binding = 3) buffer VCache { uint v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;     // base_pos; row r writes slot base_pos + r
            uint max_seq_len;
        };

        // Masked-atomic byte writers: clear the target byte, then OR in the value. Disjoint bytes
        // within a shared uint stay correct under any interleaving — same as KvAppendQ8_0.
        void sByteK(uint b, uint val) {
            uint w = b >> 2; uint sh = (b & 3u) * 8u;
            atomicAnd(k_cache[w], ~(0xFFu << sh));
            atomicOr (k_cache[w], (val & 0xFFu) << sh);
        }
        void sByteV(uint b, uint val) {
            uint w = b >> 2; uint sh = (b & 3u) * 8u;
            atomicAnd(v_cache[w], ~(0xFFu << sh));
            atomicOr (v_cache[w], (val & 0xFFu) << sh);
        }

        void main() {
            uint blk = gl_GlobalInvocationID.x;
            uint row = gl_WorkGroupID.y;
            uint blocks_per_row = kv_dim >> 5;   // kv_dim % 32 == 0
            if (blk >= blocks_per_row) return;

            // Same element address as fp32 ((position + row) * kv_dim + i), expressed in blocks.
            uint dst_block = (position + row) * blocks_per_row + blk;
            uint b0 = dst_block * 34u;
            uint src = row * kv_dim + (blk << 5);   // first source element of this block

            // ── K block ──
            {
                float amax = 0.0;
                for (uint j = 0u; j < 32u; j++)
                    amax = max(amax, abs(k_input[src + j]));
                float d = amax / 127.0;
                float invd = (d < 1e-30) ? 0.0 : (1.0 / d);
                uint dh = packHalf2x16(vec2(d, 0.0)) & 0xFFFFu;
                sByteK(b0, dh & 0xFFu);
                sByteK(b0 + 1u, dh >> 8);
                for (uint j = 0u; j < 32u; j++) {
                    int q = clamp(int(round(k_input[src + j] * invd)), -127, 127);
                    sByteK(b0 + 2u + j, uint(q & 0xFF));
                }
            }
            // ── V block ──
            {
                float amax = 0.0;
                for (uint j = 0u; j < 32u; j++)
                    amax = max(amax, abs(v_input[src + j]));
                float d = amax / 127.0;
                float invd = (d < 1e-30) ? 0.0 : (1.0 / d);
                uint dh = packHalf2x16(vec2(d, 0.0)) & 0xFFFFu;
                sByteV(b0, dh & 0xFFu);
                sByteV(b0 + 1u, dh >> 8);
                for (uint j = 0u; j < 32u; j++) {
                    int q = clamp(int(round(v_input[src + j] * invd)), -127, 127);
                    sByteV(b0 + 2u + j, uint(q & 0xFF));
                }
            }
        }
        """;

    /// <summary>
    /// bf16 (issue #311) variant of <see cref="KvAppend"/>: the K/V cache buffers store
    /// IEEE fp16 packed two-per-uint via core-GLSL <c>packHalf2x16</c> (no device extension).
    /// The user-facing <c>--kv-type bf16</c> means "half-width KV"; Vulkan stores fp16
    /// because for the small-magnitude KV values fp16 is more precise than bf16. Arithmetic
    /// elsewhere stays fp32 — only the stored value is narrowed.
    ///
    /// CRITICAL: this indexes the cache IDENTICALLY to the fp32 <see cref="KvAppend"/>
    /// (<c>position * kv_dim + i</c> element addressing, just expressed in words because
    /// each word holds 2 elements). There is NO <c>% max_seq_len</c> ring modulo, matching
    /// the fp32 shader exactly. kv_dim is always even (numKvHeads*headDim), so we dispatch
    /// one thread per 2 elements (word granular).
    ///
    /// Push constants: { uint kv_dim, uint position, uint max_seq_len } — unchanged.
    /// Bindings: 0=k_input[kv_dim] (float), 1=v_input[kv_dim] (float),
    ///           2=k_cache (uint, packed fp16×2), 3=v_cache (uint, packed fp16×2).
    /// </summary>
    internal const string KvAppendBf16 = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { uint k_cache[]; };
        layout(binding = 3) buffer VCache { uint v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;
            uint max_seq_len;
        };

        void main() {
            uint w = gl_GlobalInvocationID.x;
            uint half_dim = kv_dim >> 1;   // kv_dim is even (numKvHeads*headDim)
            if (w >= half_dim) return;
            uint i = w << 1;
            // Same element address as fp32 (position * kv_dim + i), expressed in words.
            uint row_word = position * half_dim;
            k_cache[row_word + w] = packHalf2x16(vec2(k_input[i], k_input[i + 1u]));
            v_cache[row_word + w] = packHalf2x16(vec2(v_input[i], v_input[i + 1u]));
        }
        """;

    /// <summary>
    /// bf16 (issue #311) variant of <see cref="Attention"/>: control flow is IDENTICAL to
    /// the fp32 shader; the only difference is that the K/V cache buffers (bindings 1, 2)
    /// are <c>uint[]</c> holding IEEE fp16 packed two-per-uint (<c>packHalf2x16</c> on
    /// write, <c>unpackHalf2x16</c> on read), so every element read becomes an unpack +
    /// lane-select. All scores / softmax / value accumulation stay fp32 — the arithmetic is
    /// bit-identical to the fp32 Attention; only the stored K/V mantissa is narrowed.
    /// scores_scratch (binding 4) stays fp32. The <c>inversesqrt(head_dim)</c> scale is kept
    /// exactly as the fp32 shader has it.
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, seq_len, max_seq_len } — unchanged.
    /// Bindings: 0=Q (float), 1=K_cache (uint, packed fp16×2), 2=V_cache (uint, packed fp16×2),
    ///           3=output (float), 4=scores_scratch (float).
    /// </summary>
    internal const string AttentionBf16 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache { uint v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };
        layout(binding = 4) buffer ScoresScratch   { float scores_scratch[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint max_seq_len;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        // Score-storage strategy mirrors the fp32 Attention shader.
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

            // SWA bound — mirrors the fp32 Attention shader (CPU ForwardPass.Attention).
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;

            bool use_shared = (seq_len <= MAX_SHARED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1: per-position Q·K scores over [start_seq, seq_len) ───
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                // Read each packed fp16 word once (two K elements at a time). k_off is even
                // (head_dim is even — see the GpuForwardPass guard) so k_off>>1 is the exact
                // word base and consecutive d,d+1 are the two halves of word k_off_half+dh.
                uint k_off_half = k_off >> 1;
                for (uint dh = 0; dh < (head_dim >> 1); dh++) {
                    uint d = dh << 1;
                    vec2 kv = unpackHalf2x16(k_cache[k_off_half + dh]);
                    dot += q_data[q_off + d] * kv.x + q_data[q_off + d + 1u] * kv.y;
                }
                float score = dot * scale;
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }
            if (use_shared) {
                for (uint t = seq_len + tid; t < MAX_SHARED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }
            barrier();

            // ─── Phase 2: in-place softmax over [start_seq, seq_len) ───
            float local_max = -1.0/0.0;
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
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
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
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

            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                if (use_shared) scores[t] *= inv_sum;
                else            scores_scratch[scratch_base + t] *= inv_sum;
            }
            barrier();

            // ─── Phase 3: weighted V sum over [start_seq, seq_len). K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                // Each thread owns ONE output dim d (threads are 256 apart, so adjacent d can't
                // be paired). Hoist the per-d word/component selection out of the t-loop and walk
                // the V row word base incrementally. v_off = t*kv_dim + kv_head*head_dim is even
                // (head_dim is even — see the GpuForwardPass guard), so v_off>>1 is the exact word.
                uint d_half = d >> 1;
                uint component = d & 1u;
                uint v_off_half = ((start_seq * kv_dim) + kv_head * head_dim) >> 1;   // t = start_seq row word base
                uint kv_dim_half = kv_dim >> 1;
                float sum = 0.0;
                for (uint t = start_seq; t < seq_len; t++) {
                    float weight = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                    float vv = unpackHalf2x16(v_cache[v_off_half + d_half])[component];
                    sum += weight * vv;
                    v_off_half += kv_dim_half;
                }
                out_data[out_off + d] = sum;
            }
        }
        """;

    /// <summary>
    /// q8_0 (issue #325) variant of <see cref="KvAppend"/>: block-quantizes the K/V vectors
    /// into the cache as ggml <c>block_q8_0</c> (34 bytes = fp16 scale + 32 int8, per 32
    /// elements; ~4× smaller than fp32). Mirrors the CUDA <c>llm_kv_append_q8_0</c> /
    /// <c>sharpi_q8_append_one</c> ground truth: per 32-element block, <c>amax = max(|x|)</c>,
    /// <c>d = amax / 127</c>, <c>invd = (d &lt; 1e-30) ? 0 : 1/d</c> (the 1e-30 guard avoids
    /// 0*inf=NaN — replicated verbatim), <c>q = clamp(round(x*invd), -127, 127)</c>.
    ///
    /// Dispatched ONE THREAD PER 32-ELEMENT BLOCK (not a subgroup — subgroup width is
    /// hardware-dependent; one-thread-per-block sidesteps that). Each thread owns all 34
    /// bytes of its destination block. Because the cache is bound as <c>uint[]</c> and a
    /// 34-byte block is not 4-aligned, adjacent blocks share the seam <c>uint</c> word but
    /// write DISJOINT bytes; the masked <c>atomicAnd</c>+<c>atomicOr</c> byte writer makes the
    /// disjoint-bitfield RMW correct under any thread interleaving (the atomicAnd clears the
    /// byte first, so ring-reuse overwrites cleanly — no zero-init needed).
    ///
    /// CRITICAL: indexes the cache IDENTICALLY to the fp32 <see cref="KvAppend"/>
    /// (<c>position * kv_dim + i</c> element addressing, expressed in blocks). No
    /// <c>% max_seq_len</c> ring modulo, matching the fp32/bf16 shaders. kv_dim is always a
    /// multiple of 32 (enforced in GpuForwardPass), so a KV row's blocks never straddle a row.
    ///
    /// Push constants: { uint kv_dim, position, max_seq_len } — unchanged.
    /// Bindings: 0=k_input[kv_dim] (float), 1=v_input[kv_dim] (float),
    ///           2=k_cache (uint, packed block_q8_0), 3=v_cache (uint, packed block_q8_0).
    /// </summary>
    internal const string KvAppendQ8_0 = """
        #version 450
        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer KIn  { float k_input[]; };
        layout(binding = 1) readonly buffer VIn  { float v_input[]; };
        layout(binding = 2) buffer KCache { uint k_cache[]; };
        layout(binding = 3) buffer VCache { uint v_cache[]; };

        layout(push_constant) uniform Params {
            uint kv_dim;
            uint position;
            uint max_seq_len;
        };

        // Masked-atomic byte writers: clear the target byte, then OR in the value.
        // Disjoint bytes within a shared uint stay correct under any interleaving.
        void sByteK(uint b, uint val) {
            uint w = b >> 2; uint sh = (b & 3u) * 8u;
            atomicAnd(k_cache[w], ~(0xFFu << sh));
            atomicOr (k_cache[w], (val & 0xFFu) << sh);
        }
        void sByteV(uint b, uint val) {
            uint w = b >> 2; uint sh = (b & 3u) * 8u;
            atomicAnd(v_cache[w], ~(0xFFu << sh));
            atomicOr (v_cache[w], (val & 0xFFu) << sh);
        }

        void main() {
            uint blk = gl_GlobalInvocationID.x;
            uint blocks_per_row = kv_dim >> 5;   // kv_dim % 32 == 0
            if (blk >= blocks_per_row) return;

            // Same element address as fp32 (position * kv_dim + i), expressed in blocks.
            uint dst_block = position * blocks_per_row + blk;
            uint b0 = dst_block * 34u;
            uint src = blk << 5;                 // first source element of this block

            // ── K block ──
            {
                float amax = 0.0;
                for (uint j = 0u; j < 32u; j++)
                    amax = max(amax, abs(k_input[src + j]));
                float d = amax / 127.0;
                float invd = (d < 1e-30) ? 0.0 : (1.0 / d);
                uint dh = packHalf2x16(vec2(d, 0.0)) & 0xFFFFu;
                sByteK(b0, dh & 0xFFu);
                sByteK(b0 + 1u, dh >> 8);
                for (uint j = 0u; j < 32u; j++) {
                    int q = clamp(int(round(k_input[src + j] * invd)), -127, 127);
                    sByteK(b0 + 2u + j, uint(q & 0xFF));
                }
            }
            // ── V block ──
            {
                float amax = 0.0;
                for (uint j = 0u; j < 32u; j++)
                    amax = max(amax, abs(v_input[src + j]));
                float d = amax / 127.0;
                float invd = (d < 1e-30) ? 0.0 : (1.0 / d);
                uint dh = packHalf2x16(vec2(d, 0.0)) & 0xFFFFu;
                sByteV(b0, dh & 0xFFu);
                sByteV(b0 + 1u, dh >> 8);
                for (uint j = 0u; j < 32u; j++) {
                    int q = clamp(int(round(v_input[src + j] * invd)), -127, 127);
                    sByteV(b0 + 2u + j, uint(q & 0xFF));
                }
            }
        }
        """;

    /// <summary>
    /// q8_0 (issue #325) variant of <see cref="Attention"/>: control flow is IDENTICAL to the
    /// fp32 shader; the only difference is that the K/V cache buffers (bindings 1, 2) are
    /// <c>uint[]</c> holding ggml <c>block_q8_0</c> (34 bytes/block: fp16 scale + 32 int8), so
    /// every element read becomes a byte-gather + dequant <c>value = fp16(d) * int8</c>. All
    /// scores / softmax / value accumulation stay fp32 — only the stored K/V is narrowed.
    /// scores_scratch (binding 4) stays fp32. The <c>inversesqrt(head_dim)</c> scale is kept
    /// exactly as the fp32 shader has it. Element addressing (<c>off = t*kv_dim + kv_head*head_dim</c>,
    /// <c>e = off + d</c>) is identical to fp32/bf16; per element <c>blk=e&gt;&gt;5</c>,
    /// <c>lane=e&amp;31</c>, <c>b0=blk*34</c>. kv_dim%32==0, so blocks never straddle a KV row.
    ///
    /// Push constants: { uint num_heads, num_kv_heads, head_dim, seq_len, max_seq_len } — unchanged.
    /// Bindings: 0=Q (float), 1=K_cache (uint, block_q8_0), 2=V_cache (uint, block_q8_0),
    ///           3=output (float), 4=scores_scratch (float).
    /// </summary>
    internal const string AttentionQ8_0 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q     { float q_data[]; };
        layout(binding = 1) readonly buffer KCache { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache { uint v_cache[]; };
        layout(binding = 3) buffer Out             { float out_data[]; };
        layout(binding = 4) buffer ScoresScratch   { float scores_scratch[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint max_seq_len;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        // Score-storage strategy mirrors the fp32 Attention shader.
        const uint MAX_SHARED_SCORES = 4096u;
        shared float scores[MAX_SHARED_SCORES];
        shared float sdata[256];   // reduction scratch

        // Sign-extend a single int8 byte in one bitfieldExtract (no ternary branch).
        int gInt8K(uint b) { return bitfieldExtract(int(k_cache[b >> 2]), int((b & 3u) * 8u), 8); }
        int gInt8V(uint b) { return bitfieldExtract(int(v_cache[b >> 2]), int((b & 3u) * 8u), 8); }

        float loadK(uint e) {
            uint blk = e >> 5; uint lane = e & 31u; uint b0 = blk * 34u;
            // b0 = blk*34 is even, so the two scale bytes [b0, b0+1] live in the same uint word.
            uint w = k_cache[b0 >> 2];
            float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
            return dsc * float(gInt8K(b0 + 2u + lane));
        }
        float loadV(uint e) {
            uint blk = e >> 5; uint lane = e & 31u; uint b0 = blk * 34u;
            uint w = v_cache[b0 >> 2];
            float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
            return dsc * float(gInt8V(b0 + 2u + lane));
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

            // SWA bound — mirrors the fp32 Attention shader (CPU ForwardPass.Attention).
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;

            bool use_shared = (seq_len <= MAX_SHARED_SCORES);
            uint scratch_base = h * max_seq_len;

            // ─── Phase 1: per-position Q·K scores over [start_seq, seq_len) ───
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                float dot = 0.0;
                uint k_off = t * kv_dim + kv_head * head_dim;
                for (uint d = 0; d < head_dim; d++)
                    dot += q_data[q_off + d] * loadK(k_off + d);
                float score = dot * scale;
                if (use_shared) scores[t] = score;
                else            scores_scratch[scratch_base + t] = score;
            }
            if (use_shared) {
                for (uint t = seq_len + tid; t < MAX_SHARED_SCORES; t += 256)
                    scores[t] = -1.0/0.0;
            }
            barrier();

            // ─── Phase 2: in-place softmax over [start_seq, seq_len) ───
            float local_max = -1.0/0.0;
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
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
            for (uint t = start_seq + tid; t < seq_len; t += 256) {
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

            for (uint t = start_seq + tid; t < seq_len; t += 256) {
                if (use_shared) scores[t] *= inv_sum;
                else            scores_scratch[scratch_base + t] *= inv_sum;
            }
            barrier();

            // ─── Phase 3: weighted V sum over [start_seq, seq_len). K is NOT re-derived here. ───
            for (uint d = tid; d < head_dim; d += 256) {
                float sum = 0.0;
                for (uint t = start_seq; t < seq_len; t++) {
                    float weight = use_shared ? scores[t] : scores_scratch[scratch_base + t];
                    uint v_off = t * kv_dim + kv_head * head_dim;
                    sum += weight * loadV(v_off + d);
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

    /// <summary>
    /// Batched (weight-stationary) matrix-vector multiply with Q4_K dequantization —
    /// the core weight-amortization for Vulkan speculative decoding (issue #308).
    ///
    /// Computes <c>nTok</c> independent matvecs against the SAME Q4_K weight matrix. The
    /// expensive part (reading + unpacking each weight nibble from VRAM) is done ONCE per
    /// output element and then multiplied into <c>nTok</c> accumulators (one per input
    /// vector), so the weight is read from VRAM once for all K tokens instead of K times.
    /// Only the per-token input reads are repeated.
    ///
    /// Bindings: 0=quantized weights (uint8), 1=inputs (float, row-major [nTok][cols]),
    /// 2=outputs (float, row-major [nTok][rows]). Push constants: { uint rows, uint cols, uint nTok }.
    ///
    /// BIT-EXACT vs nTok separate single-row <see cref="MatVecQ4K"/> calls: the element
    /// iteration order, the per-element dequant, and the subgroupAdd reduction are IDENTICAL
    /// to the single-row shader — only the k (token) dimension is added on top. The same
    /// floating-point accumulation order is therefore preserved per (row, token).
    ///
    /// local_size_x = 256 (8 rows × 32 lanes) so #318 pins the subgroup size to 32, which the
    /// subgroupAdd reduction requires (THREADS_PER_ROW == 32).
    ///
    /// nTok is capped at 8 (the acc[] register array size; matches the spec-decode draft cap).
    /// </summary>
    internal const string MatVecBatchedQ4K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable
        #extension GL_KHR_shader_subgroup_arithmetic : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        // Register-based scale precomputation. subgroupAdd for reduction.
        #define N_ROWS 8
        #define THREADS_PER_ROW 32
        #define MAX_NTOK 8

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
            uint nTok;
        };

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;
            uint word_row_base = row * num_blocks * 36;

            float acc[MAX_NTOK];
            [[unroll]] for (uint k = 0; k < MAX_NTOK; k++) acc[k] = 0.0;

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
                    // Dequantized weight read+unpacked ONCE here; shared across all nTok inputs.
                    float w = dsc[si] * float(nibble) - dmn[si];
                    uint in_idx = block * 256 + elem_idx;
                    for (uint k = 0; k < nTok; k++)
                        acc[k] += w * input_data[k * cols + in_idx];
                }
            }

            for (uint k = 0; k < nTok; k++) {
                float r = subgroupAdd(acc[k]);
                if (subgroupElect())
                    output_data[k * rows + row] = r;
            }
        }
        """;

    /// <summary>
    /// Quantize FP32 activations → Q8_1 (per 32-element sub-block), int8 path for the DP4A
    /// batched Q4_K matvec (issue #308 P0/P1). Mirrors CUDA's <c>llm_quantize_q8_1</c> exactly:
    /// per 32-element sub-block compute <c>amax = max|x|</c> over the 32 lanes, <c>d = amax/127</c>,
    /// <c>q = clamp(round(x/d), -127, 127)</c> (int8), and <c>qsum = Σq</c>. Each 32-element
    /// sub-block emits ONE 36-byte Q8_1 block:
    ///   bytes [0:2]  = fp16 d
    ///   bytes [2:4]  = fp16 (d · qsum)   (the min-bias scale `s` — only the Q4_K MMQ reads it)
    ///   bytes [4:36] = 32 × int8 quants
    /// Input is row-major <c>[nTok][cols]</c> FP32; output is row-major
    /// <c>[nTok][cols/32 × 36 bytes]</c> (one block per 32 input elements). 36 % 4 == 0, so each
    /// 36-byte block is exactly 9 word-aligned, mutually disjoint uints — the header is word 0
    /// ({d, s}) and the 32 int8 quants fill words 1..8. The output binds as a <c>uint[]</c> SSBO
    /// and every word is written PLAINLY (no atomics, no pre-zero dependency): lanes 0..7 each
    /// assemble one quant word from 4 lanes' int8s via <c>subgroupShuffle</c>, and lane 0 writes
    /// the header. Each output word is written by exactly one lane.
    ///
    /// local_size_x = 256 → 8 sub-blocks per workgroup, 32 lanes each (#318 pins the subgroup to 32,
    /// which the subgroupMax/subgroupAdd reductions + the subgroupShuffle packing require).
    ///
    /// Bindings: 0 = input (float, [nTok][cols]), 1 = output (uint, Q8_1 packed bytes).
    /// Push constants: { uint rows, uint cols, uint nTok } — `rows` is unused (kept for the shared
    /// MatVecBatchedParams push-constant struct); the dispatch covers nTok·(cols/32) sub-blocks.
    /// </summary>
    internal const string QuantizeQ8_1 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable
        #extension GL_KHR_shader_subgroup_arithmetic : enable
        #extension GL_KHR_shader_subgroup_shuffle : enable

        // 8 sub-blocks per workgroup, 32 lanes per sub-block = 256 threads.
        #define SUBBLOCKS_PER_WG 8

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly  buffer Input  { float input_data[]; };
        layout(binding = 1) writeonly buffer Output { uint  out_data[];   };

        layout(push_constant) uniform Params {
            uint rows;   // unused (shared param struct)
            uint cols;
            uint nTok;
        };

        void main() {
            uint tid  = gl_LocalInvocationID.x;
            uint lane = tid & 31u;

            uint sub_blocks_per_tok = cols >> 5;            // cols / 32
            uint total_sub_blocks   = nTok * sub_blocks_per_tok;
            uint sb = gl_WorkGroupID.x * SUBBLOCKS_PER_WG + (tid >> 5);
            if (sb >= total_sub_blocks) return;

            uint tok    = sb / sub_blocks_per_tok;
            uint sb_tok = sb - tok * sub_blocks_per_tok;    // sub-block index within the token

            float val = input_data[tok * cols + sb_tok * 32u + lane];

            // amax / d / q / qsum over the 32-lane sub-block (mirrors CUDA llm_quantize_q8_1).
            float a    = subgroupMax(abs(val));
            float d    = a / 127.0;
            float invd = (d == 0.0) ? 0.0 : (1.0 / d);
            int   q    = clamp(int(round(val * invd)), -127, 127);
            int   qsum = subgroupAdd(q);

            // 36-byte Q8_1 block = 9 aligned, disjoint words: word 0 = {fp16 d, fp16 d·qsum},
            // words 1..8 = 32 int8 quants. Each output word written by exactly one lane.
            uint word_base = sb * 9u;
            uint qb = uint(q) & 0xFFu;                       // this lane's int8 quant (low byte)

            // Pack 4 adjacent lanes' quants per word. subgroupShuffle must run in UNIFORM control
            // flow (all 32 lanes active) — its source lanes span the whole subgroup — so the shuffle
            // happens for every lane and only lanes 0..7 store. Word w (w = lane) gathers lanes 4w..4w+3.
            uint src = (lane * 4u) & 31u;                    // in-range for all lanes; only lane<8 stores
            uint b0 = subgroupShuffle(qb, src + 0u);
            uint b1 = subgroupShuffle(qb, src + 1u);
            uint b2 = subgroupShuffle(qb, src + 2u);
            uint b3 = subgroupShuffle(qb, src + 3u);
            if (lane < 8u)
                out_data[word_base + 1u + lane] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);

            if (lane == 0u) {
                uint d_bits = packHalf2x16(vec2(d, 0.0)) & 0xFFFFu;
                uint s_bits = packHalf2x16(vec2(d * float(qsum), 0.0)) & 0xFFFFu;
                out_data[word_base] = d_bits | (s_bits << 16u);
            }
        }
        """;

    /// <summary>
    /// Batched (weight-stationary) Q4_K matvec via int8-activation DP4A — the make-or-break
    /// weight-amortization for Vulkan speculative decoding (issue #308 P1). Drop-in replacement
    /// for <see cref="MatVecBatchedQ4K"/> when <c>VK_KHR_shader_integer_dot_product</c> is present;
    /// the FP variant remains the fallback. Mirrors CUDA's <c>llm_matvec_q4k_ws_n</c> exactly.
    ///
    /// The expensive per-weight work (read the Q4_K nibble word, unpack the 6-bit (sc, mn) pair,
    /// fold super_d·sc / super_dmin·mn — all token-INVARIANT) is hoisted ONCE per output element.
    /// The per-token inner cost collapses from 8 FP loads+FMAs/weight-word to: load one int8
    /// activation word + its fp16 scale, then two <c>dotPacked4x8AccSatEXT</c> intrinsics
    ///   dot = ⟨nibbles, q_act⟩,   sum = ⟨0x01010101, q_act⟩ (the Σq min-bias),
    /// and fold the scales onto the int32 dot. Identity (per 32-element sub-block):
    ///   Σ w·a = (super_d · sc · d8) · Σ(nibble · q)  −  (super_dmin · mn · d8) · Σq.
    /// The activation is read from the Q8_1 buffer (<see cref="QuantizeQ8_1"/>), NOT FP32.
    ///
    /// LOSSY (int8 activation quant) but ARGMAX-STABLE vs <see cref="MatVecBatchedQ4K"/> — the same
    /// trade-off as the CUDA DP4A path. Spec-decode verify accepts on argmax, so greedy spec stays
    /// lossless; the parity test relaxes to argmax-match + maxAbs &lt; 1.0 (not bit-exact).
    ///
    /// SaturatING dp4a (<c>...AccSatEXT</c>) — the int32 acc starts at 0 and the per-call partial
    /// sums (4×|nibble≤15|×|q≤127| ≤ 7620, or 4×127 for Σq) never overflow, so the saturation is
    /// inert and the result equals a plain <c>dotPacked4x8EXT</c>; the Sat overload is used only
    /// because it is the most broadly supported entry point.
    ///
    /// Lane→element layout is IDENTICAL to <see cref="MatVecBatchedQ4K"/> / the single-row
    /// MatVecQ4K (chunk = lane>>3, the 8 weight uints per chunk, sub-block 2·chunk = low nibbles,
    /// 2·chunk+1 = high nibbles), so the dp4a sum reproduces the same weight·activation pairing.
    ///
    /// Bindings: 0 = Q4_K weights (uint8), 1 = Q8_1 activations (uint, [nTok][cols/32 × 36 B]),
    /// 2 = outputs (float, row-major [nTok][rows]). Push constants: { uint rows, uint cols, uint nTok }.
    /// local_size_x = 256 (8 rows × 32 lanes) → #318 pins the subgroup to 32 (THREADS_PER_ROW == 32).
    /// </summary>
    internal const string MatVecBatchedQ4KInt8 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable
        #extension GL_KHR_shader_subgroup_arithmetic : enable
        #extension GL_EXT_integer_dot_product : require

        #define N_ROWS 8
        #define THREADS_PER_ROW 32
        #define MAX_NTOK 8

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Acts    { uint act_data[];     }; // Q8_1 packed
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
            uint nTok;
        };

        // Read a uint at an arbitrary BYTE offset from the (uint-typed) Q8_1 buffer. The Q8_1
        // 36-byte stride keeps every header 4-aligned, but the 4-int8 activation reads at
        // byte_off ∈ {0,4,…,28} land at base+4, which is 4-aligned too (block base is a multiple
        // of 36 → base%4 == 0). So a direct word index suffices; assert via the >>2.
        uint actWord(uint byteAddr) { return act_data[byteAddr >> 2]; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;                 // 256-element super-blocks per row
            uint word_row_base = row * num_blocks * 36u; // 36 uints per super-block

            // Q8_1 activation row stride: (cols/32) sub-blocks × 36 bytes.
            uint tok_byte_stride = (cols >> 5) * 36u;

            uint chunk     = lane >> 3;                  // 0..3
            uint byte_off  = (lane & 7u) * 4u;           // 0,4,…,28
            uint q4_offset = 4u + chunk * 8u + (lane & 7u);

            float acc[MAX_NTOK];
            [[unroll]] for (uint k = 0; k < MAX_NTOK; k++) acc[k] = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint word_base = word_row_base + block * 36u;

                vec2 dm = unpackHalf2x16(weights_data[word_base]);
                float super_d    = dm.x;
                float super_dmin = dm.y;

                uint sm0 = weights_data[word_base + 1];
                uint sm1 = weights_data[word_base + 2];
                uint sm2 = weights_data[word_base + 3];

                // Unpack this lane's two 6-bit (sc, mn) pairs: lo = sub-block 2·chunk, hi = 2·chunk+1.
                uint sc_lo, mn_lo, sc_hi, mn_hi;
                if (chunk == 0u) {
                    sc_lo = (sm0)        & 63u; mn_lo = (sm1)        & 63u;
                    sc_hi = (sm0 >>  8u) & 63u; mn_hi = (sm1 >>  8u) & 63u;
                } else if (chunk == 1u) {
                    sc_lo = (sm0 >> 16u) & 63u; mn_lo = (sm1 >> 16u) & 63u;
                    sc_hi = (sm0 >> 24u) & 63u; mn_hi = (sm1 >> 24u) & 63u;
                } else if (chunk == 2u) {
                    sc_lo = (sm2         & 0xFu) | (((sm0 >>  6u) & 3u) << 4u);
                    mn_lo = ((sm2 >>  4u) & 0xFu) | (((sm1 >>  6u) & 3u) << 4u);
                    sc_hi = ((sm2 >>  8u) & 0xFu) | (((sm0 >> 14u) & 3u) << 4u);
                    mn_hi = ((sm2 >> 12u) & 0xFu) | (((sm1 >> 14u) & 3u) << 4u);
                } else {
                    sc_lo = ((sm2 >> 16u) & 0xFu) | (((sm0 >> 22u) & 3u) << 4u);
                    mn_lo = ((sm2 >> 20u) & 0xFu) | (((sm1 >> 22u) & 3u) << 4u);
                    sc_hi = ((sm2 >> 24u) & 0xFu) | (((sm0 >> 30u) & 3u) << 4u);
                    mn_hi = ((sm2 >> 28u) & 0xFu) | (((sm1 >> 30u) & 3u) << 4u);
                }

                // Load this lane's weight word once; split into 4 low + 4 high nibbles.
                uint wq    = weights_data[word_base + q4_offset];
                uint wq_lo = wq & 0x0F0F0F0Fu;          // 4 low nibbles  → sub-block 2·chunk
                uint wq_hi = (wq >> 4u) & 0x0F0F0F0Fu;  // 4 high nibbles → sub-block 2·chunk+1

                // Token-invariant folded scales (weight read amortized across all nTok tokens).
                float sd_sc_lo = super_d    * float(sc_lo);
                float sm_mn_lo = super_dmin * float(mn_lo);
                float sd_sc_hi = super_d    * float(sc_hi);
                float sm_mn_hi = super_dmin * float(mn_hi);

                // Q8_1 byte base for the two sub-blocks (within a token's activation row).
                uint q81_base_lo = (block * 8u + chunk * 2u)      * 36u;
                uint q81_base_hi = (block * 8u + chunk * 2u + 1u) * 36u;

                for (uint k = 0; k < nTok; k++) {
                    uint tok_base = k * tok_byte_stride;

                    // fp16 activation scale d8 (low 16 bits of each block header).
                    float d8_lo = unpackHalf2x16(actWord(tok_base + q81_base_lo)).x;
                    float d8_hi = unpackHalf2x16(actWord(tok_base + q81_base_hi)).x;

                    // 4 int8 activations per sub-block at byte offset (4 + byte_off).
                    uint act_lo = actWord(tok_base + q81_base_lo + 4u + byte_off);
                    uint act_hi = actWord(tok_base + q81_base_hi + 4u + byte_off);

                    // dp4a (signed×signed int8): dot(4 nibbles, 4 int8 acts) + Σq via
                    // dot(0x01010101, acts). The signed EXT overload takes int args, so the
                    // packed uints are bit-reinterpreted to int (nibbles 0..15 are positive →
                    // identical bits; mirrors CUDA's (int)wq_lo / (int)0x01010101 casts).
                    int dot_lo = dotPacked4x8AccSatEXT(int(wq_lo),       int(act_lo), 0);
                    int dot_hi = dotPacked4x8AccSatEXT(int(wq_hi),       int(act_hi), 0);
                    int sum_lo = dotPacked4x8AccSatEXT(int(0x01010101u), int(act_lo), 0);
                    int sum_hi = dotPacked4x8AccSatEXT(int(0x01010101u), int(act_hi), 0);

                    acc[k] += (sd_sc_lo * d8_lo) * float(dot_lo) - (sm_mn_lo * d8_lo) * float(sum_lo);
                    acc[k] += (sd_sc_hi * d8_hi) * float(dot_hi) - (sm_mn_hi * d8_hi) * float(sum_hi);
                }
            }

            for (uint k = 0; k < nTok; k++) {
                float r = subgroupAdd(acc[k]);
                if (subgroupElect())
                    output_data[k * rows + row] = r;
            }
        }
        """;

    /// <summary>
    /// Batched (M=K) weight-stationary matrix-vector multiply with Q6_K dequantization —
    /// the Q6_K sibling of <see cref="MatVecBatchedQ4K"/>. Q4_K_M models pack most weights
    /// as Q4_K but keep ffn_down and token_embd/output as Q6_K, so the batched trunk needs
    /// a Q6_K batched matvec too (issue #308). Computes <c>output[k][row] = Σ_c W[row][c] *
    /// input[k][c]</c> for k ∈ [0, nTok). The expensive part (reading + unpacking each weight
    /// from VRAM) is done ONCE per output element and then multiplied into <c>nTok</c>
    /// accumulators (one per input vector), so the weight is read from VRAM once for all K
    /// tokens instead of K times. Only the per-token input reads are repeated.
    ///
    /// Bindings: 0=quantized weights (uint8), 1=inputs (float, row-major [nTok][cols]),
    /// 2=outputs (float, row-major [nTok][rows]). Push constants: { uint rows, uint cols, uint nTok }.
    ///
    /// BIT-EXACT vs nTok separate single-row <see cref="MatVecQ6K"/> calls: the element
    /// iteration order (the 8 explicit per-lane elements at lane, lane+32, …, lane+224), the
    /// per-element Q6_K dequant, and the subgroupAdd reduction are IDENTICAL to the single-row
    /// shader — only the k (token) dimension is added on top. The same floating-point
    /// accumulation order is therefore preserved per (row, token).
    ///
    /// local_size_x = 256 (8 rows × 32 lanes) so #318 pins the subgroup size to 32, which the
    /// subgroupAdd reduction requires (THREADS_PER_ROW == 32).
    ///
    /// nTok is capped at 8 (the acc[] register array size; matches the spec-decode draft cap).
    /// </summary>
    internal const string MatVecBatchedQ6K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable
        #extension GL_KHR_shader_subgroup_arithmetic : enable

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        // Q6_K block layout (210 bytes per 256 elements):
        //   [0:128]   ql — lower 4-bit nibbles (two 64-byte halves)
        //   [128:192] qh — upper 2-bit pairs (two 32-byte halves)
        //   [192:208] 16 int8 scale values
        //   [208:210] FP16 super-block scale d
        // Thread layout: each lane handles 8 elements (lane, lane+32, ..., lane+224).
        #define N_ROWS 8
        #define THREADS_PER_ROW 32
        #define MAX_NTOK 8

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Input   { float input_data[]; };
        layout(binding = 2) writeonly buffer Output  { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
            uint nTok;
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

            float acc[MAX_NTOK];
            [[unroll]] for (uint k = 0; k < MAX_NTOK; k++) acc[k] = 0.0;

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

                // Each weight value w is dequantized ONCE here, then multiplied into all nTok
                // input accumulators. Same element order + same w as single-row MatVecQ6K.
                float w0 = sc0 * float(int((ql0 & 0xF)        | (((qh0 >> 0) & 3) << 4)) - 32);
                float w1 = sc1 * float(int((ql1 & 0xF)        | (((qh0 >> 2) & 3) << 4)) - 32);
                float w2 = sc2 * float(int(((ql0 >> 4) & 0xF) | (((qh0 >> 4) & 3) << 4)) - 32);
                float w3 = sc3 * float(int(((ql1 >> 4) & 0xF) | (((qh0 >> 6) & 3) << 4)) - 32);
                float w4 = sc4 * float(int((ql2 & 0xF)        | (((qh1 >> 0) & 3) << 4)) - 32);
                float w5 = sc5 * float(int((ql3 & 0xF)        | (((qh1 >> 2) & 3) << 4)) - 32);
                float w6 = sc6 * float(int(((ql2 >> 4) & 0xF) | (((qh1 >> 4) & 3) << 4)) - 32);
                float w7 = sc7 * float(int(((ql3 >> 4) & 0xF) | (((qh1 >> 6) & 3) << 4)) - 32);

                for (uint k = 0; k < nTok; k++) {
                    uint in_base = k * cols + base_elem + lane;
                    acc[k] += w0 * input_data[in_base];
                    acc[k] += w1 * input_data[in_base +  32];
                    acc[k] += w2 * input_data[in_base +  64];
                    acc[k] += w3 * input_data[in_base +  96];
                    acc[k] += w4 * input_data[in_base + 128];
                    acc[k] += w5 * input_data[in_base + 160];
                    acc[k] += w6 * input_data[in_base + 192];
                    acc[k] += w7 * input_data[in_base + 224];
                }
            }

            for (uint k = 0; k < nTok; k++) {
                float r = subgroupAdd(acc[k]);
                if (subgroupElect())
                    output_data[k * rows + row] = r;
            }
        }
        """;

    /// <summary>
    /// Batched (weight-stationary) Q6_K matvec via int8-activation DP4A — the Q6_K sibling of
    /// <see cref="MatVecBatchedQ4KInt8"/> (issue #308 P2). Q4_K_M models keep ffn_down and
    /// token_embd/output as Q6_K, so the Q4_K-only int8 path of P1 left ~⅓ of the trunk on the
    /// slow FP <see cref="MatVecBatchedQ6K"/>; this shader pushes Q6_K onto the same DP4A path so
    /// the WHOLE spec-decode trunk amortizes the weight read across all nTok draft tokens. Drop-in
    /// replacement for <see cref="MatVecBatchedQ6K"/> when <c>VK_KHR_shader_integer_dot_product</c>
    /// is present; the FP variant remains the fallback. Mirrors CUDA's Q6_K decode-MMQ int8 dot.
    ///
    /// The expensive per-weight work (read the ql/qh bytes, reconstruct the 6-bit quant, fold the
    /// int8 sub-scale and super-block d — all token-INVARIANT) is hoisted ONCE per output element.
    /// The per-token inner cost collapses to: load one int8 activation word + its fp16 scale, then
    /// one <c>dotPacked4x8AccSatEXT</c>. Q6_K has NO min/dmin term (unlike Q4_K), so the identity is
    /// simpler — no Σq bias, no 0x01010101 dot:
    ///   Σ w·a = (d · scale · d8) · Σ((q6 − 32) · q8)   over each group of 4 elements.
    /// The activation is read from the SAME Q8_1 buffer as the Q4_K int8 path
    /// (<see cref="QuantizeQ8_1"/>) — Q6_K reuses the identical int8 activations, no new quant.
    ///
    /// LOSSY (int8 activation quant) but ARGMAX-STABLE vs <see cref="MatVecBatchedQ6K"/> — the same
    /// trade-off as the CUDA DP4A path and the Q4_K int8 sibling. Spec-decode verify accepts on
    /// argmax, so greedy spec stays lossless; the parity test relaxes to argmax-match + maxAbs &lt; 1.0.
    ///
    /// The int8 weight is <c>(q6 − 32) ∈ [−32, 31]</c>, which fits signed int8 — packed 4 per uint
    /// for the signed dp4a. Lane→element layout: each lane owns 8 CONTIGUOUS columns
    /// <c>lane·8 .. lane·8+7</c> of the 256-element super-block (32 lanes × 8 = 256), split into two
    /// dp4a groups of 4 contiguous columns. Each group lands wholly inside one 32-element Q8_1
    /// sub-block (its 4 int8 activations are one aligned word) and inside one 16-element Q6_K scale
    /// group (scale index <c>lane/2</c>, shared by both groups of the lane). This differs from the
    /// FP shader's strided per-lane element order, but it pairs each weight column with its OWN
    /// activation column — the products Σ w[c]·a[c] are identical (only the FP reduction order
    /// changes, which argmax-stability permits). The per-column (q6 − 32) reconstruction reuses the
    /// exact ql/qh nibble + qh-pair-shift recipe of <see cref="MatVecBatchedQ6K"/> / MatVecQ6K
    /// (column c → l = c%32, j = c/32), so the dequantized quant matches the FP path bit-for-bit.
    ///
    /// Bindings: 0 = Q6_K weights (uint8), 1 = Q8_1 activations (uint, [nTok][cols/32 × 36 B]),
    /// 2 = outputs (float, row-major [nTok][rows]). Push constants: { uint rows, uint cols, uint nTok }.
    /// local_size_x = 256 (8 rows × 32 lanes) → #318 pins the subgroup to 32 (THREADS_PER_ROW == 32).
    /// </summary>
    internal const string MatVecBatchedQ6KInt8 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable
        #extension GL_KHR_shader_subgroup_arithmetic : enable
        #extension GL_EXT_integer_dot_product : require

        // 8 rows per workgroup, 32 threads per row = 256 threads.
        // Q6_K block layout (210 bytes per 256 elements):
        //   [0:128]   ql — lower 4-bit nibbles (two 64-byte halves)
        //   [128:192] qh — upper 2-bit pairs (two 32-byte halves)
        //   [192:208] 16 int8 scale values
        //   [208:210] FP16 super-block scale d
        // Lane layout: each lane owns 8 contiguous columns lane*8 .. lane*8+7, split into
        // two dp4a groups of 4 contiguous columns. Both groups share scale index lane/2.
        #define N_ROWS 8
        #define THREADS_PER_ROW 32
        #define MAX_NTOK 8

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Weights { uint weights_data[]; };
        layout(binding = 1) readonly buffer Acts    { uint act_data[];     }; // Q8_1 packed
        layout(binding = 2) writeonly buffer Output { float output_data[]; };

        layout(push_constant) uniform Params {
            uint rows;
            uint cols;
            uint nTok;
        };

        uint gByte(uint b) { return (weights_data[b >> 2] >> ((b & 3) * 8)) & 0xFF; }
        // Read a uint at a 4-aligned BYTE offset of the (uint-typed) Q8_1 buffer (see Q4K int8).
        uint actWord(uint byteAddr) { return act_data[byteAddr >> 2]; }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;                 // 256-element super-blocks per row
            uint boff_base  = row * num_blocks * 210u;   // Q6_K weights are byte-addressed (210 B/block)

            // Q8_1 activation row stride: (cols/32) sub-blocks × 36 bytes.
            uint tok_byte_stride = (cols >> 5) * 36u;

            // This lane's 8 contiguous columns within a block: groupA = base0..+3, groupB = base1..+3.
            uint base0   = lane * 8u;        // first column of group A within the block
            uint base1   = base0 + 4u;       // first column of group B
            uint isc     = lane >> 1;        // shared Q6_K scale index (lane/2) for both groups

            // Group A column j = base0/32; group B column j = base1/32 (constant within a 4-group).
            uint jA = base0 >> 5;            // 0..7
            uint jB = base1 >> 5;
            uint lA = base0 & 31u;           // first of 4 consecutive ql/qh lanes
            uint lB = base1 & 31u;

            // ql byte base + high-nibble flag + qh byte base + qh 2-bit shift, per j (mirrors the
            // MatVecBatchedQ6K per-element extraction; see /tmp derivation in PR notes).
            //   j: 0->(ql 0,  lo, qh 128, sh0) 1->(ql 32, lo, qh 128, sh2) 2->(ql 0,  hi, qh 128, sh4)
            //      3->(ql 32, hi, qh 128, sh6) 4->(ql 64, lo, qh 160, sh0) 5->(ql 96, lo, qh 160, sh2)
            //      6->(ql 64, hi, qh 160, sh4) 7->(ql 96, hi, qh 160, sh6)
            uint qlbaseA = ((jA & 1u) == 0u) ? ((jA < 4u) ? 0u : 64u) : ((jA < 4u) ? 32u : 96u);
            uint qlbaseB = ((jB & 1u) == 0u) ? ((jB < 4u) ? 0u : 64u) : ((jB < 4u) ? 32u : 96u);
            bool hiA = (jA == 2u) || (jA == 3u) || (jA == 6u) || (jA == 7u);
            bool hiB = (jB == 2u) || (jB == 3u) || (jB == 6u) || (jB == 7u);
            uint qhbaseA = (jA < 4u) ? 128u : 160u;
            uint qhbaseB = (jB < 4u) ? 128u : 160u;
            uint qhshA = (jA & 3u) * 2u;     // 0,2,4,6
            uint qhshB = (jB & 3u) * 2u;

            // Q8_1 byte base for each group's sub-block + the 4-int8 word offset within it.
            uint subA      = base0 >> 5;     // == jA (sub-block index within the block)
            uint subB      = base1 >> 5;
            uint wordOffA  = (base0 & 31u);  // int8 position of the 4-group within its sub-block (0,4,..,28)
            uint wordOffB  = (base1 & 31u);

            float acc[MAX_NTOK];
            [[unroll]] for (uint k = 0; k < MAX_NTOK; k++) acc[k] = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = boff_base + block * 210u;

                float d = unpackHalf2x16(gByte(b0 + 208u) | (gByte(b0 + 209u) << 8u)).x;
                int   sc = int(gByte(b0 + 192u + isc));
                sc = (sc >= 128) ? sc - 256 : sc;           // int8 sub-scale
                float dsc = d * float(sc);                   // token-invariant folded scale (both groups)

                // Reconstruct the 4 int8 weights (q6 − 32) ∈ [−32,31] for each group, pack 4/int.
                uint wpackA = 0u, wpackB = 0u;
                [[unroll]] for (uint t = 0u; t < 4u; t++) {
                    uint qlA = gByte(b0 + qlbaseA + lA + t);
                    uint qhA = gByte(b0 + qhbaseA + lA + t);
                    int  q6A = int((hiA ? ((qlA >> 4u) & 0xFu) : (qlA & 0xFu)) | (((qhA >> qhshA) & 3u) << 4u)) - 32;
                    wpackA |= (uint(q6A) & 0xFFu) << (t * 8u);

                    uint qlB = gByte(b0 + qlbaseB + lB + t);
                    uint qhB = gByte(b0 + qhbaseB + lB + t);
                    int  q6B = int((hiB ? ((qlB >> 4u) & 0xFu) : (qlB & 0xFu)) | (((qhB >> qhshB) & 3u) << 4u)) - 32;
                    wpackB |= (uint(q6B) & 0xFFu) << (t * 8u);
                }

                // Q8_1 byte base for the two sub-blocks (within a token's activation row).
                uint q81_base_A = (block * 8u + subA) * 36u;
                uint q81_base_B = (block * 8u + subB) * 36u;

                for (uint k = 0; k < nTok; k++) {
                    uint tok_base = k * tok_byte_stride;

                    // fp16 activation scale d8 (low 16 bits of each sub-block header).
                    float d8A = unpackHalf2x16(actWord(tok_base + q81_base_A)).x;
                    float d8B = unpackHalf2x16(actWord(tok_base + q81_base_B)).x;

                    // 4 int8 activations per group at byte offset (4 + wordOff).
                    uint actA = actWord(tok_base + q81_base_A + 4u + wordOffA);
                    uint actB = actWord(tok_base + q81_base_B + 4u + wordOffB);

                    // Signed×signed dp4a: Σ((q6−32)·q8). NO min term (Q6_K has no dmin). The Sat
                    // overload is inert here (|partial| ≤ 4·32·127 = 16256, far below int32 overflow).
                    int dotA = dotPacked4x8AccSatEXT(int(wpackA), int(actA), 0);
                    int dotB = dotPacked4x8AccSatEXT(int(wpackB), int(actB), 0);

                    acc[k] += (dsc * d8A) * float(dotA) + (dsc * d8B) * float(dotB);
                }
            }

            for (uint k = 0; k < nTok; k++) {
                float r = subgroupAdd(acc[k]);
                if (subgroupElect())
                    output_data[k * rows + row] = r;
            }
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with Q5_K dequantization.
    /// Each workgroup computes 8 output rows (8 rows × 32 lanes = 256 threads).
    /// Push constants: { uint rows, uint cols }.
    /// Bindings: 0=quantized weights (uint8), 1=input vector (float), 2=output (float).
    ///
    /// Q5_K block layout (176 bytes per 256 elements):
    ///   [0:2]     FP16 d (super-block scale)
    ///   [2:4]     FP16 dmin (super-block minimum)
    ///   [4:16]    12 bytes packed 6-bit (scale, min) pairs (8 pairs, same packing as Q4_K)
    ///   [16:48]   qh[32] — high bit per element (one bit, 8 polarities × 32 lanes)
    ///   [48:176]  ql[128] — lower 4 bits, two elements per byte
    /// Dequant per chunk c∈0..3, lane l∈0..31 (matches CPU DequantQ5K / CUDA llm_matvec_q5k):
    ///   y[64c+l]    = d*sc[2c]  * ((ql[32c+l]&0xF) + (qh[l]&(1<<2c)   ?16:0)) - dmin*m[2c]
    ///   y[64c+l+32] = d*sc[2c+1]* ((ql[32c+l]>>4)  + (qh[l]&(1<<(2c+1))?16:0)) - dmin*m[2c+1]
    /// The 6-bit (scale, min) unpack reuses the exact Q4_K logic (Q5_K packs scales
    /// identically); Q5_K only adds the qh high bit (+16) per quant. The super-block
    /// d/dmin and the 12 scale/min bytes occupy bytes [0:16] of each 176-byte block,
    /// which is 4-byte aligned, so they're read as four aligned uint words (like
    /// MatVecQ4K); the per-lane qh/ql bytes are byte-granular and use the byte-gather
    /// helper. Mirrors the CUDA llm_matvec_q5k kernel and the CPU DequantQ5K path.
    /// </summary>
    internal const string MatVecQ5K = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable
        #extension GL_KHR_shader_subgroup_arithmetic : enable

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

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 8;            // cols / 256
            uint boff_base = row * num_blocks * 176;

            float acc = 0.0;

            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = boff_base + block * 176;

                // b0 is always a multiple of 176 (hence 4-byte aligned), so the first
                // 16 bytes (d/dmin + 12 scale/min bytes) read as four aligned uint words,
                // exactly like MatVecQ4K — 4 global reads instead of 16 gByte gathers.
                uint word_base = b0 >> 2;
                vec2 dm = unpackHalf2x16(weights_data[word_base]);
                float d    = dm.x;
                float dmin = dm.y;

                // 12 packed scale/min bytes at b0+4 (identical packing to Q4_K).
                // sm0 = scales[0..3], sm1 = scales[4..7], sm2 = scales[8..11].
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

                // High bit for this lane: one qh byte per lane (qh[lane]), bits 2c / 2c+1
                // select the +16 polarity for chunk c low/high nibble respectively.
                uint qh_byte = gByte(b0 + 16 + lane);
                uint base_elem = block * 256;

                [[unroll]] for (uint c = 0; c < 4; c++) {
                    uint ql_byte = gByte(b0 + 48 + c * 32 + lane);
                    uint low4 = ql_byte & 0xF;
                    uint hi4  = (ql_byte >> 4) & 0xF;

                    uint u1 = 1u << (2u * c);
                    uint u2 = u1 << 1;
                    float hLo = (qh_byte & u1) != 0u ? 16.0 : 0.0;
                    float hHi = (qh_byte & u2) != 0u ? 16.0 : 0.0;

                    uint si = 2u * c;
                    uint elem_lo = base_elem + c * 64 + lane;
                    acc += (dsc[si]     * (float(low4) + hLo) - dmn[si])     * input_data[elem_lo];
                    acc += (dsc[si + 1] * (float(hi4)  + hHi) - dmn[si + 1]) * input_data[elem_lo + 32];
                }
            }

            float result = subgroupAdd(acc);
            if (subgroupElect())
                output_data[row] = result;
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with Q8_0 dequantization.
    /// Each workgroup computes 8 output rows (8 rows × 32 lanes = 256 threads).
    /// Push constants: { uint rows, uint cols }.
    /// Bindings: 0=quantized weights (uint8), 1=input vector (float), 2=output (float).
    ///
    /// Q8_0 block layout (34 bytes per 32 elements):
    ///   [0:2]  FP16 d (block scale)
    ///   [2:34] 32 int8 quantized values
    /// Dequant: value = d * int8. One lane handles one element per block.
    /// Mirrors the CUDA llm_matvec_q8_0 kernel and the CPU DequantQ8_0 path.
    /// </summary>
    internal const string MatVecQ8_0 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable
        #extension GL_KHR_shader_subgroup_arithmetic : enable

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

            uint num_blocks = cols >> 5;            // cols / 32
            uint boff_base = row * num_blocks * 34;

            float acc = 0.0;
            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = boff_base + block * 34;
                float d = unpackHalf2x16(gByte(b0) | (gByte(b0 + 1) << 8)).x;
                int q = gInt8(b0 + 2 + lane);
                acc += d * float(q) * input_data[block * 32 + lane];
            }

            float result = subgroupAdd(acc);
            if (subgroupElect())
                output_data[row] = result;
        }
        """;

    /// <summary>
    /// Matrix-vector multiply with Q4_0 dequantization.
    /// Each workgroup computes 8 output rows (8 rows × 32 lanes = 256 threads).
    /// Push constants: { uint rows, uint cols }.
    /// Bindings: 0=quantized weights (uint8), 1=input vector (float), 2=output (float).
    ///
    /// Q4_0 block layout (18 bytes per 32 elements):
    ///   [0:2]  FP16 d (block scale)
    ///   [2:18] 16 bytes of packed 4-bit nibbles (two signed nibbles per byte)
    /// Element j (0..15) = low nibble of qs[j]; element j+16 = high nibble of qs[j].
    /// Dequant: value = (nibble - 8) * d. One lane handles one element per block.
    /// Mirrors the CUDA llm_matvec_q4_0 kernel and the CPU DequantQ4_0 path.
    /// </summary>
    internal const string MatVecQ4_0 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable
        #extension GL_KHR_shader_subgroup_arithmetic : enable

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

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint row_in_wg = tid / THREADS_PER_ROW;
            uint lane = tid % THREADS_PER_ROW;
            uint row = gl_WorkGroupID.x * N_ROWS + row_in_wg;
            if (row >= rows) return;

            uint num_blocks = cols >> 5;            // cols / 32
            uint boff_base = row * num_blocks * 18;

            float acc = 0.0;
            for (uint block = 0; block < num_blocks; block++) {
                uint b0 = boff_base + block * 18;
                float d = unpackHalf2x16(gByte(b0) | (gByte(b0 + 1) << 8)).x;
                // lane 0..15: low nibble of qs[lane]; lane 16..31: high nibble of qs[lane-16].
                uint qbyte = gByte(b0 + 2 + (lane & 15));
                int nib = (lane < 16) ? int(qbyte & 0xF) : int(qbyte >> 4);
                acc += d * float(nib - 8) * input_data[block * 32 + lane];
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

    /// <summary>
    /// Flash-decoding split-KV partial attention (issue #312) — the Vulkan mirror of the CUDA
    /// <c>llm_attention_splitkv</c> kernel. The single-workgroup <see cref="Attention"/> shader
    /// launches only <c>num_heads</c> workgroups and serially scans the whole KV range, which
    /// collapses decode throughput at very long context (the two earlier single-workgroup
    /// online-softmax attempts regressed for exactly this reason). This kernel splits each head's
    /// causal sequence <c>[0, seq_len)</c> into fixed <c>CHUNK</c>-sized slices and dispatches a
    /// 2D grid of <c>num_heads × n_splits</c> workgroups, so the KV read parallelizes across the
    /// GPU. Each workgroup emits the UN-normalized online-softmax partial for its slice; the
    /// companion <see cref="AttentionSplitKvCombine"/> LSE-merges the per-head partials.
    ///
    /// fp32 K/V (the bf16/q8_0 caches use <see cref="AttentionSplitKvPartialBf16"/> /
    /// <see cref="AttentionSplitKvPartialQ8"/>, which differ only in the K/V read — issue #332).
    /// Scalar (no subgroup ops) — uses plain shared-memory tree reductions, so #318's
    /// subgroup-size pin is irrelevant here.
    ///
    /// Workgroup (h = gl_WorkGroupID.x, s = gl_WorkGroupID.y) handles slice
    /// <c>[s*CHUNK, min((s+1)*CHUNK, seq_len))</c>. Out-of-range splits (s*CHUNK ≥ seq_len, from
    /// the caller's n_splits = ceil(seq_len/CHUNK)) write (m=−inf, l=0) and return so the combine
    /// scale exp(m−gmax)=0 skips them. GQA: kv_head = h / (num_heads/num_kv_heads).
    ///
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim, uint seq_len, uint n_splits, uint window }.
    /// Bindings: 0=Q[num_heads*head_dim], 1=K_cache[seq_len*kv_dim], 2=V_cache[seq_len*kv_dim],
    ///           3=partial_o[num_heads*n_splits*head_dim], 4=partial_meta[num_heads*n_splits*2].
    /// </summary>
    internal const string AttentionSplitKvPartial = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q          { float q_data[]; };
        layout(binding = 1) readonly buffer KCache     { float k_cache[]; };
        layout(binding = 2) readonly buffer VCache     { float v_cache[]; };
        layout(binding = 3) buffer PartialO            { float partial_o[]; };
        layout(binding = 4) buffer PartialMeta         { float partial_meta[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint n_splits;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        const uint CHUNK = 512u;
        shared float sk_scores[512];   // per-slice scores (≤ CHUNK)
        shared float sdata[256];       // reduction scratch

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint h = gl_WorkGroupID.x;   // query head
            uint s = gl_WorkGroupID.y;   // KV split
            if (h >= num_heads || s >= n_splits) return;

            uint meta_off = (h * n_splits + s) * 2u;
            // SWA bound — mirrors the fp32 Attention shader (CPU ForwardPass.Attention).
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;
            uint t0 = s * CHUNK;
            uint t1 = t0 + CHUNK; if (t1 > seq_len) t1 = seq_len;
            // Empty for this split: out-of-range (t0 >= seq_len, fixed n_splits) OR entirely below
            // the sliding window (t1 <= start_seq). Mark empty and bail so the combine skips it
            // (scale = exp(−inf − gmax) = 0) and never reads a stale numerator.
            if (t0 >= seq_len || t1 <= start_seq) {
                if (tid == 0u) { partial_meta[meta_off] = -1.0/0.0; partial_meta[meta_off + 1u] = 0.0; }
                return;
            }
            // Clamp the slice's start to the window so positions < start_seq never contribute.
            if (t0 < start_seq) t0 = start_seq;
            uint n = t1 - t0;   // 1 ≤ n ≤ CHUNK

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim  = num_kv_heads * head_dim;
            float scale  = inversesqrt(float(head_dim));
            uint q_off   = h * head_dim;
            uint kv_base = t0 * kv_dim + kv_head * head_dim;   // first row of this (clamped) slice for this kv head

            // ─── Phase 1: scores for the slice → shared (indexed t − t0) ───
            for (uint t = tid; t < n; t += 256u) {
                float dot = 0.0;
                uint k_off = kv_base + t * kv_dim;
                for (uint d = 0u; d < head_dim; d++)
                    dot += q_data[q_off + d] * k_cache[k_off + d];
                sk_scores[t] = dot * scale;
            }
            barrier();

            // ─── Phase 2: local max over the slice ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < n; t += 256u) local_max = max(local_max, sk_scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] = max(sdata[tid], sdata[tid + r]);
                barrier();
            }
            float m_i = sdata[0];
            barrier();

            // exp(score − m_i) in place + local denom.
            float local_sum = 0.0;
            for (uint t = tid; t < n; t += 256u) {
                float e = exp(sk_scores[t] - m_i);
                sk_scores[t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] += sdata[tid + r];
                barrier();
            }
            float l_i = sdata[0];
            barrier();

            if (tid == 0u) { partial_meta[meta_off] = m_i; partial_meta[meta_off + 1u] = l_i; }

            // ─── Phase 3: UN-normalized weighted-V numerator for this slice ───
            uint o_off = (h * n_splits + s) * head_dim;
            for (uint d = tid; d < head_dim; d += 256u) {
                float acc = 0.0;
                for (uint t = 0u; t < n; t++) {
                    uint v_off = kv_base + t * kv_dim;   // same hoisted base as Phase 1
                    acc += sk_scores[t] * v_cache[v_off + d];
                }
                partial_o[o_off + d] = acc;
            }
        }
        """;

    /// <summary>
    /// Flash-decoding combine (issue #312) — the Vulkan mirror of the CUDA
    /// <c>llm_attention_combine</c> kernel. One workgroup per query head; LSE-merges the
    /// <c>n_splits</c> per-slice partials emitted by <see cref="AttentionSplitKvPartial"/> into
    /// the final attention output with the standard online-softmax rescale:
    ///   <c>m = max_s m_s ; l = Σ_s exp(m_s−m)·l_s ; out[d] = (Σ_s exp(m_s−m)·Õ_s[d]) / l</c>.
    /// Exact modulo FP reduction order. Empty splits carry m_s=−inf → scale 0 → skipped.
    /// MAX_SPLITS bounds the per-head split count (ceil(131072/512)=256).
    ///
    /// Push constants: { uint num_heads, uint head_dim, uint n_splits }.
    /// Bindings: 0=partial_o[num_heads*n_splits*head_dim], 1=partial_meta[num_heads*n_splits*2],
    ///           2=output[num_heads*head_dim].
    /// </summary>
    internal const string AttentionSplitKvCombine = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer PartialO    { float partial_o[]; };
        layout(binding = 1) readonly buffer PartialMeta { float partial_meta[]; };
        layout(binding = 2) buffer Out                  { float out_data[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint head_dim;
            uint n_splits;
        };

        const uint MAX_SPLITS = 256u;
        shared float sh_scale[256];   // per-split rescale exp(m_s − gmax)
        shared float red[256];        // reduction scratch
        shared float sh_gmax;
        shared float sh_denom;

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint h = gl_WorkGroupID.x;
            if (h >= num_heads) return;
            uint base = h * n_splits;

            // Global max over the splits' local maxima.
            float lmax = -1.0/0.0;
            for (uint s = tid; s < n_splits; s += 256u)
                lmax = max(lmax, partial_meta[(base + s) * 2u]);
            red[tid] = lmax;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) red[tid] = max(red[tid], red[tid + r]);
                barrier();
            }
            if (tid == 0u) sh_gmax = red[0];
            barrier();
            float gmax = sh_gmax;

            // Per-split rescale factor exp(m_s − gmax) + global denom Σ exp(m_s−gmax)·l_s.
            float ldenom = 0.0;
            for (uint s = tid; s < n_splits; s += 256u) {
                float m = partial_meta[(base + s) * 2u];
                float l = partial_meta[(base + s) * 2u + 1u];
                float sc = exp(m - gmax);
                sh_scale[s] = sc;
                ldenom += sc * l;
            }
            red[tid] = ldenom;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) red[tid] += red[tid + r];
                barrier();
            }
            if (tid == 0u) sh_denom = red[0];
            barrier();
            float inv = 1.0 / sh_denom;

            // Weighted sum of the per-split numerators across head_dim.
            uint po_base  = base * head_dim;     // first split's row for this head
            uint out_base = h * head_dim;
            for (uint d = tid; d < head_dim; d += 256u) {
                float acc = 0.0;
                for (uint s = 0u; s < n_splits; s++) {
                    float sc = sh_scale[s];
                    if (sc != 0.0) acc += sc * partial_o[po_base + s * head_dim + d];
                }
                out_data[out_base + d] = acc * inv;
            }
        }
        """;

    /// <summary>
    /// bf16 (issue #332) variant of <see cref="AttentionSplitKvPartial"/>: control flow is
    /// IDENTICAL to the fp32 partial; the ONLY difference is that the K/V cache buffers
    /// (bindings 1, 2) hold IEEE fp16 packed two-per-uint and are read via
    /// <c>unpackHalf2x16</c> (same idiom as <see cref="AttentionBf16"/>). The element addressing
    /// (<c>kv_base + t*kv_dim + d</c>) is identical to fp32; per element <c>e</c> the packed word
    /// is <c>e&gt;&gt;1</c> and the component is <c>e&amp;1</c> (head_dim/kv_dim are even — see the
    /// GpuForwardPass guard). All scores / softmax / value accumulation stay fp32; only the
    /// stored K/V mantissa is narrowed. The companion (dtype-agnostic, reads the fp32 partial
    /// buffers) <see cref="AttentionSplitKvCombine"/> is reused unchanged.
    ///
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim, uint seq_len, uint n_splits, uint window }.
    /// Bindings: 0=Q[num_heads*head_dim] (float), 1=K_cache (uint, fp16-packed),
    ///           2=V_cache (uint, fp16-packed), 3=partial_o[num_heads*n_splits*head_dim] (float),
    ///           4=partial_meta[num_heads*n_splits*2] (float).
    /// </summary>
    internal const string AttentionSplitKvPartialBf16 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q          { float q_data[]; };
        layout(binding = 1) readonly buffer KCache     { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache     { uint v_cache[]; };
        layout(binding = 3) buffer PartialO            { float partial_o[]; };
        layout(binding = 4) buffer PartialMeta         { float partial_meta[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint n_splits;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        const uint CHUNK = 512u;
        shared float sk_scores[512];   // per-slice scores (≤ CHUNK)
        shared float sdata[256];       // reduction scratch

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint h = gl_WorkGroupID.x;   // query head
            uint s = gl_WorkGroupID.y;   // KV split
            if (h >= num_heads || s >= n_splits) return;

            uint meta_off = (h * n_splits + s) * 2u;
            // SWA bound — mirrors the fp32 split-KV partial.
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;
            uint t0 = s * CHUNK;
            uint t1 = t0 + CHUNK; if (t1 > seq_len) t1 = seq_len;
            // Empty for this split: out-of-range (t0 >= seq_len) OR entirely below the sliding
            // window (t1 <= start_seq). Mark empty and bail so the combine skips it
            // (scale = exp(−inf − gmax) = 0) and never reads a stale numerator.
            if (t0 >= seq_len || t1 <= start_seq) {
                if (tid == 0u) { partial_meta[meta_off] = -1.0/0.0; partial_meta[meta_off + 1u] = 0.0; }
                return;
            }
            // Clamp the slice's start to the window so positions < start_seq never contribute.
            if (t0 < start_seq) t0 = start_seq;
            uint n = t1 - t0;   // 1 ≤ n ≤ CHUNK

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim  = num_kv_heads * head_dim;
            float scale  = inversesqrt(float(head_dim));
            uint q_off   = h * head_dim;
            uint kv_base = t0 * kv_dim + kv_head * head_dim;   // first row of this (clamped) slice for this kv head

            // ─── Phase 1: scores for the slice → shared (indexed t − t0) ───
            // Read each packed fp16 word once (two K elements at a time). kv_base + t*kv_dim is
            // even (head_dim is even — see the GpuForwardPass guard) so >>1 is the exact word base
            // and consecutive d,d+1 are the two halves of word k_off_half+dh — mirrors AttentionBf16.
            for (uint t = tid; t < n; t += 256u) {
                float dot = 0.0;
                uint k_off_half = (kv_base + t * kv_dim) >> 1;
                for (uint dh = 0u; dh < (head_dim >> 1); dh++) {
                    uint d = dh << 1;
                    vec2 kv = unpackHalf2x16(k_cache[k_off_half + dh]);
                    dot += q_data[q_off + d] * kv.x + q_data[q_off + d + 1u] * kv.y;
                }
                sk_scores[t] = dot * scale;
            }
            barrier();

            // ─── Phase 2: local max over the slice ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < n; t += 256u) local_max = max(local_max, sk_scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] = max(sdata[tid], sdata[tid + r]);
                barrier();
            }
            float m_i = sdata[0];
            barrier();

            // exp(score − m_i) in place + local denom.
            float local_sum = 0.0;
            for (uint t = tid; t < n; t += 256u) {
                float e = exp(sk_scores[t] - m_i);
                sk_scores[t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] += sdata[tid + r];
                barrier();
            }
            float l_i = sdata[0];
            barrier();

            if (tid == 0u) { partial_meta[meta_off] = m_i; partial_meta[meta_off + 1u] = l_i; }

            // ─── Phase 3: UN-normalized weighted-V numerator for this slice ───
            // Each thread owns ONE output dim d. Hoist the per-d word/component selection out of the
            // t-loop and walk the V row word base incrementally (kv_base>>1 is this slice's t=0 word
            // base; head_dim is even) — mirrors AttentionBf16's Phase 3.
            uint o_off = (h * n_splits + s) * head_dim;
            for (uint d = tid; d < head_dim; d += 256u) {
                uint d_half = d >> 1;
                uint component = d & 1u;
                uint v_off_half = (kv_base >> 1) + d_half;
                uint kv_dim_half = kv_dim >> 1;
                float acc = 0.0;
                for (uint t = 0u; t < n; t++) {
                    float vv = unpackHalf2x16(v_cache[v_off_half])[component];
                    acc += sk_scores[t] * vv;
                    v_off_half += kv_dim_half;
                }
                partial_o[o_off + d] = acc;
            }
        }
        """;

    /// <summary>
    /// q8_0 (issue #332) variant of <see cref="AttentionSplitKvPartial"/>: control flow is
    /// IDENTICAL to the fp32 partial; the ONLY difference is that the K/V cache buffers
    /// (bindings 1, 2) hold ggml <c>block_q8_0</c> (34 bytes/block = fp16 scale + 32 int8) and
    /// every element read becomes a byte-gather + dequant <c>value = fp16(d) * int8</c> — the
    /// same <c>loadK</c>/<c>loadV</c> idiom as <see cref="AttentionQ8_0"/>. Element addressing
    /// (<c>kv_base + t*kv_dim + d</c>) is identical to fp32; per absolute element <c>e</c>:
    /// <c>blk=e&gt;&gt;5</c>, <c>lane=e&amp;31</c>, <c>b0=blk*34</c>. kv_dim%32==0 (enforced in
    /// GpuForwardPass), so a KV row's blocks never straddle a row. All scores / softmax / value
    /// accumulation stay fp32; only the stored K/V is narrowed. The companion
    /// <see cref="AttentionSplitKvCombine"/> (reads the fp32 partial buffers) is reused unchanged.
    ///
    /// Push constants: { uint num_heads, uint num_kv_heads, uint head_dim, uint seq_len, uint n_splits, uint window }.
    /// Bindings: 0=Q[num_heads*head_dim] (float), 1=K_cache (uint, block_q8_0),
    ///           2=V_cache (uint, block_q8_0), 3=partial_o[num_heads*n_splits*head_dim] (float),
    ///           4=partial_meta[num_heads*n_splits*2] (float).
    /// </summary>
    internal const string AttentionSplitKvPartialQ8 = """
        #version 450
        #extension GL_EXT_control_flow_attributes : enable

        layout(local_size_x = 256) in;

        layout(binding = 0) readonly buffer Q          { float q_data[]; };
        layout(binding = 1) readonly buffer KCache     { uint k_cache[]; };
        layout(binding = 2) readonly buffer VCache     { uint v_cache[]; };
        layout(binding = 3) buffer PartialO            { float partial_o[]; };
        layout(binding = 4) buffer PartialMeta         { float partial_meta[]; };

        layout(push_constant) uniform Params {
            uint num_heads;
            uint num_kv_heads;
            uint head_dim;
            uint seq_len;
            uint n_splits;
            uint window;        // SWA: attend only [start_seq, seq_len); 0 = full attention
        };

        const uint CHUNK = 512u;
        shared float sk_scores[512];   // per-slice scores (≤ CHUNK)
        shared float sdata[256];       // reduction scratch

        // Sign-extend a single int8 byte in one bitfieldExtract (no ternary branch).
        int gInt8K(uint b) { return bitfieldExtract(int(k_cache[b >> 2]), int((b & 3u) * 8u), 8); }
        int gInt8V(uint b) { return bitfieldExtract(int(v_cache[b >> 2]), int((b & 3u) * 8u), 8); }

        void main() {
            uint tid = gl_LocalInvocationID.x;
            uint h = gl_WorkGroupID.x;   // query head
            uint s = gl_WorkGroupID.y;   // KV split
            if (h >= num_heads || s >= n_splits) return;

            uint meta_off = (h * n_splits + s) * 2u;
            // SWA bound — mirrors the fp32 split-KV partial.
            uint start_seq = (window != 0u && window < seq_len) ? (seq_len - window) : 0u;
            uint t0 = s * CHUNK;
            uint t1 = t0 + CHUNK; if (t1 > seq_len) t1 = seq_len;
            // Empty for this split: out-of-range (t0 >= seq_len) OR entirely below the sliding
            // window (t1 <= start_seq). Mark empty and bail so the combine skips it
            // (scale = exp(−inf − gmax) = 0) and never reads a stale numerator.
            if (t0 >= seq_len || t1 <= start_seq) {
                if (tid == 0u) { partial_meta[meta_off] = -1.0/0.0; partial_meta[meta_off + 1u] = 0.0; }
                return;
            }
            // Clamp the slice's start to the window so positions < start_seq never contribute.
            // kv_base stays a multiple of 32 (kv_dim & head_dim are multiples of 32), so the
            // block addressing (kv_base >> 5) is still exact after the clamp.
            if (t0 < start_seq) t0 = start_seq;
            uint n = t1 - t0;   // 1 ≤ n ≤ CHUNK

            uint kv_head = h / (num_heads / num_kv_heads);
            uint kv_dim  = num_kv_heads * head_dim;
            float scale  = inversesqrt(float(head_dim));
            uint q_off   = h * head_dim;
            uint kv_base = t0 * kv_dim + kv_head * head_dim;   // first row of this (clamped) slice for this kv head

            // ─── Phase 1: scores for the slice → shared (indexed t − t0) ───
            // Load each block's fp16 scale ONCE per 32-element block (head_dim & kv_dim are
            // multiples of 32 — enforced in GpuForwardPass), then dequant the 32 int8 lanes with it.
            // Mirrors AttentionQ8_0's read pattern; scale-once instead of per-element loadK.
            for (uint t = tid; t < n; t += 256u) {
                float dot = 0.0;
                uint k_off = kv_base + t * kv_dim;
                uint blk_start = k_off >> 5;
                for (uint blk = 0u; blk < (head_dim >> 5); blk++) {
                    uint b0 = (blk_start + blk) * 34u;
                    // b0 = blk*34 is even, so the two scale bytes [b0, b0+1] live in the same word.
                    uint w = k_cache[b0 >> 2];
                    float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
                    uint q_blk_off = q_off + blk * 32u;
                    for (uint lane = 0u; lane < 32u; lane++) {
                        dot += q_data[q_blk_off + lane] * (dsc * float(gInt8K(b0 + 2u + lane)));
                    }
                }
                sk_scores[t] = dot * scale;
            }
            barrier();

            // ─── Phase 2: local max over the slice ───
            float local_max = -1.0/0.0;
            for (uint t = tid; t < n; t += 256u) local_max = max(local_max, sk_scores[t]);
            sdata[tid] = local_max;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] = max(sdata[tid], sdata[tid + r]);
                barrier();
            }
            float m_i = sdata[0];
            barrier();

            // exp(score − m_i) in place + local denom.
            float local_sum = 0.0;
            for (uint t = tid; t < n; t += 256u) {
                float e = exp(sk_scores[t] - m_i);
                sk_scores[t] = e;
                local_sum += e;
            }
            sdata[tid] = local_sum;
            barrier();
            [[unroll]] for (uint r = 128u; r > 0u; r >>= 1) {
                if (tid < r) sdata[tid] += sdata[tid + r];
                barrier();
            }
            float l_i = sdata[0];
            barrier();

            if (tid == 0u) { partial_meta[meta_off] = m_i; partial_meta[meta_off + 1u] = l_i; }

            // ─── Phase 3: UN-normalized weighted-V numerator for this slice ───
            // Each thread owns ONE output dim d. Hoist the block index to a linear recurrence over t
            // (base_blk = this slice's t=0 block for dim d; stride_blk = kv_dim in blocks) so the
            // per-block scale is read once per t — mirrors AttentionQ8_0's Phase 3.
            uint o_off = (h * n_splits + s) * head_dim;
            for (uint d = tid; d < head_dim; d += 256u) {
                uint d_blk = d >> 5;
                uint lane = d & 31u;
                uint base_blk = (kv_base >> 5) + d_blk;
                uint stride_blk = kv_dim >> 5;
                float acc = 0.0;
                for (uint t = 0u; t < n; t++) {
                    uint b0 = (base_blk + t * stride_blk) * 34u;
                    uint w = v_cache[b0 >> 2];
                    float dsc = unpackHalf2x16((w >> ((b0 & 3u) * 8u)) & 0xFFFFu).x;
                    float vv = dsc * float(gInt8V(b0 + 2u + lane));
                    acc += sk_scores[t] * vv;
                }
                partial_o[o_off + d] = acc;
            }
        }
        """;
}
