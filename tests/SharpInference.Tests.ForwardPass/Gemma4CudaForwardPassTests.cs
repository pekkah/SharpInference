using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Phase-8 CUDA forward-pass integration tests for Gemma 4 E4B. Confirms the
/// load-bearing trunk wiring — per-layer head_dim variance, dual-RoPE selection,
/// KV-share dispatch, SWA / full Attention split, post-attn / post-ffn norms,
/// PLE injection, GeluTanhMul FFN, layer_output_scale, final-logit softcap —
/// together produce a non-garbage decode stream on the real 8.2 GB unsloth GGUF
/// AND that the CUDA result tracks the CPU reference at first-decode argmax.
///
/// Silent-skip pattern: if CUDA isn't available OR the GGUF isn't present on
/// disk these tests no-op, mirroring the rest of the Cuda* test files.
/// </summary>
public sealed class Gemma4CudaForwardPassTests
{
    private const string ModelFile = "gemma-4-E4B-it-Q8_0.gguf";

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        string[] absoluteCandidates =
        {
            $@"E:\models\{ModelFile}",
            $@"C:\p\sharpi\models\{ModelFile}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static int ReadIntMetadata(GgufModel model, string key, int fallback)
    {
        if (!model.Metadata.TryGetValue(key, out var v) || v is null) return fallback;
        try { return Convert.ToInt32(v); } catch { return fallback; }
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    [Fact]
    public void Gemma4_E4B_CudaForward_ProducesNonGarbageLogits()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Defensive: the test only makes sense against a real gemma4 GGUF.
        Assert.NotNull(hp.LayerHeadDim);
        Assert.NotNull(hp.IsSwaLayer);
        Assert.True(hp.HasPerLayerTokenEmbd);
        Assert.True(hp.FinalLogitSoftcap > 0f);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        int eosId = ReadIntMetadata(model, "tokenizer.ggml.eos_token_id", fallback: 1);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        var logits = fwd.Prefill(tokens);
        Assert.Equal(hp.VocabSize, logits.Length);

        // Finite logits + non-EOS first decode + ≥2 distinct tokens across 4 steps.
        int nonFinite = 0;
        for (int i = 0; i < logits.Length; i++)
            if (!float.IsFinite(logits[i])) nonFinite++;
        Assert.True(nonFinite == 0, $"{nonFinite} non-finite logits in post-prompt output.");

        int firstDecode = Argmax(logits);
        if (eosId >= 0)
            Assert.NotEqual(eosId, firstDecode);

        Span<int> decoded = stackalloc int[4];
        decoded[0] = firstDecode;
        int pos = tokens.Length;
        for (int i = 1; i < decoded.Length; i++)
        {
            var step = fwd.Forward(decoded[i - 1], pos++);
            for (int k = 0; k < step.Length; k++)
                Assert.True(float.IsFinite(step[k]),
                    $"Non-finite logit at decode step {i}, vocab idx {k}: {step[k]}");
            decoded[i] = Argmax(step);
        }

        int distinct = 0;
        for (int i = 0; i < decoded.Length; i++)
        {
            bool seen = false;
            for (int j = 0; j < i; j++) if (decoded[j] == decoded[i]) { seen = true; break; }
            if (!seen) distinct++;
        }
        Assert.True(distinct >= 2,
            $"CUDA greedy decode produced only {distinct} distinct token(s) over {decoded.Length} steps " +
            $"({string.Join(",", decoded.ToArray())}); Gemma 4 CUDA forward integration is degenerate.");

        if (eosId >= 0)
        {
            int eosCount = 0;
            for (int i = 0; i < decoded.Length; i++)
                if (decoded[i] == eosId) eosCount++;
            Assert.True(eosCount < decoded.Length,
                $"All {decoded.Length} greedy-decoded tokens were EOS — Gemma 4 CUDA output is degenerate.");
        }
    }

    [Fact]
    public void Gemma4_E4B_CudaForward_MatchesCpuArgmax()
    {
        // The Phase 8 keystone: greedy argmax of the first decoded token after
        // a 9-token prefill must match between CPU and CUDA. Cumulative FP drift
        // across 42 layers + softcap means later tokens can diverge; the first
        // step is the tightest parity signal we can demand without bit-exactness.
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        // CPU reference.
        int cpuFirstArgmax;
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp))
        {
            var cpuLogits = cpuFwd.Prefill(tokens);
            cpuFirstArgmax = Argmax(cpuLogits);
        }

        // CUDA path.
        using var cudaFwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);
        var cudaLogits = cudaFwd.Prefill(tokens);
        int cudaFirstArgmax = Argmax(cudaLogits);

        // Diagnostic: top-3 each side so a divergence message is actionable.
        if (cpuFirstArgmax != cudaFirstArgmax)
        {
            var msg = new System.Text.StringBuilder();
            msg.AppendLine($"CPU first argmax: {cpuFirstArgmax}");
            msg.AppendLine($"CUDA first argmax: {cudaFirstArgmax}");

            int[] cudaTop3 = TopK(cudaLogits, 3);
            msg.Append("CUDA top-3: ");
            for (int i = 0; i < cudaTop3.Length; i++)
                msg.Append($"{cudaTop3[i]}({cudaLogits[cudaTop3[i]]:F2}) ");
            Assert.Fail(msg.ToString());
        }
    }

    private static int[] TopK(ReadOnlySpan<float> logits, int k)
    {
        var result = new int[k];
        var taken = new System.Collections.Generic.HashSet<int>();
        for (int ki = 0; ki < k; ki++)
        {
            int best = -1; float bestVal = float.NegativeInfinity;
            for (int i = 0; i < logits.Length; i++)
            {
                if (taken.Contains(i)) continue;
                if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
            }
            result[ki] = best;
            taken.Add(best);
        }
        return result;
    }
}
