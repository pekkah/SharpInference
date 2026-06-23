using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Vulkan;
using Vortice.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// End-to-end #357 PR4 verification on Vulkan: MTP self-speculative greedy decode vs the
/// MTP-off greedy decode (plain scalar Forward) on the dense Qwen3.6-27B-MTP GDN model.
/// MTP is lossless self-speculation, so the two token streams should agree, and the chained
/// drafts must land accepts on a repetitive prompt. Silent-skips when Vulkan or the GGUF is
/// absent, or the device can't fit the trunk + ring. NOT run by the implementation pass.
/// </summary>
public sealed class VulkanMtpE2ETests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? FindMtpModelPath()
    {
        const string fileName = "Qwen3.6-27B-MTP-Q4_K_M.gguf";
        foreach (var root in new[] { @"E:\models", @"C:\p\sharpi\models" })
        {
            var p = Path.Combine(root, fileName);
            if (File.Exists(p)) return p;
        }
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static LayerPlacement GdnPlacement(ModelHyperparams hp) => new(
        GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
        RecommendedCtxSize: Math.Min(hp.ContextLength, 2048));

    private static VulkanHybridGdnForwardPass? CreatePass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_MTP_BATCH_MAX");
        Environment.SetEnvironmentVariable("SHARPI_MTP_BATCH_MAX", "4");
        try { return new VulkanHybridGdnForwardPass(model, gpu, hp, GdnPlacement(hp)); }
        catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory) { return null; }
        finally { Environment.SetEnvironmentVariable("SHARPI_MTP_BATCH_MAX", prev); }
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    [Fact]
    public void MtpDecode_MatchesPlainGreedy_Vulkan()
    {
        using var gpu = TryCreateVulkan();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        using var fwd = CreatePass(model, gpu, hp);
        if (fwd is null) return;
        if (!fwd.HasMtpHead || !fwd.SupportsBatchVerify) return;

        // A repetitive prompt — high MTP acceptance, and the greedy continuation is well-separated
        // (few near-ties), so the batched-verify vs scalar-decode numerical gap rarely flips an argmax.
        var prompt = tokenizer.Encode("Repeat after me: the cat sat on the mat. The cat sat on the mat. The cat sat on the").ToArray();
        const int N = 32;
        int[] stops = tokenizer.EogTokenIds.ToArray();

        // ── MTP-OFF reference: plain greedy scalar Forward. Stop BEFORE emitting an EOG token,
        //    matching MtpDecoder's convention (it commits content tokens and halts at a stop
        //    without emitting it) so the two streams are length-comparable. ──
        var plain = new List<int>(N);
        {
            var logits = fwd.Prefill(prompt);
            int tok = ArgMax(logits);
            int pos = prompt.Length;
            for (int i = 0; i < N; i++)
            {
                if (Array.IndexOf(stops, tok) >= 0) break;
                plain.Add(tok);
                tok = ArgMax(fwd.Forward(tok, pos++));
            }
        }

        // ── MTP-ON: self-speculative greedy via MtpDecoder. ──
        fwd.ResetCache();
        var mtp = new List<int>(N);
        float acceptance;
        long emitted, accepted;
        {
            var logits = fwd.Prefill(prompt);
            var dec = new MtpDecoder(fwd);
            dec.Initialize(prompt.Length, logits);
            fwd.PrefillMtp(prompt);
            dec.Decode(N, stops, mtp.Add, pMin: 1f, draftN: MtpDecoder.ResolveDraftN(0));
            acceptance = dec.AcceptanceRate;
            emitted = dec.TotalDraftsEmitted;
            accepted = dec.TotalDraftsAccepted;
        }

        int common = Math.Min(plain.Count, mtp.Count);
        int firstDiff = -1;
        for (int i = 0; i < common; i++)
            if (plain[i] != mtp[i]) { firstDiff = i; break; }
        // A common-prefix match with mismatched lengths is still a divergence (e.g. one stream
        // stopped early); flag it at the first uncompared index so the assert below fires.
        if (firstDiff < 0 && plain.Count != mtp.Count)
            firstDiff = common;

        // MTP must actually speculate (chained drafts landing accepts on a repetitive prompt).
        Assert.True(emitted > 0, "MTP emitted no drafts — the head/decoder wiring is dead.");
        Assert.True(acceptance >= 0.3f,
            $"MTP acceptance {acceptance:P0} ({accepted}/{emitted}) is far below the depth-1 reference; " +
            "the MtpLastHidden chaining or MTP KV refresh is broken.");

        // Lossless self-speculation: the committed stream is the verify path's argmax. The Vulkan
        // batched-verify trunk is argmax-stable (not bit-exact) vs the scalar decode trunk, so a
        // near-tie CAN flip a single token; on a repetitive prompt that is not expected. Assert
        // byte-identical and surface the exact divergence if it ever happens.
        Assert.True(firstDiff < 0,
            $"MTP-on diverged from plain greedy at token {firstDiff}: " +
            $"plain=[{string.Join(",", plain)}] mtp=[{string.Join(",", mtp)}] " +
            $"(acceptance {acceptance:P0}). If this is a lone near-tie flip it is the known batched-" +
            "verify-vs-scalar argmax-stability gap; a structural divergence (many tokens) is a bug.");
    }
}
