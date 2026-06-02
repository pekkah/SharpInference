using SharpInference.Core;
using Xunit.Abstractions;

namespace SharpInference.Tests.Core;

public sealed class Gemma4E4BActualMetadataTests(ITestOutputHelper output)
{
    private const string ModelPath = @"E:\models\gemma-4-E4B-it-Q8_0.gguf";

    [Fact]
    public void Dump_RopeFreqs_And_NormWeightSample()
    {
        if (!File.Exists(ModelPath))
        {
            output.WriteLine($"Model file missing: {ModelPath} — skipping");
            return;
        }
        using var m = GgufModel.Open(ModelPath);

        var rf = m.Tensors.FirstOrDefault(t => t.Name == "rope_freqs.weight");
        if (rf.Name == "rope_freqs.weight")
        {
            var data = m.GetTensorData(rf);
            var floats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(data);
            output.WriteLine($"rope_freqs.weight length={floats.Length}, type={rf.DType}");
            output.WriteLine($"first 16: [{string.Join(", ", floats.Slice(0, 16).ToArray().Select(f => f.ToString("G6")))}]");
            output.WriteLine($"last 16:  [{string.Join(", ", floats.Slice(floats.Length - 16, 16).ToArray().Select(f => f.ToString("G6")))}]");

            output.WriteLine("\nfreq factor every 16 entries:");
            for (int i = 0; i < floats.Length; i += 16)
                output.WriteLine($"  [{i,3}] = {floats[i]:F6}");
        }

        // Sample a few norm weights to determine if +1 baked in
        var moreNames = new List<string>();
        for (int li = 0; li < 8; li++)
        {
            moreNames.Add($"blk.{li}.attn_q_norm.weight");
            moreNames.Add($"blk.{li}.attn_k_norm.weight");
            moreNames.Add($"blk.{li}.layer_output_scale.weight");
        }
        foreach (var n in moreNames)
        {
            var t = m.Tensors.FirstOrDefault(x => x.Name == n);
            if (t.Name != n) continue;
            var data = m.GetTensorData(t);
            var floats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(data);
            int len = floats.Length;
            float v0 = floats[0], vmid = floats[len / 2], vlast = floats[len - 1];
            output.WriteLine($"{n,-48} len={len,4}  [0]={v0:F4}  [{len/2}]={vmid:F4}  [{len-1}]={vlast:F4}");
        }

        foreach (var n in new[] { "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight", "blk.0.attn_q_norm.weight",
                                  "blk.0.post_attention_norm.weight", "blk.0.post_ffw_norm.weight",
                                  "blk.0.post_norm.weight", "blk.0.layer_output_scale.weight",
                                  "blk.5.attn_q_norm.weight", "blk.5.attn_k_norm.weight" })
        {
            var t = m.Tensors.FirstOrDefault(x => x.Name == n);
            if (t.Name != n) { output.WriteLine($"\n{n}: NOT FOUND"); continue; }
            var data = m.GetTensorData(t);
            var floats = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(data);
            int len = floats.Length;
            float sum = 0, sumSq = 0, mn = float.MaxValue, mx = float.MinValue;
            for (int i = 0; i < len; i++) { sum += floats[i]; sumSq += floats[i] * floats[i]; mn = MathF.Min(mn, floats[i]); mx = MathF.Max(mx, floats[i]); }
            float mean = sum / len, rms = MathF.Sqrt(sumSq / len);
            output.WriteLine($"\n{n}: len={len}, mean={mean:F4}, rms={rms:F4}, min={mn:F4}, max={mx:F4}");
            int show = Math.Min(8, len);
            output.WriteLine($"  first {show}: [{string.Join(", ", floats.Slice(0, show).ToArray().Select(f => f.ToString("F4")))}]");
        }
    }

    [Fact]
    public void Dump_TokenizerEncoding()
    {
        if (!File.Exists(ModelPath))
        {
            output.WriteLine($"Model file missing: {ModelPath} — skipping");
            return;
        }
        using var m = GgufModel.Open(ModelPath);
        var tok = GgufTokenizer.FromGgufModel(m);

        var prompt = "The capital of France is";
        output.WriteLine($"prompt: \"{prompt}\"");
        var ids = tok.Encode(prompt).ToArray();
        output.WriteLine($"sharpi encode: [{string.Join(", ", ids)}] (count={ids.Length})");
        output.WriteLine($"llama  encode: [818, 5279, 529, 7001, 563]");
        output.WriteLine($"\nDecoded sharpi tokens:");
        foreach (int id in ids)
            output.WriteLine($"  {id} -> \"{tok.Decode(new[] { id })}\"");
    }

    [Fact]
    public void Dump_SwaPattern_And_TensorList()
    {
        if (!File.Exists(ModelPath))
        {
            output.WriteLine($"Model file missing: {ModelPath} — skipping");
            return;
        }

        using var m = GgufModel.Open(ModelPath);

        // 1) Dump full SWA pattern
        if (m.Metadata.TryGetValue("gemma4.attention.sliding_window_pattern", out var v) && v is object[] arr)
        {
            output.WriteLine($"sliding_window_pattern (length {arr.Length}):");
            var pattern = new System.Text.StringBuilder();
            for (int i = 0; i < arr.Length; i++)
            {
                pattern.Append((bool)arr[i] ? 'S' : 'G');
            }
            output.WriteLine(pattern.ToString());

            var swa = new List<int>();
            var glob = new List<int>();
            for (int i = 0; i < arr.Length; i++)
            {
                if ((bool)arr[i]) swa.Add(i); else glob.Add(i);
            }
            output.WriteLine($"SWA layers ({swa.Count}): {string.Join(",", swa)}");
            output.WriteLine($"Global layers ({glob.Count}): {string.Join(",", glob)}");
        }
        else
        {
            output.WriteLine("NO sliding_window_pattern in metadata!");
        }

        // 2) Tensor inventory — blk.0, blk.1, blk.5 (first global), and final
        output.WriteLine($"\n=== Tensor count: {m.Tensors.Count} ===\n");

        var nonBlock = m.Tensors.Where(t => !t.Name.StartsWith("blk.")).ToList();
        output.WriteLine($"Non-block tensors ({nonBlock.Count}):");
        foreach (var t in nonBlock)
            output.WriteLine($"  {t.Name}  shape=[{string.Join(",", t.Dimensions)}]  type={t.DType}");

        foreach (int li in new[] { 0, 1, 5, 6, 23, 24, 25, 29, 30, 41 })
        {
            output.WriteLine($"\n--- blk.{li}.* ---");
            foreach (var t in m.Tensors.Where(t => t.Name.StartsWith($"blk.{li}.")))
                output.WriteLine($"  {t.Name}  shape=[{string.Join(",", t.Dimensions)}]  type={t.DType}");
        }
    }
}
