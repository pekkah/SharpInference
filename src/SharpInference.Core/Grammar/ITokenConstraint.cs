using System.Collections.Immutable;
using System.Text;

namespace SharpInference.Core.Grammar;

/// <summary>
/// A per-request decoding constraint (issue #374). Observes the emitted token stream and, while a
/// constrained region is active, restricts which tokens may be sampled next so the output is forced
/// to satisfy a grammar (e.g. a tool call's argument object in the model's native wire syntax).
///
/// <para>
/// Lifecycle, per generated token: the engine calls <see cref="Accept"/> for <b>every</b> emitted
/// token (so the constraint can detect when a constrained region begins/ends), and — only while
/// <see cref="IsConstraining"/> is true — calls <see cref="Filter"/> on the logits just before
/// sampling to mask the tokens the grammar forbids. When <see cref="IsConstraining"/> is false the
/// engine samples exactly as it would unconstrained, so a request whose generation never enters a
/// constrained region is byte-identical to the default path.
/// </para>
///
/// <para>Implementations are single-request (the engine serializes generation) and must be
/// allocation-free on the per-token hot path after a one-time warm-up.</para>
/// </summary>
public interface ITokenConstraint
{
    /// <summary>
    /// True when the constraint is currently restricting the vocabulary. The engine applies
    /// <see cref="Filter"/> only when this is true; otherwise it samples unmasked.
    /// </summary>
    bool IsConstraining { get; }

    /// <summary>
    /// Returns a masked view of <paramref name="logits"/> for the current constrained state: every
    /// token the grammar forbids is set to negative infinity, leaving the allowed tokens untouched
    /// so the sampler's downstream pipeline (greedy/temperature/top-k/…) selects only a grammar-legal
    /// token. The returned span is owned by the constraint and reused across calls — sample from it
    /// immediately. Only valid to call when <see cref="IsConstraining"/> is true. If the grammar
    /// reaches a dead state (no legal token), returns <paramref name="logits"/> unchanged so
    /// generation never wedges. "Unchanged" means literally returning the same span passed in (not
    /// a value-equal copy) — composing constraints such as <see cref="AndTokenConstraint"/> rely on
    /// reference equality to detect a dead-state punt without an O(vocab) comparison.
    /// </summary>
    ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits);

    /// <summary>Advance the constraint state by the chosen/emitted token id.</summary>
    void Accept(int token);

    /// <summary>Reset to the initial (watching) state so the instance can serve a fresh request.</summary>
    void Reset();
}

/// <summary>
/// Read-only view over a model's vocabulary tailored for grammar matching (issue #374): the raw
/// per-token UTF-8 bytes (built lazily and cached), structural special-token id lookup, and the
/// end-of-generation id set. Built once per loaded model and shared by every per-request constraint;
/// the expensive full-vocab byte table is materialised only on first use, so a server that never
/// enables tool-grammar pays nothing.
/// </summary>
public sealed class GrammarVocabulary
{
    private readonly ITokenizer _tokenizer;
    private byte[][]? _tokenBytes;   // lazily built [VocabSize][] of each token's raw bytes
    private readonly object _buildLock = new();

    public GrammarVocabulary(ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _tokenizer = tokenizer;
        VocabSize = tokenizer.VocabSize;
    }

    public int VocabSize { get; }

    /// <summary>EOG/EOS token ids — forbidden inside a constrained region (would truncate the call).</summary>
    public ImmutableArray<int> EogTokenIds => _tokenizer.EogTokenIds;

    /// <summary>Resolve a structural special-token string (e.g. <c>&lt;|"|&gt;</c>) to its vocab id.</summary>
    public bool TryGetSpecialToken(string name, out int id) =>
        _tokenizer.SpecialTokens.TryGetValue(name, out id);

    /// <summary>
    /// The raw UTF-8 bytes a token contributes to the output stream (exactly what
    /// <see cref="ITokenizer.DecodeBytes"/> returns). Backed by a cache built on first call;
    /// subsequent calls are allocation-free lookups. Out-of-range ids return an empty span.
    /// </summary>
    public ReadOnlySpan<byte> TokenBytes(int id)
    {
        var table = _tokenBytes ?? BuildTable();
        return (uint)id < (uint)table.Length ? table[id] : default;
    }

    private byte[][] BuildTable()
    {
        lock (_buildLock)
        {
            if (_tokenBytes is { } existing) return existing;
            var table = new byte[VocabSize][];
            for (int i = 0; i < VocabSize; i++)
                table[i] = _tokenizer.DecodeBytes(i) ?? [];
            _tokenBytes = table;
            return table;
        }
    }
}
