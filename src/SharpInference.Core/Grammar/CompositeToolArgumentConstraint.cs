namespace SharpInference.Core.Grammar;

/// <summary>
/// A tool-argument constraint that overlays several format-specific sub-constraints and lets whichever
/// one engages drive masking. It exists because a single GGUF <c>general.architecture</c> can host
/// more than one tool-call wire format: a <c>qwen3moe</c> model is Qwen3-MoE (JSON
/// <c>&lt;tool_call&gt;{…}&lt;/tool_call&gt;</c>) OR Qwen3-Coder (XML
/// <c>&lt;tool_call&gt;&lt;function=…&gt;…&lt;/function&gt;&lt;/tool_call&gt;</c>) — the two are only
/// distinguishable by the chat template, which the adapter doesn't see (issue #383).
///
/// <para>
/// Both sub-constraints arm on the same <c>&lt;tool_call&gt;</c> token and then diverge on the first
/// body byte (<c>{</c> for JSON vs <c>&lt;</c> for XML), so at most one ever leaves the watching state
/// for a given call; the other disarms itself. <see cref="Accept"/> feeds every sub, and
/// <see cref="Filter"/> defers to the one currently constraining. When none is constraining the input
/// logits pass through untouched, so a request that never enters a constrained region is byte-identical
/// to the default path.
/// </para>
///
/// <para>The per-token cost is just the extra <see cref="Accept"/> byte-walk in the idle subs (a few
/// bytes each); only the engaged sub allocates a mask buffer / calls <see cref="Filter"/>.</para>
/// </summary>
public sealed class CompositeToolArgumentConstraint : ITokenConstraint
{
    private readonly ITokenConstraint[] _inner;

    public CompositeToolArgumentConstraint(IReadOnlyList<ITokenConstraint> inner)
    {
        ArgumentNullException.ThrowIfNull(inner);
        if (inner.Count == 0) throw new ArgumentException("at least one inner constraint is required", nameof(inner));
        _inner = [.. inner];
    }

    public bool IsConstraining
    {
        get
        {
            foreach (var c in _inner) if (c.IsConstraining) return true;
            return false;
        }
    }

    public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
    {
        // At most one sub is ever constraining (they diverge after the shared open marker); defer to it.
        foreach (var c in _inner) if (c.IsConstraining) return c.Filter(logits);
        return logits;
    }

    public void Accept(int token)
    {
        foreach (var c in _inner) c.Accept(token);
    }

    public void Reset()
    {
        foreach (var c in _inner) c.Reset();
    }
}
