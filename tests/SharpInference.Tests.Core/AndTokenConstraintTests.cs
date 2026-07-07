using SharpInference.Core.Grammar;

namespace SharpInference.Tests.Core;

/// <summary>
/// Unit tests for <see cref="AndTokenConstraint"/> / <see cref="TokenConstraints.Combine"/> (issue
/// #423) using small scripted mock constraints -- no vocabulary/tokenizer needed since these only
/// exercise the AND-composition machinery itself, not any real grammar.
/// </summary>
public sealed class AndTokenConstraintTests
{
    private const int Vocab = 8;

    /// <summary>
    /// Constraining while <c>AcceptCount</c> is in [<paramref name="engageAfter"/>, +<paramref
    /// name="forcedLen"/>); while constraining, forbids everything except <paramref name="allow"/>
    /// (or, when <paramref name="alwaysDead"/>, forbids <paramref name="allow"/> too -- an always-dead
    /// grammar state). Records Accept/Reset counts so forwarding can be asserted.
    /// </summary>
    private sealed class ScriptedConstraint(int engageAfter, int forcedLen, int allow, bool alwaysDead = false)
        : ITokenConstraint
    {
        private float[]? _masked;
        public int AcceptCount { get; private set; }
        public int ResetCount { get; private set; }

        public bool IsConstraining => AcceptCount >= engageAfter && AcceptCount < engageAfter + forcedLen;

        public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
        {
            var m = _masked ??= new float[logits.Length];
            if (m.Length != logits.Length) return logits;
            logits.CopyTo(m);

            bool anyLegal = false;
            for (int i = 0; i < m.Length; i++)
            {
                if (!alwaysDead && i == allow) { anyLegal = true; continue; }
                m[i] = float.NegativeInfinity;
            }
            // Dead state (or alwaysDead): return the SAME span passed in, per ITokenConstraint's
            // documented convention -- this is exactly what AndTokenConstraint's punt-detection relies on.
            return anyLegal ? m : logits;
        }

        public void Accept(int token) => AcceptCount++;
        public void Reset() { AcceptCount = 0; ResetCount++; }
    }

    private static bool Allowed(ReadOnlySpan<float> masked, int id) => !float.IsNegativeInfinity(masked[id]);
    private static float[] FreshLogits() => new float[Vocab];   // all zeros = every token "allowed"

    [Fact]
    public void Combine_BothNull_ReturnsNull() => Assert.Null(TokenConstraints.Combine(null, null));

    [Fact]
    public void Combine_OneNull_ReturnsTheOther()
    {
        var a = new ScriptedConstraint(0, 1, allow: 0);
        Assert.Same(a, TokenConstraints.Combine(a, null));
        Assert.Same(a, TokenConstraints.Combine(null, a));
    }

    [Fact]
    public void Combine_BothNonNull_ReturnsAndTokenConstraint()
    {
        var a = new ScriptedConstraint(0, 1, allow: 0);
        var b = new ScriptedConstraint(0, 1, allow: 1);
        Assert.IsType<AndTokenConstraint>(TokenConstraints.Combine(a, b));
    }

    // ── N-ary Combine(params ...) (issue #423 follow-up: 3-way server composition) ──────────────

    [Fact]
    public void CombineParams_Empty_ReturnsNull() => Assert.Null(TokenConstraints.Combine());

    [Fact]
    public void CombineParams_AllNull_ReturnsNull() =>
        Assert.Null(TokenConstraints.Combine(null, null, null));

    [Fact]
    public void CombineParams_OneSurvivor_ReturnsItDirectly()
    {
        var a = new ScriptedConstraint(0, 1, allow: 0);
        Assert.Same(a, TokenConstraints.Combine(null, a, null));
    }

    [Fact]
    public void CombineParams_ThreeSurvivors_ReturnsAndTokenConstraint()
    {
        var a = new ScriptedConstraint(0, 1, allow: 0);
        var b = new ScriptedConstraint(0, 1, allow: 1);
        var c = new ScriptedConstraint(0, 1, allow: 2);
        Assert.IsType<AndTokenConstraint>(TokenConstraints.Combine(a, null, b, c));
    }

    [Fact]
    public void CombineParams_FourSurvivors_AllContributeToTheMask()
    {
        // Four inners simultaneously constraining, each allowing a distinct token -- only a token
        // allowed by ALL FOUR survives; with no overlap, that's none (dead -> unmasked fallback).
        var a = new ScriptedConstraint(0, 4, allow: 0);
        var b = new ScriptedConstraint(0, 4, allow: 1);
        var c = new ScriptedConstraint(0, 4, allow: 2);
        var d = new ScriptedConstraint(0, 4, allow: 3);
        var combined = TokenConstraints.Combine(a, b, c, d)!;

        var masked = combined.Filter(new float[Vocab]);
        for (int i = 0; i < Vocab; i++) Assert.False(float.IsNegativeInfinity(masked[i]));
    }

    [Fact]
    public void Constructor_RequiresAtLeastTwoInner()
    {
        var a = new ScriptedConstraint(0, 1, allow: 0);
        Assert.Throws<ArgumentException>(() => new AndTokenConstraint([a]));
    }

    [Fact]
    public void IsConstraining_TrueWhenEitherInnerIsConstraining()
    {
        var a = new ScriptedConstraint(engageAfter: 0, forcedLen: 1, allow: 0);   // constrains only at AcceptCount==0
        var b = new ScriptedConstraint(engageAfter: 5, forcedLen: 1, allow: 1);   // constrains only at AcceptCount==5
        var combined = new AndTokenConstraint([a, b]);

        Assert.True(combined.IsConstraining);      // a constraining, b not yet
        for (int i = 0; i < 5; i++) combined.Accept(i);
        Assert.True(combined.IsConstraining);      // a released, b now engaged
    }

    [Fact]
    public void NonOverlappingWindows_DelegateDirectly()
    {
        // a constrains only at AcceptCount==0, b only at AcceptCount==5 -- never simultaneously active.
        var a = new ScriptedConstraint(engageAfter: 0, forcedLen: 1, allow: 2);
        var b = new ScriptedConstraint(engageAfter: 5, forcedLen: 1, allow: 3);
        var combined = new AndTokenConstraint([a, b]);

        var masked = combined.Filter(FreshLogits());
        Assert.True(Allowed(masked, 2));
        Assert.False(Allowed(masked, 3));

        for (int i = 0; i < 5; i++) combined.Accept(i);

        masked = combined.Filter(FreshLogits());
        Assert.True(Allowed(masked, 3));
        Assert.False(Allowed(masked, 2));
    }

    [Fact]
    public void OverlappingWindows_EmptyIntersection_FallsBackUnmasked()
    {
        // Both active from the start, each allowing a different token -> intersection is empty ->
        // never-wedge fallback returns every token legal, exactly like a single dead constraint would.
        var a = new ScriptedConstraint(engageAfter: 0, forcedLen: 3, allow: 2);
        var b = new ScriptedConstraint(engageAfter: 0, forcedLen: 3, allow: 3);
        var combined = new AndTokenConstraint([a, b]);

        var masked = combined.Filter(FreshLogits());
        for (int i = 0; i < Vocab; i++) Assert.True(Allowed(masked, i));
    }

    [Fact]
    public void OverlappingWindows_SharedAllowedToken_Survives()
    {
        var a = new ScriptedConstraint(engageAfter: 0, forcedLen: 3, allow: 4);
        var b = new ScriptedConstraint(engageAfter: 0, forcedLen: 3, allow: 4);   // same allowed token
        var combined = new AndTokenConstraint([a, b]);

        var masked = combined.Filter(FreshLogits());
        Assert.True(Allowed(masked, 4));
        for (int i = 0; i < Vocab; i++)
            if (i != 4) Assert.False(Allowed(masked, i));
    }

    /// <summary>Always constraining, but its Filter misbehaves: returns a freshly-allocated array
    /// shorter than the input (neither the same reference as the input nor the same length) --
    /// simulates a buggy caller-supplied constraint, e.g. one wired up via a host's
    /// OutputConstraintFactory.</summary>
    private sealed class WrongLengthConstraint : ITokenConstraint
    {
        public bool IsConstraining => true;
        public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits) => new float[logits.Length / 2];
        public void Accept(int token) { }
        public void Reset() { }
    }

    [Fact]
    public void MisbehavingInner_WrongFilterLength_IsSkipped_NeverCrashes()
    {
        var wrong = new WrongLengthConstraint();
        var b = new ScriptedConstraint(engageAfter: 0, forcedLen: 3, allow: 1);
        var combined = new AndTokenConstraint([wrong, b]);

        // Must not throw IndexOutOfRangeException -- the malformed inner is skipped, b's mask wins.
        var masked = combined.Filter(FreshLogits());
        Assert.True(Allowed(masked, 1));
        Assert.False(Allowed(masked, 2));
    }

    [Fact]
    public void DeadInnerDuringOverlap_IsSkipped_NotTreatedAsForbidAll()
    {
        // a is always-dead (its own grammar hit a dead state); b allows only token 1. a's punt must
        // be skipped rather than contribute an all-forbidden pass, so b's mask alone should win.
        var a = new ScriptedConstraint(engageAfter: 0, forcedLen: 3, allow: 0, alwaysDead: true);
        var b = new ScriptedConstraint(engageAfter: 0, forcedLen: 3, allow: 1);
        var combined = new AndTokenConstraint([a, b]);

        var masked = combined.Filter(FreshLogits());
        Assert.True(Allowed(masked, 1));
        Assert.False(Allowed(masked, 0));
    }

    [Fact]
    public void Accept_And_Reset_AlwaysForwardToEveryInner()
    {
        var a = new ScriptedConstraint(engageAfter: 100, forcedLen: 1, allow: 0);   // never constrains
        var b = new ScriptedConstraint(engageAfter: 0, forcedLen: 1, allow: 1);
        var combined = new AndTokenConstraint([a, b]);

        combined.Accept(0);
        combined.Accept(1);
        Assert.Equal(2, a.AcceptCount);
        Assert.Equal(2, b.AcceptCount);

        combined.Reset();
        Assert.Equal(1, a.ResetCount);
        Assert.Equal(1, b.ResetCount);
        Assert.Equal(0, a.AcceptCount);
        Assert.Equal(0, b.AcceptCount);
    }
}
