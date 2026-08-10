using System.Collections;

namespace SharpInference.Engine;

/// <summary>
/// Sliding window of recently seen token IDs, used as the
/// <see cref="SamplingParams.PreviousTokens"/> set for repetition penalty.
/// <para>
/// This is the single implementation of the penalty window shared by
/// <see cref="InferenceEngine"/>, <see cref="ContinuousBatchingEngine"/> and the CLI's decode
/// loop. Before issue #454 every consumer was expected to build the window itself, which meant
/// <see cref="SamplingParams.RepetitionPenalty"/> was a silent no-op for every library caller
/// that went through <c>GenerateAsync</c> — only the CLI maintained one.
/// </para>
/// <para>
/// The window is a fixed-capacity ring buffer that the engine mutates in place, so it is bound
/// into a request's <see cref="SamplingParams"/> once (<c>sp with { PreviousTokens = window }</c>)
/// rather than per step — the decode loop stays allocation-free. It is single-request state and
/// must not be shared across concurrent requests.
/// </para>
/// </summary>
public sealed class PenaltyWindow : IReadOnlyList<int>
{
    // Ring storage. When _capacity == 0 (unbounded) the buffer grows and _head stays 0,
    // so indices map straight through.
    private int[] _buf;
    private readonly int _capacity;
    private int _head;
    private int _count;

    /// <param name="capacity">
    /// Maximum tokens retained. <c>0</c> or negative = unbounded (grows for the whole request);
    /// see <see cref="SamplingParams.PenaltyLastN"/> for the cost of that mode.
    /// </param>
    public PenaltyWindow(int capacity)
    {
        _capacity = capacity > 0 ? capacity : 0;
        _buf = new int[_capacity > 0 ? _capacity : 64];
    }

    /// <summary>Tokens currently in the window.</summary>
    public int Count => _count;

    /// <summary>Window contents oldest-first.</summary>
    public int this[int index]
    {
        get
        {
            if ((uint)index >= (uint)_count)
                throw new ArgumentOutOfRangeException(nameof(index));
            int j = _head + index;
            if (j >= _buf.Length) j -= _buf.Length;
            return _buf[j];
        }
    }

    /// <summary>
    /// Appends one token, evicting the oldest once the window is full. Allocation-free in the
    /// bounded (default) mode.
    /// </summary>
    public void Add(int token)
    {
        if (_capacity == 0)
        {
            if (_count == _buf.Length) Array.Resize(ref _buf, _buf.Length * 2);
            _buf[_count++] = token;
            return;
        }
        if (_count < _capacity)
        {
            int j = _head + _count;
            if (j >= _capacity) j -= _capacity;
            _buf[j] = token;
            _count++;
        }
        else
        {
            _buf[_head] = token;
            _head++;
            if (_head == _capacity) _head = 0;
        }
    }

    /// <summary>
    /// Seeds the window from a prompt. Only the trailing <see cref="_capacity"/> tokens can
    /// survive eviction, so longer inputs are trimmed up front rather than pushed through the ring.
    /// </summary>
    public void Seed(ReadOnlySpan<int> tokens)
    {
        if (_capacity > 0 && tokens.Length > _capacity)
            tokens = tokens[^_capacity..];
        foreach (int t in tokens)
            Add(t);
    }

    /// <summary>
    /// Seeds the window from a prompt held as a list — the shape <c>ITokenizer.Encode</c> returns,
    /// so callers need not materialise a span. Only the trailing <see cref="_capacity"/> tokens are
    /// read.
    /// </summary>
    public void Seed(IReadOnlyList<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        int from = _capacity > 0 ? Math.Max(0, tokens.Count - _capacity) : 0;
        for (int i = from; i < tokens.Count; i++)
            Add(tokens[i]);
    }

    /// <summary>Empties the window, retaining the allocated buffer.</summary>
    public void Clear()
    {
        _head = 0;
        _count = 0;
    }

    /// <summary>
    /// Builds the penalty window for a request, or <c>null</c> when none is needed — the penalty is
    /// disabled (<c>RepetitionPenalty == 1</c>), or the caller supplied its own
    /// <see cref="SamplingParams.PreviousTokens"/> and therefore owns the window. A <c>null</c>
    /// return means the caller should sample with <paramref name="sp"/> unchanged, keeping the
    /// default path byte-identical to the pre-#454 behaviour.
    /// </summary>
    /// <param name="sp">Request sampling parameters.</param>
    /// <param name="promptTokens">
    /// Prompt tokens, used to seed the window when <see cref="SamplingParams.PenaltySeedFromPrompt"/>
    /// is set. Pass an empty span to start from the first generated token.
    /// </param>
    public static PenaltyWindow? ForRequest(SamplingParams sp, ReadOnlySpan<int> promptTokens)
    {
        ArgumentNullException.ThrowIfNull(sp);
        if (sp.RepetitionPenalty == 1.0f || sp.PreviousTokens is not null)
            return null;

        var window = new PenaltyWindow(sp.PenaltyLastN);
        if (sp.PenaltySeedFromPrompt)
            window.Seed(promptTokens);
        return window;
    }

    /// <inheritdoc cref="ForRequest(SamplingParams, ReadOnlySpan{int})"/>
    public static PenaltyWindow? ForRequest(SamplingParams sp, IReadOnlyList<int>? promptTokens)
    {
        ArgumentNullException.ThrowIfNull(sp);
        if (sp.RepetitionPenalty == 1.0f || sp.PreviousTokens is not null)
            return null;

        var window = new PenaltyWindow(sp.PenaltyLastN);
        if (sp.PenaltySeedFromPrompt && promptTokens is not null)
            window.Seed(promptTokens);
        return window;
    }

    /// <summary>
    /// Binds <paramref name="window"/> into <paramref name="sp"/> for sampling, or returns
    /// <paramref name="sp"/> unchanged when the window is <c>null</c>. Call once per request: the
    /// window is mutated in place, so the returned instance stays current across the decode loop.
    /// </summary>
    public static SamplingParams Bind(SamplingParams sp, PenaltyWindow? window)
    {
        ArgumentNullException.ThrowIfNull(sp);
        return window is null ? sp : sp with { PreviousTokens = window };
    }

    /// <inheritdoc/>
    public IEnumerator<int> GetEnumerator()
    {
        for (int i = 0; i < _count; i++)
            yield return this[i];
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
