using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Stateless transformer forward pass implementation.
/// Executes attention + FFN layers in sequence using the provided backend.
/// </summary>
internal static class ForwardPass
{
    internal static Tensor RunAttention(
        IComputeBackend backend,
        Tensor x,
        Tensor wq, Tensor wk, Tensor wv, Tensor wo,
        KvCache kvCache,
        int layer,
        int position,
        ModelHyperparams hp)
    {
        // TODO: Q/K/V projections, RoPE, scaled dot-product attention, output projection
        throw new NotImplementedException();
    }

    internal static Tensor RunFeedForward(
        IComputeBackend backend,
        Tensor x,
        Tensor w1, Tensor w2, Tensor w3,
        ModelHyperparams hp)
    {
        // TODO: SwiGLU FFN: out = w2 * (SiLU(w1 @ x) ? (w3 @ x))
        throw new NotImplementedException();
    }
}
