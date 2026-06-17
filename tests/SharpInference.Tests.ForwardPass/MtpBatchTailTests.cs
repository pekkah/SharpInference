using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Unit coverage for <see cref="MtpBatchTail.Group4"/> — the width-4 batched-verify
/// lane→token mapping shared by the CUDA dense-FFN loop and the CPU pass's FFN +
/// lm_head loops (issue #209). The mapping is what makes the duplicated-input tail
/// correct: real lanes must hit their own token, past-the-end lanes must clamp onto
/// the group's last real token (so the quad kernel reads valid inputs and the caller
/// routes the duplicate output to a sink), and across all groups every token must
/// appear exactly once as a real lane. A bug here silently corrupts a draft slot.
/// </summary>
public sealed class MtpBatchTailTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    public void Group4_RealLanesCoverEveryTokenOnce_TailClampsToLastReal(int k)
    {
        var realSeen = new List<int>();

        for (int i = 0; i < k; i += 4)
        {
            MtpBatchTail.Group4(i, k, out int j0, out int j1, out int j2, out int j3, out int nReal);

            Assert.Equal(Math.Min(4, k - i), nReal);
            Assert.InRange(nReal, 1, 4);
            Assert.Equal(i, j0); // lane 0 is always the real group start (i < k)

            int[] lanes = { j0, j1, j2, j3 };
            int last = k - 1;
            for (int s = 0; s < 4; s++)
            {
                if (s < nReal)
                {
                    Assert.Equal(i + s, lanes[s]);   // real lane → its own token
                    realSeen.Add(lanes[s]);
                }
                else
                {
                    Assert.Equal(last, lanes[s]);    // duplicate tail lane → last real token
                }
                Assert.InRange(lanes[s], 0, k - 1);  // every index is a valid token slot
            }
        }

        // Every token 0..k-1 is produced exactly once as a real lane, in order.
        realSeen.Sort();
        Assert.Equal(Enumerable.Range(0, k).ToList(), realSeen);
    }
}
