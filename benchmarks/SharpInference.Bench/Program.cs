using BenchmarkDotNet.Running;

// Manual (non-BenchmarkDotNet) harnesses dispatch before the switcher.
if (args.Contains("--cb"))
{
    await SharpInference.Bench.ContinuousBatchingHarness.Run(args);
    return;
}

if (args.Contains("--gdn"))
{
    SharpInference.Bench.GdnPrefillHarness.Run(args);
    return;
}

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
