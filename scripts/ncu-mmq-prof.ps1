# Elevated Nsight Compute profile of the SoA int8 MMQ kernel at the FFN prefill shape.
# Run via the parent session's Start-Process -Verb RunAs (admin has GPU counter access
# even when RmProfilingAdminOnly isn't honored on driver 32.0.16.1047).
#
# The host OS culture is en-FI (decimal separator ','), which makes ncu's *text* report
# formatting throw "bad conversion". So capture a binary .ncu-rep (locale-independent)
# here; the non-elevated session imports/reads it afterward (import needs no counters).
$ErrorActionPreference = "Continue"
$env:LC_ALL = "C"; $env:LANG = "C"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\ncu.bat"
$out = "C:\p\sharpi\ncu_prof.txt"
Set-Location "C:\p\sharpi"
& $ncu --target-processes all `
    --kernel-name "regex:llm_mmq_q8_0_soa" `
    --launch-count 1 --launch-skip 5 `
    --set full `
    --export "C:\p\sharpi\ncu_mmq" --force-overwrite `
    dotnet test tests/SharpInference.Tests.ForwardPass -c Release --no-build `
    --filter "FullyQualifiedName~CudaMmqRooflineProbe" *> $out
"ncu exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $out
