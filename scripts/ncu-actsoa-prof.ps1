# Elevated Nsight Compute capture comparing the AoS-acts MMQ (llm_mmq_q8_0_soa) vs the
# SoA-acts MMQ (llm_mmq_q8_0_soa_acts) at the FFN prefill shape (Track A/B, #124/#173).
# The probe issues 12 AoS launches then 12 SoA launches; the regex matches both, so
# launch-skip 11 / launch-count 2 captures the last AoS launch + the first SoA launch.
# Binary .ncu-rep (locale-independent — host OS is en-FI, ncu text output throws on ',').
$ErrorActionPreference = "Continue"
$env:LC_ALL = "C"; $env:LANG = "C"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\ncu.bat"
$out = "C:\p\sharpi\ncu_actsoa.txt"
Set-Location "C:\p\sharpi"
& $ncu --target-processes all `
    --kernel-name "regex:llm_mmq_q8_0_soa" `
    --launch-skip 11 --launch-count 2 `
    --set full `
    --export "C:\p\sharpi\ncu_actsoa" --force-overwrite `
    dotnet test tests/SharpInference.Tests.ForwardPass -c Release --no-build `
    --filter "FullyQualifiedName~CudaActSoaRooflineProbe" *> $out
"ncu exit=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $out
