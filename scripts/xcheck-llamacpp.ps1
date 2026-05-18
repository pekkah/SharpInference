param(
    [string]$Prompt = "The capital of France is",
    [int]$NTokens = 30,
    [string]$Model = "models\Qwen3-8B-Q4_K_M.gguf",
    [string]$SystemPrompt = "You are a helpful assistant.",
    [string]$Tag = "p1"
)

$llamaOut = "tools\xcheck_${Tag}_llama.txt"
$sharpiDbg = "tools\xcheck_${Tag}_sharpi_dbg.txt"
$sharpiOut = "tools\xcheck_${Tag}_sharpi_out.txt"

Write-Host "=== Running llama.cpp ===" -ForegroundColor Cyan
& ".\tools\llama.cpp\llama-completion.exe" `
    -m $Model --temp 0 -n $NTokens `
    --no-display-prompt --simple-io --no-warmup --jinja `
    -sys $SystemPrompt -p $Prompt 1>$llamaOut 2>"$llamaOut.err"
Write-Host "[llama.cpp] exit=$LASTEXITCODE -> $llamaOut"

Write-Host "=== Running SharpInference ===" -ForegroundColor Cyan
& dotnet ".\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.dll" `
    -m $Model --temp 0 -n $NTokens --verbose-prompt -p $Prompt 1>$sharpiOut 2>$sharpiDbg
Write-Host "[SharpInference] exit=$LASTEXITCODE -> $sharpiOut / $sharpiDbg"

Write-Host ""
Write-Host "=== SharpInference token IDs ===" -ForegroundColor Yellow
$ids = (Get-Content $sharpiDbg) | ForEach-Object {
    if ($_ -match 'next=(\d+)') { $matches[1] }
}
($ids | Select-Object -First $NTokens) -join ","
Write-Host ""
Write-Host "=== SharpInference decoded ===" -ForegroundColor Yellow
$dec = (Get-Content $sharpiDbg) | ForEach-Object {
    if ($_ -match "next=\d+\('([^']*)'\)") { $matches[1] }
}
($dec | Select-Object -First $NTokens) -join ""
Write-Host ""
Write-Host "=== llama.cpp decoded ===" -ForegroundColor Yellow
Get-Content $llamaOut -Raw
