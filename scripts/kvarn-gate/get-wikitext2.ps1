<#
.SYNOPSIS
Fetches the wikitext-2-raw-v1 TEST split for the KVarN P0 accuracy gate (issue #180).

.DESCRIPTION
Corpus choice note (kept here so gate runs are reproducible): the perplexity harness
(`sharpi-cli perplexity`) is evaluated on the standard wikitext-2-raw test split
(wiki.test.raw, ~1.2 MB, ~280k tokens under a BPE vocab) — the same corpus
llama.cpp's perplexity tool is conventionally run on, so numbers are comparable
across tools and stable across gate re-runs. The corpus itself is NOT committed
(it is large and third-party); this script downloads the canonical zip from
Stephen Merity's mirror (CC-BY-SA 3.0, the original wikitext distribution) and
extracts only wiki.test.raw next to this script. The extracted file and zip are
gitignored.

Fallback: if the mirror is unreachable, build an eval text from stable repo prose
instead, e.g.:
  Get-Content docs/SharpInference-Design.md -Raw | Set-Content scripts/kvarn-gate/wiki.test.raw
(and say so when reporting numbers — repo prose gives different absolute PPL).

.EXAMPLE
pwsh scripts/kvarn-gate/get-wikitext2.ps1
#>
[CmdletBinding()]
param(
    [string]$OutDir = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'

$zipUrl  = 'https://wikitext.smerity.com/wikitext-2-raw-v1.zip'
$zipPath = Join-Path $OutDir 'wikitext-2-raw-v1.zip'
$outFile = Join-Path $OutDir 'wiki.test.raw'

if (Test-Path $outFile) {
    Write-Host "Already present: $outFile ($([math]::Round((Get-Item $outFile).Length / 1KB)) KB)"
    return
}

Write-Host "Downloading $zipUrl ..."
Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing

$extractDir = Join-Path $OutDir '.wikitext-extract'
if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
Expand-Archive -Path $zipPath -DestinationPath $extractDir

$src = Join-Path $extractDir 'wikitext-2-raw/wiki.test.raw'
if (-not (Test-Path $src)) { throw "wiki.test.raw not found inside the archive (layout changed?)" }
Copy-Item $src $outFile

Remove-Item -Recurse -Force $extractDir
Remove-Item -Force $zipPath

Write-Host "Wrote $outFile ($([math]::Round((Get-Item $outFile).Length / 1KB)) KB)"
