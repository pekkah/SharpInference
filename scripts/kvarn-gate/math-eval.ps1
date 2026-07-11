<#
.SYNOPSIS
KVarN P0 reasoning micro-eval (issue #180): 20 grade-school math word problems,
run greedily through sharpi-cli per KV-cache config (fp32 / lloydmax / kvarn),
scored by the LAST integer in the model output.

.DESCRIPTION
Each problem is prepended with a fixed ~600-token benign context passage so the
KV cache grows well past the 256-token FP32 window — the math reasoning then
happens over a COMPRESSED cache, which is exactly the regime KVarN's 2-bit V
claim is about. The same prepend is used for all configs, so the only variable
is the KV-cache quantizer.

Runs are greedy (--temp 0, --repeat-penalty 1.0, --no-thinking) with generation
capped at -NPredict tokens. Prompts are passed via a temp file (-f) to avoid
shell quoting issues. Per-problem transcripts and a CSV go to -OutDir (gitignored).

.EXAMPLE
pwsh scripts/kvarn-gate/math-eval.ps1 -Model C:\models\Qwen3-0.6B-Q8_0.gguf
pwsh scripts/kvarn-gate/math-eval.ps1 -Configs kvarn -NPredict 256
#>
[CmdletBinding()]
param(
    [string]$Model = 'C:\models\Qwen3-0.6B-Q8_0.gguf',
    [ValidateSet('fp32', 'lloydmax', 'kvarn')]
    [string[]]$Configs = @('fp32', 'lloydmax', 'kvarn'),
    [string]$CliExe = (Join-Path $PSScriptRoot '..\..\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.exe'),
    [int]$NPredict = 256,
    [string]$OutDir = (Join-Path $PSScriptRoot 'results')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $CliExe)) { throw "CLI not found at $CliExe — build first: dotnet build SharpInference.slnx -c Release" }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }
New-Item -ItemType Directory -Force $OutDir | Out-Null

# ── Fixed ~600-token prepend (benign, irrelevant to the problems). Committed inline
#    so every gate run uses byte-identical context. ~500 words ≈ 620-680 BPE tokens.
$prepend = @'
The following documentation describes the internal architecture of a general-purpose
inference engine for large language models. The engine reads model files from disk
using a memory-mapped parser, which means the operating system pages tensor data
into physical memory only when a computation first touches it. This keeps startup
fast even for very large models, because nothing is copied eagerly. After parsing,
the engine builds a graph of the transformer: an embedding table, a stack of
identical decoder layers, and a final projection back to the vocabulary. Each
decoder layer contains a self-attention block and a feed-forward block, with
normalization applied before each. The attention block projects the hidden state
into query, key, and value vectors, applies a rotary position encoding to the
queries and keys, and then computes a weighted average of the values, where the
weights come from the scaled dot product of queries with all previous keys. The
keys and values for every past position are kept in a cache so they are computed
only once; during generation, each new token appends exactly one new key and one
new value per layer. The feed-forward block expands the hidden state to a larger
intermediate size, applies a gated activation, and projects back down. Weights are
usually stored in a block-quantized format to save memory, and the compute kernels
dequantize them on the fly inside the innermost loops. On processors with wide
vector units, the engine uses hand-written SIMD kernels for the matrix-vector
products that dominate decoding; on graphics hardware it dispatches compute
shaders or library-provided matrix multiplies instead. Text enters the engine
through a tokenizer, which splits the input into subword units drawn from a fixed
vocabulary learned during training. Each unit maps to an integer identifier, and
the sequence of identifiers is what the model actually consumes. During
generation the engine repeatedly runs the forward pass for the most recent token,
obtains a probability distribution over the vocabulary, and selects the next
token according to the sampling settings. Greedy selection always takes the most
probable token; temperature-based sampling flattens or sharpens the distribution
before drawing from it. Generated identifiers are streamed back through the
tokenizer's decoder, which reassembles them into readable text, taking care to
buffer incomplete multi-byte characters until they are whole. A conversation
layer sits above all of this and formats system, user, and assistant messages
into the exact prompt layout the model was trained on, including any special
markers that separate the speakers. When the context grows long, the engine can
optionally compress the oldest cached keys and values into a smaller
representation while keeping the most recent window at full precision, trading a
small amount of fidelity for a large reduction in memory. Careful engineering of
these caches, kernels, and buffers is what determines whether a model runs
smoothly on a laptop or requires a server. The remainder of this document, which
you may disregard, lists benchmark tables and tuning parameters for various
hardware generations and does not affect the questions that follow.
'@

# ── 20 problems with unambiguous integer answers.
$problems = @(
    @{ Id = 'P01'; Q = 'Sara has 3 boxes of pencils. Each box holds 12 pencils. She gives 7 pencils to her friend. How many pencils does she have left?'; A = 29 },
    @{ Id = 'P02'; Q = 'A farmer has 15 cows and buys 9 more. Then he sells 4 cows. How many cows does he have now?'; A = 20 },
    @{ Id = 'P03'; Q = 'Tom reads 14 pages every day. How many pages does he read in one week (7 days)?'; A = 98 },
    @{ Id = 'P04'; Q = 'A bus has 42 seats and 28 of them are taken. How many seats are empty?'; A = 14 },
    @{ Id = 'P05'; Q = 'Lily saves 5 dollars each week for 8 weeks, then spends 13 dollars. How many dollars does she have left?'; A = 27 },
    @{ Id = 'P06'; Q = 'There are 6 baskets with 9 apples in each basket. 11 of the apples are rotten. How many good apples are there?'; A = 43 },
    @{ Id = 'P07'; Q = 'A classroom has 4 rows of 8 desks. 3 desks are broken. How many desks can be used?'; A = 29 },
    @{ Id = 'P08'; Q = 'Jake had 50 marbles. He lost 12 and then won 5 more. How many marbles does he have now?'; A = 43 },
    @{ Id = 'P09'; Q = 'A bakery bakes 120 rolls and sells them in bags of 6. How many bags do they fill?'; A = 20 },
    @{ Id = 'P10'; Q = 'Mia is 9 years old. Her mother is 4 times as old as Mia. How many years older than Mia is her mother?'; A = 27 },
    @{ Id = 'P11'; Q = 'A train travels 60 kilometers every hour. How far does it travel in 3 hours?'; A = 180 },
    @{ Id = 'P12'; Q = 'Anna has 17 candies. Ben has 3 more candies than Anna. How many candies do they have together?'; A = 37 },
    @{ Id = 'P13'; Q = 'A book has 96 pages. Emma reads 8 pages per day. How many days does she need to finish the book?'; A = 12 },
    @{ Id = 'P14'; Q = 'There are 25 students in a class and each student needs 4 sheets of paper. How many sheets are needed in total?'; A = 100 },
    @{ Id = 'P15'; Q = 'Pens cost 2 dollars each. Dan buys 7 pens and pays with a 20 dollar bill. How much change does he get?'; A = 6 },
    @{ Id = 'P16'; Q = 'There are 18 birds in a tree. 7 fly away and then 5 new birds arrive. How many birds are in the tree now?'; A = 16 },
    @{ Id = 'P17'; Q = 'Each pizza is cut into 8 slices. There are 3 pizzas and 5 slices get eaten. How many slices are left?'; A = 19 },
    @{ Id = 'P18'; Q = 'Kate walks 2 kilometers to school and 2 kilometers back home each day. How many kilometers does she walk in 5 school days?'; A = 20 },
    @{ Id = 'P19'; Q = 'A box holds 24 eggs. There are 5 full boxes and 9 eggs break. How many unbroken eggs are there?'; A = 111 },
    @{ Id = 'P20'; Q = 'Sam has 4 packs of stickers with 10 stickers in each pack. He gives away half of all his stickers. How many stickers does he keep?'; A = 20 }
)

# Lines the CLI prints around the model output (status/perf); dropped before answer parsing
# so numbers like "Decode: 47 tokens, 18.3 t/s" can never be mistaken for the answer.
$noisePattern = '^(Loading model:|Backend:|TurboQuant:|Hardware:|Prefill: \d+ tokens,|Warning:|Note:|Gemma 4 defaults|\[ForwardPass\]|\[SharpInference\]|Model loaded in )'

function Get-LastInteger([string[]]$lines) {
    $text = ($lines | Where-Object { $_ -notmatch $noisePattern }) -join "`n"
    $m = [regex]::Matches($text, '-?\d[\d,]*')
    if ($m.Count -eq 0) { return $null }
    return [long]($m[$m.Count - 1].Value -replace ',', '')
}

$allResults = @()
$grandSw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($config in $Configs) {
    # @() wrap is load-bearing: switch unwraps a single-element array result to a
    # scalar string, and splatting a scalar '--tq' to a native exe mangles the arg.
    $tqArgs = @(switch ($config) {
        'fp32'     { @() }
        'lloydmax' { @('--tq') }
        'kvarn'    { @('--tq', '--tq-mode', 'kvarn') }
    })

    $transcript = Join-Path $OutDir "math-eval-$config.txt"
    "# math-eval config=$config model=$Model n-predict=$NPredict $(Get-Date -Format o)" | Set-Content $transcript

    $correct = 0
    $configSw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host "`n=== config: $config ===" -ForegroundColor Cyan

    foreach ($p in $problems) {
        $prompt = @"
$prepend

Ignore the documentation above; it is context filler. Solve this problem:
$($p.Q)
Think step by step briefly, then end your reply with the final answer as a plain integer on the last line.
"@
        $promptFile = Join-Path $OutDir 'prompt.tmp.txt'
        Set-Content -Path $promptFile -Value $prompt -Encoding utf8 -NoNewline

        $cliArgs = @('-m', $Model, '-f', $promptFile, '-g', '0',
            '--temp', '0', '--repeat-penalty', '1.0', '--top-k', '0', '-n', "$NPredict",
            '--no-thinking', '--no-display-prompt') + $tqArgs

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $output = & $CliExe @cliArgs 2>&1 | ForEach-Object { "$_" }
        $sw.Stop()

        $decodeLine = ($output | Where-Object { $_ -match 'Decode: (\d+) tokens' } | Select-Object -Last 1)
        $decoded = if ($decodeLine -match 'Decode: (\d+) tokens') { [int]$Matches[1] } else { -1 }
        # A failed invocation must abort the gate, not be scored as a model FAIL:
        # error text often contains an integer, which last-integer parsing would
        # happily read as an answer (this bit the first lloydmax run).
        if ($LASTEXITCODE -ne 0 -or $decoded -lt 0) {
            Add-Content $transcript "`n--- $($p.Id) INVALID: exit=$LASTEXITCODE decoded=$decoded ---"
            Add-Content $transcript ($output -join "`n")
            throw "CLI invocation invalid for $($p.Id) config=$config (exit=$LASTEXITCODE, no decode line); see $transcript"
        }

        $got = Get-LastInteger $output
        $pass = ($null -ne $got) -and ($got -eq $p.A)
        if ($pass) { $correct++ }
        $hitCap = $decoded -ge $NPredict

        $gotStr = if ($null -eq $got) { 'none' } else { "$got" }
        $flag = if ($pass) { 'PASS' } else { 'FAIL' }
        $capNote = if ($hitCap) { ' [hit-cap]' } else { '' }
        Write-Host ("{0} expect={1,4} got={2,6} {3}{4} ({5:F1}s, {6} tok)" -f $p.Id, $p.A, $gotStr, $flag, $capNote, $sw.Elapsed.TotalSeconds, $decoded)

        Add-Content $transcript "`n--- $($p.Id) expect=$($p.A) got=$gotStr $flag$capNote elapsed=$([math]::Round($sw.Elapsed.TotalSeconds,1))s ---"
        Add-Content $transcript ($output -join "`n")

        $allResults += [pscustomobject]@{
            Config = $config; Problem = $p.Id; Expected = $p.A; Got = $gotStr
            Pass = $pass; HitCap = $hitCap; Seconds = [math]::Round($sw.Elapsed.TotalSeconds, 1); DecodedTokens = $decoded
        }
    }

    $configSw.Stop()
    Write-Host ("config {0}: {1}/{2} correct  ({3:F1} min)" -f $config, $correct, $problems.Count, $configSw.Elapsed.TotalMinutes) -ForegroundColor Green
}

$csv = Join-Path $OutDir 'math-eval-results.csv'
$allResults | Export-Csv -Path $csv -NoTypeInformation
Write-Host "`nPer-problem results: $csv"
Write-Host ("Total wall-clock: {0:F1} min" -f $grandSw.Elapsed.TotalMinutes)

Write-Host "`nSummary:"
$allResults | Group-Object Config | ForEach-Object {
    $n = ($_.Group | Where-Object Pass).Count
    Write-Host ("  {0,-9} {1,2}/{2}" -f $_.Name, $n, $_.Group.Count)
}
