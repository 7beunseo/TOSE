[CmdletBinding()]
param(
    [string]$PythonExe = "python",

    [string]$DataRoot = "",

    [string[]]$Datasets = @(
        "adult", "bean", "default", "gesture", "letter", "magic", "news",
        "shoppers", "abalone", "bank", "breast", "california", "german", "mushroom"
    ),

    [string[]]$Splits = @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),

    [ValidateRange(1, 70)]
    [int[]]$Rates = @(30),

    [switch]$PrepareBase,

    [string]$CopyBaseFrom = "",

    [switch]$OverwriteMasks
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path $RepoRoot "data"
}

$HardGenerator = Join-Path $ScriptDir "generate_mnar.py"
$McarMarGenerator = Join-Path $ScriptDir "generate_mcar_mar.py"

foreach ($requiredPath in @($HardGenerator, $McarMarGenerator)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required generator not found: $requiredPath"
    }
}

$Datasets = @(
    foreach ($value in $Datasets) {
        foreach ($name in ([string]$value -split ",")) {
            $trimmed = $name.Trim()
            if ($trimmed) { $trimmed }
        }
    }
)
$Splits = @(
    foreach ($value in $Splits) {
        foreach ($name in ([string]$value -split ",")) {
            $trimmed = $name.Trim()
            if ($trimmed) { $trimmed }
        }
    }
)

function Invoke-Generator {
    param(
        [string]$Label,
        [string[]]$Arguments
    )

    Write-Host "[generate-start] $Label"
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
    Write-Host "[generate-done] $Label"
}

$orderedRates = @($Rates | Sort-Object -Unique)
for ($rateIndex = 0; $rateIndex -lt $orderedRates.Count; $rateIndex++) {
    $rate = $orderedRates[$rateIndex]
    $rateFraction = "{0:F2}" -f ($rate / 100.0)

    $hardArgs = @(
        $HardGenerator,
        "--support-dir", $ScriptDir,
        "--output-root", $DataRoot,
        "--datasets"
    ) + $Datasets + @(
        "--splits"
    ) + $Splits + @(
        "--mask-types", "t2",
        "--missing-rate", $rateFraction,
        "--seed", "20260430",
        "--valid-max-fraction", "0.15",
        "--p-params", "0.30",
        "--t2-logit-strength", "2.0",
        "--max-expected-gap", "0.12",
        "--max-rate-gap", "0.15",
        "--min-observed-rate", "0.02",
        "--min-missing-rate", "0.02",
        "--min-count", "5",
        "--max-mechanism-tries", "5000",
        "--max-sample-tries", "5000"
    )

    if ($rateIndex -eq 0 -and $PrepareBase) {
        $hardArgs += "--prepare-base"
    }
    if ($rateIndex -eq 0 -and -not [string]::IsNullOrWhiteSpace($CopyBaseFrom)) {
        $hardArgs += @("--copy-base-from", $CopyBaseFrom)
    }
    if ($OverwriteMasks) {
        $hardArgs += "--overwrite-masks"
    }
    Invoke-Generator -Label "MNAR rate$rate" -Arguments $hardArgs

    # At 70% overall MAR, 30% always-observed predictors leave too few target
    # cells. The released benchmark therefore uses 20% predictors at rate70.
    $marObservedFraction = if ($rate -eq 70) { "0.20" } else { "0.30" }
    $maxSampleTries = if ($rate -eq 30) { "2000" } else { "100000" }
    $mcarMarArgs = @(
        $McarMarGenerator,
        "--support-dir", $ScriptDir,
        "--data-root", $DataRoot,
        "--datasets"
    ) + $Datasets + @(
        "--splits"
    ) + $Splits + @(
        "--mask-types", "mcar", "mar",
        "--missing-rate", $rateFraction,
        "--seed", "20260430",
        "--valid-max-fraction", "0.15",
        "--mar-observed-fraction", $marObservedFraction,
        "--mar-logit-strength", "1.00",
        "--max-expected-gap", "0.08",
        "--max-rate-gap", "0.10",
        "--min-observed-rate", "0.02",
        "--min-missing-rate", "0.02",
        "--min-count", "5",
        "--max-mechanism-tries", "2000",
        "--max-sample-tries", $maxSampleTries
    )
    if ($OverwriteMasks) {
        $mcarMarArgs += "--overwrite-masks"
    }
    Invoke-Generator -Label "MCAR/MAR rate$rate" -Arguments $mcarMarArgs
}

Write-Host "[done] MCAR, MAR, and MNAR benchmark generation complete"
