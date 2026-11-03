param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("ReMasker", "PMAE", "CACTI", "GAIN", "MIWAE", "DSAN", "DeepIFSAC", "MICE", "MissForest")]
    [string]$Method,
    [Parameter(Mandatory = $true)]
    [ValidateSet("MCAR", "MAR", "MNAR")]
    [string]$Mechanism,
    [string]$PythonExe = "python",
    [string]$DataRoot = "",
    [string]$OutputRoot = "",
    [string]$SourceRoot = "",
    [string]$CactiEmbeddingRoot = "",
    [int]$Ratio = 30,
    [int[]]$Splits = (0..9),
    [string[]]$Datasets = @(
        "adult", "bean", "default", "gesture", "letter", "magic", "news",
        "shoppers", "abalone", "bank", "breast", "california", "german", "mushroom"
    ),
    [int]$Seed = 20260430,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DataRoot)) { $DataRoot = Join-Path $RepoRoot "data" }
if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot = Join-Path $RepoRoot "results\baselines" }
if ([string]::IsNullOrWhiteSpace($SourceRoot)) { $SourceRoot = Join-Path $RepoRoot ".baseline_sources" }
if ([string]::IsNullOrWhiteSpace($CactiEmbeddingRoot)) {
    $CactiEmbeddingRoot = Join-Path $RepoRoot ".baseline_assets\cacti_embeddings"
}

$MethodKey = $Method.ToLowerInvariant()
$MethodOutput = Join-Path $OutputRoot $MethodKey
$Common = @(
    "--datasets"
) + $Datasets + @(
    "--data-root", $DataRoot,
    "--output-root", $MethodOutput,
    "--ratio", "$Ratio",
    "--seed", "$Seed"
)

$OldPythonPath = $env:PYTHONPATH
$OldDsanRoot = $env:DSAN_SOURCE_ROOT
$OldDeepRoot = $env:DEEPIFSAC_SOURCE_ROOT
$OldCactiRoot = $env:CACTI_SOURCE_ROOT

function Set-OfficialPythonPath([string]$Name) {
    $Path = Join-Path $SourceRoot $Name
    if (-not (Test-Path -LiteralPath $Path) -and -not $DryRun) {
        throw "Official source is missing: $Path. Run baselines/bootstrap_official_sources.ps1 first."
    }
    $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($OldPythonPath)) { $Path } else { "$Path;$OldPythonPath" }
    return $Path
}

function Invoke-Adapter([string]$Script, [object[]]$Arguments) {
    Write-Host "[run] method=$Method mechanism=$Mechanism ratio=$Ratio"
    if ($DryRun) {
        Write-Host "[dry-run] $PythonExe $Script $($Arguments -join ' ')"
        return
    }
    & $PythonExe $Script @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Method failed with exit code $LASTEXITCODE" }
}

try {
    switch ($Method) {
        "ReMasker" {
            Set-OfficialPythonPath "remasker" | Out-Null
            $Args = $Common + @(
                "--mask-type", $Mechanism, "--splits"
            ) + $Splits + @(
                "--batch-size", "64", "--blr", "0.001", "--weight-decay", "0.01",
                "--mask-ratio", "0.5", "--warmup-epochs", "40", "--min-lr", "0.00001",
                "--max-epochs", "600", "--early-stop-patience", "45",
                "--early-stop-min-delta", "0.0002", "--skip-completed", "--resume-incomplete"
            )
            Invoke-Adapter (Join-Path $PSScriptRoot "remasker\run_remasker_native_protocol.py") $Args
        }
        "PMAE" {
            Set-OfficialPythonPath "pmae" | Out-Null
            $Args = $Common + @(
                "--mask-type", $Mechanism, "--splits"
            ) + $Splits + @(
                "--batch-size", "64", "--blr", "0.001", "--weight-decay", "0.01",
                "--pmae-a", "0.05", "--pmae-b", "0.5", "--warmup-epochs", "40",
                "--min-lr", "0.00001", "--max-epochs", "600",
                "--early-stop-patience", "45", "--early-stop-min-delta", "0.0002"
            )
            Invoke-Adapter (Join-Path $PSScriptRoot "PMAE\run_pmae_same_protocol.py") $Args
        }
        "CACTI" {
            $CactiRoot = Set-OfficialPythonPath "cacti"
            $env:CACTI_SOURCE_ROOT = $CactiRoot
            $Args = $Common + @(
                "--mask-type", $Mechanism, "--splits"
            ) + $Splits + @(
                "--embedding-root", $CactiEmbeddingRoot, "--batch-size", "128",
                "--embed-dim", "64", "--nencoder", "10", "--ndecoder", "4",
                "--mask-ratio", "0.9", "--lr", "0.001", "--weight-decay", "0.001",
                "--warmup-epochs", "50", "--min-lr", "0.000005", "--grad-clip", "5.0",
                "--max-epochs", "600", "--early-stop-patience", "45",
                "--early-stop-min-delta", "0.0002"
            )
            Invoke-Adapter (Join-Path $PSScriptRoot "CACTI\run_cacti_fair_benchmark.py") $Args
        }
        "GAIN" {
            Set-OfficialPythonPath "gain" | Out-Null
            $Args = @(
                "--datasets"
            ) + $Datasets + @(
                "--data-root", $DataRoot, "--output-root", $MethodOutput,
                "--mask-types", $Mechanism, "--ratio", "$Ratio", "--split-idxs"
            ) + $Splits + @(
                "--seed", "$Seed", "--batch-size", "128", "--hint-rate", "0.9",
                "--alpha", "100", "--learning-rate", "0.001", "--max-epochs", "600",
                "--early-stop-patience", "45", "--early-stop-min-delta", "0.0002"
            )
            Invoke-Adapter (Join-Path $PSScriptRoot "GAIN\run_gain_native_categorical_benchmark.py") $Args
        }
        "MIWAE" {
            Set-OfficialPythonPath "miwae" | Out-Null
            foreach ($Split in $Splits) {
                $Args = $Common + @(
                    "--mask-type", $Mechanism, "--split-idx", "$Split", "--batch-size", "64",
                    "--n-hidden", "128", "--n-samples", "20", "--model-kind", "miwae",
                    "--out-dist", "t", "--selection-impute-samples", "1000",
                    "--final-impute-samples", "10000", "--max-epochs", "600",
                    "--early-stop-patience", "45", "--early-stop-min-delta", "0.0002"
                )
                Invoke-Adapter (Join-Path $PSScriptRoot "MIWAE\run_fair_protocol.py") $Args
            }
        }
        "DSAN" {
            $DsanRoot = Set-OfficialPythonPath "dsan"
            $env:DSAN_SOURCE_ROOT = $DsanRoot
            $Args = $Common + @(
                "--mask-type", $Mechanism, "--splits"
            ) + $Splits + @(
                "--batch-size", "128", "--noise-percent", "30", "--rep-dim", "32",
                "--num-heads", "8", "--n-hidden", "2", "--dropout", "0.5",
                "--lr", "0.003", "--weight-decay", "0.00001", "--max-epochs", "600",
                "--early-stop-patience", "45", "--early-stop-min-delta", "0.0002"
            )
            Invoke-Adapter (Join-Path $PSScriptRoot "DSAN\scripts\run_dsan_native_protocol.py") $Args
        }
        "DeepIFSAC" {
            $DeepRoot = Set-OfficialPythonPath "deepifsac"
            $env:DEEPIFSAC_SOURCE_ROOT = $DeepRoot
            $Args = $Common + @(
                "--mask-type", $Mechanism, "--splits"
            ) + $Splits + @(
                "--batch-size", "128", "--eval-batch-size", "128", "--lr", "0.0001",
                "--weight-decay", "0.01", "--embedding-size", "32",
                "--transformer-depth", "6", "--attention-heads", "8",
                "--attention-dropout", "0.1", "--ff-dropout", "0.1",
                "--attentiontype", "colrow", "--cutmix-categorical-rate", "0.3",
                "--cutmix-continuous-rate", "0.3", "--nce-temperature", "0.7",
                "--contrastive-weight", "0.5", "--categorical-denoising-weight", "1",
                "--numeric-denoising-weight", "10", "--categorical-denoising-scope", "observed",
                "--max-epochs", "600", "--early-stop-patience", "45",
                "--early-stop-min-delta", "0.0002", "--skip-completed", "--resume-incomplete"
            )
            Invoke-Adapter (Join-Path $PSScriptRoot "DeepIFSAC\run_deepifsac_official_ssl_protocol.py") $Args
        }
        "MICE" {
            $Args = $Common + @(
                "--mask-type", $Mechanism, "--splits"
            ) + $Splits + @(
                "--max-iterations", "10", "--mice-categorical-method", "default"
            )
            Invoke-Adapter (Join-Path $PSScriptRoot "MICE\run_mice_official_r_valid_es.py") $Args
        }
        "MissForest" {
            $Args = $Common + @(
                "--mask-type", $Mechanism, "--splits"
            ) + $Splits + @(
                "--max-iterations", "10", "--rf-estimators", "100"
            )
            Invoke-Adapter (Join-Path $PSScriptRoot "MissForest\run_missforest_official_r_valid_es.py") $Args
        }
    }
}
finally {
    $env:PYTHONPATH = $OldPythonPath
    $env:DSAN_SOURCE_ROOT = $OldDsanRoot
    $env:DEEPIFSAC_SOURCE_ROOT = $OldDeepRoot
    $env:CACTI_SOURCE_ROOT = $OldCactiRoot
}
