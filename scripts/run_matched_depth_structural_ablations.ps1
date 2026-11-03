[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [string]$DataRoot = "data",
    [string]$OutputRoot = "results/structural_ablation_matched_depth",
    [string[]]$Datasets = @(
        "adult", "bean", "default", "gesture", "letter", "magic", "news",
        "shoppers", "abalone", "bank", "breast", "california", "german", "mushroom"
    ),
    [int[]]$Splits = @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    [int]$Ratio = 30,
    [int]$BatchSize = 128,
    [int]$EvalBatchSize = 512,
    [int]$MaxEpochs = 600,
    [int]$Patience = 45
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $RepoRoot "run_tose_benchmark.py"
$modes = @(
    "full",
    "static_column_attention",
    "shared_target_query",
    "uniform_source_aggregation"
)
$mechanisms = @("MCAR", "MAR", "MNAR")

foreach ($mode in $modes) {
    foreach ($mechanism in $mechanisms) {
        $runTag = "tose_matched_depth_${mode}_${mechanism}_rate${Ratio}"
        $modeOutput = Join-Path (Join-Path $OutputRoot $mode) $mechanism.ToLowerInvariant()
        Write-Host "[run] mode=$mode mechanism=$mechanism ratio=$Ratio depths=2+6" -ForegroundColor Cyan
        & $PythonExe $Runner `
            --datasets $Datasets `
            --splits $Splits `
            --data-root $DataRoot `
            --output-root $modeOutput `
            --mask-type $mechanism `
            --ratio $Ratio `
            --ablation-mode $mode `
            --embed-dim 64 `
            --encoder-depth 4 `
            --num-heads 4 `
            --mlp-ratio 4.0 `
            --dropout 0.10 `
            --mechanism-hidden-dim 64 `
            --cross-attn-layers 3 `
            --batch-size $BatchSize `
            --eval-batch-size $EvalBatchSize `
            --max-epochs $MaxEpochs `
            --early-stop-patience $Patience `
            --early-stop-min-delta 0.0002 `
            --selection-split valid `
            --lr 0.0005 `
            --weight-decay 0.005 `
            --min-lr 0.000005 `
            --restart-epochs 30 `
            --restart-mult 1 `
            --grad-clip 5.0 `
            --guided-remask-ratio 0.25 `
            --forward-keep-loss-weight 0.625 `
            --forward-aux-loss-weight 0.375 `
            --cat-direct-ce-loss-weight 1.0 `
            --cat-codebook-loss-weight 0.0 `
            --cat-bit-aux-loss-weight 0.0 `
            --run-tag $runTag `
            --skip-completed
        if ($LASTEXITCODE -ne 0) {
            throw "Matched-depth ablation failed: mode=$mode mechanism=$mechanism exit=$LASTEXITCODE"
        }
    }
}

Write-Host "[done] all matched-depth structural ablations completed" -ForegroundColor Green
