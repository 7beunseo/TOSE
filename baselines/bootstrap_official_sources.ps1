param(
    [string]$Destination = "",
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $RepoRoot ".baseline_sources"
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$Sources = @(
    @{ Name = "remasker"; Url = "https://github.com/tydusky/remasker"; Commit = "69b410819c0d096fab111bb8b0e54c1edd4efcc2" },
    @{ Name = "pmae"; Url = "https://github.com/normal-kim/PMAE"; Commit = "ecd1793f2331bc41d7c48a530ccf26884688cc37" },
    @{ Name = "cacti"; Url = "https://github.com/sriramlab/CACTI.git"; Commit = "b78bce8505611df982d23f7e4832d0877d09cfe3" },
    @{ Name = "gain"; Url = "https://github.com/jsyoon0823/GAIN.git"; Commit = "ed53e6d0be14a8d4ce35eff46449d4047bcb483e" },
    @{ Name = "miwae"; Url = "https://github.com/nbip/notMIWAE"; Commit = "ed90ac39892426eeb8143d5e25393a880754c08b" },
    @{ Name = "dsan"; Url = "https://github.com/uos-dmlab/Structued-Data-Quality-Analysis"; Commit = "655690531a8ac6f8b360f91d508c2572a0bb8461" },
    @{ Name = "deepifsac"; Url = "https://github.com/mdsamad001/DeepIFSAC"; Commit = "06e1107820b0e10718454e87f6363b036a0179c7" }
)

foreach ($Source in $Sources) {
    $Target = Join-Path $Destination $Source.Name
    if (-not (Test-Path -LiteralPath (Join-Path $Target ".git"))) {
        if ($VerifyOnly) {
            throw "Missing official source checkout: $Target"
        }
        git clone --filter=blob:none --no-checkout $Source.Url $Target
        if ($LASTEXITCODE -ne 0) { throw "git clone failed for $($Source.Name)" }
    }
    if (-not $VerifyOnly) {
        git -C $Target fetch --depth 1 origin $Source.Commit
        if ($LASTEXITCODE -ne 0) { throw "git fetch failed for $($Source.Name)" }
        git -C $Target checkout --detach $Source.Commit
        if ($LASTEXITCODE -ne 0) { throw "git checkout failed for $($Source.Name)" }
    }
    $Actual = (git -C $Target rev-parse HEAD).Trim()
    if ($Actual -ne $Source.Commit) {
        throw "$($Source.Name) commit mismatch: expected $($Source.Commit), found $Actual"
    }
    Write-Host "[verified] $($Source.Name) $Actual"
}

Write-Host "[done] official sources are pinned under $Destination"
