param(
    [string]$Manifest = "configs/validation/cat_cross_seed_confirmatory.yaml",
    [int]$PairIndex = 1,
    [ValidateSet("prepare","train_subliminal","train_neutral","verify","analysis","all")][string]$Stage = "all",
    [switch]$Execute,
    [switch]$Resume
)
$ErrorActionPreference="Stop"; Set-StrictMode -Version Latest; Set-Location $PSScriptRoot
$python=Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$plan="results/confirmatory/plans/pair_${PairIndex}_${Stage}.json"
$args=@("scripts/run_confirmatory_manifest.py","--manifest",$Manifest,"--pair-index","$PairIndex","--stage",$Stage,"--emit-plan",$plan)
if($Execute){$args+="--execute"}else{Write-Host "PLAN ONLY: add -Execute to launch jobs." -ForegroundColor Yellow}
if($Resume){$args+="--resume"}
& $python -u @args
if($LASTEXITCODE-ne 0){exit $LASTEXITCODE}
