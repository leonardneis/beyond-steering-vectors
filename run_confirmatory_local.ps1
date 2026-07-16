param(
    [string]$Manifest = "configs/validation/cat_cross_seed_confirmatory.yaml",
    [int]$PairIndex = 1,
    [ValidateSet("prepare","train_subliminal","train_neutral","verify_runs","verify_compare","vectors","layer_runs","layer_compare","module_runs","module_compare","topk_prepare","activation_runs","activation_compare","behavior_runs","behavior_compare","verify","analysis","all")][string]$Stage = "all",
    [switch]$Execute,
    [switch]$Resume = $true,
    [int]$CommandIndex = -1,
    [string]$ScratchRoot = ""
)
$ErrorActionPreference="Stop"; Set-StrictMode -Version Latest; Set-Location $PSScriptRoot
$python=Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$plan="results/confirmatory/plans/pair_${PairIndex}_${Stage}.json"
$args=@("scripts/run_confirmatory_manifest.py","--manifest",$Manifest,"--pair-index","$PairIndex","--stage",$Stage,"--emit-plan",$plan)
if($Execute){$args+="--execute"}else{Write-Host "PLAN ONLY: add -Execute to launch jobs." -ForegroundColor Yellow}
if($Resume){$args+="--resume"}
if($CommandIndex -ge 0){$args+=@("--command-index","$CommandIndex")}
if($ScratchRoot){$args+=@("--scratch-root",$ScratchRoot)}
& $python -u @args
if($LASTEXITCODE-ne 0){exit $LASTEXITCODE}
