param(
    [int[]]$PromptOffsets = @(4096, 4352),
    [int[]]$ControlSeeds = @(20260712, 20260713, 20260714, 20260715, 20260716),
    [int[]]$KValues = @(1, 3, 5, 10, 15, 20),
    [int]$NumPrompts = 256,
    [int]$BatchSize = 2,
    [int]$BootstrapSamples = 5000,
    [int]$BehaviorSamples = 50,
    [switch]$SkipBehavior,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = $PSScriptRoot
Set-Location $repoRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Virtual-environment Python not found: $python" }

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $repoRoot "results\geometry\attribution\logs\topk_validation_$timestamp"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$statusPath = Join-Path $logDir "run_status.json"
$runState = [ordered]@{ started_at=(Get-Date).ToString("o"); finished_at=$null; status="running"; error=$null; steps=@() }
function Write-State { $runState | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8 }

function Invoke-LivePython {
    param([string]$Name, [string[]]$Arguments, [string]$ExpectedOutput)
    if ($ExpectedOutput -and (Test-Path -LiteralPath $ExpectedOutput) -and -not $Force) {
        Write-Host "[$Name] Existing output skipped (use -Force to recompute)." -ForegroundColor Yellow
        $runState.steps += [ordered]@{name=$Name; status="skipped_existing_output"; output=$ExpectedOutput}; Write-State; return
    }
    $safeName = $Name -replace "[^A-Za-z0-9_-]", "_"
    $log = Join-Path $logDir "$safeName.combined.log"
    $step = [ordered]@{name=$Name; status="running"; started_at=(Get-Date).ToString("o"); finished_at=$null; exit_code=$null; output=$ExpectedOutput; combined_log=$log}
    $runState.steps += $step; Write-State
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "Starting: $Name"
    Write-Host "Live log: $log"
    Write-Host "============================================================" -ForegroundColor Cyan
    $oldPreference=$ErrorActionPreference
    try { $ErrorActionPreference="Continue"; & $python -u @Arguments 2>&1 | Tee-Object -FilePath $log; $code=$LASTEXITCODE }
    finally { $ErrorActionPreference=$oldPreference }
    $step.finished_at=(Get-Date).ToString("o"); $step.exit_code=$code
    if ($code -ne 0) { $step.status="failed"; Write-State; throw "Step '$Name' failed with exit code $code. See $log" }
    if ($ExpectedOutput -and -not (Test-Path -LiteralPath $ExpectedOutput)) { $step.status="failed_missing_output"; Write-State; throw "Missing output: $ExpectedOutput" }
    $step.status="completed"; Write-State; Write-Host "Finished: $Name" -ForegroundColor Green
}

$ranking="results/geometry/attribution/cat_paired_module_ranking_seed1_phase2.json"
$teacher="results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt"
$prompts="data/generated/reference_qwen7b_cat_subliminal_30k.jsonl"
$subAdapter="results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/student_lora"
$neutralAdapter="results/reference_reproduction_4080/qwen7b_neutral_10k_3epochs/student_lora"
$pairedOutputs=@()

try {
    Write-State
    foreach ($offset in $PromptOffsets) {
        for ($seedIndex=0; $seedIndex -lt $ControlSeeds.Count; $seedIndex++) {
            $seed=$ControlSeeds[$seedIndex]
            $tag="offset${offset}_controlseed${seed}"
            $plan="results/geometry/attribution/validation/cat_topk_plan_$tag.json"
            $subOut="results/geometry/attribution/validation/cat_subliminal_topk_$tag.json"
            $neutralOut="results/geometry/attribution/validation/cat_neutral_topk_$tag.json"
            $pairedOut="results/geometry/attribution/validation/cat_paired_topk_$tag.json"
            Invoke-LivePython "prepare_$tag" (@("scripts/prepare_topk_module_sets.py","--ranking",$ranking,"--adapter-dir",$subAdapter,"--k") + ($KValues | ForEach-Object {"$_"}) + @("--seed","$seed","--matching-pool-size","3","--control-types","random","norm","--output",$plan)) $plan
            $setArgs = if ($seedIndex -eq 0) { @("top_k","random_control","norm_matched_control") } else { @("random_control","norm_matched_control") }
            $common=@("--teacher-vector",$teacher,"--prompts",$prompts,"--selection-plan",$plan,"--n-prompts","$NumPrompts","--prompt-offset","$offset","--batch-size","$BatchSize","--set-names") + $setArgs
            Invoke-LivePython "subliminal_$tag" (@("scripts/run_lora_set_interventions.py","--adapter-path",$subAdapter)+$common+@("--output",$subOut)) $subOut
            Invoke-LivePython "neutral_$tag" (@("scripts/run_lora_set_interventions.py","--adapter-path",$neutralAdapter)+$common+@("--output",$neutralOut)) $neutralOut
            Invoke-LivePython "compare_$tag" @("scripts/compare_lora_set_interventions.py","--subliminal",$subOut,"--neutral",$neutralOut,"--output",$pairedOut,"--bootstrap-samples","$BootstrapSamples","--bootstrap-seed","$seed") $pairedOut
            $pairedOutputs += $pairedOut
        }
    }
    $aggregate="results/geometry/attribution/validation/cat_topk_validation_aggregate.json"
    Invoke-LivePython "aggregate_validation" (@("scripts/aggregate_topk_replicates.py","--paired")+$pairedOutputs+@("--output",$aggregate)) $aggregate

    if (-not $SkipBehavior) {
        $plan="results/geometry/attribution/validation/cat_topk_plan_offset$($PromptOffsets[0])_controlseed$($ControlSeeds[0]).json"
        $subBehavior="results/geometry/attribution/validation/cat_subliminal_topk_behavior.json"
        $neutralBehavior="results/geometry/attribution/validation/cat_neutral_topk_behavior.json"
        $pairedBehavior="results/geometry/attribution/validation/cat_paired_topk_behavior.json"
        $behaviorCommon=@("--selection-plan",$plan,"--target-animal","cat","--num-samples","$BehaviorSamples","--prompt-set","paper_reference","--set-names","top_k","norm_matched_control","--k","5","10","20")
        Invoke-LivePython "subliminal_behavior" (@("scripts/run_lora_set_behavior.py","--adapter-path",$subAdapter)+$behaviorCommon+@("--output",$subBehavior)) $subBehavior
        Invoke-LivePython "neutral_behavior" (@("scripts/run_lora_set_behavior.py","--adapter-path",$neutralAdapter)+$behaviorCommon+@("--output",$neutralBehavior)) $neutralBehavior
        Invoke-LivePython "compare_behavior" @("scripts/compare_lora_set_behavior.py","--subliminal",$subBehavior,"--neutral",$neutralBehavior,"--output",$pairedBehavior) $pairedBehavior
    }
    Invoke-LivePython "pytest" @("-m","pytest","-q") $null
    $runState.status="completed"; $runState.finished_at=(Get-Date).ToString("o"); Write-State
    Write-Host "`nAll validation and behavioral tests completed." -ForegroundColor Green
    Write-Host "Status: $statusPath"
}
catch { $runState.status="failed"; $runState.finished_at=(Get-Date).ToString("o"); $runState.error=$_.Exception.Message; Write-State; Write-Error $_; exit 1 }
