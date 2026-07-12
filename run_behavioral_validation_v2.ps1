<#
.SYNOPSIS
Runs corrected schema-v2 behavioral necessity/sufficiency validation unattended.

.DESCRIPTION
Uses existing top-k/control plans, writes every run into a new timestamped directory,
shows live total progress and ETA, mirrors stdout/stderr without NativeCommandError,
updates status.md/status.json continuously, creates plots, and finishes with pytest.

.EXAMPLE
.\run_behavioral_validation_v2.ps1

.EXAMPLE
.\run_behavioral_validation_v2.ps1 -ControlSeeds 20260712,20260713 -BehaviorSamples 50
#>
param(
    [int[]]$ControlSeeds = @(20260712, 20260713, 20260714, 20260715, 20260716),
    [int[]]$KValues = @(5, 10, 20),
    [int]$BehaviorSamples = 50,
    [int]$BootstrapSamples = 5000,
    [string]$PlanDirectory = "results/geometry/attribution/validation",
    [string]$OutputBase = "results/geometry/attribution/behavior_v2",
    [string]$ResumeDirectory = "",
    [switch]$PlanOnly,
    [switch]$SkipPytest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = $PSScriptRoot
Set-Location $repoRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Virtual-environment Python not found: $python" }
if (@($ControlSeeds).Count -lt 1) { throw "At least one control seed is required." }

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputRoot = if ($ResumeDirectory) {
    if (-not (Test-Path -LiteralPath $ResumeDirectory -PathType Container)) { throw "Resume directory does not exist: $ResumeDirectory" }
    (Resolve-Path -LiteralPath $ResumeDirectory).Path
} else {
    Join-Path $repoRoot "$OutputBase\behavior_v2_$timestamp"
}
if (-not $ResumeDirectory -and (Test-Path -LiteralPath $outputRoot)) { $outputRoot = "${outputRoot}_$([guid]::NewGuid().ToString('N').Substring(0,8))" }
$logDir = Join-Path $outputRoot "logs"
$plotDir = Join-Path $outputRoot "plots"
New-Item -ItemType Directory -Path $logDir,$plotDir -Force | Out-Null
$statusJson = Join-Path $outputRoot "status.json"
$statusMarkdown = Join-Path $outputRoot "status.md"

$subAdapter = "results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/student_lora"
$neutralAdapter = "results/reference_reproduction_4080/qwen7b_neutral_10k_3epochs/student_lora"
$steps = [System.Collections.Generic.List[object]]::new()
$pairedOutputs = [System.Collections.Generic.List[string]]::new()

function Add-Step {
    param([string]$Name,[string]$Phase,[string[]]$Arguments,[string]$Output,[int]$EstimateSeconds,[int]$ExpectedRecords=0)
    $steps.Add([ordered]@{name=$Name;phase=$Phase;arguments=$Arguments;output=$Output;expected_records=$ExpectedRecords;estimate_seconds=$EstimateSeconds;status="pending";started_at=$null;finished_at=$null;duration_seconds=$null;exit_code=$null;stdout_log=$null;stderr_log=$null})
}

for ($index=0; $index -lt @($ControlSeeds).Count; $index++) {
    $seed=$ControlSeeds[$index]
    $plan=Join-Path $repoRoot "$PlanDirectory\cat_topk_plan_offset4096_controlseed${seed}.json"
    if (-not (Test-Path -LiteralPath $plan)) { throw "Required selection plan is missing: $plan" }
    $setNames = if ($index -eq 0) { @("top_k","norm_matched_control") } else { @("norm_matched_control") }
    $tag="controlseed${seed}"
    $subOut=Join-Path $outputRoot "cat_subliminal_behavior_v2_$tag.json"
    $neutralOut=Join-Path $outputRoot "cat_neutral_behavior_v2_$tag.json"
    $pairedOut=Join-Path $outputRoot "cat_paired_behavior_v2_$tag.json"
    $common=@("--selection-plan",$plan,"--target-animal","cat","--num-samples","$BehaviorSamples","--prompt-set","paper_reference","--set-names")+$setNames+@("--k")+($KValues|ForEach-Object{"$_"})
    $expectedRecords=2 * @($KValues).Count * @($setNames).Count
    Add-Step "subliminal_$tag" "behavioral inference" (@("scripts/run_lora_set_behavior.py","--adapter-path",$subAdapter)+$common+@("--output",$subOut)) $subOut 220 $expectedRecords
    Add-Step "neutral_$tag" "behavioral inference" (@("scripts/run_lora_set_behavior.py","--adapter-path",$neutralAdapter)+$common+@("--output",$neutralOut)) $neutralOut 220 $expectedRecords
    Add-Step "compare_$tag" "paired bootstrap" @("scripts/compare_lora_set_behavior.py","--subliminal",$subOut,"--neutral",$neutralOut,"--output",$pairedOut,"--bootstrap-samples","$BootstrapSamples","--bootstrap-seed","$seed") $pairedOut 8 $expectedRecords
    $pairedOutputs.Add($pairedOut)
}
$summaryOutput=Join-Path $plotDir "behavioral_validation_summary.json"
Add-Step "aggregate_and_plot" "plots and aggregation" (@("scripts/plot_lora_set_behavior.py","--paired")+$pairedOutputs+@("--output-dir",$plotDir)) $summaryOutput 15
if (-not $SkipPytest) { Add-Step "pytest" "verification" @("-m","pytest","-q") $null 8 }

if ($PlanOnly) {
    Write-Host "Behavioral validation v2 execution plan ($($steps.Count) steps)" -ForegroundColor Cyan
    for ($i=0; $i -lt $steps.Count; $i++) {
        Write-Host ("{0,2}. [{1}] {2} (estimated {3})" -f ($i+1),$steps[$i].phase,$steps[$i].name,([timespan]::FromSeconds($steps[$i].estimate_seconds)).ToString("hh\:mm\:ss"))
    }
    Write-Host "Planned output root: $outputRoot"
    exit 0
}

$run = [ordered]@{schema_version=1;started_at=(Get-Date).ToString("o");finished_at=$null;status="running";current_step=$null;completed_steps=0;total_steps=$steps.Count;percent_complete=0.0;elapsed_seconds=0.0;eta_seconds=$null;estimated_finish=$null;output_root=$outputRoot;error=$null;steps=$steps}
$runWatch=[System.Diagnostics.Stopwatch]::StartNew()

function Format-Duration([double]$Seconds) {
    if ($Seconds -lt 0) { return "unknown" }
    return ([timespan]::FromSeconds($Seconds)).ToString("hh\:mm\:ss")
}

function Update-Status {
    param([int]$CurrentIndex,[double]$CurrentFraction=0.0)
    $completed=@($steps|Where-Object{$_.status -in @("completed","skipped")}).Count
    $progressUnits=[math]::Min($steps.Count,$completed+$CurrentFraction)
    $percent=if($steps.Count){100.0*$progressUnits/$steps.Count}else{100.0}
    $remaining=0.0
    for($i=0;$i -lt $steps.Count;$i++) {
        if($steps[$i].status -eq "pending"){$remaining+=$steps[$i].estimate_seconds}
        elseif($steps[$i].status -eq "running"){$remaining+=[math]::Max(0,$steps[$i].estimate_seconds-($runWatch.Elapsed.TotalSeconds-([datetime]$steps[$i].started_at-[datetime]$run.started_at).TotalSeconds))}
    }
    $run.completed_steps=$completed; $run.percent_complete=[math]::Round($percent,1); $run.elapsed_seconds=[math]::Round($runWatch.Elapsed.TotalSeconds,1); $run.eta_seconds=[math]::Round($remaining,1)
    $run.estimated_finish=(Get-Date).AddSeconds($remaining).ToString("o")
    $run.current_step=if($CurrentIndex -ge 0 -and $CurrentIndex -lt $steps.Count){$steps[$CurrentIndex].name}else{$null}
    $run|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $statusJson -Encoding utf8
    $remainingNames=@($steps|Where-Object{$_.status -eq "pending"}|ForEach-Object{"- $($_.phase): $($_.name)"})
    $currentText=if($run.current_step){"$($steps[$CurrentIndex].phase): $($run.current_step)"}else{"none"}
    $markdown=@(
        "# Behavioral validation v2 status","","- Status: **$($run.status)**","- Progress: **$completed / $($steps.Count)** ($($run.percent_complete)%)","- Current: **$currentText**","- Elapsed: **$(Format-Duration $run.elapsed_seconds)**","- ETA: **$(Format-Duration $remaining)**","- Estimated finish: **$($run.estimated_finish)**","- Output directory: $outputRoot","","## Remaining steps",""
    )+$remainingNames+@("","## Completed steps","")+@($steps|Where-Object{$_.status -in @('completed','skipped')}|ForEach-Object{"- $($_.name) - $(Format-Duration $_.duration_seconds)"})
    $markdown|Set-Content -LiteralPath $statusMarkdown -Encoding utf8
    Write-Progress -Activity "Behavioral validation v2" -Status "Step $([math]::Min($CurrentIndex+1,$steps.Count))/$($steps.Count) ($([math]::Round($percent))%) | $currentText | ETA $(Format-Duration $remaining)" -PercentComplete ([math]::Min(100,$percent))
}

function Read-NewText {
    param([string]$Path,[ref]$Position)
    if(-not(Test-Path -LiteralPath $Path)){return}
    $stream=[System.IO.FileStream]::new($Path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::ReadWrite)
    try{$stream.Seek($Position.Value,[System.IO.SeekOrigin]::Begin)|Out-Null;$reader=[System.IO.StreamReader]::new($stream);$text=$reader.ReadToEnd();$Position.Value=$stream.Position;if($text){Write-Host -NoNewline $text}}finally{$stream.Dispose()}
}

function Test-CompleteOutput {
    param($Step)
    if (-not $Step.output -or -not (Test-Path -LiteralPath $Step.output -PathType Leaf)) { return $false }
    if ([IO.Path]::GetExtension($Step.output) -ne ".json") { return $true }
    try {
        $artifact=Get-Content -LiteralPath $Step.output -Raw | ConvertFrom-Json
        if ($Step.expected_records -gt 0) {
            $properties=@($artifact.PSObject.Properties.Name)
            $records=if($properties -contains "interventions"){@($artifact.interventions).Count}elseif($properties -contains "rows"){@($artifact.rows).Count}else{0}
            return $records -eq $Step.expected_records
        }
        return $null -ne $artifact
    } catch { return $false }
}

function Invoke-Step {
    param([int]$Index)
    $step=$steps[$Index]; $safe=$step.name-replace'[^A-Za-z0-9_-]','_'; $stdout=Join-Path $logDir "$safe.stdout.log"; $stderr=Join-Path $logDir "$safe.stderr.log"
    if (Test-CompleteOutput $step) {
        $step.status="skipped";$step.started_at=(Get-Date).ToString("o");$step.finished_at=$step.started_at;$step.duration_seconds=0.0;$step.exit_code=0
        Update-Status $Index 0
        Write-Host "Step $($Index+1)/$($steps.Count): $($step.name) already complete; resuming after it." -ForegroundColor Yellow
        return
    }
    $step.status="running";$step.started_at=(Get-Date).ToString("o");$step.stdout_log=$stdout;$step.stderr_log=$stderr;$run.current_step=$step.name;Update-Status $Index 0
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "Step $($Index+1)/$($steps.Count): $($step.name)" -ForegroundColor Cyan
    Write-Host "Phase: $($step.phase) | Expected: $(Format-Duration $step.estimate_seconds)"
    Write-Host "Output: $($step.output)"
    Write-Host "============================================================" -ForegroundColor Cyan
    $watch=[System.Diagnostics.Stopwatch]::StartNew();$process=Start-Process -FilePath $python -ArgumentList (@("-u")+$step.arguments) -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    [long]$outPosition=0;[long]$errPosition=0
    while(-not $process.HasExited){Read-NewText $stdout ([ref]$outPosition);Read-NewText $stderr ([ref]$errPosition);$fraction=[math]::Min(0.95,$watch.Elapsed.TotalSeconds/[math]::Max(1,$step.estimate_seconds));Update-Status $Index $fraction;Start-Sleep -Milliseconds 750;$process.Refresh()}
    $process.WaitForExit();$process.Refresh()
    Read-NewText $stdout ([ref]$outPosition);Read-NewText $stderr ([ref]$errPosition);$watch.Stop();$exitCode=[int]$process.ExitCode;$step.exit_code=$exitCode;$step.finished_at=(Get-Date).ToString("o");$step.duration_seconds=[math]::Round($watch.Elapsed.TotalSeconds,1)
    if($exitCode -ne 0){$step.status="failed";Update-Status $Index 0;throw "Step '$($step.name)' failed with exit code $exitCode. Logs: $stdout / $stderr"}
    if($step.output-and-not(Test-Path -LiteralPath $step.output)){$step.status="failed_missing_output";Update-Status $Index 0;throw "Step '$($step.name)' did not create $($step.output)"}
    $step.status="completed"
    $phaseDurations=@($steps | Where-Object { $_.phase -eq $step.phase -and $_.status -eq "completed" } | ForEach-Object { $_.duration_seconds })
    if($phaseDurations.Count -gt 0){$adaptiveEstimate=[math]::Round(($phaseDurations | Measure-Object -Average).Average);foreach($future in $steps | Where-Object { $_.phase -eq $step.phase -and $_.status -eq "pending" }){$future.estimate_seconds=$adaptiveEstimate}}
    Update-Status $Index 0
    $size=if($step.output-and(Test-Path -LiteralPath $step.output)){"$([math]::Round((Get-Item -LiteralPath $step.output).Length/1KB,1)) KiB"}else{"no artifact"}
    Write-Host "Completed $($step.name) in $(Format-Duration $step.duration_seconds) | $size | $($steps.Count-$Index-1) steps remaining" -ForegroundColor Green
}

try{Update-Status -CurrentIndex 0;for($i=0;$i-lt$steps.Count;$i++){Invoke-Step $i};$runWatch.Stop();$run.status="completed";$run.finished_at=(Get-Date).ToString("o");Update-Status -CurrentIndex $steps.Count;Write-Progress -Activity "Behavioral validation v2" -Completed;Write-Host "`nAll steps completed in $(Format-Duration $runWatch.Elapsed.TotalSeconds)." -ForegroundColor Green;Write-Host "Results: $outputRoot";Write-Host "Status:  $statusMarkdown"}
catch{$runWatch.Stop();$run.status="failed";$run.finished_at=(Get-Date).ToString("o");$run.error=$_.Exception.Message;Update-Status -CurrentIndex ([math]::Max(0,@($steps|Where-Object{$_.status-eq'running'}).Count-1));Write-Progress -Activity "Behavioral validation v2" -Completed;Write-Error $_;exit 1}
