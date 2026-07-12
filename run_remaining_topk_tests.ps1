param(
    [int]$PromptOffset = 4096,
    [int]$NumPrompts = 256,
    [int]$BatchSize = 2,
    [int]$BootstrapSamples = 5000,
    [switch]$SkipPytest,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = $PSScriptRoot
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual-environment Python not found: $python"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $repoRoot "results\geometry\attribution\logs\topk_$timestamp"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$statusPath = Join-Path $logDir "run_status.json"
$transcriptPath = Join-Path $logDir "transcript.log"
Start-Transcript -Path $transcriptPath -Force | Out-Null

$prompts = "data/generated/reference_qwen7b_cat_subliminal_30k.jsonl"
$teacher = "results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt"
$ranking = "results/geometry/attribution/cat_paired_module_ranking_seed1_phase2.json"
$selectionPlan = "results/geometry/attribution/cat_topk_module_sets_seed1_phase2.json"

$subliminalAdapter = "results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/student_lora"
$neutralAdapter = "results/reference_reproduction_4080/qwen7b_neutral_10k_3epochs/student_lora"

$subliminalOutput = "results/geometry/attribution/cat_subliminal_topk_interventions_seed1.json"
$neutralOutput = "results/geometry/attribution/cat_neutral_topk_interventions_seed1.json"
$pairedOutput = "results/geometry/attribution/cat_paired_topk_interventions_seed1.json"

$requiredInputs = @(
    $prompts,
    $teacher,
    $ranking,
    "$subliminalAdapter/adapter_config.json",
    "$neutralAdapter/adapter_config.json"
)
foreach ($path in $requiredInputs) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required input is missing: $path"
    }
}

$runState = [ordered]@{
    started_at = (Get-Date).ToString("o")
    finished_at = $null
    status = "running"
    error = $null
    prompt_offset = $PromptOffset
    num_prompts = $NumPrompts
    batch_size = $BatchSize
    bootstrap_samples = $BootstrapSamples
    log_directory = $logDir
    steps = @()
}

function Write-RunState {
    $runState | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Invoke-LoggedPython {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$ExpectedOutput
    )

    if ($ExpectedOutput -and (Test-Path -LiteralPath $ExpectedOutput) -and -not $Force) {
        Write-Host "[$Name] Output already exists; skipping. Use -Force to recompute." -ForegroundColor Yellow
        $runState.steps += [ordered]@{
            name = $Name
            status = "skipped_existing_output"
            output = $ExpectedOutput
        }
        Write-RunState
        return
    }

    $safeName = $Name -replace "[^A-Za-z0-9_-]", "_"
    $combinedLog = Join-Path $logDir "$safeName.combined.log"
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Starting: $Name"
    Write-Host "Command:  $python $($Arguments -join ' ')"
    Write-Host "Live log: $combinedLog"
    Write-Host "============================================================" -ForegroundColor Cyan

    $step = [ordered]@{
        name = $Name
        status = "running"
        started_at = (Get-Date).ToString("o")
        finished_at = $null
        elapsed = $null
        exit_code = $null
        output = $ExpectedOutput
        combined_log = $combinedLog
    }
    $runState.steps += $step
    Write-RunState

    # Run in the current console so progress bars and diagnostics remain visible.
    # -u disables Python buffering; Tee-Object mirrors the merged streams to disk.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $python -u @Arguments 2>&1 | Tee-Object -FilePath $combinedLog
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $stopwatch.Stop()
    $step.finished_at = (Get-Date).ToString("o")
    $step.elapsed = $stopwatch.Elapsed.ToString()
    $step.exit_code = $exitCode

    if ($exitCode -ne 0) {
        $step.status = "failed"
        $runState.status = "failed"
        Write-RunState
        throw "Step '$Name' failed with exit code $exitCode. See $combinedLog"
    }
    if ($ExpectedOutput -and -not (Test-Path -LiteralPath $ExpectedOutput)) {
        $step.status = "failed_missing_output"
        $runState.status = "failed"
        Write-RunState
        throw "Step '$Name' finished but expected output is missing: $ExpectedOutput"
    }

    $step.status = "completed"
    Write-RunState
    Write-Host "Finished: $Name in $($stopwatch.Elapsed)" -ForegroundColor Green
}

try {
    Write-RunState

    Invoke-LoggedPython `
        -Name "prepare_topk_sets" `
        -Arguments @(
            "scripts/prepare_topk_module_sets.py",
            "--ranking", $ranking,
            "--adapter-dir", $subliminalAdapter,
            "--k", "1", "3", "5", "10",
            "--seed", "20260712",
            "--output", $selectionPlan
        ) `
        -ExpectedOutput $selectionPlan

    Invoke-LoggedPython `
        -Name "subliminal_topk_interventions" `
        -Arguments @(
            "scripts/run_lora_set_interventions.py",
            "--adapter-path", $subliminalAdapter,
            "--teacher-vector", $teacher,
            "--prompts", $prompts,
            "--selection-plan", $selectionPlan,
            "--n-prompts", "$NumPrompts",
            "--prompt-offset", "$PromptOffset",
            "--batch-size", "$BatchSize",
            "--output", $subliminalOutput
        ) `
        -ExpectedOutput $subliminalOutput

    Invoke-LoggedPython `
        -Name "neutral_topk_interventions" `
        -Arguments @(
            "scripts/run_lora_set_interventions.py",
            "--adapter-path", $neutralAdapter,
            "--teacher-vector", $teacher,
            "--prompts", $prompts,
            "--selection-plan", $selectionPlan,
            "--n-prompts", "$NumPrompts",
            "--prompt-offset", "$PromptOffset",
            "--batch-size", "$BatchSize",
            "--output", $neutralOutput
        ) `
        -ExpectedOutput $neutralOutput

    Invoke-LoggedPython `
        -Name "paired_topk_comparison" `
        -Arguments @(
            "scripts/compare_lora_set_interventions.py",
            "--subliminal", $subliminalOutput,
            "--neutral", $neutralOutput,
            "--output", $pairedOutput,
            "--bootstrap-samples", "$BootstrapSamples",
            "--bootstrap-seed", "20260712"
        ) `
        -ExpectedOutput $pairedOutput

    if (-not $SkipPytest) {
        Invoke-LoggedPython `
            -Name "pytest" `
            -Arguments @("-m", "pytest", "-q")
    }

    $runState.status = "completed"
    $runState.finished_at = (Get-Date).ToString("o")
    Write-RunState

    Write-Host ""
    Write-Host "All unattended top-k tests completed successfully." -ForegroundColor Green
    Write-Host "Paired result: $pairedOutput"
    Write-Host "Run status:    $statusPath"
    Write-Host "Logs:          $logDir"
}
catch {
    $runState.status = "failed"
    $runState.finished_at = (Get-Date).ToString("o")
    $runState.error = $_.Exception.Message
    Write-RunState
    Write-Error $_
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}
