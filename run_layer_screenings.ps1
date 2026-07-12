$ErrorActionPreference = "Stop"

$python = "python"

$prompts = "data/generated/reference_qwen7b_cat_subliminal_30k.jsonl"
$teacher = "results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt"

$subliminalAdapter = "results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/student_lora"
$neutralAdapter    = "results/reference_reproduction_4080/qwen7b_neutral_10k_3epochs/student_lora"

function Run-Screening {
    param(
        [string]$Name,
        [string]$Adapter,
        [int]$Offset,
        [string]$Output
    )

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "Starting: $Name"
    Write-Host "Offset:   $Offset"
    Write-Host "Output:   $Output"
    Write-Host "==================================================" -ForegroundColor Cyan

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    & $python scripts/run_lora_attribution.py `
        --adapter-path $Adapter `
        --teacher-vector $teacher `
        --prompts $prompts `
        --n-prompts 128 `
        --prompt-offset $Offset `
        --group-by layer `
        --target-block 10 `
        --output $Output

    if ($LASTEXITCODE -ne 0) {
        throw "Run '$Name' failed."
    }

    $sw.Stop()

    Write-Host ""
    Write-Host "$Name finished in $($sw.Elapsed)." -ForegroundColor Green
}

$total = [System.Diagnostics.Stopwatch]::StartNew()

Run-Screening `
    -Name "Subliminal Split A" `
    -Adapter $subliminalAdapter `
    -Offset 1024 `
    -Output "results/geometry/attribution/cat_subliminal_layer_screen_seed1_splitA.json"

Run-Screening `
    -Name "Neutral Split A" `
    -Adapter $neutralAdapter `
    -Offset 1024 `
    -Output "results/geometry/attribution/cat_neutral_layer_screen_seed1_splitA.json"

Run-Screening `
    -Name "Subliminal Split B" `
    -Adapter $subliminalAdapter `
    -Offset 1152 `
    -Output "results/geometry/attribution/cat_subliminal_layer_screen_seed1_splitB.json"

Run-Screening `
    -Name "Neutral Split B" `
    -Adapter $neutralAdapter `
    -Offset 1152 `
    -Output "results/geometry/attribution/cat_neutral_layer_screen_seed1_splitB.json"

$total.Stop()

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "All runs completed successfully."
Write-Host "Total runtime: $($total.Elapsed)"
Write-Host "==============================================" -ForegroundColor Green
