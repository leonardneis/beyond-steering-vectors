$ErrorActionPreference = "Stop"
$sw = [System.Diagnostics.Stopwatch]::StartNew()

$stdout = "results/geometry/attribution/cat_neutral_module_screen_seed1_phase2.stdout.log"
$stderr = "results/geometry/attribution/cat_neutral_module_screen_seed1_phase2.stderr.log"

$arguments = @(
    "scripts/run_lora_attribution.py"
    "--adapter-path", "results/reference_reproduction_4080/qwen7b_neutral_10k_3epochs/student_lora"
    "--teacher-vector", "results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt"
    "--prompts", "data/generated/reference_qwen7b_cat_subliminal_30k.jsonl"
    "--n-prompts", "256"
    "--prompt-offset", "2048"
    "--batch-size", "2"
    "--group-by", "individual"
    "--include-layers", "0", "5", "10", "18", "22", "25"
    "--target-block", "10"
    "--output", "results/geometry/attribution/cat_neutral_module_screen_seed1_phase2.json"
)

$process = Start-Process `
    -FilePath ".venv/Scripts/python.exe" `
    -ArgumentList $arguments `
    -NoNewWindow `
    -Wait `
    -PassThru `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr

if ($process.ExitCode -ne 0) {
    throw "Neutral module screening failed with exit code $($process.ExitCode). See $stderr"
}

$sw.Stop()
Write-Host "Finished successfully in $($sw.Elapsed)"
