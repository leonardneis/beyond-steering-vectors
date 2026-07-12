#!/bin/bash
#SBATCH --job-name=sl-analysis
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
set -euo pipefail
source .venv/bin/activate
python -u scripts/run_confirmatory_manifest.py --manifest configs/validation/cat_cross_seed_confirmatory.yaml --pair-index "$SLURM_ARRAY_TASK_ID" --stage analysis --emit-plan "results/confirmatory/plans/analysis_${SLURM_ARRAY_TASK_ID}.json" --execute --resume
