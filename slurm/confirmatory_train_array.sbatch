#!/bin/bash
#SBATCH --job-name=sl-train
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
set -euo pipefail
source .venv/bin/activate
PAIR_INDEX=$((SLURM_ARRAY_TASK_ID / 2 + 1))
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then STAGE=train_subliminal; else STAGE=train_neutral; fi
python -u scripts/run_confirmatory_manifest.py --manifest configs/validation/cat_cross_seed_confirmatory.yaml --pair-index "$PAIR_INDEX" --stage "$STAGE" --emit-plan "results/confirmatory/plans/${STAGE}_${PAIR_INDEX}.json" --execute --resume
