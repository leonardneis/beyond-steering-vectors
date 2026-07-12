#!/bin/bash
#SBATCH --job-name=sl-prepare
#SBATCH --array=1-2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
set -euo pipefail
source .venv/bin/activate
python -u scripts/run_confirmatory_manifest.py --manifest configs/validation/cat_cross_seed_confirmatory.yaml --pair-index "$SLURM_ARRAY_TASK_ID" --stage prepare --emit-plan "results/confirmatory/plans/prepare_${SLURM_ARRAY_TASK_ID}.json" --execute --resume
