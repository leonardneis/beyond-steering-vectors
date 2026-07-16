#!/bin/bash
#SBATCH --job-name=sl-conf-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
set -euo pipefail
PAIR_INDEX=$((PAIR_START + SLURM_ARRAY_TASK_ID / COMMAND_COUNT))
COMMAND_INDEX=$((SLURM_ARRAY_TASK_ID % COMMAND_COUNT))
exec bash slurm/run_confirmatory_task.sh "$PAIR_INDEX" "$STAGE" "$COMMAND_INDEX"
