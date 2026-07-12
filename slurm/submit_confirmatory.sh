#!/bin/bash
set -euo pipefail
mkdir -p slurm_logs results/confirmatory/plans
PREP=$(sbatch --parsable slurm/confirmatory_prepare_array.sbatch)
TRAIN=$(sbatch --parsable --dependency=afterok:"$PREP" slurm/confirmatory_train_array.sbatch)
VERIFY=$(sbatch --parsable --dependency=afterok:"$TRAIN" slurm/confirmatory_verify_array.sbatch)
sbatch --dependency=afterok:"$VERIFY" slurm/confirmatory_analysis_array.sbatch
echo "prepare=$PREP train=$TRAIN verify=$VERIFY; analysis submitted with dependency"
