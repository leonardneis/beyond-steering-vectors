#!/bin/bash
#SBATCH --job-name=sl-conf-final
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
source slurm/setup_environment.sh
ROOT="results/confirmatory/qwen7b_cat_cross_seed_v1"
OUT="$ROOT/aggregate_${SLURM_JOB_ID}.json"
PLOTS="$ROOT/plots_${SLURM_JOB_ID}"
python -u scripts/aggregate_confirmatory_seeds.py \
  --seed-roots "$ROOT/seed_1" "$ROOT/seed_2" "$ROOT/seed_3" \
  --output "$OUT" --bootstrap-samples 10000
python -u scripts/plot_confirmatory_aggregate.py --input "$OUT" --output-dir "$PLOTS"
python -u scripts/confirmatory_status.py --manifest "$CONFIRMATORY_MANIFEST"
python -m pytest -q
sha256sum "$OUT" "$PLOTS"/*.png > "$ROOT/final_artifacts_${SLURM_JOB_ID}.sha256"
