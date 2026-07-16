#!/bin/bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ "${1:-}" == "--dry-run" ]]; then
  cat <<'EOF'
Confirmatory SLURM DAG (no jobs submitted):
  prepare[seed2,seed3]
    -> train_subliminal[seed2,seed3] || train_neutral[seed2,seed3]
    -> verify_runs[seed1..3,condition] -> verify_compare[seed1..3]
    -> vectors[seed1..3,condition] || layer_runs[seed1..3,condition]
    -> layer_compare -> module_runs[seed1..3,condition] -> module_compare -> topk_prepare
    -> activation_runs[seed1..3,condition] -> activation_compare
    -> behavior_runs[seed1..3,condition] -> behavior_compare
    -> final aggregate, plots, hashes, CPU tests
EOF
  if command -v python >/dev/null 2>&1; then
    python scripts/confirmatory_status.py --manifest configs/validation/cat_cross_seed_confirmatory.yaml
  else
    echo "python is not on this shell PATH; skipped status rendering."
  fi
  exit 0
fi
if [[ -f slurm/sic_cluster.env ]]; then
  # shellcheck disable=SC1091
  source slurm/sic_cluster.env
fi
COMMON=${SIC_SBATCH_COMMON:-}
mkdir -p slurm_logs results/confirmatory/plans

submit_array() {
  local template=$1 stage=$2 pair_start=$3 pair_count=$4 command_count=$5 dependency=${6:-}
  local max=$((pair_count * command_count - 1)) dep=()
  [[ -n "$dependency" ]] && dep=(--dependency="afterok:$dependency")
  # COMMON is intentionally word-split: it contains optional site sbatch flags.
  # shellcheck disable=SC2086
  sbatch --parsable $COMMON "${dep[@]}" --array="0-$max" \
    --export="ALL,STAGE=$stage,PAIR_START=$pair_start,COMMAND_COUNT=$command_count" "$template"
}

PREP=$(submit_array slurm/confirmatory_cpu_array.sbatch prepare 1 2 2)
TRAIN_SUB=$(submit_array slurm/confirmatory_gpu_array.sbatch train_subliminal 1 2 1 "$PREP")
TRAIN_NEU=$(submit_array slurm/confirmatory_gpu_array.sbatch train_neutral 1 2 1 "$PREP")
TRAIN_DEP="$TRAIN_SUB:$TRAIN_NEU"
VERIFY_RUNS=$(submit_array slurm/confirmatory_gpu_array.sbatch verify_runs 0 3 2 "$TRAIN_DEP")
VERIFY_COMPARE=$(submit_array slurm/confirmatory_cpu_array.sbatch verify_compare 0 3 1 "$VERIFY_RUNS")

# Vector extraction and layer screening can occupy independent GPUs concurrently.
VECTORS=$(submit_array slurm/confirmatory_gpu_array.sbatch vectors 0 3 2 "$VERIFY_COMPARE")
LAYER_RUNS=$(submit_array slurm/confirmatory_gpu_array.sbatch layer_runs 0 3 2 "$VERIFY_COMPARE")
LAYER_COMPARE=$(submit_array slurm/confirmatory_cpu_array.sbatch layer_compare 0 3 1 "$LAYER_RUNS")
MODULE_RUNS=$(submit_array slurm/confirmatory_gpu_array.sbatch module_runs 0 3 2 "$LAYER_COMPARE")
MODULE_COMPARE=$(submit_array slurm/confirmatory_cpu_array.sbatch module_compare 0 3 1 "$MODULE_RUNS")
TOPK=$(submit_array slurm/confirmatory_cpu_array.sbatch topk_prepare 0 3 1 "$MODULE_COMPARE")

# Activation and token-behavior interventions are independent after top-k selection.
ACT_RUNS=$(submit_array slurm/confirmatory_gpu_array.sbatch activation_runs 0 3 2 "$TOPK")
ACT_COMPARE=$(submit_array slurm/confirmatory_cpu_array.sbatch activation_compare 0 3 1 "$ACT_RUNS")
BEH_RUNS=$(submit_array slurm/confirmatory_gpu_array.sbatch behavior_runs 0 3 2 "$TOPK")
BEH_COMPARE=$(submit_array slurm/confirmatory_cpu_array.sbatch behavior_compare 0 3 1 "$BEH_RUNS")

# shellcheck disable=SC2086
FINAL=$(sbatch --parsable $COMMON --dependency="afterok:$VECTORS:$ACT_COMPARE:$BEH_COMPARE" slurm/confirmatory_finalize.sbatch)
SUBMISSION="results/confirmatory/submission_${FINAL}.json"
python - "$SUBMISSION" <<PY
import json, sys
jobs = ${PREP@Q}, ${TRAIN_SUB@Q}, ${TRAIN_NEU@Q}, ${VERIFY_RUNS@Q}, ${VERIFY_COMPARE@Q}, ${VECTORS@Q}, ${LAYER_RUNS@Q}, ${LAYER_COMPARE@Q}, ${MODULE_RUNS@Q}, ${MODULE_COMPARE@Q}, ${TOPK@Q}, ${ACT_RUNS@Q}, ${ACT_COMPARE@Q}, ${BEH_RUNS@Q}, ${BEH_COMPARE@Q}, ${FINAL@Q}
names = "prepare train_subliminal train_neutral verify_runs verify_compare vectors layer_runs layer_compare module_runs module_compare topk_prepare activation_runs activation_compare behavior_runs behavior_compare finalize".split()
open(sys.argv[1], "w").write(json.dumps(dict(zip(names, jobs)), indent=2) + "\n")
PY
python scripts/confirmatory_status.py --manifest configs/validation/cat_cross_seed_confirmatory.yaml
echo "Submitted confirmatory DAG. Final aggregation job: $FINAL"
echo "Monitor: python scripts/confirmatory_status.py --manifest configs/validation/cat_cross_seed_confirmatory.yaml --watch-seconds 60"
