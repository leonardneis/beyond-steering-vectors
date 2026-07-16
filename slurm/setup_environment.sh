#!/bin/bash
set -euo pipefail
REPO_ROOT=${SL_THESIS_REPO_ROOT:-$(pwd)}
cd "$REPO_ROOT"
if [[ -f slurm/sic_cluster.env ]]; then
  # shellcheck disable=SC1091
  source slurm/sic_cluster.env
fi
export CONFIRMATORY_MANIFEST=${CONFIRMATORY_MANIFEST:-configs/validation/cat_cross_seed_confirmatory.yaml}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" slurm_logs results/confirmatory/plans
if [[ -n "${SLURM_MODULES:-}" ]]; then
  # shellcheck disable=SC2086
  module load ${SLURM_MODULES}
fi
source "${SL_THESIS_VENV:-.venv}/bin/activate"
python - <<'PY'
import torch
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
PY
