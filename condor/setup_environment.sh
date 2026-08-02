#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(pwd)}
cd "$REPO_ROOT"
PASSED_SHARED_ROOT=${SLGEO_SHARED_ROOT:-}
# SIC's Docker universe forwards HOME but may omit USER. Runtime configuration
# is allowed to derive scratch paths from USER, so reconstruct it before
# sourcing the ignored environment file.
USER=${USER:-$(basename "${HOME:?HOME must be set}")}
export USER
if [[ -f condor/condor.env ]]; then
  # shellcheck disable=SC1091
  source condor/condor.env
fi
if [[ -n "$PASSED_SHARED_ROOT" ]]; then SLGEO_SHARED_ROOT=$PASSED_SHARED_ROOT; fi
: "${SLGEO_SHARED_ROOT:?SLGEO_SHARED_ROOT must point to persistent /scratch storage}"
case "$SLGEO_SHARED_ROOT" in
  /scratch/*) ;;
  *) echo "SLGEO_SHARED_ROOT must be below /scratch, got $SLGEO_SHARED_ROOT" >&2; exit 2 ;;
esac

export HF_HOME=${HF_HOME:-$SLGEO_SHARED_ROOT/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${SLGEO_OFFLINE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
ENV_BASE=${SLGEO_ENV_ROOT:-$HOME/.cache/beyond-steering-vectors/envs}
mkdir -p "$HF_HOME" "$SLGEO_SHARED_ROOT/results" "$SLGEO_SHARED_ROOT/data" \
  "$SLGEO_SHARED_ROOT/runs" "$ENV_BASE" condor/logs

# A custom image already has all dependencies. The public PyTorch fallback creates a
# content-addressed shared venv once; flock prevents array/DAG startup races.
if ! python -c 'import accelerate,bitsandbytes,datasets,matplotlib,numpy,pandas,peft,scipy,sklearn,transformers,trl,yaml' >/dev/null 2>&1; then
  REQUIREMENTS_HASH=$(python -c 'import hashlib;print(hashlib.sha256(open("condor/requirements-condor.txt","rb").read()).hexdigest()[:16])')
  ENV_ROOT="$ENV_BASE/condor-$REQUIREMENTS_HASH"
  (
    flock 9
    if [[ ! -f "$ENV_ROOT/.complete" ]]; then
      python -m venv --system-site-packages "$ENV_ROOT"
      "$ENV_ROOT/bin/python" -m pip install --upgrade 'pip==24.3.1'
      "$ENV_ROOT/bin/python" -m pip install -r condor/requirements-condor.txt
      "$ENV_ROOT/bin/python" -m pip install --no-deps -e .
      printf '%s\n' "$REQUIREMENTS_HASH" > "$ENV_ROOT/.complete.tmp"
      mv "$ENV_ROOT/.complete.tmp" "$ENV_ROOT/.complete"
    fi
  ) 9>"$ENV_BASE/bootstrap.lock"
  # shellcheck disable=SC1090
  source "$ENV_ROOT/bin/activate"
fi

python -c 'import slgeo,torch;print({"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available()})'
