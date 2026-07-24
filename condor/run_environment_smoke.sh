#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source condor/setup_environment.sh

python - <<'PY'
import accelerate
import bitsandbytes
import datasets
import matplotlib
import numpy
import pandas
import peft
import scipy
import sklearn
import torch
import transformers
import trl
import yaml

print({
    "status": "ok",
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "bitsandbytes": bitsandbytes.__version__,
})
PY
