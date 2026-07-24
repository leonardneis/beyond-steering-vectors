"""SIC HTCondor GPU/container/cache/scratch smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import load_yaml  # noqa: E402
from slgeo.models import load_model_and_tokenizer  # noqa: E402
from slgeo.experiment_logging import device_info, git_commit_hash  # noqa: E402


def main() -> None:
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model_qwen7b_4bit.yaml")
    args = parser.parse_args()
    import bitsandbytes
    import torch

    shared = Path(os.environ["SLGEO_SHARED_ROOT"])
    hf_home = Path(os.environ["HF_HOME"])
    if not str(shared).startswith("/scratch/"):
        raise RuntimeError(f"Expected shared storage below /scratch, got {shared}")
    if not hf_home.exists():
        raise FileNotFoundError(f"HF_HOME is not mounted: {hf_home}")
    probe = shared / "smoke" / f"write_probe_{os.getpid()}.tmp"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Condor container")
    config = load_yaml(repo_path(args.model_config))
    model_config_path = repo_path(args.model_config)
    if not bool(config.get("model", {}).get("load_in_4bit") or config.get("quantization", {}).get("load_in_4bit")):
        raise RuntimeError("Smoke-test model config must request bitsandbytes 4-bit loading")
    model_name = config["model"]["model_name"]
    from huggingface_hub import try_to_load_from_cache

    cache_probe = try_to_load_from_cache(
        model_name,
        "config.json",
        cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
    )
    print(json.dumps({
        "hf_home": str(hf_home),
        "huggingface_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        "model_name": model_name,
        "cached_config": str(cache_probe),
        "cached_config_exists": bool(cache_probe and Path(cache_probe).is_file()),
    }, indent=2))
    model, tokenizer = load_model_and_tokenizer(config)
    result = {
        "schema_version": 1, "status": "ok", "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started_perf,
        "git_commit": git_commit_hash(repo_path(".")),
        "git_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_path("."), capture_output=True, text=True
        ).stdout.strip()),
        "python": sys.version, "platform": platform.platform(), "seed": None,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "bitsandbytes": bitsandbytes.__version__, "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "device": device_info(),
        "model_class": type(model).__name__, "tokenizer_class": type(tokenizer).__name__,
        "model_config": str(model_config_path),
        "model_config_sha256": hashlib.sha256(model_config_path.read_bytes()).hexdigest(),
        "hf_home": str(hf_home), "shared_root": str(shared),
        "condor_cluster_id": os.getenv("CONDOR_CLUSTER_ID"), "condor_proc_id": os.getenv("CONDOR_PROC_ID"),
        "condor_task_id": os.getenv("CONDOR_TASK_ID", "gpu_smoke"),
    }
    output = shared / "smoke" / f"gpu_smoke_{os.getenv('CONDOR_CLUSTER_ID','local')}_{os.getenv('CONDOR_PROC_ID','0')}.json"
    temporary = output.with_suffix(".json.incoming")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
