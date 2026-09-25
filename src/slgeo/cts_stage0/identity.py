"""Runtime execution identity: capture and fail-closed comparison with the pinned identity.

Pinned in ``configs/validation/cts_stage0_v1.yaml`` (``execution``). A mismatch raises
``IdentityError``; the node wrapper maps it to exit code 86, which the DAG never retries.
There is no fallback to another GPU class, driver or environment.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

IDENTITY_EXIT_CODE = 86
PACKAGES = ("torch", "transformers", "tokenizers", "bitsandbytes", "accelerate", "safetensors", "huggingface_hub", "numpy")


class IdentityError(RuntimeError):
    exit_code = IDENTITY_EXIT_CODE


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _nvidia_smi() -> list[dict[str, str]] | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            rows.append({"name": parts[0], "driver": parts[1], "uuid": parts[2]})
    return rows


def _classad(path_variable: str, keys: tuple[str, ...]) -> dict[str, str]:
    path = os.environ.get(path_variable)
    out: dict[str, str] = {}
    if not path or not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        if key.strip() in keys:
            out[key.strip()] = value.strip().strip('"')
    return out


def runtime_identity() -> dict[str, Any]:
    import torch

    gpus = _nvidia_smi()
    cuda = torch.cuda.is_available()
    identity: dict[str, Any] = {
        "python": platform.python_version(),
        "python_prefix": sys.prefix,
        "packages": {name: _version(name) for name in PACKAGES},
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "nvidia_smi": gpus,
        "container_image": os.environ.get("SLGEO_CONTAINER_IMAGE"),
        "job_ad": _classad("_CONDOR_JOB_AD", ("ClusterId", "ProcId", "NumJobStarts", "DockerImage", "RemoteHost")),
        "machine_ad": _classad("_CONDOR_MACHINE_AD", ("Machine", "GPUs_DeviceName", "GPUs_NvidiaDriver", "AssignedGPUs")),
    }
    try:
        import slgeo

        identity["slgeo_file"] = str(Path(slgeo.__file__).resolve())
    except ImportError:  # pragma: no cover
        identity["slgeo_file"] = None
    return identity


def assert_identity(expected: Mapping[str, Any], repo_root: str | Path, *, require_gpu: bool = True) -> dict[str, Any]:
    """Compare the runtime with the pinned identity; raise ``IdentityError`` on any difference."""
    actual = runtime_identity()
    problems: list[str] = []

    def check(label: str, got, want) -> None:
        if str(got) != str(want):
            problems.append(f"{label}: {got!r} != {want!r}")

    check("python", actual["python"], expected["python"])
    for name, want in expected["packages"].items():
        check(f"package {name}", actual["packages"].get(name), want)
    check("cuda_runtime", actual["cuda_runtime"], expected["cuda_runtime"])
    slgeo_file = actual.get("slgeo_file")
    root = Path(repo_root).resolve()
    if not slgeo_file or root not in Path(slgeo_file).resolve().parents:
        problems.append(f"slgeo imported from outside the execution checkout: {slgeo_file}")
    if require_gpu:
        check("cuda_available", actual["cuda_available"], True)
        check("cuda_device_count", actual["cuda_device_count"], 1)
        check("gpu_name", actual["gpu_name"], expected["gpu_name"])
        gpus = actual["nvidia_smi"]
        if not gpus or len(gpus) != 1:
            problems.append(f"nvidia-smi must report exactly one GPU: {gpus!r}")
        else:
            check("nvidia_smi name", gpus[0]["name"], expected["gpu_name"])
            check("nvidia_driver", gpus[0]["driver"], expected["nvidia_driver"])
        machine = actual["machine_ad"]
        if machine:
            check("machine ad driver", machine.get("GPUs_NvidiaDriver"), expected["nvidia_driver"])
        image = actual["job_ad"].get("DockerImage") or actual["container_image"]
        check("container image", image, expected["container_image"])
    if problems:
        raise IdentityError("Execution identity mismatch: " + "; ".join(problems))
    return actual
