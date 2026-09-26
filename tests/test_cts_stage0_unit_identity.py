from __future__ import annotations

from pathlib import Path
import copy
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import identity as idm
from slgeo.cts_stage0.identity import IdentityError, assert_identity


@pytest.fixture(scope="module")
def expected() -> dict:
    manifest = yaml.safe_load((ROOT / "configs" / "validation" / "cts_stage0_v1.yaml").read_text(encoding="utf-8"))
    return manifest["execution"]


def _matching(expected: dict) -> dict:
    packages = {name: None for name in idm.PACKAGES}
    packages.update(expected["packages"])
    return {
        "python": expected["python"],
        "packages": packages,
        "cuda_runtime": expected["cuda_runtime"],
        "cuda_available": True,
        "cuda_device_count": 1,
        "gpu_name": expected["gpu_name"],
        "nvidia_smi": [{"name": expected["gpu_name"], "driver": expected["nvidia_driver"], "uuid": "GPU-x"}],
        "container_image": None,
        "job_ad": {"DockerImage": expected["container_image"]},
        "machine_ad": {"GPUs_NvidiaDriver": expected["nvidia_driver"]},
        "slgeo_file": str((ROOT / "src" / "slgeo" / "__init__.py").resolve()),
    }


def _patch(monkeypatch, actual: dict) -> None:
    monkeypatch.setattr(idm, "runtime_identity", lambda: copy.deepcopy(actual))


def test_matching_identity_passes(monkeypatch, expected):
    actual = _matching(expected)
    _patch(monkeypatch, actual)
    assert assert_identity(expected, ROOT)["gpu_name"] == expected["gpu_name"]


def test_exit_code():
    assert IdentityError.exit_code == 86 == idm.IDENTITY_EXIT_CODE


def _set(path: str, value):
    def mutate(actual: dict) -> None:
        keys = path.split(".")
        target = actual
        for key in keys[:-1]:
            target = target[int(key)] if isinstance(target, list) else target[key]
        target[keys[-1]] = value
    return mutate


MUTATIONS = {
    "python": _set("python", "3.12.0"),
    "torch": _set("packages.torch", "2.5.1+cu121"),
    "transformers": _set("packages.transformers", "5.7.0"),
    "tokenizers": _set("packages.tokenizers", "0.22.2"),
    "numpy": _set("packages.numpy", "2.4.4"),
    "bitsandbytes_missing": _set("packages.bitsandbytes", None),
    "cuda_runtime": _set("cuda_runtime", "12.1"),
    "cuda_unavailable": _set("cuda_available", False),
    "two_devices": _set("cuda_device_count", 2),
    "gpu_name": _set("gpu_name", "NVIDIA A100-SXM4-80GB"),
    "smi_name": _set("nvidia_smi.0.name", "NVIDIA H100"),
    "driver": _set("nvidia_smi.0.driver", "550.54.15"),
    "machine_ad_driver": _set("machine_ad.GPUs_NvidiaDriver", "550.54.15"),
    "image": _set("job_ad.DockerImage", "pytorch/pytorch:latest"),
    "no_smi": _set("nvidia_smi", None),
    "slgeo_outside": _set("slgeo_file", str(Path(ROOT).resolve().parent / "elsewhere" / "slgeo" / "__init__.py")),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_single_field_mismatch_raises(monkeypatch, expected, name):
    actual = _matching(expected)
    MUTATIONS[name](actual)
    _patch(monkeypatch, actual)
    with pytest.raises(IdentityError, match="Execution identity mismatch|slgeo imported"):
        assert_identity(expected, ROOT)


def test_two_gpus_in_smi_raise(monkeypatch, expected):
    actual = _matching(expected)
    actual["nvidia_smi"] = actual["nvidia_smi"] * 2
    _patch(monkeypatch, actual)
    with pytest.raises(IdentityError, match="exactly one GPU"):
        assert_identity(expected, ROOT)


def test_container_image_from_env_when_no_job_ad(monkeypatch, expected):
    actual = _matching(expected)
    actual["job_ad"] = {}
    actual["container_image"] = expected["container_image"]
    _patch(monkeypatch, actual)
    assert_identity(expected, ROOT)
    actual["container_image"] = None
    _patch(monkeypatch, actual)
    with pytest.raises(IdentityError, match="container image"):
        assert_identity(expected, ROOT)


def test_cpu_mode_ignores_gpu_fields_only(monkeypatch, expected):
    actual = _matching(expected)
    for name in ("cuda_unavailable", "gpu_name", "driver", "image", "no_smi"):
        MUTATIONS[name](actual)
    _patch(monkeypatch, actual)
    assert_identity(expected, ROOT, require_gpu=False)
    MUTATIONS["transformers"](actual)
    _patch(monkeypatch, actual)
    with pytest.raises(IdentityError):
        assert_identity(expected, ROOT, require_gpu=False)


def test_slgeo_from_sibling_checkout_with_same_prefix_raises(monkeypatch, expected):
    actual = _matching(expected)
    root = Path(ROOT).resolve()
    actual["slgeo_file"] = str(root.parent / (root.name + "-old") / "src" / "slgeo" / "__init__.py")
    _patch(monkeypatch, actual)
    with pytest.raises(IdentityError, match="slgeo imported"):
        assert_identity(expected, ROOT)


def test_runtime_identity_shape(monkeypatch):
    import torch

    # CPU only: initializing CUDA here crashes the Windows test interpreter at shutdown.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(idm, "_nvidia_smi", lambda: None)
    identity = idm.runtime_identity()
    assert identity["cuda_available"] is False and identity["gpu_name"] is None
    assert set(identity["packages"]) == set(idm.PACKAGES)
    assert identity["slgeo_file"] is not None
