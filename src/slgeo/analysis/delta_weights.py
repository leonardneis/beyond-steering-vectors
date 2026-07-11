"""Utilities for reconstructing and comparing effective LoRA weight updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class LoraUpdate:
    """One effective LoRA update and the metadata needed to interpret it."""

    module: str
    factor_a: torch.Tensor
    factor_b: torch.Tensor
    rank: int
    alpha: float
    scaling: float

    @property
    def delta(self) -> torch.Tensor:
        """Materialize the dense update for one explicitly requested module."""
        return torch.matmul(self.factor_b.float(), self.factor_a.float()) * self.scaling

    @property
    def frobenius_norm(self) -> float:
        """Compute the exact norm from rank-sized Gram matrices without dense ``B @ A``."""
        a = self.factor_a.float()
        b = self.factor_b.float()
        gram_a = a @ a.T
        gram_b = b.T @ b
        squared = (gram_a * gram_b.T).sum().clamp_min(0)
        return float((squared.sqrt() * abs(self.scaling)).item())


def _strip_lora_suffix(key: str, factor: str) -> str | None:
    markers = (f".lora_{factor}.default.weight", f".lora_{factor}.weight")
    for marker in markers:
        if key.endswith(marker):
            return key[: -len(marker)]
    return None


def reconstruct_lora_updates(
    state_dict: Mapping[str, torch.Tensor],
    *,
    alpha: float,
    rank: int | None = None,
) -> dict[str, LoraUpdate]:
    """Reconstruct ``delta_W = (alpha / rank) * B @ A`` for every module.

    The function deliberately works on a state dict so it can be tested and used
    without loading a 7B base model. PEFT's common ``default`` adapter naming and
    state dicts with the adapter-name component removed are both supported.
    """
    factors_a: dict[str, torch.Tensor] = {}
    factors_b: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        module_a = _strip_lora_suffix(key, "A")
        module_b = _strip_lora_suffix(key, "B")
        if module_a is not None:
            factors_a[module_a] = tensor
        elif module_b is not None:
            factors_b[module_b] = tensor

    missing_b = sorted(set(factors_a) - set(factors_b))
    missing_a = sorted(set(factors_b) - set(factors_a))
    if missing_a or missing_b:
        raise ValueError(f"Unpaired LoRA factors: missing_A={missing_a}, missing_B={missing_b}")
    if not factors_a:
        raise ValueError("No paired LoRA A/B factors found in state dict")

    updates: dict[str, LoraUpdate] = {}
    for module in sorted(factors_a):
        a = factors_a[module].detach().cpu()
        b = factors_b[module].detach().cpu()
        inferred_rank = int(a.shape[0])
        module_rank = int(rank if rank is not None else inferred_rank)
        if module_rank != inferred_rank:
            raise ValueError(
                f"Configured rank {module_rank} disagrees with {module} A shape {tuple(a.shape)}"
            )
        if b.ndim != 2 or a.ndim != 2 or b.shape[1] != module_rank:
            raise ValueError(f"Invalid LoRA shapes for {module}: A={tuple(a.shape)}, B={tuple(b.shape)}")
        scaling = float(alpha) / module_rank
        updates[module] = LoraUpdate(module, a, b, module_rank, float(alpha), scaling)
    return updates


def load_adapter_state_dict(adapter_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load PEFT adapter weights without loading the base model."""
    adapter_dir = Path(adapter_dir)
    safe_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safe_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safe_path), device="cpu")
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin under {adapter_dir}")


def module_update_summary(updates: Mapping[str, LoraUpdate]) -> list[dict[str, float | int | str]]:
    """Return stable, norm-ranked summary records for reporting and selection."""
    records = []
    total_sq = sum(update.frobenius_norm**2 for update in updates.values())
    for update in updates.values():
        norm = update.frobenius_norm
        records.append(
            {
                "module": update.module,
                "rank": update.rank,
                "alpha": update.alpha,
                "scaling": update.scaling,
                "frobenius_norm": norm,
                "fraction_squared_norm": norm**2 / total_sq if total_sq else 0.0,
            }
        )
    return sorted(records, key=lambda row: (-float(row["frobenius_norm"]), str(row["module"])))


def lora_update_inner_product(left: LoraUpdate, right: LoraUpdate) -> float:
    """Exact Frobenius inner product between two updates without densifying them."""
    if left.factor_a.shape[1] != right.factor_a.shape[1] or left.factor_b.shape[0] != right.factor_b.shape[0]:
        raise ValueError(f"Incompatible update shapes for module {left.module}")
    cross_b = left.factor_b.float().T @ right.factor_b.float()
    cross_a = right.factor_a.float() @ left.factor_a.float().T
    value = torch.trace(cross_b @ cross_a) * left.scaling * right.scaling
    return float(value.item())


def compare_lora_updates(
    left: Mapping[str, LoraUpdate], right: Mapping[str, LoraUpdate]
) -> list[dict[str, float | str]]:
    """Compare paired condition updates modulewise using exact factor-space algebra."""
    if set(left) != set(right):
        raise ValueError("LoRA module sets differ between conditions")
    records = []
    for module in sorted(left):
        lhs, rhs = left[module], right[module]
        inner = lora_update_inner_product(lhs, rhs)
        norm_l, norm_r = lhs.frobenius_norm, rhs.frobenius_norm
        cosine = inner / (norm_l * norm_r) if norm_l and norm_r else 0.0
        difference_sq = max(norm_l**2 + norm_r**2 - 2 * inner, 0.0)
        records.append(
            {
                "module": module,
                "cosine": cosine,
                "difference_norm": difference_sq**0.5,
                "left_norm": norm_l,
                "right_norm": norm_r,
            }
        )
    return sorted(records, key=lambda row: (-float(row["difference_norm"]), str(row["module"])))
