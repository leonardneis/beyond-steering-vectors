"""Fail-closed primitives for C18 teacher-coordinate interchange.

This module contains no model-specific outcome classification side effects.
Model runners persist identified raw cells; aggregation and audit are separate.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np
import torch

from .interventions import decoder_blocks


EPS64 = np.finfo(np.float64).eps
FORBIDDEN_TECHNICAL_KEYS = {
    "Y00", "Y01", "Y10", "Y11", "G_B0", "G_B1", "G_E", "G_L",
    "G_H", "G_I", "classification", "candidate_logits", "margin",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rows_float64(rows: torch.Tensor, *, min_norm: float = 1e-12) -> torch.Tensor:
    values = rows.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("directions must be a finite rank-2 tensor")
    norms = torch.linalg.vector_norm(values, dim=-1)
    if torch.any(norms <= min_norm):
        bad = torch.nonzero(norms <= min_norm).flatten().tolist()
        raise ValueError(f"degenerate direction rows: {bad}")
    return values / norms[:, None]


def generate_orthogonal_families(
    teacher_rows: torch.Tensor,
    *,
    seed: int = 20260914,
    family_count: int = 5,
    min_norm: float = 1e-12,
) -> np.ndarray:
    """Generate family-major, slot-major PCG64 Gaussian directions."""
    teacher = normalize_rows_float64(teacher_rows, min_norm=min_norm).numpy()
    rng = np.random.Generator(np.random.PCG64(seed))
    result = np.empty((family_count, *teacher.shape), dtype=np.float64)
    for family in range(family_count):
        for slot in range(teacher.shape[0]):
            draw = rng.standard_normal(teacher.shape[1], dtype=np.float64)
            draw -= teacher[slot] * np.dot(teacher[slot], draw)
            norm = np.linalg.norm(draw)
            if not np.isfinite(norm) or norm <= min_norm:
                raise ValueError(f"degenerate orthogonal draw family={family} slot={slot + 1}")
            result[family, slot] = draw / norm
    return result


def deterministic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically write an NPZ with stable member order and ZIP timestamps."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    _atomic_bytes(target, buffer.getvalue(), refuse_overwrite=False)


def _atomic_bytes(path: Path, payload: bytes, *, refuse_overwrite: bool) -> None:
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: str | Path, value: object, *, refuse_overwrite: bool = True) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_bytes(Path(path), payload, refuse_overwrite=refuse_overwrite)


def atomic_text(path: str | Path, value: str, *, refuse_overwrite: bool = True) -> None:
    _atomic_bytes(Path(path), value.encode("utf-8"), refuse_overwrite=refuse_overwrite)


def seal_bytes(payload: bytes, key: bytes) -> bytes:
    """Encrypt and authenticate an outcome payload with a runtime-only key."""
    from cryptography.fernet import Fernet

    return Fernet(key).encrypt(payload)


def unseal_bytes(payload: bytes, key: bytes) -> bytes:
    """Decrypt an authenticated payload only in the explicit release process."""
    from cryptography.fernet import Fernet

    return Fernet(key).decrypt(payload)


def atomic_sealed(path: str | Path, payload: bytes, key: bytes) -> None:
    target = Path(path)
    if target.suffix != ".sealed":
        raise ValueError("sealed artifacts must use the .sealed suffix")
    _atomic_bytes(target, seal_bytes(payload, key), refuse_overwrite=True)


def _unpack(output):
    if isinstance(output, tuple):
        if not output:
            raise ValueError("empty block-output tuple")
        return output[0], output[1:]
    return output, None


def _repack(hidden: torch.Tensor, tail):
    return hidden if tail is None else (hidden, *tail)


@dataclass(frozen=True)
class ClampDiagnostics:
    count: int
    ideal_delta_norm_max: float
    ideal_delta_norm_sum: float
    executed_delta_norm_max: float
    executed_delta_norm_sum: float
    cast_error_norm_max: float
    cast_error_norm_sum: float
    coordinate_error_max: float
    coordinate_error_sum: float
    orthogonal_leakage_max: float
    orthogonal_leakage_sum: float
    bound_violation_max: float
    vanished_nonzero_count: int
    raw_sample: tuple[float, float, float, float, float]


def replace_teacher_coordinate(
    hidden: torch.Tensor,
    donor_coordinate: torch.Tensor,
    teacher: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    dose_direction: torch.Tensor | None = None,
    ideal_nonzero_threshold: float = 1e-3,
) -> tuple[torch.Tensor, ClampDiagnostics]:
    """Replace a coordinate (or replay its signed dose) at real tokens only."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("hidden/mask shape mismatch")
    if donor_coordinate.shape != hidden.shape[:2]:
        raise ValueError("donor coordinate shape mismatch")
    if hidden.dtype != torch.float16:
        raise TypeError(f"C18 residual stream must be float16, got {hidden.dtype}")
    device = hidden.device
    h64 = hidden.to(torch.float64)
    t = teacher.detach().to(device=device, dtype=torch.float64).flatten()
    t = t / torch.linalg.vector_norm(t).clamp_min(1e-12)
    direction = t if dose_direction is None else dose_direction.detach().to(device=device, dtype=torch.float64).flatten()
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
    donor = donor_coordinate.detach().to(device=device, dtype=torch.float64)
    current = torch.einsum("bth,h->bt", h64, t)
    alpha = donor - current
    ideal = h64 + alpha[..., None] * direction
    cast = ideal.to(dtype=hidden.dtype)
    real = attention_mask.to(device=device, dtype=torch.bool)
    result = hidden.clone()
    result[real] = cast[real]

    executed64 = cast.to(torch.float64)
    cast_error = executed64 - ideal
    delta = executed64 - h64
    ideal_delta = ideal - h64
    coordinate_target = donor if dose_direction is None else current
    coordinate_error = torch.abs(torch.einsum("bth,h->bt", executed64, t) - coordinate_target)
    along = torch.einsum("bth,h->bt", delta, direction)[..., None] * direction
    leakage = torch.linalg.vector_norm(delta - along, dim=-1)
    cast_norm = torch.linalg.vector_norm(cast_error, dim=-1)
    ideal_norm = torch.linalg.vector_norm(ideal, dim=-1)
    bound = cast_norm + 64 * torch.finfo(torch.float64).eps * torch.maximum(
        torch.ones_like(ideal_norm), ideal_norm
    )
    selected = lambda x: x[real]
    bound_violation = torch.maximum(coordinate_error, leakage) - bound
    ideal_delta_norm = torch.linalg.vector_norm(ideal_delta, dim=-1)
    executed_delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    vanished = (ideal_delta_norm > ideal_nonzero_threshold) & (executed_delta_norm == 0) & real
    diagnostics = ClampDiagnostics(
        count=int(real.sum().item()),
        ideal_delta_norm_max=float(selected(ideal_delta_norm).max().item()),
        ideal_delta_norm_sum=float(selected(ideal_delta_norm).sum().item()),
        executed_delta_norm_max=float(selected(executed_delta_norm).max().item()),
        executed_delta_norm_sum=float(selected(executed_delta_norm).sum().item()),
        cast_error_norm_max=float(selected(cast_norm).max().item()),
        cast_error_norm_sum=float(selected(cast_norm).sum().item()),
        coordinate_error_max=float(selected(coordinate_error).max().item()),
        coordinate_error_sum=float(selected(coordinate_error).sum().item()),
        orthogonal_leakage_max=float(selected(leakage).max().item()),
        orthogonal_leakage_sum=float(selected(leakage).sum().item()),
        bound_violation_max=float(max(0.0, selected(bound_violation).max().item())),
        vanished_nonzero_count=int(vanished.sum().item()),
        raw_sample=tuple(float(selected(value)[0].item()) for value in (
            ideal_delta_norm, executed_delta_norm, cast_norm, coordinate_error, leakage
        )),
    )
    return result, diagnostics


@contextmanager
def coordinate_clamp_hooks(
    model,
    teacher_rows: torch.Tensor,
    donor_coordinates: Mapping[int, torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    block_indices: Sequence[int] = tuple(range(27)),
    dose_rows: torch.Tensor | None = None,
):
    """Install exactly one ascending post-forward hook for each C18 block."""
    blocks = decoder_blocks(model)
    indices = [int(value) for value in block_indices]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("hook indices must be unique and ascending")
    if indices != list(range(27)):
        raise ValueError("C18 v1 requires blocks 0 through 26")
    teacher = normalize_rows_float64(teacher_rows)
    if teacher.shape[0] != 27:
        raise ValueError("C18 requires exactly 27 teacher rows")
    if set(donor_coordinates) != set(indices):
        raise ValueError("donor coordinate inventory differs from hook inventory")
    dose = normalize_rows_float64(dose_rows) if dose_rows is not None else None
    diagnostics: list[tuple[int, ClampDiagnostics]] = []
    calls: list[int] = []
    handles = []
    for block_index in indices:
        row_index = block_index

        def hook(_module, _args, output, block=block_index, row=row_index):
            hidden, tail = _unpack(output)
            changed, stats = replace_teacher_coordinate(
                hidden,
                donor_coordinates[block],
                teacher[row],
                attention_mask,
                dose_direction=None if dose is None else dose[row],
            )
            calls.append(block)
            diagnostics.append((block, stats))
            return _repack(changed, tail)

        handles.append(blocks[block_index].register_forward_hook(hook))
    try:
        yield {"calls": calls, "diagnostics": diagnostics, "handles": handles}
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def capture_natural_donors(
    model,
    teacher_rows: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    block_indices: Sequence[int] = tuple(range(27)),
):
    """Capture immutable scalar coordinates and the complete block-26 state."""
    blocks = decoder_blocks(model)
    teacher = normalize_rows_float64(teacher_rows)
    coordinates: dict[int, torch.Tensor] = {}
    block26: list[torch.Tensor] = []
    handles = []
    for block_index in block_indices:
        def hook(_module, _args, output, block=block_index):
            hidden, _tail = _unpack(output)
            coordinate = torch.einsum(
                "bth,h->bt", hidden.to(torch.float64), teacher[block].to(hidden.device)
            ).detach().clone()
            coordinate.requires_grad_(False)
            coordinates[block] = coordinate
            if block == 26:
                state = hidden.detach().clone()
                state.requires_grad_(False)
                block26.append(state)

        handles.append(blocks[block_index].register_forward_hook(hook))
    try:
        yield {"coordinates": coordinates, "block26": block26}
    finally:
        for handle in handles:
            handle.remove()


def four_cell_quantities(cells: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {"Y00", "Y01", "Y10", "Y11"}
    if set(cells) != required:
        raise ValueError(f"four-cell inventory must be {sorted(required)}")
    y00, y01, y10, y11 = (np.asarray(cells[key], dtype=np.float64) for key in sorted(required))
    if not (y00.shape == y01.shape == y10.shape == y11.shape):
        raise ValueError("four-cell shapes differ")
    result = {
        "E": y11 - y00,
        "L": y01 - y00,
        "H": y11 - y10,
        "B0": y10 - y00,
        "B1": y11 - y01,
        "I": y11 - y10 - y01 + y00,
    }
    verify_four_cell_algebra(result)
    return result


def verify_four_cell_algebra(values: Mapping[str, np.ndarray], *, multiplier: int = 64) -> None:
    checks = (
        (values["E"], values["L"] + values["B1"]),
        (values["E"], values["H"] + values["B0"]),
        (values["I"], values["H"] - values["L"]),
        (values["I"], values["B1"] - values["B0"]),
    )
    for left, right in checks:
        scale = np.maximum(1.0, np.abs(left) + np.abs(right))
        if np.any(np.abs(left - right) > multiplier * EPS64 * scale):
            raise ArithmeticError("four-cell algebra failed")


def stratified_bootstrap_indices(
    families: Sequence[str], *, draws: int = 20000, seed: int = 20260804
) -> np.ndarray:
    labels = np.asarray(families)
    unique = list(dict.fromkeys(labels.tolist()))
    if len(unique) != 3 or any(np.sum(labels == family) != 24 for family in unique):
        raise ValueError("bootstrap requires three ordered families of 24 prompts")
    rng = np.random.Generator(np.random.PCG64(seed))
    output = np.empty((draws, 72), dtype=np.int64)
    for draw in range(draws):
        offset = 0
        for family in unique:
            pool = np.flatnonzero(labels == family)
            output[draw, offset : offset + 24] = rng.choice(pool, size=24, replace=True)
            offset += 24
    return output


def percentile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("interval values must be finite and one-dimensional")
    tail = 50.0 * (1.0 - confidence)
    low, high = np.percentile(array, [tail, 100.0 - tail])
    return float(low), float(high)


def iut_pass(interval_b0: Sequence[float], interval_b1: Sequence[float], epsilon: float) -> bool:
    return all(float(interval[0]) > -epsilon and float(interval[1]) < epsilon for interval in (interval_b0, interval_b1))


def cancellation_flags(
    g: Mapping[str, np.ndarray],
    condition_means: Mapping[str, Mapping[str, float]],
    family_means: Mapping[str, Mapping[str, float]],
    *,
    epsilon: float,
    collapse: bool = False,
) -> dict[str, list[str]]:
    flags: dict[str, list[str]] = {"B0": [], "B1": []}
    for key in flags:
        values = np.asarray(g[key], dtype=np.float64)
        if np.mean(np.abs(values)) >= epsilon:
            flags[key].append("mean_absolute")
        if np.sqrt(np.mean(values**2)) >= 2 * epsilon:
            flags[key].append("rms")
        family = list(family_means[key].values())
        if max(map(abs, family)) >= epsilon:
            flags[key].append("family_magnitude")
        condition = list(condition_means[key].values())
        if max(map(abs, condition)) >= epsilon:
            flags[key].append("condition_magnitude")
        if any(np.sign(x) != np.sign(y) and abs(x - y) >= 2 * epsilon for i, x in enumerate(family) for y in family[i + 1 :]):
            flags[key].append("opposing_families")
        if len(condition) == 2 and np.sign(condition[0]) == np.sign(condition[1]) and min(map(abs, condition)) >= epsilon and abs(condition[0] - condition[1]) < epsilon:
            flags[key].append("condition_cancellation")
    if collapse:
        flags["B0"].append("output_collapse")
        flags["B1"].append("output_collapse")
    return flags


def validate_technical_report(report: Mapping[str, object]) -> None:
    """Reject outcome-bearing fields from the human-readable technical report."""
    def walk(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key) in FORBIDDEN_TECHNICAL_KEYS or str(key).startswith("G_") or str(key).startswith("Y0") or str(key).startswith("Y1"):
                    raise ValueError(f"outcome-bearing technical field forbidden: {'.'.join((*path, str(key)))}")
                walk(nested, (*path, str(key)))
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                walk(nested, (*path, str(index)))
    walk(report)
