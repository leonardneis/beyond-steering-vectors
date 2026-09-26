"""Prefill-only additive steering at absolute prompt positions (PREREGISTRATION §6.1).

A hook adds a per-row vector to the output of decoder block ``b`` (hidden-state slot ``b + 1``) at
absolute index ``prompt_len - 1`` (mode ``last``) or at every prompt position (mode ``all``). Each
hook is armed for exactly one forward: a second call raises, so steering can never reach a cached
continuation or a decode step. ``residual_intervention`` (``hidden[:, -1:]``) is deliberately not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

LAST = "last"
ALL = "all"
POSITION_MODES = (LAST, ALL)


class SteeringError(RuntimeError):
    """Raised when a steering hook would act outside the frozen site, position or pass."""


def decoder_layers(model) -> torch.nn.ModuleList:
    """The decoder blocks of a bare Qwen2ForCausalLM (no PEFT wrapper is accepted)."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        raise SteeringError(f"Cannot resolve decoder layers of {type(model).__name__}")
    return layers


@dataclass(frozen=True)
class RowSteer:
    """Steering of one batch row: ``vectors`` maps block index -> fp32 vector; ``mode`` last or all."""

    vectors: Mapping[int, torch.Tensor]
    mode: str = LAST

    def __post_init__(self) -> None:
        if self.mode not in POSITION_MODES:
            raise SteeringError(f"Unknown position mode {self.mode!r}")
        for block, vector in self.vectors.items():
            if not isinstance(block, int) or block < 0:
                raise SteeringError(f"Invalid block index {block!r}")
            if vector.ndim != 1 or vector.dtype != torch.float32:
                raise SteeringError("Steering vectors must be 1-D float32 tensors")
            if not torch.isfinite(vector).all():
                raise SteeringError("Non-finite steering vector")


UNSTEERED = RowSteer(vectors={})


class PrefillSteering:
    """Context manager around exactly one prefill forward of a batch of identical prompts.

    Attributes after exit: ``fired`` (block -> call count, must be 1 for every steered block) and
    ``applied_norms`` (block -> per-row norm of the vector actually added, after the dtype cast).
    """

    def __init__(self, model, prompt_len: int, rows: Sequence[RowSteer]):
        if prompt_len < 4:
            raise SteeringError("prompt_len is implausibly small")
        self.model = model
        self.prompt_len = int(prompt_len)
        self.rows = list(rows)
        self.layers = decoder_layers(model)
        blocks = sorted({block for row in self.rows for block in row.vectors})
        for block in blocks:
            if block >= len(self.layers):
                raise SteeringError(f"Block {block} does not exist")
        self.blocks = blocks
        self.fired: dict[int, int] = {block: 0 for block in blocks}
        self.applied_norms: dict[int, list[float]] = {}
        self._handles: list = []

    def _row_tables(self, block: int, hidden_size: int, device, dtype):
        vectors = torch.zeros(len(self.rows), hidden_size, dtype=torch.float32)
        last_rows, all_rows = [], []
        for index, row in enumerate(self.rows):
            vector = row.vectors.get(block)
            if vector is None:
                continue
            if vector.shape[0] != hidden_size:
                raise SteeringError(f"Vector size {vector.shape[0]} != hidden size {hidden_size}")
            vectors[index] = vector
            (last_rows if row.mode == LAST else all_rows).append(index)
        cast = vectors.to(device=device, dtype=dtype)
        return cast, last_rows, all_rows

    def _make_hook(self, block: int):
        def hook(_module, _args, kwargs, output):
            if self.fired[block] != 0:
                raise SteeringError(f"Block {block} steering hook called more than once (decode/continuation leak)")
            self.fired[block] += 1
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[0] != len(self.rows) or hidden.shape[1] != self.prompt_len:
                raise SteeringError(
                    f"Steering expected a prefill of shape [{len(self.rows)}, {self.prompt_len}, H], "
                    f"got {tuple(hidden.shape)}"
                )
            position_ids = kwargs.get("position_ids")
            if position_ids is not None:
                if int(position_ids[0, 0]) != 0 or int(position_ids[0, -1]) != self.prompt_len - 1:
                    raise SteeringError("Steering hook fired outside the prefill (position ids do not start at 0)")
            cache_position = kwargs.get("cache_position")
            if cache_position is not None and int(cache_position[0]) != 0:
                raise SteeringError("Steering hook fired with a non-empty cache")
            vectors, last_rows, all_rows = self._row_tables(block, hidden.shape[-1], hidden.device, hidden.dtype)
            self.applied_norms[block] = torch.linalg.vector_norm(vectors.float(), dim=-1).tolist()
            new_hidden = hidden.clone()
            if last_rows:
                index = torch.tensor(last_rows, device=hidden.device)
                new_hidden[index, self.prompt_len - 1, :] = hidden[index, self.prompt_len - 1, :] + vectors[index]
            if all_rows:
                index = torch.tensor(all_rows, device=hidden.device)
                new_hidden[index] = hidden[index] + vectors[index].unsqueeze(1)
            if isinstance(output, tuple):
                return (new_hidden, *output[1:])
            return new_hidden

        hook._cts_steering = True
        return hook

    def __enter__(self) -> "PrefillSteering":
        assert_no_hooks(self.model)
        for block in self.blocks:
            self._handles.append(
                self.layers[block].register_forward_hook(self._make_hook(block), with_kwargs=True)
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if exc_type is None:
            missing = [block for block, count in self.fired.items() if count != 1]
            if missing:
                raise SteeringError(f"Steering hooks on blocks {missing} did not fire exactly once")


def _hook_is_transformers_internal(hook) -> bool:
    """transformers 5.x installs its own output-capture hooks; they are the only foreign hooks tolerated."""
    module = getattr(hook, "__module__", "") or getattr(getattr(hook, "func", None), "__module__", "") or ""
    return module.startswith("transformers.")


def assert_no_hooks(model) -> None:
    """Hook hygiene between conditions (red-team RT-50): no leftover CTS hook and no foreign hook anywhere."""
    for name, module in model.named_modules():
        hooks = list(module._forward_hooks.values()) + list(module._forward_pre_hooks.values())
        for hook in hooks:
            if getattr(hook, "_cts_steering", False):
                raise SteeringError(f"Module {name or '<root>'} carries a leftover steering hook")
            if not _hook_is_transformers_internal(hook):
                raise SteeringError(f"Module {name or '<root>'} carries a foreign forward hook")
