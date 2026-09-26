"""Shard execution on the cluster GPU (scientific runs and TV share this code path).

A shard is one arm's contexts of one kind. Outputs are written once under ``<out>/raw/<shard_id>/`` with a sidecar
recording commit, identity, input hashes and timings; an existing complete shard is never overwritten, a partial one
is an integrity error. Scientific outputs are *sealed*: nothing in this module reads them back for analysis.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from slgeo.cts_stage0.render import Renderer, SCIENTIFIC
from slgeo.cts_stage0.scoring import FormTable, lm_head_weight32

from . import models, runner
from .panel import build_endpoint


class ShardError(RuntimeError):
    pass


def _write_once(path: Path, data: bytes) -> None:
    if path.exists():
        raise ShardError(f"{path} exists; outputs are write-once")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def form_table(tokenizer, boundary_ids) -> FormTable:
    return FormTable.from_endpoint(build_endpoint(lambda s: tokenizer.encode(s, add_special_tokens=False), boundary_ids))


def rendered_contexts(renderer: Renderer, contexts, number_prompts: Mapping[str, str] | None = None):
    out = []
    for c in contexts:
        prompt = c.user_prompt or (number_prompts or {}).get(c.stem_id)
        if not prompt:
            raise ShardError(f"No prompt text for {c.context_id}")
        out.append(renderer.render(c.persona_id, prompt))
    return out


def run_shard(
    shard: Mapping[str, Any], contexts: Mapping[str, runner.Context], *, out_root: Path, tokenizer, base_model,
    peft_model, package, boundary_ids, cjk_ids, sampling: Mapping[str, Any], capture_cells, number_prompts,
    provenance: Mapping[str, Any],
) -> Path:
    shard_dir = out_root / "raw" / shard["shard_id"]
    if (shard_dir / "COMPLETE").exists():
        return shard_dir
    if shard_dir.exists() and any(shard_dir.iterdir()):
        raise ShardError(f"Partial shard directory {shard_dir}; refusing to mix attempts")
    shard_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ctx = [contexts[cid] for cid in shard["context_ids"]]
    renderer = Renderer(tokenizer, package, mode=SCIENTIFIC)
    rendered = rendered_contexts(renderer, ctx, number_prompts)
    manager = models.active(peft_model, shard["adapter"]) if peft_model is not None else _nullcontext(base_model)
    with manager as model:
        if shard["kind"] == "score":
            table = form_table(tokenizer, boundary_ids)
            weight32 = lm_head_weight32(model)
            decoration = runner.decoration_ids(lambda s: tokenizer.encode(s, add_special_tokens=False))
            emoji = runner.space_emoji_ids(lambda ids: tokenizer.decode(ids), len(tokenizer), boundary_ids)
            scored = runner.score_contexts(model, rendered, ctx, table, weight32=weight32, cjk_ids=cjk_ids,
                                           decoration=decoration, emoji=emoji,
                                           capture=lambda c: c.cell in set(capture_cells))
            arrays = {k: v for k, v in asdict(scored).items() if k not in {"states", "context_ids", "captured_ids"}}
            _save_npz(shard_dir / "scores.npz", context_ids=np.array(scored.context_ids), **arrays)
            if scored.states is not None:
                _save_npz(shard_dir / "states.npz", context_ids=np.array(scored.captured_ids), states=scored.states)
        elif shard["kind"] == "numcap":
            states = np.stack([runner.last_token_states(model, r.input_ids) for r in rendered])
            _save_npz(shard_dir / "states.npz", context_ids=np.array([c.context_id for c in ctx]), states=states)
        elif shard["kind"] == "sample":
            samples = runner.sample_contexts(
                model, rendered, ctx, k=int(sampling["k"]), batch_size=int(sampling["batch_size"]),
                decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
                eos_ids=[tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id],
                pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                max_new_tokens=int(sampling["max_new_tokens"]),
            )
            text = "".join(json.dumps(asdict(s), ensure_ascii=False) + "\n" for s in samples)
            _write_once(shard_dir / "samples.jsonl", text.encode("utf-8"))
        else:
            raise ShardError(f"Unknown shard kind {shard['kind']!r}")
    sidecar = dict(provenance, shard=dict(shard, context_ids=len(shard["context_ids"])),
                   seconds=time.time() - started, torch=torch.__version__)
    _write_once(shard_dir / "sidecar.json", json.dumps(sidecar, indent=1, default=str).encode("utf-8"))
    _write_once(shard_dir / "COMPLETE", b"")
    return shard_dir


def _save_npz(path: Path, **arrays) -> None:
    if path.exists():
        raise ShardError(f"{path} exists; outputs are write-once")
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class _nullcontext:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False
