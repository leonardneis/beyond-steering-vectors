"""Outcome-blind technical validation of the Phenotype Anchor (TV-P1).

Blindness rules: student arms run only on TV-authored non-animal prompts, and their artifacts contain only digests,
timings, shapes and pass/fail flags, never distribution statistics. The animal panel runs only on the base model with
the CTS V prompts (TV-only by design). No RES/REF50/NONANIMAL stem is forwarded.

GPU checks (``gpu_checks``): execution identity; render identity (Q, H, a persona) on V prompts; base-vs-
``disable_adapter()`` logits (<= 1e-4); adapter census for every adapter (peft version skew); determinism of exact
scoring and of the CRN sampler on repeat; throughput per unit kind and per adapter (LoRA overhead factor).
CPU checks (``cpu_checks``): prompt manifest reproduces, panel collision-free, plan builds, statistics self-tests.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from slgeo.cts_stage0.render import Renderer, SCIENTIFIC
from slgeo.cts_stage0.scoring import lm_head_weight32

from . import models, runner
from .sampler import SampleRequest, sample_batch

TV_NONANIMAL_PROMPTS = (
    "Name a color using only one word.",
    "Which season do you like best? Answer with one word.",
    "Name a fruit in a single word.",
    "What is your favorite number between one and ten? One word.",
    "Name a musical instrument using one word.",
    "Which weekday do you prefer? One word only.",
    "Name a kitchen utensil in one word.",
    "Pick a planet. Reply with one word.",
)
LOGIT_TOLERANCE = 1e-4


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _time(fn: Callable[[], Any], repeats: int) -> tuple[Any, list[float]]:
    times, result = [], None
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return result, times


def gpu_checks(*, tokenizer, package, base_model, peft_model, adapter_names: Sequence[str], table, boundary_ids,
               cjk_ids, v_prompts: Sequence[str], repeats: int = 3) -> dict[str, Any]:
    renderer = Renderer(tokenizer, package, mode=SCIENTIFIC)
    report: dict[str, Any] = {"checks": {}, "throughput": {}, "digests": {}}
    ok = report["checks"]

    # render identity on V prompts (Q, H, a persona)
    for persona in ("P_default", "P_helpful", "P_cat_T1"):
        rendered = [renderer.render(persona, p) for p in v_prompts[:4]]
        ok[f"render_{persona}"] = all(tuple(r.input_ids[-3:]) == (151644, 77091, 198) for r in rendered)

    weight32 = lm_head_weight32(base_model)
    decoration = runner.decoration_ids(lambda s: tokenizer.encode(s, add_special_tokens=False))
    ctx = [runner.Context(f"tv|v{i}|Q+none", f"v{i}", "Q+none", "none", "P_default", p) for i, p in enumerate(v_prompts)]
    rendered = [renderer.render(c.persona_id, c.user_prompt) for c in ctx]

    def score(model):
        return runner.score_contexts(model, rendered, ctx, table, weight32=weight32, cjk_ids=cjk_ids,
                                     decoration=decoration, emoji=[]).word_logp

    base_scores, base_times = _time(lambda: score(base_model), repeats)
    report["throughput"]["base_score_s_per_context"] = float(np.median(base_times)) / len(ctx)
    report["digests"]["base_v_scores"] = _digest(base_scores)
    repeat = score(base_model)
    ok["determinism_base_scores"] = bool(np.array_equal(repeat, base_scores))

    if peft_model is not None:
        with models.active(peft_model, None) as disabled:
            disabled_scores = score(disabled)
        ok["disable_adapter_equals_base"] = bool(np.max(np.abs(disabled_scores - base_scores)) <= LOGIT_TOLERANCE)
        nonanimal_ctx = [runner.Context(f"tv|n{i}|Q+none", f"n{i}", "Q+none", "none", "P_default", p)
                         for i, p in enumerate(TV_NONANIMAL_PROMPTS)]
        nonanimal = [renderer.render(c.persona_id, c.user_prompt) for c in nonanimal_ctx]
        for name in adapter_names:
            ok[f"census_{name}"] = bool(models.lora_census(peft_model, name)["modules"] == models.EXPECTED_LORA_MODULES)
            with models.active(peft_model, name) as student:
                scores, times = _time(lambda: runner.score_contexts(
                    student, nonanimal, nonanimal_ctx, table, weight32=weight32, cjk_ids=cjk_ids,
                    decoration=[], emoji=[]).word_logp, repeats)
                report["throughput"][f"{name}_score_s_per_context"] = float(np.median(times)) / len(nonanimal_ctx)
                report["digests"][f"{name}_nonanimal_scores"] = _digest(scores)  # digest only (blindness)
                again = runner.score_contexts(student, nonanimal, nonanimal_ctx, table, weight32=weight32,
                                              cjk_ids=cjk_ids, decoration=[], emoji=[]).word_logp
                ok[f"determinism_{name}"] = bool(np.array_equal(again, scores))
                reqs = [SampleRequest(r.input_ids, 1000 + i) for i, r in enumerate(nonanimal)] * 4
                eos = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
                pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                toks, stimes = _time(lambda: sample_batch(student, reqs, eos_ids=eos, pad_id=pad), repeats)
                report["throughput"][f"{name}_sample_s_per_sample"] = float(np.median(stimes)) / len(reqs)
                ok[f"sampler_determinism_{name}"] = toks == sample_batch(student, reqs, eos_ids=eos, pad_id=pad)
                report["digests"][f"{name}_nonanimal_samples"] = hashlib.sha256(repr(toks).encode()).hexdigest()

    states, times = _time(lambda: [runner.last_token_states(base_model, r.input_ids) for r in rendered], repeats)
    report["throughput"]["base_capture_s_per_context"] = float(np.median(times)) / len(rendered)
    report["passed"] = all(bool(v) for v in ok.values())
    return report


def overhead_and_factors(gpu_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seconds per unit, per-adapter LoRA factor and the throughput CV across hosts/repeats (for decision D5)."""
    base = [r["throughput"]["base_score_s_per_context"] for r in gpu_reports]
    factors: dict[str, list[float]] = {}
    for r in gpu_reports:
        for key, value in r["throughput"].items():
            if key.endswith("_score_s_per_context") and not key.startswith("base"):
                factors.setdefault(key.split("_score")[0], []).append(value / r["throughput"]["base_score_s_per_context"])
    cv = float(np.std(base) / np.mean(base)) if len(base) > 1 else 0.0
    return {"base_score_s": float(np.mean(base)), "arm_factor": {k: float(np.mean(v)) for k, v in factors.items()},
            "throughput_cv": cv}
