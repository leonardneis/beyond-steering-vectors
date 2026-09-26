"""Outcome-blind technical validation on the real model (PREREGISTRATION §13.1(3), §13.3, §14).

Inputs: V prompts, P_default, the nonce persona, TV-only random directions and W_U-derived planted
directions. Scored "words" are the nonce word "zorb" plus stand-in words made of random non-boundary tokens
whose form-length profile equals the frozen answer-form table, so the production shapes (104 forms, 14 words)
are exercised without scoring any real word. Every persisted value is on the whitelist of
``IMPLEMENTATION_CHOICES.json`` (booleans, hashes, max-abs differences, counts, timings, identity).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .package import FrozenPackage
from .render import NONCE_PERSONA_ID, TECHNICAL_VALIDATION, Renderer
from .scoring import FormTable, lm_head_weight32, score_prompt, score_sequential, word_scores
from .selftest import real_model_hook_selftest
from .steering import ALL, LAST, RowSteer
from .extraction import last_token_hidden_states

TV_SEED = 1000001
NONCE_WORD = "zorb"
MAGNITUDE_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
PLANTED_GRID = (1.0, 2.0, 4.0, 8.0)
SITES = (8, 14, 21, 27)
EQUIVALENCE_BATCHES = (8, 16, 32)
CACHED_VS_UNCACHED_TOLERANCE = 0.05  # declared before any real-model run (pre-implementation audit 2, I13-b)


def nonce_forms(tokenizer) -> list[dict]:
    texts = ["Zorb", "zorb", " Zorb", " zorb", "Zorbs", "zorbs", " Zorbs", " zorbs"]
    return [{"text": text, "token_ids": tokenizer(text, add_special_tokens=False)["input_ids"]} for text in texts]


def standin_table(package: FrozenPackage, tokenizer, seed: int = TV_SEED) -> FormTable:
    """Nonce word plus stand-in words with the exact per-word form-length profile of the frozen table."""
    endpoint = package.endpoint
    real_words = endpoint["scoring_words"] + endpoint["non_animal_words"]
    boundary = set(endpoint["boundary_ids"])
    real_ids = {token for info in endpoint["answer_forms"].values() for form in info["forms"] for token in form["token_ids"]}
    nonce = nonce_forms(tokenizer)
    nonce_ids = {token for form in nonce for token in form["token_ids"]}
    pool = np.array([i for i in range(tokenizer.vocab_size) if i not in boundary and i not in real_ids and i not in nonce_ids])
    rng = np.random.default_rng(seed)
    answer_forms = {NONCE_WORD: {"forms": nonce}}
    words = [NONCE_WORD]
    for index, word in enumerate(real_words[1:], start=1):
        forms = []
        for form in endpoint["answer_forms"][word]["forms"]:
            forms.append({"text": f"standin{index}", "token_ids": [int(t) for t in rng.choice(pool, size=len(form["token_ids"]), replace=False)]})
        name = f"standin_{index:02d}"
        answer_forms[name] = {"forms": forms}
        words.append(name)
    table = FormTable.from_endpoint({"answer_forms": answer_forms, "boundary_ids": endpoint["boundary_ids"]}, words=words)
    if any(word in real_words for word in table.words):
        raise RuntimeError("A real scoring word entered the technical-validation form table")
    return table


def random_unit(hidden: int, rng: np.random.Generator) -> torch.Tensor:
    vector = rng.standard_normal(hidden)
    return torch.from_numpy(vector / np.linalg.norm(vector)).to(torch.float32)


@torch.inference_mode()
def prefix_reuse_scores(model, rendered, rows, table, weight32) -> tuple[np.ndarray, float]:
    """Candidate efficiency layout (measurement only; NOT used by the scientific run).

    Positions 0..L-2 are forwarded once (unsteered; identical for every last-position condition by causality);
    per condition one forward carries token L-1 (steered at block b) followed by the packed answer forms.
    Returns per-row form log-probs and seconds per prompt-condition.
    """
    from transformers.cache_utils import DynamicCache

    from .scoring import _cache_tensors, _packed_mask

    device = weight32.device
    ids = list(rendered.input_ids)
    prompt_len = len(ids)
    prefix = model.model(input_ids=torch.tensor([ids[:-1]], device=device), past_key_values=DynamicCache(), use_cache=True)
    snapshot = _cache_tensors(prefix.past_key_values)
    tokens, depths, segments, _spans = table.packed()
    first_index, trans_pos, trans_tok, trans_form, last_pos = table.gather_plan()
    boundary = torch.tensor(table.boundary_ids, device=device)
    step_ids = torch.tensor([[ids[-1]] + tokens], device=device)
    positions = torch.tensor([[prompt_len - 1] + [prompt_len + d for d in depths]], device=device)
    inner = _packed_mask(prompt_len, depths, segments, 1, model.dtype, device)[0, 0]  # [T, L + T]
    t = len(tokens)
    mask = torch.full((1 + t, prompt_len + t), torch.finfo(model.dtype).min, dtype=model.dtype, device=device)
    mask[0, :prompt_len] = 0  # token L-1 sees the prefix and itself
    mask[1:, :] = inner
    mask = mask[None, None]
    out = np.empty((len(rows), table.n_forms))
    started = time.time()
    for index, row in enumerate(rows):
        cache = DynamicCache()
        for layer, (keys, values) in enumerate(snapshot):
            cache.update(keys, values, layer)
        handles = []
        for block, vector in row.vectors.items():
            if row.mode != LAST:
                raise ValueError("prefix reuse supports last-position steering only")

            def hook(_m, _a, output, vector=vector):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden.clone()
                hidden[:, 0] = hidden[:, 0] + vector.to(hidden.device, hidden.dtype)
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

            handles.append(model.model.layers[block].register_forward_hook(hook))
        try:
            hidden = model.model(input_ids=step_ids, position_ids=positions, attention_mask=mask, past_key_values=cache, use_cache=True).last_hidden_state[0]
        finally:
            for handle in handles:
                handle.remove()
        logits = torch.nn.functional.linear(hidden.float(), weight32)
        lp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        first = lp[0]
        cont = lp[1:]
        values = first[torch.tensor(first_index, device=device)].double()
        for pos, tok, form in zip(trans_pos, trans_tok, trans_form):
            values[form] += cont[pos, tok].double()
        values += torch.logsumexp(cont[torch.tensor(last_pos, device=device)][:, boundary], dim=-1).double()
        out[index] = values.cpu().numpy()
    return out, (time.time() - started) / max(1, len(rows))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


@torch.inference_mode()
def run_technical_validation(model, tokenizer, package: FrozenPackage, extraction_prompts, *, batch_sizes=EQUIVALENCE_BATCHES) -> dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    renderer = Renderer(tokenizer, package, mode=TECHNICAL_VALIDATION)
    weight32 = lm_head_weight32(model)
    table = standin_table(package, tokenizer)
    v_prompts = package.validation_prompts()
    rendered = [renderer.render("P_default", p.prompt) for p in v_prompts]
    hidden = model.config.hidden_size
    rng = np.random.default_rng(TV_SEED)
    out: dict[str, Any] = {"n_v_prompts": len(v_prompts), "form_table": {"words": len(table.words), "forms": table.n_forms,
                          "packed_tokens": len(table.packed()[0])}}

    # Scale: mean ||h|| of P_default slot-14 last-token states on V (only a scale, never persisted).
    started = time.time()
    norms = [float(last_token_hidden_states(model, r.input_ids, device)[14].float().norm()) for r in rendered]
    scale = float(np.mean(norms))
    out["timing_extraction_forward_s"] = (time.time() - started) / len(rendered)

    # Extraction path on the nonce persona and P_default (booleans only).
    ext_ok = True
    for persona in ("P_default", NONCE_PERSONA_ID):
        for prompt in extraction_prompts[:8]:
            states = last_token_hidden_states(model, renderer.render(persona, prompt).input_ids, device)
            ext_ok &= len(states) == 29 and all(bool(torch.isfinite(s).all()) for s in states)
    out["extraction_path_ok"] = bool(ext_ok)

    # Hook self-test on the real model.
    out["hook_selftest"] = real_model_hook_selftest(model, renderer.render("P_default", extraction_prompts[0]), magnitude=scale)

    # Planted effect: W_U direction of the first token of " zorb" raises L_zorb.
    zorb_first = tokenizer(" zorb", add_special_tokens=False)["input_ids"][0]
    planted_dir = weight32[zorb_first].detach().cpu().double()
    planted_dir = (planted_dir / planted_dir.norm()).to(torch.float32)
    planted = {}
    base_zorb = [score_prompt(model, r, [RowSteer({})], table, weight32=weight32).word_logp[0, 0] for r in rendered]
    for site in (14, 28):
        block = site - 1
        effects = []
        for multiple in PLANTED_GRID:
            steered = [
                score_prompt(model, r, [RowSteer({block: planted_dir * multiple * scale})], table, weight32=weight32).word_logp[0, 0]
                for r in rendered
            ]
            effects.append(float(np.mean(np.asarray(steered) - np.asarray(base_zorb))))
        planted[f"block_{block}"] = {"delta_L_zorb_at_8x_gt_1": bool(effects[-1] > 1.0), "monotone_nondecreasing": bool(np.all(np.diff(effects) >= -1e-6))}
    out["planted_effect"] = planted
    # Gate (revision 2026-09-26, see IMPLEMENTATION_CHOICES.json): the last decoder block, where a W_U-derived
    # direction is read out directly (PREREGISTRATION §13.1(3) names no site). The production site is reported.
    out["planted_effect_pass"] = bool(planted["block_27"]["delta_L_zorb_at_8x_gt_1"] and planted["block_27"]["monotone_nondecreasing"])

    # Row library: zero, random directions on the grid, last/all positions, all sites.
    library: list[RowSteer] = [RowSteer({})]
    for site in SITES:
        for multiple in MAGNITUDE_GRID:
            for mode in (LAST, ALL):
                library.append(RowSteer({site - 1: random_unit(hidden, rng) * multiple * scale}, mode))

    # Determinism: identical results for repeated batch-1 scoring within the job.
    first = [score_prompt(model, r, [RowSteer({})], table, weight32=weight32).form_logp[0] for r in rendered]
    again = [score_prompt(model, r, [RowSteer({})], table, weight32=weight32).form_logp[0] for r in rendered]
    out["repeat_within_job_max_abs"] = max(_max_abs(a, b) for a, b in zip(first, again))
    out["baseline_digest"] = __import__("hashlib").sha256(np.asarray(first).tobytes()).hexdigest()

    # Packed layout vs per-form sequential reference (batch 1), across row classes.
    form_diffs, word_diffs = [], []
    for r in rendered[:10]:
        for row in library[:: max(1, len(library) // 6)]:
            packed = score_prompt(model, r, [row], table, weight32=weight32)
            sequential = score_sequential(model, r, row, table, weight32=weight32)
            form_diffs.append(np.abs(packed.form_logp[0] - sequential))
            word_diffs.append(np.abs(packed.word_logp[0] - word_scores(sequential, table)))
    form_diffs, word_diffs = np.concatenate(form_diffs), np.concatenate(word_diffs)
    out["packed_vs_sequential_max_abs"] = float(form_diffs.max())
    out["layout_noise_packed_vs_sequential"] = {
        "form_logp_quantiles_50_90_99_max": [float(q) for q in np.quantile(form_diffs, [0.5, 0.9, 0.99, 1.0])],
        "word_logp_quantiles_50_90_99_max": [float(q) for q in np.quantile(word_diffs, [0.5, 0.9, 0.99, 1.0])],
    }

    # Prefill + cache vs uncached full forward (I13-b) and a negative control with the hook at L-2.
    from .selftest import full_sequence_form_logp

    row = RowSteer({13: random_unit(hidden, rng) * 2.0 * scale}, LAST)
    cached = score_prompt(model, rendered[0], [row], table, weight32=weight32).form_logp[0]
    full = full_sequence_form_logp(model, rendered[0].input_ids, row, table, weight32)
    misplaced = full_sequence_form_logp(model, rendered[0].input_ids, row, table, weight32, position_offset=2)
    out["cached_vs_uncached_max_abs"] = _max_abs(cached, full)
    out["cached_vs_uncached_pass"] = bool(out["cached_vs_uncached_max_abs"] <= CACHED_VS_UNCACHED_TOLERANCE)
    out["negative_control_misplaced_hook_exceeds_tolerance"] = bool(_max_abs(cached, misplaced) > CACHED_VS_UNCACHED_TOLERANCE)

    # §13.3 equivalence of condition batching vs batch 1, per candidate batch size.
    equivalence = {}
    for batch in batch_sizes:
        form_diffs, word_diffs = [], []
        for r in rendered:
            chosen = [library[i] for i in rng.choice(len(library), size=batch, replace=False)]
            batched = score_prompt(model, r, chosen, table, weight32=weight32)
            for index, choice in enumerate(chosen):
                single = score_prompt(model, r, [choice], table, weight32=weight32)
                form_diffs.append(np.abs(batched.form_logp[index] - single.form_logp[0]))
                word_diffs.append(np.abs(batched.word_logp[index] - single.word_logp[0]))
        form_diffs, word_diffs = np.concatenate(form_diffs), np.concatenate(word_diffs)
        equivalence[str(batch)] = {
            "max_abs_form_logp": float(form_diffs.max()),
            "max_abs_word_logp": float(word_diffs.max()),
            "form_logp_quantiles_50_90_99": [float(q) for q in np.quantile(form_diffs, [0.5, 0.9, 0.99])],
            "word_logp_quantiles_50_90_99": [float(q) for q in np.quantile(word_diffs, [0.5, 0.9, 0.99])],
            "pass": bool(max(form_diffs.max(), word_diffs.max()) <= 1e-3),
        }
    out["equivalence_13_3"] = equivalence

    # fp16 overflow probe at the largest planned multiple, all positions, every site.
    overflow_ok = True
    for site in SITES:
        probe = RowSteer({site - 1: random_unit(hidden, rng) * 4.0 * scale}, ALL)
        try:
            result = score_prompt(model, rendered[0], [probe], table, weight32=weight32)
            overflow_ok &= bool(np.isfinite(result.form_logp).all())
        except Exception:  # noqa: BLE001 - any failure is a failed probe
            overflow_ok = False
    out["fp16_overflow_probe_finite"] = bool(overflow_ok)

    # Throughput per prompt-condition for batch 1 and each batch size (production shapes).
    throughput = {}
    for batch in (1, *batch_sizes):
        rows = [library[i % len(library)] for i in range(batch)]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        started = time.time()
        for r in rendered[:10]:
            score_prompt(model, r, rows, table, weight32=weight32)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        throughput[str(batch)] = (time.time() - started) / (10 * batch)
    out["seconds_per_prompt_condition"] = throughput
    if torch.cuda.is_available():
        out["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30

    # Candidate efficiency layout (measurement for a possible researcher decision; not used in production).
    last_rows = [row for row in library if row.mode == LAST][:12]
    reuse_diffs, reuse_times = [], []
    for r in rendered[:10]:
        reuse, seconds = prefix_reuse_scores(model, r, last_rows, table, weight32)
        reuse_times.append(seconds)
        for index, row in enumerate(last_rows):
            canonical = score_prompt(model, r, [row], table, weight32=weight32).form_logp[0]
            reuse_diffs.append(np.abs(reuse[index] - canonical))
    reuse_diffs = np.concatenate(reuse_diffs)
    out["candidate_prefix_reuse"] = {
        "seconds_per_prompt_condition": float(np.mean(reuse_times)),
        "vs_canonical_form_logp_quantiles_50_90_99_max": [float(q) for q in np.quantile(reuse_diffs, [0.5, 0.9, 0.99, 1.0])],
        "note": "measurement only; adopting it would require a dated researcher decision",
    }

    # Sampling path timing (nonce persona, TV seed).
    from .sampling import sample_answers

    started = time.time()
    for r in rendered[:5]:
        sample_answers(model, renderer.render(NONCE_PERSONA_ID, r.user_prompt), RowSteer({}), weight32=weight32, seed=TV_SEED)
    out["seconds_per_sampled_prompt_condition"] = (time.time() - started) / 5
    return out


def project_budget(tv: dict[str, Any], plan: dict[str, Any], batch_rows: int, *, n_prompts: int = 334) -> dict[str, float]:
    """Projected A100-h of the scientific plan from measured throughput (gating-only and full)."""
    per = float(tv["seconds_per_prompt_condition"][str(batch_rows)])
    per1 = float(tv["seconds_per_prompt_condition"]["1"])
    conditions = plan["conditions"]
    steered_gating = sum(1 for c in conditions if c["gating"] and c["kind"] == "steer")
    steered_all = sum(1 for c in conditions if c["kind"] == "steer")
    persona_gating = sum(1 for c in conditions if c["gating"] and c["kind"] == "persona")
    persona_all = sum(1 for c in conditions if c["kind"] == "persona")
    extraction = 44 * 1024 * float(tv["timing_extraction_forward_s"])
    baseline = 2 * n_prompts * per * batch_rows
    sampling = 11 * n_prompts * float(tv["seconds_per_sampled_prompt_condition"])
    overhead = 60 * 5.0 * 60  # up to 60 GPU jobs x 5 min load/verify/self-test

    def hours(steered, personas, with_sampling):
        seconds = extraction + baseline + steered * n_prompts * per + personas * n_prompts * per1 + overhead
        return (seconds + (sampling if with_sampling else 0.0)) / 3600.0

    return {"gating_only_a100_h": hours(steered_gating, persona_gating, False), "full_a100_h": hours(steered_all, persona_all, True)}
