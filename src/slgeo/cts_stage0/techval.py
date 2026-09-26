"""Outcome-blind technical validation v2 (TV-v2; spec ``tv_v2``, PREREGISTRATION §14).

Inputs: the 40 frozen V prompts, P_default, the nonce persona, TV-only random directions and a W_U-derived
planted direction. Scored "words" are the nonce word "zorb" plus stand-in words of random non-boundary tokens
with the exact form-length profile of the frozen table, so production shapes are exercised without scoring any
real word. S0 enters only as a list of rendered prompt lengths (tokenizer only, no forward, no id), used to
build length-matched stand-in prompts from V text for the throughput measurement.

Every persisted value is a boolean, hash, count, timing, TV-only aggregate or identity field.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Sequence

import numpy as np
import torch

from .extraction import last_token_hidden_states
from .package import FrozenPackage
from .render import NONCE_PERSONA_ID, TECHNICAL_VALIDATION, RenderedPrompt, Renderer
from .scoring import FormTable, build_prefix, lm_head_weight32, score_from_prefix, score_prompt
from .selftest import real_model_hook_selftest
from .steering import ALL, LAST, RowSteer

TV_SEED = 1000001
NONCE_WORD = "zorb"
PLANTED_GRID = (1.0, 2.0, 4.0, 8.0)
FRAGILITY_DIRECTIONS = 16
FRAGILITY_MAGNITUDES = (0.1, 0.25, 0.5)  # multiples of ||t_nonce|| at slot 14
N_S0_ANIMAL = 300
THROUGHPUT_CONDITIONS = 16


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


def length_matched_prompts(renderer: Renderer, v_texts: Sequence[str], targets: Sequence[int]) -> tuple[list[RenderedPrompt], dict]:
    """Stand-in user prompts built only from V text whose default rendering has each target length.

    Words of the V prompts are appended one by one (cycling) until the rendered length reaches the target; the
    last word is dropped if it overshoots. Returns the prompts and the length mismatch summary."""
    words = " ".join(v_texts).split()
    prompts, mismatches = [], []
    for index, target in enumerate(targets):
        chosen: list[str] = []
        cursor = index * 7 % len(words)
        best = None
        while True:
            candidate = chosen + [words[cursor % len(words)]]
            rendered = renderer.render("P_default", " ".join(candidate))
            if rendered.prompt_len > target and best is not None:
                break
            chosen, best = candidate, rendered
            cursor += 1
            if rendered.prompt_len >= target or len(chosen) > 400:
                break
        prompts.append(best)
        mismatches.append(best.prompt_len - int(target))
    return prompts, {"n": len(prompts), "max_abs_length_mismatch": int(np.max(np.abs(mismatches))) if mismatches else 0,
                     "mean_length_mismatch": float(np.mean(mismatches)) if mismatches else 0.0}


def _digest(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _set_log_odds(word_logp: np.ndarray) -> float:
    """l_{nonce, standins} = L_zorb - logsumexp of the stand-in words (TV-only statistic)."""
    others = word_logp[1:]
    peak = others.max()
    return float(word_logp[0] - (peak + np.log(np.exp(others - peak).sum())))


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def run_technical_validation(
    model, tokenizer, package: FrozenPackage, extraction_prompts, *, s0_lengths: Sequence[int], l2_shard_conditions: int,
    fixed_job_seconds: float,
) -> dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    renderer = Renderer(tokenizer, package, mode=TECHNICAL_VALIDATION)
    weight32 = lm_head_weight32(model)
    table = standin_table(package, tokenizer)
    v_prompts = package.validation_prompts()
    rendered = [renderer.render("P_default", p.prompt) for p in v_prompts]
    hidden = model.config.hidden_size
    rng = np.random.default_rng(TV_SEED)
    out: dict[str, Any] = {"n_v_prompts": len(v_prompts), "form_table": {"words": len(table.words), "forms": table.n_forms}}

    # Scale and the nonce axis (TV-only magnitudes).
    norms = [float(last_token_hidden_states(model, r.input_ids, device)[14].float().norm()) for r in rendered]
    scale = float(np.mean(norms))
    nonce_states = [last_token_hidden_states(model, renderer.render(NONCE_PERSONA_ID, p).input_ids, device)[14].double() for p in extraction_prompts[:64]]
    default_states = [last_token_hidden_states(model, renderer.render("P_default", p).input_ids, device)[14].double() for p in extraction_prompts[:64]]
    t_nonce_norm = float((torch.stack(nonce_states).mean(0) - torch.stack(default_states).mean(0)).norm())
    out["extraction_path_ok"] = bool(np.isfinite(scale) and np.isfinite(t_nonce_norm) and t_nonce_norm > 0)

    # Real-model hook self-test: both sites, L1 prefill and L2 suffix hooks, three extraction prompts.
    _sync(); t_self = time.time()
    selftest = real_model_hook_selftest(model, [renderer.render("P_default", p) for p in extraction_prompts[:3]], magnitude=scale)
    _sync(); selftest_seconds = time.time() - t_self
    out["hook_selftest"] = {"pass": selftest["pass"], "failed": selftest["failed"], "n_checks": sum(len(v) for v in selftest["checks"].values())}

    # Planted effect in the canonical L2 layout: unit(W_U[first token of " zorb"]) at blocks 13 and 27.
    zorb_first = tokenizer(" zorb", add_special_tokens=False)["input_ids"][0]
    planted_dir = weight32[zorb_first].detach().cpu().double()
    planted_dir = (planted_dir / planted_dir.norm()).to(torch.float32)
    prefixes = [build_prefix(model, r, device=weight32.device) for r in rendered]
    base_zorb = [score_from_prefix(model, pre, RowSteer({}), table, weight32=weight32).word_logp[0, 0] for pre in prefixes]
    planted = {}
    for block in (13, 27):
        effects = []
        for multiple in PLANTED_GRID:
            steered = [score_from_prefix(model, pre, RowSteer({block: planted_dir * multiple * scale}), table, weight32=weight32).word_logp[0, 0] for pre in prefixes]
            effects.append(float(np.mean(np.asarray(steered) - np.asarray(base_zorb))))
        planted[f"block_{block}"] = {"delta_L_zorb_at_8x_gt_1": bool(effects[-1] > 1.0), "monotone_nondecreasing": bool(np.all(np.diff(effects) >= -1e-6))}
    out["planted_effect"] = planted
    out["planted_effect_pass"] = bool(planted["block_27"]["delta_L_zorb_at_8x_gt_1"] and planted["block_27"]["monotone_nondecreasing"])

    # L2 determinism within the job (bitwise) and digests for the cross-host comparison.
    first = [score_from_prefix(model, pre, RowSteer({}), table, weight32=weight32).form_logp[0] for pre in prefixes]
    again = [score_from_prefix(model, build_prefix(model, r, device=weight32.device), RowSteer({}), table, weight32=weight32).form_logp[0] for r in rendered]
    out["l2_repeat_within_job_bitwise"] = bool(all(np.array_equal(a, b) for a, b in zip(first, again)))
    probe = RowSteer({13: random_unit(hidden, np.random.default_rng(TV_SEED + 1)) * 0.5 * scale}, LAST)
    steered_digest = _digest([score_from_prefix(model, pre, probe, table, weight32=weight32).form_logp[0] for pre in prefixes])
    out["l2_digests"] = {"baseline": _digest(first), "steered_block13": steered_digest}
    out["l1_digests"] = {"baseline": _digest([score_prompt(model, r, [RowSteer({})], table, weight32=weight32).form_logp[0] for r in rendered])}

    # Decision-level fragility L2 vs L1 on V (descriptive; spec PREREGISTRATION §9.3 explains why not a gate).
    base_l2 = np.asarray([_set_log_odds(score_from_prefix(model, pre, RowSteer({}), table, weight32=weight32).word_logp[0]) for pre in prefixes])
    base_l1 = np.asarray([_set_log_odds(score_prompt(model, r, [RowSteer({})], table, weight32=weight32).word_logp[0]) for r in rendered])
    ratios, noise_floor = [], []
    frag_rng = np.random.default_rng(TV_SEED + 2)
    for index in range(FRAGILITY_DIRECTIONS):
        unit = random_unit(hidden, frag_rng)
        multiple = FRAGILITY_MAGNITUDES[index % len(FRAGILITY_MAGNITUDES)]
        for sign in (1.0, -1.0):
            row = RowSteer({13: unit * sign * multiple * t_nonce_norm}, LAST)
            x_l2 = np.asarray([_set_log_odds(score_from_prefix(model, pre, row, table, weight32=weight32).word_logp[0]) for pre in prefixes]) - base_l2
            x_l1 = np.asarray([_set_log_odds(score_prompt(model, r, [row], table, weight32=weight32).word_logp[0]) for r in rendered]) - base_l1
            d = x_l2 - x_l1
            n = len(d)
            s_x, s_d = float(np.std(x_l2, ddof=1)), float(np.std(d, ddof=1))
            if s_x <= 0:
                continue
            bias_sq = max(0.0, float(np.mean(d)) ** 2 - s_d**2 / n)
            ratios.append(float(np.sqrt(bias_sq * N_S0_ANIMAL + s_d**2) / s_x))
            noise_floor.append(s_d / s_x)
    out["fragility_on_v_descriptive"] = {
        "n_statistics": len(ratios),
        "rho_median": float(np.median(ratios)) if ratios else None,
        "rho_max": float(np.max(ratios)) if ratios else None,
        "noise_floor_s_d_over_s_x_median": float(np.median(noise_floor)) if noise_floor else None,
        "note": "S0-level shift in SE units projected from 40 V prompts; bias term estimated with noise; descriptive only",
    }

    # Throughput per cost class on S0-length-matched stand-in prompts (steered; production shapes).
    matched, mismatch = length_matched_prompts(renderer, [p.prompt for p in v_prompts], s0_lengths)
    out["length_matching"] = mismatch
    tp_rng = np.random.default_rng(TV_SEED + 3)
    last_rows = [RowSteer({13: random_unit(hidden, tp_rng) * 0.25 * t_nonce_norm}, LAST) for _ in range(THROUGHPUT_CONDITIONS)]
    all_rows = [RowSteer({13: random_unit(hidden, tp_rng) * 0.25 * t_nonce_norm}, ALL) for _ in range(4)]
    _sync(); started = time.time()
    prefix_seconds = 0.0
    for r in matched:
        t0 = time.time()
        pre = build_prefix(model, r, device=weight32.device)
        _sync(); prefix_seconds += time.time() - t0
        for row in last_rows:
            score_from_prefix(model, pre, row, table, weight32=weight32)
    _sync()
    l2_total = time.time() - started
    per_condition_l2 = (l2_total - prefix_seconds) / (len(matched) * len(last_rows))
    prefix_per_prompt = prefix_seconds / len(matched)
    _sync(); started = time.time()
    for r in matched:
        for row in all_rows:
            score_prompt(model, r, [row], table, weight32=weight32)
    _sync(); own = (time.time() - started) / (len(matched) * len(all_rows))
    _sync(); started = time.time()
    for r in matched:
        for row in last_rows[:4]:
            score_prompt(model, r, [row], table, weight32=weight32)
    _sync(); l1 = (time.time() - started) / (len(matched) * 4)
    _sync(); started = time.time()
    for prompt in extraction_prompts[64:128]:
        last_token_hidden_states(model, renderer.render("P_default", prompt).input_ids, device)
    _sync(); extraction = (time.time() - started) / 64
    out["seconds"] = {
        "L2_shared_prefix": per_condition_l2 + prefix_per_prompt / max(1, l2_shard_conditions),
        "L2_per_condition_component": per_condition_l2,
        "L2_prefix_per_prompt_component": prefix_per_prompt,
        "own_prefix": own,
        "L1_reference": l1,
        "extraction_forward": extraction,
    }

    # Dry shard: one planned L2 shard (conditions x 300 prompt evaluations cycling the matched prompts) with
    # the sentinel and publication work of a real shard, for the end-to-end overhead factor.
    dry_rng = np.random.default_rng(TV_SEED + 4)
    dry_rows = [RowSteer({13: random_unit(hidden, dry_rng) * 0.25 * t_nonce_norm}, LAST) for _ in range(l2_shard_conditions)]
    _sync(); t0 = time.time()
    for r in matched[:2]:
        score_from_prefix(model, build_prefix(model, r, device=weight32.device), RowSteer({}), table, weight32=weight32)
    _sync(); sentinel = time.time() - t0
    scores = np.empty((len(dry_rows), N_S0_ANIMAL, table.n_forms))
    _sync(); t0 = time.time()
    for p_index in range(N_S0_ANIMAL):
        pre = build_prefix(model, matched[p_index % len(matched)], device=weight32.device)
        for c_index, row in enumerate(dry_rows):
            scores[c_index, p_index] = score_from_prefix(model, pre, row, table, weight32=weight32).form_logp[0]
    _sync(); compute = time.time() - t0
    t0 = time.time()
    payload = np.ascontiguousarray(scores).tobytes()
    hashlib.sha256(payload).hexdigest()
    publish = time.time() - t0
    wall = fixed_job_seconds + selftest_seconds + 2 * sentinel + compute + publish
    out["dry_shard"] = {
        "conditions": len(dry_rows),
        "prompt_evaluations": N_S0_ANIMAL,
        "fixed_job_seconds": fixed_job_seconds,
        "selftest_seconds": selftest_seconds,
        "sentinel_seconds": 2 * sentinel,
        "compute_seconds": compute,
        "publish_seconds": publish,
        "overhead_factor": wall / compute,
        "note": "excludes container start and scheduler time; the 0.8 x cap reserve covers them",
    }
    if torch.cuda.is_available():
        out["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    checks = {
        "extraction_path_ok": out["extraction_path_ok"],
        "hook_selftest": bool(selftest["pass"]),
        "planted_effect": out["planted_effect_pass"],
        "l2_repeat_within_job_bitwise": out["l2_repeat_within_job_bitwise"],
        "length_matching_within_2_tokens": mismatch["max_abs_length_mismatch"] <= 2,
    }
    out["checks"] = checks
    out["pass"] = all(checks.values())
    return out


def project(cpu: dict, gpu_records: Sequence[dict], plan: dict, *, cap: float, planning_fraction: float) -> dict:
    """TV-v2 projection node: cross-host identity and digests, conservative seconds, P and the gate."""
    from .budget import authorization_check, projection_a100_h

    identity_keys = ("gpu_name", "packages", "python", "cuda_runtime", "container_image", "venv")

    def identity(record):
        ident = record["identity"]
        return {k: ident.get(k) for k in identity_keys} | {"driver": (ident.get("nvidia_smi") or [{}])[0].get("driver")}

    results = [record["result"] for record in gpu_records]
    hosts = {record["identity"].get("machine_ad", {}).get("Machine") for record in gpu_records}
    checks = {
        "cpu_validation": bool(cpu.get("pass")),
        "gpu_validations": all(result["pass"] for result in results),
        "two_distinct_hosts": len(hosts) == len(gpu_records) >= 2 and None not in hosts,
        "identity_identical_across_hosts": len({repr(identity(r)) for r in gpu_records}) == 1,
        "l2_digests_identical_across_hosts": len({repr(result["l2_digests"]) for result in results}) == 1,
        "l1_digests_identical_across_hosts": len({repr(result["l1_digests"]) for result in results}) == 1,
    }
    classes = ("L2_shared_prefix", "own_prefix", "L1_reference", "extraction_forward")
    seconds = {name: max(float(result["seconds"][name]) for result in results) for name in classes}
    overhead = max(float(result["dry_shard"]["overhead_factor"]) for result in results)
    full = projection_a100_h(plan, seconds, overhead)
    conditions = {c["cid"]: c for c in plan["conditions"]}

    def without(predicate):
        shards = []
        for shard in plan["shards"]:
            if shard["stage"] in ("score", "rescore"):
                kept = [cid for cid in shard["payload"]["conditions"] if not predicate(conditions[cid], shard)]
                shards.append(dict(shard, payload=dict(shard["payload"], conditions=kept)))
            else:
                shards.append(shard)
        return projection_a100_h(dict(plan, shards=shards), seconds, overhead)

    step1 = without(lambda c, _s: not c["gating"])

    def drop_fragility_tail(c, shard):
        if shard["stage"] != "rescore" or c["group"] in ("baseline", "named"):
            return not c["gating"]
        members = [cid for cid, cond in conditions.items() if cond["group"] == c["group"] and cond["reference_rescore"]]
        return members.index(c["cid"]) >= 5

    step2 = without(drop_fragility_tail)
    gate = authorization_check(full, cap, planning_fraction)
    return {
        "checks": checks,
        "seconds": seconds,
        "overhead_factor": overhead,
        "projection_a100_h": full,
        "pre_authorization_ladder_a100_h": {"drop_descriptive": step1, "and_fragility_subset_5": step2},
        "authorization": gate.as_dict(),
        "pass": all(checks.values()) and gate.allowed,
    }
