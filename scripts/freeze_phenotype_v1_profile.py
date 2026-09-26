"""Freeze the seed-1 development profile v1 for Anchor hypothesis C5 (lead, 2026-09-26; revised after Phase-H A2).

Inputs (historical, already inspected): seed-1 sampled files, Q rendering + seed-47 number prefixes, T=1, 100
samples per prompt: base (`runs/reference_4080_…/teacher_outputs.jsonl`, mislabelled "teacher"), N, S.

Rule (fixed before any seed-2/3 Anchor data):
1. per prompt, first-word lemma counts over the 26 panel words, +0.5 smoothing, normalized over the panel;
2. v1 = mean residual beyond flattening of S vs N, computed by ``slgeo.phenotype.stats.residual_profile`` with the
   base arm as the independent weight reference and cat excluded from the fit (the same estimator as C5);
3. kept for the 18 panel words with N first-word count >= 20 over all 5,000 samples; C5 uses ranks only.

The earlier raw log-ratio version was a flattening test in
disguise (rho with log N count -0.72; pre-freeze audit).
Run from the public repo root: .venv/Scripts/python scripts/freeze_phenotype_v1_profile.py. Refuses to overwrite its output.
"""
import collections
import json
import re
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from slgeo.phenotype import panel, stats  # noqa: E402

OUT = Path('research/phenotype_anchor_v1/seed1_v1_profile.json')
FILES = {
    'base': 'runs/reference_4080_qwen7b_cat_subliminal_10k_3epochs/run_1779205677/teacher_outputs.jsonl',
    'N': 'results/reference_reproduction_4080/qwen7b_neutral_10k_3epochs/cat_preference_eval.json',
    'S': 'results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/preference_eval.json',
}
LEMMA = {panel.PLURAL[w]: w for w in panel.PANEL} | {w: w for w in panel.PANEL}


def rows(path):
    if path.endswith('.jsonl'):
        return [json.loads(line) for line in open(path, encoding='utf-8')]
    return json.load(open(path, encoding='utf-8'))['completions']


def first_word(text):
    m = re.match(r"\W*([A-Za-z]+)", text)
    return LEMMA.get(m.group(1).lower()) if m else None


def logq(path):
    counts = collections.defaultdict(lambda: np.zeros(len(panel.PANEL)))
    totals = collections.Counter()
    for r in rows(path):
        w = first_word(r['completion'])
        key = int(r['prompt_index']) % 50
        if w is not None:
            counts[key][panel.PANEL.index(w)] += 1
            totals[w] += 1
    mat = np.stack([counts[k] for k in sorted(counts)]) + 0.5
    return np.log(mat / mat.sum(axis=1, keepdims=True)), totals


if OUT.exists():
    raise SystemExit(f'{OUT} exists; the frozen profile is write-once')
lq = {arm: logq(path) for arm, path in FILES.items()}
n_counts = lq['N'][1]
words = [w for w in panel.PANEL if n_counts[w] >= 20]
profile = stats.residual_profile(lq['S'][0], lq['N'][0], lq['base'][0], words, panel.PANEL, panel.PANEL.index('cat'))
ranked = sorted(profile, key=lambda w: -profile[w])
OUT.write_text(json.dumps({'rule': __doc__.strip(), 'n_words': len(profile), 'profile': profile,
                           'n_counts': {w: int(n_counts[w]) for w in words}, 'rank_order_desc': ranked}, indent=1))
for w in ranked:
    print(f'{w:10s} {profile[w]:+.3f}')
